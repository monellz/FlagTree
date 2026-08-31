#ifdef __TLE__

#include "Dialect/MUSA/IR/Dialect.h"
#include "TritonMUSACommon/BarrierUtils.h"
#include "TritonMUSAGPUTransforms/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include <cstdint>
#include <limits>
#include <utility>

namespace mlir {

#define GEN_PASS_DEF_TRITONMUSAGPUTLELOWERWARPSPECIALIZE
#include "TritonMUSAGPUTransforms/Passes.h.inc"

namespace {

namespace ttg = triton::gpu;
namespace ttmg = triton::musa;

static constexpr StringLiteral kStaticWarpSpecializeAttr =
    "musa_tle.static_warp_specialize";
static constexpr StringLiteral kLocalSyncIntrinsic = "llvm.musa.syncthreads.lm";
// `tl.debug_barrier()` reaches here as a CTA-wide barrier. Inside a partition
// that is a deadlock -- the peer partition never executes it -- so it is
// rewritten onto the region's barrier just like the shared-memory syncs the
// compiler inserts for layout conversions.
static constexpr StringLiteral kBlockSyncIntrinsic = "llvm.musa.barrier0";

static bool isPartitionSync(LLVM::CallIntrinsicOp call) {
  return call.getIntrin() == kLocalSyncIntrinsic ||
         call.getIntrin() == kBlockSyncIntrinsic;
}
static constexpr StringLiteral kBarRecordIntrinsic =
    "llvm.musa.async.bar.record";

// `llvm.musa.lma.wait` is unknown to the in-process LLVM (llvm22); emit a
// plain extern call that the backend driver textually renames to the
// intrinsic before the MUSA llc (llvm14) runs. See make_mubin in
// backend/compiler.py.
static LLVM::LLVMFuncOp getOrCreateLmaWaitShim(Operation *anchor,
                                               RewriterBase &rewriter) {
  auto mod = anchor->getParentOfType<ModuleOp>();
  StringRef name = "__musa_shim_lma_wait";
  if (auto fn = mod.lookupSymbol<LLVM::LLVMFuncOp>(name))
    return fn;
  OpBuilder::InsertionGuard g(rewriter);
  rewriter.setInsertionPointToStart(mod.getBody());
  auto fnTy = LLVM::LLVMFunctionType::get(
      LLVM::LLVMVoidType::get(mod.getContext()), {});
  return LLVM::LLVMFuncOp::create(rewriter, mod.getLoc(), name, fnTy);
}

static LogicalResult lowerWarpGroupBarriers(LLVM::LLVMFuncOp func,
                                            ttg::WarpSpecializeOp ws,
                                            IRRewriter &rewriter) {
  ModuleOp module = func->getParentOfType<ModuleOp>();
  auto consumerWarpsAttr =
      module->getAttrOfType<IntegerAttr>(ttg::AttrNumWarpsName);
  if (!consumerWarpsAttr || consumerWarpsAttr.getInt() <= 0 ||
      consumerWarpsAttr.getInt() > std::numeric_limits<int32_t>::max())
    return ws.emitOpError(
        "mthreads TLE default partition requires a positive int32 "
        "ttg.num-warps");

  // Every sync a region asked for is real: layout conversions and other shared
  // memory round-trips, reductions and tl.debug_barrier all need the warps of
  // that region to rendezvous. A CTA-wide sync cannot serve them once only part
  // of the CTA runs the region, but dropping them silently races -- the warps
  // then read scratch their peers have not written yet -- so give each one a
  // hardware barrier sized to its own region.
  // One barrier per region, not per site: a barrier is a rendezvous of the same
  // warps every time, so every site in a region can share it the way loop trips
  // already do, and a kernel with many conversions does not run the hardware
  // out of barrier ids.
  SmallVector<LLVM::CallIntrinsicOp> syncs;
  SmallVector<int32_t> syncWarps;
  SmallVector<unsigned> syncRegion;
  SmallVector<int32_t> regionWarps;
  auto collect = [&](Region &region, int64_t numWarps) -> LogicalResult {
    if (numWarps <= 0 || numWarps > std::numeric_limits<int32_t>::max())
      return ws.emitOpError(
          "mthreads TLE partition requires a positive int32 warp count");
    unsigned index = regionWarps.size();
    bool used = false;
    region.walk([&](LLVM::CallIntrinsicOp call) {
      if (isPartitionSync(call)) {
        syncs.push_back(call);
        syncWarps.push_back(static_cast<int32_t>(numWarps));
        syncRegion.push_back(index);
        used = true;
      }
    });
    if (used)
      regionWarps.push_back(static_cast<int32_t>(numWarps));
    return success();
  };

  if (failed(collect(ws.getDefaultRegion(), consumerWarpsAttr.getInt())))
    return failure();
  ArrayRef<int32_t> partitionWarps = ws.getPartitionNumWarps();
  for (auto [index, partition] : llvm::enumerate(ws.getPartitionRegions())) {
    if (index >= partitionWarps.size())
      return ws.emitOpError("partition warp counts do not cover all regions");
    if (failed(collect(*partition, partitionWarps[index])))
      return failure();
  }
  if (syncs.empty())
    return success();

  // Barrier ids are a scarce hardware resource, but a region cannot put every
  // rendezvous on one id either: consecutive rendezvous on the same barrier let
  // a fast warp enter the next one before a slow warp has left the previous,
  // which corrupts whatever the barrier was protecting. Give each region a
  // small pool and rotate through it, so neighbouring sites differ while the
  // id count stays bounded. Shrink the pools if the budget cannot cover them.
  SmallVector<unsigned> regionSites(regionWarps.size(), 0);
  for (unsigned index : syncRegion)
    ++regionSites[index];

  // Prefer one id per site. Two sites sharing an id stay in step only while
  // they run the same number of times -- pair a site inside a loop with one
  // outside it and the phases desync, which silently corrupts whatever the
  // barrier protects. Aliasing is a fallback for kernels too big to give every
  // site its own id, not the default.
  unsigned maxSites = 0;
  for (unsigned sites : regionSites)
    maxSites = std::max(maxSites, sites);

  // Barrier ids must be covered by the `bar.record` declaration or a wait on
  // them never completes. This pass re-emits the record from the function's
  // reserved count after allocating (see the end of this function), so ids up
  // to ttmg::kMaxBarrierId are all usable.
  int32_t taken = ttmg::getReservedBarrierCount(
      cast<FunctionOpInterface>(func.getOperation()));
  int32_t available = ttmg::kMaxBarrierId - taken;

  SmallVector<unsigned> poolSize;
  FailureOr<int32_t> reserved = failure();
  for (unsigned cap = std::max(maxSites, 1u); cap >= 1; cap /= 2) {
    poolSize.clear();
    int32_t total = 0;
    for (unsigned sites : regionSites) {
      unsigned size = std::min(sites, cap);
      poolSize.push_back(size);
      total += static_cast<int32_t>(size);
    }
    if (total > available) {
      if (cap == 1)
        break;
      continue;
    }
    reserved = ttmg::reserveBarrierIdRange(syncs.front(), total);
    if (succeeded(reserved))
      break;
  }
  if (failed(reserved))
    return syncs.front().emitOpError(
        "mthreads TLE partition synchronization exhausted hardware barrier "
        "ids");

  LLVM::CallIntrinsicOp initializationRendezvous;
  for (Operation *op = ws->getPrevNode(); op; op = op->getPrevNode()) {
    auto call = dyn_cast<LLVM::CallIntrinsicOp>(op);
    if (call && call.getIntrin() == kLocalSyncIntrinsic) {
      initializationRendezvous = call;
      break;
    }
  }

  Location loc = initializationRendezvous ? initializationRendezvous.getLoc()
                                          : ws.getLoc();
  if (initializationRendezvous)
    rewriter.setInsertionPoint(initializationRendezvous);
  else
    rewriter.setInsertionPoint(ws);
  Value phase = arith::ConstantIntOp::create(rewriter, loc, 0, 32);
  llvm::DenseMap<Operation *, Value> barrierIds;
  SmallVector<std::pair<Value, Value>> initializationArgs;
  int32_t nextId = *reserved;
  SmallVector<SmallVector<Value>> regionPools(regionWarps.size());
  for (auto [index, size] : llvm::enumerate(poolSize)) {
    for (unsigned slot = 0; slot < size; ++slot) {
      Value id = arith::ConstantIntOp::create(rewriter, loc, nextId++, 32);
      Value count =
          arith::ConstantIntOp::create(rewriter, loc, regionWarps[index], 32);
      initializationArgs.push_back({id, count});
      regionPools[index].push_back(id);
    }
  }
  SmallVector<unsigned> nextInRegion(regionWarps.size(), 0);
  llvm::DenseMap<Operation *, unsigned> barrierPoolSlot;
  for (auto [sync, index] : llvm::zip(syncs, syncRegion)) {
    unsigned slot = nextInRegion[index]++ % poolSize[index];
    barrierIds[sync.getOperation()] = regionPools[index][slot];
    barrierPoolSlot[sync.getOperation()] = slot;
  }

  // A barrier completes into the opposite phase, so its wait has to name the
  // phase it expects. A site inside a loop meets its own barrier again on the
  // next trip, so the phase cannot be a constant: keep it in a stack slot and
  // flip it after every rendezvous. Without this the second trip waits on a
  // phase the barrier has already left and the partition hangs.
  llvm::DenseMap<Operation *, Value> phaseSlots;
  {
    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointToStart(&func.getBody().front());
    Type i32 = rewriter.getI32Type();
    Value one = arith::ConstantIntOp::create(rewriter, loc, 1, 32);
    Value zero = arith::ConstantIntOp::create(rewriter, loc, 0, 32);
    // One phase slot per barrier id, since the phase belongs to the barrier.
    SmallVector<SmallVector<Value>> regionPhaseSlots(regionWarps.size());
    for (auto [index, size] : llvm::enumerate(poolSize)) {
      for (unsigned slot = 0; slot < size; ++slot) {
        Value storage = LLVM::AllocaOp::create(
            rewriter, loc, LLVM::LLVMPointerType::get(rewriter.getContext()),
            i32, one, /*alignment=*/0);
        LLVM::StoreOp::create(rewriter, loc, zero, storage);
        regionPhaseSlots[index].push_back(storage);
      }
    }
    for (auto [sync, index] : llvm::zip(syncs, syncRegion))
      phaseSlots[sync.getOperation()] =
          regionPhaseSlots[index][barrierPoolSlot.lookup(sync.getOperation())];
  }

  Value tid =
      LLVM::CallIntrinsicOp::create(
          rewriter, loc, rewriter.getI32Type(),
          rewriter.getStringAttr("llvm.musa.read.ptx.sreg.tid.x"), ValueRange{})
          .getResult(0);
  Value issueInit = arith::CmpIOp::create(rewriter, loc,
                                          arith::CmpIPredicate::eq, tid, phase);
  auto initIf = scf::IfOp::create(rewriter, loc, issueInit, false);
  rewriter.setInsertionPointToStart(&initIf.getThenRegion().front());
  for (auto [id, count] : initializationArgs)
    LLVM::CallIntrinsicOp::create(
        rewriter, loc, rewriter.getStringAttr("llvm.musa.async.init.arrival"),
        ValueRange{id, count, phase});

  rewriter.setInsertionPointAfter(initIf);
  if (!initializationRendezvous)
    LLVM::CallIntrinsicOp::create(rewriter, loc,
                                  rewriter.getStringAttr(kLocalSyncIntrinsic),
                                  ValueRange{});

  for (LLVM::CallIntrinsicOp sync : syncs) {
    rewriter.setInsertionPoint(sync);
    Location siteLoc = sync.getLoc();
    Value id = barrierIds.lookup(sync.getOperation());
    Value slot = phaseSlots.lookup(sync.getOperation());
    Value expected =
        LLVM::LoadOp::create(rewriter, siteLoc, rewriter.getI32Type(), slot);
    // The `syncthreads` this replaces was a memory barrier as well as a
    // rendezvous; an arrive/wait pair on its own is not, so each arriving
    // warp must flush its OWN outstanding asynchronous shared accesses
    // (loads and stores -- MP31 lma ops) before arriving. `lma.wait` drains
    // only the arriving warp's own outstanding accesses, and the unmodeled
    // call blocks compiler code motion across it, which together give the
    // release ordering the sync needs. It replaces a CTA-scope membar.cta
    // pair that would drain the whole MP's memory pipeline on both sides of
    // every region sync.
    LLVM::CallOp::create(rewriter, siteLoc,
                         getOrCreateLmaWaitShim(sync, rewriter), ValueRange{});
    LLVM::CallIntrinsicOp::create(
        rewriter, siteLoc,
        rewriter.getStringAttr("llvm.musa.async.arrive.none.phaseid"),
        ValueRange{id});
    LLVM::CallIntrinsicOp::create(
        rewriter, siteLoc, rewriter.getStringAttr("llvm.musa.async.wait"),
        ValueRange{id, expected});
    Value one = arith::ConstantIntOp::create(rewriter, siteLoc, 1, 32);
    LLVM::StoreOp::create(
        rewriter, siteLoc,
        arith::XOrIOp::create(rewriter, siteLoc, expected, one), slot);
    rewriter.eraseOp(sync);
  }

  SmallVector<LLVM::CallIntrinsicOp> oldBarRecords;
  func.walk([&](LLVM::CallIntrinsicOp call) {
    if (call.getIntrin() == kBarRecordIntrinsic)
      oldBarRecords.push_back(call);
  });
  for (LLVM::CallIntrinsicOp record : oldBarRecords)
    rewriter.eraseOp(record);
  rewriter.setInsertionPointToStart(&func.getBody().front());
  Value barCount = arith::ConstantIntOp::create(
      rewriter, func.getLoc(), ttmg::getReservedBarrierCount(func), 32);
  LLVM::CallIntrinsicOp::create(rewriter, func.getLoc(),
                                rewriter.getStringAttr(kBarRecordIntrinsic),
                                ValueRange{barCount});
  return success();
}

static LogicalResult lowerStaticWarpSpecialize(LLVM::LLVMFuncOp func,
                                               ttg::WarpSpecializeOp ws,
                                               IRRewriter &rewriter) {
  if (ws.getNumResults() != 0 || ws.getPartitionRegions().empty() ||
      ws.getPartitionNumWarps().size() != ws.getPartitionRegions().size())
    return ws.emitOpError(
        "mthreads TLE late lowering requires a warp count per worker "
        "partition and no results");

  Region &consumerRegion = ws.getDefaultRegion();
  if (consumerRegion.empty())
    return ws.emitOpError("mthreads TLE static partitions must not be empty");

  auto partitions = ws.getPartitionOp();
  ValueRange captures = partitions.getExplicitCaptures();
  for (Region *partition : ws.getPartitionRegions()) {
    if (partition->empty())
      return ws.emitOpError("mthreads TLE static partitions must not be empty");
    Block &entry = partition->front();
    if (entry.getNumArguments() != captures.size())
      return ws.emitOpError("worker capture count changed during lowering");
    for (auto [argument, capture] : llvm::zip(entry.getArguments(), captures)) {
      if (argument.getType() != capture.getType())
        return ws.emitOpError(
            "worker capture types were not converted consistently");
      argument.replaceAllUsesWith(capture);
    }
    entry.eraseArguments([](BlockArgument) { return true; });
  }

  ModuleOp module = func->getParentOfType<ModuleOp>();
  auto consumerWarpsAttr =
      module->getAttrOfType<IntegerAttr>(ttg::AttrNumWarpsName);
  auto threadsPerWarpAttr =
      module->getAttrOfType<IntegerAttr>(ttg::AttrNumThreadsPerWarp);
  if (!consumerWarpsAttr || !threadsPerWarpAttr ||
      consumerWarpsAttr.getInt() <= 0 || threadsPerWarpAttr.getInt() <= 0)
    return ws.emitOpError("late lowering requires positive ttg.num-warps and "
                          "ttg.threads-per-warp");
  int32_t threadsPerWarp = static_cast<int32_t>(threadsPerWarpAttr.getInt());
  int64_t boundary64 = consumerWarpsAttr.getInt() * threadsPerWarp;
  if (boundary64 > std::numeric_limits<int32_t>::max())
    return ws.emitOpError("static partition boundary exceeds int32 range");
  int32_t boundary = static_cast<int32_t>(boundary64);

  // Thread range of each worker: they follow the default region back to back.
  SmallVector<std::pair<int32_t, int32_t>> workerRanges;
  int64_t next = boundary64;
  for (int32_t warps : ws.getPartitionNumWarps()) {
    int64_t end = next + static_cast<int64_t>(warps) * threadsPerWarp;
    if (end > std::numeric_limits<int32_t>::max())
      return ws.emitOpError("static partition boundary exceeds int32 range");
    workerRanges.emplace_back(static_cast<int32_t>(next),
                              static_cast<int32_t>(end));
    next = end;
  }

  SmallVector<ttg::WarpYieldOp> consumerYields;
  consumerRegion.walk(
      [&](ttg::WarpYieldOp op) { consumerYields.push_back(op); });
  if (consumerYields.empty())
    return ws.emitOpError("static partitions lost their terminators");

  Block *dispatch = ws->getBlock();
  Block *continuation =
      rewriter.splitBlock(dispatch, std::next(ws->getIterator()));
  Region &funcBody = func.getBody();
  auto &funcBlocks = funcBody.getBlocks();

  // Splice each worker in, followed by the block that tests the next worker,
  // so the dispatch is a chain: worker0? -> worker1? -> ... -> default.
  SmallVector<Block *> workerRoots;
  SmallVector<Block *> joins;
  for (Region *partition : ws.getPartitionRegions()) {
    SmallVector<ttg::WarpReturnOp> returns;
    partition->walk([&](ttg::WarpReturnOp op) { returns.push_back(op); });
    if (returns.empty())
      return ws.emitOpError("static partitions lost their terminators");
    workerRoots.push_back(&partition->front());
    funcBlocks.splice(continuation->getIterator(), partition->getBlocks());
    Block *join = rewriter.createBlock(&funcBody, continuation->getIterator());
    joins.push_back(join);
    for (ttg::WarpReturnOp op : returns) {
      rewriter.setInsertionPoint(op);
      cf::BranchOp::create(rewriter, op.getLoc(), join);
      rewriter.eraseOp(op);
    }
  }

  Block *consumerRoot = &consumerRegion.front();
  funcBlocks.splice(continuation->getIterator(), consumerRegion.getBlocks());
  for (ttg::WarpYieldOp op : consumerYields) {
    if (op.getNumOperands() != 0)
      return op.emitOpError(
          "mthreads TLE static consumer must not yield values");
    rewriter.setInsertionPoint(op);
    cf::BranchOp::create(rewriter, op.getLoc(), continuation);
    rewriter.eraseOp(op);
  }

  Location loc = ws.getLoc();
  rewriter.eraseOp(ws);
  rewriter.setInsertionPointToEnd(dispatch);
  Value tid =
      LLVM::CallIntrinsicOp::create(
          rewriter, loc, rewriter.getI32Type(),
          rewriter.getStringAttr("llvm.musa.read.ptx.sreg.tid.x"), ValueRange{})
          .getResult(0);

  // Thread bounds are materialized once, in the dispatch block, so every test
  // in the chain compares against the same SSA value.
  DenseMap<int32_t, Value> bounds;
  auto boundValue = [&](int32_t bound) {
    Value &value = bounds[bound];
    if (!value)
      value = arith::ConstantIntOp::create(rewriter, loc, bound, 32);
    return value;
  };
  boundValue(boundary);
  for (auto [index, range] : llvm::enumerate(workerRanges)) {
    boundValue(range.first);
    if (index + 1 != workerRanges.size())
      boundValue(range.second);
  }

  Block *test = dispatch;
  for (auto [index, range] : llvm::enumerate(workerRanges)) {
    rewriter.setInsertionPointToEnd(test);
    Value inWorker = arith::CmpIOp::create(
        rewriter, loc, arith::CmpIPredicate::uge, tid, boundValue(range.first));
    if (index + 1 != workerRanges.size()) {
      // Not the last worker, so bound the range from above as well.
      Value below =
          arith::CmpIOp::create(rewriter, loc, arith::CmpIPredicate::ult, tid,
                                boundValue(range.second));
      inWorker = arith::AndIOp::create(rewriter, loc, inWorker, below);
    }
    cf::CondBranchOp::create(rewriter, loc, inWorker, workerRoots[index],
                             ValueRange{}, joins[index], ValueRange{});
    test = joins[index];
  }

  rewriter.setInsertionPointToEnd(test);
  Value isConsumer = arith::CmpIOp::create(
      rewriter, loc, arith::CmpIPredicate::ult, tid, boundValue(boundary));
  cf::CondBranchOp::create(rewriter, loc, isConsumer, consumerRoot,
                           ValueRange{}, continuation, ValueRange{});
  return success();
}

class LowerWarpSpecializePass
    : public impl::TritonMUSAGPUTLELowerWarpSpecializeBase<
          LowerWarpSpecializePass> {
public:
  void runOnOperation() override {
    ModuleOp module = getOperation();
    SmallVector<ttg::WarpSpecializeOp> marked;
    module.walk([&](ttg::WarpSpecializeOp ws) {
      if (ws->hasAttr(kStaticWarpSpecializeAttr))
        marked.push_back(ws);
    });
    if (marked.empty())
      return;
    if (marked.size() != 1) {
      marked[1].emitOpError(
          "mthreads TLE static warp_specialize supports exactly one marked "
          "operation per module");
      return signalPassFailure();
    }

    auto func = marked.front()->getParentOfType<LLVM::LLVMFuncOp>();
    if (!func) {
      marked.front().emitOpError(
          "mthreads TLE late lowering requires an LLVM function");
      return signalPassFailure();
    }

    IRRewriter rewriter(&getContext());
    if (failed(lowerWarpGroupBarriers(func, marked.front(), rewriter)) ||
        failed(lowerStaticWarpSpecialize(func, marked.front(), rewriter)))
      signalPassFailure();
  }
};

} // namespace
} // namespace mlir

#endif // __TLE__
