#include "PassDetail.h"
#include "clang/CIR/Dialect/IR/CIRDialect.h"
#include "clang/CIR/Dialect/Passes.h"

using namespace mlir;
using namespace cir;
using namespace affine;

namespace {

struct RaiseToAffine : public RaiseToAffineBase<RaiseToAffine> {
  using RaiseToAffineBase::RaiseToAffineBase;
  
  void runOnOperation() override;
  void raise(ForOp op);
};

int64_t getConstant(cir::ConstantOp op) {
  auto attr = op.getValue();
  const auto intAttr = mlir::cast<cir::IntAttr>(attr);
  return intAttr.getValue().getSExtValue();
}

bool isLoopInvariant(mlir::Operation *op, ForOp loop) {
  if (!op || isa<ConstantOp>(op))
    return true;

  if (auto load = dyn_cast<LoadOp>(op)) {
    auto addr = load.getAddr();
    (void) addr;
    // Make sure the address isn't stored inside the loop.
    // TODO: This is very complicated.
    return true;
  }

  for (auto operand : op->getOperands()) {
    if (!isLoopInvariant(operand.getDefiningOp(), loop))
      return false;
  }
  return false;
}

void RaiseToAffine::raise(ForOp forOp) {
  bool earlyExit = false;
  forOp.walk([&](mlir::Operation *op) {
    if (mlir::isa<BreakOp, ContinueOp, ReturnOp>(op))
      earlyExit = true;
  });
  if (earlyExit)
    return;

  // Obtain step and find out the induction variable.
  auto *stepBlock =
      (forOp.maybeGetStep() ? &forOp.maybeGetStep()->front() : nullptr);
  assert(stepBlock && "Cannot find step block");

  // Try to match "iv = load addr; ++iv; store iv, addr; yield" to find step.
  // We should match the exact pattern, in case there's something unexpected:
  // we must rule out cases like `for (int i = 0; i < n; i++, printf("\n"))`.
  auto &oplist = stepBlock->getOperations();

  auto iterator = oplist.begin();

  // We might find constants at beginning. Skip them.
  // We could have hoisted them outside the for loop in previous passes, but
  // it hasn't been done yet.
  while (iterator != oplist.end() && isa<ConstantOp>(*iterator))
    ++iterator;

  if (iterator == oplist.end())
    return;

  auto load = dyn_cast<LoadOp>(*iterator);
  if (!load)
    return;

  // We assume this is the address of induction variable (IV). The operations
  // that come next will check if that's true.
  mlir::Value addr = load.getAddr();
  mlir::Value iv = load.getResult();

  // Then we try to match either "++IV" or "IV += n". Same for reversed loops.
  if (++iterator == oplist.end())
    return;

  mlir::Operation &arith = *iterator;

  int64_t step;
  if (auto unary = dyn_cast<UnaryOp>(arith)) {
    // Not operating on induction variable. Fail.
    if (unary.getInput() != iv)
      return;

    if (unary.getKind() == UnaryOpKind::Inc)
      step = 1;
    else if (unary.getKind() == UnaryOpKind::Dec)
      step = -1;
    else
      return;
  }

  if (auto binary = dyn_cast<BinOp>(arith)) {
    if (binary.getLhs() != iv)
      return;

    mlir::Value value = binary.getRhs();
    if (auto constValue = value.getDefiningOp<cir::ConstantOp>();
        constValue.getValueAttr<cir::IntAttr>())
      step = getConstant(constValue);

    if (binary.getKind() == BinOpKind::Add)
      ; // Nothing to do. Step has been calculated above.
    else if (binary.getKind() == BinOpKind::Sub)
      step = -step;
    else
      return;
  }

  // Check whether we immediately store this value into the appropriate place.
  if (++iterator == oplist.end())
    return;

  auto store = dyn_cast<StoreOp>(*iterator);
  if (!store || store.getAddr() != addr ||
      store.getValue() != arith.getResult(0))
    return;

  if (++iterator == oplist.end())
    return;

  // Finally, this should precede a yield with nothing in between.
  if (!isa<YieldOp>(*iterator))
    return;

  Value ivAddr = addr;

  // Check the increased variable is indeed the induction variable, by looking at the condition.
  auto &condBlock = forOp.getCond().getBlocks().front();
  auto term = cast<ConditionOp>(condBlock.getOperations().back());
  auto cmp = dyn_cast<CmpOp>(term.getCondition().getDefiningOp());

  // For now, we only deal with the less-than situation.
  if (!cmp || (cmp.getKind() != CmpOpKind::lt && cmp.getKind() != CmpOpKind::le))
    return;

  auto lhs = dyn_cast<LoadOp>(cmp.getLhs().getDefiningOp());
  if (!lhs || lhs.getAddr() != ivAddr)
    return;

  // TODO: Make sure the `cond` region doesn't have any side-effect.

  // Extract the upper bound and make sure it doesn't change.
  Value ub = cmp.getRhs();
  if (!isLoopInvariant(ub.getDefiningOp(), forOp))
    return;

  // Now we need to find the lower bound.
  auto *parentBlock = forOp->getBlock();
  Value lb;
  for (auto it = ++forOp->getReverseIterator(); it != parentBlock->rbegin(); it++) {
    if (it->getNumRegions() || isa<CallOp>(*it))
      return;
    if (auto store = dyn_cast<StoreOp>(*it); store.getAddr() == ivAddr) {
      lb = store.getValue();
      break;
    }
  }

  if (!lb)
    return;

  // Create an index cast for the lower and upper bounds.
  OpBuilder builder(forOp);
  mlir::MLIRContext *ctx = forOp->getContext();

  auto createCast = [&](mlir::Value value, bool addOne) {
    if (auto *def = value.getDefiningOp()) {
      if (def->getParentOp() == forOp)
        def->moveBefore(forOp);
      builder.setInsertionPointAfter(def);
    } else {
      builder.setInsertionPointToStart(value.getParentBlock());
    }
    if (addOne) {
      auto one = builder.create<ConstantOp>(value.getLoc(), IntAttr::get(IntType::get(ctx, 32, true), 1));
      value = builder.create<BinOp>(value.getLoc(), BinOpKind::Add, value, one);
    }
    auto cast = builder.create<IndexCastOp>(value.getLoc(), value);
    return cast;
  };

  auto upperBound = createCast(ub, cmp.getKind() == CmpOpKind::le);
  auto lowerBound = createCast(lb, false);

  auto idmap = AffineMap::get(0, 1, getAffineSymbolExpr(0, ctx));
  builder.setInsertionPoint(forOp);
  auto affineFor = builder.create<AffineForOp>(forOp.getLoc(), ValueRange { lowerBound }, idmap, ValueRange { upperBound }, idmap, step);
  auto &affineBlock = affineFor.getRegion().front();
  auto &forBlock = forOp.getBody().front();
  for (auto it = forBlock.begin(); it != forBlock.end();) {
    if (isa<YieldOp>(*it))
      break;
    auto next = it; next++;
    it->moveBefore(&affineBlock.back());
    it = next;
  }
  forOp.erase();

  // Load from `ivAddr` should be changed as well.
  llvm::SmallVector<LoadOp> loads;
  affineFor.walk([&](LoadOp loadOp) {
    if (loadOp.getAddr() == ivAddr)
      loads.push_back(loadOp);
  });
  if (!loads.empty()) {
    builder.setInsertionPointToStart(&affineBlock);
    Value cast = builder.create<FromIndexOp>(affineFor->getLoc(), loads.front().getResult().getType(), affineFor.getInductionVar());
    for (auto op : loads) {
      op.replaceAllUsesWith(cast);
      op.erase();
    }
  }

  // TODO: Make sure the ivAddr gets stored back.
}

void RaiseToAffine::runOnOperation() {
  std::vector<ForOp> loops;
  getOperation()->walk([&](ForOp op) {
    loops.push_back(op);
  });

  for (auto loop : loops)
    raise(loop);
}

} // namespace

std::unique_ptr<Pass> mlir::createCIRRaiseToAffinePass() {
  return std::make_unique<RaiseToAffine>();
}
