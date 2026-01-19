#include "PassDetail.h"
#include "mlir/Transforms/DialectConversion.h"
#include "clang/CIR/Dialect/IR/CIRDialect.h"
#include "clang/CIR/Dialect/Passes.h"

using namespace mlir;
using namespace cir;

namespace {

struct ConstFold : public ConstFoldBase<ConstFold> {
  using ConstFoldBase::ConstFoldBase;

  void runOnOperation() override;
  void foldFunction(FuncOp func);
};

bool isSignedInt(mlir::Type ty) {
  if (auto intTy = dyn_cast<IntType>(ty))
    return intTy.isSigned();
  return false;
}

mlir::LogicalResult foldBinary(BinOp op) {
  auto *lhsOp = op.getLhs().getDefiningOp();
  auto *rhsOp = op.getRhs().getDefiningOp();

  if (!lhsOp || !rhsOp || !isa<ConstantOp>(lhsOp) || !isa<ConstantOp>(rhsOp))
    return mlir::failure();

  auto lhsAttr = cast<ConstantOp>(lhsOp).getValue();
  auto rhsAttr = cast<ConstantOp>(rhsOp).getValue();
  if (!isSignedInt(lhsAttr.getType()) || !isSignedInt(rhsAttr.getType()))
    return mlir::failure();

  int64_t lhs = cast<IntAttr>(lhsAttr).getValue().getSExtValue();
  int64_t rhs = cast<IntAttr>(rhsAttr).getValue().getSExtValue();
  mlir::TypedAttr result;
  switch (op.getKind()) {
  case BinOpKind::Add:
    result = IntAttr::get(op.getResult().getType(), lhs + rhs);
    break;
  case BinOpKind::Sub:
    result = IntAttr::get(op.getResult().getType(), lhs - rhs);
    break;
  case BinOpKind::Mul:
    result = IntAttr::get(op.getResult().getType(), lhs * rhs);
    break;
  case BinOpKind::Div:
    result = IntAttr::get(op.getResult().getType(), lhs / rhs);
    break;
  default:
    return mlir::failure();
  }
  
  OpBuilder builder(op);
  auto value = builder.create<ConstantOp>(op.getLoc(), result);
  op->replaceAllUsesWith(value);
  op->erase();
  return mlir::success();
}

void ConstFold::runOnOperation() {
  llvm::SmallVector<FuncOp> funcs;
  llvm::SmallVector<BinOp> binops;
  mlir::Operation *module = getOperation();
  module->walk([&](FuncOp func) {
    funcs.push_back(func);
  });
  
  for (auto func : funcs)
    foldFunction(func);

  bool changed;
  do {
    binops.clear();
    module->walk([&](BinOp bin) {
      binops.push_back(bin);
    });

    changed = false;
    for (auto bin : binops)
      changed |= mlir::succeeded(foldBinary(bin));
  } while (changed);
}

void ConstFold::foldFunction(FuncOp func) {
  llvm::DenseMap<AllocaOp, llvm::SmallVector<StoreOp>> stores;
  llvm::DenseMap<AllocaOp, llvm::SmallVector<LoadOp>> loads;
  
  // Collect stores and loads to allocas in the function.
  func.walk([&](StoreOp op) {
    auto addr = op.getAddr();
    if (auto alloca = dyn_cast<AllocaOp>(addr.getDefiningOp()))
      stores[alloca].push_back(op);
  });
  func.walk([&](LoadOp op) {
    auto addr = op.getAddr();
    if (auto alloca = dyn_cast<AllocaOp>(addr.getDefiningOp()))
      loads[alloca].push_back(op);
  });

  for (auto &[alloca, stores] : stores) {
    if (stores.size() != 1)
      continue;
    auto value = stores[0].getValue();
    for (auto load : loads[alloca]) {
      load.replaceAllUsesWith(value);
      load.erase();
    }
    stores[0].erase();
    alloca.erase();
  }
}

} // namespace

std::unique_ptr<Pass> mlir::createConstFoldPass() {
  return std::make_unique<ConstFold>();
}
