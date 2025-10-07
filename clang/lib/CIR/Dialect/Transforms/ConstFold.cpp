#include "PassDetail.h"
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

void ConstFold::runOnOperation() {
  llvm::SmallVector<FuncOp> funcs;
  getOperation()->walk([&](FuncOp op) {
    funcs.push_back(op);
  });
  for (auto func : funcs)
    foldFunction(func);
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
    if (stores.size() > 1)
      continue;
    auto value = stores[0].getValue();
    for (auto load : loads[alloca])
      load.replaceAllUsesWith(value);
  }
}

} // namespace

std::unique_ptr<Pass> mlir::createConstFoldPass() {
  return std::make_unique<ConstFold>();
}
