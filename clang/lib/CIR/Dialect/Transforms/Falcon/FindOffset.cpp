#include "PassDetail.h"
#include "FalconUtilities.h"
#include "clang/CIR/Dialect/IR/CIRDialect.h"
#include "clang/CIR/Dialect/Passes.h"

using namespace mlir;
using namespace cir;
using namespace affine;

namespace {

struct FindOffset : public FindOffsetBase<FindOffset> {
  using FindOffsetBase::FindOffsetBase;

  void runOnOperation() override;
  void findBaseInFunction(FuncOp func);
  void findOffsetInFunction(FuncOp func);
};

void FindOffset::runOnOperation() {
  auto funcs = findAll<FuncOp>(getOperation());
  for (auto func : funcs) {
    findBaseInFunction(func);
    findOffsetInFunction(func);
  }
}

void FindOffset::findBaseInFunction(FuncOp func) {
  // The ID to distinguish bases.
  static unsigned id = 0;
  auto *ctx = func->getContext();

  // Mark allocas that represent arrays.
  auto allocas = findAll<AllocaOp>(func);
  for (auto alloca : allocas) {
    auto type = alloca.getType().getPointee();
    if (!isa<ArrayType>(type))
      continue;

    // This alloca holds an array.
    alloca->setAttr(ArrayBaseAttr::getMnemonic(), ArrayBaseAttr::get(ctx, id++));
  }
}

void FindOffset::findOffsetInFunction(FuncOp func) {
}

} // namespace

std::unique_ptr<Pass> mlir::createCIRFindOffsetPass() {
  return std::make_unique<FindOffset>();
}
