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

  unsigned markBase(PtrStrideOp stride);
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
    setAttr<ArrayBaseAttr>(alloca, ctx, id++);
  }

  // Also consider decays from pointer to array.
  auto decays = findAll<CastOp>(func);
  for (auto cast : decays) {
    if (cast.getKind() != cir::CastKind::array_to_ptrdecay)
      continue;

    // The operand of the operation is an array.
    auto *operand = cast.getSrc().getDefiningOp();
    if (!operand)
      continue;

    unsigned currentId;
    if (operand->hasAttr(ArrayBaseAttr::getMnemonic()))
      currentId = getAttr<ArrayBaseAttr>(operand).getId();
    else
      setAttr<ArrayBaseAttr>(operand, ctx, currentId = id++);

    setAttr<ArrayBaseAttr>(cast, ctx, currentId);
  }

  // Climb the use-def chain of ptr_stride, to identify the base.
  auto strides = findAll<PtrStrideOp>(func);
  for (auto stride : strides)
    markBase(stride);
}

unsigned FindOffset::markBase(PtrStrideOp stride) {
  auto *pointer = stride.getBase().getDefiningOp();
  if (!pointer)
    return -1u;

  if (pointer->hasAttr(ArrayBaseAttr::getMnemonic())) {
    auto id = getAttr<ArrayBaseAttr>(pointer).getId();
    setAttr<ArrayBaseAttr>(stride, stride->getContext(), id);
    return id;
  }

  if (auto baseStride = dyn_cast<PtrStrideOp>(pointer)) {
    if (auto id = markBase(baseStride); id != -1u) {
      setAttr<ArrayBaseAttr>(stride, stride->getContext(), id);
      return id;
    }
  }

  return -1u;
}

void FindOffset::findOffsetInFunction(FuncOp func) {
}

} // namespace

std::unique_ptr<Pass> mlir::createCIRFindOffsetPass() {
  return std::make_unique<FindOffset>();
}
