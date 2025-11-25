#include "PassDetail.h"
#include "FalconUtilities.h"
#include "clang/CIR/Dialect/IR/CIRDialect.h"
#include "clang/CIR/Dialect/Passes.h"

using namespace mlir;
using namespace cir;
using namespace affine;

namespace {

class FindOffset : public FindOffsetBase<FindOffset> {
  using FindOffsetBase::FindOffsetBase;

  void runOnOperation() override;
  void findBaseInFunction(FuncOp func);
  void findOffsetInFunction(FuncOp func);
private:
  // The ID to distinguish bases.
  unsigned id = 0;
  ModuleOp module;
};

void FindOffset::runOnOperation() {
  module = cast<ModuleOp>(getOperation());
  auto globals = findAll<GlobalOp>(module);
  for (auto glob : globals)
    setAttr<ArrayBaseAttr>(glob, glob->getContext(), id++);
  auto funcs = findAll<FuncOp>(module);
  for (auto func : funcs) {
    findBaseInFunction(func);
    findOffsetInFunction(func);
  }
}

unsigned markBase(PtrStrideOp stride) {
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

void FindOffset::findBaseInFunction(FuncOp func) {
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

  // Also consider getglobals.
  auto getglobals = findAll<GetGlobalOp>(func);
  for (auto getglob : getglobals) {
    auto symName = getglob.getName();
    Operation *sym = module.lookupSymbol(symName);
    GlobalOp global = cast<GlobalOp>(sym);
    if (auto base = getAttr<ArrayBaseAttr>(global))
      setAttr<ArrayBaseAttr>(getglob, ctx, base.getId());
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

  // Do similar things to GetElementOp.
  auto getelem = findAll<GetElementOp>(func);
  for (auto get : getelem) {
    mlir::Operation *base = get.getBase().getDefiningOp();
    if (base && base->hasAttr(ArrayBaseAttr::getMnemonic()))
      setAttr<ArrayBaseAttr>(get, ctx, getAttr<ArrayBaseAttr>(base).getId());
  }
}

int getNumSurroundingFor(mlir::Operation *op) {
  int number = 0;
  for (auto *runner = op; !isa<ModuleOp, FuncOp>(runner); runner = runner->getParentOp()) {
    if (isa<AffineForOp>(runner))
      number++;
  }
  return number;
}

// Gives a fixed ID to each induction variable in affine loops.
// The outermost loop is given ID 0, and the loop directly nested
// in it is given ID 1, and so on.
llvm::DenseMap<mlir::Operation*, int> getForIds(mlir::Operation *op) {
  llvm::DenseMap<mlir::Operation*, int> map;
  llvm::SmallVector<mlir::Operation*> operations;
  for (auto *runner = op; !isa<ModuleOp, FuncOp>(runner); runner = runner->getParentOp()) {
    if (!isa<AffineForOp>(runner))
      continue;

    // According to RaiseToFor, the first operation must be `from_index`.
    auto &first = runner->getRegion(0).front().front();
    assert(isa<FromIndexOp>(first));

    operations.push_back(&first);
  }

  // We enumerate from the innermost to outermost.
  // Therefore when we assign IDs, we must reverse the operations first.
  for (auto [id, op] : llvm::enumerate(llvm::reverse(operations))) {
    map[op] = id;
  }
  return map;
}

// Gets the size (in bytes) of the type in C++.
unsigned getSize(mlir::Type ty) {
  if (auto intTy = dyn_cast<IntType>(ty))
    return intTy.getWidth();
  if (isa<DoubleType>(ty))
    return 8;
  if (isa<SingleType>(ty))
    return 4;
  if (auto arrTy = dyn_cast<ArrayType>(ty))
    return arrTy.getSize() * getSize(arrTy.getElementType());
  
  ty.dump();
  llvm_unreachable("unknown type");
}

void FindOffset::findOffsetInFunction(FuncOp func) {
  auto accesses = findAll<GetElementOp>(func);
  auto *ctx = func->getContext();
  for (auto access : accesses) {
    int dimCount = getNumSurroundingFor(access);
    int symCount = 0;
    auto forId = getForIds(access);
    
    // Calculates offset from the base by recursively examining the operands, to
    // construct an affine expression.
    std::function<std::optional<AffineExpr>(mlir::Operation*)> calculateOffset =
    [&](mlir::Operation *op) -> std::optional<AffineExpr> {
      if (!op)
        return {};

      if (isa<AllocaOp>(op))
        return getAffineConstantExpr(0, ctx);

      if (auto cast = dyn_cast<CastOp>(op); cast && cast.getKind() == CastKind::array_to_ptrdecay)
        return getAffineConstantExpr(0, ctx);

      if (forId.count(op))
        return getAffineDimExpr(forId[op], ctx);

      if (auto constop = dyn_cast<ConstantOp>(op)) {
        if (auto intAttr = dyn_cast<IntAttr>(constop.getValue()))
          return getAffineConstantExpr(intAttr.getSInt(), ctx);
      }

      if (auto stride = dyn_cast<PtrStrideOp>(op)) {
        auto base = calculateOffset(stride.getBase().getDefiningOp());
        auto offset = calculateOffset(stride.getStride().getDefiningOp());
        unsigned size = getSize(stride.getType().getPointee());
        if (base && offset)
          return *base + *offset * size;
      }

      if (auto getelem = dyn_cast<GetElementOp>(op)) {
        auto base = calculateOffset(getelem.getBase().getDefiningOp());
        auto index = calculateOffset(getelem.getIndex().getDefiningOp());
        if (base && index)
          return *base + *index;
      }

      if (auto binop = dyn_cast<BinOp>(op)) {
        auto lhs = calculateOffset(binop.getLhs().getDefiningOp());
        auto rhs = calculateOffset(binop.getRhs().getDefiningOp());
        if (!lhs || !rhs)
          return {};

        switch (binop.getKind()) {
        case BinOpKind::Mul: return *lhs * *rhs;
        case BinOpKind::Add: return *lhs + *rhs;
        case BinOpKind::Sub: return *lhs - *rhs;
        default:
          return {};
        }
      }

      return {};
    };

    auto offset = calculateOffset(access);
    if (offset)
      setAttr<ArrayOffsetAttr>(access, ctx, AffineMap::get(dimCount, symCount, *offset));
  }
}

} // namespace

std::unique_ptr<Pass> mlir::createCIRFindOffsetPass() {
  return std::make_unique<FindOffset>();
}
