#include "PassDetail.h"
#include "FalconUtilities.h"
#include "mlir/Analysis/Presburger/Barvinok.h"
#include "mlir/Analysis/Presburger/IntegerRelation.h"
#include "clang/CIR/Dialect/IR/CIRDialect.h"
#include "clang/CIR/Dialect/Passes.h"

using namespace mlir;
using namespace cir;
using namespace affine;
using namespace mlir::presburger;
using namespace mlir::presburger::detail;

namespace {

struct ComputeDeps : public ComputeDepsBase<ComputeDeps> {
  using ComputeDepsBase::ComputeDepsBase;
  
  void runOnOperation() override;
};

using LoopBound = std::pair<AffineBound, AffineBound>;

// Checks whether an operation is a CIR operation representing a constant integer.
std::optional<int64_t> getConstantOnLoopBound(Value value) {
  mlir::Operation *defining = value.getDefiningOp();

  // By implementation of RaiseToAffine, we know `defining` must be a IndexCast.
  // The real value comes from the operand of that operation.
  auto indexCast = cast<IndexCastOp>(defining);
  mlir::Operation *underlying = indexCast.getSrc().getDefiningOp();

  if (!underlying)
    return std::nullopt;

  if (auto cst = dyn_cast<ConstantOp>(underlying))
    return cast<IntAttr>(cst.getValue()).getSInt();
  
  return std::nullopt;
}

// Checks whether all operands of `bound` are constants.
bool isConstantBound(AffineBound &bound) {
  auto operands = bound.getOperands();
  return std::all_of(operands.begin(), operands.end(), [](Value value) {
    return getConstantOnLoopBound(value).has_value();
  });
}

// Extracts coefficients of dimensions and symbols from an AffineExpr.
// All valuees
llvm::SmallVector<int64_t> extractCoefficients(AffineExpr expr, const llvm::SmallVector<Value> &symbolValues, int dims) {
  llvm::SmallVector<int64_t> result(dims + 1);

  if (auto c = dyn_cast<AffineConstantExpr>(expr)) {
    result.back() += c.getValue();
    return result;
  }
  
  if (auto d = dyn_cast<AffineDimExpr>(expr)) {
    result[d.getPosition()] += 1;
    return result;
  }

  if (auto s = dyn_cast<AffineSymbolExpr>(expr)) {
    Value value = symbolValues[s.getPosition()];
    result.back() += *getConstantOnLoopBound(value);
    return result;
  }

  if (auto bin = dyn_cast<AffineBinaryOpExpr>(expr)) {
    switch (bin.getKind()) {
    case AffineExprKind::Add: {
      auto l = extractCoefficients(bin.getLHS(), symbolValues, dims);
      auto r = extractCoefficients(bin.getRHS(), symbolValues, dims);
      for (auto [i, coeff] : llvm::enumerate(r))
        l[i] += coeff;

      return l;
    }

    case AffineExprKind::Mul:
      if (auto cst = dyn_cast<AffineConstantExpr>(bin.getLHS())) {
        auto coeffs = extractCoefficients(bin.getRHS(), symbolValues, dims);
        for (auto &coeff : coeffs)
          coeff *= cst.getValue();
        return coeffs;
      }

      if (auto cst = dyn_cast<AffineConstantExpr>(bin.getRHS())) {
        auto coeffs = extractCoefficients(bin.getLHS(), symbolValues, dims);
        for (auto &coeff : coeffs)
          coeff *= cst.getValue();
        return coeffs;
      }
      break;
    
    default:
      ; // fallthrough to the unreachable below.
    }
  }

  llvm_unreachable("unable to handle AffineExpr that isn't purely affine");
}

// Gives the bounds of every loop for the entire loop nest forest that `op` is in.
// When the bounds are not compile-time determined, returns `std::nullopt`.
std::optional<llvm::SmallVector<LoopBound>> getLoopBounds(mlir::Operation *op) {
  llvm::SmallVector<LoopBound> bounds;

  // Traverse through the loop nest.
  for (mlir::Operation *runner = op->getParentOp(); !isa<FuncOp, ModuleOp>(runner); runner = runner->getParentOp()) {
    if (auto affine = dyn_cast<AffineForOp>(runner)) {
      auto upper = affine.getUpperBound();
      auto lower = affine.getLowerBound();

      // We must make sure they refer to known constants.
      if (!isConstantBound(upper) || !isConstantBound(lower)) {
        return std::nullopt;
      }

      bounds.push_back({ lower, upper });
    }
  }

  return bounds;
}

void getReuseInstances(IntegerRelation &set) {

}

void ComputeDeps::runOnOperation() {
  auto *module = getOperation();
  auto getelems = findAll<GetElementOp>(module); 
  for (auto access : getelems) {
    auto baseAttr = getAttr<ArrayBaseAttr>(access);
    auto offsetAttr = getAttr<ArrayOffsetAttr>(access);
    
    // When base and/or offset is unknown, the operation will not be subject
    // to our analysis.
    if (!baseAttr || !offsetAttr)
      continue;

    auto accessMap = offsetAttr.getMap();
    unsigned dims = accessMap.getNumDims();
    assert(accessMap.getNumSymbols() == 0);
    auto bounds = getLoopBounds(access);
    if (!bounds)
      continue;

    llvm::SmallVector<GetElementOp> statementsAbove;
    // The anchor is the operation that has a region to which `access` belongs.
    // It is maintained to find out statements above `access`, as required by
    // the algorithm.
    mlir::Operation *anchor = access;
    for (mlir::Operation *runner = access->getParentOp(); !isa<FuncOp, ModuleOp>(runner);
         runner = runner->getParentOp(), anchor = anchor->getParentOp()) {

      // Collect all relevant operations before the anchor.
      auto *block = anchor->getBlock();
      for (auto it = ++anchor->getReverseIterator(); it != block->rend(); it++) {
        if (auto get = dyn_cast<GetElementOp>(*it))
          statementsAbove.push_back(get);
      }

      // We only analyze full loops, rather than subregions of operations
      // contained in loops.
      if (!isa<AffineForOp>(runner))
        continue;

      // Obtain a presburger set, which denotes values of induction variables
      // that make the statements `get` and `access` touch the same location of
      // an array.
      for (auto get : statementsAbove) {
        // It's only possible when the operations access the same array.
        if (auto attr = getAttr<ArrayBaseAttr>(get); !attr || attr.getId() != baseAttr.getId())
          continue;

        auto offset = getAttr<ArrayOffsetAttr>(get);
        if (!offset)
          continue;

        auto depMap = offset.getMap();

        // We haven't put any identifiers in these attributes. Rather, we give
        // each nested loop a different ID for their loop variables to distinguish
        // them.
        // Therefore, it might be possible that `depMap` has fewer dimensions (as
        // it can use fewer loop variables) and fewer symbols, but the same dimension
        // number will always refer to the same value.
        PresburgerSpace space = PresburgerSpace::getSetSpace(dims, 0);
        IntegerRelation sameLoc(/*numReservedInequalities=*/2 * dims,
          /*numReservedEqualities=*/1,
          /*numReservedCols=*/space.getNumVars() + 1, space);

        
        // The offsets must be equal, and the induction variables must be in
        // loop bounds.
        AffineExpr diff = depMap.getResult(0) - accessMap.getResult(0);
        sameLoc.addEquality(extractCoefficients(diff, {}, dims));

        // An extra element (the last one) for the constant.
        for (auto [i, bound] : llvm::enumerate(*bounds)) {
          auto [low, high] = bound;
          auto coeffLow = extractCoefficients(low.getMap().getResult(0), low.getOperands(), dims);
          auto coeffHigh = extractCoefficients(high.getMap().getResult(0), high.getOperands(), dims);
          
          // d_i < high;
          // That is, -d_i + high - 1 >= 0. 
          coeffHigh[i] = -1;
          coeffHigh.back() -= 1;
          sameLoc.addInequality(coeffHigh);

          // d_i - low >= 0
          for (auto &c : coeffLow)
            c *= -1;
          coeffLow[i] = 1;
          sameLoc.addInequality(coeffLow);
        }

        // Now, for `sameLoc`, we know the statement instance of `get`.
        getReuseInstances(sameLoc);
        auto diseq = eliminateEqualities(sameLoc);
        diseq.dump();
        /*
        auto decomp = computePolytopeGeneratingFunction(sameLoc);
        for (auto &[set, gf] : decomp) {
          if (set.isIntegerEmpty())
            continue;
          
          llvm::errs() << "Region:\n";
          set.dump();
          llvm::errs() << "Generating function:\n";
          gf.dump();
          llvm::errs() << "\n";
          auto quasi = computeNumTerms(gf).simplify().collectTerms();
          quasi.dump();
          llvm::errs() << "\n";
        }*/
      }
      
      // Clear the statements inside this loop. We only deal with ones between
      // the next layer and this loop in the next iteration.
      statementsAbove.clear();

      // TODO: other instances
    }
  }
}

} // namespace

std::unique_ptr<Pass> mlir::createComputeDepsPass() {
  return std::make_unique<ComputeDeps>();
}
