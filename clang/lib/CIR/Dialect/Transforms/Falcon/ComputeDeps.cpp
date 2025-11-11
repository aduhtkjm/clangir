#include "PassDetail.h"
#include "FalconUtilities.h"
#include "mlir/Analysis/Presburger/Barvinok.h"
#include "mlir/Analysis/Presburger/IntegerRelation.h"
#include "mlir/Analysis/Presburger/Simplex.h"
#include "clang/CIR/Dialect/IR/CIRDialect.h"
#include "clang/CIR/Dialect/Passes.h"

using namespace mlir;
using namespace cir;
using namespace affine;
using namespace mlir::presburger;
using namespace mlir::presburger::detail;

namespace {

using IntVector = SmallVector<int64_t>;
using LoopBound = std::pair<AffineBound, AffineBound>;
using DependencyMap = DenseMap<Operation*, PresburgerRelation>;

struct ComputeDeps : public ComputeDepsBase<ComputeDeps> {
  using ComputeDepsBase::ComputeDepsBase;
  
  void runOnOperation() override;
  void markLexicalOrder();
  void markLoopInductionVars(Operation *op, SmallVector<unsigned> indices, unsigned &nextIndex);

  void addLexLess(PresburgerRelation &rel, IntegerRelation base, Operation *dep, Operation *sink, unsigned depDims, unsigned srcDims);

private:
  // For two operations S and T,
  // lexIndex[S] < lexIndex[T] iff. S is lexically before T in the IR.
  DenseMap<Operation*, unsigned> lexIndex;
  
  // Each loop induction variable is given a list of schedule index.
  // For example, consider this program:
  //
  //   for (int z = 0; z < ...; z++) {}
  //   for (int i = 0; i < ...; i++) {
  //     for (int j = 0; j < ...; j++) {}
  //     for (int k = 0; k < ...; k++) {}
  //   }
  //
  // The variables `z` might be given index { 1 },
  // and `i` { 2 }, `j` { 2, 1 }, `k` { 2, 2 } respectively.
  DenseMap<Operation*, SmallVector<unsigned>> loopVarIndex;
  friend class LessAccess;
};

// Checks whether an operation is a CIR operation representing a constant integer.
std::optional<int64_t> getConstantOnLoopBound(Value value) {
  Operation *defining = value.getDefiningOp();

  // By implementation of RaiseToAffine, we know `defining` must be a IndexCast.
  // The real value comes from the operand of that operation.
  auto indexCast = cast<IndexCastOp>(defining);
  Operation *underlying = indexCast.getSrc().getDefiningOp();

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
IntVector extractCoefficients(AffineExpr expr, const SmallVector<Value> &symbolValues, int dims) {
  IntVector result(dims + 1);

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
      for (auto [i, coeff] : enumerate(r))
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
std::optional<SmallVector<LoopBound>> getLoopBounds(Operation *op) {
  SmallVector<LoopBound> bounds;

  // Traverse through the loop nest.
  for (Operation *runner = op->getParentOp(); !isa<FuncOp, ModuleOp>(runner); runner = runner->getParentOp()) {
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

  // Make sure that the outermost loop comes first.
  std::reverse(bounds.begin(), bounds.end());
  return bounds;
}

unsigned findCommonLoopDepth(Operation *x, Operation *y) {
  SmallVector<Operation*> loopParents;
  for (auto *runner = x->getParentOp(); !isa<ModuleOp>(runner); runner = runner->getParentOp()) {
    if (isa<AffineForOp>(runner))
      loopParents.push_back(runner);
  }

  for (auto *runner = y->getParentOp(); !isa<ModuleOp>(runner); runner = runner->getParentOp()) {
    if (!isa<AffineForOp>(runner))
      continue;

    auto *it = std::find(loopParents.begin(), loopParents.end(), runner);
    if (it != loopParents.end())
      return loopParents.end() - it;
  }
  return 0;
}

/// Finds the outermost operation inside the function that contains op.
Operation *findOutermostOperation(Operation *op) {
  Operation *outermost = op;
  for (auto *runner = op->getParentOp(); !isa<FuncOp>(runner); runner = runner->getParentOp()) {
    if (isa<AffineForOp>(runner))
      outermost = runner;
  }
  return outermost;
}

void addBounds(IntegerRelation &rel, ArrayRef<LoopBound> loopBound, unsigned dims, unsigned offset) {
  for (auto [i, bound] : enumerate(loopBound)) {
    auto [low, high] = bound;
    // We must insert `dims(sinkMap)` elements at front.
    auto coeffLow = extractCoefficients(low.getMap().getResult(0), low.getOperands(), dims);
    auto coeffHigh = extractCoefficients(high.getMap().getResult(0), high.getOperands(), dims);
    // Adjust as done above.
    coeffHigh[i] -= 1;
    coeffHigh.back() -= 1;
    for (auto &c : coeffLow)
      c *= -1;
    coeffLow[i] += 1;

    IntVector shifted(dims + 1);

    for (unsigned i = 0; i < dims - offset; i++)
      shifted[i + offset] = coeffLow[i];
    shifted.back() = coeffLow.back();
    rel.addInequality(shifted);
    
    for (unsigned i = 0; i < dims - offset; i++)
      shifted[i + offset] = coeffHigh[i];
    shifted.back() = coeffHigh.back();
    rel.addInequality(shifted);
  }
}

void addInequality(PresburgerRelation &rel, ArrayRef<int64_t> ineq) {
  IntegerRelation constraint(rel.getSpace());
  constraint.addInequality(ineq);
  rel = rel.intersect(PresburgerRelation(constraint));
}

void addEquality(PresburgerRelation &rel, ArrayRef<int64_t> eq) {
  IntegerRelation constraint(rel.getSpace());
  constraint.addEquality(eq);
  rel = rel.intersect(PresburgerRelation(constraint));
}

DynamicAPInt countIntPointsWithoutParameters(const PresburgerRelation &rel) {
  // If `rel` is empty, then `countIntegerPoints` will return empty
  // quasi-linear affine functions.
  // To avoid the hassle later, we deal with the special case here.
  if (rel.isIntegerEmpty())
    return DynamicAPInt(0);

  auto v = countIntegerPoints(rel);
  unsigned chamberCount = 0;
  Fraction value;
  for (const auto &[chamber, result] : v) {
    if (chamber.isIntegerEmpty())
      continue;

    chamberCount++;
    value = result.getCoefficients().front();
  }
  assert(chamberCount == 1 && "there should only be 1 chamber without parameters!");
  return value.getAsInteger();
}

void ComputeDeps::markLexicalOrder() {
  unsigned id = 1;
  getOperation()->walk([&](GetElementOp op) {
    lexIndex[op] = id++;
  });
}

void ComputeDeps::markLoopInductionVars(mlir::Operation *op, SmallVector<unsigned> indices, unsigned &nextIndex) {
  if (auto forOp = dyn_cast<AffineForOp>(op)) {
    unsigned newLoopIndex = nextIndex++;
    indices.push_back(newLoopIndex);
    loopVarIndex[op] = indices;
  }

  for (Region &region : op->getRegions()) {
    for (Block &block : region.getBlocks()) {
      for (Operation &nestedOp : block) {
        markLoopInductionVars(&nestedOp, indices, nextIndex);
      }
    }
  }
}

/// Adds constraints `dep < sink` (lexicographically), based on the constraints
/// recorded in `base`. Assumes that 0.. srcDims are for `sink` and the rest are
/// for `deps`.
void ComputeDeps::addLexLess(PresburgerRelation &rel, IntegerRelation base,
  Operation *dep, Operation *sink, unsigned depDims, unsigned srcDims) {
  unsigned depth = findCommonLoopDepth(sink, dep);
  unsigned dims = depDims + srcDims;
  for (unsigned i = 0; i < std::max(depDims, srcDims); i++) {
    auto copy = base;

    // v_d[i] < v_s[i].
    // i.e. v_s[i] - v_d[i] - 1 >= 0.
    IntVector ineq(dims + 1);
    ineq[i + srcDims] = -1;
    ineq[i] = 1;
    ineq.back() = -1;
    copy.addInequality(ineq);
    rel.unionInPlace(PresburgerRelation(copy));

    // Add constraint v_d[i] = v_s[i] for the next iteration.
    IntVector eq(dims + 1);
    eq[i + srcDims] = -1;
    eq[i] = 1;
    base.addEquality(eq);

    if (i > depth) {
      // Now all common loop indices are already exhausted.
      // If `dep` goes lexically before `sink` in the sink program,
      // it will always be lexicographically smaller than `sink`.
      // Hence the whole `depend` will be added.
      if (lexIndex[dep] < lexIndex[sink])
        rel.unionInPlace(PresburgerRelation(base));

      // Otherwise, no instance of `dep` is lexicographically smaller
      // than sink, and we need to do nothing.
      
      // In both cases, there's no need to add further constraints.
      break;
    }
  }
}

class LessAccess {
  ComputeDeps *parent;

public:
  LessAccess(ComputeDeps *parent): parent(parent) {}
  bool operator()(Operation *l, Operation *r) const {
    assert(parent->lexIndex.count(l) && parent->lexIndex.count(r));
    return parent->lexIndex[l] < parent->lexIndex[r];
  }
};

void ComputeDeps::runOnOperation() {
  auto *module = getOperation();
  markLexicalOrder();
  unsigned nextIndex = 1;
  markLoopInductionVars(module, {}, nextIndex);

  // For testing purposes.
  int cacheInput;
  scanf("%d", &cacheInput);
  DynamicAPInt cacheSize(cacheInput);

  DynamicAPInt compulsoryMisses, capacityMisses;

  auto getelems = findAll<GetElementOp>(module); 
  for (auto sink : getelems) {
    auto baseAttr = getAttr<ArrayBaseAttr>(sink);
    auto offsetAttr = getAttr<ArrayOffsetAttr>(sink);
    
    // When base and/or offset is unknown, the operation will not be subject
    // to our analysis.
    if (!baseAttr || !offsetAttr)
      continue;

    auto sinkMap = offsetAttr.getMap();
    assert(sinkMap.getNumSymbols() == 0);
    unsigned srcDims = sinkMap.getNumDims();
    auto sinkBounds = getLoopBounds(sink);
    if (!sinkBounds)
      continue;

    auto sinkDomainSpace = PresburgerSpace::getSetSpace(srcDims);
    IntegerRelation sinkDomain(sinkDomainSpace);
    
    for (auto [i, bound] : enumerate(*sinkBounds)) {
      auto [low, high] = bound;
      // The extra `dims(depMap)` elements will be automatically zeroed.
      auto coeffLow = extractCoefficients(low.getMap().getResult(0), low.getOperands(), srcDims);
      auto coeffHigh = extractCoefficients(high.getMap().getResult(0), high.getOperands(), srcDims);
      
      // v_s[i] < high;
      // That is, -v_s[i] + high - 1 >= 0. 
      coeffHigh[i] -= 1;
      coeffHigh.back() -= 1;
      sinkDomain.addInequality(coeffHigh);

      // v_s[i] - low >= 0.
      for (auto &c : coeffLow)
        c *= -1;
      coeffLow[i] += 1;
      sinkDomain.addInequality(coeffLow);
    }

    // For every statement that sink depends on, record the `depend` relation,
    // defined below.
    // We compare lexIndex to maintain a well-defined order. If we use the
    // order between pointers, it is completely arbitrary.
    std::map<Operation*, PresburgerRelation, LessAccess> depsIndividual(LessAccess(this));

    // Check whether `sink` depends on `dep`.
    for (auto dep : getelems) {
      // It's only possible when the operation and sink access the same array.
      if (auto attr = getAttr<ArrayBaseAttr>(dep); !attr || attr.getId() != baseAttr.getId())
        continue;

      auto offset = getAttr<ArrayOffsetAttr>(dep);
      if (!offset)
        continue;

      auto depBounds = getLoopBounds(dep);
      if (!depBounds)
        continue;

      auto depMap = offset.getMap();

      // We haven't put any identifiers in these attributes. Rather, we give
      // each nested loop a different ID for their loop variables to distinguish
      // them.
      unsigned depDims = depMap.getNumDims();
      unsigned dims = depDims + srcDims;
      PresburgerSpace space = PresburgerSpace::getSetSpace(dims, 0);

      // Let's use the name S for the `sink` statement, and D for its dependence,
      // i.e. `dep`.
      //   depend = { (v_s, v_d) | S(v_s) depends on D(v_d) }.
      // 
      // It can be viewed as a function that maps induction variables of S, to
      // induction variables of D. However, this "function" is in fact multi-valued.
      // It records all dependencies, rather than the latest one.
      //
      // Moreover, it also carries constraints that all variables must be in bounds.
      IntegerRelation depend(/*numReservedInequalities=*/dims,
        /*numReservedEqualities=*/1,
        /*numReservedCols=*/space.getNumVars() + 1, space);
      
      // The offsets must be equal.
      auto depCoeff = extractCoefficients(depMap.getResult(0), {}, depDims);
      auto equality = extractCoefficients(sinkMap.getResult(0), {}, srcDims);

      // Concatenate the coefficients together, and pay special attention to constants
      // at the end.
      auto eqConstant = depCoeff.back();
      equality.pop_back();
      for (auto coeff : depCoeff)
        equality.push_back(-coeff);
      equality.back() += eqConstant;

      depend.addEquality(equality);

      // Add bound constraints.
      // An extra element (the last one) for the constant.
      addBounds(depend, *sinkBounds, dims, 0);
      addBounds(depend, *depBounds, dims, srcDims);

      // `dep` must be lexicographically before `sink`.
      PresburgerRelation rel = PresburgerRelation::getEmpty(depend.getSpace());
      addLexLess(rel, depend, dep, sink, depDims, srcDims);
      
      // Record this in the map.
      // Note that we can't use `deps[dep] = rel`, because operator[]
      // will attempt to default-construct a PresburgerRelation, which
      // doesn't have a default constructor.
      depsIndividual.insert({ dep, rel.simplify() });
    }

    // Currently, `depend` gives all instances of dependences between
    // the pair (sink, dep). We need to find out the lex-max of all
    // of them.
    
    // Each statement (as keys in `deps`) will be nested in zero or more loops.
    // They must be raised to a common space for presburger operations, so we
    // maintain a map from the indices of their loop variables to the indices
    // in the new space.
    //
    // We will use an example to illustrate the mapping. Suppose the statement
    // S depends on the following 3 statements D1, D2 and D3:
    //
    //   for (int z = 0; z < ...; z++) { D1 }
    //   for (int i = 0; i < ...; i++) {
    //     for (int j = 0; j < ...; j++) { D2 }
    //     for (int k = 0; k < ...; k++) { D3 }
    //   }
    //
    // Also suppose S's induction variables is called v_s.
    // We have obtained relations in `deps`, of the form { (v_s, z) }, { (v_s, i, j) }
    // and { (v_s, i, k) } respectively.
    // To raise them into a common space that is lexicographically comparable,
    // we can't simply raise it to { (v_s, z, i, j, k) }, as lexicographical
    // order in this space carries no meaning.
    // Instead, we'll raise it into
    //       { (v_s, t1, t2, t3, t4) }
    // where (v_s, z)    -> (v_s, 0, z, 0, 0)
    //       (v_s, i, j) -> (v_s, 1, i, 0, j)
    //       (v_s, i, k) -> (v_s, 1, i, 1, k)
    // Note that the constants are exactly the ones stored in `loopVarIndex`,
    // and the position of the dimensions only rely on loop depth.

    // Maps each variable to their dimension number in the common space.
    DenseMap<Operation*, unsigned> indexMap;

    // For every statement D_i, this stores the loops that contains D_i.
    // The innermost loop is at the beginning, and outermost loop is at
    // the end.
    DenseMap<Operation*, SmallVector<Operation*>> loopParents;

    // All induction variables we encountered in D_i's nesting loops.
    SmallVector<Operation*> indvars;

    // Maps each index vector to statement.
    // In the example above, it maps { 0 } to D1, { 1, 0 } to D2 and { 1, 1 }
    // to D3.
    DenseMap<SmallVector<unsigned>, Operation*> index2Statement;

    // The dimension of the common space.
    // This does not count the final constant. In other word, an equality/
    // inequality in the space should have `commonDims + 1` elements.
    unsigned commonDims = srcDims;

    for (const auto &[statement, _]: depsIndividual) {
      for (auto *runner = statement->getParentOp(); !isa<ModuleOp>(runner); runner = runner->getParentOp()) {
        if (isa<AffineForOp>(runner)) {
          indvars.push_back(runner);
          loopParents[statement].push_back(runner);
        }
      }
    }

    for (auto *indvar : indvars) {
      // Each dimension is raised to a pair: (index, value), as said above.
      // Hence the `* 2`.
      // Also, note indices should be one smaller than dimension count.
      unsigned loopDepth = loopVarIndex[indvar].size();
      unsigned index = (srcDims - 1) + loopDepth * 2;
      indexMap[indvar] = index;
      commonDims = std::max(commonDims, index + 1);
    }
    // We need an extra dimension for the statement id, the `lexIndex`.
    // This is to distinguish different statements in the same loop.
    commonDims++;

    // Now we construct this new relation, `depsTotal`.
    PresburgerSpace totalSpace = PresburgerSpace::getRelationSpace(srcDims, commonDims - srcDims);
    PresburgerRelation depsTotal = PresburgerRelation::getEmpty(totalSpace);
    for (const auto &[statement, relation] : depsIndividual) {
      for (const auto &disjunct : relation.getAllDisjuncts()) {
        IntegerRelation newRel(totalSpace);
        ArrayRef loopParent = loopParents[statement];
        SmallVector<Operation*> loopParentReversed(loopParent.size());
        std::copy(loopParent.rbegin(), loopParent.rend(), loopParentReversed.begin());
        Operation *innermostLoop = loopParent.front();

        // Record the map from indices to statement.
        SmallVector<unsigned> indices = loopVarIndex[innermostLoop];

        // Don't forget the statement identifier at the end.
        const unsigned statementId = lexIndex[statement];
        indices.push_back(statementId);
        index2Statement[indices] = statement;

        // Add equalities for this statement, that constrains loop
        // variable indices.
        for (auto [i, index] : llvm::enumerate(indices)) {
          SmallVector<DynamicAPInt> equality(commonDims + 1);
          equality[srcDims + i * 2] = 1;
          equality.back() = -index;
          newRel.addEquality(equality);
        }

        // If some indices are not present for this statement,
        // we zero out the corresponding entries.
        // Note the second-to-last entry, i.e. the last dimension is 
        // reserved for statement index, which is always present.
        for (unsigned i = indices.size() * 2 + srcDims; i < commonDims - 1; i++) {
          SmallVector<DynamicAPInt> equality(commonDims + 1);
          equality[i] = 1;
          newRel.addEquality(equality);
        }

        // An additional equality that specifies the statement id.
        SmallVector<DynamicAPInt> equality(commonDims + 1);
        equality[commonDims - 1] = 1;
        equality.back() = -statementId;
        newRel.addEquality(equality);

        const auto &raiseToCommonSpace = [&](const IntMatrix &matrix, bool isEquality) {
          for (unsigned i = 0; i < matrix.getNumRows(); i++) {
            SmallVector<DynamicAPInt> newRow(commonDims + 1);
            auto row = matrix.getRow(i);

            // The first `srcDims` elements are the same across all relations.
            for (unsigned j = 0; j < srcDims; j++)
              newRow[j] = row[j];

            // The rest needs to be mapped to their correct position.
            for (unsigned j = srcDims; j < row.size() - 1; j++) {
              // This is the (j - srcDims)'th outermost loop.
              // As loopParent goes from inner to outer, we need
              // to use its reversed version.
              Operation *affineLoop = loopParentReversed[j - srcDims];
              auto index = indexMap[affineLoop];
              // Position `index` represents the value.
              newRow[index] = row[j];
            }

            // Copy the constants.
            newRow.back() = row.back();

            if (isEquality)
              newRel.addEquality(newRow);
            else
              newRel.addInequality(newRow);
          }
        };

        raiseToCommonSpace(disjunct.getEqualities(), true);
        raiseToCommonSpace(disjunct.getInequalities(), false);
        newRel.simplify();
        depsTotal.unionInPlace(newRel);
      }
    }

    SymbolicLexOpt lexmax = depsTotal.simplify().findSymbolicIntegerLexMax();
    PWMAFunction deps = lexmax.lexopt;
    assert(deps.getDomainSpace().isCompatible(sinkDomainSpace));
    PresburgerRelation untouched(sinkDomain);

    for (const auto &piece : deps.getAllPieces())
      untouched = untouched.subtract(piece.domain);
    untouched = untouched.simplify();
    
    // The number of compulsory misses is equal to the number of integer points in `untouched`.
    // We're going to count it.
    compulsoryMisses += countIntPointsWithoutParameters(untouched);

    // Now, for each chamber of `deps`, we want to get the number of instances
    // between `sink(p)` and `deps(sink(p))`.
    // To do this, we enumerate statements between them.
    for (const auto &piece : deps.getAllPieces()) {
      if (piece.domain.isIntegerEmpty())
        continue;

      auto domain = piece.domain.simplify();
      auto output = piece.output;
      
      // First, extract the index values from `piece.output`.
      // These correspond to the exact statement.
      // TODO: index might not all have the same length - how do we do that?
      SmallVector<unsigned> indices;

      // Let the induction variables of deps(S(v_s)) be d.
      // Coefficients of d_i is specified by indexExpr[2*i+1].
      SmallVector<IntVector> d;
      for (unsigned i = 0, e = output.getNumOutputs(); i < e; i += 2) {
        const auto &indexExpr = output.getOutputExpr(i);
        // For any present index, we guarantee that it's at least 1.
        // The only situation that a zero appears is when the index is
        // zeroed out, i.e. it is not present. (See above.)
        if (indexExpr.back() == 0)
          break;

        // `indexExpr` should be a constant expression.
        // TODO: check indexExpr are all zeroes except the last element.
        indices.push_back((int64_t) indexExpr.back());

        // The last index, i.e. the statement index, is not followed by
        // any coefficient.
        if (i + 1 == e)
          break;
        const auto &coeffExpr = output.getOutputExpr(i + 1);
        d.push_back({});
        for (const auto &coeff : coeffExpr)
          d.back().push_back((int64_t) coeff);
      }

      assert(index2Statement.count(indices));
      Operation *dep = index2Statement[indices];

      // Now enumerate statements between `dep` and `sink`.
      // 
      // The range is from the outermost loop of dep, to the outermost
      // loop of sink.
      // 
      // Note that, for a statement u to be between them, 
      // it is not necessary that lexIndex[u] is between lexIndex[dep]
      // and lexIndex[sink]. It might be in the same loop with dep, but
      // lies before dep, for example.
      SmallVector<Operation*> inBetween;
      Operation *outermostDep = findOutermostOperation(dep);
      Operation *outermostSink = findOutermostOperation(sink);
      auto *stop = outermostSink->getNextNode();

      // This might include some more operations than needed.
      // But this won't affect correctness anyway, as their corresponding
      // "between" relation will be empty.
      for (auto *runner = outermostDep; runner != stop; runner = runner->getNextNode()) {
        auto stmts = findAll<GetElementOp>(runner);
        std::copy(stmts.begin(), stmts.end(), std::back_inserter(inBetween));
      }

      // For each collected operation `u`, compute the `reuseInstance` relation.
      // It is defined as:
      //   { (v_s, p) | deps(S(v_s)) < u(p) < S(v_s) }
      
      // All reuseInstances (composed with the corresponding access function)
      // is recorded in this vector.
      // For operations accessing the same array, we must raise them to a
      // common space, rather than summing the counts together directly.
      // We care only about *unique* accesses between sink and dep, and
      // a naive sum will count access to the same location multiple times.
      DenseMap<Operation*, std::vector<PresburgerRelation>> instances;

      for (Operation *u : inBetween) {
        auto uBounds = getLoopBounds(u);
        if (!uBounds)
          continue;

        // The dimension of `p` in the above definition.
        unsigned uDims = uBounds->size();
        unsigned totalDims = uDims + srcDims;

        PresburgerSpace space = PresburgerSpace::getRelationSpace(srcDims, uDims);
        IntegerRelation inBound(space);

        // Make sure `v_s` and `p` are both in bound.
        addBounds(inBound, *sinkBounds, totalDims, 0);
        addBounds(inBound, *uBounds, totalDims, srcDims);

        // Add lexicographical comparison.
        // The side u(p) < S(v_s) is essentially the same.
        auto base = PresburgerRelation::getEmpty(space);
        addLexLess(base, inBound, u, sink, uDims, srcDims);

        // Moreover, constrain v_s to be in `piece.domain`.
        // We can't intersect directly because they aren't in the same space.
        // We first convert all range variables in `domain` into domain variables,
        // to match that of `space`.
        auto domainCopy = domain;
        domainCopy.convertVarKind(VarKind::Range, 0, srcDims, VarKind::Domain, 0);
        // Then we insert `uDims` dummy (unconstrained) variables as range.
        domainCopy.insertVarInPlace(VarKind::Range, 0, uDims);
        base = base.intersect(domainCopy).simplify();

        // Now we handle the side deps(S(v_s)) < u(p).
        //
        // depDims might not be equal to srcDims. However, they are
        // expressible by those variables, according to piece.output calculated
        // above.
        unsigned depth = findCommonLoopDepth(u, dep);
        unsigned dims = uDims + srcDims;
        unsigned depDims = getLoopBounds(dep)->size();
        auto reuseInstance = PresburgerRelation::getEmpty(space);
        for (unsigned i = 0; i < std::max(uDims, depDims); i++) {
          auto copy = base;

          // d[i] < p[i].
          // i.e. p[i] - d[i] - 1 >= 0.
          IntVector ineq(dims + 1);
          // d[i] is determined by coefficients.
          // They should be placed to the dimensions of s, which are the dimensions at front.
          for (unsigned j = 0, e = d[i].size() - 1; j < e; j++)
            ineq[j] = -d[i][j];
          // The constant term should always be placed at back.
          ineq.back() -= d[i].back();

          ineq[i + srcDims] = 1;
          ineq.back() -= 1;
          addInequality(copy, ineq);
          reuseInstance.unionInPlace(copy);

          // Add constraint d[i] = p[i] for the next iteration.
          IntVector eq(dims + 1);
          for (unsigned j = 0, e = d[i].size() - 1; j < e; j++)
            eq[j] = d[i][j];
          eq.back() += d[i].back();
          eq[i + srcDims] = -1;
          addEquality(base, eq);

          if (i > depth) {
            if (lexIndex[dep] < lexIndex[sink])
              reuseInstance.unionInPlace(base);
            break;
          }
        }
        
        auto attr = getAttr<ArrayOffsetAttr>(u);
        auto getelem = cast<GetElementOp>(u);
        auto accessMap = attr.getMap();
        
        // For every point in the space of induction vector of u,
        // the statement accesses exactly one point.
        auto accessSpace = PresburgerSpace::getRelationSpace(uDims, 1);
        IntegerRelation accessRel(accessSpace);
        auto accessCoeffs = extractCoefficients(accessMap.getResult(0), {}, uDims);
        // The penultimate entry stands for this point.
        accessCoeffs.insert(accessCoeffs.end() - 1, -1);
        accessRel.addEquality(accessCoeffs);

        PresburgerRelation access(accessRel);
        reuseInstance.compose(access);
        if (reuseInstance.isIntegerEmpty())
          continue;

        reuseInstance = reuseInstance.computeReprWithOnlyDivLocals().simplify();

        // Clang IR doesn't use block arguments, so it's fine.
        instances[getelem.getBase().getDefiningOp()].push_back(reuseInstance);
      } // for each u in between

      for (const auto &[_, reuses] : instances) {
        // We must union the relations for the same array, because we only care
        // about *unique* accesses for each address.
        assert(!reuses.empty());
        PresburgerRelation total = reuses[0];
        for (unsigned i = 1, e = reuses.size(); i < e; i++)
          total.unionInPlace(reuses[i]);

        // Now for each v_s, `total` gives a set of addresses that `u` has accessed
        // between v_s and deps.
        // We hope to calculate the function that maps v_s to the cardinality
        // of this set, so we can view v_s as parameters and count the set.
        total.convertVarKind(VarKind::Domain, 0, srcDims, VarKind::Symbol, 0);
        total = total.simplify();
        auto result = countIntegerPoints(total);

        for (const auto &[region, count] : result) {
          if (region.isIntegerEmpty())
            continue;

          // Determine whether `count` exceeds cache size in this region.
          // The space here has only |v_s| range variables.
          PresburgerRelation rel(region);
          IntegerRelation exceed(region.getSpace());
          assert(region.getSpace().getNumVars() == srcDims);

          SmallVector<DynamicAPInt> ineq(srcDims + 1);
          for (unsigned i = 0, e = count.getCoefficients().size(); i < e; i++) {
            auto coeff = count.getCoefficients()[i];
            auto affine = count.getAffine()[i];
            if (affine.size() > 1)
              llvm_unreachable("NYI: more multiplications");
            
            // A constant.
            if (affine.size() == 0)
              ineq.back() += coeff.getAsInteger();

            // A single affine expression.
            if (affine.size() == 1) {
              const auto &expr = affine[0];
              assert(std::all_of(expr.begin(), expr.end(), [](const Fraction &f) { return f.num % f.den == 0; }));
              assert(expr.size() == srcDims + 1);
              for (unsigned i = 0; i < srcDims + 1; i++)
                ineq[i] += expr[i].getAsInteger();
            }
          }

          ineq.back() -= cacheSize;
          exceed.addInequality(ineq);
          rel = rel.intersect(PresburgerRelation(exceed));
          rel = rel.intersect(domain).simplify();

          auto misses = countIntPointsWithoutParameters(rel);
          capacityMisses += misses;
        }
      } // for each array base
    } // for each piece of lexmax
  } // for every sink

  llvm::errs() << "total compulsory misses = " << compulsoryMisses << "\n";
  llvm::errs() << "total capacity misses = " << capacityMisses << "\n";
  llvm::errs() << "total misses = " << capacityMisses + compulsoryMisses << "\n";
}

} // namespace

std::unique_ptr<Pass> mlir::createComputeDepsPass() {
  return std::make_unique<ComputeDeps>();
}
