//===- Barvinok.cpp - Barvinok's Algorithm ----------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Analysis/Presburger/Barvinok.h"
#include "mlir/Analysis/Presburger/Simplex.h"
#include "mlir/Analysis/Presburger/Utils.h"
#include "llvm/ADT/Sequence.h"
#include <algorithm>

using namespace mlir;
using namespace presburger;
using namespace mlir::presburger::detail;

using IntVector = llvm::SmallVector<llvm::DynamicAPInt>;

namespace {
  
template<class T>
[[gnu::unused]] static void dumpVector(const std::string &name, const T &point) {
  llvm::errs() << name << " = "; llvm::interleaveComma(point, llvm::errs()); llvm::errs() << "\n";
}

// Matrix multiplication.
template<typename T>
T matmul(const T &a, const T &b) {
  assert(a.getNumColumns() == b.getNumRows());
  unsigned n = a.getNumRows();
  unsigned m = b.getNumRows();
  unsigned p = b.getNumColumns();
  T result(n, p);

  for (unsigned i = 0; i < n; i++) {
    for (unsigned j = 0; j < m; j++) {
      for (unsigned k = 0; k < p; k++) {
        result(i, k) += a(i, j) * b(j, k);
      }
    }
  }
  return result;
}

template<typename T>
[[nodiscard]] T hstack(T a, const T &b) {
  assert(a.getNumRows() == b.getNumRows());
  unsigned aCols = a.getNumColumns();
  unsigned bCols = b.getNumColumns();
  a.insertColumns(aCols, b.getNumColumns());
  for (unsigned i = 0; i < a.getNumRows(); i++)
    for (unsigned j = 0; j < bCols; j++)
      a(i, aCols + j) = b(i, j);
  return a;
}

FracMatrix asFracMatrix(const IntMatrix &a) {
  unsigned rows = a.getNumRows();
  unsigned cols = a.getNumColumns();
  FracMatrix mat(rows, cols);
  for (unsigned i = 0; i < rows; i++) {
    for (unsigned j = 0; j < cols; j++)
      mat(i, j) = a(i, j);
  }
  return mat;
}

IntMatrix asIntMatrix(const FracMatrix &a) {
  unsigned rows = a.getNumRows();
  unsigned cols = a.getNumColumns();
  IntMatrix mat(rows, cols);
  for (unsigned i = 0; i < rows; i++) {
    for (unsigned j = 0; j < cols; j++)
      mat(i, j) = a(i, j).getAsInteger();
  }
  return mat;
}

struct ParticularSolution {
  IntegerRelation constraint;
  // Row i is an affine expression for the i'th variable.
  FracMatrix solution;
};

ParticularSolution findParticularSolution(const IntMatrix &eqs, const IntMatrix &constants) {
  auto [u, d, v] = eqs.computeSmithNormalForm();
  
  // We are solving Ax = Bp + C. Now we've obtained UAV = D, where D is the Smith Normal Form.
  // Hence Ax = (U^{-1} D V^{-1}) x = Bp + C, which means D(V^{-1}x) = U(Bp + C).
  // We denote y = V^{-1}x, and we solve for y first.
  //
  // As D is diagonal, i.e. D = diag(s_1, ..., s_n, 0, ..., 0),
  // it is obvious that the equation is reduced to the following equalities:
  //   -  s_1 y_1 = (UBp + UC)_1
  //   -  s_2 y_2 = (UBp + UC)_2
  //   -  ...
  //
  // So y_n = (UBp + UC)_n / s_n for every s_n != 0. For those s_n = 0, we can simply set y_n = 0,
  // as we are only finding *a* solution rather than all solutions.
  // Note that it is not guaranteed that the values will always be divisible. If it is not,
  // then the parameters are constrained into a sublattice. This case is not handled currently.

  unsigned numRows = eqs.getNumRows();
  unsigned numCols = eqs.getNumColumns();
  assert(numRows == constants.getNumRows());
  unsigned numRhsCols = constants.getNumColumns();

  // Bp + C is stored as [B | C]. Hence to calculate UBp + UC, simply do a matmul.
  IntMatrix rhs = matmul(u, constants);

  // The result, in a form [y_B | y_C] that represents y_B p + y_C.
  FracMatrix y(numCols, numRhsCols);
  SmallVector<unsigned> modIndex, eqIndex;
  for (unsigned i = 0; i < numRows; i++) {
    MutableArrayRef row = rhs.getRow(i);
    // It is possible that there are more rows than columns. But by Smith Normal Form,
    // the extra rows are all zero.
    auto s = i >= numCols ? DynamicAPInt() : d(i, i);

    // (UBp + UC)_i is essentially (UB)[i, :] \cdot p + (UC)_i.
    if (s == 0 && std::any_of(row.begin(), row.end(), [](const DynamicAPInt &x) { return x != 0; }))
      // This is an equality constraint, e.g. 0x + 3N - 6 == 0 implies N == 2.
      eqIndex.push_back(i);

    if (s != 0 && std::any_of(row.begin(), row.end(), [&](const DynamicAPInt &x) { return x % s != 0; }))
      // This is a modular constraint, e.g. 2x + 3N - 5 == 0 implies (3N - 5) % 2 == 0.
      modIndex.push_back(i);

    // A constraint s_i y_i = (UB)[i, :] p + (UC)_i is found and must be enforced.
    if (s != 0) {
      for (unsigned j = 0; j < numRhsCols; j++)
        y(i, j) = Fraction(rhs(i, j), s);
    }
  }

  // For modular constraints, we use div-locals to capture them.
  unsigned numParams = numRhsCols - 1;
  unsigned numLocals = modIndex.size();
  unsigned dims = numParams + numLocals;
  auto space = PresburgerSpace::getSetSpace(numParams, 0, numLocals);
  IntegerRelation constraints(space);
  
  // First, we'll create the constraints of `p`.
  for (auto i : eqIndex) {
    // These means the RHS must be equal to zero, so we can directly add them
    // as constraints, modulo the locals.
    auto row = rhs.getRow(i);
    IntVector eq(dims + 1);
    std::copy(row.begin(), row.end() - 1, eq.begin());
    eq.back() = row.back();
    constraints.addEquality(eq);
  }

  for (auto [j, i] : llvm::enumerate(modIndex)) {
    // These will use the j'th local variable to express a modulus constraint.
    auto row = rhs.getRow(i);
    IntVector eq(dims + 1);
    std::copy(row.begin(), row.end() - 1, eq.begin());
    // This is the divisor.
    eq[numParams + j] = -d(i, i);
    eq.back() = row.back();
    constraints.addEquality(eq);
  }

  return { constraints, matmul(asFracMatrix(v), y) };
}

// The L_\infty-norm.
Fraction infNorm(ArrayRef<Fraction> fracs) {
  Fraction max = abs(fracs[0]);
  for (const auto &x : fracs) {
    if (max < abs(x))
      max = abs(x);
  }
  return max;
}

void normalize(IntMatrix &mat) {
  llvm::DynamicAPInt gcd;
  for (unsigned i = 0; i < mat.getNumRows(); i++) {
    for (unsigned j = 0; j < mat.getNumColumns(); j++) {
      gcd = llvm::gcd(gcd, abs(mat(i, j)));
      if (gcd == 1)
        return;
    }
  }
  
  for (unsigned i = 0; i < mat.getNumRows(); i++) {
    for (unsigned j = 0; j < mat.getNumColumns(); j++)
      mat(i, j) /= gcd;
  }
}

// See https://math.ucdavis.edu/~deloera/researchsummary/barvinokalgorithm-latte1.pdf
std::pair<IntVector, IntVector> getSamplePoint(const ConeV &cone) {
  unsigned numRows = cone.getNumRows();
  unsigned numCols = cone.getNumColumns();
  IntMatrix adjugate(numRows, numCols);
  auto det = cone.determinant(&adjugate);

  // Use LLL to find a reduced basis, one that the generators' length are smaller.
  if (det < 0) {
    for (unsigned i = 0; i < adjugate.getNumRows(); i++)
      adjugate.negateRow(i);
  }
  normalize(adjugate);
  FracMatrix basis = asFracMatrix(adjugate);
  basis.LLL(Fraction(3, 4));

  IntMatrix transform = asIntMatrix(matmul(basis, asFracMatrix(cone)));
  for (unsigned i = 0; i < numRows; i++)
    transform.normalizeRow(i);

  unsigned index = 0;
  Fraction shortest = infNorm(basis.getRow(0));
  for (unsigned i = 0; i < numRows; i++) {
    Fraction norm = infNorm(basis.getRow(i));
    if (norm < shortest) {
      shortest = norm;
      index = i;
    }
  }

  IntVector lambda;
  lambda.reserve(numRows);
  for (const auto &e : basis.getRow(index))
    lambda.push_back(e.getAsInteger());

  IntVector w(transform.getRow(index));
  if (std::all_of(lambda.begin(), lambda.end(), [](const DynamicAPInt &x) { return x <= 0; })) {
    for (auto &x : lambda)
      x *= -1;
    for (auto &x : w)
      x *= -1;
  }
  return { w, lambda };
}

std::vector<std::pair<int, ConeV>> unimodularDecomposeSimplicial(int sign, const ConeV &cone) {
  if (getIndex(cone) == 1)
    return { { sign, cone } };

  std::vector<std::pair<int, ConeV>> cones;
  auto [w, lambda] = getSamplePoint(cone);
  // In V-representation, we store generators as rows in a matrix.
  for (unsigned i = 0; i < cone.getNumRows(); i++) {
    if (lambda[i] == 0)
      continue;
    int lSign = lambda[i] > 0 ? 1 : -1;
    // The newly split cone is formed by replacing one ray into w.
    IntMatrix mat = cone;
    for (unsigned j = 0; j < cone.getNumColumns(); j++)
      mat(i, j) = w[j];
    assert(getIndex(mat) < getIndex(cone));
    auto decomp = unimodularDecomposeSimplicial(sign * lSign, mat);
    std::copy(decomp.begin(), decomp.end(), std::back_inserter(cones));
  }
  return cones;
}

} // namespace

std::pair<IntegerRelation, PolyhedronH> mlir::presburger::detail::eliminateEqualities(const PolyhedronH &poly) {
  auto paramSpace = PresburgerSpace::getSetSpace(poly.getNumSymbolVars());
  if (poly.getNumEqualities() == 0) {
    // We have to convert all div-locals into range variables, for other procedures
    // to work correctly.
    auto copy = poly;
    copy.convertVarKind(VarKind::Local, 0, poly.getNumLocalVars(), VarKind::Range);
    return { IntegerRelation::getUniverse(paramSpace), copy };
  }

  // Divlocals are unique and hence doesn't need special treatment.
  // They can be just viewed as normal variables.
  assert(poly.hasOnlyDivLocals());

  // We want to find another polyhedron H, such that it doesn't have equalities
  // and has the same amount of integer points with P.
  // Here, P = { x | Ax = Bp + C, Dx <= Ep + F }, where p is a vector of parameters.
  // 
  // We define A's kernel as { x | Ax = 0 }. It's also called the null space of A.
  // Suppose v_1, ..., v_n form a basis of the kernel. We denote the matrix
  // V = [ v_1 | ... | v_n ].
  // Also suppose x_0 is a solution of Ax = b,
  // then every solution of the equation can be written as Vy(p) + x_0,
  // where y is a function of parameters p. Therefore, we have:
  //
  //   #P = #{ y | D(Vy + x_0(p)) <= Ep + F }
  //
  // Here #P means the number of integer points in P.
  //
  // If V and x_0 both have integer entries, we can use Barvinok's algorithm
  // on the right-hand side. Now we calculate them in the following code.
  //
  // Note that it is not always possible to find an integer x_0.
  // When there is no parameters, i.e. p = [], the polyhedron does not contain
  // any integer points; when there is, however, I still cannot think of
  // a good way. Therefore, the current implementation will not consider the case of
  // parameters when doing such conversion.

  // To obtain the matrix A explained in the previous comments, we
  // strip out the coefficients for variables, removing the parts
  // for parameters.
  // As numCols and numRows are different for each matrix, we are not using these names.
  // The equality matrix is shaped m * (n + p + 1), where the last element is constant,
  // and the front n elements are coefficients of the domain, range and local variables.

  unsigned locals = poly.getNumLocalVars();
  unsigned dims = poly.getNumDimVars();
  unsigned n = dims + locals;
  unsigned p = poly.getNumSymbolVars();
  IntMatrix eqs = poly.getEqualities();
  unsigned m = eqs.getNumRows();
  
  // Reorder the rows of `eqs`, so that its symbol variables are at position n.. n + p.
  eqs.insertColumns(dims, locals);
  for (unsigned i = dims; i < n; i++) {
    // Add the original (p + i)'th column, which is the (i - dims)'th local's column,
    // to the new empty column.
    // After inserting `locals` columns, they are at position p + i + locals.
    eqs.addToColumn(/*sourceColumn=*/p + i + locals, /*targetColumn=*/i, 1);
  }
  // Remove the locals at the end to finish the reposition.
  eqs.removeColumns(n + p, locals);

  // `eqs` consists of [ A | -B ], as it is stored in "== 0" form.
  // We split it into two halves, `coeffs` (A) and `parameters + constants` (B).
  // Note that both ends are inclusive.
  IntMatrix coeffs = eqs.getSubMatrix(0, m, 0, n);
  IntMatrix constants = eqs.getSubMatrix(0, m, n, n + p + 1);
  // Flip constant sign, as mentioned above.
  for (unsigned i = 0; i < m; i++)
    constants.negateRow(i);

  // Here hnf is the Hermite Normal Form of A^T. We'll denote it as H.
  // U is the matrix such that UA^T = H, or the transformation matrix.
  // By construction, U is unimodular, i.e. both itself and its inverse
  // have integer entries.
  // A is shaped m * n, then H is n * m, and U is n * n.
  IntMatrix transpose = coeffs.transpose();
  auto [hnf, u] = transpose.computeRowHermiteNormalForm();

  // As UA^T = H, take transposition on both sides, then AU^T = H^T.
  // Let rank(A) = r, then rank(H) is also r. As H is in row-echelon
  // form (by definition of HNF), its last n - r rows must be all zero.
  // Therefore, the last n - r columns of H^T are zero.
  // Hence, the last n - r columns of U, say u_j ... u_n, satisfy that
  // Au_j = 0; they're in A's kernel.
  // 
  // By unimodularity, det(U) = 1, so U is not singular: rank(U) = n.
  // Hence its columns u_j ... u_n are linearly independent.
  // Moreover, as rank(A) = r, we have rank(ker(A)) = n - r, which is
  // exactly the same amount of these vectors.
  //
  // So the vectors are indeed a basis. We've got the integral basis
  // we want.
  // They are the last n - r columns of U^T, and it's just the last
  // n - r rows of U.

  // To obtain rank(H), we simply count non-zero rows.
  unsigned rank = 0;
  for (int i = 0, e = hnf.getNumRows(); i < e; i++) {
    const auto &row = hnf.getRow(i);
    bool hasNonZero = std::any_of(row.begin(), row.end(), [&](const DynamicAPInt &value) {
      return value != 0;
    });
    if (hasNonZero) {
      rank++;
    } else {
      // HNF is a row-echelon form, so when we find a row with all zeroes,
      // all rows after that are also zeroes.
      // Therefore we can break early.
      break;
    }
  }
  
  // The inequalities are stored in the form Dx + Ep + F >= 0.
  // So it's changed to D(Vy + x_0(p, 1)) + [E | F](p, 1) >= 0,
  // i.e. (DV)y + (Dx_0 + [E | F])(p, 1) >= 0.
  
  auto ineqs = poly.getInequalities();
  // Similarly, reorder the rows of ineqs as done above to equalities.
  ineqs.insertColumns(dims, locals);
  for (unsigned i = dims; i < n; i++)
    ineqs.addToColumn(/*sourceColumn=*/p + i + locals, /*targetColumn=*/i, 1);
  ineqs.removeColumns(n + p, locals);

  // Strip the parameter and constant part.
  unsigned ineqRows = ineqs.getNumRows();
  assert(ineqs.getNumColumns() == n + p + 1);

  // Find the particular solution as described above.
  auto [constraint, solution] = findParticularSolution(coeffs, constants); // n * (p + 1)
  if (constraint.isIntegerEmpty())
    return { IntegerRelation::getUniverse(paramSpace), PolyhedronH::getEmpty(poly.getSpace()) };

  if (rank >= n) {
    // When the matrix is full-ranked, the kernel is empty.
    // In this case, Ax = Bp + C has a unique solution or none at all.
    assert(rank == n);
    // If there is no parameter, then don't care about parameter space.
    // Moreover, if there is no inequality, there won't be extra constraints
    // that they bring. Therefore we can directly return as well.
    if (p == 0 || ineqRows == 0)
      return { constraint, PolyhedronH::getUniverse(PresburgerSpace::getSetSpace()) };
    
    // It isn't enough if we only have the constraints for Ax = Bp + C.
    // We must also take care of Dx + Ep + F >= 0.
    // We won't worry about this in non-0-dimensional cases, because we've done
    // chambering over the parameter space. However, in this case we won't go
    // to `computePolytopeGeneratingFunction` as that function does not support
    // 0-dimensional inputs, so we must handle the constraints ourselves.
    //
    // Now let's substitute the solution x = Up + V into it.
    // Then (DU + E)p + DV + F >= 0, which are new constraints of p.
    unsigned solRows = solution.getNumRows();

    auto d = asFracMatrix(ineqs.getSubMatrix(0, ineqRows, 0, n));
    auto e = asFracMatrix(ineqs.getSubMatrix(0, ineqRows, n, n + p));
    auto f = asFracMatrix(ineqs.getSubMatrix(0, ineqRows, n + p, n + p + 1));

    FracMatrix u = solution.getSubMatrix(0, solRows, 0, p);
    FracMatrix v = solution.getSubMatrix(0, solRows, p, p + 1);
    
    auto coeffP = matmul(d, u) + e;
    auto constP = matmul(d, v) + f;
    coeffP = hstack(coeffP, constP);

    // The real constraint should be the intersection of both constraints,
    // one from the equalities and the other from the inequalities.
    for (unsigned i = 0; i < ineqRows; i++) {
      auto row = coeffP.getRow(i);
      DynamicAPInt lcm(1);
      for (const auto &elem : row)
        lcm = llvm::lcm(lcm, elem.den);
      IntVector ineq(p + constraint.getNumLocalVars() + 1);
      for (unsigned i = 0; i < p + 1; i++)
        ineq[i] = (row[i] * lcm).getAsInteger();
      constraint.addInequality(ineq);
    }

    // Returns a singleton set.
    return { constraint, PolyhedronH::getUniverse(PresburgerSpace::getSetSpace()) };
  }

  // We transpose because we want the column vectors to form the basis,
  // rather than row vectors.
  FracMatrix basis = asFracMatrix(u.getSubMatrix(rank, n, 0, n).transpose());
  auto ineqCoeffs = asFracMatrix(ineqs.getSubMatrix(0, ineqRows, 0, n));
  FracMatrix rhs = matmul(ineqCoeffs, solution);
  FracMatrix newCoeffs = matmul(ineqCoeffs, basis);

  // Add a new column for the constants, and calculate the value.
  // Be careful that `ineqCoeffs` has fewer columns after the matmul.
  auto before = asFracMatrix(ineqs.getSubMatrix(0, ineqRows, n, n + p + 1));
  rhs += before;
  newCoeffs = hstack(newCoeffs, rhs);

  auto numVars = newCoeffs.getNumColumns() - 1;
  auto space = PresburgerSpace::getSetSpace(numVars - p, p);
  auto resultPoly = PolyhedronH(ineqRows, 0, numVars + 1, space);
  for (unsigned i = 0; i < ineqRows; i++) {
    auto row = newCoeffs.getRow(i);
    DynamicAPInt lcm(1);
    for (const auto &elem : row)
      lcm = llvm::lcm(lcm, elem.den);
    IntVector ineq(numVars + 1);
    for (unsigned i = 0; i < numVars + 1; i++)
      ineq[i] = (row[i] * lcm).getAsInteger();
    resultPoly.addInequality(ineq);
  }
  resultPoly.simplify();
  return { constraint, resultPoly };
}

static IntMatrix getAffineHull(const PolyhedronH &poly) {
  IntMatrix mat(0, poly.getNumCols());
  Simplex simplex(poly);
  for (unsigned i = 0, e = poly.getNumInequalities(); i < e; i++) {
    const auto &ineq = poly.getInequality(i);
    auto maybeMinimum = simplex.computeOptimum(Simplex::Direction::Down, poly.getInequality(i));
    auto maybeMaximum = simplex.computeOptimum(Simplex::Direction::Up, poly.getInequality(i));
    if (!maybeMinimum.isBounded() || !maybeMaximum.isBounded())
      continue;
    Fraction minimum = maybeMaximum.getBoundedOptimum();
    Fraction maximum = maybeMaximum.getBoundedOptimum();
    if (minimum == maximum && minimum == 0) {
      mat.insertRow(mat.getNumRows());
      auto row = mat.getRow(mat.getNumRows() - 1);
      std::copy(ineq.begin(), ineq.end(), row.begin());
    }
  }
  return mat;
}

std::pair<IntegerRelation, PolyhedronH> mlir::presburger::detail::projectToFullDimension(PolyhedronH poly) {
  poly.simplify();
  auto affineHull = getAffineHull(poly);
  if (affineHull.getNumRows() == 0 && poly.getNumEqualities() == 0) {
    auto paramSpace = PresburgerSpace::getSetSpace(poly.getNumSymbolVars());
    poly.convertVarKind(VarKind::Local, 0, poly.getNumLocalVars(), VarKind::Range);
    return { IntegerRelation::getUniverse(paramSpace), poly };
  }

  IntegerRelation newpoly(poly.getSpace());
  Simplex simplex(poly);
  for (unsigned i = 0; i < poly.getNumInequalities(); i++) {
    auto maybeMinimum = simplex.computeOptimum(Simplex::Direction::Down, poly.getInequality(i));
    auto maybeMaximum = simplex.computeOptimum(Simplex::Direction::Up, poly.getInequality(i));
    if (!maybeMinimum.isBounded() || !maybeMaximum.isBounded()) {
      newpoly.addInequality(poly.getInequality(i));
      continue;
    }
    Fraction minimum = maybeMaximum.getBoundedOptimum();
    Fraction maximum = maybeMaximum.getBoundedOptimum();
    if (minimum == maximum && minimum == 0)
      newpoly.addEquality(poly.getInequality(i));
    else
      newpoly.addInequality(poly.getInequality(i));
  }
  for (unsigned i = 0; i < poly.getNumEqualities(); i++)
    newpoly.addEquality(poly.getEquality(i));

  newpoly.simplify();
  newpoly.removeTrivialRedundancy();
  auto result = eliminateEqualities(newpoly);
  if (getAffineHull(result.second).getNumRows() != 0) {
    auto [constraint, r] = projectToFullDimension(result.second);
    return { constraint.intersect(result.first), r };
  }
  return result;
}

/// Assuming that the input cone is pointed at the origin,
/// converts it to its dual in V-representation.
/// Essentially we just remove the all-zeroes constant column.
ConeV mlir::presburger::detail::getDual(ConeH cone) {
  unsigned numIneq = cone.getNumInequalities();
  unsigned numVar = cone.getNumCols() - 1;
  ConeV dual(numIneq, numVar, 0, 0);
  // Assuming that an inequality of the form
  // a1*x1 + ... + an*xn + b ≥ 0
  // is represented as a row [a1, ..., an, b]
  // and that b = 0.

  for (auto i : llvm::seq<int>(0, numIneq)) {
    assert(cone.atIneq(i, numVar) == 0 &&
           "H-representation of cone is not centred at the origin!");
    for (unsigned j = 0; j < numVar; ++j)
      dual.at(i, j) = cone.atIneq(i, j);

    dual.normalizeRow(i);
  }

  // Now dual is of the form [ [a1, ..., an] , ... ]
  // which is the V-representation of the dual.
  return dual;
}

/// Converts a cone in V-representation to the H-representation
/// of its dual, pointed at the origin (not at the original vertex).
/// Essentially adds a column consisting only of zeroes to the end.
ConeH mlir::presburger::detail::getDual(ConeV cone) {
  unsigned rows = cone.getNumRows();
  unsigned columns = cone.getNumColumns();
  ConeH dual = defineHRep(columns);
  // Add a new column (for constants) at the end.
  // This will be initialized to zero.
  cone.insertColumn(columns);

  for (unsigned i = 0; i < rows; ++i)
    dual.addInequality(cone.getRow(i));

  // Now dual is of the form [ [a1, ..., an, 0] , ... ]
  // which is the H-representation of the dual.
  return dual;
}

/// Find the index of a cone in V-representation.
DynamicAPInt mlir::presburger::detail::getIndex(const ConeV &cone) {
  if (cone.getNumRows() > cone.getNumColumns())
    return DynamicAPInt(0);

  return abs(cone.determinant());
}

/// Compute the generating function for a unimodular cone.
/// This consists of a single term of the form
/// sign * x^num / prod_j (1 - x^den_j)
///
/// sign is either +1 or -1.
/// den_j is defined as the set of generators of the cone.
/// num is computed by expressing the vertex as a weighted
/// sum of the generators, and then taking the floor of the
/// coefficients.
GeneratingFunction
mlir::presburger::detail::computeUnimodularConeGeneratingFunction(
    ParamPoint vertex, int sign, const ConeH &cone) {
  // Consider a cone with H-representation [0  -1].
  //                                       [-1 -2]
  // Let the vertex be given by the matrix [ 2  2   0], with 2 params.
  //                                       [-1 -1/2 1]

  // `cone` must be unimodular.
  assert(getIndex(getDual(cone)) == 1 && "input cone is not unimodular!");

  unsigned numVar = cone.getNumVars();
  unsigned numIneq = cone.getNumInequalities();

  // Thus its ray matrix, U, is the inverse of the
  // transpose of its inequality matrix, `cone`.
  // The last column of the inequality matrix is null,
  // so we remove it to obtain a square matrix.
  //
  // We must also normalize each inequality of the cone. This is done
  // in the simplify() call.
  auto normalizedCone = cone;
  normalizedCone.simplify();
  FracMatrix transp = FracMatrix(normalizedCone.getInequalities()).transpose();
  transp.removeRow(numVar);

  FracMatrix generators(numVar, numIneq);
  transp.determinant(/*inverse=*/&generators); // This is the U-matrix.
  // Thus the generators are given by U = [2  -1].
  //                                      [-1  0]

  // The powers in the denominator of the generating
  // function are given by the generators of the cone,
  // i.e., the rows of the matrix U.
  std::vector<Point> denominator(numIneq);
  ArrayRef<Fraction> row;
  for (auto i : llvm::seq<int>(0, numVar)) {
    row = generators.getRow(i);
    denominator[i] = Point(row);
  }

  // The vertex is v \in Z^{d x (n+1)}
  // We need to find affine functions of parameters λ_i(p)
  // such that v = Σ λ_i(p)*u_i,
  // where u_i are the rows of U (generators)
  // The λ_i are given by the columns of Λ = v^T U^{-1}, and
  // we have transp = U^{-1}.
  // Then the exponent in the numerator will be
  // Σ -floor(-λ_i(p))*u_i.
  // Thus we store the (exponent of the) numerator as the affine function -Λ,
  // since the generators u_i are already stored as the exponent of the
  // denominator. Note that the outer -1 will have to be accounted for, as it is
  // not stored. We will insert the negation in substituteMuInTerm().

  unsigned numColumns = vertex.getNumColumns();
  unsigned numRows = vertex.getNumRows();
  ParamPoint numerator(numColumns, numRows);
  SmallVector<Fraction> ithCol(numRows);
  for (auto i : llvm::seq<int>(0, numColumns)) {
    for (auto j : llvm::seq<int>(0, numRows))
      ithCol[j] = vertex(j, i);
    numerator.setRow(i, transp.preMultiplyWithRow(ithCol));
    numerator.negateRow(i);
  }
  // Therefore Λ will be given by [ 1    0 ] and the negation of this will be
  //                              [ 1/2 -1 ]
  //                              [ -1  -2 ]
  // stored as the numerator.
  // Algebraically, the numerator exponent is
  // [ -2 ⌊ - N - M/2 + 1 ⌋ + 1 ⌊ 0 + M + 2 ⌋ ] -> first  COLUMN of U is [2, -1]
  // [  1 ⌊ - N - M/2 + 1 ⌋ + 0 ⌊ 0 + M + 2 ⌋ ] -> second COLUMN of U is [-1, 0]

  return GeneratingFunction(numColumns - 1, SmallVector<int>(1, sign),
                            std::vector({numerator}),
                            std::vector({denominator}));
}

/// We use Gaussian elimination to find the solution to a set of d equations
/// of the form
/// a_1 x_1 + ... + a_d x_d + b_1 m_1 + ... + b_p m_p + c = 0
/// where x_i are variables,
/// m_i are parameters and
/// a_i, b_i, c are rational coefficients.
///
/// The solution expresses each x_i as an affine function of the m_i, and is
/// therefore represented as a matrix of size d x (p+1).
/// If there is no solution, we return null.
std::optional<ParamPoint>
mlir::presburger::detail::solveParametricEquations(FracMatrix equations) {
  // equations is a d x (d + p + 1) matrix.
  // Each row represents an equation.
  unsigned d = equations.getNumRows();
  unsigned numCols = equations.getNumColumns();

  // If the determinant is zero, there is no unique solution.
  // Thus we return null.
  if (FracMatrix(equations.getSubMatrix(/*fromRow=*/0, /*toRow=*/d,
                                        /*fromColumn=*/0,
                                        /*toColumn=*/d))
          .determinant() == 0)
    return std::nullopt;

  // Perform row operations to make each column all zeros except for the
  // diagonal element, which is made to be one.
  for (unsigned i = 0; i < d; ++i) {
    // First ensure that the diagonal element is nonzero, by swapping
    // it with a row that is non-zero at column i.
    if (equations(i, i) == 0) {
      for (unsigned j = i + 1; j < d; ++j) {
        if (equations(j, i) == 0)
          continue;
        equations.swapRows(j, i);
        break;
      }
    }

    Fraction diagElement = equations(i, i);

    // Apply row operations to make all elements except the diagonal to zero.
    for (unsigned j = 0; j < d; ++j) {
      if (i == j)
        continue;
      if (equations(j, i) == 0)
        continue;
      // Apply row operations to make element (j, i) zero by subtracting the
      // ith row, appropriately scaled.
      Fraction currentElement = equations(j, i);
      equations.addToRow(/*sourceRow=*/i, /*targetRow=*/j,
                         /*scale=*/-currentElement / diagElement);
    }
  }

  // Rescale diagonal elements to 1.
  for (unsigned i = 0; i < d; ++i)
    equations.scaleRow(i, 1 / equations(i, i));

  // Now we have reduced the equations to the form
  // x_i + b_1' m_1 + ... + b_p' m_p + c' = 0
  // i.e. each variable appears exactly once in the system, and has coefficient
  // one.
  //
  // Thus we have
  // x_i = - b_1' m_1 - ... - b_p' m_p - c
  // and so we return the negation of the last p + 1 columns of the matrix.
  //
  // We copy these columns and return them.
  ParamPoint vertex =
      equations.getSubMatrix(/*fromRow=*/0, /*toRow=*/d,
                             /*fromColumn=*/d, /*toColumn=*/numCols);
  vertex.negateMatrix();
  return vertex;
}

// Triangulate with the DeLaunay's method.
std::vector<ConeV> triangulate(const ConeV &dual) {
  std::vector<ConeV> triangles;
  unsigned numRays = dual.getNumRows();
  unsigned dim = dual.getNumColumns();
  IntMatrix extendedRays = dual;
  // Raise the rays from R^d to points in R^{d+1} by placing them on a parabola.
  extendedRays.insertColumn(extendedRays.getNumColumns());
  for (unsigned i = 0; i < numRays; i++) {
    DynamicAPInt sum(0);
    auto row = extendedRays.getRow(i);
    for (unsigned j = 0, e = row.size(); j < e; j++) {
      row[j] *= 1000;
      sum += row[j] * row[j];
    }
    extendedRays.getRow(i).back() = sum + i;
  }

  // We don't want a full double-description method here.
  // Since we'd expect that the dimension of `dual` isn't large,
  // and the function is rarely called, let's do it brute-force.
  //
  // We enumerate all combinations of size `dim`, denoted as
  // p_1, ..., p_d.
  // These vectors are *points* rather than *rays* now, so when added
  // so they form a hyperplane of dimension `dim` with origin, which might be
  // a facet.
  //
  // We solve the equation pn = 0 for the normal vector `n`.
  // To avoid infinite solutions, we let the last element of `n` be 1.
  SmallVector<int> indicator(numRays);
  for (unsigned i = numRays - dim; i < numRays; ++i)
    indicator[i] = 1;
  do {
    auto [subset, remainder] = extendedRays.splitByBitset(indicator);
    auto maybeSolution = solveParametricEquations(FracMatrix(subset));
    if (!maybeSolution)
      continue;
    
    // Since there is no parameter, the solution should have a single column.
    auto solution = maybeSolution->transpose();
    assert(solution.getNumRows() == 1);
    DynamicAPInt lcm(1);
    for (const auto &f : solution.getRow(0))
      lcm = llvm::lcm(lcm, f.den);

    std::vector<DynamicAPInt> normal;
    for (const auto &f : solution.getRow(0))
      normal.push_back((f * lcm).getAsInteger());

    // Now we've got a hyperplane `nx = c`. But the orientation is
    // not yet decided; we could have -nx = -c and use `-n` as the normal.
    // 
    // We need to adjust the hyperplane such that its normal points to the side that
    // doesn't contain the polytope. This is achieved by requiring nx <= c for all other
    // vertices.
    // An 1D example:
    //  ----|--------------|----
    //      |x=1           |x=2
    // Here, the supporting hyperplane at x=1 has n=-1, so that for x=2, we have nx <= -1;
    // The one at x=2 has n=1/2, so that for x=1, we have x/2 <= 1.
    //
    // If we flipped the signs, then the hyperplane at x=1 becomes x=1 rather than -x=-1 above;
    // this will mean a requirement of x <= 1, which isn't satisfied by x=2.

    bool neg = true, pos = true;
    for (unsigned i = 0, e = remainder.getNumRows(); i < e; i++) {
      auto vertex = remainder.getRow(i);
      auto product = std::inner_product(normal.begin(), normal.end(), vertex.begin(), Fraction(0));
      // Don't forget the actual n is [..normal, 1], scaled by lcm.
      product += vertex.back() * lcm;
      // After introducing a small perturbation, no d+1 points will be on the same facet.
      assert(product != 0);
      if (product > 0)
        neg = false;
      if (product < 0)
        pos = false;
    }
    // It is impossible that nx-c is both positive and negative for every vertex.
    assert(!(neg && pos));

    // If nx - c is negative for some vertices and positive for others, the vertices
    // are actually separated (lie on both sides) by this hyperplane.
    // This means it isn't a facet.
    if (!neg && !pos)
      continue;
    // When everything is positive, this means we need to flip the signs,
    // since we expected that everything is negative.
    // We only care about lower facets, i.e. those whose last dimensions' normal is
    // negative.
    auto last = 1;
    if (pos)
      last *= -1;
    if (last >= 0)
      continue;

    // Now the rays that produced this facet is just the triangulated cone.
    subset.removeColumn(subset.getNumColumns() - 1);
    triangles.emplace_back(subset);
  } while (std::next_permutation(indicator.begin(), indicator.end()));
  return triangles;
}

std::vector<std::pair<int, ConeH>> mlir::presburger::detail::unimodularDecompose(const ConeH &tangentCone) {
  auto dual = getDual(tangentCone);
  DynamicAPInt index = getIndex(dual);
  // The cone is already unimodular and needs no further decomposition.
  if (index == 1)
    return { { 1, tangentCone } };

  std::vector<std::pair<int, ConeV>> cones;
  // The cone is not simplicial; we must triangulate it.
  if (index != 0)
    cones = unimodularDecomposeSimplicial(1, dual);
  else {
    auto triangulated = triangulate(dual);
    for (const auto &cone : triangulated) {
      auto single = unimodularDecomposeSimplicial(1, cone);
      std::copy(single.begin(), single.end(), std::back_inserter(cones));
    }
  }

  std::vector<std::pair<int, ConeH>> result;
  result.reserve(cones.size());

  for (const auto &[sign, conev] : cones)
    result.emplace_back(sign, getDual(conev));
  return result;
}

/// This is an implementation of the Clauss-Loechner algorithm for chamber
/// decomposition.
///
/// We maintain a list of pairwise disjoint chambers and the generating
/// functions corresponding to each one. We iterate over the list of regions,
/// each time adding the current region's generating function to the chambers
/// where it is active and separating the chambers where it is not.
///
/// Given the region each generating function is active in, for each subset of
/// generating functions the region that (the sum of) precisely this subset is
/// in, is the intersection of the regions that these are active in,
/// intersected with the complements of the remaining regions.
std::vector<std::pair<PresburgerRelation, GeneratingFunction>>
mlir::presburger::detail::computeChamberDecomposition(
    unsigned numSymbols, ArrayRef<std::pair<PresburgerSet, GeneratingFunction>>
                             regionsAndGeneratingFunctions) {
  assert(!regionsAndGeneratingFunctions.empty() &&
         "there must be at least one chamber!");
  // We maintain a list of regions and their associated generating function
  // initialized with the universe and the empty generating function.
  std::vector<std::pair<PresburgerRelation, GeneratingFunction>> chambers = {
      {PresburgerSet::getUniverse(PresburgerSpace::getSetSpace(numSymbols + 1)),
       GeneratingFunction(numSymbols, {}, {}, {})}};

  // We iterate over the region list.
  //
  // For each activity region R_j (corresponding to the generating function
  // gf_j), we examine all the current chambers R_i.
  //
  // If R_j has a full-dimensional intersection with an existing chamber R_i,
  // then that chamber is replaced by two new ones:
  // 1. the intersection R_i \cap R_j, where the generating function is
  // gf_i + gf_j.
  // 2. the difference R_i - R_j, where the generating function is gf_i.
  //
  // At each step, we define a new chamber list after considering gf_j,
  // replacing and appending chambers as discussed above.
  //
  // The loop has the invariant that the union over all the chambers gives the
  // universe at every step.
  for (const auto &[region, gf] : regionsAndGeneratingFunctions) {
    std::vector<std::pair<PresburgerRelation, GeneratingFunction>> newChambers;
    // llvm::errs() << "looking at region: "; region.dump();

    for (const auto &[currentRegion, currentGf] : chambers) {
      PresburgerRelation intersection = currentRegion.intersect(region).simplify();

      // If the intersection is not full-dimensional, we do not modify
      // the chamber list.
      if (!intersection.isFullDim()) {
        newChambers.emplace_back(currentRegion, currentGf);
        continue;
      }

      // If it is, we add the intersection and the difference as chambers.
      PresburgerRelation diff = currentRegion.subtract(region).simplify();
      newChambers.emplace_back(intersection, currentGf + gf);
      newChambers.emplace_back(diff, currentGf);
    }
    chambers = std::move(newChambers);
  }

  // We must convert the rational representation back into integers.
  // Remember that the representation is Ax' + bd + k >= 0 for a denominator `d`;
  // we only need to divide the denominator `d` and convert them back to Ax + b + k / d >= 0.
  // Since d is unknown and can be arbitrarily large, `k / d` reduces to -1 if `k < 0`,
  // and 0 otherwise.
  std::vector<std::pair<PresburgerRelation, GeneratingFunction>> result;
  for (const auto &[region, gf] : chambers) {
    if (region.isIntegerEmpty())
      continue;
    
    auto converted = PresburgerRelation::getEmpty(PresburgerSpace::getSetSpace(numSymbols));
    for (auto &disjunct : region.getAllDisjuncts()) {
      assert(disjunct.getNumEqualities() == 0);

      IntegerRelation result(PresburgerSpace::getRelationSpace(0, numSymbols, 0, disjunct.getNumLocalVars()));
      auto dDim = result.getNumDomainVars() + result.getNumRangeVars();
      SmallVector<DynamicAPInt> add(result.getNumVars() + 1);
      for (unsigned i = 0, e = disjunct.getNumInequalities(); i < e; i++) {
        auto ineq = disjunct.getInequality(i);
        std::copy(ineq.begin(), ineq.end() - 2, add.begin());
        add.back() = ineq[dDim] + (ineq.back() >= 0 ? 0 : -1);
        result.addInequality(add);
      }

      converted.unionInPlace(result);
    }
    result.emplace_back(converted, gf);
  }
  return result;
}

/// For a polytope expressed as a set of n inequalities, compute the generating
/// function corresponding to the lattice points included in the polytope. This
/// algorithm has three main steps:
/// 1. Enumerate the vertices, by iterating over subsets of inequalities and
///    checking for satisfiability. For each d-subset of inequalities (where d
///    is the number of variables), we solve to obtain the vertex in terms of
///    the parameters, and then check for the region in parameter space where
///    this vertex satisfies the remaining (n - d) inequalities.
/// 2. For each vertex, identify the tangent cone and compute the generating
///    function corresponding to it. The generating function depends on the
///    parametric expression of the vertex and the (non-parametric) generators
///    of the tangent cone.
/// 3. [Clauss-Loechner decomposition] Identify the regions in parameter space
///    (chambers) where each vertex is active, and accordingly compute the
///    GF of the polytope in each chamber.
///
/// Verdoolaege, Sven, et al. "Counting integer points in parametric
/// polytopes using Barvinok's rational functions." Algorithmica 48 (2007):
/// 37-66.
std::vector<std::pair<PresburgerRelation, GeneratingFunction>>
mlir::presburger::detail::computePolytopeGeneratingFunction(
    const PolyhedronH &poly) {
  unsigned numVars = poly.getNumRangeVars();
  unsigned numSymbols = poly.getNumSymbolVars();
  unsigned numIneqs = poly.getNumInequalities();

  // We store a list of the computed vertices.
  std::vector<ParamPoint> vertices;
  // For each vertex, we store the corresponding active region and the
  // generating functions of the tangent cone, in order.
  std::vector<std::pair<PresburgerSet, GeneratingFunction>>
      regionsAndGeneratingFunctions;

  // We iterate over all subsets of inequalities with cardinality numVars,
  // using permutations of numVars 1's and (numIneqs - numVars) 0's.
  //
  // For a given permutation, we consider a subset which contains
  // the i'th inequality if the i'th bit in the bitset is 1.
  //
  // We start with the permutation that takes the last numVars inequalities.
  SmallVector<int> indicator(numIneqs);
  for (unsigned i = numIneqs - numVars; i < numIneqs; ++i)
    indicator[i] = 1;

  do {
    // Collect the inequalities corresponding to the bits which are set
    // and the remaining ones.
    auto [subset, remainder] = poly.getInequalities().splitByBitset(indicator);
    // All other inequalities are stored in a2 and b2c2.
    //
    // These are column-wise splits of the inequalities;
    // a2 stores the coefficients of the variables, and
    // b2c2 stores the coefficients of the parameters and the constant term.
    FracMatrix a2(numIneqs - numVars, numVars);
    FracMatrix b2c2(numIneqs - numVars, numSymbols + 1);
    a2 = FracMatrix(
        remainder.getSubMatrix(0, numIneqs - numVars, 0, numVars));
    b2c2 = FracMatrix(remainder.getSubMatrix(0, numIneqs - numVars, numVars,
                                             numVars + numSymbols + 1));
    // Find the vertex, if any, corresponding to the current subset of
    // inequalities.
    std::optional<ParamPoint> maybeVertex =
        solveParametricEquations(FracMatrix(subset)); // d x (p+1)

    if (!maybeVertex)
      continue;
    auto vertex = *maybeVertex;
    if (llvm::is_contained(vertices, vertex))
      continue;
    // If this subset corresponds to a vertex that has not been considered,
    // store it.
    vertices.emplace_back(vertex);
    // If a vertex is formed by the intersection of more than d facets, we
    // assume that any d-subset of these facets can be solved to obtain its
    // expression. This assumption is valid because, if the vertex has two
    // distinct parametric expressions, then a nontrivial equality among the
    // parameters holds, which is a contradiction as we know the parameter
    // space to be full-dimensional.

    // Let the current vertex be [X | y], where
    // X represents the coefficients of the parameters and
    // y represents the constant term.
    //
    // The region (in parameter space) where this vertex is active is given
    // by substituting the vertex into the *remaining* inequalities of the
    // polytope (those which were not collected into `subset`), i.e., into the
    // inequalities [A2 | B2 | c2].
    //
    // Thus, the coefficients of the parameters after substitution become
    // (A2 • X + B2)
    // and the constant terms become
    // (A2 • y + c2).
    //
    // The region is therefore given by
    // (A2 • X + B2) p + (A2 • y + c2) ≥ 0
    //
    // This is equivalent to A2 • [X | y] + [B2 | c2].
    //
    // Thus we premultiply [X | y] with each row of A2
    // and add each row of [B2 | c2].
    FracMatrix activeRegion(numIneqs - numVars, numSymbols + 1);
    for (unsigned i = 0; i < numIneqs - numVars; i++) {
      activeRegion.setRow(i, vertex.preMultiplyWithRow(a2.getRow(i)));
      activeRegion.addToRow(i, b2c2.getRow(i), 1);
    }

    // We convert the representation of the active region to an integers-only
    // form so as to store it as a PresburgerSet.
    auto activeRegionRel = IntegerRelation::getRational(
        PresburgerSpace::getRelationSpace(0, numSymbols, 0, 0), activeRegion);
    if (activeRegionRel.isIntegerEmpty())
      continue;

    // Now, we compute the generating function at this vertex.
    // We collect the inequalities corresponding to each vertex to compute
    // the tangent cone at that vertex.
    //
    // We only need the coefficients of the variables (NOT the parameters)
    // as the generating function only depends on these.
    // We translate the cones to be pointed at the origin by making the
    // constant terms zero.

    ConeH tangentCone = defineHRep(numVars);
    for (unsigned j = 0, e = subset.getNumRows(); j < e; ++j) {
      SmallVector<DynamicAPInt> ineq(numVars + 1);
      for (unsigned k = 0; k < numVars; ++k)
        ineq[k] = subset(j, k);
      tangentCone.addInequality(ineq);
    }
    // There might be more active inequalities other than the ones chosen.
    // For an inequality Ax + Bp + c >= 0 to be active, the equality must hold;
    // This means the corresponding row of the active region must be zero,
    // since that is produced by substituting `x` back into the inequality.
    for (unsigned j = 0, e = activeRegion.getNumRows(); j < e; ++j) {
      if (llvm::any_of(activeRegion.getRow(j), [](const Fraction &x) { return x != 0; }))
        continue;
      SmallVector<DynamicAPInt> ineq(numVars + 1);
      for (unsigned k = 0; k < numVars; ++k)
        ineq[k] = remainder(j, k);
      tangentCone.addInequality(ineq);
    }

    GeneratingFunction vertexGf(numSymbols, {}, {}, {});
    std::vector<std::pair<int, ConeH>> unimodCones = unimodularDecompose(tangentCone);
    for (const std::pair<int, ConeH> &signedCone : unimodCones) {
      auto [sign, cone] = signedCone;
      auto gf = computeUnimodularConeGeneratingFunction(vertex, sign, cone);
      vertexGf = vertexGf + gf;
    }
    // We store the vertex we computed with the generating function of its
    // tangent cone.
    regionsAndGeneratingFunctions.emplace_back(PresburgerRelation(activeRegionRel),
                                               vertexGf);
  } while (std::next_permutation(indicator.begin(), indicator.end()));

  // Now, we use Clauss-Loechner decomposition to identify regions in parameter
  // space where each vertex is active. These regions (chambers) have the
  // property that no two of them have a full-dimensional intersection, i.e.,
  // they may share "facets" or "edges", but their intersection can only have
  // up to numVars - 1 dimensions.
  //
  // In each chamber, we sum up the generating functions of the active vertices
  // to find the generating function of the polytope.
  return computeChamberDecomposition(numSymbols, regionsAndGeneratingFunctions);
}

/// We use an iterative procedure to find a vector not orthogonal
/// to a given set, ignoring the null vectors.
/// Let the inputs be {x_1, ..., x_k}, all vectors of length n.
///
/// In the following,
/// vs[:i] means the elements of vs up to and including the i'th one,
/// <vs, us> means the dot product of vs and us,
/// vs ++ [v] means the vector vs with the new element v appended to it.
///
/// We proceed iteratively; for steps d = 0, ... n-1, we construct a vector
/// which is not orthogonal to any of {x_1[:d], ..., x_n[:d]}, ignoring
/// the null vectors.
/// At step d = 0, we let vs = [1]. Clearly this is not orthogonal to
/// any vector in the set {x_1[0], ..., x_n[0]}, except the null ones,
/// which we ignore.
/// At step d > 0 , we need a number v
/// s.t. <x_i[:d], vs++[v]> != 0 for all i.
/// => <x_i[:d-1], vs> + x_i[d]*v != 0
/// => v != - <x_i[:d-1], vs> / x_i[d]
/// We compute this value for all x_i, and then
/// set v to be the maximum element of this set plus one. Thus
/// v is outside the set as desired, and we append it to vs
/// to obtain the result of the d'th step.
Point mlir::presburger::detail::getNonOrthogonalVector(
    ArrayRef<Point> vectors) {
  unsigned dim = vectors[0].size();
  assert(llvm::all_of(
             vectors,
             [&dim](const Point &vector) { return vector.size() == dim; }) &&
         "all vectors need to be the same size!");

  SmallVector<Fraction> newPoint = {Fraction(1, 1)};
  Fraction maxDisallowedValue = -Fraction(1, 0),
           disallowedValue = Fraction(0, 1);

  for (unsigned d = 1; d < dim; ++d) {
    // Compute the disallowed values  - <x_i[:d-1], vs> / x_i[d] for each i.
    maxDisallowedValue = -Fraction(1, 0);
    for (const Point &vector : vectors) {
      if (vector[d] == 0)
        continue;
      disallowedValue =
          -dotProduct(ArrayRef(vector).slice(0, d), newPoint) / vector[d];

      // Find the biggest such value
      maxDisallowedValue = std::max(maxDisallowedValue, disallowedValue);
    }
    newPoint.emplace_back(maxDisallowedValue + 1);
  }
  return newPoint;
}

/// We use the following recursive formula to find the coefficient of
/// s^power in the rational function given by P(s)/Q(s).
///
/// Let P[i] denote the coefficient of s^i in the polynomial P(s).
/// (P/Q)[r] =
/// if (r == 0) then
///   P[0]/Q[0]
/// else
///   (P[r] - {Σ_{i=1}^r (P/Q)[r-i] * Q[i])}/(Q[0])
/// We therefore recursively call `getCoefficientInRationalFunction` on
/// all i \in [0, power).
///
/// https://math.ucdavis.edu/~deloera/researchsummary/
/// barvinokalgorithm-latte1.pdf, p. 1285
QuasiPolynomial mlir::presburger::detail::getCoefficientInRationalFunction(
    unsigned power, ArrayRef<QuasiPolynomial> num, ArrayRef<Fraction> den) {
  assert(!den.empty() && "division by empty denominator in rational function!");

  unsigned numParam = num[0].getNumInputs();
  // We use the `isEqual` method of PresburgerSpace, which QuasiPolynomial
  // inherits from.
  assert(llvm::all_of(num,
                      [&num](const QuasiPolynomial &qp) {
                        return num[0].isEqual(qp);
                      }) &&
         "the quasipolynomials should all belong to the same space!");

  std::vector<QuasiPolynomial> coefficients;
  coefficients.reserve(power + 1);

  coefficients.emplace_back(num[0] / den[0]);
  for (unsigned i = 1; i <= power; ++i) {
    // If the power is not there in the numerator, the coefficient is zero.
    coefficients.emplace_back(i < num.size() ? num[i]
                                             : QuasiPolynomial(numParam, 0));

    // After den.size(), the coefficients are zero, so we stop
    // subtracting at that point (if it is less than i).
    unsigned limit = std::min<unsigned long>(i, den.size() - 1);
    for (unsigned j = 1; j <= limit; ++j)
      coefficients[i] = coefficients[i] -
                        coefficients[i - j] * QuasiPolynomial(numParam, den[j]);

    coefficients[i] = coefficients[i] / den[0];
  }
  return coefficients[power].simplify();
}

/// Substitute x_i = t^μ_i in one term of a generating function, returning
/// a quasipolynomial which represents the exponent of the numerator
/// of the result, and a vector which represents the exponents of the
/// denominator of the result.
/// If the returned value is {num, dens}, it represents the function
/// t^num / \prod_j (1 - t^dens[j]).
/// v represents the affine functions whose floors are multiplied by the
/// generators, and ds represents the list of generators.
std::pair<QuasiPolynomial, std::vector<Fraction>>
substituteMuInTerm(unsigned numParams, const ParamPoint &v,
                   const std::vector<Point> &ds, const Point &mu) {
  unsigned numDims = mu.size();
#ifndef NDEBUG
  for (const Point &d : ds)
    assert(d.size() == numDims &&
           "μ has to have the same number of dimensions as the generators!");
#endif

  // First, the exponent in the numerator becomes
  // - (μ • u_1) * (floor(first col of v))
  // - (μ • u_2) * (floor(second col of v)) - ...
  // - (μ • u_d) * (floor(d'th col of v))
  // So we store the negation of the dot products.

  // We have d terms, each of whose coefficient is the negative dot product.
  SmallVector<Fraction> coefficients;
  coefficients.reserve(numDims);
  for (const Point &d : ds)
    coefficients.emplace_back(-dotProduct(mu, d));

  // Then, the affine function is a single floor expression, given by the
  // corresponding column of v.
  ParamPoint vTranspose = v.transpose();
  std::vector<std::vector<SmallVector<Fraction>>> affine;
  affine.reserve(numDims);
  for (unsigned j = 0; j < numDims; ++j)
    affine.push_back({SmallVector<Fraction>{vTranspose.getRow(j)}});

  QuasiPolynomial num(numParams, coefficients, affine);
  num = num.simplify();

  std::vector<Fraction> dens;
  dens.reserve(ds.size());
  // Similarly, each term in the denominator has exponent
  // given by the dot product of μ with u_i.
  for (const Point &d : ds) {
    // This term in the denominator is
    // (1 - t^dens.back())
    dens.emplace_back(dotProduct(d, mu));
  }

  return {num, dens};
}

/// Normalize all denominator exponents `dens` to their absolute values
/// by multiplying and dividing by the inverses, in a function of the form
/// sign * t^num / prod_j (1 - t^dens[j]).
/// Here, sign = ± 1,
/// num is a QuasiPolynomial, and
/// each dens[j] is a Fraction.
void normalizeDenominatorExponents(int &sign, QuasiPolynomial &num,
                                   std::vector<Fraction> &dens) {
  // We track the number of exponents that are negative in the
  // denominator, and convert them to their absolute values.
  unsigned numNegExps = 0;
  Fraction sumNegExps(0, 1);
  for (const auto &den : dens) {
    if (den < 0) {
      numNegExps += 1;
      sumNegExps += den;
    }
  }

  // If we have (1 - t^-c) in the denominator, for positive c,
  // multiply and divide by t^c.
  // We convert all negative-exponent terms at once; therefore
  // we multiply and divide by t^sumNegExps.
  // Then we get
  // -(1 - t^c) in the denominator,
  // increase the numerator by c, and
  // flip the sign of the function.
  if (numNegExps % 2 == 1)
    sign = -sign;
  num = num - QuasiPolynomial(num.getNumInputs(), sumNegExps);
}

/// Compute the binomial coefficients nCi for 0 ≤ i ≤ r,
/// where n is a QuasiPolynomial.
std::vector<QuasiPolynomial> getBinomialCoefficients(const QuasiPolynomial &n,
                                                     unsigned r) {
  unsigned numParams = n.getNumInputs();
  std::vector<QuasiPolynomial> coefficients;
  coefficients.reserve(r + 1);
  coefficients.emplace_back(numParams, 1);
  for (unsigned j = 1; j <= r; ++j)
    // We use the recursive formula for binomial coefficients here and below.
    coefficients.emplace_back(
        (coefficients[j - 1] * (n - QuasiPolynomial(numParams, j - 1)) /
         Fraction(j, 1))
            .collectTerms().simplify());
  return coefficients;
}

/// Compute the binomial coefficients nCi for 0 ≤ i ≤ r,
/// where n is a QuasiPolynomial.
std::vector<Fraction> getBinomialCoefficients(const Fraction &n,
                                              const Fraction &r) {
  std::vector<Fraction> coefficients;
  coefficients.reserve((int64_t)floor(r));
  coefficients.emplace_back(1);
  for (unsigned j = 1; j <= r; ++j)
    coefficients.emplace_back(coefficients[j - 1] * (n - (j - 1)) / (j));
  return coefficients;
}

/// We have a generating function of the form
/// f_p(x) = \sum_i sign_i * (x^n_i(p)) / (\prod_j (1 - x^d_{ij})
///
/// where sign_i is ±1,
/// n_i \in Q^p -> Q^d is the sum of the vectors d_{ij}, weighted by the
/// floors of d affine functions on p parameters.
/// d_{ij} \in Q^d are vectors.
///
/// We need to find the number of terms of the form x^t in the expansion of
/// this function.
/// However, direct substitution (x = (1, ..., 1)) causes the denominator
/// to become zero.
///
/// We therefore use the following procedure instead:
/// 1. Substitute x_i = (s+1)^μ_i for some vector μ. This makes the generating
/// function a function of a scalar s.
/// 2. Write each term in this function as P(s)/Q(s), where P and Q are
/// polynomials. P has coefficients as quasipolynomials in d parameters, while
/// Q has coefficients as scalars.
/// 3. Find the constant term in the expansion of each term P(s)/Q(s). This is
/// equivalent to substituting s = 0.
///
/// Verdoolaege, Sven, et al. "Counting integer points in parametric
/// polytopes using Barvinok's rational functions." Algorithmica 48 (2007):
/// 37-66.
QuasiPolynomial
mlir::presburger::detail::computeNumTerms(const GeneratingFunction &gf) {
  // Step (1) We need to find a μ such that we can substitute x_i =
  // (s+1)^μ_i. After this substitution, the exponent of (s+1) in the
  // denominator is (μ_i • d_{ij}) in each term. Clearly, this cannot become
  // zero. Hence we find a vector μ that is not orthogonal to any of the
  // d_{ij} and substitute x accordingly.
  std::vector<Point> allDenominators;
  for (ArrayRef<Point> den : gf.getDenominators())
    llvm::append_range(allDenominators, den);
  Point mu = getNonOrthogonalVector(allDenominators);

  unsigned numParams = gf.getNumParams();
  const std::vector<std::vector<Point>> &ds = gf.getDenominators();
  QuasiPolynomial totalTerm(numParams, 0);
  for (unsigned i = 0, e = ds.size(); i < e; ++i) {
    int sign = gf.getSigns()[i];

    // Compute the new exponents of (s+1) for the numerator and the
    // denominator after substituting μ.
    auto [numExp, dens] =
        substituteMuInTerm(numParams, gf.getNumerators()[i], ds[i], mu);
    // Now the numerator is (s+1)^numExp
    // and the denominator is \prod_j (1 - (s+1)^dens[j]).

    // Step (2) We need to express the terms in the function as quotients of
    // polynomials. Each term is now of the form
    // sign_i * (s+1)^numExp / (\prod_j (1 - (s+1)^dens[j]))
    // For the i'th term, we first normalize the denominator to have only
    // positive exponents. We convert all the dens[j] to their
    // absolute values and change the sign and exponent in the numerator.
    normalizeDenominatorExponents(sign, numExp, dens);

    // Then, using the formula for geometric series, we replace each (1 -
    // (s+1)^(dens[j])) with
    // (-s)(\sum_{0 ≤ k < dens[j]} (s+1)^k).
    for (auto &j : dens)
      j = abs(j) - 1;

    // Note that at this point, the semantics of `dens[j]` changes to mean
    // a term (\sum_{0 ≤ k ≤ dens[j]} (s+1)^k). The denominator is, as before,
    // a product of these terms.

    // Since the -s are taken out, the sign changes if there is an odd number
    // of such terms.
    unsigned r = dens.size();
    if (dens.size() % 2 == 1)
      sign = -sign;

    // Thus the term overall now has the form
    // sign'_i * (s+1)^numExp /
    // (s^r * \prod_j (\sum_{0 ≤ k < dens[j]} (s+1)^k)).
    // This means that
    // the numerator is a polynomial in s, with coefficients as
    // quasipolynomials (given by binomial coefficients), and the denominator
    // is a polynomial in s, with integral coefficients (given by taking the
    // convolution over all j).

    // Step (3) We need to find the constant term in the expansion of each
    // term. Since each term has s^r as a factor in the denominator, we avoid
    // substituting s = 0 directly; instead, we find the coefficient of s^r in
    // sign'_i * (s+1)^numExp / (\prod_j (\sum_k (s+1)^k)),
    // Letting P(s) = (s+1)^numExp and Q(s) = \prod_j (...),
    // we need to find the coefficient of s^r in P(s)/Q(s),
    // for which we use the `getCoefficientInRationalFunction()` function.

    // First, we compute the coefficients of P(s), which are binomial
    // coefficients.
    // We only need the first r+1 of these, as higher-order terms do not
    // contribute to the coefficient of s^r.
    std::vector<QuasiPolynomial> numeratorCoefficients =
        getBinomialCoefficients(numExp, r);

    // Then we compute the coefficients of each individual term in Q(s),
    // which are (dens[i]+1) C (k+1) for 0 ≤ k ≤ dens[i].
    std::vector<std::vector<Fraction>> eachTermDenCoefficients;
    std::vector<Fraction> singleTermDenCoefficients;
    eachTermDenCoefficients.reserve(r);
    for (const Fraction &den : dens) {
      singleTermDenCoefficients = getBinomialCoefficients(den + 1, den + 1);
      eachTermDenCoefficients.emplace_back(
          ArrayRef<Fraction>(singleTermDenCoefficients).drop_front());
    }

    // Now we find the coefficients in Q(s) itself
    // by taking the convolution of the coefficients
    // of all the terms.
    std::vector<Fraction> denominatorCoefficients;
    denominatorCoefficients = eachTermDenCoefficients[0];
    for (unsigned j = 1, e = eachTermDenCoefficients.size(); j < e; ++j)
      denominatorCoefficients = multiplyPolynomials(denominatorCoefficients,
                                                    eachTermDenCoefficients[j]);                                         
    totalTerm =
        totalTerm + getCoefficientInRationalFunction(r, numeratorCoefficients,
                                                     denominatorCoefficients) *
                        QuasiPolynomial(numParams, sign);
  }

  return totalTerm.simplify();
}

namespace {

struct Region {
  // The region is formed by intersection of disjunctions[indices].
  SmallVector<unsigned> indices;

  // The actual region in parameter space.
  PresburgerRelation region;

  // The number of integers in the region.
  QuasiPolynomial count;

  void print(llvm::raw_ostream &os) const;
  void dump() const;
};

void Region::dump() const {
  print(llvm::errs());
}

void Region::print(llvm::raw_ostream &os) const {
  os << "==== Record ====\n";
  os << "Indices:\n";
  llvm::interleaveComma(indices, os);
  os << "\nRegion:\n";
  region.print(os);
  os << "\nQuasiPolynomial:\n";
  count.print(os);
  os << "\n";
}

/// We take disjunctions of `rel` whose indices are in `active`, intersect them, and put the result into `outRecords`.
void obtainRegions(const PresburgerRelation &rel, const PolyhedronH &current, SmallVector<unsigned> &active, unsigned next, std::vector<Region> &outRecords) {
  if (next == rel.getNumDisjuncts()) {
    if (active.empty() || current.isIntegerEmpty())
      return;
    
    // Compute generating functions on active regions.
    // By "active region", we must also take the parameter space
    // constraint into consideration.
    auto [c, projected] = projectToFullDimension(current);
    assert(projected.isFullDim());
    PresburgerRelation constraint(c);

    // A singleton set.
    if (projected.getNumVars() == 0) {
      unsigned syms = rel.getNumSymbolVars();
      auto one = QuasiPolynomial(syms, 1);
      outRecords.push_back({ active, constraint, one });
      return;
    }
    // We should guarantee that the number of integer points 
    // #projected = #current. Since current is non-empty, neither should
    // projected be.
    assert(!projected.isIntegerEmpty());
    auto gfs = computePolytopeGeneratingFunction(projected);
    for (const auto &[region_, gf] : gfs) {
      auto region = region_.intersect(constraint);
      // It is possible for a non-empty region to have an empty gf.
      // This simply means there's no integer points inside it.
      if (region.isIntegerEmpty() || gf.getSigns().size() == 0)
        continue;
      
      QuasiPolynomial q = computeNumTerms(gf);
      q = q.collectTerms().simplify();
      outRecords.push_back({ active, region.simplify(), q });
    }
    return;
  }

  // Case 1. disjunct[next] is excluded.
  obtainRegions(rel, current, active, next + 1, outRecords);

  // Case 2. disjunct[next] is included.
  PolyhedronH nextIntersection = current.intersect(rel.getDisjunct(next));
  if (!nextIntersection.isIntegerEmpty()) {
    active.push_back(next);
    obtainRegions(rel, nextIntersection, active, next + 1, outRecords);
    active.pop_back();
  }
}

} // namespace

std::vector<std::pair<PresburgerRelation, QuasiPolynomial>>
mlir::presburger::detail::countIntegerPoints(const PresburgerRelation &rel) {
  std::vector<Region> records;
  SmallVector<unsigned> active;
  auto universe = IntegerRelation::getUniverse(rel.getSpace());
  obtainRegions(rel, universe, active, 0, records);
  // for (auto &rec : records)
  //   rec.dump();

  unsigned numParams = rel.getNumSymbolVars();
  auto paramSpace = PresburgerSpace::getSetSpace(numParams);
  auto paramUniverse = PresburgerRelation::getUniverse(paramSpace);
  std::vector<PresburgerRelation> refinement { paramUniverse };

  // For each record region, split current refinement.
  for (const Region &rec : records) {
    std::vector<PresburgerRelation> nextRef;
    for (PresburgerRelation &set : refinement) {
      // Divide each set into two parts: 
      // A = S ∩ rec.region
      PresburgerRelation a = set.intersect(rec.region);
      if (!a.isIntegerEmpty())
        nextRef.push_back(a);
      
      // B = S \ rec.region
      PresburgerRelation b = set.subtract(rec.region);
      if (!b.isIntegerEmpty())
        nextRef.push_back(b);
    }
    refinement.swap(nextRef);
  }

  // Now refinement is a list of pairwise-disjoint chambers.
  // For each refinement chamber, collect contributions and sum quasi-polynomials.
  std::vector<std::pair<PresburgerRelation, QuasiPolynomial>> result;
  for (const PresburgerRelation &chamber : refinement) {
    // Build the inclusion-exclusion sum.
     
    // A zero polynomial.
    QuasiPolynomial q(numParams, 0);
    for (const Region &rec : records) {
      // The chamber is either fully inside `rec.region`, or fully disjoint with it.
      PresburgerRelation intersect = chamber.intersect(rec.region);
      if (intersect.isIntegerEmpty())
        continue;
      // rec.count contributes to the chamber.
      // According to inclusion-exclusion principle, the coefficient is
      // (-1)^{|I|+1}, where I is the number of insections.
      int parity = (rec.indices.size() % 2 == 1) ? 1 : -1;
      q = q + rec.count * parity;
    }
    result.emplace_back(chamber, q.collectTerms());
  }

  return result;
}
