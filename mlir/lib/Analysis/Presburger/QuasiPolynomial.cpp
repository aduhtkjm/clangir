//===- QuasiPolynomial.cpp - Quasipolynomial Class --------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Analysis/Presburger/QuasiPolynomial.h"
#include "mlir/Analysis/Presburger/Fraction.h"
#include "mlir/Analysis/Presburger/PresburgerSpace.h"

using namespace mlir;
using namespace presburger;

QuasiPolynomial::QuasiPolynomial(
    unsigned numVars, ArrayRef<Fraction> coeffs,
    ArrayRef<std::vector<SmallVector<Fraction>>> aff)
    : PresburgerSpace(/*numDomain=*/numVars, /*numRange=*/1, /*numSymbols=*/0,
                      /*numLocals=*/0),
      coefficients(coeffs), affine(aff) {
#ifndef NDEBUG
  // For each term which involves at least one affine function,
  for (const std::vector<SmallVector<Fraction>> &term : affine) {
    if (term.empty())
      continue;
    // the number of elements in each affine function is
    // one more than the number of symbols.
    for (const SmallVector<Fraction> &aff : term) {
      assert(aff.size() == getNumInputs() + 1 &&
             "dimensionality of affine functions does not match number of "
             "symbols!");
    }
  }
#endif // NDEBUG
}

/// Define a quasipolynomial which is a single constant.
QuasiPolynomial::QuasiPolynomial(unsigned numVars, const Fraction &constant)
    : PresburgerSpace(/*numDomain=*/numVars, /*numRange=*/1, /*numSymbols=*/0,
                      /*numLocals=*/0),
      coefficients({constant}), affine({{}}) {}

QuasiPolynomial QuasiPolynomial::operator+(const QuasiPolynomial &x) const {
  assert(getNumInputs() == x.getNumInputs() &&
         "two quasi-polynomials with different numbers of symbols cannot "
         "be added!");
  SmallVector<Fraction> sumCoeffs = coefficients;
  sumCoeffs.append(x.coefficients);
  std::vector<std::vector<SmallVector<Fraction>>> sumAff = affine;
  llvm::append_range(sumAff, x.affine);
  return QuasiPolynomial(getNumInputs(), sumCoeffs, sumAff);
}

QuasiPolynomial QuasiPolynomial::operator-(const QuasiPolynomial &x) const {
  assert(getNumInputs() == x.getNumInputs() &&
         "two quasi-polynomials with different numbers of symbols cannot "
         "be subtracted!");
  QuasiPolynomial qp(getNumInputs(), x.coefficients, x.affine);
  for (Fraction &coeff : qp.coefficients)
    coeff = -coeff;
  return *this + qp;
}

QuasiPolynomial QuasiPolynomial::operator*(const QuasiPolynomial &x) const {
  assert(getNumInputs() == x.getNumInputs() &&
         "two quasi-polynomials with different numbers of "
         "symbols cannot be multiplied!");

  SmallVector<Fraction> coeffs;
  coeffs.reserve(coefficients.size() * x.coefficients.size());
  for (const Fraction &coeff : coefficients)
    for (const Fraction &xcoeff : x.coefficients)
      coeffs.emplace_back(coeff * xcoeff);

  std::vector<SmallVector<Fraction>> product;
  std::vector<std::vector<SmallVector<Fraction>>> aff;
  aff.reserve(affine.size() * x.affine.size());
  for (const std::vector<SmallVector<Fraction>> &term : affine) {
    for (const std::vector<SmallVector<Fraction>> &xterm : x.affine) {
      product.clear();
      llvm::append_range(product, term);
      llvm::append_range(product, xterm);
      aff.emplace_back(product);
    }
  }

  return QuasiPolynomial(getNumInputs(), coeffs, aff);
}

QuasiPolynomial QuasiPolynomial::operator/(const Fraction &x) const {
  assert(x != 0 && "division by zero!");
  QuasiPolynomial qp(*this);
  for (Fraction &coeff : qp.coefficients)
    coeff /= x;
  return qp;
}

QuasiPolynomial QuasiPolynomial::operator*(const Fraction &x) const {
  QuasiPolynomial qp(*this);
  for (Fraction &coeff : qp.coefficients)
    coeff *= x;
  return qp;
}

// Removes terms which evaluate to zero from the expression and
// integrate affine functions which are constants into the
// coefficients.
QuasiPolynomial QuasiPolynomial::simplify() const {
  Fraction newCoeff = 0;
  SmallVector<Fraction> newCoeffs({});

  std::vector<SmallVector<Fraction>> newAffineTerm({});
  std::vector<std::vector<SmallVector<Fraction>>> newAffine({});

  unsigned numParam = getNumInputs();

  for (unsigned i = 0, e = coefficients.size(); i < e; i++) {
    // A term is zero if its coefficient is zero, or
    if (coefficients[i] == Fraction(0, 1))
      continue;
    bool productIsZero =
        // if any of the affine functions in the product
        llvm::any_of(affine[i], [](const SmallVector<Fraction> &affineIj) {
          // has all its coefficients as zero.
          return llvm::all_of(affineIj,
                              [](const Fraction &f) { return f == 0; });
        });
    if (productIsZero)
      continue;

    // Now, we know the term is nonzero.

    // We now eliminate the affine functions which are constant
    // by merging them into the coefficients.
    newAffineTerm = {};
    newCoeff = coefficients[i];
    for (ArrayRef<Fraction> term : affine[i]) {
      bool allCoeffsZero = llvm::all_of(
          term.slice(0, numParam), [](const Fraction &c) { return c == 0; });
      // Remember that we have a floor outside the affine function.
      if (allCoeffsZero)
        newCoeff *= floor(term[numParam]);
      else
        newAffineTerm.emplace_back(term);
    }

    newCoeffs.emplace_back(newCoeff);
    newAffine.emplace_back(newAffineTerm);
  }
  return QuasiPolynomial(getNumInputs(), newCoeffs, newAffine);
}

QuasiPolynomial QuasiPolynomial::collectTerms() const {
  SmallVector<Fraction> newCoeffs({});
  std::vector<std::vector<SmallVector<Fraction>>> newAffine({});

  for (unsigned i = 0, e = affine.size(); i < e; i++) {
    bool alreadyPresent = false;
    for (unsigned j = 0, f = newAffine.size(); j < f; j++) {
      if (affine[i] == newAffine[j]) {
        newCoeffs[j] += coefficients[i];
        alreadyPresent = true;
      }
    }
    if (alreadyPresent)
      continue;
    newCoeffs.emplace_back(coefficients[i]);
    newAffine.emplace_back(affine[i]);
  }

  return QuasiPolynomial(getNumInputs(), newCoeffs, newAffine);
}

Fraction QuasiPolynomial::getConstantTerm() {
  Fraction constTerm = 0;
  for (unsigned i = 0, e = coefficients.size(); i < e; ++i)
    if (affine[i].empty())
      constTerm += coefficients[i];
  return constTerm;
}

Fraction QuasiPolynomial::evaluate(ArrayRef<DynamicAPInt> x) const {
  Fraction value = 0;
  for (unsigned i = 0; i < affine.size(); i++) {
    Fraction term = 1;
    for (ArrayRef<Fraction> xCoeffs : affine[i]) {
      assert(x.size() == xCoeffs.size() - 1);
      Fraction sum = std::inner_product(x.begin(), x.end(), xCoeffs.begin(), Fraction());
      term *= floor(sum + xCoeffs.back());
    }
    Fraction coeff = coefficients[i];
    value += coeff * term;
  }
  return value;
}

QuasiPolynomial QuasiPolynomial::partialEvaluate(const DynamicAPInt &x) const {
  std::vector<std::vector<SmallVector<Fraction>>> aff{affine.size()};
  for (unsigned i = 0; i < affine.size(); i++) {
    for (ArrayRef<Fraction> xCoeffs : affine[i]) {
      auto sliced = xCoeffs.slice(1);
      aff[i].emplace_back(sliced.begin(), sliced.end());
      aff[i].back().back() += x * xCoeffs[0];
    }
  }
  return QuasiPolynomial(getNumInputs() - 1, coefficients, aff);
}

QuasiPolynomial QuasiPolynomial::partialEvaluate(unsigned index, ArrayRef<DynamicAPInt> x) const {
  std::vector<std::vector<SmallVector<Fraction>>> aff{affine.size()};
  for (unsigned i = 0; i < affine.size(); i++) {
    for (ArrayRef<Fraction> xCoeffs : affine[i]) {
      SmallVector<Fraction> w(xCoeffs.begin(), xCoeffs.end());
      Fraction value = w[index];
      w.erase(w.begin() + index);
      for (unsigned j = 0; j < w.size(); j++)
        w[j] += x[j] * value;
      aff[i].push_back(w);
    }
  }
  return QuasiPolynomial(getNumInputs() - 1, coefficients, aff).simplify().collectTerms();
}

struct Interval {
  Fraction min;
  Fraction max;
};

static Interval evaluateAffine(const llvm::SmallVector<Fraction> &g,
  const llvm::ArrayRef<std::pair<DynamicAPInt, DynamicAPInt>> &varBounds) {
  Fraction min = g.back();
  Fraction max = g.back();

  for (size_t k = 0; k < varBounds.size(); k++) {
    Fraction term1 = g[k] * varBounds[k].first;
    Fraction term2 = g[k] * varBounds[k].second;
    
    if (term1 < term2) {
      min += term1;
      max += term2;
    } else {
      min += term2;
      max += term1;
    }
  }
  return {min, max};
}

DynamicAPInt QuasiPolynomial::computeCertifiedLowerBound(llvm::ArrayRef<std::pair<DynamicAPInt, DynamicAPInt>> varBounds) const {
  Fraction total = 0;
  for (size_t i = 0; i < coefficients.size(); ++i) {
    Interval productInterval { 1, 1 };

    for (const auto& aff : affine[i]) {
      Interval range = evaluateAffine(aff, varBounds);
      Interval floored = { floor(range.min), floor(range.max) };

      // Interval Multiplication: [a, b] * [c, d]
      Fraction v1 = productInterval.min * floored.min;
      Fraction v2 = productInterval.min * floored.max;
      Fraction v3 = productInterval.max * floored.min;
      Fraction v4 = productInterval.max * floored.max;

      productInterval.min = std::min({v1, v2, v3, v4});
      productInterval.max = std::max({v1, v2, v3, v4});
    }

    if (coefficients[i] >= 0)
      total += coefficients[i] * productInterval.min;
    else
      total += coefficients[i] * productInterval.max;
  }

  return floor(total);
}

DynamicAPInt QuasiPolynomial::computeCertifiedUpperBound(llvm::ArrayRef<std::pair<DynamicAPInt, DynamicAPInt>> varBounds) const {
  Fraction total = 0;
  for (size_t i = 0; i < coefficients.size(); ++i) {
    Interval productInterval { 1, 1 };

    for (const auto& aff : affine[i]) {
      Interval range = evaluateAffine(aff, varBounds);

      // Interval multiplication.
      Fraction v1 = productInterval.min * range.min;
      Fraction v2 = productInterval.min * range.max;
      Fraction v3 = productInterval.max * range.min;
      Fraction v4 = productInterval.max * range.max;

      productInterval.min = std::min({v1, v2, v3, v4});
      productInterval.max = std::max({v1, v2, v3, v4});
    }

    if (coefficients[i] >= 0)
      total += coefficients[i] * productInterval.max;
    else
      total += coefficients[i] * productInterval.min;
  }

  return ceil(total);
}

void QuasiPolynomial::print(llvm::raw_ostream &os) const {
  if (affine.empty()) {
    os << "<empty>";
    return;
  }
  
  for (unsigned i = 0, e = affine.size(); i < e; i++) {
    if (i != 0)
      os << " + ";

    os << coefficients[i];
    if (affine[i].size() > 0)
      os << " * [";
    for (unsigned j = 0, f = affine[i].size(); j < f; j++) {
      if (j != 0)
        os << "] * [";
      
      llvm::interleaveComma(affine[i][j], os);
    }
    if (affine[i].size() > 0)
      os << "]";
  }
}
