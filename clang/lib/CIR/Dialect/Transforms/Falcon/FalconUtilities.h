#include "clang/CIR/Dialect/IR/CIRDialect.h"

namespace mlir {

template<class T>
llvm::SmallVector<T> findAll(mlir::Operation *op) {
  llvm::SmallVector<T> subops;
  op->walk([&](T subop) {
    subops.push_back(subop);
  });
  return subops;
}
  
} // namespace mlir
