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

template<class T, class ...Args>
void setAttr(mlir::Operation *op, Args ...args) {
  op->setAttr(T::getMnemonic(), T::get(args...));
}

template<class T>
T getAttr(mlir::Operation *op) {
  return cast<T>(op->getAttr(T::getMnemonic()));
}
  
} // namespace mlir
