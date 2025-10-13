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
  auto attr = op->getAttr(T::getMnemonic());
  if (!attr)
    return T();

  if (auto result = dyn_cast<T>(attr))
    return result;
  
  return T();
}
  
} // namespace mlir
