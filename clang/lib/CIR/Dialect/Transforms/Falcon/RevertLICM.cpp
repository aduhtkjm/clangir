#include "PassDetail.h"
#include "FalconUtilities.h"
#include "clang/CIR/Dialect/IR/CIRDialect.h"
#include "clang/CIR/Dialect/Passes.h"

using namespace mlir;
using namespace cir;
using namespace affine;

namespace {

class RevertLICM : public RevertLICMBase<RevertLICM> {
  using RevertLICMBase::RevertLICMBase;

  void runOnOperation() override;
private:
};

void RevertLICM::runOnOperation() {
  Operation *module = getOperation();

  auto loads = findAll<LoadOp>(module);
  for (auto load : loads) {
    auto *addr = load.getAddr().getDefiningOp();
    if (!isa<GetElementOp>(addr) && !isa<PtrStrideOp>(addr))
      continue;
  
    OpBuilder builder(load);
    Operation *cloned = builder.clone(*addr);
    load.getAddrMutable() = cloned;
  }

  auto stores = findAll<StoreOp>(module);
  for (auto store : stores) {
    auto *addr = store.getAddr().getDefiningOp();
    if (!isa<GetElementOp>(addr) && !isa<PtrStrideOp>(addr))
      continue;
  
    OpBuilder builder(store);
    Operation *cloned = builder.clone(*addr);
    store.getAddrMutable() = cloned;
  }
}

} // namespace

std::unique_ptr<Pass> mlir::createCIRRevertLICMPass() {
  return std::make_unique<RevertLICM>();
}
