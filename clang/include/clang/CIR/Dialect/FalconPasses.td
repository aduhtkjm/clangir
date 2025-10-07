#ifndef MLIR_DIALECT_CIR_FALCONPASSES
#define MLIR_DIALECT_CIR_FALCONPASSES

include "mlir/Pass/PassBase.td"

def RaiseToAffine : Pass<"cir-falcon-raise-to-affine"> {
  let summary = "Raise CIR loops to affine dialects when possible";
  let description = [{
    We raise loops to affine::AffineOp for further analysis in this pass.
    The details is similar to the process of raising loops to scf::ForOp in lowering.
  }];
  let constructor = "mlir::createCIRRaiseToAffinePass()";
  let dependentDialects = ["cir::CIRDialect", "affine::AffineDialect"];
}

def FindOffset : Pass<"cir-falcon-find-offset"> {
  let summary = "Find offset of arrays";
  let description = [{
    Note this only deal with compile-time known arrays, rather than arbitrary pointers.
    Arrays are assumed not aliasing with each other.
  }];
  let constructor = "mlir::createCIRFindOffsetPass()";
  let dependentDialects = ["cir::CIRDialect", "affine::AffineDialect"];
}

#endif // MLIR_DIALECT_CIR_FALCONPASSES
