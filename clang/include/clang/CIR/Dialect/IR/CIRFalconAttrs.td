#ifndef CLANG_CIR_DIALECT_IR_CIRFALCONATTRS_TD
#define CLANG_CIR_DIALECT_IR_CIRFALCONATTRS_TD

def CIR_ArrayBaseAttr : CIR_Attr<"ArrayBase", "array_base"> {
  let summary = "The starting address of the array to which the pointer belongs.";
  let description = [{
    Marked on `cir.ptr_stride` operations, function arguments and allocas. 
    When these are striding from arrays or are denoting arrays, the attribute
    shows the starting address of the array that the pointer belongs to.

    As SSA values cannot be stored in attributes, we use integer IDs to distinguish
    different arrays.
  }];
  let parameters = (ins "unsigned":$id);
  let assemblyFormat = "`<` $id `>`";
}

def CIR_ArrayOffsetAttr : CIR_Attr<"ArrayOffset", "array_offset"> {
  let summary = "The offset from the base.";
  let description = [{
    Marked on `cir.ptr_stride` operations, function arguments and allocas.
    When these are striding from arrays or are denoting arrays, the attribute
    shows the offset from the starting address (the base) of the array.

    The information is stored as an affine map.
  }];
  let parameters = (ins "mlir::AffineMap":$map);
  let assemblyFormat = "`<` $map `>`";
}

#endif
