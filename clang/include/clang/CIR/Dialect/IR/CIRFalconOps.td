#ifndef LLVM_CLANG_CIR_DIALECT_IR_CIRFALCONOPS
#define LLVM_CLANG_CIR_DIALECT_IR_CIRFALCONOPS

def CIR_IndexCastOp : CIR_Op<"index_cast", [Pure]> {
  let summary = "Cast a CIR integer to the builtin index type.";
  let description = [{
  }];
  let arguments = (ins CIR_IntType:$src);
  let results = (outs Index:$result);

  let assemblyFormat = [{
    $src `:` type($src) attr-dict
  }];
}

def CIR_FromIndexOp : CIR_Op<"from_index", [Pure]> {
  let summary = "Cast the builtin index type to a CIR integer type.";
  let description = [{
  }];
  let arguments = (ins Index:$src);
  let results = (outs CIR_IntType:$result);

  let assemblyFormat = [{
    $src `->` type($result) attr-dict
  }];
}

#endif // LLVM_CLANG_CIR_DIALECT_IR_CIRFALCONOPS
