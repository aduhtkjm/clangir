add_clang_library(Falcon
  RaiseToAffine.cpp

  LINK_LIBS PUBLIC
  MLIRAnalysis
  MLIRIR
  MLIRPass
  MLIRTransformUtils

  MLIRCIR
  MLIRCIRInterfaces
)