add_clang_library(Falcon
  RaiseToAffine.cpp
  FindOffset.cpp

  LINK_LIBS PUBLIC
  MLIRAnalysis
  MLIRIR
  MLIRPass
  MLIRTransformUtils

  MLIRCIR
  MLIRCIRInterfaces
)