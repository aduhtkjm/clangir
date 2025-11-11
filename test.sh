#!/bin/zsh

die() {
  echo -e "\e[31merror:\e[0m $1"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    -c|--cmake)
      run_cmake=1; shift;;
    -t|--test)
      if [[ -z $2 ]]; then
        die "expected test case name after $1"
        exit 1
      fi
      testcase=$2; shift 2;;
    -v|--verbose)
      verbose=1; shift;;
    -V|--valgrind)
      valgrind=1; shift;;
    -f|--fix)
      rm -f build/bin/clang build/bin/clang++
      ln -s /usr/local/bin/clang build/bin/clang
      ln -s /usr/local/bin/clang++ build/bin/clang++
      shift;;
    -d|--display)
      display=1; shift;;
    -w|--write-test)
      if [[ -z $2 ]]; then
        die "expected test case name after $1"
        exit 1
      fi
      newtest=$2; shift 2;;
    -s|--simulate)
      if [[ -z $2 ]]; then
        die "expected simulation name after $1"
        exit 1
      fi
      simul=$2; shift 2;;
    *)
      die "unknown option: $1"; failed=1; shift;;
  esac
done

if [[ -n $display && -z $testcase ]]; then
  die "--display is specified, but no test case is given"
  failed=1
fi

# Illegal arguments.
if [[ -n $failed ]]; then
  exit 1
fi

polybench=../polybench
tamper=../tamper
if [[ -n $newtest ]]; then
  if [[ $newtest != *.* ]]; then
    newtest=$newtest.c
  fi
  code $tamper/test/$newtest
  exit 0
fi

if [[ -n $simul ]]; then
  testpath=$(find $polybench $tamper/test -regextype posix-extended -regex ".*$simul(_test|_simul)\.c(pp)?")
  echo running: $testpath
  echo -n "> "
  clang++ $testpath -o a.out
  ./a.out
  rm a.out
  exit 0
fi

# Build the project.
if [[ -n $run_cmake ]]; then
cmake -G Ninja -S llvm -B build -DCMAKE_BUILD_TYPE=Release -DLLVM_ENABLE_PROJECTS="clang;mlir;compiler-rt" -DCLANG_ENABLE_CIR=ON -DLLVM_BUILD_TARGETS="X86" -DLLVM_ENABLE_ASSERTIONS=ON -DLLVM_USE_LINKER=lld -DCMAKE_C_COMPILER="/usr/local/bin/clang" -DCMAKE_CXX_COMPILER="/usr/local/bin/clang++"
fi

ninja -C build -j $(nproc)
if [[ $? -ne 0 ]]; then
  exit 1
fi

if [[ -n $testcase ]]; then
  testpath=$(find $polybench $tamper/test -name "$testcase.c")
  echo "testing: $testpath"
  output=$tamper/"$testcase.mlir"
  rm -f $output
  # Output a hint to input cache size.
  echo -n "> "
  if [[ -n $valgrind ]]; then
    valgrind clang -I$polybench/utilities -fclangir -emit-cir $testpath -o $output
  else
    clang -I$polybench/utilities -fclangir -emit-cir $testpath -o $output
  fi
  if [[ ! -f $output ]]; then
    echo "error: compilation failed."
  else
    sed -E 's/(^#loc.+)|((^| )loc\(.+\)$)//' $output | awk NF > tmp
    if [[ -n $verbose ]]; then
      cat tmp
    fi
    mv tmp $output
    if [[ -n $display ]]; then
      code $output
    fi
    echo "done."
  fi
fi
