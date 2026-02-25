#!/bin/zsh

die() {
  echo -e "\e[31merror:\e[0m $1"
}
warn() {
  echo -e "\e[33mwarning:\e[0m $1"
}

export FALCON_PYTHON_ADAPTOR="/home/aduhtkjm/llvm/model-counter/adaptor.py"
while [[ $# -gt 0 ]]; do
  case "$1" in
    -c|--cmake)
      run_cmake=1; shift;;
    -C|--cache-size)
      if [[ -z $2 ]]; then
        die "expected cache size after $1"
        exit 1
      fi
      cachesize=$2; shift 2;;
    -l|--cacheline-size)
      if [[ -z $2 ]]; then
        die "expected cache line size after $1"
        exit 1
      fi
      cachelinesize=$2; shift 2;;
    -u|--unit-test)
      ninja -C build check-mlir-unit
      exit 0;;
    -b|--barvinok)
      if [[ -z $2 ]]; then
        warn "no input file after $1, default to ./input.txt"
        input=./input.txt
      else
        input=$2
      fi
      cat "$input" | /opt/bin/barvinok_count
      exit 0;;
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
    --gdb)
      gdb=1; shift;;
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
    -S|--small)
      small=1; shift;;
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

# Give a default cache(-line) size.
if [[ -z $cachesize ]]; then
  if [[ -n $testcase || -n $simul ]]; then
    cachesize=4
    warn "cache size not specified, default to $cachesize"
  fi
fi

if [[ -z $cachelinesize ]]; then
  if [[ -n $testcase || -n $simul ]]; then
    cachelinesize=1
    warn "cacheline size not specified, default to $cachelinesize"
  fi
fi

polybench=../polybench
tamper=../tamper
if [[ -n $newtest ]]; then
  if [[ $newtest != *.* ]]; then
    die "must specify suffix when creating a file"
    exit 1
  fi
  code $tamper/test/$newtest
  exit 0
fi

if [[ -n $simul ]]; then
  testpath=$(find $polybench $tamper/test -regextype posix-extended -regex ".*$simul(_test|_simul)\.c(pp)?")
  echo running: $testpath
  if [[ -n $verbose ]]; then
    clang++ -O2 -DVERBOSE $testpath -o a.out
  else
    clang++ -O2 $testpath -o a.out
  fi
  echo $cachesize $cachelinesize | ./a.out
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
  testpath=$(find $polybench $tamper/test -regextype posix-extended -regex ".*/$testcase\.c(pp)?")
  echo "testing: $testpath"
  output=$tamper/"$testcase.mlir"
  rm -f $output
  cmd="clang -I$polybench/utilities -emit-cir $testpath -o $output"
  if [[ -n $valgrind ]]; then
    cmd="echo $cachesize $cachelinesize | valgrind $cmd"
  elif [[ -n $gdb ]]; then
    cmd="gdb --args $cmd"
  else
    cmd="echo $cachesize $cachelinesize | $cmd"
  fi
  if [[ -n $small ]]; then
    cmd="$cmd -DSMALL_DATASET"
  fi
  eval $cmd
  if [[ ! -f $output ]]; then
    die "compilation failed."
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
