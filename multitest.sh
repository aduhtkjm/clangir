polybench=~/llvm/polybench

testpath=$(find $polybench -regextype posix-extended -regex ".*/(.*)\.c(pp)?")
out=small-results.txt
rm -f $out
for f in $testpath; do
  echo "testing: $f"
  echo "$f" >> $out
  case $f in
  *Nussinov|nussinov*) continue;;
  *polybench.c) continue;;
  *template*) continue;;
  *2mmp*|*2mmp_test*) continue;;
  esac

  if ! echo 128 1 | timeout 5s clang -I"$polybench/utilities" -DSMALL_DATASET -emit-cir "$f" -o /dev/null 2>> $out; then
    echo "Timeout or error on: $f"
  fi
done
