polybench=~/llvm/polybench

testpath=$(find $polybench -regextype posix-extended -regex ".*/(.*)\.c(pp)?")
out=small-results.txt
rm -f $out
for f in $testpath; do
  case $f in
  *Nussinov*|*nussinov*) continue;;
  *polybench.c) continue;;
  *template*) continue;;
  *2mmp*|*2mmp_test*) continue;;
  esac
  echo "testing: $f"
  echo "$f" >> $out

  if ! echo 524288 1 | timeout 600s clang -I"$polybench/utilities" -DLARGE_DATASET -emit-cir "$f" -o /dev/null 2>> $out; then
    echo "Timeout or error on: $f"
  fi
done
