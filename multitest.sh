polybench=~/llvm/polybench

testpath=$(find $polybench -regextype posix-extended -regex ".*/(.*)\.c(pp)?")
for f in $testpath; do
  echo "testing: $f"
  if ! echo 524288 1 | timeout 5s clang -I"$polybench/utilities" -DSMALL_DATASET -emit-cir "$f" -o /dev/null; then
    echo "Timeout or error on: $f"
  fi
done
