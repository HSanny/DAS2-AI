#!/usr/bin/env bash
# Run the whole suite. Usage: bash tests/run_all.sh
set -u
cd "$(dirname "$0")/.."
fail=0; total=0
for t in tests/test_*.py; do
  printf '%-32s ' "$(basename "$t")"
  if out=$(timeout 300 python3 "$t" 2>&1); then
    n=$(grep -c PASS <<<"$out"); total=$((total+n)); echo "PASS ($n assertions)"
  else
    echo "FAIL"; echo "$out" | tail -20; fail=1
  fi
done
printf '%-32s ' "cluster_suppression self-test"
if python3 docker_ready/cluster_suppression.py >/dev/null 2>&1; then echo "PASS"; else echo "FAIL"; fail=1; fi
echo "-----"; echo "total: $total assertions"
exit $fail
