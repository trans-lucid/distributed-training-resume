#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"

cd "$ROOT/generated/main"
python3 -m pip install -e ".[test]" >/tmp/distributed-render-main-install.txt
set +e
python3 -m pytest tests/public/test_checkpoint_unit.py 2>&1 | tee /tmp/distributed-render-main-unit-output.txt
status=${PIPESTATUS[0]}
set -e
if [ "$status" -eq 0 ]; then
  echo "rendered candidate main unexpectedly passed public unit tests"
  exit 1
fi
for expected in optimizer_state_missing rng_state_missing resume_diverged corrupt_checkpoint_unhandled; do
  if ! grep -q "$expected" /tmp/distributed-render-main-unit-output.txt; then
    echo "rendered candidate main did not fail for expected reason: $expected"
    exit 1
  fi
done

cd "$ROOT/generated/solution"
python3 -m pip install -e ".[test]" >/tmp/distributed-render-solution-install.txt
EVAL_TARGET="$PWD/solution" python3 -m pytest tests/public/test_checkpoint_unit.py solution/tests evaluator/tests_hidden

echo "rendered repo smoke validation passed"
