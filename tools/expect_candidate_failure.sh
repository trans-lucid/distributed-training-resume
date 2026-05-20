#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT/candidate"

set +e
python3 -m pytest tests/public/test_checkpoint_unit.py 2>&1 | tee /tmp/distributed-training-unit-output.txt
status=${PIPESTATUS[0]}
set -e

if [ "$status" -eq 0 ]; then
  echo "candidate starter unexpectedly passed public unit tests"
  exit 1
fi

for expected in optimizer_state_missing rng_state_missing resume_diverged corrupt_checkpoint_unhandled; do
  if ! grep -q "$expected" /tmp/distributed-training-unit-output.txt; then
    echo "public unit tests did not fail for expected reason: $expected"
    exit 1
  fi
done

echo "candidate starter failed public unit tests as expected"
