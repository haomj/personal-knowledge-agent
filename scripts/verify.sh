#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

echo "== Offline tests =="
RUN_LIVE_TESTS=0 uv run pytest -q

if [[ "${RUN_LIVE_TESTS:-0}" == "1" ]]; then
  echo "== Live API tests (per file, to isolate async clients) =="
  for test_file in \
    tests/test_smoke_live.py \
    tests/test_rag_live.py \
    tests/test_mcp_live.py \
    tests/test_multiagent_live.py
  do
    echo "-- ${test_file}"
    RUN_LIVE_TESTS=1 uv run pytest "${test_file}" -q
  done
else
  echo "Skip live tests. Set RUN_LIVE_TESTS=1 to enable."
fi
