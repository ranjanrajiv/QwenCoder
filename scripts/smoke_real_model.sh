#!/usr/bin/env bash
#
# Manual real-model smoke test (spec 03 sections 34, 48).
#
# Loads the configured Qwen model, generates ONE candidate for ONE problem, extracts the
# Python, validates its syntax, and persists it.
#
# This is NOT part of the automated test suite and must never run in CI: it downloads
# several GB of weights on first use and needs a GPU to finish in reasonable time.
# Run it deliberately, by hand.
#
# Usage:
#   scripts/smoke_real_model.sh [problem_id]
#
# Prerequisites:
#   pip install -e '.[model]'
#   python -m python_dpo problems build      # data/problems/problems.jsonl must exist
#
# For gated or private models, export HF_TOKEN in your shell first. Never write its value
# into config.yaml, source code, or any committed file.

set -euo pipefail

PROBLEM_ID="${1:-p001}"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

cd "$PROJECT_ROOT"

echo "=============================================================================="
echo " Real-model smoke test"
echo "   problem:    $PROBLEM_ID"
echo "   candidates: 1"
echo
echo " This loads the model configured in config.yaml and may download several GB"
echo " on first run. Press Ctrl-C within 5 seconds to abort."
echo "=============================================================================="
sleep 5

python -m python_dpo generate \
    --problem-id "$PROBLEM_ID" \
    --num-candidates 1

echo
echo "Candidate written to data/candidates/candidates.jsonl:"
tail -n 1 data/candidates/candidates.jsonl \
    | python -c 'import json, sys; r = json.loads(sys.stdin.read()); print(json.dumps({k: r[k] for k in ("candidate_id", "run_id", "model", "strategy", "extraction_format", "syntax_valid", "function_name_valid")}, indent=2)); print(); print(r["code"])'
