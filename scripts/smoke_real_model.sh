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

# Stage 4: every `generate` call creates its own run directory under
# data/candidates/runs/<run_id>/, so the run this invocation just created is whichever
# one is newest.
RUN_ID="$(python -c '
from python_dpo.config import Config
from python_dpo.runs import RunRepository

repo = RunRepository(Config.load().paths.candidates / "runs")
print(repo.list_runs()[0].run_id)
')"

echo
echo "Run: $RUN_ID"
python -m python_dpo runs show "$RUN_ID"
echo
python -m python_dpo candidates show "$RUN_ID" "${PROBLEM_ID}_c001" --show-code
echo
python -m python_dpo runs validate "$RUN_ID"
