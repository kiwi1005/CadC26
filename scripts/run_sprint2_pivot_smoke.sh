#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-$ROOT/.venv/bin/python}"

echo "[1/5] Candidate coverage"
"$PYTHON" "$ROOT/scripts/check_candidate_coverage.py" \
  --case-ids 0 1 2 3 4 \
  --max-traces 2 \
  --modes semantic relaxed strict \
  --output "$ROOT/artifacts/reports/sprint2_pivot_candidate_coverage.json"

echo "[2/5] Semantic rollout"
ROLLOUT_EPOCHS=5 "$PYTHON" "$ROOT/scripts/rollout_validate.py" \
  --case-ids 0 1 2 3 4 \
  --rollout-mode semantic \
  --output "$ROOT/artifacts/reports/sprint2_pivot_semantic_rollout.json"

echo "[3/5] Repair validate"
"$PYTHON" "$ROOT/scripts/repair_validate.py" \
  --case-ids 0 1 2 3 4 \
  --output "$ROOT/artifacts/reports/sprint2_pivot_repair_validate.json"

echo "[4/5] Contest optimizer"
"$PYTHON" "$ROOT/scripts/evaluate_contest_optimizer.py"

echo "[5/5] Sprint 2 summary"
"$PYTHON" "$ROOT/scripts/make_sprint2_pivot_summary.py"
