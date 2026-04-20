#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for path in (ROOT, SRC):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from puzzleplace.actions import compute_expert_candidate_coverage
from puzzleplace.data import adapt_validation_batch
from puzzleplace.trajectory import generate_pseudo_traces
from scripts.download_smoke import FLOORSET_ROOT, _auto_approve_downloads, _ensure_import_paths


def main() -> None:
    _ensure_import_paths()
    _auto_approve_downloads()
    from iccad2026_evaluate import get_validation_dataloader

    batch = next(iter(get_validation_dataloader(data_path=str(FLOORSET_ROOT), batch_size=1)))
    case = adapt_validation_batch(batch, case_id="validation-0")
    traces = generate_pseudo_traces(case, max_traces=4)
    report = compute_expert_candidate_coverage(case, traces)
    print(json.dumps({
        "case_id": case.case_id,
        "trace_count": len(traces),
        "total_steps": report.total_steps,
        "heuristic_hits": report.heuristic_hits,
        "heuristic_coverage": report.heuristic_coverage,
        "augmented_hits": report.augmented_hits,
        "augmented_coverage": report.augmented_coverage,
        "note": "augmented coverage includes offline teacher-hint candidate injection for supervision-only analysis; heuristic coverage excludes that hint",
    }, indent=2))


if __name__ == "__main__":
    main()
