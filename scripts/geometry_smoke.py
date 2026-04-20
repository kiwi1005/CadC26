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

from puzzleplace.data import adapt_validation_batch
from puzzleplace.eval import OfficialEvaluatorWrapper
from scripts.download_smoke import FLOORSET_ROOT, _auto_approve_downloads, _ensure_import_paths


def main() -> None:
    _ensure_import_paths()
    _auto_approve_downloads()
    from iccad2026_evaluate import get_validation_dataloader

    batch = next(iter(get_validation_dataloader(data_path=str(FLOORSET_ROOT), batch_size=1)))
    case = adapt_validation_batch(batch, case_id="validation-0")
    wrapper = OfficialEvaluatorWrapper()
    result = wrapper.evaluate_validation_batch(batch, case_id="validation-0")
    print(json.dumps({
        "case_id": case.case_id,
        "block_count": case.block_count,
        "legal_feasible": result["legality"]["is_feasible"],
        "official_feasible": result["official"]["is_feasible"],
        "cost": result["official"]["cost"],
    }, indent=2))
