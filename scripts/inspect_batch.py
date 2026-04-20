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

from puzzleplace.data import adapt_training_batch, adapt_validation_batch
from scripts.download_smoke import FLOORSET_ROOT, _auto_approve_downloads, _ensure_import_paths


def main() -> None:
    _ensure_import_paths()
    _auto_approve_downloads()
    from iccad2026_evaluate import get_training_dataloader, get_validation_dataloader

    train_batch = next(iter(get_training_dataloader(data_path=str(FLOORSET_ROOT), batch_size=1, num_samples=1, shuffle=False)))
    train_case = adapt_training_batch(train_batch, case_id="train-0")

    validation_batch = next(iter(get_validation_dataloader(data_path=str(FLOORSET_ROOT), batch_size=1)))
    validation_case = adapt_validation_batch(validation_batch, case_id="validation-0")

    print(json.dumps({
        "train": {
            "case_id": train_case.case_id,
            "block_count": train_case.block_count,
            "has_target_positions": train_case.target_positions is not None,
        },
        "validation": {
            "case_id": validation_case.case_id,
            "block_count": validation_case.block_count,
            "has_target_positions": validation_case.target_positions is not None,
        },
    }, indent=2))


if __name__ == "__main__":
    main()
