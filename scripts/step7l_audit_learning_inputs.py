#!/usr/bin/env python3
"""Audit Step7L learning-input boundaries before model work."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from puzzleplace.ml.learning_data_audit import run_learning_data_audit


def _case_list(value: str) -> list[str]:
    if not value.strip():
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-dir", type=Path, default=Path("."))
    parser.add_argument("--floorset-root", type=Path, default=None)
    parser.add_argument("--train-samples", type=int, default=128)
    parser.add_argument("--validation-cases", type=_case_list, default=[])
    parser.add_argument("--grid", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--auto-download", action="store_true")
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("artifacts/research/step7l_phase0_audit.json"),
    )
    args = parser.parse_args()
    result = run_learning_data_audit(
        args.base_dir,
        args.out,
        floorset_root=args.floorset_root,
        train_samples=args.train_samples,
        validation_case_ids=args.validation_cases,
        grid=args.grid,
        batch_size=args.batch_size,
        auto_download=args.auto_download,
    )
    print(
        json.dumps(
            {"decision": result["decision"], "metrics": result["metrics"]},
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
