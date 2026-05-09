#!/usr/bin/env python3
"""Build Step7L deterministic heatmap examples from FloorSet training labels."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from puzzleplace.ml.heatmap_dataset import build_heatmap_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-dir", type=Path, default=Path("."))
    parser.add_argument("--floorset-root", type=Path, default=None)
    parser.add_argument("--train-samples", type=int, default=256)
    parser.add_argument("--grid", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--auto-download", action="store_true")
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("artifacts/research/step7l_phase1_heatmap_examples.jsonl"),
    )
    args = parser.parse_args()
    report = build_heatmap_dataset(
        args.base_dir,
        args.out,
        floorset_root=args.floorset_root,
        train_samples=args.train_samples,
        grid_size=args.grid,
        batch_size=args.batch_size,
        auto_download=args.auto_download,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
