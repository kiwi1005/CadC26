#!/usr/bin/env python3
"""Build Step7ML-G training-backed data mart artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from puzzleplace.ml.training_backed_data_mart import build_training_backed_data_mart


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-dir", type=Path, default=Path("."))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/research"))
    parser.add_argument("--floorset-root", type=Path, default=None)
    parser.add_argument("--training-samples", type=int, default=10_000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--auto-download", action="store_true")
    args = parser.parse_args()
    result = build_training_backed_data_mart(
        args.base_dir,
        args.output_dir,
        floorset_root=args.floorset_root,
        requested_training_sample_count=args.training_samples,
        batch_size=args.batch_size,
        auto_download=args.auto_download,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
