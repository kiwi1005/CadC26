#!/usr/bin/env python3
"""Validate/download FloorSet-Lite training data and export Step7DATA labels."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from puzzleplace.ml.floorset_training_corpus import run_step7data


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-dir", type=Path, default=Path("."))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/research"))
    parser.add_argument("--floorset-root", type=Path, default=None)
    parser.add_argument("--auto-download", action="store_true")
    parser.add_argument("--num-samples", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-examples", type=int, default=1000)
    parser.add_argument("--max-macro-labels", type=int, default=5000)
    args = parser.parse_args()
    result = run_step7data(
        args.base_dir,
        args.output_dir,
        floorset_root=args.floorset_root,
        auto_download=args.auto_download,
        num_samples=args.num_samples,
        batch_size=args.batch_size,
        max_examples=args.max_examples,
        max_macro_labels=args.max_macro_labels,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
