#!/usr/bin/env python3
"""Run Step7ML-I geometry-aware macro decoding probe."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from puzzleplace.ml.geometry_aware_decoder import run_geometry_aware_decoding


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-dir", type=Path, default=Path("."))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/research"))
    args = parser.parse_args()
    result = run_geometry_aware_decoding(args.base_dir, args.output_dir)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
