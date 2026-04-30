#!/usr/bin/env python3
"""Run Step7N-I quality-filtered macro slot/repack integration."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from puzzleplace.search.quality_filtered_macro_integration import run_step7n_i


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-dir", type=Path, default=Path("."))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/research"))
    args = parser.parse_args()
    result = run_step7n_i(args.base_dir, args.output_dir)
    print(
        json.dumps(
            {"decision": result["decision"], "metrics": result["metrics"]}, indent=2, sort_keys=True
        )
    )


if __name__ == "__main__":
    main()
