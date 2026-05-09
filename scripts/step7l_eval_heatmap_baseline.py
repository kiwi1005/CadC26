#!/usr/bin/env python3
"""Evaluate Step7L deterministic heatmap baselines."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from puzzleplace.ml.heatmap_baselines import evaluate_heatmap_baselines


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--examples", type=Path, required=True)
    parser.add_argument("--grid", type=int, default=16)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("artifacts/research/step7l_phase1_heatmap_metrics.json"),
    )
    args = parser.parse_args()
    report = evaluate_heatmap_baselines(args.examples, args.out)
    report["grid"] = args.grid
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
