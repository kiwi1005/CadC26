#!/usr/bin/env python3
"""Evaluate Step7ML-H macro layout predictions on Step7P payloads."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from puzzleplace.ml.supervised_macro_layout import evaluate_on_step7p, load_json, write_decision


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("artifacts/models/step7ml_h/macro_layout_mlp.pt"),
    )
    parser.add_argument("--base-dir", type=Path, default=Path("."))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/research"))
    args = parser.parse_args()
    eval_report = evaluate_on_step7p(args.checkpoint, args.base_dir, args.output_dir)
    training_report = load_json(args.output_dir / "step7ml_h_training_report.json")
    decision = write_decision(args.output_dir, training_report, eval_report)
    print(json.dumps({"decision": decision, "eval_report": eval_report}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
