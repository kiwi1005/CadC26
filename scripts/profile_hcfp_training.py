#!/usr/bin/env python3
"""Profile one HCFP supervised training configuration."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hcfp.training_profile import TrainingProfileConfig, run_training_profile  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--blocks", type=int, default=120)
    parser.add_argument("--population", type=int, default=8)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--encoder-layers", type=int, default=3)
    parser.add_argument("--stage", default="all")
    parser.add_argument("--compute-dtype", choices=("float32", "bfloat16"), default="float32")
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output")
    args = parser.parse_args(argv)
    report = run_training_profile(
        TrainingProfileConfig(
            block_count=args.blocks,
            population=args.population,
            hidden_dim=args.hidden_dim,
            encoder_layers=args.encoder_layers,
            stage=args.stage,
            compute_dtype=args.compute_dtype,
            warmups=args.warmups,
            steps=args.steps,
            device=args.device,
        )
    )
    text = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    else:
        print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
