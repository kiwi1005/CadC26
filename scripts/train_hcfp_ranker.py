#!/usr/bin/env python3
"""Fit the repair-aware ranker from exact-tail replay records."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hcfp.analytic import select_device  # noqa: E402
from hcfp.checkpoint import RUNTIME_NORMALIZATION, load_checkpoint, save_checkpoint  # noqa: E402
from hcfp.replay import iter_replay, train_ranker_steps  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("replay")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=1.0e-3)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args(argv)

    device = select_device(args.device)
    model, source = load_checkpoint(
        args.checkpoint,
        expected_normalization=RUNTIME_NORMALIZATION,
        map_location="cpu",
    )
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.ranker.parameters(), lr=args.learning_rate)
    history = train_ranker_steps(
        model,
        iter_replay(args.replay),
        optimizer,
        steps=args.steps,
    )
    checkpoint_hash = save_checkpoint(model, args.output, RUNTIME_NORMALIZATION)
    report = {
        "schema_version": 1,
        "source_checkpoint_hash": source["state_hash"],
        "checkpoint_hash": checkpoint_hash,
        "steps": args.steps,
        "first_loss": history[0],
        "last_loss": history[-1],
        "device": str(device),
        "replay": args.replay,
    }
    Path(f"{args.output}.training.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
