#!/usr/bin/env python3
"""Fit the repair-aware ranker from exact-tail replay records."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hcfp.analytic import select_device  # noqa: E402
from hcfp.checkpoint import RUNTIME_NORMALIZATION, load_checkpoint, save_checkpoint  # noqa: E402
from hcfp.data import file_sha256  # noqa: E402
from hcfp.replay import OFFICIAL_TARGET_KIND, iter_replay, train_ranker_steps  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("replay")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=1.0e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args(argv)

    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(args.seed)
    device = select_device(args.device)
    model, source = load_checkpoint(
        args.checkpoint,
        expected_normalization=RUNTIME_NORMALIZATION,
        map_location="cpu",
    )
    records = list(iter_replay(args.replay))
    if not records:
        raise ValueError("ranker training requires at least one replay record")
    if any(record.target_kind != OFFICIAL_TARGET_KIND for record in records):
        raise ValueError("ranker training requires official v10 replay targets")
    replay_hashes = {record.checkpoint_hash for record in records}
    if replay_hashes != {source["state_hash"]}:
        raise ValueError("replay checkpoint hash does not match source checkpoint")
    targets = torch.cat([record.target_score.reshape(-1) for record in records])
    if not bool(torch.isfinite(targets).all()):
        raise ValueError("replay targets must be finite")
    records_with_signal = sum(
        float(record.target_score.max() - record.target_score.min()) > 1.0e-8
        for record in records
    )
    if not records_with_signal:
        raise ValueError("replay targets contain no ranking signal")
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.ranker.parameters(), lr=args.learning_rate)
    history = train_ranker_steps(
        model,
        records,
        optimizer,
        steps=args.steps,
    )
    checkpoint_metadata = {
        "capabilities": dict(source["capabilities"]),
        "trained_heads": sorted({*source["trained_heads"], "ranker"}),
        "training_objective_version": "ranker_official_v10_v1",
        "parent_state_hash": source["state_hash"],
    }
    checkpoint_hash = save_checkpoint(
        model,
        args.output,
        RUNTIME_NORMALIZATION,
        metadata=checkpoint_metadata,
    )
    report = {
        "schema_version": 1,
        "source_checkpoint_hash": source["state_hash"],
        "checkpoint_hash": checkpoint_hash,
        "checkpoint_metadata": checkpoint_metadata,
        "checkpoint_sha256": file_sha256(args.output),
        "source_checkpoint": args.checkpoint,
        "source_checkpoint_sha256": file_sha256(args.checkpoint),
        "seed": args.seed,
        "steps": args.steps,
        "first_loss": history[0],
        "last_loss": history[-1],
        "target_kind": OFFICIAL_TARGET_KIND,
        "device": str(device),
        "replay": args.replay,
        "replay_sha256": file_sha256(args.replay),
        "records": len(records),
        "target_distribution": {
            "count": int(targets.numel()),
            "minimum": float(targets.min()),
            "maximum": float(targets.max()),
            "mean": float(targets.mean()),
            "standard_deviation": float(targets.std(unbiased=False)),
            "records_with_signal": records_with_signal,
        },
    }
    Path(f"{args.output}.training.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
