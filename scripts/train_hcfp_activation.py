#!/usr/bin/env python3
"""Fit and calibrate a hash-verified learned-tail activation policy."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hcfp.activation import (  # noqa: E402
    activation_policy_metrics,
    fit_activation_policy,
    iter_activation_replay,
    save_activation_policy,
)
from hcfp.data import file_sha256  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-replay", required=True)
    parser.add_argument("--calibration-replay", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--learning-rate", type=float, default=1.0e-2)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--min-train-positives", type=int, default=32)
    parser.add_argument("--min-calibration-positives", type=int, default=16)
    args = parser.parse_args(argv)

    train = list(iter_activation_replay(args.train_replay))
    calibration = list(iter_activation_replay(args.calibration_replay))
    train_positives = sum(record.tail_needed for record in train)
    calibration_positives = sum(record.tail_needed for record in calibration)
    if train_positives < args.min_train_positives:
        raise ValueError(
            f"activation train positives {train_positives} < required {args.min_train_positives}"
        )
    if calibration_positives < args.min_calibration_positives:
        raise ValueError(
            "activation calibration positives "
            f"{calibration_positives} < required {args.min_calibration_positives}"
        )
    policy, history = fit_activation_policy(
        train,
        calibration,
        steps=args.steps,
        learning_rate=args.learning_rate,
        device=args.device,
    )
    payload_hash = save_activation_policy(policy, args.output)
    report = {
        "schema_version": 1,
        "policy": {
            "path": str(Path(args.output).resolve()),
            "sha256": file_sha256(args.output),
            "payload_hash": payload_hash,
            "checkpoint_hash": policy.checkpoint_hash,
            "config_hash": policy.config_hash,
            "feature_version": policy.feature_version,
            "threshold": policy.threshold,
        },
        "training": {
            "steps": args.steps,
            "learning_rate": args.learning_rate,
            "device": args.device,
            "first_loss": history[0],
            "last_loss": history[-1],
            "min_train_positives": args.min_train_positives,
            "min_calibration_positives": args.min_calibration_positives,
        },
        "splits": {
            "train": _split_report(Path(args.train_replay), train),
            "calibration": _split_report(Path(args.calibration_replay), calibration),
        },
        "metrics": {
            "train": _summary_metrics(policy, train),
            "calibration": _summary_metrics(policy, calibration),
        },
    }
    report_path = Path(f"{args.output}.training.json")
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


def _split_report(path: Path, records) -> dict[str, object]:
    return {
        "path": str(path.resolve()),
        "sha256": file_sha256(path),
        "records": len(records),
        "positives": sum(record.tail_needed for record in records),
        "sample_id_sha256": hashlib.sha256(
            "\n".join(record.sample_id for record in records).encode()
        ).hexdigest(),
    }


def _summary_metrics(policy, records) -> dict[str, object]:
    metrics = activation_policy_metrics(policy, records)
    metrics.pop("probabilities")
    return metrics


if __name__ == "__main__":
    raise SystemExit(main())
