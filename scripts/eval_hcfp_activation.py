#!/usr/bin/env python3
"""Evaluate learned-tail activation recall and estimated counterfactual runtime."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from statistics import median
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hcfp.activation import (  # noqa: E402
    activation_policy_metrics,
    iter_activation_replay,
    load_activation_policy,
)
from hcfp.data import file_sha256  # noqa: E402


def _named_path(value: str) -> tuple[str, Path]:
    name, separator, raw_path = value.partition("=")
    if not separator or not name or not raw_path:
        raise argparse.ArgumentTypeError("expected NAME=PATH")
    return name, Path(raw_path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", required=True)
    parser.add_argument("--replay", action="append", type=_named_path, required=True)
    parser.add_argument("--training-report", required=True)
    parser.add_argument("--force-large-min", type=int, default=106)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)

    policy = load_activation_policy(args.policy)
    replays = _load_named_replays(args.replay)
    excluded, exclusion_report = _load_training_exclusions(
        Path(args.training_report),
        Path(args.policy),
    )
    _reject_overlap({**replays, **excluded})
    results = {}
    replay_report = {}
    for name, path in args.replay:
        records = replays[name]
        metrics = activation_policy_metrics(
            policy,
            records,
            force_large_min=args.force_large_min,
        )
        probabilities = metrics.pop("probabilities")
        active = [
            record.block_count >= args.force_large_min or probability >= policy.threshold
            for record, probability in zip(records, probabilities)
        ]
        counterfactual = [
            record.learned.runtime_seconds if run else record.analytic.runtime_seconds
            for record, run in zip(records, active)
        ]
        results[name] = {
            **metrics,
            "estimated_counterfactual_runtime_p50": median(counterfactual),
            "estimated_counterfactual_runtime_p95": _percentile(counterfactual, 0.95),
            "estimated_analytic_runtime_p50": median(
                [record.analytic.runtime_seconds for record in records]
            ),
            "estimated_learned_runtime_p50": median(
                [record.learned.runtime_seconds for record in records]
            ),
            "bucket_metrics": _bucket_metrics(records, active),
        }
        replay_report[name] = {
            "path": str(path),
            "sha256": file_sha256(path),
            "records": len(records),
            "sample_id_sha256": hashlib.sha256(
                "\n".join(record.sample_id for record in records).encode()
            ).hexdigest(),
        }

    report = {
        "schema_version": 1,
        "policy": {
            "path": args.policy,
            "sha256": file_sha256(args.policy),
            "checkpoint_hash": policy.checkpoint_hash,
            "config_hash": policy.config_hash,
            "feature_version": policy.feature_version,
            "threshold": policy.threshold,
        },
        "force_large_min": args.force_large_min,
        "runtime_warning": (
            "Replay runtimes are component estimates, not promotion evidence; "
            "run the official live benchmark before activation."
        ),
        "training_exclusions": exclusion_report,
        "replays": replay_report,
        "results": results,
    }
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


def _reject_overlap(replays: dict[str, list]) -> None:
    names = list(replays)
    for index, name in enumerate(names):
        raw_ids = [record.sample_id for record in replays[name]]
        ids = set(raw_ids)
        if len(ids) != len(raw_ids):
            raise ValueError(f"activation replay {name!r} contains duplicate samples")
        for other in names[index + 1 :]:
            overlap = ids & {record.sample_id for record in replays[other]}
            if overlap:
                raise ValueError(f"activation replay sample overlap between {name!r} and {other!r}")


def _load_named_replays(named_paths: list[tuple[str, Path]]) -> dict[str, list]:
    names = [name for name, _ in named_paths]
    if len(names) != len(set(names)):
        raise ValueError("activation replay names must be unique")
    if any(name.startswith("__excluded_") for name in names):
        raise ValueError("activation replay name uses a reserved prefix")
    return {name: list(iter_activation_replay(path)) for name, path in named_paths}


def _load_training_exclusions(
    report_path: Path,
    policy_path: Path,
) -> tuple[dict[str, list], dict[str, object]]:
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("schema_version") != 1:
        raise ValueError("activation training report schema mismatch")
    policy_report = report.get("policy")
    if not isinstance(policy_report, dict) or policy_report.get("sha256") != file_sha256(policy_path):
        raise ValueError("activation training report does not match policy")
    policy_payload = json.loads(policy_path.read_text(encoding="utf-8"))
    if policy_report.get("payload_hash") != policy_payload.get("payload_hash"):
        raise ValueError("activation training report policy payload hash mismatch")
    splits = report.get("splits")
    if not isinstance(splits, dict) or set(splits) != {"train", "calibration"}:
        raise ValueError("activation training report must contain train and calibration splits")

    excluded = {}
    verified = {}
    for name in ("train", "calibration"):
        split = splits[name]
        if not isinstance(split, dict) or not isinstance(split.get("path"), str):
            raise ValueError(f"activation training report {name} split is invalid")
        path = Path(split["path"])
        digest = file_sha256(path)
        if digest != split.get("sha256"):
            raise ValueError(f"activation training report {name} hash mismatch")
        records = list(iter_activation_replay(path))
        sample_digest = hashlib.sha256(
            "\n".join(record.sample_id for record in records).encode()
        ).hexdigest()
        if len(records) != split.get("records") or sample_digest != split.get("sample_id_sha256"):
            raise ValueError(f"activation training report {name} provenance mismatch")
        excluded[f"__excluded_{name}"] = records
        verified[name] = {
            "path": str(path),
            "sha256": digest,
            "records": len(records),
            "sample_id_sha256": sample_digest,
        }
    return excluded, {
        "path": str(report_path),
        "sha256": file_sha256(report_path),
        "splits": verified,
    }


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(fraction * (len(ordered) - 1)))]


def _bucket_metrics(records, active: list[bool]) -> dict[str, object]:
    result = {}
    for lower, upper in ((1, 32), (33, 64), (65, 96), (97, 120)):
        selected = [
            (record, run)
            for record, run in zip(records, active)
            if lower <= record.block_count <= upper
        ]
        if not selected:
            continue
        positives = sum(record.tail_needed for record, _ in selected)
        true_positive = sum(record.tail_needed and run for record, run in selected)
        result[f"{lower}-{upper}"] = {
            "records": len(selected),
            "positives": positives,
            "activation_rate": sum(run for _, run in selected) / len(selected),
            "recall": true_positive / max(positives, 1),
        }
    return result


if __name__ == "__main__":
    raise SystemExit(main())
