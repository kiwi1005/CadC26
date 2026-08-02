#!/usr/bin/env python3
"""Compare ranker top-1 regret on versioned exact-tail replay files."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from statistics import fmean, median
import sys

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hcfp.checkpoint import RUNTIME_NORMALIZATION, load_checkpoint  # noqa: E402
from hcfp.data import file_sha256  # noqa: E402
from hcfp.replay import OFFICIAL_TARGET_KIND, iter_replay  # noqa: E402


def _named_path(value: str) -> tuple[str, Path]:
    name, separator, raw_path = value.partition("=")
    if not separator or not name or not raw_path:
        raise argparse.ArgumentTypeError("expected NAME=PATH")
    return name, Path(raw_path)


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(fraction * (len(ordered) - 1)))]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay", action="append", type=_named_path, required=True)
    parser.add_argument("--checkpoint", action="append", type=_named_path, required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args(argv)

    device_name = args.device
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_name)
    replay_records = {name: list(iter_replay(path)) for name, path in args.replay}
    sample_ids = {
        name: {record.sample.sample_id for record in records}
        for name, records in replay_records.items()
    }
    for name, records in replay_records.items():
        if not records or any(record.target_kind != OFFICIAL_TARGET_KIND for record in records):
            raise ValueError(f"replay {name!r} does not contain official replay targets")
    names = list(sample_ids)
    for index, name in enumerate(names):
        for other in names[index + 1 :]:
            overlap = sample_ids[name] & sample_ids[other]
            if overlap:
                raise ValueError(f"replay sample overlap between {name!r} and {other!r}")

    results: dict[str, dict[str, object]] = {}
    checkpoints: dict[str, object] = {}
    for checkpoint_name, checkpoint_path in args.checkpoint:
        model, metadata = load_checkpoint(
            checkpoint_path,
            expected_normalization=RUNTIME_NORMALIZATION,
            map_location="cpu",
        )
        model = model.to(device=device).eval()
        checkpoints[checkpoint_name] = {
            "path": str(checkpoint_path),
            "sha256": file_sha256(checkpoint_path),
            "state_hash": metadata["state_hash"],
        }
        for replay_name, records in replay_records.items():
            regrets: list[float] = []
            index_exact = equivalent_exact = 0
            with torch.inference_mode():
                for record in records:
                    case = record.sample.case.to(device=device, dtype=torch.float32)
                    features = record.candidate_features.to(device=device)
                    target = record.target_score
                    embedding = model.encoder(case)
                    prediction = model.ranker(embedding, len(features), features)
                    selected = int(torch.argmin(prediction).to(device="cpu"))
                    oracle = int(torch.argmin(target))
                    regret = float(target[selected] - target[oracle])
                    index_exact += int(selected == oracle)
                    equivalent_exact += int(regret <= 1.0e-8)
                    regrets.append(regret)
            results.setdefault(replay_name, {})[checkpoint_name] = {
                "records": len(records),
                "top1_exact": equivalent_exact,
                "top1_rate": equivalent_exact / len(records),
                "top1_index_exact": index_exact,
                "top1_index_rate": index_exact / len(records),
                "mean_regret": fmean(regrets),
                "median_regret": median(regrets),
                "p95_regret": _percentile(regrets, 0.95),
            }

    report = {
        "schema_version": 1,
        "target_kind": OFFICIAL_TARGET_KIND,
        "device": str(device),
        "replays": {
            name: {
                "path": str(path),
                "sha256": file_sha256(path),
                "records": len(replay_records[name]),
                "sample_id_sha256": hashlib.sha256(
                    "\n".join(sorted(sample_ids[name])).encode()
                ).hexdigest(),
            }
            for name, path in args.replay
        },
        "checkpoints": checkpoints,
        "results": results,
    }
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
