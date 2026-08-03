#!/usr/bin/env python3
"""Evaluate ranker listwise quality on exact-tail replay files."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from statistics import fmean, median
import sys
from typing import Any

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hcfp.checkpoint import RUNTIME_NORMALIZATION, load_checkpoint  # noqa: E402
from hcfp.data import file_sha256  # noqa: E402
from hcfp.replay import OFFICIAL_TARGET_KIND, ReplayRecord, iter_replay  # noqa: E402


def _named_path(value: str) -> tuple[str, Path]:
    name, separator, raw_path = value.partition("=")
    if not separator or not name or not raw_path:
        raise argparse.ArgumentTypeError("expected NAME=PATH")
    return name, Path(raw_path)


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    rank = max(1, math.ceil(fraction * len(ordered)))
    return ordered[min(len(ordered) - 1, rank - 1)]


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
        name: [record.sample.sample_id for record in records]
        for name, records in replay_records.items()
    }
    record_ids = {
        name: [
            (record.sample.sample_id, record.candidate_stage or "legacy")
            for record in records
        ]
        for name, records in replay_records.items()
    }
    for name, records in replay_records.items():
        if not records:
            raise ValueError(f"replay {name!r} is empty")
        if any(record.target_kind != OFFICIAL_TARGET_KIND for record in records):
            raise ValueError(f"replay {name!r} does not contain official replay targets")
        if len(record_ids[name]) != len(set(record_ids[name])):
            raise ValueError(f"replay {name!r} contains duplicate sample-stage records")
    names = list(sample_ids)
    for index, name in enumerate(names):
        for other in names[index + 1 :]:
            overlap = set(sample_ids[name]) & set(sample_ids[other])
            if overlap:
                raise ValueError(f"replay sample overlap between {name!r} and {other!r}")

    split_results: dict[str, dict[str, object]] = {}
    overall_inputs: dict[str, list[dict[str, Any]]] = {}
    checkpoints: dict[str, object] = {}
    for checkpoint_name, checkpoint_path in args.checkpoint:
        model, metadata = load_checkpoint(
            checkpoint_path,
            expected_normalization=RUNTIME_NORMALIZATION,
            map_location="cpu",
        )
        model = model.to(device=device).eval()
        compatible_hashes = {
            str(value)
            for value in (metadata.get("state_hash"), metadata.get("parent_state_hash"))
            if value is not None
        }
        checkpoints[checkpoint_name] = {
            "path": str(checkpoint_path),
            "sha256": file_sha256(checkpoint_path),
            "state_hash": metadata["state_hash"],
            "parent_state_hash": metadata.get("parent_state_hash"),
            "compatible_replay_hashes": sorted(compatible_hashes),
        }
        for replay_name, records in replay_records.items():
            _validate_checkpoint_compatibility(
                checkpoint_name,
                replay_name,
                records,
                compatible_hashes,
            )
            cases = _evaluate_records(model, records, device)
            summary = _summary(cases)
            split_results.setdefault(replay_name, {})[checkpoint_name] = {
                "summary": summary,
                "by_stage": _stage_summaries(cases),
                "cases": cases,
            }
            overall_inputs.setdefault(checkpoint_name, []).extend(cases)

    overall = {
        checkpoint_name: _summary(cases)
        for checkpoint_name, cases in overall_inputs.items()
    }
    overall_by_stage = {
        checkpoint_name: _stage_summaries(cases)
        for checkpoint_name, cases in overall_inputs.items()
    }
    report = {
        "schema_version": 2,
        "target_kind": OFFICIAL_TARGET_KIND,
        "device": str(device),
        "promotion_gate_policy": {
            "records_required": 16,
            "top1_exact_best_required": 12,
            "top4_oracle_recall_required": 15,
        },
        "replays": {
            name: {
                "path": str(path),
                "sha256": file_sha256(path),
                "records": len(replay_records[name]),
                "samples": len(set(sample_ids[name])),
                "sample_id_sha256": hashlib.sha256(
                    "\n".join(sorted(set(sample_ids[name]))).encode()
                ).hexdigest(),
            }
            for name, path in args.replay
        },
        "checkpoints": checkpoints,
        "results": split_results,
        "overall": overall,
        "overall_by_stage": overall_by_stage,
    }
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


def _validate_checkpoint_compatibility(
    checkpoint_name: str,
    replay_name: str,
    records: list[ReplayRecord],
    compatible_hashes: set[str],
) -> None:
    replay_hashes = {record.checkpoint_hash for record in records}
    if not replay_hashes <= compatible_hashes:
        raise ValueError(
            f"checkpoint {checkpoint_name!r} is not compatible with replay {replay_name!r}: "
            f"{sorted(replay_hashes - compatible_hashes)}"
        )


def _evaluate_records(model: Any, records: list[ReplayRecord], device: torch.device) -> list[dict[str, Any]]:
    mode = _metric_mode(records)
    cases = []
    with torch.inference_mode():
        for record in records:
            case = record.sample.case.to(device=device, dtype=torch.float32)
            features = record.candidate_features.to(device=device)
            target = record.target_score.to(device="cpu", dtype=torch.float32)
            embedding = model.encoder(case)
            prediction = (
                model.ranker(embedding, len(features), features)
                .detach()
                .to(device="cpu", dtype=torch.float32)
            )
            if mode == "schema_v3_listwise":
                cases.append(_v3_case_metrics(record, prediction, target))
            else:
                cases.append(_legacy_case_metrics(record, prediction, target, mode))
    return cases


def _metric_mode(records: list[ReplayRecord]) -> str:
    has_v3 = [
        record.target_rank is not None and record.feasibility_tier is not None
        for record in records
    ]
    if all(has_v3):
        return "schema_v3_listwise"
    if not any(has_v3):
        return "legacy_v2_score_regret"
    raise ValueError("cannot mix schema v3 and legacy replay records in one split")


def _v3_case_metrics(record: ReplayRecord, prediction: torch.Tensor, target: torch.Tensor) -> dict[str, Any]:
    ranks = _required_int_vector(record.target_rank, "target_rank")
    tiers = _required_int_vector(record.feasibility_tier, "feasibility_tier")
    order = _prediction_order(prediction, record.candidate_row_ids)
    selected = int(order[0])
    top4 = [int(index) for index in order[: min(4, len(order))]]
    oracle = int(torch.argmin(ranks).item())
    rank_regret = int(ranks[selected].item() - ranks[oracle].item())
    score_delta = float(target[selected] - target[oracle])
    score_regret = max(0.0, score_delta)
    false_promotion = bool(int(tiers[selected].item()) > int(tiers[oracle].item()))
    return {
        "sample_id": record.sample.sample_id,
        "candidate_stage": record.candidate_stage or "unknown",
        "block_count": int(record.sample.case.n),
        "metric_mode": "schema_v3_listwise",
        "selected_index": selected,
        "oracle_index": oracle,
        "top4_indices": top4,
        "selected_row_id": _row_id(record, selected),
        "oracle_row_id": _row_id(record, oracle),
        "selected_target_rank": int(ranks[selected].item()),
        "oracle_target_rank": int(ranks[oracle].item()),
        "selected_feasibility_tier": int(tiers[selected].item()),
        "oracle_feasibility_tier": int(tiers[oracle].item()),
        "top1_exact_best": selected == oracle,
        "top4_oracle_recall": oracle in top4,
        "rank_regret": rank_regret,
        "score_regret": score_regret,
        "target_score_delta": score_delta,
        "false_promotion": false_promotion,
        "prediction_selected_cost": float(prediction[selected]),
        "prediction_oracle_cost": float(prediction[oracle]),
        "target_selected_score": float(target[selected]),
        "target_oracle_score": float(target[oracle]),
        "weight": _case_weight(record.sample.case.n),
    }


def _legacy_case_metrics(
    record: ReplayRecord,
    prediction: torch.Tensor,
    target: torch.Tensor,
    mode: str,
) -> dict[str, Any]:
    target_order = sorted(range(int(target.numel())), key=lambda index: (float(target[index]), index))
    target_rank = torch.empty_like(target, dtype=torch.long)
    for rank, index in enumerate(target_order):
        target_rank[index] = rank
    order = _prediction_order(prediction, None)
    selected = int(order[0])
    top4 = [int(index) for index in order[: min(4, len(order))]]
    oracle = int(target_order[0])
    regret = float(target[selected] - target[oracle])
    return {
        "sample_id": record.sample.sample_id,
        "candidate_stage": record.candidate_stage or "legacy",
        "block_count": int(record.sample.case.n),
        "metric_mode": mode,
        "selected_index": selected,
        "oracle_index": oracle,
        "top4_indices": top4,
        "top1_exact_best": regret <= 1.0e-8,
        "top4_oracle_recall": oracle in top4,
        "rank_regret": int(target_rank[selected].item()),
        "score_regret": regret,
        "target_score_delta": regret,
        "false_promotion": False,
        "prediction_selected_cost": float(prediction[selected]),
        "prediction_oracle_cost": float(prediction[oracle]),
        "target_selected_score": float(target[selected]),
        "target_oracle_score": float(target[oracle]),
        "weight": _case_weight(record.sample.case.n),
    }


def _prediction_order(
    prediction: torch.Tensor,
    row_ids: tuple[str, ...] | None,
) -> list[int]:
    if prediction.ndim != 1:
        raise ValueError("ranker prediction must be one-dimensional")
    if not bool(torch.isfinite(prediction).all()):
        raise ValueError("ranker predictions must be finite")
    if row_ids is not None and len(row_ids) != int(prediction.numel()):
        raise ValueError("candidate row ids must align with ranker prediction")
    return sorted(
        range(int(prediction.numel())),
        key=lambda index: (
            float(prediction[index]),
            row_ids[index] if row_ids is not None else f"{index:020d}",
        ),
    )


def _required_int_vector(value: torch.Tensor | None, name: str) -> torch.Tensor:
    if value is None:
        raise ValueError(f"{name} is required")
    tensor = value.to(device="cpu")
    if tensor.ndim != 1 or tensor.dtype is not torch.long:
        raise ValueError(f"{name} must be a 1-D torch.long tensor")
    return tensor


def _row_id(record: ReplayRecord, index: int) -> str | None:
    return None if record.candidate_row_ids is None else record.candidate_row_ids[index]


def _summary(cases: list[dict[str, Any]]) -> dict[str, Any]:
    if not cases:
        raise ValueError("cannot summarize empty ranker evaluation")
    weights = [float(case["weight"]) for case in cases]
    rank_regrets = [float(case["rank_regret"]) for case in cases]
    score_regrets = [float(case["score_regret"]) for case in cases]
    top1 = sum(bool(case["top1_exact_best"]) for case in cases)
    top4 = sum(bool(case["top4_oracle_recall"]) for case in cases)
    false_promotions = sum(bool(case["false_promotion"]) for case in cases)
    records = len(cases)
    return {
        "metric_mode": _summary_mode(cases),
        "records": records,
        "top1_exact_best": top1,
        "top1_exact_best_rate": top1 / records,
        "top4_oracle_recall": top4,
        "top4_oracle_recall_rate": top4 / records,
        "false_promotion": false_promotions,
        "false_promotion_rate": false_promotions / records,
        "mean_rank_regret": fmean(rank_regrets),
        "median_rank_regret": median(rank_regrets),
        "p95_rank_regret": _percentile(rank_regrets, 0.95),
        "weighted_rank_regret": _weighted_mean(rank_regrets, weights),
        "mean_score_regret": fmean(score_regrets),
        "median_score_regret": median(score_regrets),
        "p95_score_regret": _percentile(score_regrets, 0.95),
        "weighted_score_regret": _weighted_mean(score_regrets, weights),
        "promotion_gates": _promotion_gates(records, top1, top4),
    }


def _summary_mode(cases: list[dict[str, Any]]) -> str:
    modes = {str(case["metric_mode"]) for case in cases}
    if len(modes) != 1:
        raise ValueError("cannot summarize mixed metric modes")
    return next(iter(modes))


def _stage_summaries(cases: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    stages = sorted({str(case["candidate_stage"]) for case in cases})
    return {
        stage: _summary(
            [case for case in cases if str(case["candidate_stage"]) == stage]
        )
        for stage in stages
    }


def _promotion_gates(records: int, top1: int, top4: int) -> dict[str, Any]:
    evaluable = records == 16
    return {
        "records_required": 16,
        "records": records,
        "evaluable": evaluable,
        "top1_exact_best_required": 12,
        "top1_exact_best": top1,
        "top1_12_of_16_met": (top1 >= 12) if evaluable else None,
        "top4_oracle_recall_required": 15,
        "top4_oracle_recall": top4,
        "top4_15_of_16_met": (top4 >= 15) if evaluable else None,
    }


def _case_weight(block_count: int) -> float:
    return math.exp((int(block_count) - 120) / 12.0)


def _weighted_mean(values: list[float], weights: list[float]) -> float:
    total = sum(weights)
    if total <= 0.0:
        raise ValueError("weights must be positive")
    return sum(value * weight for value, weight in zip(values, weights, strict=True)) / total


if __name__ == "__main__":
    raise SystemExit(main())
