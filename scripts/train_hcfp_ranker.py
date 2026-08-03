#!/usr/bin/env python3
"""Fit the repair-aware ranker from exact-tail replay records."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
import os
from pathlib import Path
from statistics import fmean
import sys

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hcfp.analytic import select_device  # noqa: E402
from hcfp.checkpoint import RUNTIME_NORMALIZATION, load_checkpoint, save_checkpoint  # noqa: E402
from hcfp.data import file_sha256  # noqa: E402
from hcfp.ranker_features import RANKER_FEATURE_DIM, RANKER_FEATURE_VERSION  # noqa: E402
from hcfp.ranker_upgrade import upgrade_candidate_metric_dim  # noqa: E402
from hcfp.replay import (  # noqa: E402
    OFFICIAL_TARGET_KIND,
    RANKER_OBJECTIVES,
    RANKER_SAMPLING_PRESETS,
    iter_replay,
    ranker_features_for_record,
    ranker_training_schedule,
    train_ranker_steps,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("replay", nargs="+")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=1.0e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--stage", choices=("all", "initial", "post_relax"), default="all")
    parser.add_argument(
        "--objective-preset",
        choices=tuple(RANKER_OBJECTIVES),
        default="default",
        help="Ranker loss preset. The default preserves the existing objective.",
    )
    parser.add_argument(
        "--sampling-preset",
        choices=RANKER_SAMPLING_PRESETS,
        default="legacy",
        help="Ranker record sampler. The default preserves legacy modulo order.",
    )
    parser.add_argument(
        "--sampling-seed",
        type=int,
        help="Deterministic sampler seed. Defaults to --seed.",
    )
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
    records_by_replay = {
        str(path): list(iter_replay(path))
        for path in args.replay
    }
    replay_names = list(records_by_replay)
    for index, name in enumerate(replay_names):
        sample_ids = {record.sample.sample_id for record in records_by_replay[name]}
        for other in replay_names[index + 1 :]:
            other_ids = {record.sample.sample_id for record in records_by_replay[other]}
            if sample_ids & other_ids:
                raise ValueError("ranker training replay files must have disjoint sample IDs")
    records = [record for name in replay_names for record in records_by_replay[name]]
    if args.stage != "all":
        records = [record for record in records if record.candidate_stage == args.stage]
    selected_record_counts = {
        name: sum(
            args.stage == "all" or record.candidate_stage == args.stage
            for record in replay_records
        )
        for name, replay_records in records_by_replay.items()
    }
    replay_stats = {
        name: {
            "records": len(replay_records),
            "selected_records": selected_record_counts[name],
            "samples": len({record.sample.sample_id for record in replay_records}),
        }
        for name, replay_records in records_by_replay.items()
    }
    if not records:
        raise ValueError("ranker training requires at least one replay record for the selected stage")
    if any(record.target_kind != OFFICIAL_TARGET_KIND for record in records):
        raise ValueError("ranker training requires official v10 replay targets")
    replay_hashes = {record.checkpoint_hash for record in records}
    if replay_hashes != {source["state_hash"]}:
        raise ValueError("replay checkpoint hash does not match source checkpoint")
    targets = torch.cat([record.target_score.reshape(-1) for record in records])
    if not bool(torch.isfinite(targets).all()):
        raise ValueError("replay targets must be finite")
    score_records_with_signal = sum(
        float(record.target_score.max() - record.target_score.min()) > 1.0e-8
        for record in records
    )
    listwise_records = sum(record.target_rank is not None for record in records)
    if listwise_records not in {0, len(records)}:
        raise ValueError("ranker training cannot mix listwise and legacy replay records")
    records_with_signal = (
        sum(
            float(record.target_score.max() - record.target_score.min()) > 1.0e-8
            or (
                record.feasibility_tier is not None
                and int(torch.unique(record.feasibility_tier).numel()) > 1
            )
            for record in records
        )
        if listwise_records
        else score_records_with_signal
    )
    if not records_with_signal:
        raise ValueError("replay targets contain no ranking signal")
    feature_dim = (
        RANKER_FEATURE_DIM
        if listwise_records == len(records)
        else model.config.candidate_metric_dim
    )
    feature_version = (
        RANKER_FEATURE_VERSION
        if feature_dim == RANKER_FEATURE_DIM
        else "stored_candidate_features_v1"
    )
    records = [
        replace(
            record,
            candidate_features=ranker_features_for_record(
                record,
                expected_dim=feature_dim,
                expected_version=feature_version,
            ),
            candidate_geometry=None,
            post_bdp_geometry=None,
            post_repair_geometry=None,
            teacher_delta_xy=None,
            repair_displacement=None,
            post_repair_log_uncapped_cost=None,
            boundary_violations=None,
            grouping_violations=None,
            mib_violations=None,
        )
        for record in records
    ]
    del records_by_replay
    continuing_ranker = _checkpoint_has_trained_ranker(source)
    objective = (
        RANKER_OBJECTIVES[args.objective_preset]
        if listwise_records
        else None
    )
    expected_scene_embedding = (
        False if listwise_records else model.config.ranker_use_scene_embedding
    )
    if continuing_ranker:
        if listwise_records and source.get("training_objective_version") != objective.name:
            raise ValueError("ranker continuation requires the same training objective")
        if listwise_records and source.get("training_objective_weights") != objective.as_metadata():
            raise ValueError("ranker continuation requires the same objective weights")
        if (
            model.config.candidate_metric_dim != feature_dim
            or model.config.ranker_feature_version != feature_version
            or model.config.ranker_use_scene_embedding != expected_scene_embedding
            or not model.config.ranker_feature_mean
            or not model.config.ranker_feature_scale
        ):
            raise ValueError(
                "ranker continuation requires the same feature contract and "
                "checkpoint normalization"
            )
        feature_mean = torch.tensor(
            model.config.ranker_feature_mean,
            dtype=torch.float32,
        )
        feature_scale = torch.tensor(
            model.config.ranker_feature_scale,
            dtype=torch.float32,
        )
        normalization_source = "source_checkpoint"
    else:
        feature_rows = torch.cat([record.candidate_features for record in records], dim=0)
        feature_mean = feature_rows.mean(dim=0)
        feature_scale = feature_rows.std(dim=0, unbiased=False)
        feature_scale = torch.where(
            feature_scale > 1.0e-6,
            feature_scale,
            torch.ones_like(feature_scale),
        )
        normalization_source = "training_replay"
    model = upgrade_candidate_metric_dim(
        model,
        feature_dim,
        source_metadata=source,
        feature_mean=feature_mean.tolist(),
        feature_scale=feature_scale.tolist(),
        feature_version=feature_version,
        use_scene_embedding=expected_scene_embedding,
    )
    objective_version = (
        objective.name
        if listwise_records and objective is not None
        else "ranker_official_v10_pointwise_v1"
    )
    objective_weights = (
        objective.as_metadata()
        if listwise_records and objective is not None
        else {
            "name": "ranker_official_v10_pointwise_v1",
            "listwise": 0.0,
            "feasibility_order": 0.0,
            "pointwise": 1.0,
            "top_one": 0.0,
        }
    )
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.ranker.parameters(), lr=args.learning_rate)
    sampling_seed = args.seed if args.sampling_seed is None else args.sampling_seed
    sampling_plan = ranker_training_schedule(
        records,
        steps=args.steps,
        seed=sampling_seed,
        preset=args.sampling_preset,
    )
    history = train_ranker_steps(
        model,
        records,
        optimizer,
        steps=args.steps,
        report_components=True,
        objective=objective or RANKER_OBJECTIVES["default"],
        sampling_indices=(
            None if args.sampling_preset == "legacy" else sampling_plan.indices
        ),
    )
    loss_window = min(len(records), len(history))
    capabilities = dict(source["capabilities"])
    capabilities["ranker"] = False
    checkpoint_metadata = {
        "capabilities": capabilities,
        "trained_heads": sorted({*source["trained_heads"], "ranker"}),
        "training_objective_version": objective_version,
        "training_objective_weights": objective_weights,
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
        "first_loss": history[0]["combined"],
        "last_loss": history[-1]["combined"],
        "first_window_mean_loss": fmean(
            item["combined"] for item in history[:loss_window]
        ),
        "last_window_mean_loss": fmean(
            item["combined"] for item in history[-loss_window:]
        ),
        "loss_window_records": loss_window,
        "first_loss_components": history[0],
        "last_loss_components": history[-1],
        "target_kind": OFFICIAL_TARGET_KIND,
        "device": str(device),
        "replays": [
            {
                "path": path,
                "sha256": file_sha256(path),
                **replay_stats[path],
            }
            for path in replay_names
        ],
        "records": len(records),
        "listwise_records": listwise_records,
        "objective_preset": args.objective_preset if listwise_records else "pointwise",
        "training_objective_version": objective_version,
        "training_objective_weights": objective_weights,
        "sampling": sampling_plan.metadata,
        "candidate_stage_filter": args.stage,
        "candidate_feature_version": feature_version,
        "candidate_feature_dim": feature_dim,
        "ranker_use_scene_embedding": model.config.ranker_use_scene_embedding,
        "training_record_storage": "ranker_features_only_v1",
        "ranker_initialization": (
            "continued_from_source_checkpoint"
            if continuing_ranker
            else "reset_from_non_ranker_source"
        ),
        "candidate_feature_normalization": {
            "kind": "global_zscore_constant_identity_v2",
            "source": normalization_source,
            "mean": list(model.config.ranker_feature_mean),
            "scale": list(model.config.ranker_feature_scale),
        },
        "target_distribution": {
            "count": int(targets.numel()),
            "minimum": float(targets.min()),
            "maximum": float(targets.max()),
            "mean": float(targets.mean()),
            "standard_deviation": float(targets.std(unbiased=False)),
            "records_with_signal": records_with_signal,
            "score_records_with_signal": score_records_with_signal,
        },
    }
    Path(f"{args.output}.training.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


def _checkpoint_has_trained_ranker(metadata: dict[str, object]) -> bool:
    trained_heads = metadata.get("trained_heads", ())
    return isinstance(trained_heads, (list, tuple)) and "ranker" in trained_heads


if __name__ == "__main__":
    raise SystemExit(main())
