#!/usr/bin/env python3
"""Generate repair-aware ranker replay from the exact HCFP tail."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hcfp.analytic import AnalyticConfig, select_device  # noqa: E402
from hcfp.checkpoint import RUNTIME_NORMALIZATION, load_checkpoint  # noqa: E402
from hcfp.data import file_sha256  # noqa: E402
from hcfp.dynamics import DynamicsConfig  # noqa: E402
from hcfp.floorset_lite import iter_floorset_lite_with_source  # noqa: E402
from hcfp.learned import (  # noqa: E402
    LearnedConfig,
    analyze_case_with_checkpoint,
    effective_collective_steps,
    effective_flow_steps,
)
from hcfp.ranker_features import RANKER_FEATURE_DIM, RANKER_FEATURE_VERSION  # noqa: E402
from hcfp.replay import (  # noqa: E402
    OFFICIAL_TARGET_KIND,
    iter_replay,
    records_from_learned_analysis,
    write_replay_v3,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--floorset-lite-root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--limit", type=_positive_int, default=32)
    parser.add_argument("--population", type=int, default=8)
    parser.add_argument("--dynamics-steps", type=int, default=4)
    parser.add_argument("--projection-steps", type=int, default=8)
    parser.add_argument("--flow-steps", type=int, default=0)
    parser.add_argument("--collective-steps", type=_non_negative_int, default=0)
    parser.add_argument("--topology-seeds", type=_non_negative_int, default=0)
    parser.add_argument("--constraint-seeds", type=_non_negative_int, default=0)
    parser.add_argument("--flow-seed", type=int, default=0)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--score-aware", action="store_true")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--exclude-replay", action="append", default=[])
    parser.add_argument(
        "--record-stage",
        choices=("both", "initial", "post_relax"),
        default="both",
    )
    args = parser.parse_args(argv)
    if args.score_aware and args.seed is None:
        parser.error("--score-aware requires an explicit --seed")

    device = select_device(args.device)
    model, checkpoint_metadata = load_checkpoint(
        args.checkpoint,
        expected_normalization=RUNTIME_NORMALIZATION,
        map_location="cpu",
    )
    flow_steps = effective_flow_steps(args.flow_steps, checkpoint_metadata)
    collective_steps = effective_collective_steps(
        args.collective_steps,
        checkpoint_metadata,
        getattr(model, "config", checkpoint_metadata.get("config", {})),
    )
    analytic = AnalyticConfig(
        dynamics=DynamicsConfig(population=args.population, steps=args.dynamics_steps),
        projection_iterations=args.projection_steps,
        direction_beam=2,
    )
    config = LearnedConfig(
        analytic=analytic,
        flow_steps=flow_steps,
        collective_steps=collective_steps,
        seed=args.flow_seed,
        topology_seeds=args.topology_seeds,
        constraint_seeds=args.constraint_seeds,
    )
    excluded_ids = {
        record.sample.sample_id
        for replay_path in args.exclude_replay
        for record in iter_replay(replay_path)
    }
    exclusion_stats = {"skipped": 0, "accepted": 0}

    def records():
        for sample, source in iter_floorset_lite_with_source(
            args.floorset_lite_root,
            limit=None,
            seed=args.seed,
            score_aware=args.score_aware,
        ):
            if sample.sample_id in excluded_ids:
                exclusion_stats["skipped"] += 1
                continue
            device_sample = sample.case.to(device=device, dtype=None)
            analysis = analyze_case_with_checkpoint(device_sample, args.checkpoint, config)
            if not analysis.result.used_checkpoint or analysis.result.checkpoint_hash is None:
                raise RuntimeError(analysis.result.failure_reason or "checkpoint was not used")
            stage_records = records_from_learned_analysis(
                sample,
                source,
                analysis.result.checkpoint_hash,
                analysis,
                analytic_population=args.population,
                population_seed=args.flow_seed,
                stages=(
                    ("initial", "post_relax")
                    if args.record_stage == "both"
                    else (args.record_stage,)
                ),
            )
            yield from stage_records
            exclusion_stats["accepted"] += 1
            if exclusion_stats["accepted"] >= args.limit:
                return

    value_count = 0
    value_sum = 0.0
    value_square_sum = 0.0
    value_min = math.inf
    value_max = -math.inf
    unique_values: set[float] = set()
    records_with_signal = 0
    sample_ids: list[str] = []
    seen_samples: set[str] = set()
    stage_counts: dict[str, int] = {}

    def observed_records():
        nonlocal value_count, value_sum, value_square_sum, value_min, value_max
        nonlocal records_with_signal
        for record in records():
            values = [float(value) for value in record.target_score]
            value_count += len(values)
            value_sum += sum(values)
            value_square_sum += sum(value * value for value in values)
            value_min = min(value_min, *values)
            value_max = max(value_max, *values)
            unique_values.update(values)
            records_with_signal += float(record.target_score.max() - record.target_score.min()) > 1.0e-8
            sample_id = record.sample.sample_id
            if sample_id not in seen_samples:
                seen_samples.add(sample_id)
                sample_ids.append(sample_id)
            stage = str(record.candidate_stage)
            stage_counts[stage] = stage_counts.get(stage, 0) + 1
            yield record

    count = write_replay_v3(observed_records(), args.output)
    if not count or not value_count:
        raise ValueError("replay generation produced no records")
    sample_id_sha256 = hashlib.sha256("\n".join(sample_ids).encode()).hexdigest()
    value_mean = value_sum / value_count
    value_variance = max(0.0, value_square_sum / value_count - value_mean * value_mean)
    report = {
        "schema_version": 3,
        "records": count,
        "target_kind": OFFICIAL_TARGET_KIND,
        "stages": stage_counts,
        "mid_flow_state_recorded": False,
        "mid_flow_state_note": "not available in current LearnedAnalysis; replay records initial and post_relax only",
        "dataset": {
            "root": str(Path(args.floorset_lite_root).resolve()),
            "seed": args.seed,
            "score_aware": args.score_aware,
            "samples": len(sample_ids),
            "sample_id_sha256": sample_id_sha256,
            "exclusions": {
                "sample_count": len(excluded_ids),
                "skipped": exclusion_stats["skipped"],
                "replays": [
                    {
                        "path": str(Path(path).resolve()),
                        "sha256": file_sha256(path),
                    }
                    for path in args.exclude_replay
                ],
            },
        },
        "checkpoint": {
            "path": args.checkpoint,
            "sha256": file_sha256(args.checkpoint),
            "state_hash": checkpoint_metadata["state_hash"],
            "capabilities": checkpoint_metadata["capabilities"],
            "trained_heads": checkpoint_metadata["trained_heads"],
        },
        "candidate_config": {
            "population": args.population,
            "dynamics_steps": args.dynamics_steps,
            "projection_steps": args.projection_steps,
            "direction_beam": analytic.direction_beam,
            "requested_flow_steps": args.flow_steps,
            "flow_steps": flow_steps,
            "requested_collective_steps": args.collective_steps,
            "collective_steps": collective_steps,
            "topology_seeds": args.topology_seeds,
            "constraint_seeds": args.constraint_seeds,
            "flow_seed": args.flow_seed,
            "record_stage": args.record_stage,
        },
        "ranker_feature_views": {
            "stored": "stored_candidate_features_v1",
            "repair_aware": RANKER_FEATURE_VERSION,
            "repair_aware_dim": RANKER_FEATURE_DIM,
        },
        "target_distribution": {
            "count": value_count,
            "unique": len(unique_values),
            "minimum": value_min,
            "maximum": value_max,
            "mean": value_mean,
            "standard_deviation": math.sqrt(value_variance),
            "records_with_signal": records_with_signal,
        },
        "device": str(device),
        "output": args.output,
        "output_sha256": file_sha256(args.output),
    }
    Path(f"{args.output}.report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


def _non_negative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be non-negative")
    return parsed


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


if __name__ == "__main__":
    raise SystemExit(main())
