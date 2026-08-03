#!/usr/bin/env python3
"""Generate repair-aware ranker replay from the exact HCFP tail."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from statistics import fmean, pstdev
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
from hcfp.replay import OFFICIAL_TARGET_KIND, records_from_learned_analysis, write_replay_v3  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--floorset-lite-root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--limit", type=int, default=32)
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

    def records():
        for sample, source in iter_floorset_lite_with_source(
            args.floorset_lite_root,
            limit=args.limit,
            seed=args.seed,
            score_aware=args.score_aware,
        ):
            device_sample = sample.case.to(device=device, dtype=None)
            analysis = analyze_case_with_checkpoint(device_sample, args.checkpoint, config)
            if not analysis.result.used_checkpoint or analysis.result.checkpoint_hash is None:
                raise RuntimeError(analysis.result.failure_reason or "checkpoint was not used")
            yield from records_from_learned_analysis(
                sample,
                source,
                analysis.result.checkpoint_hash,
                analysis,
                analytic_population=args.population,
                population_seed=args.flow_seed,
            )

    replay_records = list(records())
    count = write_replay_v3(replay_records, args.output)
    values = [float(value) for record in replay_records for value in record.target_score]
    sample_ids = list(dict.fromkeys(record.sample.sample_id for record in replay_records))
    sample_id_sha256 = hashlib.sha256("\n".join(sample_ids).encode()).hexdigest()
    stage_counts = {
        stage: sum(record.candidate_stage == stage for record in replay_records)
        for stage in sorted({str(record.candidate_stage) for record in replay_records})
    }
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
        },
        "target_distribution": {
            "count": len(values),
            "unique": len(set(values)),
            "minimum": min(values),
            "maximum": max(values),
            "mean": fmean(values),
            "standard_deviation": pstdev(values),
            "records_with_signal": sum(
                float(record.target_score.max() - record.target_score.min()) > 1.0e-8
                for record in replay_records
            ),
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


if __name__ == "__main__":
    raise SystemExit(main())
