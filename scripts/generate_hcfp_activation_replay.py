#!/usr/bin/env python3
"""Generate disjoint training-only replay for learned-tail activation."""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import sys
import time

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hcfp.activation import (  # noqa: E402
    ACTIVATION_FEATURE_VERSION,
    ACTIVATION_SCHEMA_VERSION,
    ActivationOutcome,
    ActivationRecord,
    activation_features,
    activation_outcome,
    write_activation_replay,
)
from hcfp.analytic import (  # noqa: E402
    AnalyticConfig,
    select_device,
    solve_case_from_population_with_telemetry,
    solve_case_with_telemetry,
)
from hcfp.candidates import candidate_features  # noqa: E402
from hcfp.checkpoint import RUNTIME_NORMALIZATION, load_checkpoint  # noqa: E402
from hcfp.collective_runtime import CollectiveForceController  # noqa: E402
from hcfp.data import DataSample, file_sha256  # noqa: E402
from hcfp.dynamics import DynamicsConfig  # noqa: E402
from hcfp.fallback import safe_shelf  # noqa: E402
from hcfp.floorset_lite import iter_floorset_lite_with_source  # noqa: E402
from hcfp.learned import (  # noqa: E402
    LearnedAnalysis,
    LearnedConfig,
    LearnedResult,
    _learned_population,
    _merge_tail_analyses,
    effective_collective_steps,
    effective_flow_steps,
    select_official_from_analysis,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--floorset-lite-root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-prefix", required=True)
    parser.add_argument("--train-count", type=int, default=64)
    parser.add_argument("--calibration-count", type=int, default=16)
    parser.add_argument("--heldout-count", type=int, default=16)
    parser.add_argument("--population", type=int, default=8)
    parser.add_argument("--dynamics-steps", type=int, default=12)
    parser.add_argument("--projection-steps", type=int, default=24)
    parser.add_argument("--direction-beam", type=int, default=4)
    parser.add_argument("--flow-steps", type=int, default=0)
    parser.add_argument("--collective-steps", type=_non_negative_int, default=0)
    parser.add_argument("--tail-topk", type=int, default=4)
    parser.add_argument("--flow-seed", type=int, default=0)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--score-aware", action="store_true")
    parser.add_argument("--layouts-per-file", type=int, default=16)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args(argv)
    counts = {
        "train": args.train_count,
        "calibration": args.calibration_count,
        "heldout": args.heldout_count,
    }
    if any(value <= 0 for value in counts.values()):
        parser.error("all activation split counts must be positive")
    if args.tail_topk <= 0 or args.tail_topk > args.population:
        parser.error("--tail-topk must be in [1, population]")
    if args.layouts_per_file <= 0:
        parser.error("--layouts-per-file must be positive")

    torch.use_deterministic_algorithms(True)
    torch.manual_seed(args.seed)
    device = select_device(args.device)
    model, metadata = load_checkpoint(
        args.checkpoint,
        expected_normalization=RUNTIME_NORMALIZATION,
        map_location="cpu",
    )
    model = model.to(device=device).eval()
    flow_steps = effective_flow_steps(args.flow_steps, metadata)
    collective_steps = effective_collective_steps(
        args.collective_steps,
        metadata,
        getattr(model, "config", metadata.get("config", {})),
    )
    analytic_config = AnalyticConfig(
        dynamics=DynamicsConfig(population=args.population, steps=args.dynamics_steps),
        projection_iterations=args.projection_steps,
        direction_beam=args.direction_beam,
    )
    config = LearnedConfig(
        analytic=analytic_config,
        flow_steps=flow_steps,
        collective_steps=collective_steps,
        tail_topk=args.tail_topk,
        seed=args.flow_seed,
    )
    config_payload = asdict(config)
    config_hash = hashlib.sha256(
        json.dumps(config_payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()

    total = sum(counts.values())
    split_records: dict[str, list[ActivationRecord]] = {name: [] for name in counts}
    source_stream = iter_floorset_lite_with_source(
        args.floorset_lite_root,
        limit=total,
        seed=args.seed,
        score_aware=args.score_aware,
        max_layouts_per_file=args.layouts_per_file,
    )
    for split_name, (sample, source) in zip(_interleaved_split_names(counts), source_stream):
        split_records[split_name].append(
            _record(
                sample,
                model,
                str(metadata["state_hash"]),
                config_hash,
                config,
                device,
                source,
            )
        )
    produced = sum(len(records) for records in split_records.values())
    if produced != total:
        raise RuntimeError(f"activation source produced {produced} records, expected {total}")
    sample_ids = [record.sample_id for records in split_records.values() for record in records]
    if len(sample_ids) != len(set(sample_ids)):
        raise RuntimeError("activation source produced duplicate sample IDs")

    prefix = Path(args.output_prefix)
    paths = {name: Path(f"{prefix}.{name}.jsonl") for name in counts}
    reports = {}
    for name, subset in split_records.items():
        write_activation_replay(subset, paths[name])
        reports[name] = {
            "path": str(paths[name]),
            "sha256": file_sha256(paths[name]),
            "records": len(subset),
            "positives": sum(record.tail_needed for record in subset),
            "learned_failures": sum(record.failure_reason is not None for record in subset),
            "block_count_buckets": _block_count_buckets(subset),
            "sample_id_sha256": hashlib.sha256(
                "\n".join(record.sample_id for record in subset).encode()
            ).hexdigest(),
        }

    report = {
        "schema_version": 1,
        "replay_schema_version": ACTIVATION_SCHEMA_VERSION,
        "feature_version": ACTIVATION_FEATURE_VERSION,
        "dataset": {
            "root": str(Path(args.floorset_lite_root).resolve()),
            "seed": args.seed,
            "score_aware": args.score_aware,
            "layouts_per_file": args.layouts_per_file,
        },
        "checkpoint": {
            "path": args.checkpoint,
            "sha256": file_sha256(args.checkpoint),
            "state_hash": metadata["state_hash"],
            "capabilities": metadata["capabilities"],
            "trained_heads": metadata["trained_heads"],
        },
        "requested_flow_steps": args.flow_steps,
        "requested_collective_steps": args.collective_steps,
        "collective_steps": collective_steps,
        "candidate_config": config_payload,
        "candidate_config_hash": config_hash,
        "device": str(device),
        "splits": reports,
    }
    report_path = Path(f"{prefix}.report.json")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


def _non_negative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be non-negative")
    return parsed


def _record(
    sample: DataSample,
    model,
    checkpoint_hash: str,
    config_hash: str,
    config: LearnedConfig,
    device: torch.device,
    source: dict[str, object],
) -> ActivationRecord:
    case = sample.case.to(device=device, dtype=torch.float32)
    torch.cuda.synchronize() if case.area.is_cuda else None
    start = time.perf_counter()
    learned_population = _learned_population(case, model, config, seed=int(config.seed or 0))
    torch.cuda.synchronize() if case.area.is_cuda else None
    population_seconds = time.perf_counter() - start

    torch.cuda.synchronize() if case.area.is_cuda else None
    start = time.perf_counter()
    analytic = solve_case_with_telemetry(case, config.analytic)
    torch.cuda.synchronize() if case.area.is_cuda else None
    analytic_tail_seconds = time.perf_counter() - start

    torch.cuda.synchronize() if case.area.is_cuda else None
    feature_start = time.perf_counter()
    rank_features = candidate_features(
        case,
        learned_population,
        safe_shelf(case).to(device=device, dtype=torch.float32),
    )
    with torch.inference_mode():
        embedding = model.encoder(case)
        rank_scores = model.ranker(embedding, len(learned_population), rank_features)
    features = activation_features(case, analytic, learned_population, rank_scores)
    torch.cuda.synchronize() if case.area.is_cuda else None
    feature_seconds = time.perf_counter() - feature_start

    analytic_wrapper = _standalone_wrapper(analytic)
    selector_start = time.perf_counter()
    analytic_placements = select_official_from_analysis(
        source,
        case,
        analytic_wrapper,
        config=config,
        device=device,
    )
    analytic_selector_seconds = time.perf_counter() - selector_start

    baseline_area = float(sample.labels.baseline_area)
    baseline_hpwl = float(sample.labels.baseline_hpwl)
    analytic_seconds = (
        population_seconds
        + analytic_tail_seconds
        + feature_seconds
        + analytic_selector_seconds
    )
    analytic_outcome = activation_outcome(
        source,
        analytic_placements,
        baseline_area=baseline_area,
        baseline_hpwl=baseline_hpwl,
        runtime_seconds=analytic_seconds,
    )
    failure_reason = None
    torch.cuda.synchronize() if case.area.is_cuda else None
    learned_branch_start = time.perf_counter()
    try:
        force_controller = None
        if config.collective_steps:
            device_type = "cuda" if case.area.is_cuda else "cpu"
            with torch.inference_mode(), torch.autocast(
                device_type=device_type,
                dtype=torch.bfloat16,
                enabled=model.config.compute_dtype == "bfloat16",
            ):
                static_embedding = model.encoder(case).float()
            force_controller = CollectiveForceController.from_guidance(
                model,
                static_embedding,
                None,
            )
        learned_config = replace(
            config.analytic,
            dynamics=replace(
                config.analytic.dynamics,
                population=int(learned_population.shape[0]),
                steps=(
                    config.collective_steps
                    if config.collective_steps
                    else config.analytic.dynamics.steps
                ),
            ),
        )
        learned = solve_case_from_population_with_telemetry(
            case,
            learned_population,
            learned_config,
            force_controller=force_controller,
        )
        merged = _merge_tail_analyses(case, analytic, learned)
        learned_wrapper = LearnedAnalysis(
            LearnedResult(
                merged.selected,
                True,
                checkpoint_hash,
                None,
                config.flow_steps,
                config.analytic.dynamics.population + int(learned_population.shape[0]),
                collective_steps=config.collective_steps,
                collective_used=bool(config.collective_steps),
                collective_calls=config.collective_steps,
            ),
            merged,
        )
        learned_placements = select_official_from_analysis(
            source,
            case,
            learned_wrapper,
            config=config,
            device=device,
        )
        torch.cuda.synchronize() if case.area.is_cuda else None
        full_seconds = analytic_seconds + time.perf_counter() - learned_branch_start
        learned_outcome = activation_outcome(
            source,
            learned_placements,
            baseline_area=baseline_area,
            baseline_hpwl=baseline_hpwl,
            runtime_seconds=full_seconds,
        )
    except Exception as exc:
        torch.cuda.synchronize() if case.area.is_cuda else None
        full_seconds = analytic_seconds + time.perf_counter() - learned_branch_start
        failure_reason = f"{type(exc).__name__}: {exc}"
        learned_outcome = ActivationOutcome(False, 0.0, 0.0, 0.0, 10.0, full_seconds)
    margin = learned_outcome.objective - analytic_outcome.objective
    return ActivationRecord(
        sample.sample_id,
        case.n,
        checkpoint_hash,
        config_hash,
        features,
        learned_outcome.feasible and margin < -1.0e-6,
        margin,
        analytic_outcome,
        learned_outcome,
        failure_reason,
    )


def _standalone_wrapper(analytic) -> LearnedAnalysis:
    snapshot = dict(analytic.incumbent_snapshot)
    snapshot.update(
        analytic_exact_source=snapshot.get("exact_source"),
        analytic_fast_source=snapshot.get("fast_source"),
    )
    guarded = replace(analytic, incumbent_snapshot=snapshot)
    return LearnedAnalysis(LearnedResult(analytic.selected, False, None, None), guarded)


def _interleaved_split_names(counts: dict[str, int]) -> list[str]:
    """Interleave exact split counts so source-file buckets cannot segregate them."""

    total = sum(counts.values())
    assigned = {name: 0 for name in counts}
    result = []
    for position in range(total):
        available = [name for name in counts if assigned[name] < counts[name]]
        selected = max(
            available,
            key=lambda name: ((position + 1) * counts[name] / total - assigned[name]),
        )
        assigned[selected] += 1
        result.append(selected)
    return result


def _block_count_buckets(records: list[ActivationRecord]) -> dict[str, int]:
    return {
        f"{lower}-{upper}": sum(lower <= record.block_count <= upper for record in records)
        for lower, upper in ((1, 32), (33, 64), (65, 96), (97, 105), (106, 120))
    }


if __name__ == "__main__":
    raise SystemExit(main())
