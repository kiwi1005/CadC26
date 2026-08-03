#!/usr/bin/env python3
"""Attribute raw and post-BDP candidate quality with the pinned official evaluator."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Any

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from benchmark_hcfp import _case_ids, _load_evaluator, _provenance  # noqa: E402
from hcfp.analytic import AnalyticConfig, select_device, to_official_placements  # noqa: E402
from hcfp.benchmark import (  # noqa: E402
    candidate_oracles,
    candidate_source_layout,
    select_candidate_oracle,
    summarize_attribution_cases,
    summarize_candidate_types,
    uncapped_objective,
)
from hcfp.case import from_official  # noqa: E402
from hcfp.checkpoint import RUNTIME_NORMALIZATION, load_checkpoint  # noqa: E402
from hcfp.dynamics import DynamicsConfig  # noqa: E402
from hcfp.learned import (  # noqa: E402
    LearnedConfig,
    analyze_case_with_checkpoint,
    effective_collective_steps,
    effective_flow_steps,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-path", default="artifacts/floorset-v10")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--cases", default="all", help="all or comma-separated validation ids")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--population", type=int, default=8)
    parser.add_argument("--dynamics-steps", type=int, default=DynamicsConfig().steps)
    parser.add_argument(
        "--projection-steps",
        type=int,
        default=AnalyticConfig().projection_iterations,
    )
    parser.add_argument(
        "--direction-beam",
        type=int,
        default=AnalyticConfig().direction_beam,
    )
    parser.add_argument("--flow-steps", type=int, default=0)
    parser.add_argument("--collective-steps", type=_non_negative_int, default=0)
    parser.add_argument("--flow-seed", type=int, default=0)
    parser.add_argument("--topology-seeds", type=int, default=0)
    parser.add_argument(
        "--allow-missing-topology",
        action="store_true",
        help="diagnostic only: keep cases whose requested topology decode was rejected",
    )
    parser.add_argument("--tail-topk", type=int)
    args = parser.parse_args(argv)

    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(0)
    checkpoint = Path(args.checkpoint)
    model, checkpoint_metadata = load_checkpoint(
        checkpoint,
        expected_normalization=RUNTIME_NORMALIZATION,
        map_location="cpu",
    )
    checkpoint_hash = str(checkpoint_metadata["state_hash"])
    flow_steps = effective_flow_steps(args.flow_steps, checkpoint_metadata)
    collective_steps = effective_collective_steps(
        args.collective_steps,
        checkpoint_metadata,
        getattr(model, "config", checkpoint_metadata.get("config", {})),
    )
    device = select_device(args.device)
    analytic = AnalyticConfig(
        dynamics=DynamicsConfig(population=args.population, steps=args.dynamics_steps),
        projection_iterations=args.projection_steps,
        direction_beam=args.direction_beam,
    )
    config = LearnedConfig(
        analytic=analytic,
        flow_steps=flow_steps,
        collective_steps=collective_steps,
        tail_topk=args.tail_topk,
        seed=args.flow_seed,
        topology_seeds=args.topology_seeds,
    )

    data_path = Path(args.data_path)
    evaluator_module = _load_evaluator(data_path)
    evaluator = evaluator_module.ContestEvaluator(str(data_path), verbose=False)
    evaluator._load_dataset()
    requested = _case_ids(args.cases)
    test_ids = list(range(len(evaluator.dataset))) if requested is None else sorted(set(requested))
    cases = [
        _audit_case(
            evaluator_module,
            evaluator,
            test_id,
            checkpoint,
            checkpoint_hash,
            device,
            config,
            args.population,
            args.allow_missing_topology,
        )
        for test_id in test_ids
    ]
    large_cases = [case for case in cases if 106 <= int(case["block_count"]) <= 120]
    provenance = _provenance(data_path, str(device), "oracle_attribution")
    provenance.update(
        {
            "checkpoint": str(checkpoint),
            "checkpoint_hash": checkpoint_hash,
            "checkpoint_normalization": checkpoint_metadata["normalization"],
            "checkpoint_capabilities": checkpoint_metadata["capabilities"],
            "checkpoint_trained_heads": checkpoint_metadata["trained_heads"],
            "requested_collective_steps": args.collective_steps,
            "collective_steps": collective_steps,
        }
    )
    report = {
        "schema_version": 1,
        "provenance": provenance,
        "config": {
            "population": args.population,
            "dynamics_steps": args.dynamics_steps,
            "projection_steps": args.projection_steps,
            "direction_beam": args.direction_beam,
            "requested_flow_steps": args.flow_steps,
            "flow_steps": flow_steps,
            "requested_collective_steps": args.collective_steps,
            "collective_steps": collective_steps,
            "flow_seed": args.flow_seed,
            "topology_seeds": args.topology_seeds,
            "allow_missing_topology": args.allow_missing_topology,
            "tail_topk": args.tail_topk,
        },
        "cases": cases,
        "summary": {
            "all": _summary(cases),
            "106-120": _summary(large_cases),
        },
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(output)
    print(json.dumps(report["summary"], indent=2, sort_keys=True))
    return 0


def _non_negative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be non-negative")
    return parsed


def _audit_case(
    evaluator_module: Any,
    evaluator: Any,
    test_id: int,
    checkpoint: Path,
    checkpoint_hash: str,
    device: torch.device,
    config: LearnedConfig,
    population: int,
    allow_missing_topology: bool,
) -> dict[str, Any]:
    sample = evaluator.dataset[test_id]
    inputs, labels = sample["input"], sample["label"]
    area, b2b, p2b, pins, constraints = inputs
    block_count = int((area != -1).sum().item())
    baseline, target_positions = evaluator._extract_baseline(
        test_id,
        labels,
        b2b,
        p2b,
        pins,
        block_count,
    )
    optimizer_targets = _optimizer_targets(constraints, target_positions, block_count)
    source = {
        "block_count": block_count,
        "area_targets": area,
        "b2b_connectivity": b2b,
        "p2b_connectivity": p2b,
        "pins_pos": pins,
        "constraints": constraints,
        "target_positions": optimizer_targets,
    }
    case = from_official(
        block_count,
        area,
        b2b,
        p2b,
        pins,
        constraints,
        optimizer_targets,
        device=device,
    )
    analysis = analyze_case_with_checkpoint(case, checkpoint, config)
    if not analysis.result.used_checkpoint or analysis.result.checkpoint_hash != checkpoint_hash:
        raise RuntimeError(
            f"case {test_id}: {analysis.result.failure_reason or 'checkpoint was not used'}"
        )
    if (
        config.topology_seeds > 0
        and analysis.result.topology_seed_count != config.topology_seeds
        and not allow_missing_topology
    ):
        reason = analysis.analytic.incumbent_snapshot.get(
            "topology_seed_failure_reason",
            "topology decode produced an incomplete candidate set",
        )
        raise RuntimeError(
            f"case {test_id}: requested {config.topology_seeds} topology seeds, "
            f"produced {analysis.result.topology_seed_count}: {reason}"
        )

    learned_count = analysis.result.candidate_count - population
    sources = candidate_source_layout(population, learned_count)
    raw = analysis.analytic.raw_candidates
    projected = analysis.analytic.projected_candidates
    if len(sources) != raw.shape[0] or raw.shape != projected.shape:
        raise RuntimeError(
            f"case {test_id}: source layout {len(sources)} does not match "
            f"raw/projected candidates {tuple(raw.shape)}/{tuple(projected.shape)}"
        )
    snapshot = analysis.analytic.incumbent_snapshot
    topology_indices = _topology_indices(snapshot.get("topology_seed_sources"))
    metric_args = (baseline, constraints, b2b, p2b, pins, area, target_positions)
    raw_records = _candidate_records(
        evaluator_module,
        source,
        case,
        raw,
        sources,
        topology_indices,
        metric_args,
    )
    projected_records = _candidate_records(
        evaluator_module,
        source,
        case,
        projected,
        sources,
        topology_indices,
        metric_args,
    )
    incumbent_index, incumbent_source = _incumbent_source(
        snapshot.get("exact_source"),
        sources,
    )
    incumbent = _candidate_record(
        evaluator_module,
        source,
        case,
        analysis.result.selected,
        incumbent_index,
        incumbent_source,
        topology_indices,
        metric_args,
    )
    return {
        "test_id": test_id,
        "block_count": block_count,
        "baseline": {
            "hpwl": float(baseline["hpwl_baseline"]),
            "area": float(baseline["area_baseline"]),
        },
        "candidate_layout": {
            "population": population,
            "learned_count": learned_count,
            "topology_count": len(topology_indices) // 2,
            "candidate_count": len(sources),
        },
        "topology_provenance": {
            key: value
            for key, value in snapshot.items()
            if str(key).startswith("topology_")
        },
        "raw": {"candidates": raw_records, "oracles": _oracles(raw_records)},
        "post_bdp": {
            "candidates": projected_records,
            "oracles": _oracles(projected_records),
        },
        "incumbent": incumbent,
    }


def _optimizer_targets(constraints: torch.Tensor, targets: list[Any], block_count: int) -> torch.Tensor:
    result = torch.full((block_count, 4), -1.0)
    columns = constraints.shape[1] if constraints.dim() > 1 else 0
    for index in range(block_count):
        fixed = columns > 0 and constraints[index, 0] != 0
        preplaced = columns > 1 and constraints[index, 1] != 0
        if preplaced:
            result[index] = torch.as_tensor(targets[index], dtype=torch.float32)
        elif fixed:
            result[index, 2:4] = torch.as_tensor(targets[index][2:4], dtype=torch.float32)
    return result


def _candidate_records(
    evaluator_module: Any,
    source: dict[str, Any],
    case: Any,
    boxes: torch.Tensor,
    sources: tuple[str, ...],
    topology_indices: frozenset[int],
    metric_args: tuple[Any, ...],
) -> list[dict[str, Any]]:
    return [
        _candidate_record(
            evaluator_module,
            source,
            case,
            candidate,
            index,
            candidate_source,
            topology_indices,
            metric_args,
        )
        for index, (candidate_source, candidate) in enumerate(zip(sources, boxes))
    ]


def _candidate_record(
    evaluator_module: Any,
    source: dict[str, Any],
    case: Any,
    candidate: torch.Tensor,
    index: int,
    candidate_source: str,
    topology_indices: frozenset[int],
    metric_args: tuple[Any, ...],
) -> dict[str, Any]:
    positions = to_official_placements(source, case, candidate.detach().to(device="cpu"))
    metrics = evaluator_module.evaluate_solution(
        {"positions": positions, "runtime": 1.0},
        *metric_args,
        median_runtime=1.0,
    )
    objective = uncapped_objective(
        metrics.hpwl_gap,
        metrics.area_gap,
        metrics.violations_relative,
    )
    return {
        "candidate_index": int(index),
        "source": candidate_source,
        "candidate_type": (
            "topology"
            if index in topology_indices
            else "fallback"
            if index == 0
            else "learned_residual"
            if candidate_source.startswith("learned_")
            else "analytic"
        ),
        "hard_feasible": bool(metrics.is_feasible),
        "overlap_violations": int(metrics.overlap_violations),
        "area_violations": int(metrics.area_violations),
        "dimension_violations": int(metrics.dimension_violations),
        "fixed_violations": int(metrics.fixed_violations),
        "preplaced_violations": int(metrics.preplaced_violations),
        "hpwl_gap": float(metrics.hpwl_gap),
        "area_gap": float(metrics.area_gap),
        "boundary_violations": int(metrics.boundary_violations),
        "grouping_violations": int(metrics.grouping_violations),
        "mib_violations": int(metrics.mib_violations),
        "total_soft_violations": int(metrics.total_soft_violations),
        "max_possible_violations": int(metrics.max_possible_violations),
        "violations_relative": float(metrics.violations_relative),
        "official_capped_cost": float(metrics.cost),
        "uncapped_objective": objective,
    }


def _topology_indices(value: object) -> frozenset[int]:
    indices: set[int] = set()
    for source in value if isinstance(value, (list, tuple)) else ():
        text = str(source)
        if text.startswith("candidate_"):
            indices.add(int(text.removeprefix("candidate_")))
    return frozenset(indices)


def _oracles(candidates: list[dict[str, Any]]) -> dict[str, Any]:
    result = candidate_oracles(candidates)
    topology = [row for row in candidates if row["candidate_type"] == "topology"]
    residual = [row for row in candidates if row["candidate_type"] == "learned_residual"]
    result["topology"] = select_candidate_oracle(topology)
    result["learned_residual"] = select_candidate_oracle(residual)
    return result


def _summary(cases: list[dict[str, Any]]) -> dict[str, Any]:
    result = summarize_attribution_cases(cases)
    result["candidate_types"] = summarize_candidate_types(cases)
    return result


def _incumbent_source(exact_source: object, sources: tuple[str, ...]) -> tuple[int, str]:
    value = str(exact_source)
    if value == "fallback":
        return 0, "fallback"
    prefix = "candidate_"
    if not value.startswith(prefix):
        raise RuntimeError(f"unknown exact incumbent source: {value}")
    index = int(value.removeprefix(prefix))
    if not 0 <= index < len(sources):
        raise RuntimeError(f"exact incumbent index {index} is outside candidate layout")
    return index, sources[index]


if __name__ == "__main__":
    raise SystemExit(main())
