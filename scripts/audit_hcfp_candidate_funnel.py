#!/usr/bin/env python3
"""Trace exact candidate quality through raw, BDP, repair, and selection stages."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import sys
from typing import Any

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from benchmark_hcfp import _case_ids, _load_evaluator, _provenance  # noqa: E402
from hcfp.analytic import AnalyticConfig, select_device, to_official_placements  # noqa: E402
from hcfp.benchmark import candidate_source_layout, uncapped_objective  # noqa: E402
from hcfp.case import from_official  # noqa: E402
from hcfp.checkpoint import RUNTIME_NORMALIZATION, load_checkpoint  # noqa: E402
from hcfp.dynamics import DynamicsConfig  # noqa: E402
from hcfp.geometry import centers_from_xywh, normalize_xywh  # noqa: E402
from hcfp.learned import (  # noqa: E402
    LearnedConfig,
    _post_tail_group_repair,
    _repair_constraint_candidate,
    analyze_case_with_checkpoint,
    effective_collective_steps,
    effective_flow_steps,
    select_official_from_analysis,
)
from hcfp.verify import bbox_area  # noqa: E402


CAP = 10.0
EPS = 1.0e-9


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-path", default="artifacts/floorset-v10")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--cases", default="88,89,93,98")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--population", type=int, default=8)
    parser.add_argument("--dynamics-steps", type=int, default=DynamicsConfig().steps)
    parser.add_argument("--projection-steps", type=int, default=AnalyticConfig().projection_iterations)
    parser.add_argument("--direction-beam", type=int, default=AnalyticConfig().direction_beam)
    parser.add_argument("--flow-steps", type=int, default=0)
    parser.add_argument("--collective-steps", type=int, default=0)
    parser.add_argument("--flow-seed", type=int, default=6501)
    parser.add_argument("--topology-seeds", type=int, default=16)
    parser.add_argument("--constraint-seeds", type=int, default=16)
    parser.add_argument("--treemap-seeds", type=int, default=1)
    parser.add_argument("--btree-seeds", type=int, default=0)
    parser.add_argument("--tail-topk", type=int)
    parser.add_argument("--max-repair-moves", type=int, default=12)
    args = parser.parse_args(argv)
    if args.constraint_seeds and not args.topology_seeds:
        parser.error("--constraint-seeds requires --topology-seeds")

    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(args.flow_seed)
    checkpoint = Path(args.checkpoint)
    model, metadata = load_checkpoint(
        checkpoint,
        expected_normalization=RUNTIME_NORMALIZATION,
        map_location="cpu",
    )
    flow_steps = effective_flow_steps(args.flow_steps, metadata)
    collective_steps = effective_collective_steps(
        args.collective_steps,
        metadata,
        getattr(model, "config", metadata.get("config", {})),
    )
    device = select_device(args.device)
    config = LearnedConfig(
        analytic=AnalyticConfig(
            dynamics=DynamicsConfig(population=args.population, steps=args.dynamics_steps),
            projection_iterations=args.projection_steps,
            direction_beam=args.direction_beam,
        ),
        flow_steps=flow_steps,
        collective_steps=collective_steps,
        tail_topk=args.tail_topk,
        seed=args.flow_seed,
        topology_seeds=args.topology_seeds,
        constraint_seeds=args.constraint_seeds,
        treemap_seeds=args.treemap_seeds,
        btree_seeds=args.btree_seeds,
    )

    data_path = Path(args.data_path)
    evaluator_module = _load_evaluator(data_path)
    evaluator = evaluator_module.ContestEvaluator(str(data_path), verbose=False)
    evaluator._load_dataset()
    requested = _case_ids(args.cases)
    case_ids = list(range(len(evaluator.dataset))) if requested is None else sorted(set(requested))
    cases = [
        _audit_case(
            evaluator_module,
            evaluator,
            test_id,
            checkpoint,
            device,
            config,
            args.population,
            args.max_repair_moves,
        )
        for test_id in case_ids
    ]
    report = {
        "schema_version": 1,
        "provenance": {
            **_provenance(data_path, str(device), "candidate_funnel"),
            "checkpoint": str(checkpoint),
            "checkpoint_hash": str(metadata["state_hash"]),
        },
        "config": {
            "population": args.population,
            "dynamics_steps": args.dynamics_steps,
            "projection_steps": args.projection_steps,
            "direction_beam": args.direction_beam,
            "flow_steps": flow_steps,
            "collective_steps": collective_steps,
            "flow_seed": args.flow_seed,
            "topology_seeds": args.topology_seeds,
            "constraint_seeds": args.constraint_seeds,
            "treemap_seeds": args.treemap_seeds,
            "btree_seeds": args.btree_seeds,
            "tail_topk": args.tail_topk,
            "max_repair_moves": args.max_repair_moves,
        },
        "cases": cases,
        "summary": _summary(cases),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(output)
    print(json.dumps(report["summary"], indent=2, sort_keys=True))
    return 0


def _audit_case(
    evaluator_module: Any,
    evaluator: Any,
    test_id: int,
    checkpoint: Path,
    device: torch.device,
    config: LearnedConfig,
    population: int,
    max_repair_moves: int,
) -> dict[str, Any]:
    sample = evaluator.dataset[test_id]
    inputs, labels = sample["input"], sample["label"]
    area, b2b, p2b, pins, constraints = inputs
    block_count = int((area != -1).sum().item())
    baseline, target_positions = evaluator._extract_baseline(
        test_id, labels, b2b, p2b, pins, block_count
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
    if not analysis.result.used_checkpoint:
        raise RuntimeError(f"case {test_id}: {analysis.result.failure_reason or 'checkpoint was not used'}")

    learned_count = analysis.result.candidate_count - population
    sources = candidate_source_layout(population, learned_count)
    raw = analysis.analytic.raw_candidates
    projected = analysis.analytic.projected_candidates
    if len(sources) != raw.shape[0] or raw.shape != projected.shape:
        raise RuntimeError(f"case {test_id}: candidate layout mismatch")
    snapshot = analysis.analytic.incumbent_snapshot
    catalog = _candidate_catalog(snapshot)
    metric_args = (baseline, constraints, b2b, p2b, pins, area, target_positions)
    total_area = float(area[:block_count].sum().item())
    telemetry = analysis.analytic.telemetry

    raw_rows = []
    projected_rows = []
    repaired_rows = []
    repaired_geometry: list[torch.Tensor] = []
    for index, (source_name, raw_box, projected_box) in enumerate(zip(sources, raw, projected, strict=True)):
        family, stage = _candidate_identity(index, source_name, catalog)
        common = {
            "candidate_index": index,
            "source": source_name,
            "family": family,
            "candidate_stage": stage,
        }
        raw_rows.append(
            {
                **common,
                **_measure(
                    evaluator_module,
                    source,
                    case,
                    raw_box,
                    metric_args,
                    total_area,
                ),
            }
        )
        projected_rows.append(
            {
                **common,
                **_measure(
                    evaluator_module,
                    source,
                    case,
                    projected_box,
                    metric_args,
                    total_area,
                ),
                "projection_displacement": float(telemetry.projection_displacement[index].detach().cpu()),
                "selector_key": [
                    float(telemetry.soft_violation[index].detach().cpu()),
                    float((telemetry.bbox_area[index] + 0.05 * telemetry.hpwl[index]).detach().cpu()),
                    index,
                ],
            }
        )
        repair_input = raw_box if family == "treemap" and stage == "initial" else projected_box
        placements = to_official_placements(source, case, repair_input.detach().cpu())
        placements = _repair_constraint_candidate(
            source,
            case,
            placements,
            snapshot,
            f"candidate_{index}",
        )
        placements = _post_tail_group_repair(
            source,
            case,
            placements,
            max_moves=max_repair_moves,
        )
        repaired_box = normalize_xywh(
            case.to(device="cpu", dtype=torch.float32),
            torch.as_tensor(placements, dtype=torch.float32),
        )
        repaired_geometry.append(repaired_box)
        repaired_rows.append(
            {
                **common,
                **_measure_positions(
                    evaluator_module,
                    placements,
                    metric_args,
                    total_area,
                ),
                "repair_input_stage": "raw" if repair_input is raw_box else "post_bdp",
                "repair_displacement": float(
                    torch.linalg.vector_norm(
                        centers_from_xywh(repaired_box) - centers_from_xywh(repair_input.detach().cpu()),
                        dim=-1,
                    ).sum()
                ),
            }
        )

    selected_positions = select_official_from_analysis(
        source,
        case,
        analysis,
        config=config,
        device=device,
    )
    selected_box = normalize_xywh(
        case.to(device="cpu", dtype=torch.float32),
        torch.as_tensor(selected_positions, dtype=torch.float32),
    )
    selected_matches = [
        index
        for index, candidate in enumerate(repaired_geometry)
        if torch.allclose(candidate, selected_box, rtol=0.0, atol=1.0e-5)
    ]
    selected = {
        **_measure_positions(evaluator_module, selected_positions, metric_args, total_area),
        "snapshot_exact_source": str(snapshot.get("exact_source")),
        "candidate_funnel_proxy_source": snapshot.get("candidate_funnel_proxy_source"),
        "candidate_funnel_proxy_records": snapshot.get("candidate_funnel_proxy_records", ()),
        "matched_post_repair_indices": selected_matches,
    }
    oracle = {
        "raw": _oracle(raw_rows),
        "post_bdp": _oracle(projected_rows),
        "post_repair": _oracle(repaired_rows),
    }
    for row in repaired_rows:
        if row["candidate_index"] in selected_matches:
            row["selected"] = True
            row["rejection_reason"] = None
        elif not row["hard_feasible"]:
            row["selected"] = False
            row["rejection_reason"] = "hard_infeasible"
        elif float(row["uncapped_objective"]) + EPS < float(selected["uncapped_objective"]):
            row["selected"] = False
            row["rejection_reason"] = "selector_regret"
        else:
            row["selected"] = False
            row["rejection_reason"] = "higher_offline_cost"
    failure = _classify_case(oracle, selected)
    return {
        "test_id": test_id,
        "block_count": block_count,
        "candidate_count": len(sources),
        "family_counts": _family_counts(repaired_rows),
        "raw": {"candidates": raw_rows, "oracle": oracle["raw"]},
        "post_bdp": {"candidates": projected_rows, "oracle": oracle["post_bdp"]},
        "post_repair": {"candidates": repaired_rows, "oracle": oracle["post_repair"]},
        "selected": selected,
        "failure": failure,
    }


def _measure(
    evaluator_module: Any,
    source: dict[str, Any],
    case: Any,
    boxes: torch.Tensor,
    metric_args: tuple[Any, ...],
    total_area: float,
) -> dict[str, Any]:
    positions = to_official_placements(source, case, boxes.detach().cpu())
    return _measure_positions(evaluator_module, positions, metric_args, total_area)


def _measure_positions(
    evaluator_module: Any,
    positions: list[tuple[float, float, float, float]],
    metric_args: tuple[Any, ...],
    total_area: float,
) -> dict[str, Any]:
    metrics = evaluator_module.evaluate_solution(
        {"positions": positions, "runtime": 1.0},
        *metric_args,
        median_runtime=1.0,
    )
    objective = uncapped_objective(metrics.hpwl_gap, metrics.area_gap, metrics.violations_relative)
    layout_area = float(bbox_area(positions))
    return {
        "hard_feasible": bool(metrics.is_feasible),
        "overlap_violations": int(metrics.overlap_violations),
        "area_violations": int(metrics.area_violations),
        "dimension_violations": int(metrics.dimension_violations),
        "fixed_violations": int(metrics.fixed_violations),
        "preplaced_violations": int(metrics.preplaced_violations),
        "bbox_area": layout_area,
        "utilization": total_area / max(layout_area, 1.0e-12),
        "hpwl_gap": float(metrics.hpwl_gap),
        "area_gap": float(metrics.area_gap),
        "boundary_violations": int(metrics.boundary_violations),
        "grouping_violations": int(metrics.grouping_violations),
        "mib_violations": int(metrics.mib_violations),
        "violations_relative": float(metrics.violations_relative),
        "official_capped_cost": float(metrics.cost),
        "uncapped_objective": objective,
        "cap_margin": math.log(CAP) - math.log(objective) if metrics.is_feasible else None,
    }


def _candidate_catalog(snapshot: dict[str, object]) -> dict[int, dict[str, object]]:
    catalog: dict[int, dict[str, object]] = {}
    for name in ("topology_seed_provenance", "constraint_seed_provenance", "treemap_seed_provenance"):
        records = snapshot.get(name, ())
        for record in records if isinstance(records, (tuple, list)) else ():
            if not isinstance(record, dict):
                continue
            index = _candidate_index(record.get("source"))
            if index is not None:
                catalog[index] = record
    return catalog


def _candidate_identity(
    index: int,
    source: str,
    catalog: dict[int, dict[str, object]],
) -> tuple[str, str]:
    record = catalog.get(index)
    if record is not None:
        return str(record.get("candidate_type", "learned_residual")), str(record.get("stage", "unknown"))
    if index == 0:
        return "fallback", "safe"
    family = "learned_residual" if source.startswith("learned_") else "analytic"
    stage = "initial" if source.endswith("_initial") else "post_relax"
    return family, stage


def _candidate_index(source: object) -> int | None:
    text = str(source)
    if not text.startswith("candidate_"):
        return None
    try:
        return int(text.removeprefix("candidate_"))
    except ValueError:
        return None


def _oracle(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    feasible = [row for row in rows if bool(row["hard_feasible"])]
    if not feasible:
        return None
    winner = min(feasible, key=lambda row: (float(row["uncapped_objective"]), int(row["candidate_index"])))
    return {
        key: winner[key]
        for key in (
            "candidate_index",
            "source",
            "family",
            "candidate_stage",
            "official_capped_cost",
            "uncapped_objective",
            "cap_margin",
            "hpwl_gap",
            "area_gap",
            "boundary_violations",
            "grouping_violations",
            "mib_violations",
        )
    }


def _classify_case(oracles: dict[str, dict[str, Any] | None], selected: dict[str, Any]) -> dict[str, Any]:
    raw = oracles.get("raw")
    repaired = oracles.get("post_repair")
    raw_objective = float(raw["uncapped_objective"]) if raw is not None else math.inf
    repaired_objective = float(repaired["uncapped_objective"]) if repaired is not None else math.inf
    selected_objective = (
        float(selected["uncapped_objective"])
        if bool(selected.get("hard_feasible"))
        else math.inf
    )
    generation_gap = raw_objective >= CAP
    repair_gap = raw_objective < CAP and repaired_objective >= CAP
    repair_regret = max(0.0, repaired_objective - raw_objective) if math.isfinite(raw_objective) else None
    selection_gap = repaired_objective + EPS < selected_objective
    if selection_gap:
        primary = "selection"
    elif repair_gap:
        primary = "repair"
    elif generation_gap:
        primary = "generation"
    else:
        primary = "none"
    return {
        "primary": primary,
        "generation_gap": generation_gap,
        "repair_gap": repair_gap,
        "selection_gap": selection_gap,
        "repair_regret": repair_regret,
        "selection_regret": max(0.0, selected_objective - repaired_objective)
        if math.isfinite(repaired_objective)
        else None,
    }


def _family_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    result: dict[str, int] = {}
    for row in rows:
        family = str(row["family"])
        result[family] = result.get(family, 0) + 1
    return result


def _summary(cases: list[dict[str, Any]]) -> dict[str, Any]:
    classes: dict[str, int] = {}
    for case in cases:
        primary = str(case["failure"]["primary"])
        classes[primary] = classes.get(primary, 0) + 1
    return {
        "case_count": len(cases),
        "failure_classes": classes,
        "raw_below_cap": sum(
            case["raw"]["oracle"] is not None
            and float(case["raw"]["oracle"]["uncapped_objective"]) < CAP
            for case in cases
        ),
        "post_repair_below_cap": sum(
            case["post_repair"]["oracle"] is not None
            and float(case["post_repair"]["oracle"]["uncapped_objective"]) < CAP
            for case in cases
        ),
        "selected_below_cap": sum(
            bool(case["selected"]["hard_feasible"])
            and float(case["selected"]["uncapped_objective"]) < CAP
            for case in cases
        ),
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


if __name__ == "__main__":
    raise SystemExit(main())
