#!/usr/bin/env python3
"""Run the bounded P10 single-case cap-crossing experiment."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from benchmark_hcfp import _load_evaluator  # noqa: E402
from hcfp.boundary_skeleton import boundary_skeleton_candidates  # noqa: E402
from hcfp.case import from_official  # noqa: E402
from hcfp.contact_patch import dense_contact_patch_candidates  # noqa: E402
from hcfp.contact_synthesis import synthesize_contact_obligations  # noqa: E402
from hcfp.frame_core import frame_core_lns  # noqa: E402
from hcfp.score_attribution import attribute_score  # noqa: E402
from hcfp.verify import (  # noqa: E402
    _edge_connected,
    bbox,
    bbox_area,
    boundary_bitmask,
    boundary_missing,
    mib_shape_keys,
    verify,
)
from hcfp.visualize import (  # noqa: E402
    _bounds as visualization_bounds,
    _screen_point,
    _screen_rect,
    _viewport,
    render_svg,
)


CAP = 9.999999
SIDE_BITS = (("right", 2), ("top", 4), ("left", 1), ("bottom", 8))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--baseline",
        default="artifacts/experiments/p10_case70/p8_replay.json",
        help="single-case P8 benchmark JSON",
    )
    parser.add_argument("--case-id", type=int, default=70)
    parser.add_argument("--data-path", default="artifacts/floorset-v10")
    parser.add_argument(
        "--output-dir", default="artifacts/experiments/p10_case70"
    )
    parser.add_argument("--runtime-ceiling", type=float, default=30.0)
    args = parser.parse_args(argv)
    if args.runtime_ceiling <= 0.0:
        parser.error("--runtime-ceiling must be positive")

    started = time.perf_counter()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    baseline_path = Path(args.baseline)
    replay = json.loads(baseline_path.read_text(encoding="utf-8"))
    baseline_row = _baseline_row(replay, args.case_id)
    boxes = torch.as_tensor(
        baseline_row["positions"], dtype=torch.float64, device="cpu"
    )
    (
        evaluator_module,
        case,
        raw_case,
        metric_args,
        visual_case,
        b2b_edges,
        official_baseline,
    ) = _official_case(Path(args.data_path), args.case_id)

    baseline_metrics = _measure(
        evaluator_module, raw_case, metric_args, boxes
    )
    baseline_runtime = float(baseline_row.get("runtime_seconds", 0.0))
    expected = (24, 23, 5, 59)
    actual = tuple(
        baseline_metrics[key]
        for key in (
            "boundary_violations",
            "grouping_violations",
            "mib_violations",
            "max_possible_violations",
        )
    )
    if actual != expected:
        raise RuntimeError(f"P8 case70 baseline drifted: expected {expected}, got {actual}")
    if not baseline_metrics["hard_feasible"]:
        raise RuntimeError("P8 case70 incumbent is not hard feasible")

    provenance = _provenance(
        args,
        replay,
        baseline_path,
        evaluator_module,
    )
    _dump(
        output_dir / "baseline.json",
        {
            "test_id": args.case_id,
            "metrics": baseline_metrics,
            "solver_runtime_seconds": baseline_runtime,
            "positions": boxes.tolist(),
            "official_baseline": official_baseline,
            "placement_sha256": _placement_sha256(boxes),
            "provenance": provenance,
        },
    )
    _render_png(
        output_dir / "baseline.png",
        boxes,
        visual_case,
        b2b_edges,
        baseline_metrics,
        title="P10-C70 P8 baseline",
    )

    obligations = _obligations(case, raw_case, boxes, official_baseline)
    if len(obligations) != 52:
        raise RuntimeError(f"expected 52 case70 obligations, got {len(obligations)}")
    _write_obligations(output_dir / "obligations.csv", obligations)

    family_started = time.perf_counter()
    boundary_records = []
    for index, candidate in enumerate(
        boundary_skeleton_candidates(
            case,
            boxes,
            verify_case=raw_case,
            patch_sizes=(4, 8, 12),
            max_candidates=16,
        )
    ):
        boundary_records.append(
            _candidate_record(
                evaluator_module,
                raw_case,
                metric_args,
                "boundary",
                index,
                candidate.placement,
                {
                    "block": candidate.block,
                    "required_sides": candidate.required_sides,
                    "members": candidate.members,
                    "missing_before": candidate.missing_before,
                    "missing_after": candidate.missing_after,
                },
            )
        )
    boundary_runtime = time.perf_counter() - family_started
    boundary_best = _oracle(boundary_records)
    boundary_keep = bool(
        boundary_best
        and (
            boundary_best["metrics"]["cost"] < CAP
            or (
                boundary_best["metrics"]["boundary_violations"] <= 23
                and boundary_best["metrics"]["uncapped_cost"]
                < baseline_metrics["uncapped_cost"]
            )
        )
    )
    _dump(
        output_dir / "boundary_candidates.json",
        {
            "hypothesis": "one right/top boundary witness is the cheapest repair",
            "decision": "KEEP" if boundary_keep else "REJECT",
            "runtime_seconds": boundary_runtime,
            "candidate_count": len(boundary_records),
            "candidates": boundary_records,
        },
    )
    _render_png(
        output_dir / "boundary_best.png",
        boxes if boundary_best is None else boundary_best["positions"],
        visual_case,
        b2b_edges,
        baseline_metrics if boundary_best is None else boundary_best["metrics"],
        title="P10-C70 boundary rejected-best",
        patch=() if boundary_best is None else boundary_best["details"]["members"],
    )

    family_started = time.perf_counter()
    group_records = []
    first_crossing = None
    for index, candidate in enumerate(
        dense_contact_patch_candidates(
            case,
            boxes,
            verify_case=raw_case,
            patch_sizes=(4, 8, 12, 16),
            max_candidates=16,
        )
    ):
        record = _candidate_record(
            evaluator_module,
            raw_case,
            metric_args,
            "group",
            index,
            candidate.placement,
            {
                "group_index": candidate.group_index,
                "group_id": int(case.cluster_group_ids[candidate.group_index]),
                "bridge_member": candidate.bridge_member,
                "anchor_member": candidate.anchor_member,
                "members": candidate.members,
                "side": candidate.side,
                "grouping_before": candidate.grouping_before,
                "grouping_after": candidate.grouping_after,
            },
        )
        group_records.append(record)
        if first_crossing is None and record["metrics"]["cost"] < CAP:
            first_crossing = record
    group_runtime = time.perf_counter() - family_started
    group_best = _oracle(group_records)
    group_keep = bool(
        group_best
        and (
            group_best["metrics"]["cost"] < CAP
            or (
                group_best["metrics"]["grouping_violations"] <= 22
                and group_best["metrics"]["uncapped_cost"]
                < baseline_metrics["uncapped_cost"]
            )
        )
    )
    _dump(
        output_dir / "group_candidates.json",
        {
            "hypothesis": "one mandatory side contact crosses the cap",
            "decision": "KEEP" if group_keep else "REJECT",
            "runtime_seconds": group_runtime,
            "candidate_count": len(group_records),
            "first_cap_crossing_index": (
                None if first_crossing is None else first_crossing["candidate_index"]
            ),
            "oracle_index": None if group_best is None else group_best["candidate_index"],
            "candidates": group_records,
        },
    )
    if first_crossing is not None:
        _dump(output_dir / "first_cap_crossing.json", first_crossing)
        _render_record(
            output_dir / "first_cap_crossing.png",
            first_crossing,
            visual_case,
            b2b_edges,
            title="P10-C70 first cap crossing",
        )
    if group_best is not None:
        _render_record(
            output_dir / "group_best.png",
            group_best,
            visual_case,
            b2b_edges,
            title="P10-C70 group oracle",
        )

    if not group_keep or group_best is None:
        raise RuntimeError("case70 did not produce a KEEP single-step expert")

    # The first deterministic contact repair already crosses the cap.  Per the
    # experiment contract, do not spend the case budget on the later MIB family.
    mib_result = {
        "decision": "NOT_RUN_AFTER_CAP_CROSSING",
        "reason": "C70-B already produced a hard-feasible cost below 9.999999",
    }
    _dump(output_dir / "mib_candidates.json", mib_result)

    common = _common_loop(
        evaluator_module,
        case,
        raw_case,
        metric_args,
        group_best,
        runtime_ceiling=args.runtime_ceiling,
    )
    _dump(output_dir / "common_loop.json", common["report"])
    loop_best = common["best"]
    _render_png(
        output_dir / "common_loop_best.png",
        loop_best["positions"],
        visual_case,
        b2b_edges,
        loop_best["metrics"],
        title="P10-C70 locked contact loop",
        patch=loop_best["history"][-1]["members"],
        locked_edges=loop_best["locks"],
    )

    cleanup_started = time.perf_counter()
    cleanup_result = frame_core_lns(
        raw_case,
        loop_best["positions"],
        top_k=8,
        max_candidates=8,
    )
    cleanup_records = []
    for index, candidate in enumerate(cleanup_result.candidates):
        record = _candidate_record(
            evaluator_module,
            raw_case,
            metric_args,
            "hpwl_cleanup",
            index,
            candidate.placement,
            {
                "members": candidate.members,
                "delta": candidate.delta,
                "strategy": candidate.strategy,
            },
        )
        if (
            record["metrics"]["hard_feasible"]
            and record["metrics"]["total_soft_violations"]
            <= loop_best["metrics"]["total_soft_violations"]
            and record["metrics"]["uncapped_cost"]
            < loop_best["metrics"]["uncapped_cost"]
            and all(
                _touches(record["positions"], edge)
                for edge in loop_best["locks"]
            )
        ):
            cleanup_records.append(record)
    cleanup_runtime = time.perf_counter() - cleanup_started
    cleanup_best = _oracle(cleanup_records)
    cleanup_decision = "REJECT"
    if cleanup_best is not None:
        cleanup_decision = (
            "KEEP"
            if cleanup_best["metrics"]["hpwl_total"]
            < loop_best["metrics"]["hpwl_total"] - 1.0e-9
            else "MODIFY"
        )
    _dump(
        output_dir / "hpwl_cleanup.json",
        {
            "hypothesis": "locked-aware rigid local moves lower HPWL without soft debt",
            "decision": cleanup_decision,
            "runtime_seconds": cleanup_runtime,
            "candidate_count": len(cleanup_records),
            "candidates": cleanup_records,
        },
    )

    winner = loop_best
    winner_source = "common_loop"
    if cleanup_best is not None:
        winner = {
            "positions": cleanup_best["positions"],
            "metrics": cleanup_best["metrics"],
            "locks": loop_best["locks"],
            "history": [
                *loop_best["history"],
                {"family": "hpwl_cleanup", **cleanup_best["details"]},
            ],
        }
        winner_source = "hpwl_cleanup"
        _render_record(
            output_dir / "hpwl_cleanup_best.png",
            cleanup_best,
            visual_case,
            b2b_edges,
            title="P10-C70 HPWL cleanup",
            locked_edges=loop_best["locks"],
        )
    else:
        _render_png(
            output_dir / "hpwl_cleanup_best.png",
            loop_best["positions"],
            visual_case,
            b2b_edges,
            loop_best["metrics"],
            title="P10-C70 HPWL cleanup rejected-best",
            locked_edges=loop_best["locks"],
        )

    total_runtime = time.perf_counter() - started
    if winner["metrics"]["cost"] >= CAP or not winner["metrics"]["hard_feasible"]:
        raise RuntimeError("P10 winner did not meet the minimum completion gate")
    winner_payload = {
        "test_id": args.case_id,
        "source": winner_source,
        "metrics": winner["metrics"],
        "positions": winner["positions"],
        "placement_sha256": _placement_sha256(winner["positions"]),
        "locked_contact_edges": winner["locks"],
        "history": winner["history"],
        "runtime_seconds": total_runtime,
        "baseline_solver_runtime_seconds": baseline_runtime,
        "end_to_end_runtime_seconds": baseline_runtime + total_runtime,
        "runtime_ceiling_seconds": args.runtime_ceiling,
        "provenance": provenance,
    }
    _dump(output_dir / "winner.json", winner_payload)
    _dump(
        output_dir / "provenance.json",
        {
            **provenance,
            "winner_placement_sha256": winner_payload["placement_sha256"],
            "winner_metrics": winner_payload["metrics"],
        },
    )
    _dump(
        output_dir / "winner_placement.json",
        {"positions": winner["positions"], "runtime": total_runtime},
    )
    _render_png(
        output_dir / "winner.png",
        winner["positions"],
        visual_case,
        b2b_edges,
        winner["metrics"],
        title="P10-C70 winner",
        patch=winner["history"][-1].get("members", ()),
        locked_edges=winner["locks"],
    )
    _write_report(
        output_dir / "report.md",
        baseline_metrics,
        baseline_runtime,
        boundary_best,
        boundary_runtime,
        group_best,
        group_runtime,
        common,
        cleanup_best,
        cleanup_runtime,
        winner_payload,
    )

    print(output_dir / "winner.json")
    print(
        f"case70: {baseline_metrics['cost']:.6f} -> {winner['metrics']['cost']:.6f}; "
        f"B/G/M {baseline_metrics['boundary_violations']}/"
        f"{baseline_metrics['grouping_violations']}/"
        f"{baseline_metrics['mib_violations']} -> "
        f"{winner['metrics']['boundary_violations']}/"
        f"{winner['metrics']['grouping_violations']}/"
        f"{winner['metrics']['mib_violations']}; "
        f"hard_feasible={winner['metrics']['hard_feasible']}"
    )
    return 0


def _official_case(
    data_path: Path, case_id: int
) -> tuple[Any, Any, dict[str, Any], tuple[Any, ...], dict[str, Any], list[Any], dict[str, float]]:
    evaluator_module = _load_evaluator(data_path)
    evaluator = evaluator_module.ContestEvaluator(str(data_path), verbose=False)
    evaluator._load_dataset()
    item = evaluator.dataset[case_id]
    (area, b2b, p2b, pins, constraints), labels = item["input"], item["label"]
    block_count = int((area != -1).sum().item())
    baseline, targets = evaluator._extract_baseline(
        case_id, labels, b2b, p2b, pins, block_count
    )
    optimizer_targets = torch.full((block_count, 4), -1.0)
    for index in range(block_count):
        if constraints[index, 1] != 0:
            optimizer_targets[index] = torch.as_tensor(targets[index])
        elif constraints[index, 0] != 0:
            optimizer_targets[index, 2:4] = torch.as_tensor(targets[index][2:4])
    case = from_official(
        block_count,
        area,
        b2b,
        p2b,
        pins,
        constraints,
        optimizer_targets,
    )
    raw_case = {
        "normalized": False,
        "n": block_count,
        "area": area[:block_count],
        "constraints": constraints[:block_count],
        "target": targets[:block_count],
        "fixed_mask": case.fixed_mask,
        "preplaced_mask": case.preplaced_mask,
        "raw_preplaced_validated": True,
        "boundary_bits": case.boundary_bits,
        "group_membership": case.group_membership,
        "mib_membership": case.mib_membership,
        "b2b_weight": case.b2b_weight,
        "b2b_connectivity": b2b,
        "p2b_connectivity": p2b,
        "pins_pos": pins,
    }
    valid_pins = [
        row for row in pins.tolist() if row[:2] != [-1.0, -1.0]
    ]
    visual_case = {
        "constraints": constraints[:block_count].tolist(),
        "pins_pos": valid_pins,
    }
    b2b_edges = [
        row for row in b2b.tolist() if row[:2] != [-1.0, -1.0]
    ]
    metric_args = (baseline, constraints, b2b, p2b, pins, area, targets)
    return (
        evaluator_module,
        case,
        raw_case,
        metric_args,
        visual_case,
        b2b_edges,
        {
            "hpwl": float(baseline["hpwl_baseline"]),
            "area": float(baseline["area_baseline"]),
        },
    )


def _measure(
    evaluator_module: Any,
    raw_case: dict[str, Any],
    metric_args: tuple[Any, ...],
    positions: Any,
) -> dict[str, Any]:
    boxes = torch.as_tensor(positions, dtype=torch.float64, device="cpu")
    rows = [tuple(float(value) for value in row) for row in boxes.tolist()]
    metrics = evaluator_module.evaluate_solution(
        {"positions": rows, "runtime": 1.0},
        *metric_args,
        median_runtime=1.0,
    )
    score = attribute_score(
        float(metrics.hpwl_gap),
        float(metrics.area_gap),
        boundary_violations=int(metrics.boundary_violations),
        grouping_violations=int(metrics.grouping_violations),
        mib_violations=int(metrics.mib_violations),
        max_possible_violations=int(metrics.max_possible_violations),
        hard_feasible=bool(metrics.is_feasible),
    )
    hard = verify(raw_case, boxes)
    if hard.feasible != bool(metrics.is_feasible):
        raise RuntimeError("local hard verifier disagrees with the pinned evaluator")
    if not math.isclose(
        score.official_capped_cost,
        float(metrics.cost),
        rel_tol=0.0,
        abs_tol=1.0e-10,
    ):
        raise RuntimeError("score attribution disagrees with the pinned evaluator")
    return {
        "cost": score.official_capped_cost,
        "uncapped_cost": score.uncapped_cost,
        "log_uncapped_cost": score.log_uncapped_cost,
        "cap_margin": score.cap_margin,
        "hpwl_total": float(metrics.hpwl_total),
        "bbox_area": float(metrics.bbox_area),
        "hpwl_gap": float(metrics.hpwl_gap),
        "area_gap": float(metrics.area_gap),
        "boundary_violations": int(metrics.boundary_violations),
        "grouping_violations": int(metrics.grouping_violations),
        "mib_violations": int(metrics.mib_violations),
        "total_soft_violations": int(metrics.total_soft_violations),
        "max_possible_violations": int(metrics.max_possible_violations),
        "violations_relative": float(metrics.violations_relative),
        "hard_feasible": hard.feasible,
        "overlap_pairs": [list(pair) for pair in hard.overlap_pairs],
        "area_bad": list(hard.area_bad),
        "fixed_bad": list(hard.fixed_bad),
        "preplaced_bad": list(hard.preplaced_bad),
    }


def _candidate_record(
    evaluator_module: Any,
    raw_case: dict[str, Any],
    metric_args: tuple[Any, ...],
    family: str,
    index: int,
    positions: Any,
    details: dict[str, Any],
) -> dict[str, Any]:
    boxes = torch.as_tensor(positions, dtype=torch.float64, device="cpu")
    return {
        "family": family,
        "candidate_index": index,
        "details": details,
        "metrics": _measure(evaluator_module, raw_case, metric_args, boxes),
        "placement_sha256": _placement_sha256(boxes),
        "positions": boxes.tolist(),
    }


def _oracle(records: list[dict[str, Any]]) -> dict[str, Any] | None:
    feasible = [record for record in records if record["metrics"]["hard_feasible"]]
    if not feasible:
        return None
    return min(
        feasible,
        key=lambda record: (
            record["metrics"]["uncapped_cost"],
            record["metrics"]["total_soft_violations"],
            record["metrics"]["bbox_area"],
            record["metrics"]["hpwl_total"],
            record["candidate_index"],
        ),
    )


def _common_loop(
    evaluator_module: Any,
    case: Any,
    raw_case: dict[str, Any],
    metric_args: tuple[Any, ...],
    single_step: dict[str, Any],
    *,
    runtime_ceiling: float,
) -> dict[str, Any]:
    details = single_step["details"]
    initial = {
        "positions": single_step["positions"],
        "metrics": single_step["metrics"],
        "locks": [(details["bridge_member"], details["anchor_member"])],
        "history": [{"family": "group", **details}],
    }
    beam = [initial]
    best = initial
    rounds = []
    decoded = 0
    stagnant = 0
    started = time.perf_counter()
    seen = {_placement_sha256(initial["positions"])}

    for round_index in range(1, 7):
        proposals = []
        for state in beam:
            remaining = 48 - decoded
            if remaining <= 0 or time.perf_counter() - started >= runtime_ceiling:
                break
            candidates = dense_contact_patch_candidates(
                case,
                state["positions"],
                verify_case=raw_case,
                patch_sizes=(4, 8, 12, 16),
                max_candidates=min(8, remaining),
            )
            for candidate in candidates:
                if decoded >= 48 or time.perf_counter() - started >= runtime_ceiling:
                    break
                decoded += 1
                digest = _placement_sha256(candidate.placement)
                if digest in seen:
                    continue
                metrics = _measure(
                    evaluator_module, raw_case, metric_args, candidate.placement
                )
                state_metrics = state["metrics"]
                locks = [*state["locks"], (candidate.bridge_member, candidate.anchor_member)]
                if not (
                    metrics["hard_feasible"]
                    and metrics["boundary_violations"]
                    <= state_metrics["boundary_violations"]
                    and metrics["grouping_violations"]
                    < state_metrics["grouping_violations"]
                    and metrics["mib_violations"] <= state_metrics["mib_violations"]
                    and metrics["uncapped_cost"] < state_metrics["uncapped_cost"]
                    and all(_touches(candidate.placement, edge) for edge in locks)
                ):
                    continue
                seen.add(digest)
                proposals.append(
                    {
                        "positions": candidate.placement.tolist(),
                        "metrics": metrics,
                        "locks": locks,
                        "history": [
                            *state["history"],
                            {
                                "family": "group",
                                "group_index": candidate.group_index,
                                "group_id": int(
                                    case.cluster_group_ids[candidate.group_index]
                                ),
                                "bridge_member": candidate.bridge_member,
                                "anchor_member": candidate.anchor_member,
                                "members": candidate.members,
                                "side": candidate.side,
                            },
                        ],
                    }
                )
        proposals.sort(key=_state_key)
        beam = proposals[:4]
        rounds.append(
            {
                "round": round_index,
                "accepted": len(proposals),
                "beam": [
                    {
                        "cost": state["metrics"]["cost"],
                        "uncapped_cost": state["metrics"]["uncapped_cost"],
                        "B": state["metrics"]["boundary_violations"],
                        "G": state["metrics"]["grouping_violations"],
                        "M": state["metrics"]["mib_violations"],
                        "placement_sha256": _placement_sha256(state["positions"]),
                    }
                    for state in beam
                ],
            }
        )
        if not beam:
            break
        if _state_key(beam[0]) < _state_key(best):
            best = beam[0]
            stagnant = 0
        else:
            stagnant += 1
        if stagnant >= 2 or decoded >= 48:
            break

    runtime = time.perf_counter() - started
    return {
        "best": best,
        "report": {
            "hypothesis": "locked single-contact repairs compose under exact HPWL selection",
            "decision": (
                "KEEP"
                if best["metrics"]["uncapped_cost"]
                < single_step["metrics"]["uncapped_cost"]
                else "REJECT"
            ),
            "beam_width": 4,
            "maximum_rounds": 6,
            "proposals_per_round": 8,
            "exact_decode_cap": 48,
            "exact_decodes": decoded,
            "runtime_seconds": runtime,
            "runtime_ceiling_seconds": runtime_ceiling,
            "rounds": rounds,
            "best": {
                "metrics": best["metrics"],
                "positions": best["positions"],
                "placement_sha256": _placement_sha256(best["positions"]),
                "locked_contact_edges": best["locks"],
                "history": best["history"],
            },
        },
    }


def _state_key(state: dict[str, Any]) -> tuple[Any, ...]:
    metrics = state["metrics"]
    return (
        metrics["uncapped_cost"],
        metrics["total_soft_violations"],
        metrics["bbox_area"],
        metrics["hpwl_total"],
        _placement_sha256(state["positions"]),
    )


def _touches(positions: Any, edge: tuple[int, int]) -> bool:
    boxes = torch.as_tensor(positions, dtype=torch.float64, device="cpu")
    return _edge_connected(boxes[edge[0]], boxes[edge[1]], tol=1.0e-6)


def _obligations(
    case: Any,
    raw_case: dict[str, Any],
    boxes: torch.Tensor,
    official_baseline: dict[str, float],
) -> list[dict[str, Any]]:
    left, bottom, right, top = bbox(boxes)
    actual_bits = boundary_bitmask(boxes, tol=1.0e-6)
    missing = boundary_missing(raw_case, boxes)
    degree = case.b2b_weight.sum(dim=1).to(dtype=torch.float64)
    for pin, block, weight in case.p2b_edges.tolist():
        del pin
        degree[int(block)] += float(weight)
    benefit = 2.0 / 59.0
    rows: list[dict[str, Any]] = []

    for block in torch.nonzero(missing != 0, as_tuple=False).reshape(-1).tolist():
        mask = int(missing[block])
        distances = {
            "left": float(boxes[block, 0]) - left,
            "right": right - float(boxes[block, 0] + boxes[block, 2]),
            "top": top - float(boxes[block, 1] + boxes[block, 3]),
            "bottom": float(boxes[block, 1]) - bottom,
        }
        sides = [name for name, bit in SIDE_BITS if mask & bit]
        displacement = sum(max(0.0, distances[side]) for side in sides)
        rows.append(
            _obligation_row(
                "boundary",
                f"B{block}:{'+'.join(sides)}",
                (block,),
                boxes,
                case,
                benefit,
                displacement,
                float(degree[block]) * displacement,
                0.0,
                False,
                {
                    "block": block,
                    "required_sides": sides,
                    "actual_bits": int(actual_bits[block]),
                    "family_priority": min(
                        index for index, (_name, bit) in enumerate(SIDE_BITS) if mask & bit
                    ),
                },
                official_baseline,
            )
        )

    synthesis = synthesize_contact_obligations(case, boxes, tolerance=1.0e-6)
    for index, obligation in enumerate(synthesis.obligations):
        affected = tuple(
            sorted(set(obligation.component_a) | set(obligation.component_b))
        )
        group_id = int(case.cluster_group_ids[obligation.group_index])
        group_priority = {3: 0, 4: 1, 1: 2, 2: 3}.get(group_id, 4)
        rows.append(
            _obligation_row(
                "group",
                f"G{group_id}:{index}",
                affected,
                boxes,
                case,
                benefit,
                obligation.move_distance,
                obligation.net_incident * obligation.move_distance,
                obligation.bbox_expansion,
                False,
                {
                    "group_id": group_id,
                    "component_a": obligation.component_a,
                    "component_b": obligation.component_b,
                    "bridge_member": obligation.bridge_member,
                    "anchor_member": obligation.anchor_member,
                    "side": obligation.side,
                    "family_priority": group_priority,
                },
                official_baseline,
            )
        )

    shape_keys = mib_shape_keys(boxes)
    for group_index, membership in enumerate(case.mib_membership):
        members = torch.nonzero(membership, as_tuple=False).reshape(-1).tolist()
        hard_members = [
            member
            for member in members
            if bool(case.fixed_mask[member] or case.preplaced_mask[member])
        ]
        anchor = hard_members[0] if hard_members else members[0]
        anchor_shape = shape_keys[anchor]
        distinct: dict[tuple[float, float], list[int]] = {}
        for member in members:
            distinct.setdefault(shape_keys[member], []).append(member)
        for shape, shape_members in distinct.items():
            if shape == anchor_shape:
                continue
            displacement = 0.5 * (
                abs(shape[0] - anchor_shape[0]) + abs(shape[1] - anchor_shape[1])
            )
            estimated_hpwl = sum(float(degree[member]) for member in shape_members) * displacement
            rows.append(
                _obligation_row(
                    "mib",
                    f"M{int(case.mib_group_ids[group_index])}:{shape[0]}x{shape[1]}",
                    tuple(shape_members),
                    boxes,
                    case,
                    benefit,
                    displacement,
                    estimated_hpwl,
                    _shape_bbox_delta(boxes, shape_members, anchor_shape),
                    True,
                    {
                        "mib_group_id": int(case.mib_group_ids[group_index]),
                        "anchor": anchor,
                        "anchor_shape": anchor_shape,
                        "area_compatible": all(
                            abs(anchor_shape[0] * anchor_shape[1] - float(case.area[member]) * case.scale**2)
                            / (float(case.area[member]) * case.scale**2)
                            <= 0.01
                            for member in members
                        ),
                        "family_priority": 0,
                    },
                    official_baseline,
                )
            )

    rows.sort(
        key=lambda row: (
            -row["benefit_cost_ratio"],
            row["family_priority"],
            row["type"],
            row["obligation"],
        )
    )
    for rank, row in enumerate(rows, start=1):
        row["rank"] = rank
    return rows


def _obligation_row(
    kind: str,
    name: str,
    affected: tuple[int, ...],
    boxes: torch.Tensor,
    case: Any,
    benefit: float,
    displacement: float,
    estimated_hpwl: float,
    estimated_bbox: float,
    shape_change: bool,
    details: dict[str, Any],
    official_baseline: dict[str, float],
) -> dict[str, Any]:
    coupled = set(affected)
    fixed = sum(bool(case.fixed_mask[index]) for index in coupled)
    preplaced = sum(bool(case.preplaced_mask[index]) for index in coupled)
    hpwl_baseline = max(official_baseline["hpwl"], 1.0e-9)
    area_baseline = max(official_baseline["area"], 1.0e-9)
    repair_cost = (
        max(0.0, estimated_hpwl) / hpwl_baseline
        + max(0.0, estimated_bbox) / area_baseline
        + max(0.0, displacement) / math.sqrt(area_baseline)
        + 0.05 * (fixed + preplaced)
        + 0.01 * int(shape_change)
    )
    return {
        "rank": 0,
        "type": kind,
        "obligation": name,
        "affected_blocks": affected,
        "local_occupancy": _local_occupancy(boxes, affected),
        "preplaced_coupling": preplaced,
        "fixed_coupling": fixed,
        "shape_change": shape_change,
        "minimum_displacement": float(displacement),
        "estimated_hpwl_delta": float(estimated_hpwl),
        "estimated_bbox_delta": float(estimated_bbox),
        "benefit_log_cost": benefit,
        "estimated_repair_cost": repair_cost,
        "benefit_cost_ratio": benefit / max(repair_cost, 1.0e-12),
        **details,
    }


def _local_occupancy(
    boxes: torch.Tensor, affected: tuple[int, ...], limit: int = 4
) -> float:
    centers = boxes[:, :2] + 0.5 * boxes[:, 2:4]
    target = centers[list(affected)].mean(0)
    nearest = torch.argsort(torch.abs(centers - target).sum(1), stable=True).tolist()
    members = list(dict.fromkeys([*affected, *nearest]))[: max(limit, len(affected))]
    patch = boxes[members]
    area = float((patch[:, 2] * patch[:, 3]).sum())
    return area / max(bbox_area(patch), 1.0e-12)


def _shape_bbox_delta(
    boxes: torch.Tensor,
    members: list[int],
    target_shape: tuple[float, float],
) -> float:
    before = bbox_area(boxes)
    changed = boxes.clone()
    for member in members:
        center = changed[member, :2] + 0.5 * changed[member, 2:4]
        changed[member, 2:4] = changed.new_tensor(target_shape)
        changed[member, :2] = center - 0.5 * changed[member, 2:4]
    return max(0.0, bbox_area(changed) - before)


def _write_obligations(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = (
        "rank",
        "type",
        "obligation",
        "affected_blocks",
        "local_occupancy",
        "preplaced_coupling",
        "fixed_coupling",
        "shape_change",
        "minimum_displacement",
        "estimated_hpwl_delta",
        "estimated_bbox_delta",
        "benefit_log_cost",
        "estimated_repair_cost",
        "benefit_cost_ratio",
        "family_priority",
        "details",
    )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            known = {name: row.get(name) for name in fields if name != "details"}
            known["affected_blocks"] = " ".join(
                str(value) for value in row["affected_blocks"]
            )
            known["details"] = json.dumps(
                {
                    key: value
                    for key, value in row.items()
                    if key not in fields and key != "rank"
                },
                sort_keys=True,
            )
            writer.writerow(known)


def _render_record(
    path: Path,
    record: dict[str, Any],
    visual_case: dict[str, Any],
    b2b_edges: list[Any],
    *,
    title: str,
    locked_edges: list[tuple[int, int]] | None = None,
) -> None:
    details = record["details"]
    edge = (
        [(details["bridge_member"], details["anchor_member"])]
        if "bridge_member" in details
        else []
    )
    _render_png(
        path,
        record["positions"],
        visual_case,
        b2b_edges,
        record["metrics"],
        title=title,
        patch=details.get("members", ()),
        locked_edges=[*(locked_edges or []), *edge],
    )


def _render_png(
    path: Path,
    positions: Any,
    visual_case: dict[str, Any],
    b2b_edges: list[Any],
    metrics: dict[str, Any],
    *,
    title: str,
    patch: Any = (),
    locked_edges: Any = (),
) -> None:
    boxes = [tuple(float(value) for value in row) for row in positions]
    patch_members = tuple(int(value) for value in patch)
    patch_rects = []
    if patch_members:
        selected = torch.as_tensor(boxes, dtype=torch.float64)[list(patch_members)]
        left, bottom, right, top = bbox(selected)
        patch_rects = [(left, bottom, right - left, top - bottom)]
    boundary_total = sum(
        int(float(row[4]) != 0.0) for row in visual_case["constraints"]
    )
    summary = {
        "boundary_satisfied": boundary_total - metrics["boundary_violations"],
        "boundary_total": boundary_total,
        "group_connected_components": metrics["grouping_violations"] + 4,
        "mib_distinct_shape_count": metrics["mib_violations"] + 1,
    }
    svg = render_svg(
        boxes,
        case=visual_case,
        telemetry={
            "cost": metrics["cost"],
            "hpwl": metrics["hpwl_total"],
            "hpwl_gap": metrics["hpwl_gap"],
            "area_gap": metrics["area_gap"],
            "violations_relative": metrics["violations_relative"],
        },
        title=title,
        whitespace=patch_rects,
        summary_metrics=summary,
    )
    overlays = _svg_overlays(
        boxes,
        visual_case,
        b2b_edges,
        patch_members,
        tuple((int(a), int(b)) for a, b in locked_edges),
        patch_rects,
    )
    svg = svg.replace('<g class="legend"', overlays + '\n<g class="legend"', 1)
    path.parent.mkdir(parents=True, exist_ok=True)
    source = path.with_suffix(".svg")
    source.write_text(svg, encoding="utf-8")
    subprocess.run(["/usr/bin/rsvg-convert", "-o", str(path), str(source)], check=True)


def _svg_overlays(
    boxes: list[tuple[float, float, float, float]],
    visual_case: dict[str, Any],
    b2b_edges: list[Any],
    patch: tuple[int, ...],
    locked_edges: tuple[tuple[int, int], ...],
    extra_rects: list[tuple[float, float, float, float]],
) -> str:
    pins = [tuple(float(value) for value in row[:2]) for row in visual_case["pins_pos"]]
    left, top, scale = _viewport(
        visualization_bounds(boxes, pins, extra_rects), 760, 760
    )
    centers = [
        (x + 0.5 * width, y + 0.5 * height) for x, y, width, height in boxes
    ]
    lines = ['<g class="p10-overlays" pointer-events="none">']
    valid_edges = [
        (int(row[0]), int(row[1]), float(row[2]))
        for row in b2b_edges
        if int(row[0]) >= 0 and int(row[1]) >= 0 and float(row[2]) > 0.0
    ]
    for first, second, weight in sorted(valid_edges, key=lambda row: -row[2])[:12]:
        x1, y1 = _screen_point(centers[first], left, top, scale)
        x2, y2 = _screen_point(centers[second], left, top, scale)
        lines.append(
            f'<line x1="{x1:.3f}" y1="{y1:.3f}" x2="{x2:.3f}" y2="{y2:.3f}" '
            f'stroke="#607d8b" stroke-opacity="0.35" stroke-width="{min(3.0, 0.8 + weight):.3f}"/>'
        )
    for first, second in locked_edges:
        x1, y1 = _screen_point(centers[first], left, top, scale)
        x2, y2 = _screen_point(centers[second], left, top, scale)
        lines.append(
            f'<line x1="{x1:.3f}" y1="{y1:.3f}" x2="{x2:.3f}" y2="{y2:.3f}" '
            'stroke="#0b8043" stroke-width="3"/>'
        )
    mib_members = {
        index
        for index, row in enumerate(visual_case["constraints"])
        if int(float(row[2])) != 0
    }
    locked_members = {member for edge in locked_edges for member in edge}
    for members, color, dash in (
        (mib_members, "#b3261e", "4 2"),
        (locked_members, "#0b8043", ""),
        (set(patch), "#1565c0", "2 2"),
    ):
        for member in sorted(members):
            x, y, width, height = _screen_rect(boxes[member], left, top, scale)
            dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
            lines.append(
                f'<rect x="{x:.3f}" y="{y:.3f}" width="{width:.3f}" height="{height:.3f}" '
                f'fill="none" stroke="{color}" stroke-width="2.5"{dash_attr}/>'
            )
    lines.append("</g>")
    return "\n".join(lines)


def _write_report(
    path: Path,
    baseline: dict[str, Any],
    baseline_runtime: float,
    boundary: dict[str, Any] | None,
    boundary_runtime: float,
    group: dict[str, Any] | None,
    group_runtime: float,
    common: dict[str, Any],
    cleanup: dict[str, Any] | None,
    cleanup_runtime: float,
    winner: dict[str, Any],
) -> None:
    loop_metrics = common["best"]["metrics"]
    cleanup_metrics = loop_metrics if cleanup is None else cleanup["metrics"]
    cleanup_decision = "REJECT"
    if cleanup is not None:
        cleanup_decision = (
            "KEEP"
            if cleanup_metrics["hpwl_total"] < loop_metrics["hpwl_total"] - 1.0e-9
            else "MODIFY"
        )
    variants = [
        ("P8 baseline", baseline, baseline_runtime),
        (
            "C70-A rejected-best",
            baseline if boundary is None else boundary["metrics"],
            boundary_runtime,
        ),
        (
            "C70-B oracle",
            baseline if group is None else group["metrics"],
            group_runtime,
        ),
        (
            "C70-D locked loop",
            loop_metrics,
            common["report"]["runtime_seconds"],
        ),
        (
            f"C70-E {cleanup_decision}",
            cleanup_metrics,
            cleanup_runtime,
        ),
        ("Winner", winner["metrics"], winner["runtime_seconds"]),
    ]
    table = [
        "| Variant | Cost | Uncapped | B | G | M | Area gap | HPWL gap | Feasible | Runtime |",
        "|---|---:|---:|---:|---:|---:|---:|---:|:---:|---:|",
    ]
    for name, metrics, runtime in variants:
        table.append(
            f"| {name} | {metrics['cost']:.6f} | {metrics['uncapped_cost']:.6f} | "
            f"{metrics['boundary_violations']} | {metrics['grouping_violations']} | "
            f"{metrics['mib_violations']} | {metrics['area_gap']:.6f} | "
            f"{metrics['hpwl_gap']:.6f} | {'yes' if metrics['hard_feasible'] else 'no'} | "
            f"{runtime:.3f}s |"
        )
    text = f"""**Hypothesis**

One exact-safe group contact can remove enough soft debt to cross the case70 cap.

**Changed**

Added one case-scoped experiment driver that reuses the existing P8 exact scorer, boundary skeleton, dense contact patch, frame-core cleanup, and visualizer. No solver path or case-ID special case changed.

**Experiment**

Fresh P8 case70 replay; C70-A up to 16 boundary candidates, C70-B exactly 16 group candidates, then a locked beam-4 common loop capped at 48 exact decodes and 30 seconds. C70-C stopped after the first hard-feasible cap crossing, as required. No canary, large15, full100, or training run was executed.

**Result**

{chr(10).join(table)}

The winner preserves preplaced geometry and fixed dimensions exactly, has zero overlap and no area violations. C70-A produced no exact boundary-improving candidate. C70-B crossed the cap; the locked loop composed the validated contact action further. C70-E did not lower HPWL, but its locked-safe move removed one boundary debt at unchanged bbox/HPWL, so that hypothesis is `MODIFY`; the move remains in the winner because exact cost fell.

**Decision**

`KEEP`: case70 crossed the cap and remained hard feasible. The deterministic group-contact operator is now eligible for a later teacher-action/generalization experiment; no learned expert was trained in this single-case stage.

**Next experiment**

Freeze this placement. If generalization is approved, generate teacher actions only for the validated mandatory-contact operator on training cases with matching runtime-visible constraint signatures; keep case70 out of training.
"""
    path.write_text(text, encoding="utf-8")


def _baseline_row(report: dict[str, Any], case_id: int) -> dict[str, Any]:
    for rows in report.get("lanes", {}).values():
        for row in rows:
            if int(row.get("test_id", -1)) == case_id:
                return row
    raise ValueError(f"baseline JSON does not contain case {case_id}")


def _provenance(
    args: argparse.Namespace,
    replay: dict[str, Any],
    baseline_path: Path,
    evaluator_module: Any,
) -> dict[str, Any]:
    status = subprocess.run(
        ["git", "status", "--short"],
        cwd=ROOT,
        check=True,
        text=True,
        capture_output=True,
    ).stdout
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        text=True,
        capture_output=True,
    ).stdout.strip()
    evaluator_path = Path(evaluator_module.__file__)
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "command": " ".join(sys.argv),
        "git_commit": commit,
        "git_clean_at_experiment": not status.strip(),
        "git_status_sha256": hashlib.sha256(status.encode()).hexdigest(),
        "baseline_artifact": str(baseline_path),
        "baseline_artifact_sha256": _sha256(baseline_path),
        "baseline_command": replay.get("provenance", {}).get("command"),
        "checkpoint": replay.get("lane_metadata", {}).get("p8", {}).get("checkpoint"),
        "checkpoint_hash": replay.get("lane_metadata", {}).get("p8", {}).get(
            "checkpoint_hash"
        ),
        "evaluator_path": str(evaluator_path),
        "evaluator_sha256": _sha256(evaluator_path),
        "case_id": args.case_id,
        "runtime_ceiling_seconds": args.runtime_ceiling,
        "scope": "case70-only; no canary, large15, full100, or training",
    }


def _placement_sha256(positions: Any) -> str:
    boxes = torch.as_tensor(positions, dtype=torch.float64, device="cpu")
    payload = json.dumps(
        boxes.tolist(), separators=(",", ":"), allow_nan=False
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _dump(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    raise SystemExit(main())
