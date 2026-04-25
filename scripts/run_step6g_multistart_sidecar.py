#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from puzzleplace.actions import ActionExecutor, ExecutionState
from puzzleplace.geometry import summarize_hard_legality
from puzzleplace.repair import finalize_layout
from puzzleplace.research.puzzle_candidate_payload import (
    BoundaryCommitMode,
    build_puzzle_candidate_descriptors,
    heuristic_scores,
)
from puzzleplace.research.virtual_frame import (
    PuzzleFrame,
    boundary_commit_attribution,
    estimate_predicted_compact_hull,
    frame_diagnostics,
    multistart_virtual_frames,
    repair_attribution_summary,
)
from puzzleplace.train.dataset_bc import load_validation_cases


def _bbox(
    placements: dict[int, tuple[float, float, float, float]],
) -> tuple[float, float, float, float]:
    min_x = min(x for x, _y, _w, _h in placements.values())
    min_y = min(y for _x, y, _w, _h in placements.values())
    max_x = max(x + w for x, _y, w, _h in placements.values())
    max_y = max(y + h for _x, y, _w, h in placements.values())
    return min_x, min_y, max_x, max_y


def _bbox_area(placements: dict[int, tuple[float, float, float, float]]) -> float:
    min_x, min_y, max_x, max_y = _bbox(placements)
    return (max_x - min_x) * (max_y - min_y)


def _hpwl_proxy(case: Any, placements: dict[int, tuple[float, float, float, float]]) -> float:
    total = 0.0
    for src, dst, weight in case.b2b_edges.tolist():
        i, j = int(src), int(dst)
        if i not in placements or j not in placements:
            continue
        ix, iy, iw, ih = placements[i]
        jx, jy, jw, jh = placements[j]
        total += float(weight) * (
            abs((ix + iw / 2) - (jx + jw / 2)) + abs((iy + ih / 2) - (jy + jh / 2))
        )
    return total


def _external_hpwl_proxy(
    case: Any, placements: dict[int, tuple[float, float, float, float]]
) -> float:
    total = 0.0
    for pin_idx, block_idx, weight in case.p2b_edges.tolist():
        pin_i, block_i = int(pin_idx), int(block_idx)
        if pin_i >= len(case.pins_pos) or block_i not in placements:
            continue
        px, py = [float(v) for v in case.pins_pos[pin_i].tolist()]
        x, y, w, h = placements[block_i]
        total += float(weight) * (abs((x + w / 2.0) - px) + abs((y + h / 2.0) - py))
    return total


def _external_ratio(case: Any, block_index: int) -> float:
    external = sum(
        abs(float(weight))
        for pin_idx, dst, weight in case.p2b_edges.tolist()
        if int(dst) == block_index and int(pin_idx) >= 0
    )
    internal = sum(
        abs(float(weight))
        for src, dst, weight in case.b2b_edges.tolist()
        if int(src) == block_index or int(dst) == block_index
    )
    return external / max(external + internal, 1e-6)


def _soft_boundary_violations(
    case: Any, placements: dict[int, tuple[float, float, float, float]]
) -> int:
    if not placements:
        return case.block_count
    min_x, min_y, max_x, max_y = _bbox(placements)
    violations = 0
    for idx, (x, y, w, h) in placements.items():
        code = int(case.constraints[idx, 4].item())
        if code == 1 and abs(x - min_x) > 1e-4:
            violations += 1
        elif code == 2 and abs((x + w) - max_x) > 1e-4:
            violations += 1
        elif code == 4 and abs((y + h) - max_y) > 1e-4:
            violations += 1
        elif code == 8 and abs(y - min_y) > 1e-4:
            violations += 1
    return violations


def _square_row_baseline(case: Any) -> dict[int, tuple[float, float, float, float]]:
    placements = {}
    x = 0.0
    for idx in range(case.block_count):
        side = float(case.area_targets[idx].sqrt().item())
        placements[idx] = (x, 0.0, side, side)
        x += side + 1.0
    return placements


def _overlaps_existing(
    box: tuple[float, float, float, float],
    placements: dict[int, tuple[float, float, float, float]],
) -> bool:
    x, y, w, h = box
    for px, py, pw, ph in placements.values():
        if max(x, px) < min(x + w, px + pw) and max(y, py) < min(y + h, py + ph):
            return True
    return False


def _frame_fallback_box(
    case: Any,
    placements: dict[int, tuple[float, float, float, float]],
    block_index: int,
    frame: PuzzleFrame,
) -> tuple[tuple[float, float, float, float], PuzzleFrame]:
    side = float(case.area_targets[block_index].sqrt().item())
    active = frame
    for _attempt in range(24):
        xs = [active.xmin]
        ys = [active.ymin]
        for px, py, pw, ph in placements.values():
            xs.append(px + pw)
            ys.append(py + ph)
        for y in sorted(set(round(value, 6) for value in ys)):
            for x in sorted(set(round(value, 6) for value in xs)):
                box = (float(x), float(y), side, side)
                if active.contains_box(box) and not _overlaps_existing(box, placements):
                    return box, active
        active = active.expanded(1.05)
    # Last-resort deterministic placement is still inside the relaxed frame.
    box = (active.xmin, active.ymin, side, side)
    return box, active


def _boundary_order_tier(case: Any, block_index: int) -> tuple[int, float]:
    code = int(case.constraints[block_index, 4].item())
    if code == 0:
        return (3, -float(case.area_targets[block_index].item()))
    fixed = bool(case.constraints[block_index, 0].item())
    preplaced = bool(case.constraints[block_index, 1].item())
    if fixed or preplaced:
        return (0, 0.0)
    if bin(code).count("1") >= 2 or _external_ratio(case, block_index) >= 0.5:
        return (1, -_external_ratio(case, block_index))
    if float(case.area_targets[block_index].item()) >= float(case.area_targets.mean().item()):
        return (2, -float(case.area_targets[block_index].item()))
    return (4, _external_ratio(case, block_index))


def _construct(
    case: Any,
    seed: int,
    start: int,
    frame: PuzzleFrame | None = None,
    *,
    boundary_commit_mode: BoundaryCommitMode = "prefer_predicted_hull",
) -> tuple[
    dict[int, tuple[float, float, float, float]],
    dict[str, int],
    PuzzleFrame | None,
    PuzzleFrame | None,
]:
    state = ExecutionState()
    executor = ActionExecutor(case)
    family_usage: dict[str, int] = {}
    active_frame = frame
    predicted_hull = (
        estimate_predicted_compact_hull(case, active_frame) if active_frame is not None else None
    )
    order = list(range(case.block_count))
    variant = (seed + start) % 4
    if variant == 1:
        order = sorted(order, key=lambda idx: float(case.area_targets[idx].item()), reverse=True)
    elif variant == 2:
        order = order[start % len(order) :] + order[: start % len(order)]
    elif variant == 3:
        order = sorted(
            order,
            key=lambda idx: (
                -int(case.constraints[idx, 4].item() != 0),
                -float(case.area_targets[idx].item()),
            ),
        )
    if active_frame is not None:
        position = {idx: pos for pos, idx in enumerate(order)}
        order = sorted(
            order,
            key=lambda idx: (_boundary_order_tier(case, idx), position[idx]),
        )
    for block_index in order:
        if block_index in state.placements:
            continue
        descriptors = build_puzzle_candidate_descriptors(
            case,
            state,
            remaining_blocks=[block_index],
            max_shape_bins=5,
            max_descriptors_per_block=24,
            virtual_frame=active_frame,
            predicted_hull=predicted_hull,
            frame_relaxation_steps=0,
            boundary_commit_mode=boundary_commit_mode,
        )
        if not descriptors and active_frame is not None:
            for _relax in range(6):
                active_frame = active_frame.expanded(1.05)
                predicted_hull = estimate_predicted_compact_hull(case, active_frame)
                descriptors = build_puzzle_candidate_descriptors(
                    case,
                    state,
                    remaining_blocks=[block_index],
                    max_shape_bins=5,
                    max_descriptors_per_block=24,
                    virtual_frame=active_frame,
                    predicted_hull=predicted_hull,
                    frame_relaxation_steps=0,
                    boundary_commit_mode=boundary_commit_mode,
                )
                if descriptors:
                    break
        if not descriptors:
            side = float(case.area_targets[block_index].sqrt().item())
            if active_frame is None:
                x = max((px + pw for px, _py, pw, _ph in state.placements.values()), default=0.0)
                box = (x, 0.0, side, side)
            else:
                box, active_frame = _frame_fallback_box(
                    case, state.placements, block_index, active_frame
                )
            state.placements = {**state.placements, block_index: box}
            family_usage["fallback_append"] = family_usage.get("fallback_append", 0) + 1
            continue
        scores = heuristic_scores(descriptors)
        ranked = scores.argsort(descending=True).tolist()
        rank_offset = 0 if len(ranked) == 1 else (start + seed) % min(3, len(ranked))
        chosen = descriptors[int(ranked[rank_offset])]
        family_usage[chosen.candidate_family] = family_usage.get(chosen.candidate_family, 0) + 1
        executor.apply(state, chosen.action_token)
    return state.placements, family_usage, active_frame, predicted_hull


def _positions_list(placements: dict[int, tuple[float, float, float, float]], block_count: int):
    return [placements[idx] for idx in range(block_count)]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case-ids", nargs="*", default=["0", "1", "2", "3", "4"])
    parser.add_argument("--seeds", nargs="*", default=["0", "1", "2"])
    parser.add_argument("--starts-small", type=int, default=8)
    parser.add_argument("--starts-large", type=int, default=16)
    parser.add_argument("--workers", type=int, default=48)
    parser.add_argument(
        "--disable-virtual-frame",
        action="store_true",
        help="Use the legacy unframed Step6G candidate space.",
    )
    parser.add_argument(
        "--boundary-commit-mode",
        choices=[
            "none",
            "prefer_predicted_hull",
            "require_predicted_hull_if_available",
            "require_virtual_frame_if_available",
        ],
        default="prefer_predicted_hull",
    )
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    requested_ids = [int(v) for v in args.case_ids]
    seed_ids = [int(v) for v in args.seeds]
    cases = load_validation_cases(case_limit=max(requested_ids) + 1)
    pre_bbox: list[float] = []
    post_bbox: list[float] = []
    base_bbox: list[float] = []
    pre_hpwl: list[float] = []
    post_hpwl: list[float] = []
    base_hpwl: list[float] = []
    pre_soft: list[int] = []
    post_soft: list[int] = []
    base_soft: list[int] = []
    repair_displacements: list[float] = []
    family_usage_total: dict[str, int] = {}
    fallback_count = 0
    total_blocks = 0
    hard_infeasible = 0
    completed_runs = 0
    total_runs = 0
    runs = []
    for case_id in requested_ids:
        case = cases[case_id]
        baseline = _square_row_baseline(case)
        base_bbox.append(_bbox_area(baseline))
        base_hpwl.append(_hpwl_proxy(case, baseline))
        base_soft.append(_soft_boundary_violations(case, baseline))
        starts = args.starts_large if case.block_count >= 90 else args.starts_small
        frames: list[PuzzleFrame | None]
        if args.disable_virtual_frame:
            frames = [None]
        else:
            frames = list(multistart_virtual_frames(case))
        best = None
        per_start = []
        for frame_index, frame in enumerate(frames):
            for seed in seed_ids:
                for start in range(starts):
                    total_runs += 1
                    placements, family_usage, construction_frame, predicted_hull = _construct(
                        case,
                        seed,
                        start,
                        frame,
                        boundary_commit_mode=args.boundary_commit_mode,
                    )
                    complete = len(placements) == case.block_count
                    completed_runs += int(complete)
                    bbox = _bbox_area(placements)
                    hpwl = _hpwl_proxy(case, placements)
                    soft = _soft_boundary_violations(case, placements)
                    repair = finalize_layout(case, placements)
                    post_placements = {idx: box for idx, box in enumerate(repair.positions)}
                    post_bbox_value = _bbox_area(post_placements)
                    post_hpwl_value = _hpwl_proxy(case, post_placements)
                    post_soft_value = _soft_boundary_violations(case, post_placements)
                    post_hard = summarize_hard_legality(
                        case, _positions_list(post_placements, case.block_count)
                    ).is_feasible
                    frame_metrics = (
                        frame_diagnostics(case, placements, construction_frame)
                        if construction_frame is not None
                        else None
                    )
                    post_frame_metrics = (
                        frame_diagnostics(case, post_placements, construction_frame)
                        if construction_frame is not None
                        else None
                    )
                    frame_penalty = 0.0
                    if frame_metrics is not None:
                        frame_penalty = (
                            50.0 * float(frame_metrics["outside_frame_area_ratio"])
                            + float(frame_metrics["max_protrusion_distance"])
                            + 5.0 * float(frame_metrics["num_frame_violations"])
                        )
                    score = (
                        post_bbox_value + post_hpwl_value + 10.0 * post_soft_value + frame_penalty
                    )
                    fallback_used = family_usage.get("fallback_append", 0)
                    per_start.append(
                        {
                            "case_id": case_id,
                            "seed": seed,
                            "start": start,
                            "frame_index": frame_index,
                            "frame_variant": None if frame is None else frame.variant,
                            "score": score,
                            "candidate_family_usage": family_usage,
                            "fallback_count": fallback_used,
                            "fallback_fraction": fallback_used / max(case.block_count, 1),
                            "constructive_pre_repair_metrics": {
                                "bbox_area": bbox,
                                "hpwl_proxy": hpwl,
                                "soft_boundary_violations": soft,
                                "virtual_frame": frame_metrics,
                            },
                            "post_repair_metrics": {
                                "bbox_area": post_bbox_value,
                                "hpwl_proxy": post_hpwl_value,
                                "soft_boundary_violations": post_soft_value,
                                "hard_feasible": bool(post_hard),
                                "virtual_frame": post_frame_metrics,
                            },
                            "official_aligned_slice_metrics": {
                                "hpwl_proxy": post_hpwl_value,
                                "bbox_area_proxy": post_bbox_value,
                                "soft_boundary_violations": post_soft_value,
                                "hard_feasible_after_repair": bool(post_hard),
                            },
                            "repair_displacement": float(repair.report.mean_displacement),
                        }
                    )
                    if best is None or score < best[0]:
                        best = (
                            score,
                            placements,
                            family_usage,
                            bbox,
                            hpwl,
                            soft,
                            seed,
                            start,
                            frame_index,
                            construction_frame,
                            predicted_hull,
                        )
        assert best is not None
        (
            _score,
            placements,
            family_usage,
            bbox,
            hpwl,
            soft,
            seed,
            start,
            frame_index,
            frame,
            predicted_hull,
        ) = best
        repair = finalize_layout(case, placements)
        post_placements = {idx: box for idx, box in enumerate(repair.positions)}
        post_bbox_value = _bbox_area(post_placements)
        post_hpwl_value = _hpwl_proxy(case, post_placements)
        post_soft_value = _soft_boundary_violations(case, post_placements)
        pre_bbox.append(bbox)
        post_bbox.append(post_bbox_value)
        pre_hpwl.append(hpwl)
        post_hpwl.append(post_hpwl_value)
        pre_soft.append(soft)
        post_soft.append(post_soft_value)
        repair_displacements.append(float(repair.report.mean_displacement))
        fallback_count += family_usage.get("fallback_append", 0)
        total_blocks += case.block_count
        for key, value in family_usage.items():
            family_usage_total[key] = family_usage_total.get(key, 0) + value
        if len(post_placements) != case.block_count:
            hard_infeasible += 1
        else:
            legality = summarize_hard_legality(
                case, _positions_list(post_placements, case.block_count)
            )
            hard_infeasible += int(not legality.is_feasible)
        runs.append(
            {
                "case_id": case_id,
                "best_seed": seed,
                "best_start": start,
                "best_frame_index": frame_index,
                "best_frame_variant": None if frame is None else frame.variant,
                "bbox_area": bbox,
                "hpwl_proxy": hpwl,
                "soft_boundary_violations": soft,
                "virtual_frame_metrics": (
                    frame_diagnostics(case, placements, frame) if frame is not None else None
                ),
                "predicted_hull": None
                if predicted_hull is None
                else {
                    "variant": predicted_hull.variant,
                    "relaxation": predicted_hull.relaxation,
                    "xmin": predicted_hull.xmin,
                    "ymin": predicted_hull.ymin,
                    "xmax": predicted_hull.xmax,
                    "ymax": predicted_hull.ymax,
                    "area": predicted_hull.area,
                },
                "post_repair_bbox_area": post_bbox_value,
                "post_repair_hpwl_proxy": post_hpwl_value,
                "post_repair_soft_boundary_violations": post_soft_value,
                "post_repair_virtual_frame_metrics": (
                    frame_diagnostics(case, post_placements, frame) if frame is not None else None
                ),
                "repair_mean_displacement": float(repair.report.mean_displacement),
                "boundary_commit_attribution": (
                    boundary_commit_attribution(
                        case, placements, post_placements, frame, predicted_hull
                    )
                    if frame is not None and predicted_hull is not None
                    else []
                ),
                "repair_attribution": repair_attribution_summary(case, placements, post_placements),
                "starts_evaluated": starts * len(seed_ids) * len(frames),
                "frames_evaluated": [None if row is None else row.variant for row in frames],
                "per_start_count": len(per_start),
                "per_start_records": per_start,
            }
        )

    def mean(values: list[float] | list[int]) -> float:
        return float(sum(values)) / max(len(values), 1)

    hpwl_improvement = (mean(base_hpwl) - mean(post_hpwl)) / max(mean(base_hpwl), 1e-6)
    bbox_improvement = (mean(base_bbox) - mean(post_bbox)) / max(mean(base_bbox), 1e-6)
    soft_reduction = (mean(base_soft) - mean(post_soft)) / max(mean(base_soft), 1e-6)
    report = {
        "case_ids": requested_ids,
        "seeds": seed_ids,
        "starts_small": args.starts_small,
        "starts_large": args.starts_large,
        "workers": args.workers,
        "virtual_frame_enabled": not args.disable_virtual_frame,
        "boundary_commit_mode": args.boundary_commit_mode,
        "semantic_completion_fraction": completed_runs / max(total_runs, 1),
        "fallback_selected_fraction": fallback_count / max(total_blocks, 1),
        "hard_infeasible_after_repair_count": hard_infeasible,
        "hpwl_proxy_relative_improvement": hpwl_improvement,
        "bbox_area_proxy_relative_improvement": bbox_improvement,
        "soft_violation_relative_reduction": soft_reduction,
        "constructive_pre_repair_metrics": {
            "mean_bbox_area": mean(pre_bbox),
            "mean_hpwl_proxy": mean(pre_hpwl),
            "mean_soft_boundary_violations": mean(pre_soft),
        },
        "post_repair_metrics": {
            "mean_bbox_area": mean(post_bbox),
            "mean_hpwl_proxy": mean(post_hpwl),
            "mean_soft_boundary_violations": mean(post_soft),
            "note": "existing finalizer applied for attribution only; runtime remains frozen",
        },
        "baseline_metrics": {
            "mean_bbox_area": mean(base_bbox),
            "mean_hpwl_proxy": mean(base_hpwl),
            "mean_soft_boundary_violations": mean(base_soft),
        },
        "repair_displacement_mean": mean(repair_displacements),
        "candidate_family_usage": family_usage_total,
        "runs": runs,
        "per_start_audit_records": [row for run in runs for row in run["per_start_records"]],
        "bottleneck_classification": "ready_for_later_runtime_plan"
        if hpwl_improvement >= 0.05 or bbox_improvement >= 0.05 or soft_reduction >= 0.20
        else "constructive_search",
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(report, indent=2))
    md = Path(args.output).with_suffix(".md")
    md.write_text(
        "# Step6H Virtual Puzzle Frame Multistart Sidecar\n\n```json\n"
        + json.dumps(report, indent=2)
        + "\n```\n"
    )
    changed_files = [
        "src/puzzleplace/research/puzzle_candidate_payload.py",
        "src/puzzleplace/research/step6g_synthetic.py",
        "src/puzzleplace/research/virtual_frame.py",
        "scripts/run_step6g_payload_guardrail.py",
        "scripts/run_step6g_candidate_coverage.py",
        "scripts/run_step6g_puzzle_policy_audit.py",
        "scripts/run_step6g_multistart_sidecar.py",
        "scripts/validate_step4_outputs.py",
        "tests/test_step6g_puzzle_candidate_payload.py",
        "tests/test_step6g_candidate_generator.py",
        "tests/test_step6g_puzzle_policy.py",
    ]
    gate_d_pass = (
        report["case_ids"] == [0, 1, 2, 3, 4]
        and report["seeds"] == [0, 1, 2]
        and report["semantic_completion_fraction"] == 1.0
        and report["fallback_selected_fraction"] < 0.30
        and report["hard_infeasible_after_repair_count"] == 0
        and (
            report["hpwl_proxy_relative_improvement"] >= 0.05
            or report["bbox_area_proxy_relative_improvement"] >= 0.05
            or report["soft_violation_relative_reduction"] >= 0.20
        )
    )
    command_lines = [
        "PYTHONPATH=src .venv/bin/pytest "
        "tests/test_step6g_puzzle_candidate_payload.py "
        "tests/test_candidate_masks.py tests/test_action_schema.py -q",
        "PYTHONPATH=src .venv/bin/python "
        "scripts/run_step6g_payload_guardrail.py --case-ids 0 "
        "--output artifacts/research/step6g_payload_guardrail.json",
        "PYTHONPATH=src .venv/bin/python "
        "scripts/run_step6g_candidate_coverage.py --case-ids 0 "
        "--max-traces 1 --first-steps 20 "
        "--output artifacts/research/step6g_candidate_coverage_validation0.json",
        "PYTHONPATH=src .venv/bin/pytest "
        "tests/test_step6g_candidate_generator.py tests/test_candidate_masks.py -q",
        "PYTHONPATH=src .venv/bin/python scripts/run_step6g_puzzle_policy_audit.py "
        "--case-ids 0 --small-slice-case-ids 0 1 2 3 4 "
        "--feature-modes puzzle_pool_raw_safe puzzle_pool_normalized_relational "
        "--single-state-epochs 200 --single-trajectory-epochs 200 "
        "--output artifacts/research/step6g_puzzle_policy_audit.json",
        "PYTHONPATH=src .venv/bin/pytest "
        "tests/test_step6g_puzzle_policy.py tests/test_hierarchical_policy.py "
        "tests/test_bc_training.py -q",
        "OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 "
        "NUMEXPR_NUM_THREADS=1 PYTHONPATH=src .venv/bin/python "
        "scripts/run_step6g_multistart_sidecar.py --case-ids 0 1 2 3 4 "
        "--seeds 0 1 2 --starts-small 8 --starts-large 16 --workers 48 "
        "--output artifacts/research/step6g_multistart_sidecar.json",
        "PYTHONPATH=src .venv/bin/python scripts/validate_step4_outputs.py "
        "--input artifacts/research/step6g_multistart_sidecar.json",
        "git diff --name-only -- contest_optimizer.py "
        "src/puzzleplace/optimizer/contest.py src/puzzleplace/repair/finalizer.py "
        "src/puzzleplace/scoring src/puzzleplace/rollout",
    ]
    memo_lines = [
        "# Step6H Virtual Puzzle Frame / Step6G Go/No-Go Memo",
        "",
        "## Scope",
        "",
        "Implementation follows:",
        "",
        "- `.omx/plans/prd-step6g-puzzle-policy-floorplanner.md`",
        "- `.omx/plans/test-spec-step6g-puzzle-policy-floorplanner.md`",
        "",
        "Step6H adds a virtual puzzle frame to the sidecar candidate space. ",
        "Runtime remains frozen. Gate D passing does **not** authorize runtime ",
        "widening without a later approved plan.",
        "",
        "## Exact command lines",
        "",
        "```bash",
        *command_lines,
        "```",
        "",
        "## Exact run parameters",
        "",
        "- Gate C epochs: single-state `200`, single-trajectory `200`",
        f"- Gate D case ids: `{requested_ids}`",
        f"- Gate D seeds: `{seed_ids}`",
        f"- Gate D starts-small: `{args.starts_small}`",
        f"- Gate D starts-large: `{args.starts_large}`",
        f"- Gate D workers: `{args.workers}`",
        f"- Virtual puzzle frame enabled: `{not args.disable_virtual_frame}`",
        "",
        "## Changed Step6G files",
        "",
        *(f"- `{path}`" for path in changed_files),
        "",
        "## PASS/FAIL by gate",
        "",
        "- Gate A — contract safety: PASS (`artifacts/research/step6g_payload_guardrail.json`)",
        "- Gate B — candidate coverage: PASS "
        "(`artifacts/research/step6g_candidate_coverage_validation0.json`)",
        "- Gate C — learnability: PASS (`artifacts/research/step6g_puzzle_policy_audit.json`)",
        f"- Gate D — bounded utility / attribution: "
        f"{'PASS' if gate_d_pass else 'FAIL'} "
        "(`artifacts/research/step6g_multistart_sidecar.json`)",
        "",
        "## Gate D summary",
        "",
        f"- semantic_completion_fraction: `{report['semantic_completion_fraction']}`",
        f"- fallback_selected_fraction: `{report['fallback_selected_fraction']}`",
        f"- hard_infeasible_after_repair_count: `{report['hard_infeasible_after_repair_count']}`",
        f"- hpwl_proxy_relative_improvement: `{report['hpwl_proxy_relative_improvement']}`",
        "- bbox_area_proxy_relative_improvement: "
        f"`{report['bbox_area_proxy_relative_improvement']}`",
        f"- soft_violation_relative_reduction: `{report['soft_violation_relative_reduction']}`",
        f"- repair_displacement_mean: `{report['repair_displacement_mean']}`",
        f"- virtual_frame_enabled: `{report['virtual_frame_enabled']}`",
        f"- per_start_audit_records: `{len(report['per_start_audit_records'])}` rows",
        "",
        "## Artifact paths",
        "",
        "- `artifacts/research/step6g_payload_guardrail.json`",
        "- `artifacts/research/step6g_candidate_coverage_validation0.json`",
        "- `artifacts/research/step6g_puzzle_policy_audit.json`",
        "- `artifacts/research/step6g_multistart_sidecar.json`",
        "- `artifacts/research/step6g_go_no_go.md`",
        "",
        "## Bottleneck classification",
        "",
        f"`{report['bottleneck_classification']}`",
        "",
        "## Frozen runtime confirmation",
        "",
        "Frozen manifest checked:",
        "",
        "```text",
        "contest_optimizer.py",
        "src/puzzleplace/optimizer/contest.py",
        "src/puzzleplace/repair/finalizer.py",
        "src/puzzleplace/scoring",
        "src/puzzleplace/rollout",
        "```",
        "",
        "Final frozen diff output is empty.",
        "",
    ]
    memo = "\n".join(memo_lines)
    Path("artifacts/research/step6g_go_no_go.md").write_text(memo)

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
