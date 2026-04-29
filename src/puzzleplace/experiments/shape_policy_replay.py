from __future__ import annotations

import importlib.util
import time
from pathlib import Path
from types import ModuleType
from typing import Any, cast

from puzzleplace.actions import ActionExecutor, ExecutionState
from puzzleplace.alternatives.shape_policy import (
    SHAPE_POLICIES,
    ShapePolicyName,
    cap_for_block,
    capped_shape,
    pareto_front,
    posthoc_shape_probe,
    select_shape_policy_representatives,
    shape_policy_eval_row,
)
from puzzleplace.repair import finalize_layout
from puzzleplace.research.puzzle_candidate_payload import (
    BoundaryCommitMode,
    build_puzzle_candidate_descriptors,
    heuristic_scores,
)
from puzzleplace.research.virtual_frame import (
    Placement,
    PuzzleFrame,
    estimate_predicted_compact_hull,
    multistart_virtual_frames,
)


def _load_step6g_module() -> ModuleType:
    path = Path("scripts/run_step6g_multistart_sidecar.py")
    spec = importlib.util.spec_from_file_location("step6g_multistart_sidecar", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_STEP6G = _load_step6g_module()


def reconstruct_original_layout(
    case: Any,
    case_id: int,
    step6j: dict[str, Any] | None = None,
    boundary_commit_mode: BoundaryCommitMode = "prefer_predicted_hull",
) -> tuple[dict[int, Placement], PuzzleFrame, dict[str, int]]:
    step6j = step6j or {}
    frames = multistart_virtual_frames(case)
    runs_by_case = {int(row["case_id"]): row for row in step6j.get("runs", [])}
    if case_id in runs_by_case:
        run = runs_by_case[case_id]
        frame = frames[int(run["best_frame_index"])]
        seed = int(run["best_seed"])
        start = int(run["best_start"])
    else:
        frame = frames[case_id % len(frames)]
        seed = case_id % 3
        start = case_id % 5
    pre, family_usage, construction_frame, _predicted_hull = _STEP6G._construct(
        case,
        seed,
        start,
        frame,
        boundary_commit_mode=boundary_commit_mode,
    )
    repair = finalize_layout(case, pre)
    post = {
        idx: (float(box[0]), float(box[1]), float(box[2]), float(box[3]))
        for idx, box in enumerate(repair.positions)
    }
    return post, cast(PuzzleFrame, construction_frame or frame), dict(family_usage)


def construct_with_shape_policy(
    case: Any,
    seed: int,
    start: int,
    frame: PuzzleFrame | None,
    *,
    policy: ShapePolicyName,
    boundary_commit_mode: BoundaryCommitMode,
) -> tuple[dict[int, Placement], dict[str, int], PuzzleFrame | None]:
    state = ExecutionState()
    executor = ActionExecutor(case)
    family_usage: dict[str, int] = {}
    active_frame = frame
    predicted_hull = (
        estimate_predicted_compact_hull(case, active_frame) if active_frame is not None else None
    )
    max_shape_bins = 3 if case.block_count >= 50 else 5
    max_descriptors = 8 if case.block_count >= 50 else 24
    relax_steps = 2 if case.block_count >= 50 else 6
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
            order, key=lambda idx: (_STEP6G._boundary_order_tier(case, idx), position[idx])
        )
    for block_index in order:
        if block_index in state.placements:
            continue
        cap = cap_for_block(policy, case, block_index, state.placements)
        descriptors = build_puzzle_candidate_descriptors(
            case,
            state,
            remaining_blocks=[block_index],
            max_shape_bins=max_shape_bins,
            max_descriptors_per_block=max_descriptors,
            virtual_frame=active_frame,
            predicted_hull=predicted_hull,
            frame_relaxation_steps=0,
            boundary_commit_mode=boundary_commit_mode,
            shape_log_aspect_limit=cap,
        )
        if not descriptors and active_frame is not None:
            for _relax in range(relax_steps):
                active_frame = active_frame.expanded(1.05)
                predicted_hull = estimate_predicted_compact_hull(case, active_frame)
                descriptors = build_puzzle_candidate_descriptors(
                    case,
                    state,
                    remaining_blocks=[block_index],
                    max_shape_bins=max_shape_bins,
                    max_descriptors_per_block=max_descriptors,
                    virtual_frame=active_frame,
                    predicted_hull=predicted_hull,
                    frame_relaxation_steps=0,
                    boundary_commit_mode=boundary_commit_mode,
                    shape_log_aspect_limit=cap,
                )
                if descriptors:
                    break
        if not descriptors:
            side = float(case.area_targets[block_index].sqrt().item())
            if active_frame is None:
                x = max((px + pw for px, _py, pw, _ph in state.placements.values()), default=0.0)
                box = (x, 0.0, side, side)
            else:
                box, active_frame = _STEP6G._frame_fallback_box(
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
    return state.placements, family_usage, active_frame


def construction_shape_policy_replay(
    case: Any,
    frame: PuzzleFrame,
    policy: ShapePolicyName,
    boundary_commit_mode: BoundaryCommitMode,
    *,
    seed: int = 0,
    start: int = 0,
) -> tuple[dict[int, Placement], dict[str, int], PuzzleFrame]:
    if case.block_count >= 50:
        return fast_construction_shape_policy_replay(case, frame, policy)
    pre, family_usage, active_frame = construct_with_shape_policy(
        case,
        seed,
        start,
        frame,
        policy=policy,
        boundary_commit_mode=boundary_commit_mode,
    )
    repair = finalize_layout(case, pre)
    post = {
        idx: (float(box[0]), float(box[1]), float(box[2]), float(box[3]))
        for idx, box in enumerate(repair.positions)
    }
    return post, family_usage, active_frame or frame


def fast_construction_shape_policy_replay(
    case: Any,
    frame: PuzzleFrame,
    policy: ShapePolicyName,
) -> tuple[dict[int, Placement], dict[str, int], PuzzleFrame]:
    active = frame
    placements: dict[int, Placement] = {}
    x = active.xmin
    y = active.ymin
    row_h = 0.0
    order = sorted(
        range(case.block_count),
        key=lambda idx: (
            -int(case.constraints[idx, 4].item() != 0),
            -float(case.area_targets[idx].item()),
        ),
    )
    for idx in order:
        area = float(case.area_targets[idx].item())
        side = area**0.5
        cap = cap_for_block(policy, case, idx, placements)
        width, height = capped_shape(area, side, side, cap)
        if x + width > active.xmax and placements:
            x = active.xmin
            y += row_h + 0.5
            row_h = 0.0
        while y + height > active.ymax:
            active = active.expanded(1.10)
        placements[idx] = (float(x), float(y), float(width), float(height))
        x += width + 0.5
        row_h = max(row_h, height)
    repair = finalize_layout(case, placements)
    post = {
        idx: (float(box[0]), float(box[1]), float(box[2]), float(box[3]))
        for idx, box in enumerate(repair.positions)
    }
    return post, {f"fast_construction:{policy}": case.block_count}, active


def evaluate_shape_policy_case(
    *,
    case: Any,
    baseline: dict[int, Placement],
    frame: PuzzleFrame,
    boundary_commit_mode: BoundaryCommitMode,
    policies: tuple[ShapePolicyName, ...] = SHAPE_POLICIES,
) -> tuple[list[dict[str, Any]], dict[tuple[str, str], tuple[dict[int, Placement], PuzzleFrame]]]:
    rows: list[dict[str, Any]] = []
    layouts: dict[tuple[str, str], tuple[dict[int, Placement], PuzzleFrame]] = {}
    for policy in policies:
        posthoc_t0 = time.perf_counter()
        if policy == "original_shape_policy":
            alternative = dict(baseline)
            reasons: list[dict[str, Any]] = []
        else:
            alternative, reasons = posthoc_shape_probe(policy, case, baseline, frame)
        posthoc = shape_policy_eval_row(
            case=case,
            policy=policy,
            track="posthoc_shape_probe",
            baseline=baseline,
            alternative=alternative,
            frame=frame,
            role_cap_reasons=reasons,
        )
        posthoc["runtime_estimate_ms"] = (time.perf_counter() - posthoc_t0) * 1000.0
        rows.append(posthoc)
        layouts[(posthoc["track"], policy)] = (alternative, frame)

        replay_t0 = time.perf_counter()
        replay, family_usage, replay_frame = construction_shape_policy_replay(
            case,
            frame,
            policy,
            boundary_commit_mode,
        )
        replay_row = shape_policy_eval_row(
            case=case,
            policy=policy,
            track="construction_shape_policy_replay",
            baseline=baseline,
            alternative=replay,
            frame=replay_frame,
            role_cap_reasons=[],
        )
        replay_row["runtime_estimate_ms"] = (time.perf_counter() - replay_t0) * 1000.0
        replay_row["candidate_family_usage"] = family_usage
        rows.append(replay_row)
        layouts[(replay_row["track"], policy)] = (replay, replay_frame)
    return rows, layouts


def shape_policy_pareto_representatives(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return select_shape_policy_representatives(pareto_front(rows))
