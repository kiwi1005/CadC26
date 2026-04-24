#!/usr/bin/env python3
# ruff: noqa: E402
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for path in (ROOT, SRC):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from puzzleplace.actions import ActionExecutor, ExecutionState, canonical_action_key, generate_candidate_actions  # noqa: E402
from puzzleplace.actions.schema import ActionPrimitive  # noqa: E402
from puzzleplace.eval import evaluate_positions  # noqa: E402
from puzzleplace.repair.finalizer import finalize_layout  # noqa: E402
from puzzleplace.data import ConstraintColumns  # noqa: E402
from puzzleplace.models import (  # noqa: E402
    CandidateComponentRanker,
    CandidateLateFusionRanker,
    CandidateQualityRanker,
    CandidateRelationalActionQRanker,
    CandidateConstraintTokenRanker,
    CandidateSetPairwiseRanker,
)
from puzzleplace.roles import label_case_roles  # noqa: E402
from puzzleplace.train import build_bc_dataset_from_cases, load_validation_cases  # noqa: E402

from scripts.run_step6_hierarchical_rollout_control_audit import (  # noqa: E402
    RolloutJob,
    _forced_progress_action,
    _score_hierarchical_action,
    _seed_first_action,
    _train_hierarchical_policy,
)
from scripts.run_step6c_hierarchical_quality_alignment_audit import _quality_after_action  # noqa: E402


FEATURE_NAMES = [
    "policy_score",
    "block_logit",
    "primitive_logit",
    "target_logit",
    "heuristic_score",
    "primitive_id_norm",
    "block_index_norm",
    "has_target",
    "target_index_norm",
    "area_error",
    "total_overlap_area",
    "boundary_distance",
    "connectivity_proxy_cost",
    "is_place_relative",
    "is_place_absolute",
    "is_freeze",
    "x_norm",
    "y_norm",
    "w_norm",
    "h_norm",
    "dx_norm",
    "dy_norm",
    "source_pin",
    "source_strip",
    "intent_attach",
    "intent_boundary",
    "intent_pin",
    "intent_strip",
]

OBJECTIVE_COMPONENTS = ("HPWLgap", "Areagap_bbox", "Violationsrelative")
POOL_RELATIVE_BASE_FEATURES = (
    "policy_score",
    "block_logit",
    "primitive_logit",
    "target_logit",
    "heuristic_score",
    "area_error",
    "total_overlap_area",
    "boundary_distance",
    "connectivity_proxy_cost",
)
STATE_POOL_EXTRA_FEATURE_NAMES = [
    "pool_policy_score_z",
    "pool_policy_score_rank",
    "pool_block_logit_z",
    "pool_block_logit_rank",
    "pool_primitive_logit_z",
    "pool_primitive_logit_rank",
    "pool_target_logit_z",
    "pool_target_logit_rank",
    "pool_heuristic_score_z",
    "pool_heuristic_score_rank",
    "pool_area_error_z",
    "pool_area_error_rank",
    "pool_total_overlap_area_z",
    "pool_total_overlap_area_rank",
    "pool_boundary_distance_z",
    "pool_boundary_distance_rank",
    "pool_connectivity_proxy_cost_z",
    "pool_connectivity_proxy_cost_rank",
    "case_block_count_norm",
    "case_total_area_log",
    "step_fraction",
    "placed_fraction",
    "semantic_placed_fraction",
    "occupied_area_fraction",
    "layout_bbox_area_fraction",
    "layout_bbox_aspect_log",
    "layout_width_norm",
    "layout_height_norm",
    "block_area_fraction",
    "block_sqrt_area_fraction",
    "candidate_area_fraction",
    "candidate_area_ratio_error",
    "candidate_aspect_log",
    "resolved_x_norm",
    "resolved_y_norm",
    "resolved_center_x_norm",
    "resolved_center_y_norm",
    "resolved_w_norm",
    "resolved_h_norm",
    "candidate_left_touch",
    "candidate_bottom_touch",
    "candidate_inside_layout_x",
    "candidate_inside_layout_y",
    "candidate_center_to_layout_center_norm",
    "target_area_fraction",
    "target_area_ratio",
    "b2b_degree_norm",
    "p2b_degree_norm",
    "boundary_code_norm",
    "is_fixed",
    "is_preplaced",
    "is_mib",
    "is_cluster",
]
STATE_POOL_FEATURE_NAMES = FEATURE_NAMES + STATE_POOL_EXTRA_FEATURE_NAMES
RELATIONAL_STATE_EXTRA_FEATURE_NAMES = [
    "rel_b2b_weight_total_norm",
    "rel_b2b_weight_to_placed_norm",
    "rel_b2b_weight_to_unplaced_norm",
    "rel_b2b_placed_fraction",
    "rel_placed_neighbor_count_norm",
    "rel_unplaced_neighbor_count_norm",
    "rel_weighted_placed_distance_norm",
    "rel_nearest_placed_distance_norm",
    "rel_target_b2b_weight_norm",
    "rel_target_is_fixed",
    "rel_target_is_preplaced",
    "rel_connected_placed_area_fraction",
    "rel_connected_unplaced_area_fraction",
    "rel_pin_weight_total_norm",
    "rel_weighted_pin_distance_norm",
    "rel_nearest_pin_distance_norm",
    "rel_pin_pull_x_norm",
    "rel_pin_pull_y_norm",
]
RELATIONAL_STATE_POOL_FEATURE_NAMES = STATE_POOL_FEATURE_NAMES + RELATIONAL_STATE_EXTRA_FEATURE_NAMES
CONSTRAINT_RELATION_EXTRA_FEATURE_NAMES = [
    "crel_same_cluster_as_target",
    "crel_same_mib_as_target",
    "crel_boundary_code_match_target",
    "crel_boundary_side_compatible_with_action",
    "crel_boundary_touch_distance_norm",
    "crel_nearest_preplaced_distance_norm",
    "crel_nearest_fixed_distance_norm",
    "crel_same_cluster_placed_fraction",
    "crel_same_cluster_unplaced_fraction",
    "crel_same_mib_placed_fraction",
    "crel_same_mib_unplaced_fraction",
    "crel_preplaced_anchor_b2b_weight_norm",
    "crel_fixed_anchor_b2b_weight_norm",
    "crel_boundary_group_b2b_weight_norm",
    "crel_cluster_group_b2b_weight_norm",
    "crel_mib_group_b2b_weight_norm",
]
CONSTRAINT_RELATION_POOL_FEATURE_NAMES = (
    RELATIONAL_STATE_POOL_FEATURE_NAMES + CONSTRAINT_RELATION_EXTRA_FEATURE_NAMES
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train/evaluate a Step6C hierarchical action-Q ranker on semantic candidate pools."
    )
    parser.add_argument("--case-ids", nargs="*", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument(
        "--train-case-ids",
        nargs="*",
        type=int,
        default=None,
        help=(
            "Case indexes used to train the ranker. Defaults to all collected "
            "cases except --eval-case-ids when an eval split is provided."
        ),
    )
    parser.add_argument(
        "--eval-case-ids",
        nargs="*",
        type=int,
        default=None,
        help=(
            "Case indexes used for held-out evaluation. Omit to keep the "
            "original micro-overfit behavior."
        ),
    )
    parser.add_argument(
        "--leave-one-case-out",
        action="store_true",
        help=(
            "Run one held-out split per collected case and report aggregate "
            "generalization diagnostics."
        ),
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--policy-seeds",
        nargs="*",
        type=int,
        default=None,
        help=(
            "Train one sidecar hierarchical policy per seed and collect one "
            "candidate-pool trajectory per case/seed. Defaults to --seed."
        ),
    )
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument(
        "--encoder-kind",
        choices=["graph", "relation_aware", "typed_constraint_graph", "typed_constraint_graph_no_anchor", "typed_constraint_graph_no_boundary", "typed_constraint_graph_no_groups"],
        default="graph",
        help="Sidecar hierarchical policy state encoder.",
    )
    parser.add_argument("--policy-epochs", type=int, default=120)
    parser.add_argument("--ranker-epochs", type=int, default=200)
    parser.add_argument(
        "--feature-mode",
        choices=[
            "legacy",
            "state_pool",
            "state_pool_no_raw_logits",
            "relational_state_pool_no_raw_logits",
            "constraint_relation_pool_no_raw_logits",
        ],
        default="legacy",
        help=(
            "Feature set for candidate ranking. state_pool adds within-pool "
            "normalization and state/case-conditioned geometry features. "
            "state_pool_no_raw_logits keeps pool-relative logit features but "
            "zeros raw cross-case policy/logit scale features."
        ),
    )
    parser.add_argument(
        "--feature-normalization",
        choices=["global", "per_case"],
        default="global",
        help=(
            "Normalize ranker feature rows globally as before, or first "
            "standardize each case's candidate rows to remove cross-scale drift."
        ),
    )
    parser.add_argument(
        "--ranker-kind",
        choices=[
            "scalar",
            "component",
            "pairwise_set",
            "hybrid_set",
            "late_fusion",
            "relational_action_q",
            "constraint_token_action_q",
        ],
        default="scalar",
        help=(
            "Use a scalar ranker, a component-aware ranker, or a set-attentive "
            "pairwise/hybrid comparator."
        ),
    )
    parser.add_argument(
        "--component-loss-weight",
        type=float,
        default=0.5,
        help="Auxiliary component CE loss weight for --ranker-kind component.",
    )
    parser.add_argument(
        "--component-score-weight",
        type=float,
        default=0.25,
        help="Component-logit contribution to selection score for --ranker-kind component.",
    )
    parser.add_argument(
        "--target-kind",
        choices=["oracle_ce", "soft_quality", "topk_quality"],
        default="oracle_ce",
        help=(
            "Ranker training target. soft_quality uses a listwise distribution "
            "over quality_cost_runtime1 instead of only the top-1 oracle."
        ),
    )
    parser.add_argument(
        "--quality-temperature",
        type=float,
        default=0.5,
        help="Temperature for --target-kind soft_quality.",
    )
    parser.add_argument(
        "--pairwise-loss-weight-kind",
        choices=["quality_delta", "uniform"],
        default="quality_delta",
        help="Pair loss weighting for --ranker-kind pairwise_set.",
    )
    parser.add_argument(
        "--pairwise-listwise-loss-weight",
        type=float,
        default=0.0,
        help=(
            "Optional pool-level CE/listwise auxiliary loss for "
            "--ranker-kind pairwise_set."
        ),
    )
    parser.add_argument(
        "--hybrid-scalar-loss-weight",
        type=float,
        default=1.0,
        help="Scalar listwise loss weight for --ranker-kind hybrid_set.",
    )
    parser.add_argument(
        "--hybrid-pairwise-score-weight",
        type=float,
        default=0.5,
        help="Pairwise score mixture weight for --ranker-kind hybrid_set.",
    )
    parser.add_argument(
        "--relational-pairwise-loss-weight",
        type=float,
        default=0.5,
        help="Pairwise preference auxiliary loss for --ranker-kind relational_action_q.",
    )
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--ranker-lr", type=float, default=1e-2)
    parser.add_argument("--primitive-set-weight", type=float, default=1.0)
    parser.add_argument("--block-weight", type=float, default=1.0)
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Parallel worker cap for independent case/seed or LOCO split jobs. Keep at 1 for smoke; use 48 only after the relevant smoke/neutral gate passes.",
    )
    parser.add_argument("--max-steps", type=int, default=8)
    parser.add_argument("--max-candidates-per-step", type=int, default=16)
    parser.add_argument(
        "--label-kind",
        choices=["immediate", "rollout_return"],
        default="immediate",
        help="Quality label used to train/evaluate candidate ranking pools.",
    )
    parser.add_argument(
        "--continuation-horizon",
        type=int,
        default=8,
        help="Greedy continuation horizon for --label-kind rollout_return.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "artifacts" / "research" / "step6c_hierarchical_action_q_audit.json",
    )
    return parser.parse_args()


def _feature_vector(case, action, policy_score: float, components: dict[str, Any]) -> list[float]:
    primitive_id = list(ActionPrimitive).index(action.primitive)
    block_denom = max(case.block_count - 1, 1)
    target_index = -1 if action.target_index is None else int(action.target_index)
    return [
        float(policy_score),
        float(components.get("block_logit") or 0.0),
        float(components.get("primitive_logit") or 0.0),
        float(components.get("target_logit") or 0.0),
        float(components.get("heuristic_score") or 0.0),
        primitive_id / max(len(ActionPrimitive) - 1, 1),
        float(action.block_index) / float(block_denom),
        0.0 if action.target_index is None else 1.0,
        float(target_index) / float(block_denom) if target_index >= 0 else -1.0,
        float(action.metadata.get("area_error", 0.0)),
        float(action.metadata.get("total_overlap_area", 0.0)),
        float(action.metadata.get("boundary_distance", 0.0)),
        float(action.metadata.get("connectivity_proxy_cost", 0.0)),
        1.0 if action.primitive is ActionPrimitive.PLACE_RELATIVE else 0.0,
        1.0 if action.primitive is ActionPrimitive.PLACE_ABSOLUTE else 0.0,
        1.0 if action.primitive is ActionPrimitive.FREEZE else 0.0,
        float(action.x or 0.0) / 1000.0,
        float(action.y or 0.0) / 1000.0,
        float(action.w or 0.0) / 100.0,
        float(action.h or 0.0) / 100.0,
        float(action.dx or 0.0) / 100.0,
        float(action.dy or 0.0) / 100.0,
        1.0 if "pin" in str(action.metadata.get("source", "")) else 0.0,
        1.0 if "strip" in str(action.metadata.get("source", "")) else 0.0,
        1.0 if "attach" in str(action.metadata.get("intent_type", "")) else 0.0,
        1.0 if "boundary" in str(action.metadata.get("intent_type", "")) else 0.0,
        1.0 if "pin" in str(action.metadata.get("intent_type", "")) else 0.0,
        1.0 if "strip" in str(action.metadata.get("intent_type", "")) else 0.0,
    ]


def _feature_names(feature_mode: str) -> list[str]:
    if feature_mode == "constraint_relation_pool_no_raw_logits":
        return CONSTRAINT_RELATION_POOL_FEATURE_NAMES
    if feature_mode == "relational_state_pool_no_raw_logits":
        return RELATIONAL_STATE_POOL_FEATURE_NAMES
    if feature_mode in {"state_pool", "state_pool_no_raw_logits"}:
        return STATE_POOL_FEATURE_NAMES
    return FEATURE_NAMES


def _state_bbox(
    placements: dict[int, tuple[float, float, float, float]],
) -> tuple[float, float, float, float]:
    if not placements:
        return (0.0, 0.0, 0.0, 0.0)
    min_x = min(x for x, _y, _w, _h in placements.values())
    min_y = min(y for _x, y, _w, _h in placements.values())
    max_x = max(x + width for x, _y, width, _h in placements.values())
    max_y = max(y + height for _x, y, _w, height in placements.values())
    return (min_x, min_y, max_x, max_y)


def _candidate_box_from_action(
    state: ExecutionState,
    action,
) -> tuple[float, float, float, float] | None:
    if action.primitive is ActionPrimitive.PLACE_ABSOLUTE:
        if action.x is None or action.y is None or action.w is None or action.h is None:
            return None
        return (float(action.x), float(action.y), float(action.w), float(action.h))
    if action.primitive is ActionPrimitive.PLACE_RELATIVE:
        if (
            action.target_index is None
            or action.dx is None
            or action.dy is None
            or action.w is None
            or action.h is None
            or int(action.target_index) not in state.placements
        ):
            return None
        tx, ty, tw, _th = state.placements[int(action.target_index)]
        return (tx + tw + float(action.dx), ty + float(action.dy), float(action.w), float(action.h))
    if action.block_index in state.placements:
        return state.placements[action.block_index]
    return None


def _degree_features(case, block_index: int) -> tuple[float, float]:
    b2b_degree = 0.0
    for src, dst, weight in case.b2b_edges.tolist():
        if int(src) == block_index or int(dst) == block_index:
            b2b_degree += abs(float(weight))
    p2b_degree = 0.0
    for _pin, block, weight in case.p2b_edges.tolist():
        if int(block) == block_index:
            p2b_degree += abs(float(weight))
    return b2b_degree / max(case.block_count, 1), p2b_degree / max(case.block_count, 1)


def _state_pool_extra_features(case, state: ExecutionState, action) -> list[float]:
    total_area = max(float(case.area_targets.sum().item()), 1e-6)
    case_scale = max(math.sqrt(total_area), 1e-6)
    bbox_min_x, bbox_min_y, bbox_max_x, bbox_max_y = _state_bbox(state.placements)
    bbox_w = max(bbox_max_x - bbox_min_x, 0.0)
    bbox_h = max(bbox_max_y - bbox_min_y, 0.0)
    bbox_area = bbox_w * bbox_h
    occupied_area = sum(w * h for _x, _y, w, h in state.placements.values())
    bbox_aspect_log = math.log((bbox_w + 1e-6) / (bbox_h + 1e-6))
    block_area = max(float(case.area_targets[action.block_index].item()), 1e-6)
    box = _candidate_box_from_action(state, action)
    if box is None:
        cx = cy = cw = ch = 0.0
    else:
        cx, cy, cw, ch = box
    candidate_area = max(cw * ch, 0.0)
    center_x = cx + cw / 2.0
    center_y = cy + ch / 2.0
    layout_center_x = bbox_min_x + bbox_w / 2.0
    layout_center_y = bbox_min_y + bbox_h / 2.0
    center_distance = abs(center_x - layout_center_x) + abs(center_y - layout_center_y)
    target_area = 0.0
    if action.target_index is not None:
        target_area = float(case.area_targets[int(action.target_index)].item())
    b2b_degree, p2b_degree = _degree_features(case, action.block_index)
    constraints = case.constraints[action.block_index]
    boundary_code = float(constraints[ConstraintColumns.BOUNDARY].item())
    return [
        case.block_count / 100.0,
        math.log1p(total_area),
        state.step / max(case.block_count * 4, 1),
        len(state.placements) / max(case.block_count, 1),
        len(state.semantic_placed) / max(case.block_count, 1),
        occupied_area / total_area,
        bbox_area / total_area,
        bbox_aspect_log,
        bbox_w / case_scale,
        bbox_h / case_scale,
        block_area / total_area,
        math.sqrt(block_area) / case_scale,
        candidate_area / total_area,
        (candidate_area - block_area) / block_area,
        math.log((cw + 1e-6) / (ch + 1e-6)),
        cx / case_scale,
        cy / case_scale,
        center_x / case_scale,
        center_y / case_scale,
        cw / case_scale,
        ch / case_scale,
        1.0 if abs(cx) <= 1e-6 else 0.0,
        1.0 if abs(cy) <= 1e-6 else 0.0,
        1.0 if bbox_min_x <= center_x <= bbox_max_x and bbox_w > 0 else 0.0,
        1.0 if bbox_min_y <= center_y <= bbox_max_y and bbox_h > 0 else 0.0,
        center_distance / case_scale,
        target_area / total_area,
        target_area / block_area if target_area > 0 else 0.0,
        b2b_degree,
        p2b_degree,
        boundary_code / 8.0,
        float(constraints[ConstraintColumns.FIXED].item()),
        float(constraints[ConstraintColumns.PREPLACED].item()),
        float(constraints[ConstraintColumns.MIB].item()),
        float(constraints[ConstraintColumns.CLUSTER].item()),
    ]


def _relational_state_extra_features(case, state: ExecutionState, action) -> list[float]:
    total_area = max(float(case.area_targets.sum().item()), 1e-6)
    case_scale = max(math.sqrt(total_area), 1e-6)
    box = _candidate_box_from_action(state, action)
    if box is None:
        center_x = center_y = 0.0
    else:
        x, y, w, h = box
        center_x = x + w / 2.0
        center_y = y + h / 2.0

    b2b_total = 0.0
    b2b_placed = 0.0
    b2b_unplaced = 0.0
    placed_count = 0
    unplaced_count = 0
    weighted_distance = 0.0
    nearest_distance = None
    target_weight = 0.0
    connected_placed_area = 0.0
    connected_unplaced_area = 0.0
    connected_seen: set[int] = set()
    for src, dst, weight in case.b2b_edges.tolist():
        src_idx = int(src)
        dst_idx = int(dst)
        if src_idx == -1 or dst_idx == -1:
            continue
        if src_idx == action.block_index:
            other_idx = dst_idx
        elif dst_idx == action.block_index:
            other_idx = src_idx
        else:
            continue
        edge_weight = abs(float(weight))
        b2b_total += edge_weight
        if action.target_index is not None and other_idx == int(action.target_index):
            target_weight += edge_weight
        if other_idx in state.placements:
            b2b_placed += edge_weight
            placed_count += 1
            ox, oy, ow, oh = state.placements[other_idx]
            other_center_x = ox + ow / 2.0
            other_center_y = oy + oh / 2.0
            distance = abs(center_x - other_center_x) + abs(center_y - other_center_y)
            weighted_distance += edge_weight * distance
            nearest_distance = distance if nearest_distance is None else min(nearest_distance, distance)
            if other_idx not in connected_seen:
                connected_placed_area += float(case.area_targets[other_idx].item())
                connected_seen.add(other_idx)
        else:
            b2b_unplaced += edge_weight
            unplaced_count += 1
            if other_idx not in connected_seen and 0 <= other_idx < case.block_count:
                connected_unplaced_area += float(case.area_targets[other_idx].item())
                connected_seen.add(other_idx)

    pin_weight_total = 0.0
    pin_weighted_x = 0.0
    pin_weighted_y = 0.0
    weighted_pin_distance = 0.0
    nearest_pin_distance = None
    for pin_idx, block_idx, weight in case.p2b_edges.tolist():
        if int(block_idx) != action.block_index or int(pin_idx) < 0:
            continue
        edge_weight = abs(float(weight))
        px, py = [float(v) for v in case.pins_pos[int(pin_idx)].tolist()]
        pin_weight_total += edge_weight
        pin_weighted_x += edge_weight * px
        pin_weighted_y += edge_weight * py
        distance = abs(center_x - px) + abs(center_y - py)
        weighted_pin_distance += edge_weight * distance
        nearest_pin_distance = (
            distance if nearest_pin_distance is None else min(nearest_pin_distance, distance)
        )
    if pin_weight_total > 0:
        pin_center_x = pin_weighted_x / pin_weight_total
        pin_center_y = pin_weighted_y / pin_weight_total
        pin_pull_x = (pin_center_x - center_x) / case_scale
        pin_pull_y = (pin_center_y - center_y) / case_scale
        weighted_pin_distance_norm = weighted_pin_distance / pin_weight_total / case_scale
    else:
        pin_pull_x = pin_pull_y = weighted_pin_distance_norm = 0.0

    target_is_fixed = 0.0
    target_is_preplaced = 0.0
    if action.target_index is not None and 0 <= int(action.target_index) < case.block_count:
        target_constraints = case.constraints[int(action.target_index)]
        target_is_fixed = float(target_constraints[ConstraintColumns.FIXED].item())
        target_is_preplaced = float(target_constraints[ConstraintColumns.PREPLACED].item())

    weight_denom = max(case.block_count, 1)
    return [
        b2b_total / weight_denom,
        b2b_placed / weight_denom,
        b2b_unplaced / weight_denom,
        b2b_placed / max(b2b_total, 1e-6),
        placed_count / max(case.block_count, 1),
        unplaced_count / max(case.block_count, 1),
        weighted_distance / max(b2b_placed, 1e-6) / case_scale if b2b_placed > 0 else 0.0,
        0.0 if nearest_distance is None else nearest_distance / case_scale,
        target_weight / weight_denom,
        target_is_fixed,
        target_is_preplaced,
        connected_placed_area / total_area,
        connected_unplaced_area / total_area,
        pin_weight_total / weight_denom,
        weighted_pin_distance_norm,
        0.0 if nearest_pin_distance is None else nearest_pin_distance / case_scale,
        pin_pull_x,
        pin_pull_y,
    ]


def _constraint_relation_extra_features(case, state: ExecutionState, action) -> list[float]:
    total_area = max(float(case.area_targets.sum().item()), 1e-6)
    case_scale = max(math.sqrt(total_area), 1e-6)
    constraints = case.constraints[action.block_index]
    cluster_id = int(constraints[ConstraintColumns.CLUSTER].item())
    mib_flag = bool(constraints[ConstraintColumns.MIB].item())
    boundary_code = int(constraints[ConstraintColumns.BOUNDARY].item())
    target_constraints = None
    if action.target_index is not None and 0 <= int(action.target_index) < case.block_count:
        target_constraints = case.constraints[int(action.target_index)]
    same_cluster_as_target = 0.0
    same_mib_as_target = 0.0
    boundary_code_match_target = 0.0
    if target_constraints is not None:
        target_cluster = int(target_constraints[ConstraintColumns.CLUSTER].item())
        same_cluster_as_target = float(cluster_id != 0 and cluster_id == target_cluster)
        same_mib_as_target = float(mib_flag and bool(target_constraints[ConstraintColumns.MIB].item()))
        boundary_code_match_target = float(
            boundary_code != 0
            and boundary_code == int(target_constraints[ConstraintColumns.BOUNDARY].item())
        )

    box = _candidate_box_from_action(state, action)
    if box is None:
        x = y = w = h = center_x = center_y = 0.0
    else:
        x, y, w, h = box
        center_x = x + w / 2.0
        center_y = y + h / 2.0
    bbox_min_x, bbox_min_y, bbox_max_x, bbox_max_y = _state_bbox(state.placements)
    if not state.placements:
        bbox_max_x = max(bbox_max_x, x + w)
        bbox_max_y = max(bbox_max_y, y + h)
    boundary_touch_distance = 0.0
    boundary_side_compatible = 0.0
    if boundary_code == 1:
        boundary_touch_distance = abs(x)
        boundary_side_compatible = float(abs(x) <= 1e-6)
    elif boundary_code == 2:
        boundary_touch_distance = abs(max(bbox_max_x, x + w) - (x + w))
        boundary_side_compatible = float(boundary_touch_distance <= 1e-6)
    elif boundary_code == 4:
        boundary_touch_distance = abs(max(bbox_max_y, y + h) - (y + h))
        boundary_side_compatible = float(boundary_touch_distance <= 1e-6)
    elif boundary_code == 8:
        boundary_touch_distance = abs(y)
        boundary_side_compatible = float(abs(y) <= 1e-6)

    same_cluster_total = same_cluster_placed = 0
    same_mib_total = same_mib_placed = 0
    nearest_preplaced = None
    nearest_fixed = None
    for idx in range(case.block_count):
        other_constraints = case.constraints[idx]
        if idx != action.block_index:
            if cluster_id != 0 and int(other_constraints[ConstraintColumns.CLUSTER].item()) == cluster_id:
                same_cluster_total += 1
                same_cluster_placed += int(idx in state.placements)
            if mib_flag and bool(other_constraints[ConstraintColumns.MIB].item()):
                same_mib_total += 1
                same_mib_placed += int(idx in state.placements)
        if idx in state.placements:
            ox, oy, ow, oh = state.placements[idx]
        else:
            tx, ty, tw, th = [float(v) for v in case.target_positions[idx].tolist()]
            # Only use anchor-like target geometry for fixed/preplaced distance diagnostics.
            ox, oy, ow, oh = tx, ty, tw, th
        if bool(other_constraints[ConstraintColumns.PREPLACED].item()):
            dist = abs(center_x - (ox + ow / 2.0)) + abs(center_y - (oy + oh / 2.0))
            nearest_preplaced = dist if nearest_preplaced is None else min(nearest_preplaced, dist)
        if bool(other_constraints[ConstraintColumns.FIXED].item()):
            dist = abs(center_x - (ox + ow / 2.0)) + abs(center_y - (oy + oh / 2.0))
            nearest_fixed = dist if nearest_fixed is None else min(nearest_fixed, dist)

    preplaced_anchor_weight = 0.0
    fixed_anchor_weight = 0.0
    boundary_group_weight = 0.0
    cluster_group_weight = 0.0
    mib_group_weight = 0.0
    for src, dst, weight in case.b2b_edges.tolist():
        src_idx = int(src)
        dst_idx = int(dst)
        if src_idx == action.block_index:
            other_idx = dst_idx
        elif dst_idx == action.block_index:
            other_idx = src_idx
        else:
            continue
        if not (0 <= other_idx < case.block_count):
            continue
        edge_weight = abs(float(weight))
        other_constraints = case.constraints[other_idx]
        preplaced_anchor_weight += edge_weight * float(
            bool(other_constraints[ConstraintColumns.PREPLACED].item())
        )
        fixed_anchor_weight += edge_weight * float(
            bool(other_constraints[ConstraintColumns.FIXED].item())
        )
        boundary_group_weight += edge_weight * float(
            boundary_code != 0
            and boundary_code == int(other_constraints[ConstraintColumns.BOUNDARY].item())
        )
        cluster_group_weight += edge_weight * float(
            cluster_id != 0 and cluster_id == int(other_constraints[ConstraintColumns.CLUSTER].item())
        )
        mib_group_weight += edge_weight * float(
            mib_flag and bool(other_constraints[ConstraintColumns.MIB].item())
        )
    weight_denom = max(case.block_count, 1)
    return [
        same_cluster_as_target,
        same_mib_as_target,
        boundary_code_match_target,
        boundary_side_compatible,
        boundary_touch_distance / case_scale,
        0.0 if nearest_preplaced is None else nearest_preplaced / case_scale,
        0.0 if nearest_fixed is None else nearest_fixed / case_scale,
        same_cluster_placed / max(same_cluster_total, 1),
        (same_cluster_total - same_cluster_placed) / max(same_cluster_total, 1),
        same_mib_placed / max(same_mib_total, 1),
        (same_mib_total - same_mib_placed) / max(same_mib_total, 1),
        preplaced_anchor_weight / weight_denom,
        fixed_anchor_weight / weight_denom,
        boundary_group_weight / weight_denom,
        cluster_group_weight / weight_denom,
        mib_group_weight / weight_denom,
    ]


def _pool_relative_features(base_rows: list[list[float]]) -> list[list[float]]:
    feature_indexes = [FEATURE_NAMES.index(name) for name in POOL_RELATIVE_BASE_FEATURES]
    values_by_feature = []
    for idx in feature_indexes:
        values = [float(row[idx]) for row in base_rows]
        mean = sum(values) / max(len(values), 1)
        variance = sum((value - mean) ** 2 for value in values) / max(len(values), 1)
        std = math.sqrt(max(variance, 1e-12))
        ordered = sorted(range(len(values)), key=lambda item: values[item])
        ranks = [0.0 for _ in values]
        denom = max(len(values) - 1, 1)
        for rank, original_idx in enumerate(ordered):
            ranks[original_idx] = rank / denom
        values_by_feature.append((values, mean, std, ranks))
    relative_rows = []
    for row_idx in range(len(base_rows)):
        extras = []
        for values, mean, std, ranks in values_by_feature:
            extras.append((values[row_idx] - mean) / std)
            extras.append(ranks[row_idx])
        relative_rows.append(extras)
    return relative_rows


def _build_pool_features(
    case,
    state: ExecutionState,
    selected: list[tuple[float, Any, dict[str, Any]]],
    *,
    feature_mode: str,
) -> list[list[float]]:
    base_rows = [
        _feature_vector(case, candidate, policy_score, components)
        for policy_score, candidate, components in selected
    ]
    if feature_mode == "legacy":
        return base_rows
    relative_rows = _pool_relative_features(base_rows)
    if feature_mode in {
        "state_pool_no_raw_logits",
        "relational_state_pool_no_raw_logits",
        "constraint_relation_pool_no_raw_logits",
    }:
        raw_logit_indexes = [
            FEATURE_NAMES.index(name)
            for name in (
                "policy_score",
                "block_logit",
                "primitive_logit",
                "target_logit",
                "heuristic_score",
            )
        ]
        for row in base_rows:
            for idx in raw_logit_indexes:
                row[idx] = 0.0
    rows = []
    for row_idx, (base, relative) in enumerate(zip(base_rows, relative_rows, strict=True)):
        candidate = selected[row_idx][1]
        row = base + relative + _state_pool_extra_features(case, state, candidate)
        if feature_mode in {"relational_state_pool_no_raw_logits", "constraint_relation_pool_no_raw_logits"}:
            row += _relational_state_extra_features(case, state, candidate)
        if feature_mode == "constraint_relation_pool_no_raw_logits":
            row += _constraint_relation_extra_features(case, state, candidate)
        rows.append(row)
    return rows



def _clone_execution_state(state: ExecutionState) -> ExecutionState:
    return ExecutionState(
        placements=dict(state.placements),
        frozen_blocks=set(state.frozen_blocks),
        proposed_positions=dict(state.proposed_positions),
        shape_assigned=set(state.shape_assigned),
        semantic_placed=set(state.semantic_placed),
        physically_placed=set(state.physically_placed),
        step=state.step,
        history=list(state.history),
        last_rollout_mode=state.last_rollout_mode,
    )


def _quality_from_positions(case, positions: dict[int, tuple[float, float, float, float]]) -> dict[str, Any]:
    repair = finalize_layout(case, positions)
    evaluation = evaluate_positions(case, repair.positions, runtime=1.0, median_runtime=1.0)
    return evaluation["quality"]


def _rollout_return_after_action(
    case,
    state: ExecutionState,
    action,
    *,
    policy,
    role_evidence,
    horizon: int,
    max_candidates: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    trial = _clone_execution_state(state)
    executor = ActionExecutor(case)
    executor.apply(trial, action)
    actions_taken = 1
    forced_count = 0
    no_progress = 0
    while (
        len(trial.semantic_placed) < case.block_count
        and trial.step < case.block_count * 4
        and actions_taken < horizon + 1
    ):
        remaining = [idx for idx in range(case.block_count) if idx not in trial.semantic_placed]
        if not remaining:
            break
        candidates = generate_candidate_actions(
            case,
            trial,
            remaining_blocks=remaining,
            mode="semantic",
            max_per_primitive=8,
        )
        if not candidates:
            chosen = _forced_progress_action(case, trial, remaining[0])
            forced_count += 1
        else:
            scored = []
            for candidate in candidates[: max(max_candidates * 4, max_candidates)]:
                policy_score, _components = _score_hierarchical_action(
                    case, policy, role_evidence, trial, candidate
                )
                scored.append((policy_score, candidate))
            scored.sort(key=lambda item: item[0], reverse=True)
            chosen = scored[0][1]
            if no_progress >= 2 and chosen.block_index in trial.semantic_placed:
                chosen = _forced_progress_action(case, trial, remaining[0])
                forced_count += 1
        before = len(trial.semantic_placed)
        executor.apply(trial, chosen)
        no_progress = 0 if len(trial.semantic_placed) > before else no_progress + 1
        actions_taken += 1
    quality = _quality_from_positions(case, trial.proposed_positions)
    return quality, {
        "continuation_actions_taken": actions_taken - 1,
        "forced_continuation_count": forced_count,
        "completed": len(trial.semantic_placed) >= case.block_count,
        "semantic_placed_fraction": len(trial.semantic_placed) / max(case.block_count, 1),
    }

def _collect_pools(args_dict: dict[str, Any]) -> dict[str, Any]:
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"
    torch.set_num_threads(1)

    case_id = int(args_dict["case_id"])
    max_steps = int(args_dict["max_steps"])
    max_candidates = int(args_dict["max_candidates_per_step"])
    feature_mode = str(args_dict["feature_mode"])
    label_kind = str(args_dict.get("label_kind", "immediate"))
    continuation_horizon = int(args_dict.get("continuation_horizon", 8))
    job = RolloutJob(
        case_id=case_id,
        seed=int(args_dict["seed"]),
        hidden_dim=int(args_dict["hidden_dim"]),
        epochs=int(args_dict["policy_epochs"]),
        lr=float(args_dict["lr"]),
        primitive_set_weight=float(args_dict["primitive_set_weight"]),
        block_weight=float(args_dict["block_weight"]),
        encoder_kind=str(args_dict["encoder_kind"]),
    )
    cases = load_validation_cases(case_limit=case_id + 1)
    case = cases[case_id]
    dataset = build_bc_dataset_from_cases([case], max_traces_per_case=1)
    policy, training_summary, _primitive_sets = _train_hierarchical_policy(dataset, job)
    role_evidence = label_case_roles(case)
    state = ExecutionState(last_rollout_mode="semantic")
    executor = ActionExecutor(case)
    seed = _seed_first_action(
        case,
        [idx for idx in range(case.block_count) if idx not in state.semantic_placed],
    )
    executor.apply(state, seed)

    pools: list[dict[str, Any]] = []
    no_progress = 0
    step = 0
    while (
        len(state.semantic_placed) < case.block_count
        and state.step < case.block_count * 4
        and step < max_steps
    ):
        remaining = [idx for idx in range(case.block_count) if idx not in state.semantic_placed]
        candidates = generate_candidate_actions(
            case,
            state,
            remaining_blocks=remaining,
            mode="semantic",
            max_per_primitive=8,
        )
        if not candidates:
            chosen = _forced_progress_action(case, state, remaining[0])
            executor.apply(state, chosen)
            step += 1
            continue
        scored = []
        for candidate in candidates:
            policy_score, components = _score_hierarchical_action(
                case, policy, role_evidence, state, candidate
            )
            scored.append((policy_score, candidate, components))
        scored.sort(key=lambda item: item[0], reverse=True)
        selected = scored[:max_candidates]
        feature_rows = _build_pool_features(case, state, selected, feature_mode=feature_mode)
        candidate_rows = []
        quality_values = []
        for policy_rank, (_policy_score, candidate, _components) in enumerate(selected, start=1):
            evaluated = _quality_after_action(case, state, candidate)
            immediate_quality = evaluated["quality"]
            rollout_meta = None
            if label_kind == "rollout_return":
                quality, rollout_meta = _rollout_return_after_action(
                    case,
                    state,
                    candidate,
                    policy=policy,
                    role_evidence=role_evidence,
                    horizon=continuation_horizon,
                    max_candidates=max_candidates,
                )
            else:
                quality = immediate_quality
            q = float(quality["quality_cost_runtime1"])
            quality_values.append(q)
            candidate_rows.append(
                {
                    "policy_rank": policy_rank,
                    "action_key": canonical_action_key(candidate),
                    "primitive": candidate.primitive.value,
                    "block_index": candidate.block_index,
                    "target_index": candidate.target_index,
                    "source": candidate.metadata.get("source"),
                    "intent_type": candidate.metadata.get("intent_type"),
                    "quality_cost_runtime1": q,
                    "label_kind": label_kind,
                    "immediate_quality_cost_runtime1": float(immediate_quality["quality_cost_runtime1"]),
                    "HPWLgap": float(quality["HPWLgap"]),
                    "Areagap_bbox": float(quality["Areagap_bbox"]),
                    "Violationsrelative": float(quality["Violationsrelative"]),
                    "immediate_HPWLgap": float(immediate_quality["HPWLgap"]),
                    "immediate_Areagap_bbox": float(immediate_quality["Areagap_bbox"]),
                    "immediate_Violationsrelative": float(immediate_quality["Violationsrelative"]),
                    "rollout_meta": rollout_meta,
                    "feature": feature_rows[policy_rank - 1],
                }
            )
        oracle_index = min(range(len(quality_values)), key=lambda idx: quality_values[idx])
        pools.append(
            {
                "case_id": str(case.case_id),
                "case_index": case_id,
                "policy_seed": int(args_dict["seed"]),
                "step": step,
                "feature_rows": feature_rows,
                "quality_values": quality_values,
                "oracle_index": oracle_index,
                "candidate_rows": candidate_rows,
            }
        )
        chosen = scored[0][1]
        if no_progress >= 2 and chosen.block_index in state.semantic_placed:
            chosen = _forced_progress_action(case, state, remaining[0])
        before = len(state.semantic_placed)
        executor.apply(state, chosen)
        no_progress = 0 if len(state.semantic_placed) > before else no_progress + 1
        step += 1
    return {
        "case_id": str(case.case_id),
        "case_index": case_id,
        "policy_seed": int(args_dict["seed"]),
        "training_summary": training_summary,
        "pools": pools,
    }


def _train_ranker(
    pools: list[dict[str, Any]],
    *,
    epochs: int,
    lr: float,
    seed: int,
    ranker_kind: str = "scalar",
    component_loss_weight: float = 0.5,
    target_kind: str = "oracle_ce",
    quality_temperature: float = 0.5,
    pairwise_loss_weight_kind: str = "quality_delta",
    pairwise_listwise_loss_weight: float = 0.0,
    hybrid_scalar_loss_weight: float = 1.0,
    relational_pairwise_loss_weight: float = 0.5,
) -> tuple[torch.nn.Module, dict[str, list[float]]]:
    if not pools:
        raise ValueError("cannot train CandidateQualityRanker with zero pools")
    torch.manual_seed(seed)
    feature_dim = len(pools[0]["feature_rows"][0])
    if ranker_kind == "late_fusion":
        ranker = CandidateLateFusionRanker(feature_dim=feature_dim, hidden_dim=64, num_heads=4)
    elif ranker_kind == "relational_action_q":
        ranker = CandidateRelationalActionQRanker(
            feature_dim=feature_dim,
            hidden_dim=64,
            num_heads=4,
        )
    elif ranker_kind == "constraint_token_action_q":
        ranker = CandidateConstraintTokenRanker(
            feature_dim=feature_dim,
            hidden_dim=64,
            num_heads=4,
            constraint_feature_count=len(CONSTRAINT_RELATION_EXTRA_FEATURE_NAMES),
        )
    elif ranker_kind in {"pairwise_set", "hybrid_set"}:
        ranker = CandidateSetPairwiseRanker(feature_dim=feature_dim, hidden_dim=64, num_heads=4)
    elif ranker_kind == "component":
        ranker = CandidateComponentRanker(
            feature_dim=feature_dim,
            hidden_dim=64,
            component_count=len(OBJECTIVE_COMPONENTS),
        )
    else:
        ranker = CandidateQualityRanker(feature_dim=feature_dim, hidden_dim=64)
    all_features = torch.tensor(
        [feature for pool in pools for feature in pool["feature_rows"]],
        dtype=torch.float32,
    )
    mean = all_features.mean(dim=0)
    std = all_features.std(dim=0).clamp_min(1e-6)
    optimizer = torch.optim.Adam(ranker.parameters(), lr=lr)
    for _epoch in range(epochs):
        for pool in pools:
            features = torch.tensor(pool["feature_rows"], dtype=torch.float32)
            features = (features - mean) / std
            target = torch.tensor([int(pool["oracle_index"])], dtype=torch.long)
            quality_values = torch.tensor(pool["quality_values"], dtype=torch.float32)
            if target_kind == "topk_quality":
                topk_count = min(3, int(quality_values.numel()))
                topk_indices = torch.argsort(quality_values)[:topk_count]
                soft_target = torch.zeros_like(quality_values)
                soft_target[topk_indices] = 1.0 / max(topk_count, 1)
            else:
                soft_target = torch.softmax(
                    -(quality_values - quality_values.min()) / max(float(quality_temperature), 1e-6),
                    dim=0,
                )
            optimizer.zero_grad(set_to_none=True)
            if ranker_kind in {"pairwise_set", "hybrid_set", "late_fusion"}:
                pair_logits = (
                    ranker.pairwise_ranker.pair_logits(features)
                    if ranker_kind == "late_fusion"
                    else ranker.pair_logits(features)
                )
                pair_targets = (quality_values.unsqueeze(1) < quality_values.unsqueeze(0)).float()
                if pairwise_loss_weight_kind == "uniform":
                    pair_weight = torch.ones_like(pair_targets)
                else:
                    pair_weight = (quality_values.unsqueeze(1) - quality_values.unsqueeze(0)).abs()
                    pair_weight = pair_weight / pair_weight.mean().clamp_min(1e-6)
                pair_mask = ~torch.eye(
                    pair_logits.shape[0],
                    dtype=torch.bool,
                    device=pair_logits.device,
                )
                loss_raw = torch.nn.functional.binary_cross_entropy_with_logits(
                    pair_logits[pair_mask],
                    pair_targets[pair_mask],
                    reduction="none",
                )
                loss = (loss_raw * pair_weight[pair_mask]).mean()
                scalar_loss_weight = (
                    hybrid_scalar_loss_weight
                    if ranker_kind == "hybrid_set"
                    else pairwise_listwise_loss_weight
                )
                if ranker_kind == "late_fusion":
                    scalar_loss_weight = hybrid_scalar_loss_weight
                if scalar_loss_weight > 0:
                    if ranker_kind in {"late_fusion", "hybrid_set"}:
                        pool_scores = ranker.scalar_scores(features)
                    else:
                        pool_scores = ranker.score_candidates(features)
                    if target_kind in {"soft_quality", "topk_quality"}:
                        listwise_loss = -(
                            soft_target * torch.nn.functional.log_softmax(pool_scores, dim=0)
                        ).sum()
                    else:
                        listwise_loss = torch.nn.functional.cross_entropy(
                            pool_scores.unsqueeze(0),
                            target,
                        )
                    loss = loss + scalar_loss_weight * listwise_loss
            elif ranker_kind in {"relational_action_q", "constraint_token_action_q"}:
                scores = ranker.score_candidates(features)
                if target_kind in {"soft_quality", "topk_quality"}:
                    loss = -(soft_target * torch.nn.functional.log_softmax(scores, dim=0)).sum()
                else:
                    loss = torch.nn.functional.cross_entropy(scores.unsqueeze(0), target)
                if relational_pairwise_loss_weight > 0 and features.shape[0] > 1:
                    pair_logits = ranker.pair_logits(features)
                    pair_targets = (quality_values.unsqueeze(1) < quality_values.unsqueeze(0)).float()
                    if pairwise_loss_weight_kind == "uniform":
                        pair_weight = torch.ones_like(pair_targets)
                    else:
                        pair_weight = (quality_values.unsqueeze(1) - quality_values.unsqueeze(0)).abs()
                        pair_weight = pair_weight / pair_weight.mean().clamp_min(1e-6)
                    pair_mask = ~torch.eye(
                        pair_logits.shape[0],
                        dtype=torch.bool,
                        device=pair_logits.device,
                    )
                    pair_loss_raw = torch.nn.functional.binary_cross_entropy_with_logits(
                        pair_logits[pair_mask],
                        pair_targets[pair_mask],
                        reduction="none",
                    )
                    loss = loss + relational_pairwise_loss_weight * (
                        pair_loss_raw * pair_weight[pair_mask]
                    ).mean()
            elif ranker_kind == "component":
                overall_logits, component_logits = ranker(features)
                if target_kind in {"soft_quality", "topk_quality"}:
                    loss = -(
                        soft_target * torch.nn.functional.log_softmax(overall_logits, dim=0)
                    ).sum()
                else:
                    loss = torch.nn.functional.cross_entropy(overall_logits.unsqueeze(0), target)
                component_losses = []
                for component_idx, component in enumerate(OBJECTIVE_COMPONENTS):
                    component_target = min(
                        range(len(pool["candidate_rows"])),
                        key=lambda idx: float(pool["candidate_rows"][idx][component]),
                    )
                    component_losses.append(
                        torch.nn.functional.cross_entropy(
                            component_logits[:, component_idx].unsqueeze(0),
                            torch.tensor([component_target], dtype=torch.long),
                        )
                    )
                loss = loss + component_loss_weight * torch.stack(component_losses).mean()
            else:
                logits = ranker(features)
                if target_kind in {"soft_quality", "topk_quality"}:
                    loss = -(soft_target * torch.nn.functional.log_softmax(logits, dim=0)).sum()
                else:
                    loss = torch.nn.functional.cross_entropy(logits.unsqueeze(0), target)
            loss.backward()
            optimizer.step()
    return ranker, {"mean": mean.tolist(), "std": std.tolist()}


def _normalize_pools_for_ranker(
    pools: list[dict[str, Any]],
    *,
    mode: str,
) -> list[dict[str, Any]]:
    if mode == "global":
        return pools
    if mode != "per_case":
        raise ValueError(f"unknown feature normalization mode: {mode}")
    by_case: dict[int, list[dict[str, Any]]] = {}
    for pool in pools:
        by_case.setdefault(int(pool["case_index"]), []).append(pool)
    normalized: list[dict[str, Any]] = []
    for case_pools in by_case.values():
        rows = [feature for pool in case_pools for feature in pool["feature_rows"]]
        features = torch.tensor(rows, dtype=torch.float32)
        mean = features.mean(dim=0)
        std = features.std(dim=0).clamp_min(1e-6)
        for pool in case_pools:
            pool_copy = dict(pool)
            pool_features = torch.tensor(pool["feature_rows"], dtype=torch.float32)
            pool_copy["feature_rows"] = ((pool_features - mean) / std).tolist()
            normalized.append(pool_copy)
    return normalized


def _ranker_scores(
    ranker: torch.nn.Module,
    features: torch.Tensor,
    *,
    ranker_kind: str,
    component_score_weight: float,
    hybrid_pairwise_score_weight: float,
) -> torch.Tensor:
    if ranker_kind in {"relational_action_q", "constraint_token_action_q"}:
        return ranker.score_candidates(features)
    if ranker_kind in {"hybrid_set", "late_fusion"}:
        return ranker.hybrid_scores(features, pairwise_weight=hybrid_pairwise_score_weight)
    if ranker_kind == "pairwise_set":
        return ranker.score_candidates(features)
    if ranker_kind == "component":
        overall_logits, component_logits = ranker(features)
        return overall_logits + component_score_weight * component_logits.mean(dim=1)
    return ranker(features)


def _evaluate_ranker(
    ranker: torch.nn.Module,
    pools: list[dict[str, Any]],
    feature_stats: dict[str, list[float]],
    *,
    ranker_kind: str = "scalar",
    component_score_weight: float = 0.25,
    hybrid_pairwise_score_weight: float = 0.5,
) -> dict[str, Any]:
    if not pools:
        return {
            "pool_count": 0,
            "mean_selected_quality_rank": 0.0,
            "mean_selected_quality_regret": 0.0,
            "oracle_top1_selected_fraction": 0.0,
            "objective_component_regret": {
                component: 0.0 for component in OBJECTIVE_COMPONENTS
            },
            "rows": [],
        }
    mean = torch.tensor(feature_stats["mean"], dtype=torch.float32)
    std = torch.tensor(feature_stats["std"], dtype=torch.float32)
    ranks = []
    regrets = []
    top1 = 0
    rows = []
    for pool in pools:
        features = torch.tensor(pool["feature_rows"], dtype=torch.float32)
        features = (features - mean) / std
        scores = _ranker_scores(
            ranker,
            features,
            ranker_kind=ranker_kind,
            component_score_weight=component_score_weight,
            hybrid_pairwise_score_weight=hybrid_pairwise_score_weight,
        ).detach()
        pred_index = int(scores.argmax().item())
        ordered_by_quality = sorted(
            range(len(pool["quality_values"])), key=lambda idx: float(pool["quality_values"][idx])
        )
        rank = ordered_by_quality.index(pred_index) + 1
        regret = float(pool["quality_values"][pred_index]) - float(
            pool["quality_values"][ordered_by_quality[0]]
        )
        ranks.append(rank)
        regrets.append(regret)
        top1 += int(rank == 1)
        rows.append(
            {
                "case_id": pool["case_id"],
                "step": pool["step"],
                "predicted_index": pred_index,
                "oracle_index": int(pool["oracle_index"]),
                "predicted_quality_rank": rank,
                "predicted_quality_regret": regret,
                "predicted_action": pool["candidate_rows"][pred_index],
                "oracle_action": pool["candidate_rows"][int(pool["oracle_index"])],
            }
        )
    denom = max(len(pools), 1)
    objective_regret = {}
    for component in OBJECTIVE_COMPONENTS:
        objective_regret[component] = sum(
            float(row["predicted_action"].get(component, 0.0))
            - float(row["oracle_action"].get(component, 0.0))
            for row in rows
        ) / denom
    return {
        "pool_count": len(pools),
        "mean_selected_quality_rank": sum(ranks) / denom,
        "mean_selected_quality_regret": sum(regrets) / denom,
        "oracle_top1_selected_fraction": top1 / denom,
        "objective_component_regret": objective_regret,
        "rows": rows,
    }


def _feature_shift(
    train_pools: list[dict[str, Any]],
    eval_pools: list[dict[str, Any]],
    *,
    feature_names: list[str],
) -> dict[str, Any]:
    train_features = torch.tensor(
        [feature for pool in train_pools for feature in pool["feature_rows"]],
        dtype=torch.float32,
    )
    eval_features = torch.tensor(
        [feature for pool in eval_pools for feature in pool["feature_rows"]],
        dtype=torch.float32,
    )
    train_mean = train_features.mean(dim=0)
    train_std = train_features.std(dim=0).clamp_min(1e-6)
    eval_mean = eval_features.mean(dim=0)
    standardized_delta = (eval_mean - train_mean) / train_std
    top = sorted(
        (
            {
                "feature": feature_names[idx],
                "train_mean": float(train_mean[idx].item()),
                "eval_mean": float(eval_mean[idx].item()),
                "standardized_mean_delta": float(standardized_delta[idx].item()),
                "abs_standardized_mean_delta": float(abs(standardized_delta[idx]).item()),
            }
            for idx in range(len(feature_names))
        ),
        key=lambda item: item["abs_standardized_mean_delta"],
        reverse=True,
    )
    return {
        "train_candidate_count": int(train_features.shape[0]),
        "eval_candidate_count": int(eval_features.shape[0]),
        "max_abs_standardized_mean_delta": top[0]["abs_standardized_mean_delta"] if top else 0.0,
        "top_features": top[:8],
    }


def _split_gate(evaluation: dict[str, Any]) -> dict[str, Any]:
    gate = {
        "mean_selected_quality_rank_lt_4": evaluation["mean_selected_quality_rank"] < 4.0,
        "oracle_top1_selected_fraction_gt_0_30": (
            evaluation["oracle_top1_selected_fraction"] > 0.30
        ),
    }
    gate["pass"] = all(bool(value) for value in gate.values())
    return gate


def _run_split(
    pools: list[dict[str, Any]],
    *,
    train_case_ids: set[int],
    eval_case_ids: set[int],
    args: argparse.Namespace,
) -> dict[str, Any]:
    train_pools = [pool for pool in pools if int(pool["case_index"]) in train_case_ids]
    eval_pools = [pool for pool in pools if int(pool["case_index"]) in eval_case_ids]
    if not train_pools:
        raise ValueError(f"train split has zero pools for case ids {sorted(train_case_ids)}")
    if not eval_pools:
        raise ValueError(f"eval split has zero pools for case ids {sorted(eval_case_ids)}")
    train_pools = _normalize_pools_for_ranker(train_pools, mode=str(args.feature_normalization))
    eval_pools = _normalize_pools_for_ranker(eval_pools, mode=str(args.feature_normalization))
    ranker, feature_stats = _train_ranker(
        train_pools,
        epochs=int(args.ranker_epochs),
        lr=float(args.ranker_lr),
        seed=int(args.seed),
        ranker_kind=str(args.ranker_kind),
        component_loss_weight=float(args.component_loss_weight),
        target_kind=str(args.target_kind),
        quality_temperature=float(args.quality_temperature),
        pairwise_loss_weight_kind=str(args.pairwise_loss_weight_kind),
        pairwise_listwise_loss_weight=float(args.pairwise_listwise_loss_weight),
        hybrid_scalar_loss_weight=float(args.hybrid_scalar_loss_weight),
        relational_pairwise_loss_weight=float(args.relational_pairwise_loss_weight),
    )
    train_evaluation = _evaluate_ranker(
        ranker,
        train_pools,
        feature_stats,
        ranker_kind=str(args.ranker_kind),
        component_score_weight=float(args.component_score_weight),
        hybrid_pairwise_score_weight=float(args.hybrid_pairwise_score_weight),
    )
    evaluation = _evaluate_ranker(
        ranker,
        eval_pools,
        feature_stats,
        ranker_kind=str(args.ranker_kind),
        component_score_weight=float(args.component_score_weight),
        hybrid_pairwise_score_weight=float(args.hybrid_pairwise_score_weight),
    )
    return {
        "train_case_ids": sorted(train_case_ids),
        "eval_case_ids": sorted(eval_case_ids),
        "train_pool_count": len(train_pools),
        "eval_pool_count": len(eval_pools),
        "feature_stats": feature_stats,
        "feature_shift": _feature_shift(
            train_pools,
            eval_pools,
            feature_names=_feature_names(str(args.feature_mode)),
        ),
        "train_evaluation": train_evaluation,
        "evaluation": evaluation,
        "heldout_gate": _split_gate(evaluation),
    }


def _aggregate_loco_results(split_results: list[dict[str, Any]]) -> dict[str, Any]:
    denom = max(len(split_results), 1)
    component_regret = {}
    for component in OBJECTIVE_COMPONENTS:
        component_regret[component] = sum(
            float(
                result["evaluation"]["objective_component_regret"].get(
                    component,
                    0.0,
                )
            )
            for result in split_results
        ) / denom
    evaluation = {
        "split_count": len(split_results),
        "mean_selected_quality_rank": sum(
            float(result["evaluation"]["mean_selected_quality_rank"])
            for result in split_results
        )
        / denom,
        "mean_selected_quality_regret": sum(
            float(result["evaluation"]["mean_selected_quality_regret"])
            for result in split_results
        )
        / denom,
        "oracle_top1_selected_fraction": sum(
            float(result["evaluation"]["oracle_top1_selected_fraction"])
            for result in split_results
        )
        / denom,
        "objective_component_regret": component_regret,
    }
    gate = {
        "mean_selected_quality_rank_lt_4": evaluation["mean_selected_quality_rank"] < 4.0,
        "oracle_top1_selected_fraction_gt_0_30": (
            evaluation["oracle_top1_selected_fraction"] > 0.30
        ),
        "no_individual_oracle_top1_zero": all(
            float(result["evaluation"]["oracle_top1_selected_fraction"]) > 0.0
            for result in split_results
        ),
    }
    gate["pass"] = all(bool(value) for value in gate.values())
    return {"evaluation": evaluation, "leave_one_case_out_gate": gate}


def _aggregate(collections: list[dict[str, Any]], workers: int, args: argparse.Namespace) -> dict[str, Any]:
    pools = [pool for collection in collections for pool in collection["pools"]]
    collected_case_ids = {int(collection["case_index"]) for collection in collections}
    collected_policy_seeds = {int(collection.get("policy_seed", args.seed)) for collection in collections}
    if getattr(args, "leave_one_case_out", False):
        split_results = []
        for eval_case_id in sorted(collected_case_ids):
            split_results.append(
                _run_split(
                    pools,
                    train_case_ids=collected_case_ids - {eval_case_id},
                    eval_case_ids={eval_case_id},
                    args=args,
                )
            )
        loco = _aggregate_loco_results(split_results)
        return {
            "status": "complete",
            "purpose": "Step6C hierarchical action-Q leave-one-case-out generalization audit",
            "evaluation_mode": "leave_one_case_out",
            "feature_mode": str(args.feature_mode),
            "feature_normalization": str(args.feature_normalization),
            "ranker_kind": str(args.ranker_kind),
            "encoder_kind": str(args.encoder_kind),
            "label_kind": str(getattr(args, "label_kind", "immediate")),
            "continuation_horizon": int(getattr(args, "continuation_horizon", 8)),
            "component_loss_weight": float(args.component_loss_weight),
            "component_score_weight": float(args.component_score_weight),
            "target_kind": str(args.target_kind),
            "label_kind": str(getattr(args, "label_kind", "immediate")),
            "quality_temperature": float(args.quality_temperature),
            "pairwise_loss_weight_kind": str(args.pairwise_loss_weight_kind),
            "pairwise_listwise_loss_weight": float(args.pairwise_listwise_loss_weight),
            "hybrid_scalar_loss_weight": float(args.hybrid_scalar_loss_weight),
            "hybrid_pairwise_score_weight": float(args.hybrid_pairwise_score_weight),
            "relational_pairwise_loss_weight": float(args.relational_pairwise_loss_weight),
            "feature_names": _feature_names(str(args.feature_mode)),
            "case_count": len(collected_case_ids),
            "collection_count": len(collections),
            "policy_seeds": sorted(collected_policy_seeds),
            "pool_count": len(pools),
            "workers_requested": workers,
            "ranker_epochs": int(args.ranker_epochs),
            "train_case_ids": None,
            "eval_case_ids": None,
            "train_pool_count": None,
            "eval_pool_count": None,
            "feature_stats": None,
            "train_evaluation": None,
            "evaluation": loco["evaluation"],
            "heldout_gate": None,
            "leave_one_case_out_gate": loco["leave_one_case_out_gate"],
            "split_results": split_results,
            "collections": sorted(
                collections,
                key=lambda item: (int(item["case_index"]), int(item.get("policy_seed", 0))),
            ),
        }

    train_case_ids = set(args.train_case_ids or [])
    eval_case_ids = set(args.eval_case_ids or [])
    is_split = bool(train_case_ids or eval_case_ids)
    if eval_case_ids and not train_case_ids:
        train_case_ids = collected_case_ids - eval_case_ids
    elif train_case_ids and not eval_case_ids:
        eval_case_ids = collected_case_ids - train_case_ids

    if is_split:
        split_result = _run_split(
            pools,
            train_case_ids=train_case_ids,
            eval_case_ids=eval_case_ids,
            args=args,
        )
        train_pools = [pool for pool in pools if int(pool["case_index"]) in train_case_ids]
        eval_pools = [pool for pool in pools if int(pool["case_index"]) in eval_case_ids]
        feature_stats = split_result["feature_stats"]
        train_evaluation = split_result["train_evaluation"]
        evaluation = split_result["evaluation"]
        heldout_gate = split_result["heldout_gate"]
        feature_shift = split_result["feature_shift"]
    else:
        train_pools = pools
        eval_pools = pools
        ranker, feature_stats = _train_ranker(
            _normalize_pools_for_ranker(
                train_pools,
                mode=str(args.feature_normalization),
            ),
            epochs=int(args.ranker_epochs),
            lr=float(args.ranker_lr),
            seed=int(args.seed),
            ranker_kind=str(args.ranker_kind),
            component_loss_weight=float(args.component_loss_weight),
            target_kind=str(args.target_kind),
            quality_temperature=float(args.quality_temperature),
            pairwise_loss_weight_kind=str(args.pairwise_loss_weight_kind),
            pairwise_listwise_loss_weight=float(args.pairwise_listwise_loss_weight),
            hybrid_scalar_loss_weight=float(args.hybrid_scalar_loss_weight),
            relational_pairwise_loss_weight=float(args.relational_pairwise_loss_weight),
        )
        train_evaluation = _evaluate_ranker(
            ranker,
            _normalize_pools_for_ranker(
                train_pools,
                mode=str(args.feature_normalization),
            ),
            feature_stats,
            ranker_kind=str(args.ranker_kind),
            component_score_weight=float(args.component_score_weight),
            hybrid_pairwise_score_weight=float(args.hybrid_pairwise_score_weight),
        )
        evaluation = _evaluate_ranker(
            ranker,
            _normalize_pools_for_ranker(
                eval_pools,
                mode=str(args.feature_normalization),
            ),
            feature_stats,
            ranker_kind=str(args.ranker_kind),
            component_score_weight=float(args.component_score_weight),
            hybrid_pairwise_score_weight=float(args.hybrid_pairwise_score_weight),
        )
        heldout_gate = None
        feature_shift = None
    return {
        "status": "complete",
        "purpose": (
            "Step6C hierarchical action-Q candidate-pool held-out audit"
            if is_split
            else "Step6C hierarchical action-Q candidate-pool micro-overfit audit"
        ),
        "evaluation_mode": "heldout_split" if is_split else "micro_overfit",
        "feature_mode": str(args.feature_mode),
        "feature_normalization": str(args.feature_normalization),
        "ranker_kind": str(args.ranker_kind),
        "encoder_kind": str(args.encoder_kind),
        "component_loss_weight": float(args.component_loss_weight),
        "component_score_weight": float(args.component_score_weight),
        "target_kind": str(args.target_kind),
        "label_kind": str(getattr(args, "label_kind", "immediate")),
        "quality_temperature": float(args.quality_temperature),
        "pairwise_loss_weight_kind": str(args.pairwise_loss_weight_kind),
        "pairwise_listwise_loss_weight": float(args.pairwise_listwise_loss_weight),
        "hybrid_scalar_loss_weight": float(args.hybrid_scalar_loss_weight),
        "hybrid_pairwise_score_weight": float(args.hybrid_pairwise_score_weight),
        "relational_pairwise_loss_weight": float(args.relational_pairwise_loss_weight),
        "feature_names": _feature_names(str(args.feature_mode)),
        "feature_stats": feature_stats,
        "case_count": len(collected_case_ids),
        "collection_count": len(collections),
        "policy_seeds": sorted(collected_policy_seeds),
        "pool_count": len(pools),
        "train_case_ids": sorted(train_case_ids) if is_split else None,
        "eval_case_ids": sorted(eval_case_ids) if is_split else None,
        "train_pool_count": len(train_pools),
        "eval_pool_count": len(eval_pools),
        "workers_requested": workers,
        "ranker_epochs": int(args.ranker_epochs),
        "train_evaluation": train_evaluation,
        "evaluation": evaluation,
        "heldout_gate": heldout_gate,
        "leave_one_case_out_gate": None,
        "feature_shift": feature_shift,
        "collections": sorted(
            collections,
            key=lambda item: (int(item["case_index"]), int(item.get("policy_seed", 0))),
        ),
    }


def _write_markdown(path: Path, payload: dict[str, Any]) -> None:
    ev = payload["evaluation"]
    train_ev = payload["train_evaluation"]
    is_split = payload["evaluation_mode"] == "heldout_split"
    is_loco = payload["evaluation_mode"] == "leave_one_case_out"
    lines = [
        "# Step6C Hierarchical Action-Q Audit",
        "",
        f"- purpose: `{payload['purpose']}`",
        f"- evaluation mode: `{payload['evaluation_mode']}`",
        f"- feature mode: `{payload['feature_mode']}`",
        f"- feature normalization: `{payload['feature_normalization']}`",
        f"- encoder kind: `{payload['encoder_kind']}`",
        f"- ranker kind: `{payload['ranker_kind']}`",
        f"- target kind: `{payload['target_kind']}`",
        f"- label kind: `{payload.get('label_kind', 'immediate')}`",
        f"- pairwise loss weight: `{payload['pairwise_loss_weight_kind']}`",
        f"- pairwise listwise loss weight: `{payload['pairwise_listwise_loss_weight']}`",
        f"- hybrid scalar loss weight: `{payload['hybrid_scalar_loss_weight']}`",
        f"- hybrid pairwise score weight: `{payload['hybrid_pairwise_score_weight']}`",
        f"- relational pairwise loss weight: `{payload['relational_pairwise_loss_weight']}`",
        f"- cases: `{payload['case_count']}`",
        f"- policy seeds: `{payload['policy_seeds']}`",
        f"- collections: `{payload['collection_count']}`",
        f"- pools: `{payload['pool_count']}`",
        f"- workers requested: `{payload['workers_requested']}`",
        f"- ranker epochs: `{payload['ranker_epochs']}`",
    ]
    if is_loco:
        gate = payload["leave_one_case_out_gate"]
        lines.extend(
            [
                f"- split count: `{ev['split_count']}`",
                f"- mean held-out quality rank: `{ev['mean_selected_quality_rank']:.4f}`",
                f"- mean held-out quality regret: `{ev['mean_selected_quality_regret']:.4f}`",
                f"- mean held-out oracle-top1 fraction: `{ev['oracle_top1_selected_fraction']:.4f}`",
                "",
                "## Leave-One-Case-Out Gate",
                "",
                f"- mean selected quality rank < 4.0: `{gate['mean_selected_quality_rank_lt_4']}`",
                "- mean oracle-top1 selected fraction > 0.30: "
                f"`{gate['oracle_top1_selected_fraction_gt_0_30']}`",
                f"- no individual oracle-top1 fraction is 0.0: `{gate['no_individual_oracle_top1_zero']}`",
                f"- pass: `{gate['pass']}`",
                "",
                "## Per-Held-Out Case",
                "",
            ]
        )
        for result in payload["split_results"]:
            eval_id = result["eval_case_ids"][0]
            result_ev = result["evaluation"]
            shift = result["feature_shift"]
            component_regret = result_ev["objective_component_regret"]
            top_shift = ", ".join(
                f"{item['feature']}={item['standardized_mean_delta']:.2f}"
                for item in shift["top_features"][:3]
            )
            lines.extend(
                [
                    f"- validation-{eval_id}: rank `{result_ev['mean_selected_quality_rank']:.4f}`, "
                    f"top1 `{result_ev['oracle_top1_selected_fraction']:.4f}`, "
                    f"regret `{result_ev['mean_selected_quality_regret']:.4f}`, "
                    f"HPWL delta `{component_regret['HPWLgap']:.4f}`, "
                    f"area delta `{component_regret['Areagap_bbox']:.4f}`, "
                    f"violation delta `{component_regret['Violationsrelative']:.4f}`, "
                    f"top feature shift `{top_shift}`",
                ]
            )
        lines.extend(
            [
                "",
                "Interpretation: this is a leave-one-case-out action-Q "
                "generalization diagnostic. It tests whether the candidate "
                "features and ranker transfer across validation cases, and "
                "reports feature shift plus objective-component regret for the "
                "held-out predictions.",
            ]
        )
    else:
        lines.extend(
            [
                f"- train pools: `{payload['train_pool_count']}`",
                f"- eval pools: `{payload['eval_pool_count']}`",
                f"- train mean selected quality rank: `{train_ev['mean_selected_quality_rank']:.4f}`",
                f"- train oracle-top1 selected fraction: `{train_ev['oracle_top1_selected_fraction']:.4f}`",
                f"- eval mean selected quality rank: `{ev['mean_selected_quality_rank']:.4f}`",
                f"- eval mean selected quality regret: `{ev['mean_selected_quality_regret']:.4f}`",
                f"- eval oracle-top1 selected fraction: `{ev['oracle_top1_selected_fraction']:.4f}`",
                "",
            ]
        )
    if is_split:
        gate = payload["heldout_gate"]
        lines.extend(
            [
                "## Held-Out Gate",
                "",
                f"- mean selected quality rank < 4.0: `{gate['mean_selected_quality_rank_lt_4']}`",
                "- oracle-top1 selected fraction > 0.30: "
                f"`{gate['oracle_top1_selected_fraction_gt_0_30']}`",
                f"- pass: `{gate['pass']}`",
                "",
                "Interpretation: this is a held-out action-Q/ranking diagnostic. "
                "It tests whether the current candidate features and ranker "
                "generalize beyond the case pools used for ranker training. It "
                "does not modify contest runtime, repair/finalizer behavior, "
                "reranker settings, or proxy weights.",
            ]
        )
        if payload["feature_shift"] is not None:
            lines.extend(["", "## Feature Shift", ""])
            for item in payload["feature_shift"]["top_features"]:
                lines.append(
                    f"- `{item['feature']}`: standardized mean delta "
                    f"`{item['standardized_mean_delta']:.4f}`"
                )
            component_regret = ev["objective_component_regret"]
            lines.extend(
                [
                    "",
                    "## Objective Component Regret",
                    "",
                    f"- HPWLgap delta: `{component_regret['HPWLgap']:.4f}`",
                    f"- Areagap_bbox delta: `{component_regret['Areagap_bbox']:.4f}`",
                    f"- Violationsrelative delta: `{component_regret['Violationsrelative']:.4f}`",
                ]
            )
    elif not is_loco:
        lines.append(
            "Interpretation: this is a micro-overfit action-Q/ranking diagnostic. It "
            "shows whether the current hierarchical action representation can express "
            "quality preferences over candidate pools. It does not modify contest "
            "runtime, repair/finalizer behavior, reranker settings, or proxy weights."
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = _parse_args()
    requested_case_ids = set(int(case_id) for case_id in args.case_ids)
    if args.train_case_ids is not None:
        requested_case_ids.update(int(case_id) for case_id in args.train_case_ids)
    if args.eval_case_ids is not None:
        requested_case_ids.update(int(case_id) for case_id in args.eval_case_ids)
    policy_seeds = (
        [int(seed) for seed in args.policy_seeds]
        if args.policy_seeds is not None and len(args.policy_seeds) > 0
        else [int(args.seed)]
    )
    jobs = [
        {
            "case_id": int(case_id),
            "seed": int(seed),
            "hidden_dim": int(args.hidden_dim),
            "policy_epochs": int(args.policy_epochs),
            "lr": float(args.lr),
            "primitive_set_weight": float(args.primitive_set_weight),
            "block_weight": float(args.block_weight),
            "max_steps": int(args.max_steps),
            "max_candidates_per_step": int(args.max_candidates_per_step),
            "feature_mode": str(args.feature_mode),
            "feature_normalization": str(args.feature_normalization),
            "encoder_kind": str(args.encoder_kind),
            "label_kind": str(getattr(args, "label_kind", "immediate")),
            "continuation_horizon": int(getattr(args, "continuation_horizon", 8)),
        }
        for case_id in sorted(requested_case_ids)
        for seed in policy_seeds
    ]
    max_workers = max(1, min(int(args.workers), len(jobs)))
    collections = []
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(_collect_pools, job) for job in jobs]
        for future in as_completed(futures):
            collections.append(future.result())
    payload = _aggregate(collections, max_workers, args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    _write_markdown(args.output.with_suffix(".md"), payload)
    print(
        json.dumps(
            {
                "status": payload["status"],
                "case_count": payload["case_count"],
                "collection_count": payload["collection_count"],
                "policy_seeds": payload["policy_seeds"],
                "pool_count": payload["pool_count"],
                "evaluation_mode": payload["evaluation_mode"],
                "feature_mode": payload["feature_mode"],
                "ranker_kind": payload["ranker_kind"],
                "target_kind": payload["target_kind"],
                "label_kind": payload.get("label_kind", "immediate"),
                "pairwise_loss_weight_kind": payload["pairwise_loss_weight_kind"],
                "pairwise_listwise_loss_weight": payload["pairwise_listwise_loss_weight"],
                "hybrid_scalar_loss_weight": payload["hybrid_scalar_loss_weight"],
                "hybrid_pairwise_score_weight": payload["hybrid_pairwise_score_weight"],
                "relational_pairwise_loss_weight": payload["relational_pairwise_loss_weight"],
                "train_pool_count": payload["train_pool_count"],
                "eval_pool_count": payload["eval_pool_count"],
                "mean_selected_quality_rank": payload["evaluation"][
                    "mean_selected_quality_rank"
                ],
                "mean_selected_quality_regret": payload["evaluation"][
                    "mean_selected_quality_regret"
                ],
                "oracle_top1_selected_fraction": payload["evaluation"][
                    "oracle_top1_selected_fraction"
                ],
                "heldout_gate_pass": (
                    None
                    if payload["heldout_gate"] is None
                    else payload["heldout_gate"]["pass"]
                ),
                "leave_one_case_out_gate_pass": (
                    None
                    if payload["leave_one_case_out_gate"] is None
                    else payload["leave_one_case_out_gate"]["pass"]
                ),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
