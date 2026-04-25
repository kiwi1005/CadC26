from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

import torch

from puzzleplace.actions.executor import ExecutionState
from puzzleplace.actions.schema import ActionPrimitive, TypedAction, canonical_action_key
from puzzleplace.data import ConstraintColumns, FloorSetCase
from puzzleplace.geometry.boxes import pairwise_intersection_area
from puzzleplace.research.virtual_frame import (
    PuzzleFrame,
    boundary_frame_satisfaction_fraction,
)

ContactMode = Literal[
    "origin",
    "right",
    "left",
    "top",
    "bottom",
    "free_rect",
    "group_mate",
    "boundary",
    "pin_pull",
]
AnchorKind = Literal["origin", "placed_block", "free_rect", "boundary", "group", "pin_pull"]
FeatureMode = Literal["puzzle_pool_raw_safe", "puzzle_pool_normalized_relational"]
BoundaryCommitMode = Literal[
    "none",
    "prefer_predicted_hull",
    "require_predicted_hull_if_available",
    "require_virtual_frame_if_available",
]

ALLOWED_PUZZLE_INFERENCE_FIELDS = frozenset(
    {
        "block_index",
        "shape_bin_id",
        "exact_shape_flag",
        "site_id",
        "contact_mode",
        "anchor_kind",
        "candidate_family",
        "legality_status",
        "action_token",
        "normalized_features",
    }
)

TRAINING_ONLY_PUZZLE_FIELDS = frozenset(
    {"expert_label", "teacher_distance", "oracle_cost", "oracle_rank", "reference_xywh"}
)

DENIED_PUZZLE_INFERENCE_FIELDS = frozenset(
    {
        "target_positions",
        "target_xywh",
        "teacher_shape",
        "teacher_site",
        "teacher_contact",
        "oracle_score_components",
        "case_id",
        "case_index",
        "hard_case_selector",
        "raw_logits",
        "manual_delta_rows",
        "fallback_result",
        "repair_outcome",
    }
)


def _scan_denied_keys(value: Any, *, path: str = "payload") -> list[str]:
    denied: list[str] = []
    if isinstance(value, Mapping):
        for key, nested in value.items():
            key_text = str(key)
            nested_path = f"{path}.{key_text}"
            if key_text in DENIED_PUZZLE_INFERENCE_FIELDS:
                denied.append(nested_path)
            denied.extend(_scan_denied_keys(nested, path=nested_path))
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for idx, nested in enumerate(value):
            denied.extend(_scan_denied_keys(nested, path=f"{path}[{idx}]"))
    return denied


def _action_metadata_denied(action: TypedAction) -> list[str]:
    return _scan_denied_keys(action.metadata, path="action_token.metadata")


@dataclass(slots=True)
class PuzzleCandidateDescriptor:
    block_index: int
    shape_bin_id: int
    exact_shape_flag: bool
    site_id: int
    contact_mode: str
    anchor_kind: str
    candidate_family: str
    legality_status: str
    action_token: TypedAction
    normalized_features: torch.Tensor
    training_only: dict[str, Any] = field(default_factory=dict)

    def inference_payload(self) -> dict[str, Any]:
        return {
            "block_index": self.block_index,
            "shape_bin_id": self.shape_bin_id,
            "exact_shape_flag": self.exact_shape_flag,
            "site_id": self.site_id,
            "contact_mode": self.contact_mode,
            "anchor_kind": self.anchor_kind,
            "candidate_family": self.candidate_family,
            "legality_status": self.legality_status,
            "action_token": self.action_token,
            "normalized_features": self.normalized_features,
        }


def validate_puzzle_candidate_payload(payload: Mapping[str, Any]) -> None:
    """Fail closed on Step6G candidate/scorer payload violations."""

    keys = set(payload.keys())
    denied = sorted(keys & DENIED_PUZZLE_INFERENCE_FIELDS)
    nested_denied = sorted(_scan_denied_keys(payload))
    if denied or nested_denied:
        all_denied = sorted(set(denied + nested_denied))
        raise ValueError(f"denied puzzle candidate payload fields: {', '.join(all_denied)}")

    missing = sorted(ALLOWED_PUZZLE_INFERENCE_FIELDS - keys)
    if missing:
        raise ValueError(f"missing required puzzle candidate fields: {', '.join(missing)}")

    extras = sorted(keys - ALLOWED_PUZZLE_INFERENCE_FIELDS)
    if extras:
        raise ValueError(f"unsupported puzzle candidate fields: {', '.join(extras)}")

    action = payload["action_token"]
    if not isinstance(action, TypedAction):
        raise TypeError("action_token must be a TypedAction")
    metadata_denied = sorted(_action_metadata_denied(action))
    if metadata_denied:
        raise ValueError(f"denied puzzle action metadata fields: {', '.join(metadata_denied)}")

    features = payload["normalized_features"]
    if not isinstance(features, torch.Tensor):
        raise TypeError("normalized_features must be a torch.Tensor")
    if features.ndim != 1:
        raise ValueError("normalized_features must be a rank-1 tensor")


def validate_puzzle_descriptor(descriptor: PuzzleCandidateDescriptor) -> None:
    validate_puzzle_candidate_payload(descriptor.inference_payload())
    bad_training = sorted(set(descriptor.training_only) - TRAINING_ONLY_PUZZLE_FIELDS)
    if bad_training:
        raise ValueError(f"unsupported training-only fields: {', '.join(bad_training)}")


def masked_anchor_xywh(case: FloorSetCase) -> torch.Tensor:
    """Return inference-safe xywh: fixed/preplaced anchors only, soft rows hidden."""

    anchors = torch.full((case.block_count, 4), -1.0, dtype=torch.float32)
    if case.target_positions is None:
        return anchors
    for idx in range(case.block_count):
        fixed = bool(case.constraints[idx, ConstraintColumns.FIXED].item())
        preplaced = bool(case.constraints[idx, ConstraintColumns.PREPLACED].item())
        if fixed or preplaced:
            anchors[idx] = case.target_positions[idx].to(torch.float32)
    return anchors


def shape_bin_log_aspects(bin_count: int = 17, *, log_limit: float = 3.0) -> list[float]:
    limit = abs(float(log_limit))
    values = torch.linspace(-limit, limit, steps=bin_count).tolist()
    values.append(0.0)
    return sorted(set(round(float(v), 6) for v in values))


def _mib_exact_shape_candidates(
    case: FloorSetCase,
    state: ExecutionState,
    block_index: int,
    anchors: torch.Tensor,
) -> list[tuple[int, bool, float, float]]:
    mib_id = float(case.constraints[block_index, ConstraintColumns.MIB].item())
    if mib_id <= 0:
        return []

    area = max(float(case.area_targets[block_index].item()), 1e-6)
    reference_boxes: list[tuple[float, float]] = []
    for other_idx, (_x, _y, other_w, other_h) in state.placements.items():
        other_mib = float(case.constraints[other_idx, ConstraintColumns.MIB].item())
        if other_mib == mib_id and other_w > 0 and other_h > 0:
            reference_boxes.append((other_w, other_h))
    for other_idx in range(case.block_count):
        if other_idx == block_index:
            continue
        other_mib = float(case.constraints[other_idx, ConstraintColumns.MIB].item())
        if other_mib != mib_id:
            continue
        _ax, _ay, anchor_w, anchor_h = [float(v) for v in anchors[other_idx].tolist()]
        if anchor_w > 0 and anchor_h > 0:
            reference_boxes.append((anchor_w, anchor_h))

    rows: list[tuple[int, bool, float, float]] = []
    seen: set[tuple[float, float]] = set()
    for ref_w, ref_h in reference_boxes:
        ratio = max(ref_w / max(ref_h, 1e-6), 1e-6)
        width = math.sqrt(area * ratio)
        height = math.sqrt(area / ratio)
        key = (round(width, 6), round(height, 6))
        if key in seen:
            continue
        seen.add(key)
        rows.append((-2, True, width, height))
    return rows


def _shape_candidates(
    case: FloorSetCase,
    state: ExecutionState,
    block_index: int,
    anchors: torch.Tensor,
    *,
    max_shape_bins: int | None = None,
    virtual_frame: PuzzleFrame | None = None,
    log_aspect_limit: float = 3.0,
) -> list[tuple[int, bool, float, float]]:
    fixed = bool(case.constraints[block_index, ConstraintColumns.FIXED].item())
    preplaced = bool(case.constraints[block_index, ConstraintColumns.PREPLACED].item())
    ax, ay, aw, ah = [float(v) for v in anchors[block_index].tolist()]
    if (fixed or preplaced) and aw > 0 and ah > 0:
        return [(-1, True, aw, ah)]

    area = max(float(case.area_targets[block_index].item()), 1e-6)
    rows: list[tuple[int, bool, float, float]] = _mib_exact_shape_candidates(
        case, state, block_index, anchors
    )
    for shape_bin_id, log_r in enumerate(shape_bin_log_aspects(log_limit=log_aspect_limit)):
        ratio = math.exp(log_r)
        rows.append((shape_bin_id, False, math.sqrt(area * ratio), math.sqrt(area / ratio)))
    if virtual_frame is not None:
        rows = [
            row
            for row in rows
            if row[2] <= virtual_frame.width + 1e-6 and row[3] <= virtual_frame.height + 1e-6
        ]
    if max_shape_bins is not None and len(rows) > max_shape_bins:
        # Keep constraint-derived exact rows, square, and low/aspect extremes.
        exact_rows = [row for row in rows if row[1]]
        bin_rows = [row for row in rows if not row[1]]
        keep_bins = max(max_shape_bins - len(exact_rows), 1)
        square = min(
            bin_rows,
            key=lambda row: abs(
                row[0] - len(shape_bin_log_aspects(log_limit=log_aspect_limit)) // 2
            ),
        )
        sampled = bin_rows[:: max(1, len(bin_rows) // keep_bins)][: max(keep_bins - 1, 0)]
        rows = (exact_rows + sampled + [square])[:max_shape_bins]
    return rows


def _free_rect_lower_left_sites(
    state: ExecutionState,
    *,
    max_sites: int = 12,
    virtual_frame: PuzzleFrame | None = None,
) -> list[tuple[float, float, str, str, str]]:
    if not state.placements:
        return []
    xs = {0.0}
    ys = {0.0}
    if virtual_frame is not None:
        xs.add(virtual_frame.xmin)
        ys.add(virtual_frame.ymin)
    for x, y, w, h in state.placements.values():
        xs.add(float(x + w))
        ys.add(float(y + h))
    sites: list[tuple[float, float, str, str, str]] = []
    seen: set[tuple[float, float]] = set()
    for x in sorted(xs):
        for y in sorted(ys):
            key = (round(x, 6), round(y, 6))
            if key in seen:
                continue
            seen.add(key)
            inside_existing = any(
                px < x < px + pw and py < y < py + ph
                for px, py, pw, ph in state.placements.values()
            )
            if inside_existing:
                continue
            sites.append((x, y, f"free_rect_ll_{len(sites)}", "free_rect", "free_rect"))
            if len(sites) >= max_sites:
                return sites
    return sites


def _clamp_to_frame(value: float, low: float, high: float) -> float:
    if high < low:
        return low
    return min(max(value, low), high)


def _candidate_sites(
    case: FloorSetCase,
    state: ExecutionState,
    block_index: int,
    width: float,
    height: float,
    anchors: torch.Tensor,
    virtual_frame: PuzzleFrame | None = None,
    predicted_hull: PuzzleFrame | None = None,
) -> list[tuple[float, float, str, str, str]]:
    if virtual_frame is None:
        sites: list[tuple[float, float, str, str, str]] = [(0.0, 0.0, "origin", "origin", "origin")]
    else:
        sites = [
            (virtual_frame.xmin, virtual_frame.ymin, "frame_lower_left", "boundary", "boundary")
        ]
    ax, ay, aw, ah = [float(v) for v in anchors[block_index].tolist()]
    preplaced = bool(case.constraints[block_index, ConstraintColumns.PREPLACED].item())
    if preplaced and ax >= 0 and ay >= 0 and aw > 0 and ah > 0:
        return [(ax, ay, "preplaced", "boundary", "boundary")]

    for target_idx, (tx, ty, tw, th) in state.placements.items():
        sites.extend(
            [
                (tx + tw, ty, f"right_of_{target_idx}", "right", "placed_block"),
                (tx - width, ty, f"left_of_{target_idx}", "left", "placed_block"),
                (tx, ty + th, f"top_of_{target_idx}", "top", "placed_block"),
                (tx, ty - height, f"bottom_of_{target_idx}", "bottom", "placed_block"),
            ]
        )
        same_group = float(
            case.constraints[block_index, ConstraintColumns.CLUSTER].item()
        ) > 0 and float(case.constraints[block_index, ConstraintColumns.CLUSTER].item()) == float(
            case.constraints[target_idx, ConstraintColumns.CLUSTER].item()
        )
        if same_group:
            sites.append((tx + tw, ty, f"group_right_of_{target_idx}", "group_mate", "group"))

    sites.extend(_free_rect_lower_left_sites(state, virtual_frame=virtual_frame))

    boundary = int(case.constraints[block_index, ConstraintColumns.BOUNDARY].item())
    if boundary:
        for edge_box, edge_prefix in (
            (predicted_hull, "predicted_hull_boundary"),
            (virtual_frame, "frame_boundary"),
        ):
            if edge_box is None:
                continue
            x = edge_box.xmin
            y = edge_box.ymin
            labels: list[str] = []
            if boundary & 1:
                x = edge_box.xmin
                labels.append("left")
            if boundary & 2:
                x = edge_box.xmax - width
                labels.append("right")
            if boundary & 8:
                y = edge_box.ymin
                labels.append("bottom")
            if boundary & 4:
                y = edge_box.ymax - height
                labels.append("top")
            sites.append((x, y, edge_prefix + "_" + "_".join(labels), "boundary", "boundary"))
            x_fixed = (boundary & 1) or (boundary & 2)
            y_fixed = (boundary & 8) or (boundary & 4)
            if bool(x_fixed) != bool(y_fixed):
                x_candidates = {edge_box.xmin}
                y_candidates = {edge_box.ymin}
                for px, py, pw, ph in state.placements.values():
                    x_candidates.add(px + pw)
                    x_candidates.add(px - width)
                    y_candidates.add(py + ph)
                    y_candidates.add(py - height)
                if x_fixed:
                    edge_x = edge_box.xmin if boundary & 1 else edge_box.xmax - width
                    for candidate_y in sorted(y_candidates):
                        edge_y = _clamp_to_frame(candidate_y, edge_box.ymin, edge_box.ymax - height)
                        sites.append(
                            (
                                edge_x,
                                edge_y,
                                f"{edge_prefix}_slide_y_{len(sites)}",
                                "boundary",
                                "boundary",
                            )
                        )
                else:
                    edge_y = edge_box.ymin if boundary & 8 else edge_box.ymax - height
                    for candidate_x in sorted(x_candidates):
                        edge_x = _clamp_to_frame(candidate_x, edge_box.xmin, edge_box.xmax - width)
                        sites.append(
                            (
                                edge_x,
                                edge_y,
                                f"{edge_prefix}_slide_x_{len(sites)}",
                                "boundary",
                                "boundary",
                            )
                        )
        if virtual_frame is None:
            sites.append((0.0, 0.0, "boundary_origin", "boundary", "boundary"))

    for pin_idx, pin_block, _weight in case.p2b_edges.tolist():
        if int(pin_block) == block_index and 0 <= int(pin_idx) < len(case.pins_pos):
            px, py = [float(v) for v in case.pins_pos[int(pin_idx)].tolist()]
            if virtual_frame is None:
                pin_x = max(0.0, px - width / 2.0)
                pin_y = max(0.0, py - height / 2.0)
            else:
                pin_x = _clamp_to_frame(
                    px - width / 2.0, virtual_frame.xmin, virtual_frame.xmax - width
                )
                pin_y = _clamp_to_frame(
                    py - height / 2.0, virtual_frame.ymin, virtual_frame.ymax - height
                )
            sites.append((pin_x, pin_y, "pin_pull", "pin_pull", "pin_pull"))
            break
    return sites


def _bbox(
    placements: Mapping[int, tuple[float, float, float, float]],
) -> tuple[float, float, float, float]:
    if not placements:
        return (0.0, 0.0, 0.0, 0.0)
    return (
        min(x for x, _y, _w, _h in placements.values()),
        min(y for _x, y, _w, _h in placements.values()),
        max(x + w for x, _y, w, _h in placements.values()),
        max(y + h for _x, y, _w, h in placements.values()),
    )


def _is_legal_box(
    case: FloorSetCase,
    state: ExecutionState,
    block_index: int,
    box: tuple[float, float, float, float],
    anchors: torch.Tensor,
    virtual_frame: PuzzleFrame | None = None,
) -> tuple[bool, str]:
    x, y, w, h = box
    if w <= 0 or h <= 0:
        return False, "non_positive_dimensions"
    target_area = float(case.area_targets[block_index].item())
    fixed = bool(case.constraints[block_index, ConstraintColumns.FIXED].item())
    preplaced = bool(case.constraints[block_index, ConstraintColumns.PREPLACED].item())
    if (
        not (fixed or preplaced)
        and target_area > 0
        and abs(w * h - target_area) / target_area > 0.01
    ):
        return False, "area_tolerance_exceeded"
    ax, ay, aw, ah = [float(v) for v in anchors[block_index].tolist()]
    if (fixed or preplaced) and aw > 0 and ah > 0:
        if abs(w - aw) > 1e-4 or abs(h - ah) > 1e-4:
            return False, "fixed_preplaced_dimensions_mismatch"
        if preplaced and (abs(x - ax) > 1e-4 or abs(y - ay) > 1e-4):
            return False, "preplaced_location_mismatch"
    if virtual_frame is not None and not virtual_frame.contains_box(box):
        return False, "frame_violation"
    tensor_box = torch.tensor(box, dtype=torch.float32)
    for other_idx, other_box in state.placements.items():
        if other_idx == block_index:
            continue
        if (
            pairwise_intersection_area(tensor_box, torch.tensor(other_box, dtype=torch.float32))
            > 1e-9
        ):
            return False, f"overlaps_{other_idx}"
    return True, "legal"


def _feature_vector(
    case: FloorSetCase,
    state: ExecutionState,
    block_index: int,
    box: tuple[float, float, float, float],
    *,
    shape_bin_id: int,
    exact_shape_flag: bool,
    contact_mode: str,
    feature_mode: FeatureMode,
    virtual_frame: PuzzleFrame | None = None,
    predicted_hull: PuzzleFrame | None = None,
) -> torch.Tensor:
    x, y, w, h = box
    total_area = max(float(case.area_targets.sum().item()), 1e-6)
    scale = max(math.sqrt(total_area), 1e-6)
    bbox = _bbox(state.placements)
    bbox_area_before = max((bbox[2] - bbox[0]) * (bbox[3] - bbox[1]), 0.0)
    new_bbox = (
        min(bbox[0], x) if state.placements else x,
        min(bbox[1], y) if state.placements else y,
        max(bbox[2], x + w) if state.placements else x + w,
        max(bbox[3], y + h) if state.placements else y + h,
    )
    bbox_area_after = max((new_bbox[2] - new_bbox[0]) * (new_bbox[3] - new_bbox[1]), 0.0)
    connected = 0.0
    connected_distance = 0.0
    cx, cy = x + w / 2.0, y + h / 2.0
    for src, dst, weight in case.b2b_edges.tolist():
        src_i, dst_i = int(src), int(dst)
        other = None
        if src_i == block_index and dst_i in state.placements:
            other = state.placements[dst_i]
        elif dst_i == block_index and src_i in state.placements:
            other = state.placements[src_i]
        if other is not None:
            ox, oy, ow, oh = other
            connected += float(weight)
            connected_distance += (
                float(weight) * (abs(cx - (ox + ow / 2.0)) + abs(cy - (oy + oh / 2.0))) / scale
            )
    boundary = float(case.constraints[block_index, ConstraintColumns.BOUNDARY].item() != 0)
    boundary_code = int(case.constraints[block_index, ConstraintColumns.BOUNDARY].item())
    boundary_frame_satisfaction = (
        boundary_frame_satisfaction_fraction(boundary_code, box, virtual_frame)
        if virtual_frame is not None
        else 0.0
    )
    predicted_hull_satisfaction = (
        boundary_frame_satisfaction_fraction(boundary_code, box, predicted_hull)
        if predicted_hull is not None
        else 0.0
    )
    cluster = float(case.constraints[block_index, ConstraintColumns.CLUSTER].item() > 0)
    mib = float(case.constraints[block_index, ConstraintColumns.MIB].item() > 0)
    contact_vocab = {
        "origin": 0,
        "right": 1,
        "left": 2,
        "top": 3,
        "bottom": 4,
        "free_rect": 5,
        "group_mate": 6,
        "boundary": 7,
        "pin_pull": 8,
    }
    raw = torch.tensor(
        [
            float(case.area_targets[block_index].item()) / total_area,
            math.log(max(w / max(h, 1e-6), 1e-6)),
            x / scale,
            y / scale,
            w / scale,
            h / scale,
            len(state.placements) / max(case.block_count, 1),
            (bbox_area_after - bbox_area_before) / total_area,
            connected,
            connected_distance,
            boundary,
            cluster,
            mib,
            1.0 if exact_shape_flag else 0.0,
            float(shape_bin_id),
            1.0 if contact_mode in {"right", "left", "top", "bottom", "group_mate"} else 0.0,
            block_index / max(case.block_count - 1, 1),
            contact_vocab.get(contact_mode, 0) / max(len(contact_vocab) - 1, 1),
            boundary_frame_satisfaction,
            predicted_hull_satisfaction,
        ],
        dtype=torch.float32,
    )
    if virtual_frame is not None:
        frame_raw = torch.tensor(
            [
                (x - virtual_frame.xmin) / max(virtual_frame.width, 1e-6),
                (y - virtual_frame.ymin) / max(virtual_frame.height, 1e-6),
                (virtual_frame.xmax - (x + w)) / max(virtual_frame.width, 1e-6),
                (virtual_frame.ymax - (y + h)) / max(virtual_frame.height, 1e-6),
                (w * h) / max(virtual_frame.area, 1e-6),
            ],
            dtype=torch.float32,
        )
        raw = torch.cat([raw, frame_raw])
    if feature_mode == "puzzle_pool_raw_safe":
        return raw
    normalized = raw.clone()
    normalized[8] = connected / max(
        float(case.b2b_edges[:, 2].abs().max().item()) if case.b2b_edges.numel() else 1.0, 1.0
    )
    normalized[14] = shape_bin_id / max(len(shape_bin_log_aspects()) - 1, 1)
    return normalized


def _mask_reason_bucket(reason: str) -> str:
    if reason.startswith("overlaps_"):
        return "overlap"
    if reason == "non_positive_dimensions":
        return "non_positive"
    if reason == "area_tolerance_exceeded":
        return "area_tolerance"
    if reason in {"fixed_preplaced_dimensions_mismatch", "preplaced_location_mismatch"}:
        return "fixed_preplaced"
    if reason == "frame_violation":
        return "frame_violation"
    return reason


def empty_mask_reason_buckets() -> dict[str, int]:
    return {
        "overlap": 0,
        "non_positive": 0,
        "area_tolerance": 0,
        "fixed_preplaced": 0,
        "frame_violation": 0,
    }


def build_puzzle_candidate_descriptors(
    case: FloorSetCase,
    state: ExecutionState,
    *,
    remaining_blocks: list[int] | None = None,
    feature_mode: FeatureMode = "puzzle_pool_normalized_relational",
    max_shape_bins: int | None = 5,
    max_descriptors_per_block: int | None = 16,
    mask_reason_buckets: dict[str, int] | None = None,
    virtual_frame: PuzzleFrame | None = None,
    frame_relaxation_steps: int = 2,
    frame_expand_factor: float = 1.05,
    shape_log_aspect_limit: float | None = None,
    predicted_hull: PuzzleFrame | None = None,
    commit_boundary_to_frame: bool = False,
    boundary_commit_mode: BoundaryCommitMode = "none",
) -> list[PuzzleCandidateDescriptor]:
    """Build Step6 puzzle candidates, optionally bounded by a virtual frame.

    When `virtual_frame` is supplied, candidates are valid only if the whole
    rectangle is inside the frame.  Empty candidate pools trigger the Step6H
    relaxation policy: expand the frame by `frame_expand_factor` and retry.  The
    first pass uses a mild log-aspect range of [-2, 2] unless the caller
    overrides it; if all relaxed attempts are empty, a final [-3, 3] retry is
    allowed before returning an empty pool.
    """

    anchors = masked_anchor_xywh(case)
    remaining = remaining_blocks or [
        idx for idx in range(case.block_count) if idx not in state.placements
    ]
    initial_log_limit = 3.0 if virtual_frame is None else 2.0
    base_log_limit = initial_log_limit if shape_log_aspect_limit is None else shape_log_aspect_limit

    def build_once(frame: PuzzleFrame | None, log_limit: float) -> list[PuzzleCandidateDescriptor]:
        descriptors: list[PuzzleCandidateDescriptor] = []
        site_counter = 0
        for block_index in remaining:
            per_block = 0
            per_block_descriptors: list[PuzzleCandidateDescriptor] = []
            for shape_bin_id, exact_shape, width, height in _shape_candidates(
                case,
                state,
                block_index,
                anchors,
                max_shape_bins=max_shape_bins,
                virtual_frame=frame,
                log_aspect_limit=log_limit,
            ):
                for x, y, site_key, contact_mode, anchor_kind in _candidate_sites(
                    case,
                    state,
                    block_index,
                    width,
                    height,
                    anchors,
                    virtual_frame=frame,
                    predicted_hull=predicted_hull,
                ):
                    box = (float(x), float(y), float(width), float(height))
                    allowed, reason = _is_legal_box(case, state, block_index, box, anchors, frame)
                    if not allowed:
                        if mask_reason_buckets is not None:
                            bucket = _mask_reason_bucket(reason)
                            mask_reason_buckets[bucket] = mask_reason_buckets.get(bucket, 0) + 1
                        continue
                    shape_kind = "exact" if exact_shape else "bin"
                    family = f"shape_bin:{shape_kind}|anchor:{anchor_kind}|contact:{contact_mode}"
                    metadata: dict[str, Any] = {"source": "step6g_descriptor", "site_key": site_key}
                    boundary_code = int(
                        case.constraints[block_index, ConstraintColumns.BOUNDARY].item()
                    )
                    boundary_satisfaction = (
                        boundary_frame_satisfaction_fraction(boundary_code, box, frame)
                        if frame is not None
                        else 0.0
                    )
                    metadata["boundary_frame_satisfaction"] = boundary_satisfaction
                    predicted_hull_satisfaction = (
                        boundary_frame_satisfaction_fraction(boundary_code, box, predicted_hull)
                        if predicted_hull is not None
                        else 0.0
                    )
                    metadata["predicted_hull_satisfaction"] = predicted_hull_satisfaction
                    if frame is not None:
                        metadata.update(
                            {
                                "virtual_frame": frame.variant,
                                "frame_relaxation": frame.relaxation,
                            }
                        )
                    action = TypedAction(
                        ActionPrimitive.PLACE_ABSOLUTE,
                        block_index=block_index,
                        x=box[0],
                        y=box[1],
                        w=box[2],
                        h=box[3],
                        metadata=metadata,
                    )
                    descriptor = PuzzleCandidateDescriptor(
                        block_index=block_index,
                        shape_bin_id=shape_bin_id,
                        exact_shape_flag=exact_shape,
                        site_id=site_counter,
                        contact_mode=contact_mode,
                        anchor_kind=anchor_kind,
                        candidate_family=family,
                        legality_status=reason,
                        action_token=action,
                        normalized_features=_feature_vector(
                            case,
                            state,
                            block_index,
                            box,
                            shape_bin_id=shape_bin_id,
                            exact_shape_flag=exact_shape,
                            contact_mode=contact_mode,
                            feature_mode=feature_mode,
                            virtual_frame=frame,
                            predicted_hull=predicted_hull,
                        ),
                    )
                    validate_puzzle_descriptor(descriptor)
                    per_block_descriptors.append(descriptor)
                    site_counter += 1
                    per_block += 1
                    if (
                        max_descriptors_per_block is not None
                        and per_block >= max_descriptors_per_block
                    ):
                        break
                if max_descriptors_per_block is not None and per_block >= max_descriptors_per_block:
                    break
            boundary_code = int(case.constraints[block_index, ConstraintColumns.BOUNDARY].item())
            if (
                boundary_commit_mode == "require_predicted_hull_if_available"
                and predicted_hull is not None
                and boundary_code != 0
            ):
                committed = [
                    row
                    for row in per_block_descriptors
                    if float(row.action_token.metadata.get("predicted_hull_satisfaction", 0.0))
                    >= 1.0
                ]
                if committed:
                    per_block_descriptors = committed
            require_virtual_frame = (
                commit_boundary_to_frame
                or boundary_commit_mode == "require_virtual_frame_if_available"
            )
            if require_virtual_frame and frame is not None and boundary_code != 0:
                committed = [
                    row
                    for row in per_block_descriptors
                    if float(row.action_token.metadata.get("boundary_frame_satisfaction", 0.0))
                    >= 1.0
                ]
                if committed:
                    per_block_descriptors = committed
            descriptors.extend(per_block_descriptors)
        return descriptors

    if virtual_frame is None:
        return build_once(None, base_log_limit)

    frame = virtual_frame
    relaxation_steps = max(frame_relaxation_steps, 0)
    for attempt in range(relaxation_steps + 1):
        descriptors = build_once(frame, base_log_limit)
        if descriptors:
            return descriptors
        if attempt < relaxation_steps:
            frame = frame.expanded(frame_expand_factor)

    if base_log_limit < 3.0:
        descriptors = build_once(frame, 3.0)
        if descriptors:
            return descriptors
    return []


def choose_expert_descriptor(
    descriptors: Sequence[PuzzleCandidateDescriptor],
    teacher_action: TypedAction,
) -> int | None:
    if not descriptors:
        return None
    same_block = [
        (idx, desc)
        for idx, desc in enumerate(descriptors)
        if desc.block_index == teacher_action.block_index
    ]
    if not same_block:
        return None
    tx = 0.0 if teacher_action.x is None else float(teacher_action.x)
    ty = 0.0 if teacher_action.y is None else float(teacher_action.y)
    tw = 0.0 if teacher_action.w is None else float(teacher_action.w)
    th = 0.0 if teacher_action.h is None else float(teacher_action.h)
    best_idx = min(
        same_block,
        key=lambda pair: (
            abs((pair[1].action_token.x or 0.0) - tx)
            + abs((pair[1].action_token.y or 0.0) - ty)
            + 0.25
            * abs(
                math.log(
                    max(
                        (pair[1].action_token.w or 1e-6)
                        / max(pair[1].action_token.h or 1e-6, 1e-6),
                        1e-6,
                    )
                )
                - math.log(max(tw / max(th, 1e-6), 1e-6))
            )
        ),
    )[0]
    return int(best_idx)


def descriptor_signature(descriptor: PuzzleCandidateDescriptor) -> tuple[object, ...]:
    return (
        canonical_action_key(descriptor.action_token),
        descriptor.shape_bin_id,
        descriptor.site_id,
        descriptor.contact_mode,
        descriptor.anchor_kind,
    )


def heuristic_scores(descriptors: Sequence[PuzzleCandidateDescriptor]) -> torch.Tensor:
    if not descriptors:
        return torch.empty((0,), dtype=torch.float32)
    scores = []
    for desc in descriptors:
        f = desc.normalized_features
        # Prefer small bbox expansion, neighbor proximity, boundary/group contact, anchors.
        score = -2.0 * float(f[7]) - 0.5 * float(f[9])
        score += 0.3 * float(f[10]) + 0.2 * float(f[11]) + 0.1 * float(f[13]) + 0.2 * float(f[15])
        boundary_satisfaction = float(
            desc.action_token.metadata.get("boundary_frame_satisfaction", 0.0)
        )
        hull_satisfaction = float(
            desc.action_token.metadata.get("predicted_hull_satisfaction", 0.0)
        )
        if float(f[10]) > 0.0:
            score += 3.0 * hull_satisfaction + 0.4 * boundary_satisfaction
            score -= 0.8 * (1.0 - max(hull_satisfaction, boundary_satisfaction))
        scores.append(score)
    return torch.tensor(scores, dtype=torch.float32)
