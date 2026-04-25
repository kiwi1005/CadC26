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


def shape_bin_log_aspects(bin_count: int = 17) -> list[float]:
    values = torch.linspace(-3.0, 3.0, steps=bin_count).tolist()
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
    for shape_bin_id, log_r in enumerate(shape_bin_log_aspects()):
        ratio = math.exp(log_r)
        rows.append((shape_bin_id, False, math.sqrt(area * ratio), math.sqrt(area / ratio)))
    if max_shape_bins is not None and len(rows) > max_shape_bins:
        # Keep constraint-derived exact rows, square, and low/aspect extremes.
        exact_rows = [row for row in rows if row[1]]
        bin_rows = [row for row in rows if not row[1]]
        keep_bins = max(max_shape_bins - len(exact_rows), 1)
        square = min(bin_rows, key=lambda row: abs(row[0] - len(shape_bin_log_aspects()) // 2))
        sampled = bin_rows[:: max(1, len(bin_rows) // keep_bins)][: max(keep_bins - 1, 0)]
        rows = (exact_rows + sampled + [square])[:max_shape_bins]
    return rows


def _free_rect_lower_left_sites(
    state: ExecutionState,
    *,
    max_sites: int = 12,
) -> list[tuple[float, float, str, str, str]]:
    if not state.placements:
        return []
    xs = {0.0}
    ys = {0.0}
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


def _candidate_sites(
    case: FloorSetCase,
    state: ExecutionState,
    block_index: int,
    width: float,
    height: float,
    anchors: torch.Tensor,
) -> list[tuple[float, float, str, str, str]]:
    sites: list[tuple[float, float, str, str, str]] = [(0.0, 0.0, "origin", "origin", "origin")]
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

    sites.extend(_free_rect_lower_left_sites(state))

    boundary = int(case.constraints[block_index, ConstraintColumns.BOUNDARY].item())
    if boundary:
        sites.append((0.0, 0.0, "boundary_origin", "boundary", "boundary"))

    for pin_idx, pin_block, _weight in case.p2b_edges.tolist():
        if int(pin_block) == block_index and 0 <= int(pin_idx) < len(case.pins_pos):
            px, py = [float(v) for v in case.pins_pos[int(pin_idx)].tolist()]
            sites.append(
                (
                    max(0.0, px - width / 2.0),
                    max(0.0, py - height / 2.0),
                    "pin_pull",
                    "pin_pull",
                    "pin_pull",
                )
            )
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
        ],
        dtype=torch.float32,
    )
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
    return reason


def empty_mask_reason_buckets() -> dict[str, int]:
    return {
        "overlap": 0,
        "non_positive": 0,
        "area_tolerance": 0,
        "fixed_preplaced": 0,
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
) -> list[PuzzleCandidateDescriptor]:
    anchors = masked_anchor_xywh(case)
    remaining = remaining_blocks or [
        idx for idx in range(case.block_count) if idx not in state.placements
    ]
    descriptors: list[PuzzleCandidateDescriptor] = []
    site_counter = 0
    for block_index in remaining:
        per_block = 0
        for shape_bin_id, exact_shape, width, height in _shape_candidates(
            case, state, block_index, anchors, max_shape_bins=max_shape_bins
        ):
            for x, y, site_key, contact_mode, anchor_kind in _candidate_sites(
                case, state, block_index, width, height, anchors
            ):
                box = (float(x), float(y), float(width), float(height))
                allowed, reason = _is_legal_box(case, state, block_index, box, anchors)
                if not allowed:
                    if mask_reason_buckets is not None:
                        bucket = _mask_reason_bucket(reason)
                        mask_reason_buckets[bucket] = mask_reason_buckets.get(bucket, 0) + 1
                    continue
                shape_kind = "exact" if exact_shape else "bin"
                family = f"shape_bin:{shape_kind}|anchor:{anchor_kind}|contact:{contact_mode}"
                action = TypedAction(
                    ActionPrimitive.PLACE_ABSOLUTE,
                    block_index=block_index,
                    x=box[0],
                    y=box[1],
                    w=box[2],
                    h=box[3],
                    metadata={"source": "step6g_descriptor", "site_key": site_key},
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
                    ),
                )
                validate_puzzle_descriptor(descriptor)
                descriptors.append(descriptor)
                site_counter += 1
                per_block += 1
                if max_descriptors_per_block is not None and per_block >= max_descriptors_per_block:
                    break
            if max_descriptors_per_block is not None and per_block >= max_descriptors_per_block:
                break
    return descriptors


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
        scores.append(score)
    return torch.tensor(scores, dtype=torch.float32)
