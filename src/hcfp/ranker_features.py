"""Repair-aware candidate feature builder for ranker training."""

from __future__ import annotations

from collections.abc import Sequence
import math

import torch

from hcfp.candidates import candidate_features
from hcfp.case import FloorplanCase
from hcfp.geometry import centers_from_xywh


Tensor = torch.Tensor
STORED_RANKER_FEATURE_VERSION = "stored_candidate_features_v1"
RANKER_FEATURE_VERSION = "repair_aware_ranker_features_v5_family_identity"
RANKER_FEATURE_DIM = 28
KIND_FEATURE_OFFSET = 18
PROXY_FEATURE_OFFSET = 23
STAGE_FEATURE_INDEX = 26
PROJECTION_OK_FEATURE_INDEX = 27
RANKER_FEATURE_NAMES = (
    *(f"raw_{index}" for index in range(8)),
    *(f"post_bdp_{index}" for index in range(8)),
    "raw_to_post_center_movement_mean_log1p",
    "raw_to_post_center_movement_max_log1p",
    "kind_learned",
    "kind_constraint",
    "kind_topology",
    "kind_treemap",
    "kind_btree",
    "post_bdp_boundary_proxy",
    "post_bdp_group_proxy",
    "post_bdp_mib_proxy",
    "stage_post_relax",
    "projection_ok",
)
_KIND_TO_INDEX = {
    "learned": 0,
    "constraint": 1,
    "topology": 2,
    "treemap": 3,
    "btree": 4,
}


def repair_aware_ranker_features(
    case: FloorplanCase,
    raw: Tensor,
    post_bdp: Tensor,
    anchor: Tensor,
    candidate_kinds: Sequence[str],
    candidate_stage: str,
) -> Tensor:
    """Build repair-aware candidate features with explicit family identity."""

    raw_boxes = _boxes(raw, case, "raw")
    post_boxes = _boxes(post_bdp, case, "post_bdp")
    if post_boxes.shape != raw_boxes.shape:
        raise ValueError("raw and post_bdp must have equal shape")
    anchor_boxes = _anchor(anchor, case, raw_boxes.device)
    count = int(raw_boxes.shape[0])
    kind_features = _kind_one_hot(candidate_kinds, count, raw_boxes.device)
    stage = _stage_feature(candidate_stage, count, raw_boxes.device)
    ok = _projection_ok_from_geometry(case, raw_boxes, post_boxes)

    raw_features = candidate_features(case, raw_boxes, anchor_boxes)
    post_features = candidate_features(case, post_boxes, anchor_boxes)
    movement = torch.linalg.vector_norm(
        centers_from_xywh(post_boxes) - centers_from_xywh(raw_boxes),
        dim=-1,
    )
    movement_features = torch.stack(
        (torch.log1p(movement.mean(dim=1)), torch.log1p(movement.amax(dim=1))),
        dim=1,
    )
    proxies = torch.stack(
        (
            _boundary_proxy(case, post_boxes),
            _group_proxy(case, post_boxes),
            _mib_proxy(case, post_boxes),
        ),
        dim=1,
    )
    features = torch.cat(
        (
            raw_features,
            post_features,
            movement_features,
            kind_features,
            proxies,
            stage[:, None],
            ok[:, None].to(dtype=torch.float32),
        ),
        dim=1,
    ).float()
    if features.shape != (count, RANKER_FEATURE_DIM):
        raise RuntimeError("repair-aware ranker feature dimension mismatch")
    if not bool(torch.isfinite(features).all()):
        raise ValueError("repair-aware ranker features must be finite")
    return features


def _boxes(value: Tensor, case: FloorplanCase, name: str) -> Tensor:
    boxes = torch.as_tensor(value, dtype=torch.float32, device=case.area.device)
    if boxes.ndim != 3 or boxes.shape[1:] != (case.n, 4):
        raise ValueError(f"{name} must have shape [C,N,4]")
    if not bool(torch.isfinite(boxes).all()) or bool((boxes[..., 2:] <= 0.0).any()):
        raise ValueError(f"{name} must be finite with positive dimensions")
    return boxes


def _anchor(value: Tensor, case: FloorplanCase, device: torch.device) -> Tensor:
    anchor = torch.as_tensor(value, dtype=torch.float32, device=device)
    if anchor.shape != (case.n, 4):
        raise ValueError("anchor must have shape [N,4]")
    if not bool(torch.isfinite(anchor).all()) or bool((anchor[:, 2:] <= 0.0).any()):
        raise ValueError("anchor must be finite with positive dimensions")
    return anchor


def _kind_one_hot(kinds: Sequence[str], count: int, device: torch.device) -> Tensor:
    if len(kinds) != count:
        raise ValueError("candidate_kinds must align with candidate count")
    try:
        indices = torch.tensor([_KIND_TO_INDEX[str(kind)] for kind in kinds], dtype=torch.long, device=device)
    except KeyError as exc:
        raise ValueError("candidate_kinds contains an unsupported family") from exc
    return torch.nn.functional.one_hot(indices, num_classes=5).to(dtype=torch.float32)


def _projection_ok_from_geometry(case: FloorplanCase, raw: Tensor, post: Tensor) -> Tensor:
    """Reconstruct BDP success from pre-tail geometry without target leakage."""

    work = post.to(dtype=torch.float64)
    low = work[..., :2]
    high = low + work[..., 2:4]
    overlap = torch.minimum(high[:, :, None], high[:, None, :]) - torch.maximum(
        low[:, :, None],
        low[:, None, :],
    )
    tolerance = 1.0e-6 / max(float(case.scale), 1.0e-30)
    active = (overlap > tolerance).all(dim=-1)
    eye = torch.eye(case.n, dtype=torch.bool, device=work.device).unsqueeze(0)
    dimensions_ok = (work[..., 2:4] > 0.0).all(dim=(1, 2))
    overlap_ok = ~(active & ~eye).any(dim=(1, 2))
    preplaced = case.preplaced_mask.to(device=work.device, dtype=torch.bool)
    if bool(preplaced.any()):
        preplaced_ok = torch.isclose(
            work[:, preplaced, :2],
            raw.to(dtype=torch.float64)[:, preplaced, :2],
        ).all(dim=(1, 2))
    else:
        preplaced_ok = torch.ones(len(work), dtype=torch.bool, device=work.device)
    return dimensions_ok & overlap_ok & preplaced_ok


def _stage_feature(stage: str, count: int, device: torch.device) -> Tensor:
    if stage not in {"initial", "post_relax"}:
        raise ValueError("candidate_stage must be initial or post_relax")
    return torch.full(
        (count,),
        float(stage == "post_relax"),
        dtype=torch.float32,
        device=device,
    )


def _boundary_proxy(case: FloorplanCase, boxes: Tensor) -> Tensor:
    bits = case.boundary_bits.to(device=boxes.device, dtype=torch.bool)
    sides = _sides(boxes.to(dtype=torch.float64))
    frame = torch.stack(
        (
            sides[..., 0].amin(dim=1),
            sides[..., 1].amax(dim=1),
            sides[..., 2].amax(dim=1),
            sides[..., 3].amin(dim=1),
        ),
        dim=1,
    )
    tolerance = 1.0e-6 / max(float(case.scale), 1.0e-30)
    touching = (sides - frame[:, None, :]).abs() < tolerance
    missing_blocks = (bits.unsqueeze(0) & ~touching).any(dim=2).sum(dim=1)
    return torch.log1p(missing_blocks.to(dtype=boxes.dtype))


def _group_proxy(case: FloorplanCase, boxes: Tensor) -> Tensor:
    members = case.group_membership.to(device=boxes.device, dtype=torch.bool)
    if not members.numel():
        return torch.zeros(len(boxes), dtype=boxes.dtype, device=boxes.device)
    if boxes.device.type == "cpu":
        from hcfp.verify import connected_components_for_group

        tolerance = 1.0e-6 / max(float(case.scale), 1.0e-30)
        violations = []
        for candidate in boxes:
            components = sum(
                max(
                    0,
                    connected_components_for_group(candidate, group, tol=tolerance) - 1,
                )
                for group in members
            )
            violations.append(math.log1p(components))
        return torch.tensor(violations, dtype=boxes.dtype, device=boxes.device)
    sides = _sides(boxes.to(dtype=torch.float64))
    left, right, top, bottom = sides[..., 0], sides[..., 1], sides[..., 2], sides[..., 3]
    x_overlap = torch.minimum(right[:, :, None], right[:, None, :]) - torch.maximum(
        left[:, :, None],
        left[:, None, :],
    )
    y_overlap = torch.minimum(top[:, :, None], top[:, None, :]) - torch.maximum(
        bottom[:, :, None],
        bottom[:, None, :],
    )
    x_gap = torch.relu(torch.maximum(left[:, :, None], left[:, None, :]) - torch.minimum(right[:, :, None], right[:, None, :]))
    y_gap = torch.relu(torch.maximum(bottom[:, :, None], bottom[:, None, :]) - torch.minimum(top[:, :, None], top[:, None, :]))
    tolerance = 1.0e-6 / max(float(case.scale), 1.0e-30)
    contact = ((x_overlap > tolerance) & (y_gap <= tolerance)) | (
        (y_overlap > tolerance) & (x_gap <= tolerance)
    )
    same_group = (members.transpose(0, 1).to(dtype=torch.float32) @ members.to(dtype=torch.float32)) > 0.0
    grouped = members.any(dim=0)
    reach = contact & same_group.unsqueeze(0)
    reach = reach | torch.diag_embed(grouped.expand(len(boxes), -1))
    for _ in range(max(0, math.ceil(math.log2(max(case.n, 1))))):
        reach = reach | ((reach.to(dtype=boxes.dtype) @ reach.to(dtype=boxes.dtype)) > 0.0)
    component_size = reach.sum(dim=2).clamp_min(1).to(dtype=boxes.dtype)
    component_count = (
        grouped.to(dtype=boxes.dtype).unsqueeze(0) / component_size
    ).sum(dim=1)
    violation = (component_count - float(members.shape[0])).clamp_min(0.0)
    return torch.log1p(violation)


def _mib_proxy(case: FloorplanCase, boxes: Tensor) -> Tensor:
    same_mib = _same_membership(case.mib_membership, boxes.device)
    mask = torch.triu(same_mib, diagonal=1)
    log_shape = torch.log(boxes[..., 2:].clamp_min(1.0e-12))
    mismatch = (log_shape[:, :, None, :] - log_shape[:, None, :, :]).abs().sum(dim=-1)
    selected = mask.to(dtype=boxes.dtype)
    return torch.log1p((mismatch * selected.unsqueeze(0)).sum(dim=(1, 2)) / selected.sum().clamp_min(1.0))


def _same_membership(membership: Tensor, device: torch.device) -> Tensor:
    members = membership.to(device=device, dtype=torch.float32)
    if not members.numel():
        return torch.zeros((membership.shape[1], membership.shape[1]), dtype=torch.bool, device=device)
    return (members.transpose(0, 1) @ members) > 0.0


def _sides(boxes: Tensor) -> Tensor:
    return torch.stack(
        (
            boxes[..., 0],
            boxes[..., 0] + boxes[..., 2],
            boxes[..., 1] + boxes[..., 3],
            boxes[..., 1],
        ),
        dim=-1,
    )


__all__ = [
    "KIND_FEATURE_OFFSET",
    "PROJECTION_OK_FEATURE_INDEX",
    "PROXY_FEATURE_OFFSET",
    "RANKER_FEATURE_DIM",
    "RANKER_FEATURE_NAMES",
    "RANKER_FEATURE_VERSION",
    "STAGE_FEATURE_INDEX",
    "STORED_RANKER_FEATURE_VERSION",
    "repair_aware_ranker_features",
]
