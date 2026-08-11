"""FP32 rectangle geometry shared by HCFP dynamics and projection."""

from __future__ import annotations

import torch

from hcfp.case import FloorplanCase
from hcfp.constraints.mib_shapes import OFFICIAL_AREA_REL_TOL


Tensor = torch.Tensor


def _batched(value: Tensor, width: int) -> tuple[Tensor, bool]:
    tensor = torch.as_tensor(value)
    if tensor.ndim not in (2, 3) or tensor.shape[-1] != width:
        raise ValueError(f"expected [N,{width}] or [K,N,{width}]")
    return (tensor.unsqueeze(0), True) if tensor.ndim == 2 else (tensor, False)


def centers_from_xywh(xywh: Tensor) -> Tensor:
    boxes = torch.as_tensor(xywh)
    if boxes.shape[-1] != 4:
        raise ValueError("xywh must end in four coordinates")
    return boxes[..., :2] + 0.5 * boxes[..., 2:4]


def log_aspect_from_xywh(xywh: Tensor) -> Tensor:
    boxes = torch.as_tensor(xywh)
    if boxes.shape[-1] != 4 or bool((boxes[..., 2:4] <= 0).any()):
        raise ValueError("xywh must contain positive dimensions")
    return torch.log(boxes[..., 2] / boxes[..., 3])


def initializer_anchor(
    case: FloorplanCase,
    center: Tensor,
    log_aspect: Tensor,
    *,
    absolute: bool,
) -> tuple[Tensor, Tensor]:
    """Return the state added to initializer outputs.

    Absolute models predict normalized centers and soft-block log aspects from
    zero; hard geometry remains anchored to the exact official targets.
    """

    if not absolute:
        return center, log_aspect
    anchored_center = torch.zeros_like(center)
    anchored_aspect = torch.zeros_like(log_aspect)
    target_center = centers_from_xywh(case.target).to(
        device=anchored_center.device,
        dtype=anchored_center.dtype,
    )
    target_aspect = log_aspect_from_xywh(case.target.clamp_min(1.0e-30)).to(
        device=anchored_aspect.device,
        dtype=anchored_aspect.dtype,
    )
    anchored_center[..., case.preplaced_mask, :] = target_center[case.preplaced_mask]
    hard_shape = case.fixed_mask | case.preplaced_mask
    anchored_aspect[..., hard_shape] = target_aspect[hard_shape]
    return anchored_center, anchored_aspect


def exact_shape_projection(
    case: FloorplanCase,
    log_aspect: Tensor,
    *,
    enforce_mib: bool = True,
) -> Tensor:
    """Reconstruct dimensions with hard targets and compatible shared MIB shapes."""

    ratio_log = torch.as_tensor(log_aspect, dtype=torch.float32, device=case.area.device)
    if ratio_log.ndim not in (1, 2) or ratio_log.shape[-1] != case.n:
        raise ValueError("log_aspect must have shape [N] or [K,N]")
    ratio_log = ratio_log.clamp(-4.0, 4.0)
    area = case.area.to(dtype=torch.float32)
    while area.ndim < ratio_log.ndim:
        area = area.unsqueeze(0)
    width = torch.sqrt(area * torch.exp(ratio_log))
    height = area / width
    dimensions = torch.stack((width, height), dim=-1)
    hard = (case.fixed_mask | case.preplaced_mask).to(device=dimensions.device)
    target_wh = case.target[:, 2:4].to(device=dimensions.device, dtype=torch.float32)
    while hard.ndim < dimensions.ndim - 1:
        hard = hard.unsqueeze(0)
        target_wh = target_wh.unsqueeze(0)
    dimensions = torch.where(hard.unsqueeze(-1), target_wh, dimensions)
    return (
        _project_compatible_mib_shapes(case, dimensions)
        if enforce_mib
        else dimensions
    )


def _project_compatible_mib_shapes(
    case: FloorplanCase,
    dimensions: Tensor,
) -> Tensor:
    membership = case.mib_membership.to(device=dimensions.device, dtype=torch.bool)
    if not membership.numel():
        return dimensions

    batched = dimensions.ndim == 3
    projected = dimensions if batched else dimensions.unsqueeze(0)
    area = case.area.to(device=dimensions.device, dtype=dimensions.dtype)
    hard = (case.fixed_mask | case.preplaced_mask).to(device=dimensions.device)
    tolerance = OFFICIAL_AREA_REL_TOL

    for row in membership:
        members = torch.nonzero(row, as_tuple=False).reshape(-1)
        if members.numel() < 2:
            continue
        member_area = area[members]
        low = torch.max(member_area * (1.0 - tolerance))
        high = torch.min(member_area * (1.0 + tolerance))
        if float(low) > float(high):
            continue

        hard_members = members[hard[members]]
        if hard_members.numel():
            shared = projected[:, hard_members[0]]
            if not all(
                bool(torch.equal(projected[:, index], shared))
                for index in hard_members[1:]
            ):
                continue
            shared_area = shared[:, 0] * shared[:, 1]
            relative_error = torch.abs(shared_area[:, None] - member_area) / member_area
            if bool((relative_error > tolerance).any()):
                continue
        else:
            target_area = (low + high) * 0.5
            log_ratio = torch.log(projected[:, members, 0] / projected[:, members, 1]).mean(dim=1)
            width = torch.sqrt(target_area * torch.exp(log_ratio))
            shared = torch.stack((width, target_area / width), dim=1)

        mask = row.view(1, -1, 1)
        projected = torch.where(mask, shared[:, None, :], projected)
    return projected if batched else projected[0]


def xywh_from_state(
    case: FloorplanCase,
    center: Tensor,
    log_aspect: Tensor,
    *,
    enforce_mib: bool = True,
) -> Tensor:
    center_tensor = torch.as_tensor(center, dtype=torch.float32, device=case.area.device)
    dimensions = exact_shape_projection(case, log_aspect, enforce_mib=enforce_mib)
    if center_tensor.shape != dimensions.shape:
        raise ValueError("center and reconstructed dimensions must have matching shapes")
    return torch.cat((center_tensor - 0.5 * dimensions, dimensions), dim=-1)


def normalize_xywh(case: FloorplanCase, xywh: Tensor) -> Tensor:
    boxes = torch.as_tensor(xywh, dtype=torch.float32, device=case.area.device).clone()
    if boxes.shape[-1] != 4:
        raise ValueError("xywh must end in four coordinates")
    origin = case.origin.to(device=boxes.device, dtype=boxes.dtype)
    boxes[..., :2] = (boxes[..., :2] - origin) / case.scale
    boxes[..., 2:4] /= case.scale
    return boxes


def denormalize_xywh(case: FloorplanCase, xywh: Tensor) -> Tensor:
    boxes = torch.as_tensor(xywh).clone()
    if boxes.shape[-1] != 4:
        raise ValueError("xywh must end in four coordinates")
    origin = case.origin.to(device=boxes.device, dtype=boxes.dtype)
    boxes[..., :2] = boxes[..., :2] * case.scale + origin
    boxes[..., 2:4] *= case.scale
    return boxes


def overlap_extents(xywh: Tensor) -> Tensor:
    boxes, single = _batched(torch.as_tensor(xywh), 4)
    left = boxes[..., 0]
    bottom = boxes[..., 1]
    right = left + boxes[..., 2]
    top = bottom + boxes[..., 3]
    overlap_x = torch.minimum(right[:, :, None], right[:, None, :]) - torch.maximum(
        left[:, :, None], left[:, None, :]
    )
    overlap_y = torch.minimum(top[:, :, None], top[:, None, :]) - torch.maximum(
        bottom[:, :, None], bottom[:, None, :]
    )
    result = torch.stack((overlap_x.clamp_min(0.0), overlap_y.clamp_min(0.0)), dim=-1)
    eye = torch.eye(boxes.shape[1], dtype=torch.bool, device=boxes.device).view(1, boxes.shape[1], boxes.shape[1], 1)
    result = torch.where(eye, torch.zeros_like(result), result)
    return result[0] if single else result


def overlap_area_matrix(xywh: Tensor) -> Tensor:
    extents = overlap_extents(xywh)
    return extents[..., 0] * extents[..., 1]


def bbox_tensor(xywh: Tensor) -> Tensor:
    boxes, single = _batched(torch.as_tensor(xywh), 4)
    left = boxes[..., 0].amin(dim=1)
    bottom = boxes[..., 1].amin(dim=1)
    right = (boxes[..., 0] + boxes[..., 2]).amax(dim=1)
    top = (boxes[..., 1] + boxes[..., 3]).amax(dim=1)
    result = torch.stack((left, bottom, right, top), dim=-1)
    return result[0] if single else result


def bbox_area_tensor(xywh: Tensor) -> Tensor:
    bounds = bbox_tensor(xywh)
    return (bounds[..., 2] - bounds[..., 0]) * (bounds[..., 3] - bounds[..., 1])


def hpwl_tensor(case: FloorplanCase, center: Tensor) -> Tensor:
    centers, single = _batched(torch.as_tensor(center, dtype=torch.float32, device=case.area.device), 2)
    delta = torch.abs(centers[:, :, None, :] - centers[:, None, :, :]).sum(dim=-1)
    weights = torch.triu(case.b2b_weight.to(device=centers.device, dtype=torch.float32), diagonal=1)
    total = (delta * weights).sum(dim=(1, 2))
    if case.p2b_edges.numel():
        edges = case.p2b_edges.to(device=centers.device)
        pin_index = edges[:, 0].to(torch.long)
        block_index = edges[:, 1].to(torch.long)
        edge_weight = edges[:, 2].to(torch.float32)
        pins = case.pins.to(device=centers.device, dtype=torch.float32)
        distance = torch.abs(centers[:, block_index] - pins[pin_index].unsqueeze(0)).sum(dim=-1)
        total = total + (distance * edge_weight.unsqueeze(0)).sum(dim=1)
    return total[0] if single else total
