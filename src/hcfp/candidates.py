"""Cheap candidate features shared by ranking, replay, and inference."""

from __future__ import annotations

import torch

from hcfp.case import FloorplanCase
from hcfp.geometry import (
    bbox_area_tensor,
    centers_from_xywh,
    hpwl_tensor,
    log_aspect_from_xywh,
    overlap_area_matrix,
)


Tensor = torch.Tensor


def candidate_features(case: FloorplanCase, boxes: Tensor, anchor: Tensor) -> Tensor:
    work = torch.as_tensor(boxes, dtype=torch.float32, device=case.area.device)
    if work.ndim != 3 or work.shape[1:] != (case.n, 4):
        raise ValueError("boxes must have shape [C,N,4]")
    centers = centers_from_xywh(work)
    anchor_center = centers_from_xywh(anchor.to(device=work.device, dtype=torch.float32))
    displacement = torch.linalg.vector_norm(centers - anchor_center, dim=-1)
    overlap_upper = torch.triu(overlap_area_matrix(work), diagonal=1)
    overlap = overlap_upper.sum(dim=(-2, -1))
    overlap_pair_count = (overlap_upper > 0.0).sum(dim=(-2, -1)).to(dtype=torch.float32)
    hpwl = hpwl_tensor(case, centers)
    area = bbox_area_tensor(work)
    aspect = log_aspect_from_xywh(work).abs()
    return torch.stack(
        (
            torch.log1p(overlap),
            torch.log1p(hpwl),
            torch.log1p(area),
            displacement.mean(dim=1),
            displacement.amax(dim=1),
            aspect.mean(dim=1),
            aspect.amax(dim=1),
            torch.log1p(overlap_pair_count),
        ),
        dim=1,
    ).float()
