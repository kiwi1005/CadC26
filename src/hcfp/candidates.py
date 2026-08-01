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
    overlap = torch.triu(overlap_area_matrix(work), diagonal=1).sum(dim=(-2, -1))
    hpwl = hpwl_tensor(case, centers)
    area = bbox_area_tensor(work)
    aspect = log_aspect_from_xywh(work).abs()
    index = torch.linspace(0.0, 1.0, work.shape[0], device=work.device)
    return torch.stack(
        (
            torch.log1p(overlap),
            torch.log1p(hpwl),
            torch.log1p(area),
            displacement.mean(dim=1),
            displacement.amax(dim=1),
            aspect.mean(dim=1),
            aspect.amax(dim=1),
            index,
        ),
        dim=1,
    ).float()
