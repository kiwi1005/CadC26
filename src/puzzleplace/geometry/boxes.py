from __future__ import annotations

from typing import Iterable

import torch


def pairwise_intersection_area(box_a: torch.Tensor, box_b: torch.Tensor) -> float:
    ax, ay, aw, ah = [float(v) for v in box_a]
    bx, by, bw, bh = [float(v) for v in box_b]
    overlap_w = max(0.0, min(ax + aw, bx + bw) - max(ax, bx))
    overlap_h = max(0.0, min(ay + ah, by + bh) - max(ay, by))
    return overlap_w * overlap_h


def bbox_area(boxes: Iterable[torch.Tensor]) -> float:
    xs0, ys0, xs1, ys1 = [], [], [], []
    for box in boxes:
        x, y, w, h = [float(v) for v in box]
        xs0.append(x)
        ys0.append(y)
        xs1.append(x + w)
        ys1.append(y + h)
    if not xs0:
        return 0.0
    return (max(xs1) - min(xs0)) * (max(ys1) - min(ys0))
