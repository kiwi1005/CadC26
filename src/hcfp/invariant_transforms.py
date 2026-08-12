"""Pure placement transforms that preserve useful geometry invariants.

The helpers operate in the case's normalized ``(x, y, width, height)``
coordinates.  They do not inspect or rewrite directional boundary labels;
callers can therefore choose which reflected candidate is compatible with a
case's boundary requirements.
"""

from __future__ import annotations

from collections.abc import Sequence
import math
from typing import TypeAlias

import torch

from hcfp.case import FloorplanCase
from hcfp.geometry import bbox_tensor, centers_from_xywh
from hcfp.runtime import Placement


Tensor = torch.Tensor
PlacementInput: TypeAlias = Tensor | Sequence[Placement]


def weighted_median_translation(
    case: FloorplanCase, placement: PlacementInput
) -> Tensor:
    """Translate all blocks to minimize weighted P2B Manhattan distance.

    The two translation components are independent weighted medians of
    ``pin - block_center`` over P2B edges.  A translation cannot preserve a
    preplaced target, so this transform is intentionally available only when
    the case has no preplaced blocks.
    """

    boxes = _as_boxes(placement, case)
    if bool(case.preplaced_mask.any()):
        raise ValueError("weighted-median translation requires no preplaced blocks")
    if not case.p2b_edges.numel():
        return boxes

    edges = case.p2b_edges.to(device=boxes.device, dtype=boxes.dtype)
    pins = case.pins.to(device=boxes.device, dtype=boxes.dtype)
    pin_index = edges[:, 0].to(dtype=torch.long)
    block_index = edges[:, 1].to(dtype=torch.long)
    weights = edges[:, 2]
    positive = weights > 0.0
    if not bool(positive.any()):
        return boxes

    desired = (
        pins[pin_index[positive]] - centers_from_xywh(boxes)[block_index[positive]]
    )
    active_weights = weights[positive]
    delta = torch.stack(
        (
            _weighted_median(desired[:, 0], active_weights),
            _weighted_median(desired[:, 1], active_weights),
        )
    )
    translated = boxes.clone()
    translated[:, :2] += delta
    return translated


def mirror_candidates(
    placement: PlacementInput,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Return ``(original, x, y, xy)`` reflections of one placement.

    ``x`` reflects across the placement bbox's vertical midpoint and ``y``
    reflects across its horizontal midpoint.  Reflection is global, so block
    areas, non-overlap, bbox area, and B2B Manhattan distances are unchanged.
    """

    boxes = _as_boxes(placement)
    left, bottom, right, top = bbox_tensor(boxes)
    mirrored_x = boxes.clone()
    mirrored_x[:, 0] = left + right - (boxes[:, 0] + boxes[:, 2])
    mirrored_y = boxes.clone()
    mirrored_y[:, 1] = bottom + top - (boxes[:, 1] + boxes[:, 3])
    mirrored_xy = mirrored_x.clone()
    mirrored_xy[:, 1] = bottom + top - (boxes[:, 1] + boxes[:, 3])
    return boxes, mirrored_x, mirrored_y, mirrored_xy


def reciprocal_affine_transform(placement: PlacementInput, s: float) -> Tensor:
    """Apply a positive reciprocal x/y affine scale around the bbox center.

    The x axis is scaled by ``s`` and the y axis by ``1/s``.  Rectangle areas
    and the occupied bbox area are preserved, while aspect ratio is allowed to
    change.  ``s`` must be finite and strictly positive.
    """

    boxes = _as_boxes(placement)
    try:
        scalar = float(s)
    except (TypeError, ValueError) as exc:
        raise TypeError("s must be a finite positive scalar") from exc
    if not math.isfinite(scalar) or scalar <= 0.0:
        raise ValueError("s must be a finite positive scalar")

    left, bottom, right, top = bbox_tensor(boxes)
    bbox_center = torch.stack(((left + right) * 0.5, (bottom + top) * 0.5))
    scale = torch.tensor((scalar, 1.0 / scalar), dtype=boxes.dtype, device=boxes.device)
    centers = centers_from_xywh(boxes)
    transformed_centers = bbox_center + (centers - bbox_center) * scale
    dimensions = boxes[:, 2:4] * scale
    return torch.cat((transformed_centers - 0.5 * dimensions, dimensions), dim=1)


def candidate_placements(
    case: FloorplanCase,
    placement: PlacementInput,
    *,
    include_translation: bool = True,
    affine_scale: float | None = None,
) -> tuple[Tensor, ...]:
    """Build a small deterministic candidate tuple from one placement.

    The first four entries are always ``original, x, y, xy`` mirrors.  A
    weighted-median P2B translation is appended by default and an affine
    candidate is appended when ``affine_scale`` is supplied.  Set
    ``include_translation=False`` for cases with preplaced anchors.
    """

    boxes = _as_boxes(placement, case)
    candidates = list(mirror_candidates(boxes))
    if include_translation:
        candidates.append(weighted_median_translation(case, boxes))
    if affine_scale is not None:
        candidates.append(reciprocal_affine_transform(boxes, affine_scale))
    return tuple(candidates)


def _as_boxes(placement: PlacementInput, case: FloorplanCase | None = None) -> Tensor:
    try:
        boxes = torch.as_tensor(placement)
    except (TypeError, ValueError) as exc:
        raise ValueError("placement must have shape [N, 4]") from exc
    if boxes.ndim != 2 or boxes.shape[1] != 4:
        raise ValueError("placement must have shape [N, 4]")
    if not torch.is_floating_point(boxes):
        boxes = boxes.to(dtype=torch.float32)
    if not bool(torch.isfinite(boxes).all()) or bool((boxes[:, 2:4] <= 0.0).any()):
        raise ValueError("placement must contain finite positive rectangles")
    if case is not None and boxes.shape[0] != int(case.n):
        raise ValueError(f"placement must have shape [{case.n}, 4]")
    return boxes.clone()


def _weighted_median(values: Tensor, weights: Tensor) -> Tensor:
    order = torch.argsort(values, stable=True)
    sorted_values = values[order]
    cumulative = weights[order].cumsum(dim=0)
    halfway = cumulative[-1] * 0.5
    index = torch.searchsorted(cumulative, halfway, right=False)
    return sorted_values[index.clamp_max(sorted_values.numel() - 1)]


__all__ = [
    "candidate_placements",
    "mirror_candidates",
    "reciprocal_affine_transform",
    "weighted_median_translation",
]
