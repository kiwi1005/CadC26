"""Deterministic latent-outline hypotheses for a normalized FloorplanCase.

The official FloorSet input does not contain a fixed canvas.  This module
therefore only proposes *latent* envelopes for conditioning and audit; it does
not change the official verifier or place blocks.  Compact, area-derived
hypotheses use a ``[0.95, 1.00]`` utilization prior.  A separate pin-perimeter
hypothesis preserves the observed pin bounds even when its utilization falls
outside that compact prior.  Positional containment is checked for preplaced
rectangles; fixed blocks contribute only their required dimensions because
their target coordinates are not official anchors.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from typing import Any, Mapping, Sequence

import torch

from hcfp.case import FloorplanCase


_UTILIZATION_PRIORS = (1.0, 0.975, 0.95)
_SIDES = ("left", "right", "bottom", "top")
_EPS = 1.0e-10


@dataclass(frozen=True)
class OutlineHypothesis:
    """One auditable latent outline in the case's normalized coordinates.

    ``(x_left, y_bottom, x_right, y_top)`` is the envelope.  The individual
    coordinate fields intentionally mirror the P1 audit schema.  ``target``
    rectangles are never rewritten; the bounds and anchor metrics can be
    rechecked by a downstream decoder.

    ``pin_side_coverage`` is the fraction of pins that lie inside this
    envelope.  ``side_coverage`` separately reports how many of the four
    perimeter sides received at least one nearest-side assignment.
    """

    hypothesis_id: str
    x_left: float
    x_right: float
    y_bottom: float
    y_top: float
    source: str
    provenance: tuple[str, ...]
    score: float
    scores: Mapping[str, float]
    confidence: float
    pin_residual: float
    area_prior_residual: float
    anchor_residual: float
    pin_side_assignment: tuple[str, ...]
    pin_side_coverage: float
    anchor_coverage: float
    anchor_span: tuple[float, float]
    block_area: float
    utilization: float
    side_coverage: float = 0.0
    pin_side_counts: tuple[tuple[str, int], ...] = ()

    @property
    def bounds(self) -> tuple[float, float, float, float]:
        """Return ``(x_left, y_bottom, x_right, y_top)``."""

        return (self.x_left, self.y_bottom, self.x_right, self.y_top)

    def contains_targets(
        self,
        targets: Any,
        *,
        mask: Any | None = None,
        tolerance: float = 1.0e-8,
    ) -> bool:
        """Return whether every selected ``(x, y, w, h)`` target is inside."""

        boxes = torch.as_tensor(targets, dtype=torch.float64, device="cpu")
        if boxes.ndim != 2 or boxes.shape[-1] != 4:
            raise ValueError("targets must have shape [N, 4]")
        if mask is None:
            selected = boxes
        else:
            selected_mask = torch.as_tensor(mask, dtype=torch.bool, device="cpu").reshape(-1)
            if selected_mask.numel() != boxes.shape[0]:
                raise ValueError("target mask length does not match targets")
            selected = boxes[selected_mask]
        if not selected.numel():
            return True
        return bool(
            (
                (selected[:, 0] >= self.x_left - tolerance)
                & (selected[:, 1] >= self.y_bottom - tolerance)
                & (selected[:, 0] + selected[:, 2] <= self.x_right + tolerance)
                & (selected[:, 1] + selected[:, 3] <= self.y_top + tolerance)
            ).all()
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly, deterministic audit record."""
        width = self.x_right - self.x_left
        height = self.y_top - self.y_bottom
        return {
            "hypothesis_id": self.hypothesis_id,
            "bounds": list(self.bounds),
            "x_left": self.x_left,
            "x_right": self.x_right,
            "y_bottom": self.y_bottom,
            "y_top": self.y_top,
            "width": width,
            "height": height,
            "area": width * height,
            "aspect_ratio": width / height,
            "utilization": self.utilization,
            "source": self.source,
            "provenance": list(self.provenance),
            "score": self.score,
            "scores": dict(self.scores),
            "confidence": self.confidence,
            "pin_residual": self.pin_residual,
            "area_prior_residual": self.area_prior_residual,
            "anchor_residual": self.anchor_residual,
            "pin_side_assignment": list(self.pin_side_assignment),
            "pin_side_counts": {key: value for key, value in self.pin_side_counts},
            "pin_side_coverage": self.pin_side_coverage,
            "side_coverage": self.side_coverage,
            "anchor_coverage": self.anchor_coverage,
            "anchor_span": list(self.anchor_span),
            "block_area": self.block_area,
        }


def infer_outline_hypotheses(
    case: FloorplanCase,
    max_hypotheses: int = 8,
) -> tuple[OutlineHypothesis, ...]:
    """Infer a deterministic beam of latent outline hypotheses.

    The case is already normalized by :func:`hcfp.case.from_official`, so the
    returned coordinates and areas use the same coordinate frame.  A feasible
    case normally yields 4--8 hypotheses; if positional preplaced anchors or
    fixed dimensions cannot fit a compact outline, that candidate is rejected.
    """

    if not isinstance(max_hypotheses, int) or isinstance(max_hypotheses, bool):
        raise TypeError("max_hypotheses must be an integer")
    if not 4 <= max_hypotheses <= 8:
        raise ValueError("max_hypotheses must be in [4, 8]")

    area = _cpu_tensor(case.area, torch.float64).reshape(-1)
    if area.numel() != int(case.n):
        raise ValueError("case.area length does not match case.n")
    if not bool(torch.isfinite(area).all()) or bool((area <= 0).any()):
        raise ValueError("case.area must be finite and positive")
    block_area = float(area.sum().item())

    pins = _cpu_tensor(case.pins, torch.float64)
    if pins.numel() == 0:
        pins = torch.empty((0, 2), dtype=torch.float64)
    if pins.ndim != 2 or pins.shape[1] != 2:
        raise ValueError("case.pins must have shape [P, 2]")
    if not bool(torch.isfinite(pins).all()):
        raise ValueError("case.pins must be finite")

    fixed = _cpu_tensor(case.fixed_mask, torch.bool).reshape(-1)
    preplaced = _cpu_tensor(case.preplaced_mask, torch.bool).reshape(-1)
    if fixed.numel() != int(case.n) or preplaced.numel() != int(case.n):
        raise ValueError("fixed/preplaced masks must match case.n")
    targets = _cpu_tensor(case.target, torch.float64)
    if targets.shape != (int(case.n), 4):
        raise ValueError("case.target must have shape [N, 4]")
    hard = fixed | preplaced
    if bool(hard.any()):
        _validate_hard_targets(targets, hard)
    # Fixed blocks have exact dimensions, but their input x/y is not a
    # positional anchor.  Only preplaced targets determine an anchor span.
    anchor_bounds = _anchor_bounds(targets, preplaced)
    fixed_shape = _fixed_shape_limits(targets, fixed)

    pin_bounds = _point_bounds(pins)
    pin_centroid = pins.mean(dim=0) if pins.numel() else torch.zeros(2, dtype=torch.float64)
    pin_bbox_center = (
        torch.tensor(
            [
                0.5 * (pin_bounds[0] + pin_bounds[2]),
                0.5 * (pin_bounds[1] + pin_bounds[3]),
            ],
            dtype=torch.float64,
        )
        if pin_bounds is not None
        else torch.zeros(2, dtype=torch.float64)
    )

    pin_ratio = _safe_ratio(
        pin_bounds[2] - pin_bounds[0], pin_bounds[3] - pin_bounds[1]
    ) if pin_bounds is not None else 1.0
    anchor_ratio = _safe_ratio(
        anchor_bounds[2] - anchor_bounds[0], anchor_bounds[3] - anchor_bounds[1]
    ) if anchor_bounds is not None else 1.0
    reference_ratio = pin_ratio if pin_bounds is not None else anchor_ratio

    aspect_variants = _aspect_variants(
        pin_ratio, anchor_ratio, pin_bounds is not None, anchor_bounds is not None
    )
    centers = _center_modes(anchor_bounds, pins, pin_centroid, pin_bbox_center)

    candidates: list[OutlineHypothesis] = []
    for utilization in _UTILIZATION_PRIORS:
        envelope_area = block_area / utilization
        for aspect_label, raw_ratio in aspect_variants:
            ratio = _fit_ratio(raw_ratio, envelope_area, anchor_bounds, fixed_shape)
            if ratio is None:
                continue
            width = math.sqrt(envelope_area * ratio)
            height = envelope_area / width
            if not (math.isfinite(width) and math.isfinite(height)):
                continue
            for mode, center in centers:
                bounds = _place_bounds(width, height, center, anchor_bounds)
                if bounds is None:
                    continue
                x_left, y_bottom, x_right, y_top = bounds
                anchor_residual, anchor_coverage = _anchor_metrics(
                    targets, preplaced, bounds
                )
                if anchor_residual > 1.0e-8:
                    continue
                fixed_shape_residual = _fixed_shape_residual(width, height, fixed_shape)
                (
                    assignment,
                    pin_coverage,
                    side_coverage,
                    pin_residual,
                    side_residual,
                    side_counts,
                ) = _pin_metrics(pins, bounds)
                area_prior_residual = abs(utilization - 0.975) / 0.025
                aspect_residual = abs(math.log(max(ratio, _EPS) / max(reference_ratio, _EPS)))
                scores = {
                    "pin_residual": pin_residual,
                    "area_prior_residual": area_prior_residual,
                    "anchor_residual": anchor_residual,
                    "fixed_shape_residual": fixed_shape_residual,
                    "aspect_residual": aspect_residual,
                    "side_residual": side_residual,
                }
                score = (
                    pin_residual
                    + 0.25 * area_prior_residual
                    + 0.05 * aspect_residual
                    + 0.10 * side_residual
                )
                score = float(score)
                confidence = float(math.exp(-max(score, 0.0)))
                source = f"{mode}:{aspect_label}"
                provenance = (
                    "official_input",
                    "pins" if pin_bounds is not None else "no_pins",
                    f"mode:{mode}",
                    f"aspect:{aspect_label}",
                    "anchor_span" if anchor_bounds is not None else "no_preplaced_anchor",
                    "fixed_shape_fit" if fixed_shape is not None else "no_fixed_shape",
                    "area_utilization_prior",
                )
                hypothesis_id = _hypothesis_id(
                    source,
                    provenance,
                    bounds,
                    utilization,
                )
                candidates.append(
                    OutlineHypothesis(
                        hypothesis_id=hypothesis_id,
                        x_left=x_left,
                        x_right=x_right,
                        y_bottom=y_bottom,
                        y_top=y_top,
                        source=source,
                        provenance=provenance,
                        score=score,
                        scores=scores,
                        confidence=confidence,
                        pin_residual=pin_residual,
                        area_prior_residual=area_prior_residual,
                        anchor_residual=anchor_residual,
                        pin_side_assignment=assignment,
                        pin_side_coverage=pin_coverage,
                        anchor_coverage=anchor_coverage,
                        anchor_span=_span(anchor_bounds),
                        block_area=block_area,
                        utilization=float(
                            min(1.0, max(0.95, block_area / (width * height)))
                        ),
                        side_coverage=side_coverage,
                        pin_side_counts=side_counts,
                    )
                )

    if pin_bounds is not None:
        perimeter = _pin_perimeter_candidate(
            pins=pins,
            pin_bounds=pin_bounds,
            targets=targets,
            preplaced=preplaced,
            fixed_shape=fixed_shape,
            block_area=block_area,
        )
        if perimeter is not None:
            candidates.append(perimeter)

    candidates = _deduplicate(candidates)
    if not candidates:
        return ()

    ranked = sorted(candidates, key=_rank_key)
    selected: list[OutlineHypothesis] = []
    perimeter = next(
        (candidate for candidate in ranked if candidate.source == "pin_perimeter"),
        None,
    )
    if perimeter is not None:
        selected.append(perimeter)
    families: set[str] = set()
    if perimeter is not None:
        families.update(
            part for part in perimeter.provenance if part.startswith("family:")
        )
    for candidate in ranked:
        if candidate in selected:
            continue
        family = next(
            (
                part
                for part in candidate.provenance
                if part.startswith("family:") or part.startswith("aspect:")
            ),
            candidate.source,
        )
        if family in families:
            continue
        families.add(family)
        selected.append(candidate)
        if len(selected) >= max_hypotheses:
            break
    if len(selected) < max_hypotheses:
        selected_ids = {candidate.hypothesis_id for candidate in selected}
        for candidate in ranked:
            if candidate.hypothesis_id in selected_ids:
                continue
            selected.append(candidate)
            if len(selected) >= max_hypotheses:
                break
    return tuple(sorted(selected, key=_rank_key))


def _cpu_tensor(value: Any, dtype: torch.dtype) -> torch.Tensor:
    return torch.as_tensor(value).detach().to(device="cpu", dtype=dtype)


def _validate_hard_targets(targets: torch.Tensor, hard: torch.Tensor) -> None:
    selected = targets[hard]
    if not bool(torch.isfinite(selected).all()) or bool((selected[:, 2:] <= 0).any()):
        raise ValueError("fixed/preplaced target rectangles must be finite and positive")


def _anchor_bounds(
    targets: torch.Tensor, mask: torch.Tensor
) -> tuple[float, float, float, float] | None:
    if not bool(mask.any()):
        return None
    selected = targets[mask]
    left = float(selected[:, 0].min().item())
    bottom = float(selected[:, 1].min().item())
    right = float((selected[:, 0] + selected[:, 2]).max().item())
    top = float((selected[:, 1] + selected[:, 3]).max().item())
    return (left, bottom, right, top)


def _fixed_shape_limits(
    targets: torch.Tensor, fixed: torch.Tensor
) -> tuple[float, float] | None:
    """Return the largest required fixed ``(width, height)`` dimensions."""

    if not bool(fixed.any()):
        return None
    selected = targets[fixed, 2:4]
    return (float(selected[:, 0].max().item()), float(selected[:, 1].max().item()))


def _point_bounds(points: torch.Tensor) -> tuple[float, float, float, float] | None:
    if not points.numel():
        return None
    return (
        float(points[:, 0].min().item()),
        float(points[:, 1].min().item()),
        float(points[:, 0].max().item()),
        float(points[:, 1].max().item()),
    )


def _span(bounds: tuple[float, float, float, float] | None) -> tuple[float, float]:
    if bounds is None:
        return (0.0, 0.0)
    return (bounds[2] - bounds[0], bounds[3] - bounds[1])


def _safe_ratio(width: float, height: float) -> float:
    if width <= _EPS or height <= _EPS:
        return 1.0
    return float(min(max(width / height, 0.125), 8.0))


def _aspect_variants(
    pin_ratio: float,
    anchor_ratio: float,
    has_pins: bool,
    has_anchor: bool,
) -> tuple[tuple[str, float], ...]:
    raw: list[tuple[str, float]] = [
        ("square", 1.0),
        ("horizontal", 2.0),
        ("vertical", 0.5),
    ]
    if has_pins:
        raw.extend(
            [
                ("pin_spread", pin_ratio),
                ("pin_spread_inverse", 1.0 / max(pin_ratio, _EPS)),
            ]
        )
    if has_anchor:
        raw.append(("anchor_span", anchor_ratio))
    raw.extend([("wide", 1.5), ("tall", 2.0 / 3.0)])

    result: list[tuple[str, float]] = []
    seen: set[float] = set()
    for label, ratio in raw:
        ratio = float(min(max(ratio, 0.125), 8.0))
        key = round(ratio, 10)
        if key in seen:
            continue
        seen.add(key)
        result.append((label, ratio))
    return tuple(result)


def _center_modes(
    anchor_bounds: tuple[float, float, float, float] | None,
    pins: torch.Tensor,
    pin_centroid: torch.Tensor,
    pin_bbox_center: torch.Tensor,
) -> tuple[tuple[str, tuple[float, float]], ...]:
    centers: list[tuple[str, tuple[float, float]]] = []
    if anchor_bounds is not None:
        centers.append(
            (
                "anchor_center",
                (
                    0.5 * (anchor_bounds[0] + anchor_bounds[2]),
                    0.5 * (anchor_bounds[1] + anchor_bounds[3]),
                ),
            )
        )
    if pins.numel():
        centers.extend(
            [
                ("pin_centroid", (float(pin_centroid[0]), float(pin_centroid[1]))),
                ("pin_bbox", (float(pin_bbox_center[0]), float(pin_bbox_center[1]))),
            ]
        )
    if not centers:
        centers.append(("origin", (0.0, 0.0)))
    return tuple(centers)


def _fixed_shape_residual(
    width: float, height: float, fixed_shape: tuple[float, float] | None
) -> float:
    if fixed_shape is None:
        return 0.0
    return max(fixed_shape[0] - width, fixed_shape[1] - height, 0.0) / max(
        width, height, _EPS
    )


def _fit_ratio(
    raw_ratio: float,
    envelope_area: float,
    anchor_bounds: tuple[float, float, float, float] | None,
    fixed_shape: tuple[float, float] | None,
) -> float | None:
    ratio = float(min(max(raw_ratio, 0.125), 8.0))
    low = 0.125
    high = 8.0
    if anchor_bounds is None:
        pass
    else:
        anchor_width, anchor_height = _span(anchor_bounds)
        if anchor_width <= 0.0 or anchor_height <= 0.0:
            return None
        if envelope_area + 1.0e-9 < anchor_width * anchor_height:
            return None
        low = max(low, anchor_width * anchor_width / envelope_area)
        high = min(high, envelope_area / (anchor_height * anchor_height))
    if fixed_shape is not None:
        fixed_width, fixed_height = fixed_shape
        low = max(low, fixed_width * fixed_width / envelope_area)
        high = min(high, envelope_area / (fixed_height * fixed_height))
    if low > high + 1.0e-9:
        return None
    return float(min(max(ratio, low), high))


def _place_bounds(
    width: float,
    height: float,
    center: tuple[float, float],
    anchor_bounds: tuple[float, float, float, float] | None,
) -> tuple[float, float, float, float] | None:
    desired_left = center[0] - 0.5 * width
    desired_bottom = center[1] - 0.5 * height
    if anchor_bounds is None:
        return (desired_left, desired_bottom, desired_left + width, desired_bottom + height)

    anchor_left, anchor_bottom, anchor_right, anchor_top = anchor_bounds
    left_low = anchor_right - width
    left_high = anchor_left
    bottom_low = anchor_top - height
    bottom_high = anchor_bottom
    if left_low > left_high + 1.0e-9 or bottom_low > bottom_high + 1.0e-9:
        return None
    left = min(max(desired_left, left_low), left_high)
    bottom = min(max(desired_bottom, bottom_low), bottom_high)
    return (left, bottom, left + width, bottom + height)


def _anchor_metrics(
    targets: torch.Tensor,
    hard: torch.Tensor,
    bounds: tuple[float, float, float, float],
) -> tuple[float, float]:
    if not bool(hard.any()):
        return 0.0, 1.0
    selected = targets[hard]
    left, bottom, right, top = bounds
    violations = torch.stack(
        (
            left - selected[:, 0],
            bottom - selected[:, 1],
            selected[:, 0] + selected[:, 2] - right,
            selected[:, 1] + selected[:, 3] - top,
        ),
        dim=1,
    ).clamp_min(0.0)
    residual = float(violations.max().item()) / max(right - left, top - bottom, _EPS)
    coverage = float((violations.max(dim=1).values <= 1.0e-8).to(torch.float64).mean().item())
    return residual, coverage


def _pin_metrics(
    pins: torch.Tensor,
    bounds: tuple[float, float, float, float],
) -> tuple[
    tuple[str, ...],
    float,
    float,
    float,
    float,
    tuple[tuple[str, int], ...],
]:
    if not pins.numel():
        return (), 1.0, 0.0, 0.0, 0.0, ()
    left, bottom, right, top = bounds
    distances = torch.stack(
        (
            (pins[:, 0] - left).abs(),
            (right - pins[:, 0]).abs(),
            (pins[:, 1] - bottom).abs(),
            (top - pins[:, 1]).abs(),
        ),
        dim=1,
    )
    side_index = distances.argmin(dim=1)
    inside = (
        (pins[:, 0] >= left - 1.0e-8)
        & (pins[:, 0] <= right + 1.0e-8)
        & (pins[:, 1] >= bottom - 1.0e-8)
        & (pins[:, 1] <= top + 1.0e-8)
    )
    scale = max(right - left, top - bottom, _EPS)
    outside_x = (left - pins[:, 0]).clamp_min(0.0) + (pins[:, 0] - right).clamp_min(0.0)
    outside_y = (bottom - pins[:, 1]).clamp_min(0.0) + (pins[:, 1] - top).clamp_min(0.0)
    pin_coverage = float(inside.to(torch.float64).mean().item())
    counts_tensor = torch.bincount(side_index, minlength=4)
    side_coverage = float((counts_tensor > 0).to(torch.float64).mean().item())
    pin_residual = float(torch.sqrt(outside_x.square() + outside_y.square()).mean().item()) / scale
    side_residual = float(distances.gather(1, side_index[:, None]).mean().item()) / scale
    assignments = tuple(_SIDES[int(index)] for index in side_index.tolist())
    counts = tuple(
        (side, int(counts_tensor[index].item()))
        for index, side in enumerate(_SIDES)
        if int(counts_tensor[index])
    )
    return (
        assignments,
        float(pin_coverage),
        float(side_coverage),
        float(pin_residual),
        float(side_residual),
        counts,
    )


def _pin_perimeter_candidate(
    *,
    pins: torch.Tensor,
    pin_bounds: tuple[float, float, float, float],
    targets: torch.Tensor,
    preplaced: torch.Tensor,
    fixed_shape: tuple[float, float] | None,
    block_area: float,
) -> OutlineHypothesis | None:
    """Build the exact pin-coordinate envelope as a separate audit family."""

    left, bottom, right, top = pin_bounds
    width = right - left
    height = top - bottom
    # A point or collinear pin set has no positive geometric extent.  Preserve
    # its coordinate line and add only the smallest deterministic thickness
    # needed to represent a rectangle.
    minimum_extent = max(math.sqrt(block_area) * 1.0e-6, 1.0e-8)
    if width <= _EPS:
        left -= 0.5 * minimum_extent
        right += 0.5 * minimum_extent
        width = minimum_extent
    if height <= _EPS:
        bottom -= 0.5 * minimum_extent
        top += 0.5 * minimum_extent
        height = minimum_extent
    if fixed_shape is not None and (
        width + 1.0e-9 < fixed_shape[0] or height + 1.0e-9 < fixed_shape[1]
    ):
        return None
    bounds = (left, bottom, right, top)
    anchor_residual, anchor_coverage = _anchor_metrics(targets, preplaced, bounds)
    if anchor_residual > 1.0e-8:
        return None
    (
        assignment,
        pin_coverage,
        side_coverage,
        pin_residual,
        side_residual,
        side_counts,
    ) = _pin_metrics(pins, bounds)
    utilization = float(block_area / (width * height))
    area_prior_residual = abs(utilization - 0.975) / 0.025
    scores = {
        "pin_residual": pin_residual,
        "area_prior_residual": area_prior_residual,
        "anchor_residual": anchor_residual,
        "fixed_shape_residual": 0.0,
        "side_residual": side_residual,
    }
    score = float(
        pin_residual + 0.25 * area_prior_residual + 0.10 * side_residual
    )
    provenance = (
        "official_input",
        "pins",
        "coordinate_mode:pin_perimeter",
        "pin_bounds_exact",
        "anchor_span" if preplaced.any() else "no_preplaced_anchor",
        "family:pin_perimeter",
    )
    return OutlineHypothesis(
        hypothesis_id=_hypothesis_id("pin_perimeter", provenance, bounds, utilization),
        x_left=left,
        x_right=right,
        y_bottom=bottom,
        y_top=top,
        source="pin_perimeter",
        provenance=provenance,
        score=score,
        scores=scores,
        confidence=float(math.exp(-max(score, 0.0))),
        pin_residual=pin_residual,
        area_prior_residual=area_prior_residual,
        anchor_residual=anchor_residual,
        pin_side_assignment=assignment,
        pin_side_coverage=pin_coverage,
        anchor_coverage=anchor_coverage,
        anchor_span=_span(_anchor_bounds(targets, preplaced)),
        block_area=block_area,
        utilization=utilization,
        side_coverage=side_coverage,
        pin_side_counts=side_counts,
    )


def _hypothesis_id(
    source: str,
    provenance: Sequence[str],
    bounds: tuple[float, float, float, float],
    utilization: float,
) -> str:
    payload = "|".join(
        (
            source,
            *provenance,
            *(f"{value:.12g}" for value in bounds),
            f"{utilization:.12g}",
        )
    )
    digest = hashlib.sha256(payload.encode("ascii")).hexdigest()[:16]
    return f"outline-{digest}"


def _deduplicate(candidates: Sequence[OutlineHypothesis]) -> list[OutlineHypothesis]:
    unique: dict[tuple[float, ...], OutlineHypothesis] = {}
    for candidate in sorted(candidates, key=_rank_key):
        key = tuple(round(value, 10) for value in (*candidate.bounds, candidate.utilization))
        if key not in unique:
            unique[key] = candidate
    return list(unique.values())


def _rank_key(candidate: OutlineHypothesis) -> tuple[Any, ...]:
    return (
        round(candidate.score, 12),
        candidate.source,
        candidate.hypothesis_id,
    )


__all__ = [
    "OutlineHypothesis",
    "infer_outline_hypotheses",
]
