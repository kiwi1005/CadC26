"""Deterministic hard-feasible shelf fallback."""

from __future__ import annotations

from typing import Any

import torch

from hcfp.verify import OVERLAP_EPS, coordinate_tolerance, overlap_pairs, verify_feasible

try:  # Optional while case.py is supplied by another lane.
    from hcfp.case import FloorplanCase
except Exception:  # pragma: no cover
    FloorplanCase = Any  # type: ignore


def _field(source: Any, names: tuple[str, ...], default: Any = None) -> Any:
    if source is None:
        return default
    if isinstance(source, dict):
        for name in names:
            if name in source:
                return source[name]
        return default
    for name in names:
        if hasattr(source, name):
            return getattr(source, name)
    return default


def _mask(mask: Any, n: int) -> torch.Tensor:
    if mask is None:
        return torch.zeros(n, dtype=torch.bool)
    out = torch.as_tensor(mask, dtype=torch.bool, device="cpu").reshape(-1)
    if out.numel() != n:
        raise ValueError(f"mask length {out.numel()} does not match block count {n}")
    return out


def _constraint_masks(case: Any, n: int) -> tuple[torch.Tensor, torch.Tensor]:
    fixed_source = _field(case, ("fixed_mask", "is_fixed"))
    preplaced_source = _field(case, ("preplaced_mask", "is_preplaced"))
    constraints = _field(case, ("constraints", "target_constraints"))
    if constraints is not None:
        rows = torch.as_tensor(constraints, dtype=torch.long, device="cpu")
        if rows.ndim == 2 and rows.shape[0] >= n:
            if fixed_source is None and rows.shape[1] > 0:
                fixed_source = rows[:n, 0] != 0
            if preplaced_source is None and rows.shape[1] > 1:
                preplaced_source = rows[:n, 1] != 0
    return _mask(fixed_source, n), _mask(preplaced_source, n)


def _areas(case: Any) -> torch.Tensor:
    values = _field(case, ("area", "areas", "area_targets", "block_areas"))
    if values is None:
        n = _field(case, ("n", "num_blocks", "block_count"))
        if n is None:
            raise ValueError("case must provide area targets or block count")
        return torch.ones(int(n), dtype=torch.float64)
    area = torch.as_tensor(values, dtype=torch.float64, device="cpu").reshape(-1)
    if not bool((area > 0).all()):
        raise ValueError("area targets must be positive")
    return area


def _target(case: Any, n: int) -> torch.Tensor:
    values = _field(case, ("target", "targets", "target_positions", "preplaced_xywh"))
    if values is None:
        return torch.zeros((n, 4), dtype=torch.float64)
    return torch.as_tensor(values, dtype=torch.float64, device="cpu").reshape(n, 4)


def _fixed_wh(case: Any, target: torch.Tensor, n: int) -> torch.Tensor:
    values = _field(case, ("fixed_wh", "fixed_shape", "fixed_shapes"))
    if values is None:
        return target[:, 2:4]
    return torch.as_tensor(values, dtype=torch.float64, device="cpu").reshape(n, 2)


def _valid_shape(case: Any, i: int, area: torch.Tensor, fixed: torch.Tensor, preplaced: torch.Tensor, target: torch.Tensor, fixed_wh: torch.Tensor) -> tuple[float, float]:
    if bool(preplaced[i]):
        return float(target[i, 2]), float(target[i, 3])
    if bool(fixed[i]):
        return float(fixed_wh[i, 0]), float(fixed_wh[i, 1])
    side = float(torch.sqrt(area[i]).item())
    return side, side


def _max_right(boxes: torch.Tensor, mask: torch.Tensor) -> float:
    if not bool(mask.any()):
        return 0.0
    selected = boxes[mask]
    return float(torch.max(selected[:, 0] + selected[:, 2]).item())


def _choose_shelf_y(boxes: torch.Tensor, mask: torch.Tensor, gap: float) -> float:
    if not bool(mask.any()):
        return 0.0
    selected = boxes[mask]
    return float(torch.max(selected[:, 1] + selected[:, 3]).item()) + gap


def safe_shelf(case: FloorplanCase, *, gap: float | None = None) -> torch.Tensor:
    """Return deterministic ``(x, y, w, h)`` boxes with hard targets preserved.

    The fallback assumes hard anchors are internally valid. If preplaced anchors
    overlap each other, it raises instead of moving them, because moving them
    would violate the official hard target semantics.
    """

    target_source = _field(case, ("target", "targets", "target_positions", "preplaced_xywh"))
    area_source = _field(case, ("area", "areas", "area_targets", "block_areas"))
    out_dtype = (
        torch.as_tensor(target_source).dtype
        if target_source is not None and torch.as_tensor(target_source).is_floating_point()
        else torch.as_tensor(area_source).dtype
        if area_source is not None and torch.as_tensor(area_source).is_floating_point()
        else torch.float32
    )
    area = _areas(case)
    n = int(area.numel())
    target = _target(case, n)
    fixed, preplaced = _constraint_masks(case, n)
    fixed_wh = _fixed_wh(case, target, n)
    scale = float(_field(case, ("scale",), 1.0))
    raw_clearance = 1.0e-4 * max(scale, 1.0)
    if bool(_field(case, ("normalized",), False)):
        raw_clearance /= max(scale, 1.0e-30)
    clearance = float(gap) if gap is not None else raw_clearance

    pos = torch.zeros((n, 4), dtype=torch.float64)
    for i in range(n):
        w, h = _valid_shape(case, i, area, fixed, preplaced, target, fixed_wh)
        pos[i, 2] = w
        pos[i, 3] = h
    pos[preplaced] = target[preplaced]

    overlap_eps = coordinate_tolerance(case, OVERLAP_EPS)
    if overlap_pairs(pos[preplaced], eps=overlap_eps):
        raise ValueError("preplaced anchors overlap; hard-feasible fallback is impossible")

    x = _max_right(pos, preplaced) + clearance
    y = _choose_shelf_y(pos, preplaced, clearance)
    for i in sorted(int(i) for i in torch.nonzero(~preplaced, as_tuple=False).reshape(-1).tolist()):
        pos[i, 0] = x
        pos[i, 1] = y
        x += float(pos[i, 2]) + clearance

    shift = 0
    while not verify_feasible(case, pos):
        if shift >= n + 2:
            raise RuntimeError("safe shelf fallback could not produce a hard-feasible placement")
        movable = ~preplaced
        width = float(torch.sum(pos[movable, 2]).item()) if bool(movable.any()) else 0.0
        pos[movable, 0] += 2.0 * max(width, 1.0) + clearance
        shift += 1
    return pos.to(dtype=out_dtype)


def deterministic_shelf_fallback(case: FloorplanCase, **kwargs: Any) -> torch.Tensor:
    return safe_shelf(case, **kwargs)


def safe_fallback(case: Any) -> torch.Tensor:
    """Accept either a normalized ``FloorplanCase`` or official solve inputs."""

    if bool(_field(case, ("normalized",), False)):
        return safe_shelf(case)

    from hcfp.case import from_official

    normalized = from_official(
        _field(case, ("block_count", "n")),
        _field(case, ("area_targets", "area")),
        _field(case, ("b2b_connectivity", "b2b_weight"), []),
        _field(case, ("p2b_connectivity", "p2b_edges"), []),
        _field(case, ("pins_pos", "pins"), []),
        _field(case, ("constraints",), []),
        _field(case, ("target_positions", "target")),
    )
    placed = safe_shelf(normalized).to(dtype=torch.float64)
    placed[:, :2] = placed[:, :2] * normalized.scale + normalized.origin.to(torch.float64)
    placed[:, 2:4] *= normalized.scale

    raw_target = _field(case, ("target_positions", "target"))
    if raw_target is not None:
        target = torch.as_tensor(raw_target, dtype=torch.float64, device="cpu")[: normalized.n]
        placed[normalized.preplaced_mask] = target[normalized.preplaced_mask]
        hard_shape = normalized.fixed_mask | normalized.preplaced_mask
        placed[hard_shape, 2:4] = target[hard_shape, 2:4]
    return placed
