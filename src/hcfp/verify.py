"""Official-v10-style exact geometry verification primitives."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import torch

try:  # Keep this optional while case.py is still owned elsewhere.
    from hcfp.case import FloorplanCase
except Exception:  # pragma: no cover - import surface may not exist yet.
    FloorplanCase = Any  # type: ignore


OVERLAP_EPS = 1.0e-6
SOFT_AREA_REL_TOL = 1.0e-2
HARD_TARGET_TOL = 1.0e-4
BOUNDARY_TOL = 1.0e-6
BOUNDARY_LEFT = 1
BOUNDARY_RIGHT = 2
BOUNDARY_TOP = 4
BOUNDARY_BOTTOM = 8
ALPHA = 0.5
BETA = 2.0
GAMMA = 0.3
INFEASIBLE_COST = 10.0


@dataclass(frozen=True)
class Verification:
    feasible: bool
    overlap_pairs: tuple[tuple[int, int], ...]
    area_bad: tuple[int, ...]
    fixed_bad: tuple[int, ...]
    preplaced_bad: tuple[int, ...]


@dataclass(frozen=True)
class SoftViolations:
    boundary: float
    grouping: float
    mib: float
    raw_boundary: int
    raw_grouping: int
    raw_mib: int
    maximum: int

    @property
    def total(self) -> float:
        return self.boundary + self.grouping + self.mib

    @property
    def raw_total(self) -> int:
        return self.raw_boundary + self.raw_grouping + self.raw_mib


@dataclass(frozen=True)
class ExactMetrics:
    verification: Verification
    hpwl_total: float
    bbox_area: float
    hpwl_gap: float
    area_gap: float
    soft: SoftViolations
    runtime_factor: float
    cost: float


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


def as_xywh(xywh: Any) -> torch.Tensor:
    boxes = torch.as_tensor(xywh, dtype=torch.float64, device="cpu")
    if boxes.ndim != 2 or boxes.shape[1] != 4:
        raise ValueError("xywh must have shape [N, 4]")
    if not bool(torch.isfinite(boxes).all()):
        raise ValueError("xywh must be finite")
    if not bool((boxes[:, 2:] > 0).all()):
        raise ValueError("rectangle width and height must be positive")
    return boxes


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


def coordinate_tolerance(case: Any, raw_tolerance: float) -> float:
    """Map an official raw-coordinate tolerance into the case coordinate space."""

    if bool(_field(case, ("normalized",), False)):
        scale = float(_field(case, ("scale",), 1.0))
        return raw_tolerance / max(scale, 1.0e-30)
    return raw_tolerance


def _areas(case: Any, n: int) -> torch.Tensor:
    values = _field(case, ("area", "areas", "area_targets", "block_areas"))
    if values is None:
        raise ValueError("case area targets are required")
    out = torch.as_tensor(values, dtype=torch.float64, device="cpu").reshape(-1)
    if out.numel() != n:
        raise ValueError(f"area length {out.numel()} does not match block count {n}")
    return out


def _target(case: Any, n: int) -> torch.Tensor | None:
    values = _field(case, ("target", "targets", "target_positions", "preplaced_xywh"))
    if values is None:
        return None
    out = torch.as_tensor(values, dtype=torch.float64, device="cpu").reshape(n, 4)
    return out


def _fixed_wh(case: Any, target: torch.Tensor | None, n: int) -> torch.Tensor | None:
    values = _field(case, ("fixed_wh", "fixed_shape", "fixed_shapes"))
    if values is not None:
        return torch.as_tensor(values, dtype=torch.float64, device="cpu").reshape(n, 2)
    if target is not None:
        return target[:, 2:4]
    return None


def centers(xywh: Any) -> torch.Tensor:
    boxes = as_xywh(xywh)
    return boxes[:, :2] + 0.5 * boxes[:, 2:4]


def bbox(xywh: Any) -> tuple[float, float, float, float]:
    boxes = as_xywh(xywh)
    left = torch.min(boxes[:, 0])
    bottom = torch.min(boxes[:, 1])
    right = torch.max(boxes[:, 0] + boxes[:, 2])
    top = torch.max(boxes[:, 1] + boxes[:, 3])
    return (float(left), float(bottom), float(right), float(top))


def bbox_area(xywh: Any) -> float:
    left, bottom, right, top = bbox(xywh)
    return max(0.0, right - left) * max(0.0, top - bottom)


def pair_overlap_area(a: torch.Tensor, b: torch.Tensor) -> float:
    dx = min(float(a[0] + a[2]), float(b[0] + b[2])) - max(float(a[0]), float(b[0]))
    dy = min(float(a[1] + a[3]), float(b[1] + b[3])) - max(float(a[1]), float(b[1]))
    return max(0.0, dx) * max(0.0, dy)


def overlap_pairs(xywh: Any, eps: float = OVERLAP_EPS) -> tuple[tuple[int, int], ...]:
    boxes = as_xywh(xywh)
    pairs: list[tuple[int, int]] = []
    for i in range(int(boxes.shape[0])):
        for j in range(i + 1, int(boxes.shape[0])):
            a, b = boxes[i], boxes[j]
            overlap_x = min(float(a[0] + a[2]), float(b[0] + b[2])) - max(float(a[0]), float(b[0]))
            overlap_y = min(float(a[1] + a[3]), float(b[1] + b[3])) - max(float(a[1]), float(b[1]))
            if overlap_x > eps and overlap_y > eps:
                pairs.append((i, j))
    return tuple(pairs)


def area_bad_blocks(case: Any, xywh: Any, rel_tol: float = SOFT_AREA_REL_TOL) -> tuple[int, ...]:
    boxes = as_xywh(xywh)
    n = int(boxes.shape[0])
    target_area = _areas(case, n)
    fixed, preplaced = _constraint_masks(case, n)
    actual = boxes[:, 2] * boxes[:, 3]
    rel = torch.abs(actual - target_area) / torch.clamp(target_area, min=1.0e-300)
    bad = (rel > rel_tol) & ~fixed & ~preplaced
    return tuple(int(i) for i in torch.nonzero(bad, as_tuple=False).reshape(-1).tolist())


def fixed_bad_blocks(case: Any, xywh: Any, tol: float = HARD_TARGET_TOL) -> tuple[int, ...]:
    boxes = as_xywh(xywh)
    n = int(boxes.shape[0])
    fixed, _ = _constraint_masks(case, n)
    target = _target(case, n)
    fixed_wh = _fixed_wh(case, target, n)
    if fixed_wh is None:
        return ()
    delta = torch.max(torch.abs(boxes[:, 2:4] - fixed_wh), dim=1).values
    bad = fixed & (delta > tol)
    return tuple(int(i) for i in torch.nonzero(bad, as_tuple=False).reshape(-1).tolist())


def preplaced_bad_blocks(case: Any, xywh: Any, tol: float = HARD_TARGET_TOL) -> tuple[int, ...]:
    boxes = as_xywh(xywh)
    n = int(boxes.shape[0])
    _, preplaced = _constraint_masks(case, n)
    target = _target(case, n)
    if target is None:
        return ()
    delta = torch.max(torch.abs(boxes - target), dim=1).values
    bad = preplaced & (delta > tol)
    return tuple(int(i) for i in torch.nonzero(bad, as_tuple=False).reshape(-1).tolist())


def boundary_bitmask(xywh: Any, tol: float = BOUNDARY_TOL) -> torch.Tensor:
    boxes = as_xywh(xywh)
    left, bottom, right, top = bbox(boxes)
    bits = torch.zeros(int(boxes.shape[0]), dtype=torch.int64)
    bits |= torch.where(torch.abs(boxes[:, 0] - left) < tol, BOUNDARY_LEFT, 0)
    bits |= torch.where(torch.abs(boxes[:, 0] + boxes[:, 2] - right) < tol, BOUNDARY_RIGHT, 0)
    bits |= torch.where(torch.abs(boxes[:, 1] + boxes[:, 3] - top) < tol, BOUNDARY_TOP, 0)
    bits |= torch.where(torch.abs(boxes[:, 1] - bottom) < tol, BOUNDARY_BOTTOM, 0)
    return bits


def boundary_missing(case: Any, xywh: Any) -> torch.Tensor:
    boxes = as_xywh(xywh)
    n = int(boxes.shape[0])
    required = _field(case, ("boundary_codes", "boundary_mask", "boundary_bits"))
    if required is None:
        return torch.zeros(n, dtype=torch.int64)
    req = torch.as_tensor(required, dtype=torch.int64, device="cpu")
    if req.ndim == 2 and req.shape[1] == 4:
        req = req[:, 0] * 1 + req[:, 1] * 2 + req[:, 2] * 4 + req[:, 3] * 8
    req = req.reshape(-1)
    if req.numel() != n:
        raise ValueError(f"boundary length {req.numel()} does not match block count {n}")
    actual = boundary_bitmask(boxes, tol=coordinate_tolerance(case, BOUNDARY_TOL))
    return req & ~actual


def _edge_connected(a: torch.Tensor, b: torch.Tensor, tol: float = 0.0) -> bool:
    ax1, ay1, aw, ah = (float(v) for v in a.tolist())
    bx1, by1, bw, bh = (float(v) for v in b.tolist())
    ax2, ay2 = ax1 + aw, ay1 + ah
    bx2, by2 = bx1 + bw, by1 + bh
    x_overlap = min(ax2, bx2) - max(ax1, bx1)
    y_overlap = min(ay2, by2) - max(ay1, by1)
    if x_overlap > tol and (abs(ay2 - by1) <= tol or abs(by2 - ay1) <= tol):
        return True
    if y_overlap > tol and (abs(ax2 - bx1) <= tol or abs(bx2 - ax1) <= tol):
        return True
    return x_overlap > tol and y_overlap > tol


def connected_components_for_group(xywh: Any, members: Any, *, tol: float = 0.0) -> int:
    boxes = as_xywh(xywh)
    member_idx = [int(i) for i in torch.nonzero(torch.as_tensor(members, dtype=torch.bool).reshape(-1), as_tuple=False).reshape(-1).tolist()]
    if not member_idx:
        return 0
    seen: set[int] = set()
    comps = 0
    for start in member_idx:
        if start in seen:
            continue
        comps += 1
        stack = [start]
        seen.add(start)
        while stack:
            cur = stack.pop()
            for other in member_idx:
                if other not in seen and _edge_connected(boxes[cur], boxes[other], tol=tol):
                    seen.add(other)
                    stack.append(other)
    return comps


def grouping_violation(case: Any, xywh: Any) -> int:
    groups = _field(case, ("group_membership", "groups", "cluster_membership"))
    if groups is None:
        return 0
    membership = torch.as_tensor(groups, dtype=torch.bool, device="cpu")
    if membership.ndim != 2:
        raise ValueError("group membership must have shape [G, N]")
    total = 0
    for row in membership:
        comps = connected_components_for_group(xywh, row)
        total += max(0, comps - 1)
    return int(total)


def mib_shape_keys(xywh: Any, *, scale: float = 1.0) -> tuple[tuple[float, float], ...]:
    boxes = as_xywh(xywh)
    return tuple((round(float(w) * scale, 4), round(float(h) * scale, 4)) for w, h in boxes[:, 2:4].tolist())


def mib_violation(case: Any, xywh: Any) -> int:
    mib = _field(case, ("mib_membership", "mibs", "mib_groups"))
    if mib is None:
        return 0
    membership = torch.as_tensor(mib, dtype=torch.bool, device="cpu")
    if membership.ndim != 2:
        raise ValueError("MIB membership must have shape [M, N]")
    scale = float(_field(case, ("scale",), 1.0)) if bool(_field(case, ("normalized",), False)) else 1.0
    keys = mib_shape_keys(xywh, scale=scale)
    total = 0
    for row in membership:
        idx = [int(i) for i in torch.nonzero(row, as_tuple=False).reshape(-1).tolist()]
        if idx:
            total += max(0, len({keys[i] for i in idx}) - 1)
    return int(total)


def b2b_hpwl(xywh: Any, b2b_weight: Any) -> float:
    c = centers(xywh)
    w = torch.as_tensor(b2b_weight, dtype=torch.float64, device="cpu")
    if w.ndim != 2 or w.shape[0] != w.shape[1] or w.shape[0] != c.shape[0]:
        raise ValueError("b2b_weight must have shape [N, N]")
    dist = torch.abs(c[:, None, :] - c[None, :, :]).sum(dim=2)
    return float(torch.sum(torch.triu(w, diagonal=1) * dist).item())


def b2b_edge_hpwl(xywh: Any, b2b_edges: Any) -> float:
    c = centers(xywh)
    edges = torch.as_tensor(b2b_edges, dtype=torch.float64, device="cpu")
    if edges.numel() == 0:
        return 0.0
    if edges.ndim != 2 or edges.shape[1] < 3:
        raise ValueError("b2b_edges must have shape [E, >=3]")
    valid = (edges[:, 0] >= 0) & (edges[:, 1] >= 0)
    edges = edges[valid]
    if not edges.numel():
        return 0.0
    source = edges[:, 0].to(torch.long)
    target = edges[:, 1].to(torch.long)
    weight = edges[:, 2]
    distance = torch.abs(c[source] - c[target]).sum(dim=1)
    return float(torch.sum(weight * distance).item())


def p2b_hpwl(xywh: Any, p2b_edges: Any, pins: Any) -> float:
    c = centers(xywh)
    edges = torch.as_tensor(p2b_edges, dtype=torch.float64, device="cpu")
    pin_xy = torch.as_tensor(pins, dtype=torch.float64, device="cpu")
    if edges.numel() == 0:
        return 0.0
    if edges.ndim != 2 or edges.shape[1] != 3:
        raise ValueError("p2b_edges must have shape [E, 3] as pin, block, weight")
    valid = (edges[:, 0] >= 0) & (edges[:, 1] >= 0)
    edges = edges[valid]
    if not edges.numel():
        return 0.0
    pin_idx = edges[:, 0].to(torch.long)
    block_idx = edges[:, 1].to(torch.long)
    weight = edges[:, 2]
    dist = torch.abs(pin_xy[pin_idx] - c[block_idx]).sum(dim=1)
    return float(torch.sum(weight * dist).item())


def total_hpwl(case: Any, xywh: Any) -> float:
    b2b_dense = _field(case, ("b2b_weight", "b2b_weights", "net_weight_matrix"))
    b2b_edges = _field(case, ("b2b_connectivity",))
    p2b = _field(case, ("p2b_edges", "p2b_connectivity"))
    pins = _field(case, ("pins", "pins_pos", "pin_positions"))
    total = 0.0
    if b2b_dense is not None:
        total += b2b_hpwl(xywh, b2b_dense)
    elif b2b_edges is not None:
        total += b2b_edge_hpwl(xywh, b2b_edges)
    if p2b is not None and pins is not None:
        total += p2b_hpwl(xywh, p2b, pins)
    if bool(_field(case, ("normalized",), False)):
        total *= float(_field(case, ("scale",), 1.0))
    return total


def soft_violation_normalized(case: Any, xywh: Any) -> SoftViolations:
    boxes = as_xywh(xywh)
    n = max(1, int(boxes.shape[0]))
    missing = boundary_missing(case, boxes)
    required = torch.as_tensor(
        _field(case, ("boundary_codes", "boundary_mask", "boundary_bits"), torch.zeros(n)),
        dtype=torch.int64,
    )
    if required.ndim == 2 and required.shape[1] == 4:
        required = required[:, 0] + 2 * required[:, 1] + 4 * required[:, 2] + 8 * required[:, 3]
    raw_boundary = int(torch.count_nonzero(missing).item())
    maximum = int(torch.count_nonzero(required).item())

    groups = _field(case, ("group_membership", "groups", "cluster_membership"))
    raw_grouping = grouping_violation(case, boxes)
    if groups is not None:
        membership = torch.as_tensor(groups, dtype=torch.bool)
        maximum += int(torch.clamp(membership.sum(dim=1) - 1, min=0).sum().item())

    mib = _field(case, ("mib_membership", "mibs", "mib_groups"))
    raw_mib = mib_violation(case, boxes)
    if mib is not None:
        membership = torch.as_tensor(mib, dtype=torch.bool)
        maximum += int(torch.clamp(membership.sum(dim=1) - 1, min=0).sum().item())

    denominator = max(maximum, 1)
    return SoftViolations(
        boundary=raw_boundary / denominator,
        grouping=raw_grouping / denominator,
        mib=raw_mib / denominator,
        raw_boundary=raw_boundary,
        raw_grouping=raw_grouping,
        raw_mib=raw_mib,
        maximum=maximum,
    )


def verify(case: Any, xywh: Any) -> Verification:
    boxes = as_xywh(xywh)
    overlaps = overlap_pairs(boxes, eps=coordinate_tolerance(case, OVERLAP_EPS))
    areas = area_bad_blocks(case, boxes)
    hard_tolerance = coordinate_tolerance(case, HARD_TARGET_TOL)
    fixed = fixed_bad_blocks(case, boxes, tol=hard_tolerance)
    preplaced = preplaced_bad_blocks(case, boxes, tol=hard_tolerance)
    return Verification(
        feasible=not overlaps and not areas and not fixed and not preplaced,
        overlap_pairs=overlaps,
        area_bad=areas,
        fixed_bad=fixed,
        preplaced_bad=preplaced,
    )


def verify_feasible(case: Any, xywh: Any) -> bool:
    return verify(case, xywh).feasible


def compute_cost(
    hpwl_gap: float,
    area_gap: float,
    violations_relative: float,
    runtime_factor: float,
    feasible: bool,
) -> float:
    if not feasible:
        return INFEASIBLE_COST
    quality = 1.0 + ALPHA * (max(0.0, hpwl_gap) + max(0.0, area_gap))
    violation = math.exp(BETA * violations_relative)
    runtime = max(0.7, math.pow(max(0.01, runtime_factor), GAMMA))
    return min(quality * violation * runtime, INFEASIBLE_COST - 1.0e-6)


def exact_metrics(
    case: Any,
    xywh: Any,
    *,
    baseline_hpwl: float,
    baseline_area: float,
    runtime_seconds: float = 1.0,
    median_runtime: float = 1.0,
) -> ExactMetrics:
    hard = verify(case, xywh)
    hpwl = total_hpwl(case, xywh)
    layout_area = bbox_area(xywh)
    if bool(_field(case, ("normalized",), False)):
        layout_area *= float(_field(case, ("scale",), 1.0)) ** 2
    hpwl_gap = (hpwl - baseline_hpwl) / max(baseline_hpwl, 1.0e-6)
    area_gap = (layout_area - baseline_area) / max(baseline_area, 1.0e-6)
    soft = soft_violation_normalized(case, xywh)
    runtime_factor = runtime_seconds / max(median_runtime, 0.01)
    cost = compute_cost(hpwl_gap, area_gap, soft.total, runtime_factor, hard.feasible)
    return ExactMetrics(
        verification=hard,
        hpwl_total=hpwl,
        bbox_area=layout_area,
        hpwl_gap=hpwl_gap,
        area_gap=area_gap,
        soft=soft,
        runtime_factor=runtime_factor,
        cost=cost,
    )


def compute_total_score(costs: list[float], block_counts: list[int]) -> float:
    if not costs:
        return 0.0
    if len(costs) != len(block_counts):
        raise ValueError("costs and block_counts must have equal length")
    if not block_counts or all(count == 0 for count in block_counts):
        return sum(costs) / len(costs)
    max_blocks = max(block_counts)
    weights = [math.exp((count - max_blocks) / 12.0) for count in block_counts]
    return sum(cost * weight for cost, weight in zip(costs, weights)) / sum(weights)
