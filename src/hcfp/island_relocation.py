"""Bounded rigid translations for spatially fragmented ``xywh`` placements."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import torch

from hcfp.geometry import centers_from_xywh, overlap_area_matrix
from hcfp.verify import OVERLAP_EPS, as_xywh, bbox_area, overlap_pairs, total_hpwl


Tensor = torch.Tensor
_SIDES = ("left", "right", "top", "bottom")


@dataclass(frozen=True)
class RelocationCandidate:
    """One collision-free rigid translation of a spatial component."""

    placement: Tensor
    members: tuple[int, ...]
    delta: tuple[float, float]
    strategy: str
    side: str | None
    bbox_area: float
    hpwl: float | None

    @property
    def island(self) -> tuple[int, ...]:
        return self.members


@dataclass(frozen=True)
class IslandRelocationResult:
    """Selected candidate and the bounded candidate population."""

    placement: Tensor
    islands: tuple[tuple[int, ...], ...]
    core: tuple[int, ...]
    candidates: tuple[RelocationCandidate, ...]
    selected: RelocationCandidate | None

    @property
    def moved(self) -> bool:
        return self.selected is not None

    @property
    def boxes(self) -> Tensor:
        return self.placement


def detect_islands(
    placements: Any, *, proximity: float = 0.0
) -> tuple[tuple[int, ...], ...]:
    """Find components joined by contact, overlap, or bounded proximity."""

    _nonnegative(proximity, "proximity")
    boxes = _boxes(placements)
    adjacency = [[] for _ in range(len(boxes))]
    overlap = overlap_area_matrix(boxes)
    for first in range(len(boxes)):
        for second in range(first + 1, len(boxes)):
            if _connected(
                boxes[first], boxes[second], proximity, float(overlap[first, second])
            ):
                adjacency[first].append(second)
                adjacency[second].append(first)

    result: list[tuple[int, ...]] = []
    seen: set[int] = set()
    for start in range(len(boxes)):
        if start in seen:
            continue
        stack, members = [start], []
        seen.add(start)
        while stack:
            current = stack.pop()
            members.append(current)
            for other in adjacency[current]:
                if other not in seen:
                    seen.add(other)
                    stack.append(other)
        result.append(tuple(sorted(members)))
    return tuple(result)


def generate_island_relocations(
    case: Any | None = None,
    placements: Any | None = None,
    *,
    components: Any | None = None,
    proximity: float = 0.0,
    max_candidates: int = 16,
    preplaced_mask: Any | None = None,
    overlap_tolerance: float = OVERLAP_EPS,
) -> tuple[RelocationCandidate, ...]:
    """Generate bounded core-abutment and pin-median rigid translations.

    ``case`` follows the existing ``FloorplanCase``/``verify`` field names,
    but is optional for placement-only use.  Explicit components may be
    member sequences or a boolean ``[G,N]`` membership matrix; omitted blocks
    become singleton components.  A component containing a preplaced block is
    never moved.
    """

    source, boxes = _source(case, placements)
    _nonnegative(proximity, "proximity")
    _nonnegative(overlap_tolerance, "overlap_tolerance")
    if (
        not isinstance(max_candidates, int)
        or isinstance(max_candidates, bool)
        or max_candidates < 0
    ):
        raise ValueError("max_candidates must be a non-negative integer")
    if not max_candidates:
        return ()

    islands = _components(boxes, components, proximity)
    core = _core(islands, boxes)
    protected = _preplaced(source, len(boxes), preplaced_mask)
    found: list[RelocationCandidate] = []
    seen: set[tuple[tuple[int, ...], tuple[float, float]]] = set()
    for island in islands:
        if island == core or bool(protected[list(island)].any()):
            continue
        for side in _SIDES:
            _add(
                found,
                seen,
                _make_candidate(
                    source,
                    boxes,
                    island,
                    _abut(boxes, core, island, side),
                    strategy="core_abutment",
                    side=side,
                    protected=protected,
                    overlap_tolerance=overlap_tolerance,
                ),
            )
        _add(
            found,
            seen,
            _make_candidate(
                source,
                boxes,
                island,
                _pin_delta(source, boxes, island),
                strategy="pin_weighted_median",
                side=None,
                protected=protected,
                overlap_tolerance=overlap_tolerance,
            ),
        )
    found.sort(key=_sort_key)
    return tuple(found[:max_candidates])


def relocate_islands(
    case: Any | None = None,
    placements: Any | None = None,
    *,
    components: Any | None = None,
    proximity: float = 0.0,
    max_candidates: int = 16,
    preplaced_mask: Any | None = None,
    overlap_tolerance: float = OVERLAP_EPS,
) -> IslandRelocationResult:
    """Return the valid candidate with the smallest bbox area, if any."""

    source, boxes = _source(case, placements)
    islands = _components(boxes, components, proximity)
    candidates = generate_island_relocations(
        source,
        boxes,
        components=islands,
        proximity=proximity,
        max_candidates=max_candidates,
        preplaced_mask=preplaced_mask,
        overlap_tolerance=overlap_tolerance,
    )
    selected = candidates[0] if candidates else None
    return IslandRelocationResult(
        selected.placement.clone() if selected else boxes.clone(),
        islands,
        _core(islands, boxes),
        candidates,
        selected,
    )


island_relocation_oracle = relocate_islands


def _make_candidate(
    case: Any | None,
    boxes: Tensor,
    members: tuple[int, ...],
    delta: Tensor | None,
    *,
    strategy: str,
    side: str | None,
    protected: Tensor,
    overlap_tolerance: float,
) -> RelocationCandidate | None:
    if delta is None or bool(protected[list(members)].any()):
        return None
    delta = torch.as_tensor(delta, dtype=torch.float64).reshape(-1)
    if (
        delta.numel() != 2
        or not bool(torch.isfinite(delta).all())
        or float(torch.linalg.vector_norm(delta)) <= 1.0e-9
    ):
        return None
    moved = boxes.clone()
    moved[list(members), :2] += delta
    if overlap_pairs(moved, eps=overlap_tolerance):
        return None
    try:
        hpwl = float(total_hpwl(case, moved)) if case is not None else None
    except (AttributeError, TypeError, ValueError):
        hpwl = None
    return RelocationCandidate(
        moved,
        members,
        (float(delta[0]), float(delta[1])),
        strategy,
        side,
        bbox_area(moved),
        hpwl,
    )


def _add(
    found: list[RelocationCandidate],
    seen: set[tuple[tuple[int, ...], tuple[float, float]]],
    candidate: RelocationCandidate | None,
) -> None:
    if candidate is None:
        return
    key = (candidate.members, tuple(round(value, 12) for value in candidate.delta))
    if key not in seen:
        seen.add(key)
        found.append(candidate)


def _sort_key(candidate: RelocationCandidate) -> tuple[Any, ...]:
    return (
        candidate.bbox_area,
        math.inf if candidate.hpwl is None else candidate.hpwl,
        math.hypot(*candidate.delta),
        candidate.members,
        candidate.strategy,
        candidate.side or "",
    )


def _abut(
    boxes: Tensor, core: tuple[int, ...], island: tuple[int, ...], side: str
) -> Tensor:
    left, bottom, right, top = _bounds(boxes, core)
    i_left, i_bottom, i_right, i_top = _bounds(boxes, island)
    cx, cy = (left + right) / 2.0, (bottom + top) / 2.0
    ix, iy = (i_left + i_right) / 2.0, (i_bottom + i_top) / 2.0
    if side == "left":
        return boxes.new_tensor((left - i_right, cy - iy))
    if side == "right":
        return boxes.new_tensor((right - i_left, cy - iy))
    if side == "top":
        return boxes.new_tensor((cx - ix, top - i_bottom))
    if side == "bottom":
        return boxes.new_tensor((cx - ix, bottom - i_top))
    raise ValueError(f"unsupported side: {side}")


def _pin_delta(
    case: Any | None, boxes: Tensor, members: tuple[int, ...]
) -> Tensor | None:
    if case is None:
        return None
    raw_edges, raw_pins = (
        _field(case, ("p2b_edges", "p2b_connectivity")),
        _field(case, ("pins", "pins_pos", "pin_positions")),
    )
    if raw_edges is None or raw_pins is None:
        return None
    edges = torch.as_tensor(raw_edges, dtype=torch.float64)
    pins = torch.as_tensor(raw_pins, dtype=torch.float64)
    if not edges.numel():
        return None
    if edges.ndim != 2 or edges.shape[1] < 3 or pins.ndim != 2 or pins.shape[1] != 2:
        raise ValueError("p2b_edges must have shape [E, >=3] and pins [P, 2]")
    if not bool(torch.isfinite(edges).all() and torch.isfinite(pins).all()):
        raise ValueError("pin data must be finite")
    member_mask = torch.zeros(len(boxes), dtype=torch.bool)
    member_mask[list(members)] = True
    valid = (
        (edges[:, 0] >= 0)
        & (edges[:, 0] < len(pins))
        & (edges[:, 1] >= 0)
        & (edges[:, 1] < len(boxes))
        & (edges[:, 2] > 0)
    )
    valid &= member_mask[edges[:, 1].clamp(0, len(boxes) - 1).long()]
    if not bool(valid.any()):
        return None
    selected = edges[valid]
    pin_index, block_index, weights = (
        selected[:, 0].long(),
        selected[:, 1].long(),
        selected[:, 2],
    )
    offsets = pins[pin_index] - centers_from_xywh(boxes)[block_index]
    return torch.stack(
        tuple(_weighted_median(offsets[:, axis], weights) for axis in range(2))
    )


def _weighted_median(values: Tensor, weights: Tensor) -> Tensor:
    order = torch.argsort(values, stable=True)
    cumulative = weights[order].cumsum(0)
    index = int(torch.nonzero(cumulative >= cumulative[-1] * 0.5, as_tuple=False)[0])
    return values[order[index]]


def _components(
    boxes: Tensor, explicit: Any | None, proximity: float
) -> tuple[tuple[int, ...], ...]:
    if explicit is None:
        return detect_islands(boxes, proximity=proximity)
    n = len(boxes)
    try:
        value = torch.as_tensor(explicit)
    except (TypeError, ValueError):
        value = None
    if value is not None and value.dtype == torch.bool:
        if value.ndim == 1 and value.shape[0] == n:
            raw = [torch.nonzero(value).reshape(-1).tolist()]
        elif value.ndim == 2 and value.shape[1] == n:
            raw = [torch.nonzero(row).reshape(-1).tolist() for row in value]
        else:
            raise ValueError("boolean components must have shape [N] or [G,N]")
    else:
        raw = explicit.tolist() if isinstance(explicit, Tensor) else list(explicit)
        if raw and all(
            isinstance(item, (int, float)) and not isinstance(item, bool)
            for item in raw
        ):
            raw = [raw]
    if not raw:
        raise ValueError("components must not be empty")
    assigned: set[int] = set()
    result: list[tuple[int, ...]] = []
    for component in raw:
        members = tuple(sorted({_index(item, n) for item in component}))
        if not members or assigned.intersection(members):
            raise ValueError("components must be non-empty and disjoint")
        assigned.update(members)
        result.append(members)
    result.extend((index,) for index in range(n) if index not in assigned)
    return tuple(sorted(result))


def _core(islands: tuple[tuple[int, ...], ...], boxes: Tensor) -> tuple[int, ...]:
    return min(
        islands,
        key=lambda part: (
            -sum(float(boxes[i, 2] * boxes[i, 3]) for i in part),
            -len(part),
            part,
        ),
    )


def _bounds(
    boxes: Tensor, members: tuple[int, ...]
) -> tuple[float, float, float, float]:
    selected = boxes[list(members)]
    return (
        float(selected[:, 0].min()),
        float(selected[:, 1].min()),
        float((selected[:, 0] + selected[:, 2]).max()),
        float((selected[:, 1] + selected[:, 3]).max()),
    )


def _connected(first: Tensor, second: Tensor, proximity: float, overlap: float) -> bool:
    if overlap > OVERLAP_EPS:
        return True
    ax, ay, aw, ah = (float(value) for value in first)
    bx, by, bw, bh = (float(value) for value in second)
    ar, at, br, bt = ax + aw, ay + ah, bx + bw, by + bh
    x_overlap, y_overlap = min(ar, br) - max(ax, bx), min(at, bt) - max(ay, by)
    if x_overlap > OVERLAP_EPS and min(abs(at - by), abs(bt - ay)) <= proximity:
        return True
    if y_overlap > OVERLAP_EPS and min(abs(ar - bx), abs(br - ax)) <= proximity:
        return True
    if proximity <= 0.0:
        return False
    return max(max(ax - br, bx - ar, 0.0), max(ay - bt, by - at, 0.0)) <= proximity


def _source(case: Any | None, placements: Any | None) -> tuple[Any | None, Tensor]:
    if placements is None:
        if case is None:
            raise ValueError("placements are required")
        return None, _boxes(case)
    boxes = _boxes(placements)
    expected = _field(case, ("n", "block_count"))
    if expected is not None and int(expected) != len(boxes):
        raise ValueError(f"placements must have shape [{int(expected)}, 4]")
    return case, boxes


def _boxes(value: Any) -> Tensor:
    boxes = as_xywh(value).clone()
    if not len(boxes):
        raise ValueError("placements must contain at least one rectangle")
    return boxes


def _preplaced(case: Any | None, n: int, override: Any | None) -> Tensor:
    value = (
        override
        if override is not None
        else _field(case, ("preplaced_mask", "is_preplaced"))
    )
    if value is None:
        constraints = _field(case, ("constraints", "target_constraints"))
        rows = torch.as_tensor(constraints) if constraints is not None else None
        value = (
            rows[:n, 1] != 0
            if rows is not None and rows.ndim == 2 and rows.shape[1] > 1
            else None
        )
    if value is None:
        return torch.zeros(n, dtype=torch.bool)
    mask = torch.as_tensor(value, dtype=torch.bool).reshape(-1)
    if len(mask) != n:
        raise ValueError("preplaced_mask must have shape [N]")
    return mask


def _field(source: Any | None, names: tuple[str, ...]) -> Any:
    if source is None:
        return None
    if isinstance(source, dict):
        return next((source[name] for name in names if name in source), None)
    return next(
        (getattr(source, name) for name in names if hasattr(source, name)), None
    )


def _index(value: Any, n: int) -> int:
    number = float(value)
    if not math.isfinite(number) or not number.is_integer() or not 0 <= int(number) < n:
        raise ValueError("component member index is out of range")
    return int(number)


def _nonnegative(value: float, name: str) -> None:
    if not math.isfinite(float(value)) or float(value) < 0.0:
        raise ValueError(f"{name} must be finite and non-negative")
