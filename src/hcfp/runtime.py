"""Submission runtime adapter for HCFP.

This module is intentionally small.  It owns the official solve contract,
failover behavior, and optional injection points for future HCFP case/fallback
or analytic solver implementations.
"""

from __future__ import annotations

import importlib
import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any


Placement = tuple[float, float, float, float]
Solver = Callable[["SolveCase"], Sequence[Sequence[float]]]


@dataclass(frozen=True)
class SolveCase:
    block_count: int
    area_targets: Any
    b2b_connectivity: Any
    p2b_connectivity: Any
    pins_pos: Any
    constraints: Any
    target_positions: Any = None


class HCFPRuntime:
    """Official-contract runtime with immediate safe fallback."""

    def __init__(
        self,
        solver: Solver | None = None,
        fallback: Callable[[SolveCase], Sequence[Sequence[float]]] | None = None,
    ) -> None:
        self._solver = solver if solver is not None else _load_default_solver()
        self._fallback = fallback if fallback is not None else _load_fallback()

    def solve(
        self,
        block_count: int,
        area_targets: Any,
        b2b_connectivity: Any,
        p2b_connectivity: Any,
        pins_pos: Any,
        constraints: Any,
        target_positions: Any = None,
    ) -> list[Placement]:
        case = SolveCase(
            block_count=int(block_count),
            area_targets=area_targets,
            b2b_connectivity=b2b_connectivity,
            p2b_connectivity=p2b_connectivity,
            pins_pos=pins_pos,
            constraints=constraints,
            target_positions=target_positions,
        )
        _validate_case_contract(case)
        fallback = self._safe_fallback(case)
        if self._solver is None:
            return fallback
        try:
            candidate = self._solver(case)
            normalized = _normalize_output(candidate, case.block_count)
            return normalized if _is_hard_feasible(case, normalized) else fallback
        except Exception:
            return fallback

    def _safe_fallback(self, case: SolveCase) -> list[Placement]:
        try:
            candidate = _normalize_output(self._fallback(case), case.block_count)
            if _is_hard_feasible(case, candidate):
                return candidate
        except Exception:
            pass
        builtin = _builtin_shelf_fallback(case)
        if _is_hard_feasible(case, builtin):
            return builtin
        raise RuntimeError("no hard-feasible fallback exists for the supplied case")


def solve(
    block_count: int,
    area_targets: Any,
    b2b_connectivity: Any,
    p2b_connectivity: Any,
    pins_pos: Any,
    constraints: Any,
    target_positions: Any = None,
) -> list[Placement]:
    """Module-level convenience entrypoint matching the official contract."""

    return HCFPRuntime().solve(
        block_count,
        area_targets,
        b2b_connectivity,
        p2b_connectivity,
        pins_pos,
        constraints,
        target_positions,
    )


def _load_default_solver() -> Solver | None:
    for module_name, attr_name in (
        ("hcfp.analytic", "solve"),
        ("hcfp.solver", "solve"),
    ):
        try:
            module = importlib.import_module(module_name)
        except Exception:
            continue
        solver = getattr(module, attr_name, None)
        if callable(solver):
            return solver
    return None


def _load_fallback() -> Callable[[SolveCase], Sequence[Sequence[float]]]:
    try:
        module = importlib.import_module("hcfp.fallback")
    except Exception:
        return _builtin_shelf_fallback
    for attr_name in ("safe_fallback", "shelf_fallback", "solve"):
        fallback = getattr(module, attr_name, None)
        if callable(fallback):
            return fallback
    return _builtin_shelf_fallback


def _builtin_shelf_fallback(case: SolveCase) -> list[Placement]:
    n = int(case.block_count)
    areas = [_positive_float(value, 1.0) for value in _as_list(case.area_targets, n)]
    constraints = [_constraint_row(case.constraints, i) for i in range(n)]
    targets = _target_rows(case.target_positions, n)
    preplaced = [_is_truthy(_field(row, "preplaced", 1)) for row in constraints]
    fixed = [_is_truthy(_field(row, "fixed", 0)) for row in constraints]

    output: list[Placement | None] = [None] * n
    max_top = 0.0
    for i in range(n):
        if preplaced[i] and _valid_target(targets[i]):
            output[i] = _placement_from_target(targets[i])
            max_top = max(max_top, output[i][1] + output[i][3])

    gap = max(1.0e-3, math.sqrt(max(areas, default=1.0)) * 1.0e-4)
    x_cursor = 0.0
    y_cursor = max_top + gap
    for i in range(n):
        if output[i] is not None:
            continue
        w, h = _shape_for_block(i, areas[i], fixed[i], targets[i])
        output[i] = (x_cursor, y_cursor, w, h)
        x_cursor += w + gap

    return [_coerce_placement(rect) for rect in output if rect is not None]


def _normalize_output(raw: Sequence[Sequence[float]], n: int) -> list[Placement]:
    if len(raw) != n:
        raise ValueError(f"solver returned {len(raw)} placements for {n} blocks")
    return [_coerce_placement(rect) for rect in raw]


def _coerce_placement(rect: Sequence[float]) -> Placement:
    if len(rect) != 4:
        raise ValueError("each placement must be length 4")
    x, y, w, h = (float(value) for value in rect)
    if not all(math.isfinite(value) for value in (x, y, w, h)):
        raise ValueError("placement contains non-finite value")
    if w <= 0.0 or h <= 0.0:
        raise ValueError("placement width and height must be positive")
    return (x, y, w, h)


def _shape_for_block(i: int, area: float, fixed: bool, target: Sequence[Any] | None) -> tuple[float, float]:
    if fixed and _valid_target(target):
        _, _, w, h = _placement_from_target(target)
        return w, h
    side = math.sqrt(max(area, 1.0e-12))
    return side, side


def _placement_from_target(target: Sequence[Any] | None) -> Placement:
    if target is None:
        raise ValueError("missing target placement")
    return _coerce_placement(target)


def _valid_target(target: Sequence[Any] | None) -> bool:
    if target is None or len(target) < 4:
        return False
    try:
        x, y, w, h = (float(value) for value in target[:4])
    except (TypeError, ValueError):
        return False
    return all(math.isfinite(value) for value in (x, y, w, h)) and w > 0.0 and h > 0.0


def _target_rows(target_positions: Any, n: int) -> list[Sequence[Any] | None]:
    if target_positions is None:
        return [None] * n
    rows = _as_list(target_positions, n, default=None)
    return [row if isinstance(row, Sequence) and not isinstance(row, (str, bytes)) else None for row in rows]


def _constraint_row(constraints: Any, index: int) -> Any:
    if constraints is None:
        return ()
    if isinstance(constraints, dict):
        return {key: _index_or_default(value, index, None) for key, value in constraints.items()}
    try:
        return constraints[index]
    except (TypeError, IndexError, KeyError):
        return ()


def _field(row: Any, name: str, index: int) -> Any:
    if isinstance(row, dict):
        return row.get(name, False)
    if not isinstance(row, (str, bytes)) and hasattr(row, "__len__") and hasattr(row, "__getitem__"):
        return row[index] if len(row) > index else False
    return getattr(row, name, False)


def _is_truthy(value: Any) -> bool:
    try:
        return bool(float(value))
    except (TypeError, ValueError):
        return bool(value)


def _positive_float(value: Any, default: float) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(result) or result <= 0.0:
        return default
    return result


def _as_list(values: Any, n: int, default: Any = 0.0) -> list[Any]:
    if values is None:
        return [default] * n
    if hasattr(values, "detach"):
        values = values.detach().cpu().tolist()
    elif hasattr(values, "tolist"):
        values = values.tolist()
    try:
        items = list(values)
    except TypeError:
        items = [values]
    if len(items) < n:
        items.extend([default] * (n - len(items)))
    return items[:n]


def _index_or_default(values: Any, index: int, default: Any) -> Any:
    if hasattr(values, "detach"):
        values = values.detach().cpu().tolist()
    elif hasattr(values, "tolist"):
        values = values.tolist()
    try:
        return values[index]
    except (TypeError, IndexError, KeyError):
        return default


def _is_hard_feasible(case: SolveCase, placements: Sequence[Placement]) -> bool:
    from hcfp.verify import verify_feasible

    return verify_feasible(case, placements)


def _validate_case_contract(case: SolveCase) -> None:
    from hcfp.case import from_official

    from_official(
        case.block_count,
        case.area_targets,
        case.b2b_connectivity,
        case.p2b_connectivity,
        case.pins_pos,
        case.constraints,
        case.target_positions,
    )
