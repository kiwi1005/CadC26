"""Active-soft boundary repair post-processor for contest runtime.

Applies bounded boundary-snap repair to positions produced by the contest
optimizer.  Only replaces positions when a strict meaningful winner exists
(all four metric deltas non-regressing beyond MEANINGFUL_COST_EPS).
"""

from __future__ import annotations

from typing import Any

from puzzleplace.data.schema import ConstraintColumns, FloorSetCase
from puzzleplace.eval.official import evaluate_positions
from puzzleplace.geometry.legality import summarize_hard_legality
from puzzleplace.ml.step7q_fresh_metric_replay import (
    actual_delta,
    strict_meaningful_winner,
)

Box = tuple[float, float, float, float]
EPS = 1e-9
SNAP_FRACTIONS = (1.0, 0.75, 0.5)


def _bbox_of(positions: list[Box]) -> tuple[float, float, float, float]:
    return (
        min(x for x, _, _, _ in positions),
        min(y for _, y, _, _ in positions),
        max(x + w for x, _, w, _ in positions),
        max(y + h for _, y, _, h in positions),
    )


def _boundary_edges(code: int) -> list[str]:
    out: list[str] = []
    if code & 1:
        out.append("left")
    if code & 2:
        out.append("right")
    if code & 4:
        out.append("top")
    if code & 8:
        out.append("bottom")
    return out


def _boundary_margin(box: Box, bbox: tuple[float, float, float, float], edge: str) -> float:
    x, y, w, h = box
    bx0, by0, bx1, by1 = bbox
    if edge == "left":
        return x - bx0
    elif edge == "right":
        return bx1 - (x + w)
    elif edge == "top":
        return by1 - (y + h)
    elif edge == "bottom":
        return y - by0
    return 0.0


def _snap_delta(edge: str, margin: float, fraction: float) -> tuple[float, float]:
    m = margin * fraction
    if edge == "left":
        return (-m, 0.0)
    elif edge == "right":
        return (m, 0.0)
    elif edge == "top":
        return (0.0, m)
    elif edge == "bottom":
        return (0.0, -m)
    return (0.0, 0.0)


def active_soft_postprocess(
    case: FloorSetCase,
    positions: list[Box],
    *,
    snap_fractions: tuple[float, ...] = SNAP_FRACTIONS,
) -> tuple[list[Box], dict[str, Any]]:
    """Try boundary-snap repairs; return improved positions if a strict winner exists."""

    if case.target_positions is None:
        return positions, {
            "active_soft_applied": False,
            "active_soft_candidates_evaluated": 0,
            "active_soft_strict_winners": 0,
            "active_soft_skipped_reason": "no_target_positions",
        }

    before = evaluate_positions(case, positions, runtime=1.0)
    bbox = _bbox_of(positions)

    snaps: list[tuple[int, float, float, str, float]] = []
    for idx in range(case.block_count):
        code = int(case.constraints[idx, ConstraintColumns.BOUNDARY].item())
        if code <= 0:
            continue
        for edge in _boundary_edges(code):
            margin = _boundary_margin(positions[idx], bbox, edge)
            if margin <= EPS:
                continue
            for frac in snap_fractions:
                dx, dy = _snap_delta(edge, margin, frac)
                if abs(dx) <= EPS and abs(dy) <= EPS:
                    continue
                snaps.append((idx, dx, dy, edge, frac))

    best_positions = None
    best_cost_delta = 0.0
    best_id = ""
    report: dict[str, Any] = {
        "active_soft_applied": False,
        "active_soft_candidates_evaluated": 0,
        "active_soft_strict_winners": 0,
    }

    for block_id, dx, dy, edge, frac in snaps:
        trial = list(positions)
        x, y, w, h = trial[block_id]
        trial[block_id] = (x + dx, y + dy, w, h)

        legality = summarize_hard_legality(case, trial)
        if not legality.is_feasible:
            report["active_soft_candidates_evaluated"] += 1
            continue

        after = evaluate_positions(case, trial, runtime=1.0)
        if not after["quality"].get("feasible"):
            report["active_soft_candidates_evaluated"] += 1
            continue

        delta = actual_delta(before, after)
        report["active_soft_candidates_evaluated"] += 1

        if strict_meaningful_winner(delta, True):
            report["active_soft_strict_winners"] += 1
            cost_delta = float(delta.get("official_like_cost_delta", 0.0))
            if best_positions is None or cost_delta < best_cost_delta:
                best_positions = trial
                best_cost_delta = cost_delta
                best_id = f"b{block_id}_{edge}_snap" + (
                    f"_{int(frac * 100)}pct" if frac < 1.0 - 1e-9 else ""
                )

    if best_positions is not None:
        report["active_soft_applied"] = True
        report["active_soft_selected"] = best_id
        report["active_soft_cost_delta"] = best_cost_delta
        return best_positions, report

    return positions, report
