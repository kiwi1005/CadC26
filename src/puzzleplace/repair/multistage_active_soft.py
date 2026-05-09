"""Multi-stage active-soft boundary repair post-processor.

Stages (integrated per-candidate):
  1. HPWL-sensitive, bbox-preserving partial snap
  2. Joint push: if snap causes overlap, push obstructing block out of the way
  3. HPWL compensation: if snap is feasible but HPWL regresses, try small
     compensating moves on connected non-boundary blocks

Stage 2 and 3 are triggered conditionally for each candidate that needs them,
rather than running as independent global passes.

See docs/METHOD.md for the mathematical formulation.
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
HPWL_SENSITIVITY_THRESHOLD = 5.0  # skip if |sigma*dx| exceeds this (heuristic)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


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


def _boundary_margin(
    box: Box, bbox: tuple[float, float, float, float], edge: str
) -> float:
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


def _block_center(box: Box) -> tuple[float, float]:
    x, y, w, h = box
    return (x + w / 2.0, y + h / 2.0)


# ---------------------------------------------------------------------------
# HPWL sensitivity
# ---------------------------------------------------------------------------


def compute_hpwl_sensitivity(
    case: FloorSetCase, positions: list[Box]
) -> tuple[list[float], list[float]]:
    """Compute per-block HPWL sensitivity (subgradient of pairwise Manhattan HPWL).

    sigma_x[k] = dHPWL/d(cx_k). Positive means moving right increases HPWL.
    """
    n = case.block_count
    sigma_x = [0.0] * n
    sigma_y = [0.0] * n
    centers = [_block_center(b) for b in positions]

    for src, dst, weight in case.b2b_edges.tolist():
        i, j = int(src), int(dst)
        if i >= n or j >= n:
            continue
        w = float(weight)
        cxi, cyi = centers[i]
        cxj, cyj = centers[j]
        dx = cxi - cxj
        dy = cyi - cyj
        if abs(dx) > EPS:
            sx = w if dx > 0 else -w
            sigma_x[i] += sx
            sigma_x[j] -= sx
        if abs(dy) > EPS:
            sy = w if dy > 0 else -w
            sigma_y[i] += sy
            sigma_y[j] -= sy

    for pin_id, block_id, weight in case.p2b_edges.tolist():
        bid = int(block_id)
        pid = int(pin_id)
        if bid >= n or pid >= case.pins_pos.shape[0]:
            continue
        w = float(weight)
        px = float(case.pins_pos[pid, 0].item())
        py = float(case.pins_pos[pid, 1].item())
        cxb, cyb = centers[bid]
        dx = cxb - px
        dy = cyb - py
        if abs(dx) > EPS:
            sigma_x[bid] += w if dx > 0 else -w
        if abs(dy) > EPS:
            sigma_y[bid] += w if dy > 0 else -w

    return sigma_x, sigma_y


# ---------------------------------------------------------------------------
# Bbox edge owners
# ---------------------------------------------------------------------------


def compute_bbox_edge_owners(
    positions: list[Box], bbox: tuple[float, float, float, float]
) -> dict[str, set[int]]:
    bx0, by0, bx1, by1 = bbox
    owners: dict[str, set[int]] = {
        "left": set(), "right": set(), "top": set(), "bottom": set(),
    }
    for idx, (x, y, w, h) in enumerate(positions):
        if abs(x - bx0) <= EPS:
            owners["left"].add(idx)
        if abs((x + w) - bx1) <= EPS:
            owners["right"].add(idx)
        if abs((y + h) - by1) <= EPS:
            owners["top"].add(idx)
        if abs(y - by0) <= EPS:
            owners["bottom"].add(idx)
    return owners


# ---------------------------------------------------------------------------
# Netlist neighborhood
# ---------------------------------------------------------------------------


def build_net_neighborhood(
    case: FloorSetCase,
) -> dict[int, list[tuple[int, float]]]:
    """Build block-id -> list of (connected_block, weight) from b2b edges."""
    neighbors: dict[int, list[tuple[int, float]]] = {
        i: [] for i in range(case.block_count)
    }
    for src, dst, weight in case.b2b_edges.tolist():
        i, j = int(src), int(dst)
        w = float(weight)
        if i < case.block_count and j < case.block_count:
            neighbors[i].append((j, w))
            neighbors[j].append((i, w))
    return neighbors


# ---------------------------------------------------------------------------
# Overlap resolution
# ---------------------------------------------------------------------------


def _rects_overlap(a: Box, b: Box) -> bool:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    return (
        ax < bx + bw and ax + aw > bx
        and ay < by + bh and ay + ah > by
    )


def _push_to_resolve_overlap(
    moving: Box, obstructing: Box
) -> tuple[float, float] | None:
    """Compute minimal (dx, dy) on obstructing to resolve overlap."""
    mx, my, mw, mh = moving
    ox, oy, ow, oh = obstructing

    overlap_x = (mx + mw) - ox
    overlap_y = (my + mh) - oy

    if overlap_x <= EPS or overlap_y <= EPS:
        return None

    # Choose smaller push: right or up
    if abs(overlap_x) <= abs(overlap_y):
        return (overlap_x, 0.0)
    else:
        return (0.0, overlap_y)


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------


def multistage_active_soft_postprocess(
    case: FloorSetCase,
    positions: list[Box],
    *,
    snap_fractions: tuple[float, ...] = SNAP_FRACTIONS,
    max_candidates: int | None = 200,
) -> tuple[list[Box], dict[str, Any]]:
    """Run integrated multi-stage active-soft boundary repair.

    Args:
        case: The floorplan case.
        positions: Current block positions (from optimizer).
        snap_fractions: Fractions of boundary margin to try snapping.
        max_candidates: Maximum candidates to evaluate across all stages.
            None means no limit. Default 200.
    """

    if case.target_positions is None:
        return positions, {
            "multistage_applied": False,
            "multistage_skipped_reason": "no_target_positions",
        }

    before_eval = evaluate_positions(case, positions, runtime=1.0)
    bbox = _bbox_of(positions)
    sigma_x, sigma_y = compute_hpwl_sensitivity(case, positions)
    edge_owners = compute_bbox_edge_owners(positions, bbox)
    neighbors = build_net_neighborhood(case)

    # Identify boundary violations
    boundary_violations: list[dict[str, Any]] = []
    for idx in range(case.block_count):
        code = int(case.constraints[idx, ConstraintColumns.BOUNDARY].item())
        if code <= 0:
            continue
        box = positions[idx]
        margins = {}
        for edge in _boundary_edges(code):
            margins[edge] = _boundary_margin(box, bbox, edge)
        violated = [e for e, m in margins.items() if m > EPS]
        if violated:
            boundary_violations.append({
                "block_id": idx,
                "code": code,
                "required_edges": _boundary_edges(code),
                "violated_edges": violated,
                "margins": margins,
            })

    boundary_set = {int(c["block_id"]) for c in boundary_violations}

    best_positions: list[Box] | None = None
    best_cost_delta = 0.0
    best_id = ""
    candidates_evaluated = 0
    strict_winners = 0
    stage1_count = 0
    stage2_count = 0
    stage3_count = 0
    stage3_strict = 0
    limit_reached = False

    def _check_limit() -> bool:
        nonlocal limit_reached
        if max_candidates is not None and candidates_evaluated >= max_candidates:
            limit_reached = True
            return True
        return False

    compensation_steps = (0.5, 1.0, 2.0, 4.0)

    for comp in boundary_violations:
        if limit_reached:
            break
        block_id = int(comp["block_id"])
        margins = comp["margins"]

        for edge in comp["violated_edges"]:
            if limit_reached:
                break
            margin = float(margins[edge])
            if margin <= EPS:
                continue

            # HPWL impact estimation
            dx_full, dy_full = _snap_delta(edge, margin, 1.0)
            hpwl_impact = (
                sigma_x[block_id] * dx_full + sigma_y[block_id] * dy_full
            )

            # Bbox expansion check
            bbox_expands = False
            if edge == "right" and block_id in edge_owners.get("right", set()):
                bbox_expands = True
            if edge == "top" and block_id in edge_owners.get("top", set()):
                bbox_expands = True

            # Determine fractions to try
            if bbox_expands:
                continue  # skip bbox-expanding snaps

            if hpwl_impact > HPWL_SENSITIVITY_THRESHOLD:
                fractions_to_try = [
                    f for f in snap_fractions if f < 0.8
                ] or snap_fractions
            else:
                fractions_to_try = snap_fractions

            for frac in fractions_to_try:
                if limit_reached:
                    break
                dx, dy = _snap_delta(edge, margin, frac)
                if abs(dx) <= EPS and abs(dy) <= EPS:
                    continue

                # --- Stage 1: direct snap ---
                trial = list(positions)
                x, y, w, h = trial[block_id]
                trial[block_id] = (x + dx, y + dy, w, h)

                legality = summarize_hard_legality(case, trial)
                label_base = (
                    f"b{block_id}_{edge}_snap"
                    if frac >= 1.0 - 1e-9
                    else f"b{block_id}_{edge}_snap_{int(frac * 100)}pct"
                )

                if legality.is_feasible:
                    after = evaluate_positions(case, trial, runtime=1.0)
                    if after["quality"].get("feasible"):
                        delta = actual_delta(before_eval, after)
                        candidates_evaluated += 1
                        stage1_count += 1
                        _check_limit()
                        if limit_reached:
                            break

                        if strict_meaningful_winner(delta, True):
                            strict_winners += 1
                            cost_delta = float(delta.get("official_like_cost_delta", 0.0))
                            if best_positions is None or cost_delta < best_cost_delta:
                                best_positions = trial
                                best_cost_delta = cost_delta
                                best_id = f"stage1_{label_base}"
                            continue  # got a winner, skip further processing

                        # --- Stage 3: HPWL compensation ---
                        # Snap is feasible but not a strict winner.
                        # Try small compensating moves on connected non-boundary blocks.
                        hpwl_before_comp = float(delta.get("hpwl_delta", 0.0))
                        if hpwl_before_comp > EPS:
                            # Snap caused HPWL regression; try compensation
                            conn_blocks = neighbors.get(block_id, [])
                            for conn_idx, conn_weight in conn_blocks:
                                if limit_reached:
                                    break
                                if conn_idx in boundary_set:
                                    continue  # don't move boundary blocks

                                cx_conn, cy_conn = _block_center(trial[conn_idx])
                                cx_snap, cy_snap = _block_center(trial[block_id])

                                # Gradient: if snapped block is now above/below/left/right
                                # of connected block, move connected block TOWARD snapped block
                                grad_x = 0.0
                                grad_y = 0.0
                                dx_conn = cx_snap - cx_conn
                                dy_conn = cy_snap - cy_conn
                                if abs(dx_conn) > EPS:
                                    grad_x = conn_weight if dx_conn > 0 else -conn_weight
                                if abs(dy_conn) > EPS:
                                    grad_y = conn_weight if dy_conn > 0 else -conn_weight

                                if abs(grad_x) <= EPS and abs(grad_y) <= EPS:
                                    continue

                                for step in compensation_steps:
                                    if limit_reached:
                                        break
                                    cdx = step * (1.0 if grad_x > 0 else (-1.0 if grad_x < 0 else 0))
                                    cdy = step * (1.0 if grad_y > 0 else (-1.0 if grad_y < 0 else 0))

                                    if abs(cdx) <= EPS and abs(cdy) <= EPS:
                                        continue

                                    comp_trial = list(trial)
                                    cox, coy, cow, coh = comp_trial[conn_idx]
                                    comp_trial[conn_idx] = (cox + cdx, coy + cdy, cow, coh)

                                    clegality = summarize_hard_legality(case, comp_trial)
                                    if not clegality.is_feasible:
                                        continue

                                    cafter = evaluate_positions(case, comp_trial, runtime=1.0)
                                    if not cafter["quality"].get("feasible"):
                                        continue

                                    cdelta = actual_delta(before_eval, cafter)
                                    candidates_evaluated += 1
                                    stage3_count += 1
                                    _check_limit()
                                    if limit_reached:
                                        break

                                    if strict_meaningful_winner(cdelta, True):
                                        stage3_strict += 1
                                        strict_winners += 1
                                        cost_delta = float(cdelta.get("official_like_cost_delta", 0.0))
                                        if best_positions is None or cost_delta < best_cost_delta:
                                            best_positions = comp_trial
                                            best_cost_delta = cost_delta
                                            best_id = f"stage3_{label_base}_comp{conn_idx}_step{step}"

                # --- Stage 2: joint push for infeasible snaps ---
                if not legality.is_feasible:
                    # Find blocks that overlap with the snapped block
                    snapped_box = trial[block_id]
                    for other_idx in range(case.block_count):
                        if limit_reached:
                            break
                        if other_idx == block_id:
                            continue
                        if _rects_overlap(snapped_box, trial[other_idx]):
                            push = _push_to_resolve_overlap(
                                snapped_box, trial[other_idx]
                            )
                            if push is None:
                                continue

                            pdx, pdy = push
                            joint_trial = list(positions)
                            jx, jy, jw, jh = joint_trial[block_id]
                            joint_trial[block_id] = (jx + dx, jy + dy, jw, jh)
                            ox, oy, ow, oh = joint_trial[other_idx]
                            joint_trial[other_idx] = (ox + pdx, oy + pdy, ow, oh)

                            jlegality = summarize_hard_legality(case, joint_trial)
                            if not jlegality.is_feasible:
                                continue

                            jafter = evaluate_positions(case, joint_trial, runtime=1.0)
                            if not jafter["quality"].get("feasible"):
                                continue

                            jdelta = actual_delta(before_eval, jafter)
                            candidates_evaluated += 1
                            stage2_count += 1
                            _check_limit()
                            if limit_reached:
                                break

                            if strict_meaningful_winner(jdelta, True):
                                strict_winners += 1
                                cost_delta = float(jdelta.get("official_like_cost_delta", 0.0))
                                if best_positions is None or cost_delta < best_cost_delta:
                                    best_positions = joint_trial
                                    best_cost_delta = cost_delta
                                    best_id = f"stage2_{label_base}_push{other_idx}"

    report: dict[str, Any] = {
        "multistage_applied": best_positions is not None,
        "multistage_candidates_evaluated": candidates_evaluated,
        "multistage_strict_winners": strict_winners,
        "stage1_direct_snaps": stage1_count,
        "stage2_joint_pushes": stage2_count,
        "stage3_hpwl_compensations": stage3_count,
        "stage3_strict_winners": stage3_strict,
    }
    if best_positions is not None:
        report["multistage_selected"] = best_id
        report["multistage_cost_delta"] = best_cost_delta
        return best_positions, report

    return positions, report
