#!/usr/bin/env python3
"""Step7S Phase S1: Critical-Cone Active-Set Descent Certificate (LP form).

Formulates the per-case CCQP from the Oracle plan in LP form:

    max  rho
    s.t. g_C . delta + rho <= 0
         g_H . delta <= 0
         g_A . delta <= 0
         A_sep . delta <= s_clearance     (linearized non-overlap, including
                                           active and non-active pairs in closure)
         E_fixed . delta = 0              (fixed/preplaced blocks frozen)
         -r <= delta_i <= r               (trust region)
         rho >= 0

After solving, performs an exact line search alpha in (0,1] with
x(alpha) = x0 + alpha * delta and accepts only if exact replay gives
ΔC < -1e-7 AND ΔH <= EPS AND ΔA <= EPS AND ΔS <= EPS AND hard_feasible.

Outputs both the primal solution and the dual multipliers, so the artifact is
a usable KKT certificate when rho == 0.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import linprog

from puzzleplace.experiments.step7l_learning_guided_replay import load_validation_cases
from puzzleplace.geometry.legality import positions_from_case_targets, summarize_hard_legality
from puzzleplace.ml.step7q_fresh_metric_replay import (
    actual_delta,
    all_vector_nonregressing,
    official_like_evaluator,
    strict_meaningful_winner,
)

MEANINGFUL_COST_EPS = 1e-7
DEFAULT_FD_STEP = 1e-4
DEFAULT_TRUST_REGION = 0.5  # absolute units of x/y in case scale
LINESEARCH_ALPHAS = (1.0, 0.75, 0.5, 0.25, 0.125, 0.0625)
Box = tuple[float, float, float, float]


# ---------- Finite-difference gradients ----------


def _eval_quality(case: Any, positions: list[Box]) -> dict[str, float]:
    q = official_like_evaluator(case, positions)["quality"]
    return {
        "cost": float(q.get("cost", 0.0)),
        "hpwl": float(q.get("HPWLgap", 0.0)),
        "bbox": float(q.get("Areagap_bbox", 0.0)),
        "soft": float(q.get("Violationsrelative", 0.0)),
        "feasible": bool(q.get("feasible", False)),
    }


def _shifted_positions(positions: list[Box], block_id: int, dx: float, dy: float) -> list[Box]:
    out = list(positions)
    x, y, w, h = out[block_id]
    out[block_id] = (x + dx, y + dy, w, h)
    return out


def _fd_gradient_one_var(
    args: tuple[Any, list[Box], int, str, float],
) -> dict[str, float]:
    """Picklable worker. Returns d(cost,hpwl,bbox,soft)/d(var) at x0 via central FD."""

    case, positions, block_id, axis, h_step = args
    if axis == "x":
        plus = _shifted_positions(positions, block_id, h_step, 0.0)
        minus = _shifted_positions(positions, block_id, -h_step, 0.0)
    else:
        plus = _shifted_positions(positions, block_id, 0.0, h_step)
        minus = _shifted_positions(positions, block_id, 0.0, -h_step)
    qp = _eval_quality(case, plus)
    qm = _eval_quality(case, minus)
    return {
        "block_id": block_id,
        "axis": axis,
        "g_cost": (qp["cost"] - qm["cost"]) / (2.0 * h_step),
        "g_hpwl": (qp["hpwl"] - qm["hpwl"]) / (2.0 * h_step),
        "g_bbox": (qp["bbox"] - qm["bbox"]) / (2.0 * h_step),
        "g_soft": (qp["soft"] - qm["soft"]) / (2.0 * h_step),
    }


def compute_finite_diff_gradients(
    case: Any,
    positions: list[Box],
    closure: list[int],
    h_step: float,
    n_workers: int,
    *,
    smooth_soft_assumption: bool = True,
) -> dict[str, np.ndarray]:
    """Compute gradients of cost, hpwl, bbox, soft at `positions`.

    `g_cost` is built analytically from the verified hinge formula
    `c = (1 + 0.5·max(0,h) + 0.5·max(0,a))·exp(2·v)` so that discrete-soft FD
    noise does NOT pollute the LP cost gradient. FD is only used for the
    smooth components (h, a) and optionally for v.

    When `smooth_soft_assumption=True` (default), `g_soft` is forced to zero
    on the assumption that the move stays in the locally smooth interior of
    the soft-violation count. Set to False to use raw FD on soft (noisy).
    """

    args_list = [
        (case, positions, block_id, axis, h_step)
        for block_id in closure
        for axis in ("x", "y")
    ]
    if n_workers > 1:
        with ProcessPoolExecutor(max_workers=n_workers) as pool:
            rows = list(pool.map(_fd_gradient_one_var, args_list))
    else:
        rows = [_fd_gradient_one_var(args) for args in args_list]

    n = len(closure) * 2
    g_hpwl = np.zeros(n)
    g_bbox = np.zeros(n)
    g_soft_fd = np.zeros(n)
    for idx, row in enumerate(rows):
        g_hpwl[idx] = row["g_hpwl"]
        g_bbox[idx] = row["g_bbox"]
        g_soft_fd[idx] = row["g_soft"]
    g_soft = np.zeros(n) if smooth_soft_assumption else g_soft_fd

    # Build analytic g_cost from baseline hinge state
    baseline_q = _eval_quality(case, positions)
    h0 = float(baseline_q["hpwl"])
    a0 = float(baseline_q["bbox"])
    v0 = float(baseline_q["soft"])
    exp2v = math.exp(2.0 * v0)
    hinge_factor = 1.0 + 0.5 * max(0.0, h0) + 0.5 * max(0.0, a0)
    indicator_h = 1.0 if h0 > 0.0 else 0.0
    indicator_a = 1.0 if a0 > 0.0 else 0.0

    g_cost = (
        0.5 * indicator_h * exp2v * g_hpwl
        + 0.5 * indicator_a * exp2v * g_bbox
        + hinge_factor * 2.0 * exp2v * g_soft
    )

    return {
        "cost": g_cost,
        "hpwl": g_hpwl,
        "bbox": g_bbox,
        "soft": g_soft,
        "soft_fd_norm": float(np.linalg.norm(g_soft_fd)),
        "baseline_h0": h0,
        "baseline_a0": a0,
        "baseline_v0": v0,
        "indicator_h_active": indicator_h,
        "indicator_a_active": indicator_a,
    }


# ---------- Active-separation constraints ----------


def build_separation_rows(
    closure_index: dict[int, int],
    active_pairs: list[dict[str, Any]],
    eps_contact: float,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    """Translate active pairs into A_sep rows.

    For a pair (i, j) with direction "x_left" (i.e. i is left of j), the
    no-overlap constraint at first order is:
        (x_i + dx_i) + w_i <= (x_j + dx_j)  ->  dx_i - dx_j <= x_j - x_i - w_i
    Replace right-hand side with the current binding gap g >= 0.
    For active contacts (g near 0), dx_i - dx_j <= 0.
    For non-active pairs in closure but with positive clearance, we still
    record the constraint with the actual gap to avoid creating overlap.
    """

    rows: list[list[float]] = []
    rhs: list[float] = []
    info: list[dict[str, Any]] = []
    n = len(closure_index) * 2

    for pair in active_pairs:
        i = int(pair["i"])
        j = int(pair["j"])
        if i not in closure_index or j not in closure_index:
            continue
        gi = closure_index[i] * 2
        gj = closure_index[j] * 2
        direction = str(pair["direction"])
        gap = float(pair["binding_gap"])

        row = np.zeros(n)
        if direction == "x_left":
            # dx_i - dx_j <= gap
            row[gi] = 1.0
            row[gj] = -1.0
        elif direction == "x_right":
            # dx_j - dx_i <= gap
            row[gj] = 1.0
            row[gi] = -1.0
        elif direction == "y_below":
            row[gi + 1] = 1.0
            row[gj + 1] = -1.0
        elif direction == "y_above":
            row[gj + 1] = 1.0
            row[gi + 1] = -1.0
        else:
            continue

        # Use binding gap as RHS but never tighter than 0 (active contact).
        rhs_value = max(0.0, gap)
        rows.append(row.tolist())
        rhs.append(rhs_value)
        info.append(
            {
                "i": i,
                "j": j,
                "direction": direction,
                "binding_gap": gap,
                "rhs": rhs_value,
                "is_active": abs(gap) <= eps_contact,
            }
        )

    if not rows:
        return np.zeros((0, n)), np.zeros((0,)), info
    return np.array(rows), np.array(rhs), info


# ---------- LP construction and solve ----------


def solve_ccqp_lp(
    g_cost: np.ndarray,
    g_hpwl: np.ndarray,
    g_bbox: np.ndarray,
    g_soft: np.ndarray,
    a_sep: np.ndarray,
    b_sep: np.ndarray,
    fixed_indices: list[int],
    trust_region: float,
    enforce_soft_nonregression: bool,
) -> dict[str, Any]:
    """Solve the LP-form CCQP and return primal/dual report."""

    n_vars = len(g_cost)  # 2 * |closure|
    n_total = n_vars + 1  # plus rho

    # Objective: maximize rho -> minimize -rho
    c = np.zeros(n_total)
    c[-1] = -1.0

    # Inequality rows
    a_ub_rows: list[np.ndarray] = []
    b_ub_rows: list[float] = []

    # g_cost . delta + rho <= 0
    row_cost = np.concatenate([g_cost, np.array([1.0])])
    a_ub_rows.append(row_cost)
    b_ub_rows.append(0.0)

    # g_hpwl . delta <= 0
    row_h = np.concatenate([g_hpwl, np.array([0.0])])
    a_ub_rows.append(row_h)
    b_ub_rows.append(0.0)

    # g_bbox . delta <= 0
    row_a = np.concatenate([g_bbox, np.array([0.0])])
    a_ub_rows.append(row_a)
    b_ub_rows.append(0.0)

    if enforce_soft_nonregression:
        row_s = np.concatenate([g_soft, np.array([0.0])])
        a_ub_rows.append(row_s)
        b_ub_rows.append(0.0)

    # Active-separation rows
    if a_sep.shape[0] > 0:
        sep_block = np.hstack([a_sep, np.zeros((a_sep.shape[0], 1))])
        for r, b in zip(sep_block, b_sep, strict=False):
            a_ub_rows.append(r)
            b_ub_rows.append(float(b))

    a_ub = np.array(a_ub_rows)
    b_ub = np.array(b_ub_rows)

    # Equality: fixed-block indices
    a_eq_rows: list[np.ndarray] = []
    for idx in fixed_indices:
        for axis_offset in (0, 1):
            r = np.zeros(n_total)
            r[idx + axis_offset] = 1.0
            a_eq_rows.append(r)
    if a_eq_rows:
        a_eq = np.array(a_eq_rows)
        b_eq = np.zeros(len(a_eq_rows))
    else:
        a_eq = None
        b_eq = None

    # Bounds: -r <= delta_i <= r, rho >= 0
    bounds = [(-trust_region, trust_region)] * n_vars + [(0.0, None)]

    res = linprog(
        c=c,
        A_ub=a_ub,
        b_ub=b_ub,
        A_eq=a_eq,
        b_eq=b_eq,
        bounds=bounds,
        method="highs",
    )

    out: dict[str, Any] = {
        "status": int(res.status),
        "message": str(res.message),
        "success": bool(res.success),
        "n_vars": n_vars,
        "n_total_decision": n_total,
        "n_ub_rows": int(a_ub.shape[0]),
        "n_eq_rows": 0 if a_eq is None else int(a_eq.shape[0]),
        "trust_region": trust_region,
        "enforce_soft_nonregression": enforce_soft_nonregression,
    }
    if res.success:
        x = np.asarray(res.x)
        delta = x[:n_vars]
        rho = float(x[-1])
        out["rho_pred"] = rho
        out["delta"] = delta.tolist()
        out["delta_inf_norm"] = float(np.max(np.abs(delta)))
        out["delta_2_norm"] = float(np.linalg.norm(delta))
        # Linearized predictions
        out["linear_pred_cost_delta"] = float(g_cost @ delta)
        out["linear_pred_hpwl_delta"] = float(g_hpwl @ delta)
        out["linear_pred_bbox_delta"] = float(g_bbox @ delta)
        out["linear_pred_soft_delta"] = float(g_soft @ delta)
        # Dual multipliers from HiGHS marginals
        if hasattr(res, "ineqlin") and res.ineqlin is not None:
            marginals = np.asarray(res.ineqlin.marginals)
            out["ub_dual_multipliers"] = (-marginals).tolist()
        else:
            out["ub_dual_multipliers"] = None
        if a_eq is not None and hasattr(res, "eqlin") and res.eqlin is not None:
            eq_marginals = np.asarray(res.eqlin.marginals)
            out["eq_dual_multipliers"] = (-eq_marginals).tolist()
        else:
            out["eq_dual_multipliers"] = None
    else:
        out["rho_pred"] = 0.0
        out["delta"] = None
    return out


# ---------- Line search and exact replay ----------


def line_search_and_replay(
    case: Any,
    baseline: list[Box],
    closure: list[int],
    delta: np.ndarray,
    alphas: tuple[float, ...],
) -> dict[str, Any]:
    before = _eval_quality(case, baseline)
    if not before["feasible"]:
        return {"status": "baseline_not_feasible", "before": before}
    legality0 = summarize_hard_legality(case, baseline)
    if not bool(legality0.is_feasible):
        return {"status": "baseline_not_hard_feasible", "before": before}

    best: dict[str, Any] | None = None
    history: list[dict[str, Any]] = []

    for alpha in alphas:
        positions = list(baseline)
        for idx, block_id in enumerate(closure):
            x, y, w, h = positions[block_id]
            dx = float(delta[idx * 2]) * alpha
            dy = float(delta[idx * 2 + 1]) * alpha
            positions[block_id] = (x + dx, y + dy, w, h)
        legality = summarize_hard_legality(case, positions)
        # Quick overlap check among closure boxes (cheap pre-screen)
        if not bool(legality.is_feasible):
            history.append({"alpha": alpha, "status": "hard_infeasible"})
            continue
        # Full evaluator
        q = _eval_quality(case, positions)
        before_full = official_like_evaluator(case, baseline)
        after_full = official_like_evaluator(case, positions)
        delta_metrics = actual_delta(before_full, after_full)
        hard_feasible = bool(legality.is_feasible) and bool(after_full["quality"].get("feasible"))
        record = {
            "alpha": alpha,
            "status": "evaluated",
            "hard_feasible": hard_feasible,
            **{k: float(v) for k, v in delta_metrics.items()},
            "all_vector_nonregressing": all_vector_nonregressing(delta_metrics, hard_feasible),
            "strict_meaningful_winner": strict_meaningful_winner(delta_metrics, hard_feasible),
            "after_cost": q["cost"],
            "after_hpwl": q["hpwl"],
            "after_bbox": q["bbox"],
            "after_soft": q["soft"],
        }
        history.append(record)
        if record["strict_meaningful_winner"] and (
            best is None
            or float(record["official_like_cost_delta"])
            < float(best["official_like_cost_delta"])
        ):
            best = record

    return {"status": "ok", "before": before, "history": history, "best": best}


# ---------- Main ----------


def run_case(
    base_dir: Path,
    case_id: int,
    seed_block: int,
    apply_avnr_target: tuple[float, float, float, float] | None,
    contact_graph_path: Path,
    trust_region: float,
    fd_step: float,
    n_workers: int,
    floorset_root: Path | None,
    auto_download: bool,
    enforce_soft_nonregression: bool,
) -> dict[str, Any]:
    cases = load_validation_cases(
        base_dir, [case_id], floorset_root=floorset_root, auto_download=auto_download
    )
    case = cases[case_id]
    baseline = positions_from_case_targets(case)
    boxes: list[Box] = [tuple(map(float, b)) for b in baseline]  # type: ignore[misc]
    if apply_avnr_target is not None:
        boxes[seed_block] = apply_avnr_target

    contact = json.loads(contact_graph_path.read_text(encoding="utf-8"))
    closure = sorted(contact["closure_with_mib_block_ids"])
    closure_index = {block_id: idx for idx, block_id in enumerate(closure)}
    fixed_indices = [
        closure_index[b] * 2
        for b in contact["fixed_or_preplaced_in_closure"]
        if b in closure_index
    ]
    active_pairs = contact["active_pairs_in_closure"]

    # FD gradients at baseline-with-AVNR layout
    grads = compute_finite_diff_gradients(
        case, boxes, closure, h_step=fd_step, n_workers=n_workers,
        smooth_soft_assumption=True,
    )

    a_sep, b_sep, sep_info = build_separation_rows(
        closure_index, active_pairs, eps_contact=float(contact.get("eps_contact", 1e-3))
    )

    lp_result = solve_ccqp_lp(
        g_cost=grads["cost"],
        g_hpwl=grads["hpwl"],
        g_bbox=grads["bbox"],
        g_soft=grads["soft"],
        a_sep=a_sep,
        b_sep=b_sep,
        fixed_indices=fixed_indices,
        trust_region=trust_region,
        enforce_soft_nonregression=enforce_soft_nonregression,
    )

    line_search: dict[str, Any] = {"status": "skipped_no_delta"}
    delta_value = lp_result.get("delta")
    if delta_value is not None:
        delta_arr = np.asarray(delta_value)
        line_search = line_search_and_replay(
            case, boxes, closure, delta_arr, alphas=LINESEARCH_ALPHAS
        )

    sigma_norm = float(np.linalg.norm(grads["cost"]))
    rho_pred = float(lp_result.get("rho_pred", 0.0) or 0.0)
    est_units_to_strict = (
        MEANINGFUL_COST_EPS / rho_pred if rho_pred > 1e-300 else math.inf
    )

    if line_search.get("best") is not None:
        result_label = "strict_winner"
    elif rho_pred <= 1e-12:
        result_label = "kkt_stationary"
    elif rho_pred < MEANINGFUL_COST_EPS:
        result_label = "avnr_only_below_threshold"
    else:
        result_label = "linearized_promising_no_strict_after_linesearch"

    return {
        "schema": "step7s_critical_cone_certificate_v1",
        "case_id": case_id,
        "seed_block": seed_block,
        "applied_avnr_target": list(apply_avnr_target) if apply_avnr_target else None,
        "closure_size": len(closure),
        "closure_block_ids": closure,
        "fixed_indices_in_closure": fixed_indices,
        "active_pair_count": len(sep_info),
        "trust_region": trust_region,
        "fd_step": fd_step,
        "gradient_2_norm_cost": sigma_norm,
        "linear_pred_units_to_strict": est_units_to_strict,
        "lp_result": lp_result,
        "line_search": line_search,
        "result": result_label,
        "meaningful_cost_eps": MEANINGFUL_COST_EPS,
        "separation_pairs_used": sep_info,
        "gradient_diagnostics": {
            "g_cost_2_norm": sigma_norm,
            "g_hpwl_2_norm": float(np.linalg.norm(grads["hpwl"])),
            "g_bbox_2_norm": float(np.linalg.norm(grads["bbox"])),
            "g_soft_used_2_norm": float(np.linalg.norm(grads["soft"])),
            "g_soft_raw_fd_2_norm": grads["soft_fd_norm"],
            "baseline_h": grads["baseline_h0"],
            "baseline_a": grads["baseline_a0"],
            "baseline_v": grads["baseline_v0"],
            "hpwl_hinge_active": bool(grads["indicator_h_active"]),
            "bbox_hinge_active": bool(grads["indicator_a_active"]),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-dir", type=Path, default=Path("."))
    parser.add_argument("--case-id", type=int, required=True)
    parser.add_argument("--seed-block", type=int, required=True)
    parser.add_argument(
        "--apply-avnr-target",
        type=str,
        default=None,
        help="'x,y,w,h' to overwrite seed-block target before computing gradients.",
    )
    parser.add_argument("--contact-graph", type=Path, required=True)
    parser.add_argument("--trust-region", type=float, default=DEFAULT_TRUST_REGION)
    parser.add_argument("--fd-step", type=float, default=DEFAULT_FD_STEP)
    parser.add_argument("--n-workers", type=int, default=48)
    parser.add_argument("--floorset-root", type=Path, default=None)
    parser.add_argument("--auto-download", action="store_true")
    parser.add_argument(
        "--enforce-soft-nonregression",
        action="store_true",
        help="Add g_soft·delta <= 0 row (recommended when soft is locally smooth).",
    )
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    apply_avnr = None
    if args.apply_avnr_target:
        parts = [float(v) for v in args.apply_avnr_target.split(",")]
        if len(parts) != 4:
            raise SystemExit(
                f"--apply-avnr-target must be 4 floats, got {parts!r}"
            )
        apply_avnr = (parts[0], parts[1], parts[2], parts[3])

    started = time.perf_counter()
    summary = run_case(
        args.base_dir,
        args.case_id,
        args.seed_block,
        apply_avnr,
        args.contact_graph,
        trust_region=args.trust_region,
        fd_step=args.fd_step,
        n_workers=args.n_workers,
        floorset_root=args.floorset_root,
        auto_download=args.auto_download,
        enforce_soft_nonregression=args.enforce_soft_nonregression,
    )
    summary["runtime_seconds"] = time.perf_counter() - started

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    headline = {
        "case_id": summary["case_id"],
        "result": summary["result"],
        "rho_pred": summary["lp_result"].get("rho_pred"),
        "linear_pred_cost_delta": summary["lp_result"].get("linear_pred_cost_delta"),
        "best_strict": (
            summary["line_search"].get("best") or {}
        ).get("official_like_cost_delta"),
        "closure_size": summary["closure_size"],
        "active_pair_count": summary["active_pair_count"],
        "runtime_seconds": round(summary["runtime_seconds"], 2),
    }
    print(json.dumps(headline))


if __name__ == "__main__":
    main()
