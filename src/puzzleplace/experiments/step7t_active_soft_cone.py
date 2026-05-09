"""Step7T active-soft constrained descent sidecar.

Audits active soft violations on representative Step7S cases and tries a small,
bounded direct-repair deck with exact official-like evaluator replay. This module
is sidecar-only: it does not touch contest/runtime/finalizer paths.
"""

from __future__ import annotations

import json
import time
from collections import Counter, defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from puzzleplace.data.schema import ConstraintColumns, FloorSetCase
from puzzleplace.eval.official import evaluate_positions
from puzzleplace.experiments.step7l_learning_guided_replay import load_validation_cases
from puzzleplace.geometry.legality import positions_from_case_targets, summarize_hard_legality
from puzzleplace.ml.step7q_fresh_metric_replay import (
    MEANINGFUL_COST_EPS,
    actual_delta,
    all_vector_nonregressing,
    strict_meaningful_winner,
)

Box = tuple[float, float, float, float]
EPS = 1e-9
BOUNDARY_NAMES = {1: "left", 2: "right", 4: "top", 8: "bottom"}
REPRESENTATIVE_CASES = (19, 24, 25, 51, 76, 79, 91, 99)


@dataclass(frozen=True, slots=True)
class Candidate:
    candidate_id: str
    moves: tuple[tuple[int, float, float], ...]
    repair_kind: str
    repaired_component: dict[str, Any]
    companion_block: int | None = None


def load_step7s_case_specs(summary_path: Path) -> list[dict[str, Any]]:
    data = json.loads(summary_path.read_text(encoding="utf-8"))
    specs: list[dict[str, Any]] = []
    for row in data.get("per_case", []):
        specs.append(
            {
                "case_id": int(row["case_id"]),
                "seed_block": int(row["seed_block"]),
                "applied_avnr_target": row.get("applied_avnr_target"),
                "step7s_result": row.get("result_with_hinge_cap", row.get("result")),
            }
        )
    return specs


def bbox_of(positions: Iterable[Box]) -> tuple[float, float, float, float]:
    boxes = list(positions)
    return (
        min(x for x, _, _, _ in boxes),
        min(y for _, y, _, _ in boxes),
        max(x + w for x, _, w, _ in boxes),
        max(y + h for _, y, _, h in boxes),
    )


def apply_avnr(positions: list[Box], seed_block: int, target: list[float] | None) -> list[Box]:
    out = list(positions)
    if target is not None:
        out[seed_block] = tuple(float(v) for v in target)  # type: ignore[assignment]
    return out


def boundary_edges(code: int) -> list[str]:
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


def boundary_margins(box: Box, bbox: tuple[float, float, float, float]) -> dict[str, float]:
    x, y, w, h = box
    bx0, by0, bx1, by1 = bbox
    return {
        "left": x - bx0,
        "right": bx1 - (x + w),
        "top": by1 - (y + h),
        "bottom": y - by0,
    }


def active_soft_audit(case: FloorSetCase, positions: list[Box]) -> dict[str, Any]:
    bbox = bbox_of(positions)
    boundary_rows: list[dict[str, Any]] = []
    for idx, box in enumerate(positions):
        code = int(case.constraints[idx, ConstraintColumns.BOUNDARY].item())
        if code <= 0:
            continue
        margins = boundary_margins(box, bbox)
        req = boundary_edges(code)
        violated = [edge for edge in req if margins[edge] > EPS]
        boundary_rows.append(
            {
                "block_id": idx,
                "code": code,
                "required_edges": req,
                "violated_edges": violated,
                "margins": margins,
                "box": list(box),
            }
        )

    group_rows: list[dict[str, Any]] = []
    for column, kind in ((ConstraintColumns.CLUSTER, "group"), (ConstraintColumns.MIB, "mib")):
        groups: dict[int, list[int]] = defaultdict(list)
        for idx in range(case.block_count):
            gid = int(case.constraints[idx, column].item())
            if gid > 0:
                groups[gid].append(idx)
        for gid, members in sorted(groups.items()):
            if kind == "mib":
                shapes = sorted(
                    {(round(positions[i][2], 6), round(positions[i][3], 6)) for i in members}
                )
                violation_count = int(len(shapes) > 1)
                margin = float(max(len(shapes) - 1, 0))
            else:
                # A minimal sidecar margin: zero means one member/no obvious gap audit here;
                # exact grouping violation count is taken from official evaluator below.
                violation_count = 0
                margin = 0.0
            group_rows.append(
                {
                    "kind": kind,
                    "group_id": gid,
                    "members": members,
                    "margin": margin,
                    "local_violation_count": violation_count,
                }
            )
    official = evaluate_positions(case, positions, runtime=1.0)["official"]
    return {
        "bbox": list(bbox),
        "official_soft_counts": {
            "boundary_violations": int(official["boundary_violations"]),
            "grouping_violations": int(official["grouping_violations"]),
            "mib_violations": int(official["mib_violations"]),
            "total_soft_violations": int(official["total_soft_violations"]),
            "max_possible_violations": int(official["max_possible_violations"]),
        },
        "boundary_components": boundary_rows,
        "active_violated_boundary_components": [
            row for row in boundary_rows if row["violated_edges"]
        ],
        "group_mib_components": group_rows,
    }


def _snap_deltas(
    edge: str, margin: float, fractions: tuple[float, ...] = (1.0,)
) -> list[tuple[float, float, float]]:
    results: list[tuple[float, float, float]] = []
    for frac in fractions:
        dx = dy = 0.0
        m = margin * frac
        if edge == "left":
            dx = -m
        elif edge == "right":
            dx = m
        elif edge == "top":
            dy = m
        elif edge == "bottom":
            dy = -m
        if abs(dx) <= EPS and abs(dy) <= EPS:
            continue
        results.append((dx, dy, frac))
    return results


def _snap_label(block_id: int, edge: str, frac: float) -> str:
    if frac >= 1.0 - 1e-9:
        return f"b{block_id}_{edge}_snap"
    return f"b{block_id}_{edge}_snap_{int(frac * 100)}pct"


def generate_boundary_repair_candidates(
    case: FloorSetCase,
    positions: list[Box],
    seed_block: int,
    *,
    companion_radius: float = 4.0,
    companion_step: float = 0.25,
    snap_fractions: tuple[float, ...] = (1.0, 0.75, 0.5),
) -> list[Candidate]:
    audit = active_soft_audit(case, positions)
    snap_only: list[Candidate] = []
    seed_comp: list[Candidate] = []
    group_snap_only: list[Candidate] = []
    group_seed_comp: list[Candidate] = []
    grid = [
        round(-companion_radius + i * companion_step, 10)
        for i in range(int(2 * companion_radius / companion_step) + 1)
    ]

    boundary_block_ids = {
        int(c["block_id"]) for c in audit["active_violated_boundary_components"]
    }

    for component in audit["active_violated_boundary_components"]:
        block_id = int(component["block_id"])
        margins = component["margins"]
        for edge in component["violated_edges"]:
            for dx, dy, frac in _snap_deltas(edge, float(margins[edge]), snap_fractions):
                label = _snap_label(block_id, edge, frac)
                is_full = frac >= 1.0 - 1e-9
                kind = "boundary_snap_only" if is_full else "partial_boundary_snap"
                seed_kind = (
                    "boundary_snap_plus_seed_compensation"
                    if is_full
                    else "partial_boundary_snap_plus_seed_compensation"
                )
                repaired = {
                    **component,
                    "repaired_edge": edge,
                    "repair_delta": {"dx": dx, "dy": dy},
                    "snap_fraction": frac,
                }
                snap_only.append(
                    Candidate(label, ((block_id, dx, dy),), kind, repaired)
                )
                if seed_block != block_id:
                    for cdx in grid:
                        for cdy in grid:
                            if abs(cdx) <= EPS and abs(cdy) <= EPS:
                                continue
                            seed_comp.append(
                                Candidate(
                                    f"{label}_seed{seed_block}_{cdx:+.2f}_{cdy:+.2f}",
                                    ((block_id, dx, dy), (seed_block, cdx, cdy)),
                                    seed_kind,
                                    repaired,
                                    companion_block=seed_block,
                                )
                            )

    for group_row in audit.get("group_mib_components", []):
        if group_row.get("kind") != "group":
            continue
        members_with_boundary = [
            m for m in group_row.get("members", []) if m in boundary_block_ids
        ]
        if not members_with_boundary:
            continue
        for component in audit["active_violated_boundary_components"]:
            if int(component["block_id"]) not in members_with_boundary:
                continue
            block_id = int(component["block_id"])
            margins = component["margins"]
            for edge in component["violated_edges"]:
                for dx, dy, frac in _snap_deltas(edge, float(margins[edge]), snap_fractions):
                    label = f"grp{group_row['group_id']}_{_snap_label(block_id, edge, frac)}"
                    repaired = {
                        **component,
                        "repaired_edge": edge,
                        "repair_delta": {"dx": dx, "dy": dy},
                        "snap_fraction": frac,
                        "group_id": group_row["group_id"],
                    }
                    group_snap_only.append(
                        Candidate(label, ((block_id, dx, dy),), "group_boundary_snap", repaired)
                    )
                    if seed_block != block_id:
                        for cdx in grid:
                            for cdy in grid:
                                if abs(cdx) <= EPS and abs(cdy) <= EPS:
                                    continue
                                group_seed_comp.append(
                                    Candidate(
                                        f"{label}_seed{seed_block}_{cdx:+.2f}_{cdy:+.2f}",
                                        ((block_id, dx, dy), (seed_block, cdx, cdy)),
                                        "group_boundary_snap_plus_seed_compensation",
                                        repaired,
                                        companion_block=seed_block,
                                    )
                                )

    return snap_only + group_snap_only + seed_comp + group_seed_comp


def apply_candidate(positions: list[Box], candidate: Candidate) -> list[Box]:
    out = list(positions)
    for block_id, dx, dy in candidate.moves:
        x, y, w, h = out[block_id]
        out[block_id] = (x + dx, y + dy, w, h)
    return out


def replay_candidate(
    case: FloorSetCase,
    baseline: list[Box],
    before: dict[str, Any],
    candidate: Candidate,
) -> dict[str, Any]:
    after_positions = apply_candidate(baseline, candidate)
    legality = summarize_hard_legality(case, after_positions)
    after = evaluate_positions(case, after_positions, runtime=1.0)
    hard_feasible = bool(legality.is_feasible) and bool(after["quality"].get("feasible"))
    delta = actual_delta(before, after)
    return {
        "candidate_id": candidate.candidate_id,
        "repair_kind": candidate.repair_kind,
        "moves": [
            {"block_id": bid, "dx": dx, "dy": dy} for bid, dx, dy in candidate.moves
        ],
        "repaired_component": candidate.repaired_component,
        "hard_feasible": hard_feasible,
        "hard_summary": after["legality"],
        "official_before_quality": before["quality"],
        "official_after_quality": after["quality"],
        "official_after_soft_counts": {
            key: int(after["official"][key])
            for key in (
                "boundary_violations",
                "grouping_violations",
                "mib_violations",
                "total_soft_violations",
            )
        },
        **{k: float(v) for k, v in delta.items()},
        "all_vector_nonregressing": all_vector_nonregressing(delta, hard_feasible),
        "strict_meaningful_winner": strict_meaningful_winner(delta, hard_feasible),
    }


def run_active_soft_cone(
    base_dir: Path,
    specs: list[dict[str, Any]],
    *,
    floorset_root: Path | None = None,
    auto_download: bool = False,
    max_candidates_per_case: int | None = None,
) -> dict[str, Any]:
    started = time.perf_counter()
    case_ids = [int(s["case_id"]) for s in specs]
    cases = load_validation_cases(
        base_dir, case_ids, floorset_root=floorset_root, auto_download=auto_download
    )
    per_case: list[dict[str, Any]] = []
    all_rows: list[dict[str, Any]] = []
    for spec in specs:
        case_id = int(spec["case_id"])
        case = cases[case_id]
        baseline = apply_avnr(
            positions_from_case_targets(case),
            int(spec["seed_block"]),
            spec.get("applied_avnr_target"),
        )
        before = evaluate_positions(case, baseline, runtime=1.0)
        audit = active_soft_audit(case, baseline)
        candidates = generate_boundary_repair_candidates(case, baseline, int(spec["seed_block"]))
        if max_candidates_per_case is not None:
            candidates = candidates[:max_candidates_per_case]
        rows = [replay_candidate(case, baseline, before, cand) for cand in candidates]
        all_rows.extend({"case_id": case_id, **row} for row in rows)
        strict_rows = [row for row in rows if row["strict_meaningful_winner"]]
        best = min(rows, key=lambda r: float(r["official_like_cost_delta"]), default=None)
        selected = (
            min(strict_rows, key=lambda r: float(r["official_like_cost_delta"]))
            if strict_rows
            else best
        )
        per_case.append(
            {
                "case_id": case_id,
                "seed_block": int(spec["seed_block"]),
                "applied_avnr_target": spec.get("applied_avnr_target"),
                "step7s_result": spec.get("step7s_result"),
                "baseline_quality": before["quality"],
                "baseline_soft_counts": audit["official_soft_counts"],
                "active_violated_boundary_components": audit["active_violated_boundary_components"],
                "candidate_count": len(rows),
                "strict_winner_count": len(strict_rows),
                "has_strict_winner": bool(strict_rows),
                "best_candidate": best,
                "selected_candidate": selected,
                "blocker": classify_blocker(rows, audit),
            }
        )
    strict_total = sum(int(row["strict_winner_count"]) for row in per_case)
    strict_case_count = sum(int(bool(row["has_strict_winner"])) for row in per_case)
    return {
        "schema": "step7t_active_soft_summary_v1",
        "poc_certificate_kind": "margin_audit_bounded_boundary_snap_seed_compensation_exact_replay",
        "meaningful_cost_eps": MEANINGFUL_COST_EPS,
        "case_count": len(specs),
        "strict_winner_count": strict_total,
        "strict_winner_case_count": strict_case_count,
        "phase4_gate_open": strict_case_count >= 3,
        "max_candidates_per_case": max_candidates_per_case,
        "status_counts": dict(
            Counter(str(row.get("strict_meaningful_winner")) for row in all_rows)
        ),
        "candidate_count": len(all_rows),
        "runtime_seconds": time.perf_counter() - started,
        "per_case": per_case,
        "candidate_rows": all_rows,
    }


def classify_blocker(rows: list[dict[str, Any]], audit: dict[str, Any]) -> str:
    if not audit["active_violated_boundary_components"]:
        return "no_active_boundary_violation_in_margin_audit"
    if any(row.get("strict_meaningful_winner") for row in rows):
        return "strict_active_soft_repair_found"
    feasible = [row for row in rows if row.get("hard_feasible")]
    if not feasible:
        return "all_direct_repairs_hard_infeasible"
    soft_fixed = [row for row in feasible if float(row.get("soft_constraint_delta", 0.0)) < -EPS]
    if not soft_fixed:
        return "direct_repairs_do_not_reduce_soft_violation"
    if all(float(row.get("hpwl_delta", 0.0)) > EPS for row in soft_fixed):
        return "soft_repair_requires_hpwl_regression_under_bounded_compensation"
    if all(float(row.get("bbox_area_delta", 0.0)) > EPS for row in soft_fixed):
        return "soft_repair_requires_bbox_regression_under_bounded_compensation"
    return "cost_or_vector_gate_not_met"


def write_outputs(summary: dict[str, Any], summary_path: Path, markdown_path: Path) -> None:
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    lines = [
        "# Step7T Active-Soft Cone POC",
        "",
        f"- poc_certificate_kind: `{summary['poc_certificate_kind']}`",
        f"- strict_winner_count: `{summary['strict_winner_count']}` candidates",
        (
            f"- strict_winner_case_count: `{summary['strict_winner_case_count']}` "
            f"/ `{summary['case_count']}` cases"
        ),
        f"- candidate_count: `{summary['candidate_count']}`",
        f"- max_candidates_per_case: `{summary['max_candidates_per_case']}`",
        f"- phase4_gate_open: `{summary['phase4_gate_open']}`",
        f"- meaningful_cost_eps: `{summary['meaningful_cost_eps']}` (unchanged)",
        "",
        (
            "| case | active soft counts | candidates | strict | blocker | "
            "selected ΔC | selected ΔH | selected ΔA | selected ΔS |"
        ),
        "|---:|---|---:|---:|---|---:|---:|---:|---:|",
    ]
    for row in summary["per_case"]:
        counts = row["baseline_soft_counts"]
        best = row.get("selected_candidate") or {}
        lines.append(
            (
                "| {case_id} | B={b}/G={g}/M={m} | {cand} | {strict} | "
                "{blocker} | {dc:.6g} | {dh:.6g} | {da:.6g} | {ds:.6g} |"
            ).format(
                case_id=row["case_id"],
                b=counts["boundary_violations"],
                g=counts["grouping_violations"],
                m=counts["mib_violations"],
                cand=row["candidate_count"],
                strict=row["strict_winner_count"],
                blocker=row["blocker"],
                dc=float(best.get("official_like_cost_delta", 0.0)),
                dh=float(best.get("hpwl_delta", 0.0)),
                da=float(best.get("bbox_area_delta", 0.0)),
                ds=float(best.get("soft_constraint_delta", 0.0)),
            )
        )
    markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
