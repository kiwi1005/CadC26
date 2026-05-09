"""Step7V live-layout active-soft adapter experiment.

Step7T found strict winners from validation-label replay baselines.  Step7V tests
the next, stricter question: if the same active-soft repair primitive starts from
positions produced by the current contest optimizer, does the strict signal
survive?  The module remains sidecar-only and does not modify runtime/finalizer
code.
"""

from __future__ import annotations

import json
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

from puzzleplace.eval.official import evaluate_positions
from puzzleplace.experiments.step7l_learning_guided_replay import load_validation_cases
from puzzleplace.experiments.step7t_active_soft_cone import (
    active_soft_audit,
    generate_boundary_repair_candidates,
    load_step7s_case_specs,
    replay_candidate,
)
from puzzleplace.geometry.legality import summarize_hard_legality


def _ensure_contest_import_paths(base_dir: Path) -> None:
    for path in (
        base_dir,
        base_dir / "src",
        base_dir / "external" / "FloorSet" / "iccad2026contest",
    ):
        path_str = str(path.resolve())
        if path_str not in sys.path:
            sys.path.insert(0, path_str)


def _live_optimizer_positions(
    base_dir: Path,
    case: Any,
    *,
    objective_selection_k: int = 1,
) -> tuple[list[tuple[float, float, float, float]], dict[str, Any]]:
    _ensure_contest_import_paths(base_dir)
    from puzzleplace.optimizer.contest import ContestOptimizer

    optimizer = ContestOptimizer(objective_selection_k=objective_selection_k)
    positions, report = optimizer.solve_with_report(
        case.block_count,
        case.area_targets,
        case.b2b_edges,
        case.p2b_edges,
        case.pins_pos,
        case.constraints,
        case.target_positions,
    )
    return [tuple(map(float, box)) for box in positions], dict(report)


def _load_cached_baseline(
    cache_dir: Path, case_id: int
) -> tuple[list[tuple[float, float, float, float]], dict[str, Any]] | None:
    cache_path = cache_dir / f"case{case_id}.json"
    if not cache_path.exists():
        return None
    data = json.loads(cache_path.read_text(encoding="utf-8"))
    positions = [tuple(map(float, box)) for box in data["positions"]]
    return positions, dict(data.get("optimizer_report", {}))


def run_live_active_soft_adapter(
    base_dir: Path,
    specs: list[dict[str, Any]],
    *,
    floorset_root: Path | None = None,
    auto_download: bool = False,
    max_candidates_per_case: int | None = 50,
    objective_selection_k: int = 1,
    baseline_cache_dir: Path | None = None,
) -> dict[str, Any]:
    started = time.perf_counter()
    case_ids = [int(s["case_id"]) for s in specs]
    cases = load_validation_cases(
        base_dir, case_ids, floorset_root=floorset_root, auto_download=auto_download
    )
    per_case: list[dict[str, Any]] = []
    candidate_rows: list[dict[str, Any]] = []
    for spec in specs:
        case_id = int(spec["case_id"])
        case = cases[case_id]
        seed_block = int(spec["seed_block"])
        cached = (
            _load_cached_baseline(baseline_cache_dir, case_id)
            if baseline_cache_dir is not None
            else None
        )
        cached_baseline_used = cached is not None
        if cached is not None:
            live_positions, optimizer_report = cached
        else:
            live_positions, optimizer_report = _live_optimizer_positions(
                base_dir, case, objective_selection_k=objective_selection_k
            )
        before = evaluate_positions(case, live_positions, runtime=1.0)
        hard = summarize_hard_legality(case, live_positions)
        hard_feasible_before = bool(hard.is_feasible) and bool(before["quality"].get("feasible"))
        audit = active_soft_audit(case, live_positions)
        candidates = generate_boundary_repair_candidates(case, live_positions, seed_block)
        if max_candidates_per_case is not None:
            candidates = candidates[:max_candidates_per_case]
        rows = [replay_candidate(case, live_positions, before, cand) for cand in candidates]
        for row in rows:
            candidate_rows.append(
                {
                    "case_id": case_id,
                    "seed_block": seed_block,
                    "baseline_source": "contest_optimizer.solve_with_report",
                    "cached_baseline_used": cached_baseline_used,
                    **row,
                }
            )
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
                "seed_block": seed_block,
                "step7s_result": spec.get("step7s_result"),
                "baseline_source": "contest_optimizer.solve_with_report",
                "cached_baseline_used": cached_baseline_used,
                "baseline_cache_path": (
                    str(baseline_cache_dir / f"case{case_id}.json")
                    if cached_baseline_used and baseline_cache_dir is not None
                    else None
                ),
                "target_positions_passed_to_optimizer": case.target_positions is not None,
                "optimizer_report": optimizer_report,
                "baseline_quality": before["quality"],
                "baseline_hard_feasible": hard_feasible_before,
                "baseline_hard_summary": before["legality"],
                "baseline_soft_counts": audit["official_soft_counts"],
                "active_violated_boundary_components": audit[
                    "active_violated_boundary_components"
                ],
                "candidate_count": len(rows),
                "strict_winner_count": len(strict_rows),
                "has_strict_winner": bool(strict_rows),
                "selected_candidate": selected,
                "blocker": classify_live_blocker(rows, hard_feasible_before, audit),
            }
        )

    strict_case_count = sum(int(row["has_strict_winner"]) for row in per_case)
    strict_count = sum(int(row["strict_winner_count"]) for row in per_case)
    cached_baseline_case_count = sum(int(row["cached_baseline_used"]) for row in per_case)
    blocker_counts = Counter(row["blocker"] for row in per_case)
    return {
        "schema": "step7v_live_active_soft_adapter_summary_v1",
        "decision": (
            "live_adapter_phase4_gate_open"
            if strict_case_count >= 3
            else "live_adapter_phase4_gate_closed"
        ),
        "case_count": len(per_case),
        "candidate_count": len(candidate_rows),
        "strict_winner_count": strict_count,
        "strict_winner_case_count": strict_case_count,
        "phase4_gate_open": strict_case_count >= 3,
        "baseline_source": "contest_optimizer.solve_with_report",
        "baseline_cache_dir": str(baseline_cache_dir) if baseline_cache_dir is not None else None,
        "cached_baseline_used": cached_baseline_case_count > 0,
        "cached_baseline_case_count": cached_baseline_case_count,
        "validation_label_baseline_used": False,
        "target_positions_passed_to_optimizer": True,
        "objective_selection_k": objective_selection_k,
        "max_candidates_per_case": max_candidates_per_case,
        "blocker_counts": dict(sorted(blocker_counts.items())),
        "runtime_seconds": time.perf_counter() - started,
        "per_case": per_case,
        "candidate_rows": candidate_rows,
        "next_recommendation": next_recommendation(strict_case_count, blocker_counts),
    }


def classify_live_blocker(
    rows: list[dict[str, Any]],
    baseline_hard_feasible: bool,
    audit: dict[str, Any],
) -> str:
    if not baseline_hard_feasible:
        return "live_optimizer_baseline_hard_infeasible"
    if not audit["active_violated_boundary_components"]:
        return "live_optimizer_has_no_active_boundary_violation"
    if any(row.get("strict_meaningful_winner") for row in rows):
        return "strict_live_active_soft_repair_found"
    feasible = [row for row in rows if row.get("hard_feasible")]
    if not feasible:
        return "all_live_repairs_hard_infeasible"
    soft_fixed = [row for row in feasible if float(row.get("soft_constraint_delta", 0.0)) < 0.0]
    if not soft_fixed:
        return "live_repairs_do_not_reduce_soft_violation"
    if all(float(row.get("hpwl_delta", 0.0)) > 1e-9 for row in soft_fixed):
        return "live_soft_repair_requires_hpwl_regression"
    if all(float(row.get("bbox_area_delta", 0.0)) > 1e-9 for row in soft_fixed):
        return "live_soft_repair_requires_bbox_regression"
    return "live_cost_or_vector_gate_not_met"


def next_recommendation(strict_case_count: int, blocker_counts: Counter[str]) -> str:
    if strict_case_count >= 3:
        return "promote_live_active_soft_adapter_to_runtime_integration_design_review"
    if blocker_counts.get("live_optimizer_baseline_hard_infeasible", 0):
        return "fix_or_select_hard_feasible_live_baselines_before_active_soft_runtime_port"
    if blocker_counts.get("live_optimizer_has_no_active_boundary_violation", 0):
        return "active_soft_signal_is_replay_baseline_specific_seek_other_live_soft_sources"
    return "widen_live_active_soft_candidate_generation_or_add_compensating_repack"


def write_outputs(summary: dict[str, Any], out: Path, markdown_out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    lines = [
        "# Step7V Live Active-Soft Adapter",
        "",
        f"Decision: `{summary['decision']}`",
        "",
        f"- strict winners: `{summary['strict_winner_count']}`",
        (
            f"- strict winner cases: `{summary['strict_winner_case_count']}` / "
            f"`{summary['case_count']}`"
        ),
        f"- phase4 gate open: `{summary['phase4_gate_open']}`",
        f"- validation label baseline used: `{summary['validation_label_baseline_used']}`",
        (
            f"- cached live baselines: `{summary.get('cached_baseline_case_count', 0)}` / "
            f"`{summary['case_count']}`"
        ),
        f"- next: `{summary['next_recommendation']}`",
        "",
        (
            "| case | baseline hard | soft counts | candidates | strict | blocker | "
            "selected ΔC | selected ΔH | selected ΔA | selected ΔS |"
        ),
        "|---:|:---:|---|---:|---:|---|---:|---:|---:|---:|",
    ]
    for row in summary["per_case"]:
        if row.get("status") and row.get("status") != "completed":
            lines.append(
                (
                    "| {case_id} | n/a | n/a | n/a | n/a | {blocker} | "
                    "0 | 0 | 0 | 0 |"
                ).format(
                    case_id=row.get("case_id"),
                    blocker=row.get("blocker", row.get("status")),
                )
            )
            continue
        counts = row["baseline_soft_counts"]
        selected = row.get("selected_candidate") or {}
        lines.append(
            (
                "| {case_id} | {hard} | B={b}/G={g}/M={m} | {cand} | {strict} | "
                "{blocker} | {dc:.8g} | {dh:.8g} | {da:.8g} | {ds:.8g} |"
            ).format(
                case_id=row["case_id"],
                hard="yes" if row["baseline_hard_feasible"] else "no",
                b=counts["boundary_violations"],
                g=counts["grouping_violations"],
                m=counts["mib_violations"],
                cand=row["candidate_count"],
                strict=row["strict_winner_count"],
                blocker=row["blocker"],
                dc=float(selected.get("official_like_cost_delta", 0.0)),
                dh=float(selected.get("hpwl_delta", 0.0)),
                da=float(selected.get("bbox_area_delta", 0.0)),
                ds=float(selected.get("soft_constraint_delta", 0.0)),
            )
        )
    lines.append("")
    markdown_out.write_text("\n".join(lines), encoding="utf-8")


def load_specs_from_step7s(path: Path, case_ids: list[int] | None = None) -> list[dict[str, Any]]:
    specs = load_step7s_case_specs(path)
    if case_ids is not None:
        wanted = set(case_ids)
        specs = [spec for spec in specs if int(spec["case_id"]) in wanted]
    return specs


def aggregate_case_summaries(case_summaries: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate one-case Step7V summaries into the 8-case gate artifact."""

    per_case: list[dict[str, Any]] = []
    candidate_rows: list[dict[str, Any]] = []
    strict_winner_count = 0
    strict_winner_case_count = 0
    completed_case_count = 0
    failed_case_count = 0
    for summary in sorted(case_summaries, key=lambda row: int(row.get("case_id", 10**9))):
        status = str(summary.get("status", "completed"))
        if status != "completed":
            failed_case_count += 1
            per_case.append(summary)
            continue
        completed_case_count += 1
        strict_winner_count += int(summary.get("strict_winner_count", 0))
        strict_winner_case_count += int(summary.get("strict_winner_case_count", 0))
        per_case.extend(summary.get("per_case", []))
        candidate_rows.extend(summary.get("candidate_rows", []))

    blocker_counts = Counter(str(row.get("blocker")) for row in per_case)
    cached_baseline_case_count = sum(
        int(bool(row.get("cached_baseline_used"))) for row in per_case
    )
    gate_open = failed_case_count == 0 and strict_winner_case_count >= 3
    decision = (
        "live_adapter_phase4_gate_open"
        if gate_open
        else "live_adapter_partial_failures_present"
        if failed_case_count
        else "live_adapter_phase4_gate_closed"
    )
    recommendation = (
        "resolve_parallel_timeouts_or_add_fast_live_baseline_cache_then_rerun_failed_cases"
        if failed_case_count
        else next_recommendation(strict_winner_case_count, blocker_counts)
    )
    return {
        "schema": "step7v_live_active_soft_parallel_summary_v1",
        "decision": decision,
        "case_count": len(per_case),
        "completed_case_count": completed_case_count,
        "failed_case_count": failed_case_count,
        "candidate_count": len(candidate_rows),
        "strict_winner_count": strict_winner_count,
        "strict_winner_case_count": strict_winner_case_count,
        "phase4_gate_open": gate_open,
        "baseline_source": "contest_optimizer.solve_with_report",
        "baseline_cache_dir": next(
            (
                str(summary.get("baseline_cache_dir"))
                for summary in case_summaries
                if summary.get("baseline_cache_dir") is not None
            ),
            None,
        ),
        "cached_baseline_used": cached_baseline_case_count > 0,
        "cached_baseline_case_count": cached_baseline_case_count,
        "validation_label_baseline_used": False,
        "blocker_counts": dict(sorted(blocker_counts.items())),
        "per_case": per_case,
        "candidate_rows": candidate_rows,
        "next_recommendation": recommendation,
    }
