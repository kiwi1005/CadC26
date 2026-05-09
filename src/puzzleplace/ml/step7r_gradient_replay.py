"""Step7R-C fresh-metric replay for HPWL-gradient-nudged variants.

Reuses the Step7Q-F evaluator, slot finder, legality summarizer, and
strict/AVNR helpers verbatim. This module only adds:

- variant generation via the Step7R-C gradient operator;
- per-variant fresh-metric replay (parallelized);
- a Step7R-C-shaped summary that mirrors the Step7Q-F summary keys.

No new metric math is implemented here. No coordinate generation occurs
outside this sidecar.
"""

from __future__ import annotations

import time
from collections import Counter, defaultdict
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

from puzzleplace.alternatives.hpwl_gradient_nudge import (
    DEFAULT_STEP_LADDER,
    propose_gradient_variants,
)
from puzzleplace.data.schema import FloorSetCase
from puzzleplace.experiments.step7l_learning_guided_replay import load_validation_cases
from puzzleplace.geometry.legality import positions_from_case_targets, summarize_hard_legality
from puzzleplace.ml.step7q_fresh_metric_replay import (
    EPS,
    actual_delta,
    all_vector_nonregressing,
    failure_attribution,
    objective_aware_nonoverlap_slot,
    official_like_evaluator,
    overlaps_any,
    quality_gate_status,
    same_box,
    strict_meaningful_winner,
)
from puzzleplace.ml.step7q_operator_learning import read_jsonl, write_json, write_jsonl

Box = tuple[float, float, float, float]
SCHEMA_ROW = "step7r_gradient_replay_row_v1"
SCHEMA_SUMMARY = "step7r_gradient_replay_summary_v1"

__all__ = [
    "evaluate_variant",
    "replay_gradient_deck",
    "summarize_gradient_replay",
]


def replay_gradient_deck(
    base_dir: Path,
    examples_path: Path,
    step7q_rows_path: Path,
    replay_rows_out_path: Path,
    summary_out_path: Path,
    failures_out_path: Path,
    *,
    n_workers: int | None = None,
    step_ladder: Sequence[float] = DEFAULT_STEP_LADDER,
    floorset_root: Path | None = None,
    auto_download: bool = False,
    avnr_only: bool = True,
) -> dict[str, Any]:
    """Run the Step7R-C HPWL gradient nudge replay and emit summary artifacts."""

    started = time.perf_counter()
    rows = read_jsonl(step7q_rows_path)
    avnr_rows = (
        [row for row in rows if bool(row.get("actual_all_vector_nonregressing"))]
        if avnr_only
        else list(rows)
    )
    case_ids = sorted(
        {int(row["case_id"]) for row in avnr_rows if str(row.get("case_id", "")).isdigit()}
    )
    cases = load_validation_cases(
        base_dir, case_ids, floorset_root=floorset_root, auto_download=auto_download
    )

    variants: list[dict[str, Any]] = []
    parent_zero_gradient: list[dict[str, Any]] = []
    for row in avnr_rows:
        case_id_str = str(row.get("case_id"))
        if not case_id_str.isdigit():
            continue
        case = cases.get(int(case_id_str))
        if case is None:
            continue
        row_variants = propose_gradient_variants(case, row, step_ladder=step_ladder)
        if not row_variants:
            parent_zero_gradient.append(
                {
                    "parent_candidate_id": row.get("candidate_id"),
                    "case_id": case_id_str,
                    "block_id": row.get("block_id"),
                    "reason": "zero_gradient",
                }
            )
            continue
        variants.extend(row_variants)

    baseline_cache: dict[str, tuple[list[Box], dict[str, Any]]] = {}
    for case_id_int in case_ids:
        case = cases.get(case_id_int)
        if case is None:
            continue
        baseline = positions_from_case_targets(case)
        baseline_cache[str(case_id_int)] = (baseline, official_like_evaluator(case, baseline))

    workers_used = max(1, min(int(n_workers or 0) or min(48, len(variants)), len(variants) or 1))
    if workers_used > 1 and variants:
        worker_args = [
            (variant, _serialize_case(cases[int(variant["case_id"])])) for variant in variants
        ]
        with ProcessPoolExecutor(max_workers=workers_used) as pool:
            replay_rows = list(pool.map(_replay_worker, worker_args))
    else:
        replay_rows = []
        for variant in variants:
            case = cases[int(variant["case_id"])]
            baseline, before_eval = baseline_cache[str(variant["case_id"])]
            replay_rows.append(evaluate_variant(variant, case, baseline, before_eval))

    if workers_used > 1 and variants:
        # Workers used local baselines; nothing else to merge.
        pass

    write_jsonl(replay_rows_out_path, replay_rows)
    summary = summarize_gradient_replay(
        replay_rows,
        avnr_input_count=len(avnr_rows),
        zero_gradient_parent_count=len(parent_zero_gradient),
        n_workers_used=workers_used,
        runtime_proxy_ms=(time.perf_counter() - started) * 1000.0,
        step_ladder=tuple(step_ladder),
    )
    summary["replay_rows_path"] = str(replay_rows_out_path)
    summary["step7q_rows_path"] = str(step7q_rows_path)
    write_json(summary_out_path, summary)
    write_json(failures_out_path, _failures_by_case(replay_rows))
    summary_out_path.with_suffix(".md").write_text(_summary_markdown(summary), encoding="utf-8")
    return summary


def evaluate_variant(
    variant: dict[str, Any],
    case: FloorSetCase,
    baseline: list[Box],
    before_eval: dict[str, Any],
) -> dict[str, Any]:
    """Evaluate one gradient variant against the baseline; reuses Step7Q helpers."""

    block_id = int(variant["block_id"])
    target_field = variant["post_nudge_target_box"]
    target: Box = (
        float(target_field[0]),
        float(target_field[1]),
        float(target_field[2]),
        float(target_field[3]),
    )

    base_row = _row_base(variant)

    if same_box(target, baseline[block_id]):
        return _unrealized(base_row, "no_op_after_gradient_nudge", target)

    slot_adjustment: dict[str, Any] | None = None
    if overlaps_any(target, baseline, skip=block_id):
        slot_choice = objective_aware_nonoverlap_slot(
            case, baseline, before_eval, official_like_evaluator, block_id, target
        )
        if slot_choice is None:
            return _unrealized(base_row, "no_feasible_slot_after_overlap", target)
        slot_target = slot_choice["slot_target_box"]
        target = (
            float(slot_target[0]),
            float(slot_target[1]),
            float(slot_target[2]),
            float(slot_target[3]),
        )
        slot_adjustment = {
            "objective_aware_slot": True,
            "requested_target_box": list(variant["post_nudge_target_box"]),
            **slot_choice,
        }

    after = list(baseline)
    after[block_id] = target
    after_eval = official_like_evaluator(case, after)
    legality = summarize_hard_legality(case, after)
    delta = actual_delta(before_eval, after_eval)
    hard_feasible = bool(legality.is_feasible) and bool(after_eval["quality"].get("feasible"))
    status = quality_gate_status(delta, hard_feasible)
    strict = strict_meaningful_winner(delta, hard_feasible)
    avnr = all_vector_nonregressing(delta, hard_feasible)
    return {
        **base_row,
        "fresh_metric_available": True,
        "post_nudge_target_box": list(target),
        "slot_adjustment": slot_adjustment,
        "hard_feasible": hard_feasible,
        "hard_feasible_nonnoop": hard_feasible,
        "quality_gate_status": status,
        "strict_meaningful_winner": strict,
        "actual_all_vector_nonregressing": avnr,
        "actual_objective_vector": delta,
        **delta,
        "official_like_cost_improving": delta["official_like_cost_delta"] < -EPS,
        "metric_regressing": delta["official_like_cost_delta"] > EPS,
        "hpwl_strict_improvement": delta["hpwl_delta"] < -EPS,
        "legality": {
            "is_feasible": legality.is_feasible,
            "overlap_violations": legality.overlap_violations,
            "area_violations": legality.area_violations,
            "dimension_violations": legality.dimension_violations,
        },
        "failure_attribution": failure_attribution(status, delta, legality.overlap_violations),
        "rejection_reason": None,
    }


def summarize_gradient_replay(
    rows: list[dict[str, Any]],
    *,
    avnr_input_count: int,
    zero_gradient_parent_count: int,
    n_workers_used: int,
    runtime_proxy_ms: float,
    step_ladder: tuple[float, ...],
) -> dict[str, Any]:
    fresh = [row for row in rows if row.get("fresh_metric_available")]
    hard = [row for row in fresh if row.get("hard_feasible_nonnoop")]
    strict = [row for row in hard if row.get("strict_meaningful_winner")]
    avnr = [row for row in hard if row.get("actual_all_vector_nonregressing")]
    hpwl_strict = [row for row in hard if row.get("hpwl_strict_improvement")]
    overlap_count = sum(
        int(str(row.get("failure_attribution")) == "overlap_after_splice") for row in rows
    )
    soft_reg = sum(int(float(row.get("soft_constraint_delta", 0.0)) > EPS) for row in hard)
    bbox_reg = sum(int(float(row.get("bbox_area_delta", 0.0)) > EPS) for row in hard)
    hpwl_reg = sum(int(float(row.get("hpwl_delta", 0.0)) > EPS) for row in hard)
    represented_cases = {str(row.get("case_id")) for row in rows}
    case_counter = Counter(str(row.get("case_id")) for row in rows)
    largest_share = (
        max(case_counter.values()) / max(len(rows), 1) if case_counter else 0.0
    )
    step_factor_winners: dict[str, int] = {f"{factor:g}": 0 for factor in step_ladder}
    for row in strict:
        key = f"{float(row.get('step_factor', 0.0)):g}"
        step_factor_winners[key] = step_factor_winners.get(key, 0) + 1

    phase2_gate_open = len(strict) >= 1
    decision = (
        "gradient_replay_strict_gate_open"
        if phase2_gate_open
        else "gradient_replay_strict_gate_closed"
    )
    allowed_next_phase = "step7r_c_phase2_full_replay" if phase2_gate_open else "close_step7r"

    return {
        "schema": SCHEMA_SUMMARY,
        "decision": decision,
        "avnr_input_count": avnr_input_count,
        "zero_gradient_parent_count": zero_gradient_parent_count,
        "variant_count": len(rows),
        "fresh_metric_available_count": len(fresh),
        "fresh_hard_feasible_nonnoop_count": len(hard),
        "overlap_after_splice_count": overlap_count,
        "soft_regression_count": soft_reg,
        "soft_regression_rate": soft_reg / max(len(rows), 1),
        "bbox_regression_count": bbox_reg,
        "bbox_regression_rate": bbox_reg / max(len(rows), 1),
        "hpwl_regression_count": hpwl_reg,
        "hpwl_regression_rate": hpwl_reg / max(len(rows), 1),
        "actual_all_vector_nonregressing_count": len(avnr),
        "hpwl_strict_improvement_count": len(hpwl_strict),
        "strict_meaningful_winner_count": len(strict),
        "represented_case_count": len(represented_cases),
        "largest_case_share": largest_share,
        "step_factor_winner_breakdown": step_factor_winners,
        "step_ladder": list(step_ladder),
        "n_workers_used": n_workers_used,
        "phase2_gate_open": phase2_gate_open,
        "allowed_next_phase": allowed_next_phase,
        "phase4_gate_open": False,
        "step7q_f_baseline_all_vector_nonregressing_count": 27,
        "status_counts": dict(Counter(str(row.get("quality_gate_status", "")) for row in rows)),
        "failure_counts": dict(Counter(str(row.get("failure_attribution", "")) for row in rows)),
        "runtime_proxy_ms": runtime_proxy_ms,
        "runtime_proxy_per_variant_ms": runtime_proxy_ms / max(len(rows), 1),
    }


def _row_base(variant: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema": SCHEMA_ROW,
        "parent_candidate_id": variant.get("parent_candidate_id"),
        "case_id": str(variant.get("case_id")),
        "block_id": int(variant["block_id"]),
        "step_factor": float(variant.get("step_factor", 0.0)),
        "current_box": list(variant["current_box"]),
        "gradient_vec": list(variant.get("gradient_vec", [0.0, 0.0])),
        "gradient_magnitude": float(variant.get("gradient_magnitude", 0.0)),
    }


def _unrealized(base_row: dict[str, Any], reason: str, target: Box) -> dict[str, Any]:
    return {
        **base_row,
        "fresh_metric_available": False,
        "post_nudge_target_box": list(target),
        "slot_adjustment": None,
        "hard_feasible": False,
        "hard_feasible_nonnoop": False,
        "quality_gate_status": reason,
        "strict_meaningful_winner": False,
        "actual_all_vector_nonregressing": False,
        "rejection_reason": reason,
        "failure_attribution": reason,
    }


def _failures_by_case(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_case: dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        by_case[str(row.get("case_id"))][str(row.get("failure_attribution", ""))] += 1
    return {
        "schema": "step7r_gradient_failures_by_case_v1",
        "failures_by_case": {
            case_id: dict(counter) for case_id, counter in sorted(by_case.items())
        },
    }


def _summary_markdown(summary: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# Step7R-C HPWL Gradient Replay Summary",
            "",
            f"Decision: `{summary['decision']}`",
            "",
            f"- avnr_input_count: {summary['avnr_input_count']}",
            f"- variant_count: {summary['variant_count']}",
            f"- fresh_hard_feasible_nonnoop_count: {summary['fresh_hard_feasible_nonnoop_count']}",
            f"- strict_meaningful_winner_count: {summary['strict_meaningful_winner_count']}",
            f"- actual_all_vector_nonregressing_count: "
            f"{summary['actual_all_vector_nonregressing_count']}",
            f"- hpwl_strict_improvement_count: {summary['hpwl_strict_improvement_count']}",
            f"- represented_case_count: {summary['represented_case_count']}",
            f"- phase2_gate_open: {summary['phase2_gate_open']}",
            f"- allowed_next_phase: {summary['allowed_next_phase']}",
            f"- step_factor_winner_breakdown: {summary['step_factor_winner_breakdown']}",
            f"- runtime_proxy_ms: {summary['runtime_proxy_ms']:.1f}",
            "",
        ]
    )


# ---------- multiprocessing worker plumbing ----------


def _serialize_case(case: FloorSetCase) -> FloorSetCase:
    """Identity for now; FloorSetCase already pickles via dataclass+tensor."""

    return case


def _replay_worker(args: tuple[dict[str, Any], FloorSetCase]) -> dict[str, Any]:
    variant, case = args
    baseline = positions_from_case_targets(case)
    before_eval = official_like_evaluator(case, baseline)
    return evaluate_variant(variant, case, baseline, before_eval)
