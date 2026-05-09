"""Step7P Phase1 causal subproblem atlas builder."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from puzzleplace.repack.causal_subproblem import (
    bbox_hull_risk_class,
    bool_value,
    extract_block_ids,
    forbidden_term_count,
    has_boundary_member,
    has_mib_member,
    infer_failure_bucket,
    infer_intent_family,
    int_value,
    metric_confidence,
)

VALIDATION_LABEL_POLICY = "labels used for replay/evaluation only, not request generation"


def build_causal_subproblem_atlas(
    stagnation_lock_path: Path,
    step7ml_i_candidates_path: Path,
    step7ml_k_candidates_path: Path,
    step7m_phase2_summary_path: Path,
    step7m_phase4_summary_path: Path,
    step7n_i_target_quality_path: Path,
    validation_cases: Iterable[int],
    out_path: Path,
    summary_out_path: Path,
    markdown_out_path: Path,
    failures_out_path: Path,
) -> dict[str, Any]:
    """Build Phase1 causal attribution rows and summary."""

    lock = load_json(stagnation_lock_path)
    if lock.get("phase1_gate_open") is not True:
        rows: list[dict[str, Any]] = []
        summary = fail_closed_summary(
            lock,
            out_path,
            summary_out_path,
            markdown_out_path,
            failures_out_path,
            reason="stagnation_lock_phase1_gate_closed",
        )
        write_jsonl(out_path, rows)
        write_json(summary_out_path, summary)
        write_json(failures_out_path, {})
        markdown_out_path.write_text(causal_markdown(summary), encoding="utf-8")
        return summary

    cases = {str(int(case_id)) for case_id in validation_cases}
    rows = []
    rows.extend(
        normalize_candidate_rows(
            candidate_rows(load_json(step7ml_i_candidates_path), "rows"),
            source="step7ml_i",
            source_path=step7ml_i_candidates_path,
            validation_cases=cases,
        )
    )
    rows.extend(
        normalize_candidate_rows(
            candidate_rows(load_json(step7ml_k_candidates_path), "selected_rows"),
            source="step7ml_k",
            source_path=step7ml_k_candidates_path,
            validation_cases=cases,
        )
    )
    rows.extend(
        normalize_replay_rows_from_summary(
            step7m_phase2_summary_path, source="step7m_phase2", validation_cases=cases
        )
    )
    rows.extend(
        normalize_replay_rows_from_summary(
            step7m_phase4_summary_path, source="step7m_phase4", validation_cases=cases
        )
    )
    target_quality = load_json(step7n_i_target_quality_path)
    for row in rows:
        enrich_with_target_failure_context(row, target_quality)
    write_jsonl(out_path, rows)
    failures = failures_by_cause(rows)
    write_json(failures_out_path, failures)
    summary = summarize_atlas(
        rows,
        out_path=out_path,
        summary_out_path=summary_out_path,
        markdown_out_path=markdown_out_path,
        failures_out_path=failures_out_path,
        input_paths={
            "stagnation_lock": stagnation_lock_path,
            "step7ml_i_candidates": step7ml_i_candidates_path,
            "step7ml_k_candidates": step7ml_k_candidates_path,
            "step7m_phase2_summary": step7m_phase2_summary_path,
            "step7m_phase4_summary": step7m_phase4_summary_path,
            "step7n_i_target_quality": step7n_i_target_quality_path,
        },
    )
    write_json(summary_out_path, summary)
    markdown_out_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_out_path.write_text(causal_markdown(summary), encoding="utf-8")
    return summary


def normalize_candidate_rows(
    source_rows: list[dict[str, Any]],
    *,
    source: str,
    source_path: Path,
    validation_cases: set[str],
) -> list[dict[str, Any]]:
    rows = []
    for index, row in enumerate(source_rows):
        case_id = str(row.get("case_id"))
        if validation_cases and case_id not in validation_cases:
            continue
        rows.append(
            normalize_source_row(row, source=source, source_path=source_path, row_index=index)
        )
    return rows


def normalize_replay_rows_from_summary(
    summary_path: Path, *, source: str, validation_cases: set[str]
) -> list[dict[str, Any]]:
    summary = load_json(summary_path)
    rows_path = Path(str(summary.get("replay_rows_path", "")))
    if not rows_path.exists():
        return []
    rows = []
    for index, row in enumerate(read_jsonl(rows_path)):
        case_id = str(row.get("case_id"))
        if validation_cases and case_id not in validation_cases:
            continue
        rows.append(
            normalize_source_row(row, source=source, source_path=rows_path, row_index=index)
        )
    return rows


def normalize_source_row(
    row: dict[str, Any], *, source: str, source_path: Path, row_index: int
) -> dict[str, Any]:
    failure_bucket = infer_failure_bucket(row)
    intent_family = infer_intent_family(row, failure_bucket)
    block_ids = extract_block_ids(row)
    case_id = str(row.get("case_id"))
    candidate_id = str(row.get("candidate_id") or row.get("request_id") or f"row{row_index}")
    return {
        "schema": "step7p_phase1_causal_subproblem_row_v1",
        "case_id": case_id,
        "subproblem_id": f"{source}:{case_id}:{row_index:05d}",
        "seed_source": source,
        "seed_artifact": str(source_path),
        "seed_candidate_id": candidate_id,
        "seed_source_candidate_id": row.get("source_candidate_id"),
        "intent_family": intent_family,
        "seed_failure_bucket": failure_bucket,
        "affected_block_ids": block_ids,
        "blocker_block_ids": inferred_blockers(block_ids, failure_bucket),
        "soft_linked_block_ids": inferred_soft_links(block_ids, failure_bucket),
        "critical_net_ids": [],
        "boundary_constraint_touched": has_boundary_member(row)
        or intent_family == "boundary_contact_guarded",
        "mib_constraint_touched": has_mib_member(row)
        or intent_family == "mib_shape_preserve_repack",
        "group_constraint_touched": bool(row.get("closure_bbox"))
        or int_value(row.get("block_count")) > 1,
        "bbox_hull_risk_class": bbox_hull_risk_class(row),
        "allowed_repack_families": allowed_repack_families(intent_family),
        "metric_confidence": metric_confidence(row, source),
        "hard_feasible_nonnoop": bool_value(
            row.get("hard_feasible_non_noop") or row.get("hard_feasible_nonnoop")
        ),
        "metric_regressing": bool_value(row.get("metric_regressing")),
        "dominated_by_original": bool_value(row.get("dominated_by_original")),
        "quality_gate_pass": bool_value(row.get("quality_gate_pass")),
        "official_like_improving": bool_value(
            row.get("official_like_improving") or row.get("official_like_cost_improving")
        ),
        "objective_vector": objective_vector(row),
        "moved_block_count": int_value(row.get("moved_block_count") or len(block_ids)),
        "route_class": row.get("route_class") or row.get("request_route_class"),
        "decoder": row.get("decoder"),
        "validation_label_policy": VALIDATION_LABEL_POLICY,
    }


def enrich_with_target_failure_context(row: dict[str, Any], target_quality: dict[str, Any]) -> None:
    retained = target_quality.get("target_failure_bucket_counts_retained")
    row["target_failure_bucket_context_available"] = isinstance(retained, dict)
    if isinstance(retained, dict):
        row["known_target_failure_buckets"] = sorted(retained)


def inferred_blockers(block_ids: list[int], failure_bucket: str) -> list[int]:
    if failure_bucket not in {"wrong_slot", "overlap_after_splice", "bad_internal_repack"}:
        return []
    return block_ids[1:4]


def inferred_soft_links(block_ids: list[int], failure_bucket: str) -> list[int]:
    if failure_bucket != "soft_regression":
        return []
    return block_ids[:4]


def allowed_repack_families(intent_family: str) -> list[str]:
    if intent_family == "soft_guarded_repair":
        return ["soft_guarded_repair", "order_preserving_row_repack", "pareto_vector_filter"]
    if intent_family == "blocker_chain_unblock":
        return ["blocker_chain_unblock", "closure_translate_with_repair", "pareto_vector_filter"]
    if intent_family == "mib_shape_preserve_repack":
        return [
            "mib_shape_preserve_repack",
            "closure_translate_with_repair",
            "pareto_vector_filter",
        ]
    return [intent_family, "order_preserving_row_repack", "pareto_vector_filter"]


def objective_vector(row: dict[str, Any]) -> dict[str, float | None]:
    raw = row.get("actual_objective_vector")
    vector = raw if isinstance(raw, dict) else row
    return {
        "hpwl_delta": float_or_none(vector.get("hpwl_delta")),
        "bbox_area_delta": float_or_none(vector.get("bbox_area_delta")),
        "soft_constraint_delta": float_or_none(vector.get("soft_constraint_delta")),
        "official_like_cost_delta": float_or_none(vector.get("official_like_cost_delta")),
    }


def summarize_atlas(
    rows: list[dict[str, Any]],
    *,
    out_path: Path,
    summary_out_path: Path,
    markdown_out_path: Path,
    failures_out_path: Path,
    input_paths: dict[str, Path],
) -> dict[str, Any]:
    case_counts = Counter(str(row["case_id"]) for row in rows)
    intent_counts = Counter(str(row["intent_family"]) for row in rows)
    failure_counts = Counter(str(row["seed_failure_bucket"]) for row in rows)
    largest_case_id, largest_case_count = largest_case(case_counts)
    forbidden_count = forbidden_term_count(rows)
    unknown_share = failure_counts.get("unknown", 0) / max(len(rows), 1)
    represented_case_count = len(case_counts)
    nonzero_intent_family_count = sum(1 for count in intent_counts.values() if count > 0)
    largest_case_share = largest_case_count / max(len(rows), 1)
    phase2_gate_open = (
        len(rows) >= 80
        and represented_case_count >= 8
        and largest_case_share <= 0.35
        and nonzero_intent_family_count >= 4
        and unknown_share <= 0.40
        and forbidden_count == 0
        and (
            failure_counts.get("soft_regression", 0)
            + failure_counts.get("bbox_regression", 0)
        )
        > 0
    )
    return {
        "schema": "step7p_phase1_causal_subproblem_summary_v1",
        "decision": "promote_to_synthetic_causal_repacker"
        if phase2_gate_open
        else "stop_or_rework_causal_attribution",
        "atlas_path": str(out_path),
        "summary_path": str(summary_out_path),
        "markdown_path": str(markdown_out_path),
        "failures_by_cause_path": str(failures_out_path),
        "input_paths": {name: str(path) for name, path in input_paths.items()},
        "subproblem_count": len(rows),
        "represented_case_count": represented_case_count,
        "case_counts": dict(case_counts),
        "largest_case_id": largest_case_id,
        "largest_case_share": largest_case_share,
        "intent_family_counts": dict(intent_counts),
        "nonzero_intent_family_count": nonzero_intent_family_count,
        "failure_bucket_counts": dict(failure_counts),
        "unknown_failure_bucket_share": unknown_share,
        "metric_confidence_counts": dict(Counter(str(row["metric_confidence"]) for row in rows)),
        "forbidden_validation_label_term_count": forbidden_count,
        "soft_or_bbox_regression_attribution_count": failure_counts.get("soft_regression", 0)
        + failure_counts.get("bbox_regression", 0),
        "phase2_gate_open": phase2_gate_open,
        "gnn_rl_gate_open": False,
        "next_recommendation": "implement_phase2_synthetic_repacker"
        if phase2_gate_open
        else "rework_failure_attribution_before_repacker",
    }


def fail_closed_summary(
    lock: dict[str, Any],
    out_path: Path,
    summary_out_path: Path,
    markdown_out_path: Path,
    failures_out_path: Path,
    *,
    reason: str,
) -> dict[str, Any]:
    return {
        "schema": "step7p_phase1_causal_subproblem_summary_v1",
        "decision": "stop_or_rework_causal_attribution",
        "reason": reason,
        "stagnation_lock_decision": lock.get("decision"),
        "atlas_path": str(out_path),
        "summary_path": str(summary_out_path),
        "markdown_path": str(markdown_out_path),
        "failures_by_cause_path": str(failures_out_path),
        "subproblem_count": 0,
        "represented_case_count": 0,
        "phase2_gate_open": False,
        "gnn_rl_gate_open": False,
    }


def failures_by_cause(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_cause: dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        by_cause[str(row["seed_failure_bucket"])][str(row["case_id"])] += 1
    return {cause: dict(counter) for cause, counter in sorted(by_cause.items())}


def causal_markdown(summary: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# Step7P Phase1 Causal Subproblem Atlas",
            "",
            f"Decision: `{summary['decision']}`",
            "",
            f"- subproblem_count: {summary.get('subproblem_count', 0)}",
            f"- represented_case_count: {summary.get('represented_case_count', 0)}",
            f"- largest_case_share: {summary.get('largest_case_share')}",
            f"- nonzero_intent_family_count: {summary.get('nonzero_intent_family_count')}",
            f"- unknown_failure_bucket_share: {summary.get('unknown_failure_bucket_share')}",
            f"- phase2_gate_open: {summary.get('phase2_gate_open')}",
            f"- gnn_rl_gate_open: {summary.get('gnn_rl_gate_open')}",
            f"- next_recommendation: {summary.get('next_recommendation')}",
            "",
        ]
    )


def candidate_rows(payload: Any, key: str) -> list[dict[str, Any]]:
    if isinstance(payload, dict) and isinstance(payload.get(key), list):
        return [row for row in payload[key] if isinstance(row, dict)]
    if isinstance(payload, list):
        return [row for row in payload if isinstance(row, dict)]
    return []


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
            count += 1
    return count


def largest_case(case_counts: Counter[str]) -> tuple[str | None, int]:
    if not case_counts:
        return None, 0
    return max(case_counts.items(), key=lambda item: (item[1], item[0]))


def float_or_none(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
