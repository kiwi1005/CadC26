"""Step7N-ALR Phase 0 archive lineage mining.

This module is intentionally read-only over previous Step7 artifacts. It
normalizes replay/archive rows into one sidecar lineage table, reports source
coverage, and makes a fail-closed Phase0 decision before any reservoir/request
generator is allowed.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from puzzleplace.ml.floorset_training_corpus import write_json

EPS = 1e-9
DEFAULT_MEANINGFUL_COST_EPS = 1e-7
DEFAULT_NON_MICRO_THRESHOLDS = (1e-4, 1e-3, 1e-2)
STRICT_METRIC_CONFIDENCE = "exact_replay_comparable"
DECISIONS = {
    "stop_no_archive_signal",
    "diagnostic_only_due_concentration",
    "promote_to_reservoir_atlas",
}
FORBIDDEN_LABEL_TERMS = (
    "target_positions",
    "fp_sol",
    "tree_sol",
    "supervised_target",
    "label_target",
)
OBJECTIVE_FIELDS = (
    "hpwl_delta",
    "bbox_area_delta",
    "soft_constraint_delta",
    "official_like_cost_delta",
)
SELF_OUTPUT_PREFIX = "step7n_phase0_"


@dataclass(frozen=True, slots=True)
class SourceLoad:
    path: Path
    status: str
    reason: str
    extracted_row_count: int
    normalized_row_count: int


def mine_archive_lineage(
    source_manifest_path: Path,
    rows_out_path: Path,
    summary_path: Path,
    source_ledger_path: Path,
    by_case_path: Path,
    taxonomy_path: Path,
    markdown_path: Path,
    *,
    meaningful_cost_eps: float = DEFAULT_MEANINGFUL_COST_EPS,
    non_micro_thresholds: Iterable[float] = DEFAULT_NON_MICRO_THRESHOLDS,
) -> dict[str, Any]:
    """Normalize manifest-listed artifacts and write Phase0 lineage reports."""

    thresholds = tuple(float(value) for value in non_micro_thresholds)
    manifest_sources = load_source_manifest(source_manifest_path)
    rows: list[dict[str, Any]] = []
    source_loads: list[SourceLoad] = []
    for source_path in manifest_sources:
        source_rows, load = load_and_normalize_source(
            source_path,
            meaningful_cost_eps=meaningful_cost_eps,
            non_micro_thresholds=thresholds,
        )
        rows.extend(source_rows)
        source_loads.append(load)

    write_jsonl(rows_out_path, rows)
    source_ledger = source_ledger_payload(
        source_manifest_path, source_loads, rows_out_path=rows_out_path
    )
    write_json(source_ledger_path, source_ledger)
    by_case = summarize_by_case(rows)
    write_json(by_case_path, by_case)
    taxonomy = summarize_taxonomy(rows, thresholds)
    write_json(taxonomy_path, taxonomy)
    summary = summarize_lineage(
        rows,
        source_ledger,
        rows_out_path=rows_out_path,
        summary_path=summary_path,
        source_ledger_path=source_ledger_path,
        by_case_path=by_case_path,
        taxonomy_path=taxonomy_path,
        markdown_path=markdown_path,
        meaningful_cost_eps=meaningful_cost_eps,
        non_micro_thresholds=thresholds,
    )
    write_json(summary_path, summary)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.write_text(lineage_summary_markdown(summary), encoding="utf-8")
    return summary


def load_source_manifest(path: Path) -> list[Path]:
    sources: list[Path] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        item = line.strip()
        if not item or item.startswith("#"):
            continue
        sources.append(Path(item))
    return sources


def load_and_normalize_source(
    path: Path,
    *,
    meaningful_cost_eps: float,
    non_micro_thresholds: tuple[float, ...],
) -> tuple[list[dict[str, Any]], SourceLoad]:
    if is_self_output(path):
        return [], SourceLoad(
            path=path,
            status="skipped_schema",
            reason="phase0_self_output_denied",
            extracted_row_count=0,
            normalized_row_count=0,
        )
    if not path.exists():
        return [], SourceLoad(
            path=path,
            status="skipped_missing",
            reason="manifest_path_missing",
            extracted_row_count=0,
            normalized_row_count=0,
        )
    try:
        extracted = read_candidate_rows(path)
    except (json.JSONDecodeError, OSError) as exc:
        return [], SourceLoad(
            path=path,
            status="skipped_schema",
            reason=f"load_error:{type(exc).__name__}",
            extracted_row_count=0,
            normalized_row_count=0,
        )
    if not extracted:
        return [], SourceLoad(
            path=path,
            status="diagnostic_only",
            reason="no_candidate_rows_extracted",
            extracted_row_count=0,
            normalized_row_count=0,
        )

    rows = [
        normalize_row(
            row,
            path,
            row_index=index,
            meaningful_cost_eps=meaningful_cost_eps,
            non_micro_thresholds=non_micro_thresholds,
        )
        for index, row in enumerate(extracted)
    ]
    confidence_counts = Counter(str(row["metric_confidence"]) for row in rows)
    if confidence_counts.get(STRICT_METRIC_CONFIDENCE, 0) > 0:
        status = "included"
        reason = "contains_exact_replay_comparable_rows"
    elif confidence_counts:
        status = "diagnostic_only"
        reason = "no_exact_replay_comparable_rows"
    else:
        status = "skipped_metric_confidence"
        reason = "metric_confidence_unavailable"
    return rows, SourceLoad(
        path=path,
        status=status,
        reason=reason,
        extracted_row_count=len(extracted),
        normalized_row_count=len(rows),
    )


def read_candidate_rows(path: Path) -> list[dict[str, Any]]:
    if path.suffix == ".jsonl":
        return [
            with_source_list_key(json.loads(line), "jsonl")
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    payload = json.loads(path.read_text(encoding="utf-8"))
    return extract_candidate_rows(payload)


def extract_candidate_rows(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return [with_source_list_key(row, "root_list") for row in payload if is_candidate_row(row)]
    if not isinstance(payload, dict):
        return []
    if isinstance(payload.get("rows"), list):
        return [
            with_source_list_key(row, "rows")
            for row in payload["rows"]
            if is_candidate_row(row)
        ]
    rows: list[dict[str, Any]] = []
    for key, value in payload.items():
        if not isinstance(value, list):
            continue
        for item in value:
            if is_candidate_row(item):
                rows.append(with_source_list_key(item, key))
    return rows


def with_source_list_key(row: Any, key: str) -> dict[str, Any]:
    if not isinstance(row, dict):
        return {}
    copied = dict(row)
    copied["_source_list_key"] = key
    return copied


def is_candidate_row(value: Any) -> bool:
    if not isinstance(value, dict):
        return False
    candidate_markers = {
        "candidate_id",
        "request_id",
        "case_id",
        "objective_vector",
        "actual_objective_vector",
        "official_like_cost_delta",
    }
    return bool(candidate_markers.intersection(value))


def normalize_row(
    row: dict[str, Any],
    source_path: Path,
    *,
    row_index: int,
    meaningful_cost_eps: float,
    non_micro_thresholds: tuple[float, ...],
) -> dict[str, Any]:
    objective = objective_vector(row)
    confidence = metric_confidence(row, source_path, objective)
    case_id = normalize_case_id(row.get("case_id") or row.get("source_case"))
    candidate_id = str(
        row.get("candidate_id")
        or row.get("source_candidate_id")
        or row.get("request_id")
        or f"{source_path.name}:{row_index}"
    )
    taxonomy, move_scale = classify_move(row, candidate_id)
    threshold_flags = {
        threshold_key(threshold): is_non_micro(taxonomy, move_scale, threshold)
        for threshold in non_micro_thresholds
    }
    default_threshold = non_micro_thresholds[min(1, len(non_micro_thresholds) - 1)]
    non_micro = threshold_flags[threshold_key(default_threshold)]
    all_vector_nonregressing = vector_nonregressing(objective)
    hard_feasible = boolish(row.get("hard_feasible")) or boolish(row.get("hard_feasible_non_noop"))
    non_original_non_noop = (
        boolish(row.get("non_original_non_noop"))
        or boolish(row.get("hard_feasible_nonnoop"))
        or boolish(row.get("hard_feasible_non_noop"))
    ) and not boolish(row.get("no_op")) and not boolish(row.get("is_anchor"))
    meaningful_improving = objective["official_like_cost_delta"] < -meaningful_cost_eps
    strict_archive_candidate = (
        confidence == STRICT_METRIC_CONFIDENCE
        and hard_feasible
        and non_original_non_noop
        and all_vector_nonregressing
        and meaningful_improving
    )
    lineage_key = f"{case_id}:{candidate_id}"
    normalized = {
        "schema": "step7n_phase0_lineage_row_v1",
        "source_artifact": str(source_path),
        "source_list_key": row.get("_source_list_key"),
        "source_branch": row.get("source_branch") or row.get("source_family"),
        "source_phase": infer_source_phase(source_path),
        "source_candidate_id": row.get("source_candidate_id"),
        "case_id": case_id,
        "candidate_id": candidate_id,
        "lineage_key": lineage_key,
        "moved_block_count": int_or_none(
            row.get("moved_block_count")
            or row.get("changed_block_count")
            or row.get("macro_closure_size")
        ),
        "moved_block_ids": list_or_empty(
            row.get("moved_block_ids") or row.get("macro_closure_block_ids")
        ),
        "locality_class": row.get("actual_locality_class")
        or row.get("request_route_class")
        or row.get("route_class"),
        "move_taxonomy": taxonomy,
        "movement_scale": move_scale,
        "non_micro": non_micro,
        "non_micro_by_threshold": threshold_flags,
        "hard_feasible": hard_feasible,
        "non_original_non_noop": non_original_non_noop,
        "fresh_metric_available": boolish(row.get("fresh_metric_available"))
        or boolish(row.get("metrics_available"))
        or confidence.startswith("exact_replay"),
        "metric_confidence": confidence,
        "hpwl_delta": objective["hpwl_delta"],
        "bbox_area_delta": objective["bbox_area_delta"],
        "soft_constraint_delta": objective["soft_constraint_delta"],
        "official_like_cost_delta": objective["official_like_cost_delta"],
        "all_vector_nonregressing": all_vector_nonregressing,
        "meaningful_improving": meaningful_improving,
        "strict_archive_candidate": strict_archive_candidate,
        "dominated_by_original": boolish(row.get("dominated_by_original")),
        "failure_attribution": row.get("failure_attribution")
        or row.get("metric_regression_reason"),
        "quality_filter_reason": row.get("quality_filter_reason")
        or row.get("step7n_h_c_drop_reason"),
        "validation_label_policy": row.get("validation_label_policy"),
    }
    normalized["forbidden_label_term_count"] = forbidden_label_term_count(normalized)
    return normalized


def objective_vector(row: dict[str, Any]) -> dict[str, float]:
    nested = row.get("actual_objective_vector")
    if not isinstance(nested, dict):
        nested = row.get("objective_vector")
    vector = nested if isinstance(nested, dict) else {}
    return {
        "hpwl_delta": float_or_default(row.get("hpwl_delta", vector.get("hpwl_delta"))),
        "bbox_area_delta": float_or_default(
            row.get("bbox_area_delta", vector.get("bbox_area_delta"))
        ),
        "soft_constraint_delta": float_or_default(
            row.get("soft_constraint_delta", vector.get("soft_constraint_delta"))
        ),
        "official_like_cost_delta": float_or_default(
            row.get("official_like_cost_delta", vector.get("official_like_cost_delta"))
        ),
    }


def metric_confidence(
    row: dict[str, Any], source_path: Path, objective: dict[str, float]
) -> str:
    if "sweep" in source_path.name or row.get("sensitivity_vector") is not None:
        return "proxy_or_sweep_only"
    if not has_objective_components(row):
        return "summary_only" if likely_summary_source(source_path, row) else "unknown"
    if is_exact_replay_row(row, source_path) and all(
        value is not None for value in objective.values()
    ):
        return STRICT_METRIC_CONFIDENCE
    if boolish(row.get("fresh_metric_available")) or boolish(row.get("metrics_available")):
        return "exact_replay_partial"
    if row.get("objective_vector") is not None or any(field in row for field in OBJECTIVE_FIELDS):
        return "proxy_or_sweep_only"
    return "unknown"


def has_objective_components(row: dict[str, Any]) -> bool:
    nested = row.get("actual_objective_vector")
    if not isinstance(nested, dict):
        nested = row.get("objective_vector")
    vector = nested if isinstance(nested, dict) else {}
    return all(field in row or field in vector for field in OBJECTIVE_FIELDS)


def is_exact_replay_row(row: dict[str, Any], source_path: Path) -> bool:
    schema = str(row.get("schema", ""))
    if "replay_row" in schema:
        return True
    if source_path.name.endswith("_replay_rows.jsonl"):
        return True
    if row.get("artifact_trace_confidence") == "metric_ready_artifact_replay":
        return boolish(row.get("metrics_available")) or boolish(row.get("hard_feasible"))
    return False


def likely_summary_source(source_path: Path, row: dict[str, Any]) -> bool:
    if "summary" in source_path.name or "report" in source_path.name:
        return True
    return row.get("_source_list_key") not in {"rows", "jsonl", "root_list"}


def classify_move(row: dict[str, Any], candidate_id: str) -> tuple[str, float | None]:
    text = " ".join(
        str(value)
        for value in (
            candidate_id,
            row.get("source_family"),
            row.get("source_branch"),
            row.get("lane"),
            row.get("move_family"),
            row.get("assignment_type"),
            row.get("actual_locality_class"),
        )
        if value is not None
    ).lower()
    move_scale = movement_scale(row)
    if boolish(row.get("is_anchor")) or boolish(row.get("no_op")):
        return "anchor_noop", move_scale
    if "micro_axis" in text or "micro-axis" in text:
        return "micro_axis", move_scale
    if "global" in text:
        return "global_route", move_scale
    if "multiblock" in text or "block_shift" in text or "paired" in text:
        return "multi_block_shift", move_scale
    if "slot" in text or "macro" in text or "repack" in text:
        return "macro_slot_repack", move_scale
    if "window" in text:
        return "window_repack", move_scale
    moved_count = int_or_none(row.get("moved_block_count") or row.get("changed_block_count"))
    if moved_count == 1:
        return "single_block_region", move_scale
    if moved_count is not None and moved_count > 1:
        return "multi_block_shift", move_scale
    return "unknown", move_scale


def movement_scale(row: dict[str, Any]) -> float | None:
    candidates = [
        row.get("displacement"),
        row.get("observed_displacement"),
        row.get("target_distance"),
    ]
    costs = row.get("cost_components")
    if isinstance(costs, dict):
        candidates.append(costs.get("displacement"))
    for value in candidates:
        parsed = float_or_none(value)
        if parsed is not None:
            return abs(parsed)
    return None


def is_non_micro(taxonomy: str, move_scale: float | None, threshold: float) -> bool:
    if taxonomy in {"anchor_noop", "micro_axis"}:
        return False
    if move_scale is not None:
        return move_scale >= threshold
    return taxonomy in {
        "single_block_region",
        "multi_block_shift",
        "macro_slot_repack",
        "window_repack",
        "global_route",
    }


def summarize_lineage(
    rows: list[dict[str, Any]],
    source_ledger: dict[str, Any],
    *,
    rows_out_path: Path,
    summary_path: Path,
    source_ledger_path: Path,
    by_case_path: Path,
    taxonomy_path: Path,
    markdown_path: Path,
    meaningful_cost_eps: float,
    non_micro_thresholds: tuple[float, ...],
) -> dict[str, Any]:
    strict_rows = unique_rows(
        [
            row
            for row in rows
            if row["strict_archive_candidate"]
            and row["metric_confidence"] == STRICT_METRIC_CONFIDENCE
        ]
    )
    strict_non_micro = [row for row in strict_rows if row["non_micro"]]
    strict_cases = {str(row["case_id"]) for row in strict_rows if row.get("case_id") is not None}
    strict_non_micro_cases = {
        str(row["case_id"]) for row in strict_non_micro if row.get("case_id") is not None
    }
    case_counts = Counter(
        str(row["case_id"]) for row in strict_rows if row.get("case_id") is not None
    )
    largest_case_id, largest_case_count = largest_case(case_counts)
    exact_rows = [row for row in rows if row["metric_confidence"] == STRICT_METRIC_CONFIDENCE]
    decision = phase0_decision(strict_non_micro, strict_non_micro_cases, case_counts)
    threshold_sensitivity = {
        threshold_key(threshold): {
            "strict_meaningful_non_micro_winner_count": len(
                [
                    row
                    for row in strict_rows
                    if row["non_micro_by_threshold"].get(threshold_key(threshold), False)
                ]
            )
        }
        for threshold in non_micro_thresholds
    }
    return {
        "schema": "step7n_phase0_lineage_summary_v1",
        "decision": decision,
        "meaningful_cost_eps": meaningful_cost_eps,
        "non_micro_thresholds": list(non_micro_thresholds),
        "current_prior": "known Step7N-G/I winners are case024-centered until proven otherwise",
        "rows_path": str(rows_out_path),
        "summary_path": str(summary_path),
        "source_ledger_path": str(source_ledger_path),
        "by_case_path": str(by_case_path),
        "taxonomy_path": str(taxonomy_path),
        "markdown_path": str(markdown_path),
        "source_manifest_path": source_ledger["source_manifest_path"],
        "source_count": source_ledger["source_count"],
        "source_status_counts": source_ledger["status_counts"],
        "normalized_row_count": len(rows),
        "unique_lineage_key_count": len({row["lineage_key"] for row in rows}),
        "metric_confidence_counts": dict(Counter(str(row["metric_confidence"]) for row in rows)),
        "exact_replay_comparable_count": len(exact_rows),
        "fresh_metric_available_count": sum(
            int(bool(row["fresh_metric_available"])) for row in rows
        ),
        "strict_archive_candidate_count": len(strict_rows),
        "strict_meaningful_non_micro_winner_count": len(strict_non_micro),
        "strict_winner_case_count": len(strict_cases),
        "strict_non_micro_winner_case_count": len(strict_non_micro_cases),
        "non_case025_strict_winner_count": sum(
            int(str(row.get("case_id")) != "25") for row in strict_rows
        ),
        "case024_share": case_share(case_counts, "24"),
        "case025_share": case_share(case_counts, "25"),
        "largest_case_id": largest_case_id,
        "largest_case_share": largest_case_count / max(sum(case_counts.values()), 1),
        "threshold_sensitivity": threshold_sensitivity,
        "move_taxonomy_counts": dict(Counter(str(row["move_taxonomy"]) for row in rows)),
        "forbidden_validation_label_term_count": sum(
            int(row["forbidden_label_term_count"]) for row in rows
        ),
        "phase1_gate_open": decision == "promote_to_reservoir_atlas",
        "gnn_rl_gate_open": False,
        "next_recommendation": next_recommendation(decision),
    }


def summarize_by_case(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_case: dict[str, dict[str, Any]] = {}
    grouped: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get("case_id"))].append(row)
    for case_id, case_rows in sorted(grouped.items()):
        strict = [row for row in case_rows if row["strict_archive_candidate"]]
        by_case[case_id] = {
            "row_count": len(case_rows),
            "exact_replay_comparable_count": sum(
                int(row["metric_confidence"] == STRICT_METRIC_CONFIDENCE) for row in case_rows
            ),
            "strict_archive_candidate_count": len(unique_rows(strict)),
            "strict_meaningful_non_micro_winner_count": len(
                unique_rows([row for row in strict if row["non_micro"]])
            ),
            "move_taxonomy_counts": dict(Counter(str(row["move_taxonomy"]) for row in case_rows)),
            "metric_confidence_counts": dict(
                Counter(str(row["metric_confidence"]) for row in case_rows)
            ),
        }
    return {"schema": "step7n_phase0_lineage_by_case_v1", "cases": by_case}


def summarize_taxonomy(rows: list[dict[str, Any]], thresholds: tuple[float, ...]) -> dict[str, Any]:
    taxonomy_rows = {}
    for taxonomy, count in Counter(str(row["move_taxonomy"]) for row in rows).items():
        family_rows = [row for row in rows if row["move_taxonomy"] == taxonomy]
        taxonomy_rows[taxonomy] = {
            "row_count": count,
            "strict_archive_candidate_count": len(
                unique_rows([row for row in family_rows if row["strict_archive_candidate"]])
            ),
            "numeric_scale_available_count": sum(
                int(row.get("movement_scale") is not None) for row in family_rows
            ),
        }
    threshold_counts = {
        threshold_key(threshold): sum(
            int(bool(row["non_micro_by_threshold"].get(threshold_key(threshold), False)))
            for row in rows
        )
        for threshold in thresholds
    }
    return {
        "schema": "step7n_phase0_move_taxonomy_v1",
        "thresholds": list(thresholds),
        "taxonomy": taxonomy_rows,
        "non_micro_counts_by_threshold": threshold_counts,
    }


def source_ledger_payload(
    source_manifest_path: Path, loads: list[SourceLoad], *, rows_out_path: Path
) -> dict[str, Any]:
    sources = [
        {
            "path": str(load.path),
            "status": load.status,
            "reason": load.reason,
            "extracted_row_count": load.extracted_row_count,
            "normalized_row_count": load.normalized_row_count,
        }
        for load in loads
    ]
    return {
        "schema": "step7n_phase0_source_ledger_v1",
        "source_manifest_path": str(source_manifest_path),
        "rows_out_path": str(rows_out_path),
        "source_count": len(loads),
        "status_counts": dict(Counter(load.status for load in loads)),
        "sources": sources,
    }


def phase0_decision(
    strict_non_micro: list[dict[str, Any]],
    strict_non_micro_cases: set[str],
    case_counts: Counter[str],
) -> str:
    if not strict_non_micro:
        return "stop_no_archive_signal"
    largest_share = largest_case(case_counts)[1] / max(sum(case_counts.values()), 1)
    noncase25 = sum(int(str(row.get("case_id")) != "25") for row in strict_non_micro)
    if (
        len(strict_non_micro) >= 3
        and len(strict_non_micro_cases) >= 2
        and noncase25 >= 1
        and largest_share <= 0.70
    ):
        return "promote_to_reservoir_atlas"
    return "diagnostic_only_due_concentration"


def lineage_summary_markdown(summary: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# Step7N-ALR Phase0 Archive Lineage Summary",
            "",
            f"Decision: `{summary['decision']}`",
            "",
            f"- normalized_row_count: {summary['normalized_row_count']}",
            f"- exact_replay_comparable_count: {summary['exact_replay_comparable_count']}",
            f"- strict_archive_candidate_count: {summary['strict_archive_candidate_count']}",
            "- strict_meaningful_non_micro_winner_count: "
            f"{summary['strict_meaningful_non_micro_winner_count']}",
            f"- strict_winner_case_count: {summary['strict_winner_case_count']}",
            f"- case024_share: {summary['case024_share']}",
            f"- case025_share: {summary['case025_share']}",
            f"- largest_case_id: {summary['largest_case_id']}",
            f"- largest_case_share: {summary['largest_case_share']}",
            f"- forbidden_validation_label_term_count: "
            f"{summary['forbidden_validation_label_term_count']}",
            f"- phase1_gate_open: {summary['phase1_gate_open']}",
            f"- gnn_rl_gate_open: {summary['gnn_rl_gate_open']}",
            f"- next_recommendation: {summary['next_recommendation']}",
            "",
            "## Metric confidence counts",
            "",
            "```json",
            json.dumps(summary["metric_confidence_counts"], indent=2, sort_keys=True),
            "```",
            "",
            "## Source status counts",
            "",
            "```json",
            json.dumps(summary["source_status_counts"], indent=2, sort_keys=True),
            "```",
            "",
        ]
    )


def next_recommendation(decision: str) -> str:
    if decision == "promote_to_reservoir_atlas":
        return "implement_phase1_reservoir_atlas_only_after_verifier_review"
    if decision == "diagnostic_only_due_concentration":
        return "report_archive_signal_concentration_before_any_generator_work"
    return "stop_step7n_alr_without_reservoir_generator"


def unique_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    unique: list[dict[str, Any]] = []
    for row in rows:
        key = str(row.get("lineage_key"))
        if key in seen:
            continue
        seen.add(key)
        unique.append(row)
    return unique


def vector_nonregressing(objective: dict[str, float]) -> bool:
    return (
        objective["hpwl_delta"] <= EPS
        and objective["bbox_area_delta"] <= EPS
        and objective["soft_constraint_delta"] <= EPS
    )


def case_share(case_counts: Counter[str], case_id: str) -> float:
    return case_counts.get(case_id, 0) / max(sum(case_counts.values()), 1)


def largest_case(case_counts: Counter[str]) -> tuple[str | None, int]:
    if not case_counts:
        return None, 0
    return max(case_counts.items(), key=lambda item: (item[1], item[0]))


def infer_source_phase(path: Path) -> str:
    name = path.name
    for token in ("step7l", "step7m", "step7n_g", "step7n_h", "step7n_i"):
        if token in name:
            return token
    return "unknown"


def forbidden_label_term_count(row: dict[str, Any]) -> int:
    text = json.dumps(row, sort_keys=True).lower()
    return sum(text.count(term) for term in FORBIDDEN_LABEL_TERMS)


def is_self_output(path: Path) -> bool:
    return path.name.startswith(SELF_OUTPUT_PREFIX)


def threshold_key(value: float) -> str:
    return f"{value:.0e}"


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
            count += 1
    return count


def float_or_default(value: Any, default: float = 0.0) -> float:
    parsed = float_or_none(value)
    return default if parsed is None else parsed


def float_or_none(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def int_or_none(value: Any) -> int | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def boolish(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    if isinstance(value, str):
        return value.lower() in {"1", "true", "yes", "y"}
    return False


def normalize_case_id(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if text.lower().startswith("case"):
        text = text[4:]
    try:
        return str(int(text))
    except ValueError:
        return text


def list_or_empty(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []
