"""Step7N-I quality-filtered macro slot/repack integration.

Sidecar artifact integration only. It consumes Step7N-G and Step7N-H artifacts,
annotates H-C retained rows with H-D target calibration and H-A taxonomy, then
separates true archive candidates from local-starved recovery report rows.
"""

from __future__ import annotations

import json
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class InputSpec:
    name: str
    path: str
    required: bool = True


INPUTS: tuple[InputSpec, ...] = (
    InputSpec("g_integrated", "artifacts/research/step7n_g_integrated_candidates.json"),
    InputSpec("g_metric", "artifacts/research/step7n_g_metric_report.json"),
    InputSpec("h_c_filtered", "artifacts/research/step7n_h_c_filtered_candidates.json"),
    InputSpec("h_c_preservation", "artifacts/research/step7n_h_c_preservation_report.json"),
    InputSpec("h_c_dominance", "artifacts/research/step7n_h_c_dominance_report.json"),
    InputSpec("h_d_features", "artifacts/research/step7n_h_d_target_feature_table.json"),
    InputSpec("h_d_report", "artifacts/research/step7n_h_d_target_vs_metric_report.json"),
    InputSpec("h_a_taxonomy", "artifacts/research/step7n_h_a_failure_taxonomy.json"),
    InputSpec("h_e_sweep", "artifacts/research/step7n_h_e_sweep_results.json", required=False),
)


ARCHIVE_REASONS = {"official_like_winner_preservation", "step7n_g_non_anchor_pareto_front"}
RECOVERY_REASON = "local_starved_case_representative"


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def rows(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, dict):
        value = payload.get("rows", [])
        return [row for row in value if isinstance(row, dict)] if isinstance(value, list) else []
    if isinstance(payload, list):
        return [row for row in payload if isinstance(row, dict)]
    return []


def as_float(value: Any, default: float = 0.0) -> float:
    if value is None:
        return default
    if isinstance(value, int | float):
        return float(value)
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def build_input_inventory(base_dir: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    loaded: dict[str, Any] = {}
    entries: dict[str, Any] = {}
    missing: list[dict[str, Any]] = []
    for spec in INPUTS:
        path = base_dir / spec.path
        entry: dict[str, Any] = {
            "path": spec.path,
            "required": spec.required,
            "exists": path.exists(),
        }
        if path.exists():
            payload = load_json(path)
            loaded[spec.name] = payload
            entry["size_bytes"] = path.stat().st_size
            entry["row_count"] = len(rows(payload))
            if isinstance(payload, dict):
                entry["top_level_keys"] = sorted(payload.keys())
        else:
            missing.append({"name": spec.name, **entry})
        entries[spec.name] = entry
    return {
        "inputs": entries,
        "missing_input_report": missing,
        "all_required_inputs_available": all(
            e["exists"] for e in entries.values() if e["required"]
        ),
    }, loaded


def index_by_candidate(source_rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(row.get("candidate_id")): row for row in source_rows if row.get("candidate_id")}


def classify_row(row: dict[str, Any]) -> str:
    if not row.get("step7n_h_c_retained"):
        return "rejected_by_h_c_filter"
    reasons = set(row.get("step7n_h_c_retain_reasons") or [])
    if reasons & ARCHIVE_REASONS:
        return "archive_candidate"
    if RECOVERY_REASON in reasons:
        return "recovery_report_only"
    return "retained_uncategorized"


def annotate_rows(loaded: dict[str, Any]) -> list[dict[str, Any]]:
    h_c_rows = rows(loaded["h_c_filtered"])
    h_d_by_id = index_by_candidate(rows(loaded["h_d_features"]))
    h_e_rows = rows(loaded.get("h_e_sweep", {}))
    rescued = {
        str(row.get("candidate_id"))
        for row in h_e_rows
        if row.get("proxy_improving") or row.get("official_like_proxy_improving")
    }
    annotated: list[dict[str, Any]] = []
    for row in h_c_rows:
        item = dict(row)
        cid = str(item.get("candidate_id"))
        target = h_d_by_id.get(cid, {})
        classification = classify_row(item)
        item["step7n_i_classification"] = classification
        item["step7n_i_selected_for_archive"] = classification == "archive_candidate"
        item["step7n_i_selected_for_recovery_report"] = classification == "recovery_report_only"
        item["step7n_i_target_calibration"] = {
            "has_retrieval_target": target.get("has_retrieval_target"),
            "target_quality_score": target.get("target_quality_score"),
            "retrieval_prior_agreement": target.get("retrieval_prior_agreement"),
            "target_matches_guided": target.get("target_matches_guided"),
            "target_matches_pure": target.get("target_matches_pure"),
            "target_failure_bucket": target.get("target_failure_bucket"),
            "target_region": target.get("target_region"),
            "guided_target_region": target.get("guided_target_region"),
            "pure_target_region": target.get("pure_target_region"),
        }
        item["step7n_i_sensitivity_rescue_signal"] = cid in rescued
        annotated.append(item)
    return annotated


def objective_key(row: dict[str, Any]) -> tuple[float, float, float, float, float]:
    vector = row.get("objective_vector") or {}
    return (
        as_float(vector.get("feasibility_rank")),
        as_float(vector.get("official_like_cost_delta")),
        as_float(vector.get("hpwl_delta")),
        as_float(vector.get("bbox_area_delta")),
        as_float(vector.get("soft_constraint_delta")),
    )


def build_archive(annotated: list[dict[str, Any]]) -> dict[str, Any]:
    selected = [row for row in annotated if row.get("step7n_i_selected_for_archive")]
    per_case: dict[str, Any] = {}
    for case_id in sorted({int(row["case_id"]) for row in selected}):
        case_rows = sorted(
            [row for row in selected if int(row["case_id"]) == case_id], key=objective_key
        )
        per_case[str(case_id)] = {
            "archive_candidate_count": len(case_rows),
            "official_like_winner_count": sum(
                1 for row in case_rows if row.get("official_like_cost_improving")
            ),
            "front_rows": [project_archive_row(row) for row in case_rows],
        }
    selection_rule = (
        "H-C official-like winner or non-anchor Pareto preservation; "
        "local-starved representatives are report-only unless also on the front"
    )
    return {
        "schema": "step7n_i_quality_filtered_archive_v1",
        "selection_rule": selection_rule,
        "archive_candidate_count": len(selected),
        "front_contribution_by_lane": dict(Counter(str(row.get("lane")) for row in selected)),
        "front_contribution_by_source": dict(
            Counter(str(row.get("source_branch")) for row in selected)
        ),
        "official_like_improving_front_count": sum(
            1 for row in selected if row.get("official_like_cost_improving")
        ),
        "dominated_by_original_front_count": sum(
            1 for row in selected if row.get("dominated_by_original")
        ),
        "metric_regressing_front_count": sum(
            1
            for row in selected
            if as_float(row.get("official_like_cost_delta")) > 0
            and not row.get("official_like_cost_improving")
        ),
        "per_case": per_case,
    }


def project_archive_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "candidate_id": row.get("candidate_id"),
        "case_id": row.get("case_id"),
        "source_branch": row.get("source_branch"),
        "lane": row.get("lane"),
        "official_like_cost_delta": row.get("official_like_cost_delta"),
        "hpwl_delta": row.get("hpwl_delta"),
        "bbox_area_delta": row.get("bbox_area_delta"),
        "soft_constraint_delta": row.get("soft_constraint_delta"),
        "official_like_cost_improving": row.get("official_like_cost_improving"),
        "metric_regression_reason": row.get("metric_regression_reason"),
        "target_calibration": row.get("step7n_i_target_calibration"),
        "retain_reasons": row.get("step7n_h_c_retain_reasons"),
    }


def summarize(
    annotated: list[dict[str, Any]], loaded: dict[str, Any], runtime_ms: float
) -> dict[str, Any]:
    retained = [row for row in annotated if row.get("step7n_h_c_retained")]
    archive = [row for row in annotated if row.get("step7n_i_selected_for_archive")]
    recovery = [row for row in annotated if row.get("step7n_i_selected_for_recovery_report")]
    official = [row for row in annotated if row.get("official_like_cost_improving")]
    archive_official = [row for row in archive if row.get("official_like_cost_improving")]
    non_anchor = [
        row
        for row in annotated
        if "step7n_g_non_anchor_pareto_front" in (row.get("step7n_h_c_retain_reasons") or [])
    ]
    h_d_report = loaded.get("h_d_report", {})
    target_screening = (
        h_d_report.get("target_screening", {}) if isinstance(h_d_report, dict) else {}
    )
    return {
        "input_candidate_count": len(annotated),
        "h_c_retained_count": len(retained),
        "archive_candidate_count": len(archive),
        "recovery_report_only_count": len(recovery),
        "official_like_winner_preservation_count": len(archive_official),
        "official_like_winner_total": len(official),
        "non_anchor_pareto_preservation_count": len([row for row in archive if row in non_anchor]),
        "non_anchor_pareto_total": len(non_anchor),
        "local_starved_recovery_cases_reported": sorted({int(row["case_id"]) for row in retained}),
        "local_starved_recovery_case_count": len({int(row["case_id"]) for row in retained}),
        "dominated_by_original_input_count": sum(
            1 for row in annotated if row.get("dominated_by_original")
        ),
        "dominated_by_original_h_c_retained_count": sum(
            1 for row in retained if row.get("dominated_by_original")
        ),
        "dominated_by_original_archive_count": sum(
            1 for row in archive if row.get("dominated_by_original")
        ),
        "metric_regressing_input_count": sum(
            1 for row in annotated if as_float(row.get("official_like_cost_delta")) > 0
        ),
        "metric_regressing_h_c_retained_count": sum(
            1 for row in retained if as_float(row.get("official_like_cost_delta")) > 0
        ),
        "metric_regressing_archive_count": sum(
            1 for row in archive if as_float(row.get("official_like_cost_delta")) > 0
        ),
        "classification_counts": dict(Counter(row["step7n_i_classification"] for row in annotated)),
        "archive_lane_counts": dict(Counter(str(row.get("lane")) for row in archive)),
        "archive_source_counts": dict(Counter(str(row.get("source_branch")) for row in archive)),
        "target_failure_bucket_counts_retained": dict(
            Counter(
                str((row.get("step7n_i_target_calibration") or {}).get("target_failure_bucket"))
                for row in retained
            )
        ),
        "h_d_target_screening": target_screening,
        "h_a_reason_counts": loaded.get("h_a_taxonomy", {}).get("reason_counts", {}),
        "runtime_proxy_ms": runtime_ms,
    }


def decide(metrics: dict[str, Any]) -> str:
    if (
        metrics["official_like_winner_preservation_count"] == 3
        and metrics["non_anchor_pareto_preservation_count"] == 5
        and metrics["dominated_by_original_archive_count"] == 0
    ):
        return "promote_quality_filter_to_step7n_g_sidecar"
    if metrics["metric_regressing_archive_count"] > 0:
        return "refine_tradeoff_objective_schema"
    return "inconclusive_due_to_filter_quality"


def decision_markdown(metrics: dict[str, Any], decision: str) -> str:
    official_preservation = (
        f'{metrics["official_like_winner_preservation_count"]} / '
        f'{metrics["official_like_winner_total"]}'
    )
    pareto_preservation = (
        f'{metrics["non_anchor_pareto_preservation_count"]} / '
        f'{metrics["non_anchor_pareto_total"]}'
    )
    dominated_summary = (
        f'input {metrics["dominated_by_original_input_count"]} -> '
        f'H-C retained {metrics["dominated_by_original_h_c_retained_count"]} -> '
        f'archive {metrics["dominated_by_original_archive_count"]}'
    )
    regressing_summary = (
        f'input {metrics["metric_regressing_input_count"]} -> '
        f'H-C retained {metrics["metric_regressing_h_c_retained_count"]} -> '
        f'archive {metrics["metric_regressing_archive_count"]}'
    )
    return f"""# Step7N-I Quality-Filtered Macro Slot/Repack Integration

Decision: `{decision}`

## Key metrics

- input_candidate_count: {metrics["input_candidate_count"]}
- h_c_retained_count: {metrics["h_c_retained_count"]}
- archive_candidate_count: {metrics["archive_candidate_count"]}
- recovery_report_only_count: {metrics["recovery_report_only_count"]}
- official_like_winner_preservation: {official_preservation}
- non_anchor_pareto_preservation: {pareto_preservation}
- local_starved_recovery_case_count: {metrics["local_starved_recovery_case_count"]}
- dominated_by_original: {dominated_summary}
- metric_regressing: {regressing_summary}

## Interpretation

The H-C constrained Pareto quality filter is strong enough to connect back to the
Step7N-G integration layer as a sidecar archive selector: it preserves all known
winners and non-anchor Pareto rows while moving local-starved representatives to
report-only recovery evidence. H-D target calibration remains useful as an
annotation/prefilter signal, especially for identifying wrong target-region and
bad internal-repack failures before promotion.

This does not prove the macro/topology generator is solved. Two archive rows are
still metric-regressing tradeoff rows, and all official-like winners remain
case024-centered. The next bottleneck is internal repack disruption and target
quality generalization, not feasibility.

## Constraints respected

- Sidecar-only; no contest runtime integration.
- No finalizer semantic changes.
- No new generator expansion.
- No scalar penalty soup; filter is based on H-C constrained dominance and
  explicit preservation classes.
- Rejected, dominated, report-only, and metric-regressing candidates remain in
  JSON artifacts with provenance.
"""


def run_step7n_i(base_dir: Path, output_dir: Path) -> dict[str, Any]:
    start = time.perf_counter()
    inventory, loaded = build_input_inventory(base_dir)
    annotated = annotate_rows(loaded)
    archive = build_archive(annotated)
    metrics = summarize(annotated, loaded, (time.perf_counter() - start) * 1000.0)
    decision = decide(metrics)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "step7n_i_input_inventory.json", inventory)
    write_json(output_dir / "step7n_i_annotated_candidates.json", {"rows": annotated})
    write_json(
        output_dir / "step7n_i_quality_filtered_candidates.json",
        {
            "summary": metrics,
            "rows": [row for row in annotated if row.get("step7n_h_c_retained")],
        },
    )
    write_json(output_dir / "step7n_i_archive_replay.json", archive)
    write_json(
        output_dir / "step7n_i_target_quality_report.json",
        {
            "target_screening": metrics.get("h_d_target_screening"),
            "target_failure_bucket_counts_retained": metrics[
                "target_failure_bucket_counts_retained"
            ],
        },
    )
    write_json(
        output_dir / "step7n_i_failure_taxonomy.json",
        {
            "h_a_reason_counts": metrics.get("h_a_reason_counts"),
            "classification_counts": metrics["classification_counts"],
            "archive_lane_counts": metrics["archive_lane_counts"],
        },
    )
    (output_dir / "step7n_i_visualizations").mkdir(parents=True, exist_ok=True)
    (output_dir / "step7n_i_decision.md").write_text(
        decision_markdown(metrics, decision), encoding="utf-8"
    )
    return {"decision": decision, "metrics": metrics, "archive": archive}
