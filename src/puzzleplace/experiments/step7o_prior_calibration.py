"""Step7O Phase2 prior calibration over existing candidates.

This phase annotates Step7ML-I/K decoded candidates and Step7N-I report rows
with Step7O prior agreement. It does not emit requests, run replay, or train a
model.
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Iterable
from pathlib import Path
from typing import Any

EPS = 1e-9
TOP_BUDGET_SIZE = 6
MAX_BUDGET_ROWS_PER_CASE = 2
STEP7ML_WINNER_BASELINE = 2
STEP7N_ARCHIVE_WINNER_BASELINE = 3
STEP7N_NON_ANCHOR_PARETO_BASELINE = 5


def run_prior_calibration(
    atlas_path: Path,
    step7ml_i_candidates_path: Path,
    step7ml_k_candidates_path: Path,
    step7n_i_candidates_path: Path,
    rows_out_path: Path,
    summary_out_path: Path,
    markdown_out_path: Path,
    *,
    step7ml_j_quality_path: Path | None = None,
    step7ml_k_quality_path: Path | None = None,
) -> dict[str, Any]:
    """Write Phase2 calibration rows and summary."""

    atlas_rows = read_jsonl(atlas_path)
    atlas_prior = build_atlas_prior(atlas_rows)
    step7ml_rows = normalize_step7ml_candidates(
        step7ml_i_candidates_path, step7ml_k_candidates_path, atlas_prior
    )
    step7n_rows = normalize_step7n_archive_rows(step7n_i_candidates_path, atlas_prior)
    rows = step7ml_rows + step7n_rows
    top_budget = select_top_budget(step7ml_rows)
    write_jsonl(rows_out_path, rows)

    summary = summarize_calibration(
        rows,
        top_budget,
        atlas_path=atlas_path,
        rows_out_path=rows_out_path,
        summary_out_path=summary_out_path,
        markdown_out_path=markdown_out_path,
        step7ml_j_quality_path=step7ml_j_quality_path,
        step7ml_k_quality_path=step7ml_k_quality_path,
    )
    write_json(summary_out_path, summary)
    markdown_out_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_out_path.write_text(calibration_markdown(summary), encoding="utf-8")
    return summary


def build_atlas_prior(atlas_rows: list[dict[str, Any]]) -> dict[str, Any]:
    route_counts: dict[str, Counter[str]] = {}
    region_scores: dict[str, dict[str, float]] = {}
    closure_buckets: dict[str, set[str]] = {}
    for row in atlas_rows:
        case_id = str(row.get("case_id"))
        route_counts.setdefault(case_id, Counter())[str(row.get("route_locality_proxy"))] += 1
        bucket = row.get("closure_size_bucket")
        if bucket is not None:
            closure_buckets.setdefault(case_id, set()).add(str(bucket))
        region_id = row.get("region_id")
        if region_id is not None:
            region_scores.setdefault(case_id, {})[normalize_region(str(region_id))] = float_value(
                row.get("mean_region_prior")
            )
    return {
        "route_scores": {
            case_id: {
                route: count / max(max(counter.values()), 1)
                for route, count in counter.items()
            }
            for case_id, counter in route_counts.items()
        },
        "region_scores": normalize_case_region_scores(region_scores),
        "closure_buckets": closure_buckets,
    }


def normalize_step7ml_candidates(
    step7ml_i_candidates_path: Path,
    step7ml_k_candidates_path: Path,
    atlas_prior: dict[str, Any],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in candidate_rows(load_json(step7ml_i_candidates_path), key="rows"):
        rows.append(normalize_candidate(row, "step7ml_i", step7ml_i_candidates_path, atlas_prior))
    k_payload = load_json(step7ml_k_candidates_path)
    for row in candidate_rows(k_payload, key="selected_rows"):
        rows.append(normalize_candidate(row, "step7ml_k", step7ml_k_candidates_path, atlas_prior))
    return rows


def normalize_step7n_archive_rows(
    step7n_i_candidates_path: Path, atlas_prior: dict[str, Any]
) -> list[dict[str, Any]]:
    rows = []
    for row in candidate_rows(load_json(step7n_i_candidates_path), key="rows"):
        normalized = normalize_candidate(
            row, "step7n_i_archive", step7n_i_candidates_path, atlas_prior
        )
        normalized["metric_confidence"] = "archive_preservation_only"
        normalized["report_only_archive_row"] = True
        normalized["step7n_archive_candidate"] = bool(
            row.get("step7n_i_selected_for_archive")
            or row.get("step7n_i_classification") == "archive_candidate"
        )
        reasons = row.get("step7n_h_c_retain_reasons")
        normalized["step7n_official_like_winner_preservation"] = (
            isinstance(reasons, list) and "official_like_winner_preservation" in reasons
        )
        normalized["step7n_non_anchor_pareto_preservation"] = (
            isinstance(reasons, list) and "step7n_g_non_anchor_pareto_front" in reasons
        )
        rows.append(normalized)
    return rows


def normalize_candidate(
    row: dict[str, Any],
    source_model: str,
    source_path: Path,
    atlas_prior: dict[str, Any],
) -> dict[str, Any]:
    objective = objective_vector(row)
    confidence = metric_confidence(row, source_model, objective)
    case_id = str(row.get("case_id"))
    prior = prior_agreement(row, atlas_prior)
    all_vector_nonregressing = (
        confidence == "exact_component_comparable"
        and component_le(objective, "hpwl_delta", EPS)
        and component_le(objective, "bbox_area_delta", EPS)
        and component_le(objective, "soft_constraint_delta", EPS)
    )
    official_like_improving = bool_value(row.get("official_like_improving")) or component_lt(
        objective, "official_like_cost_delta", -EPS
    )
    official_like_regression = (
        confidence == "exact_component_comparable"
        and component_gt(objective, "official_like_cost_delta", EPS)
    )
    return {
        "schema": "step7o_phase2_prior_calibration_row_v1",
        "source_model": source_model,
        "source_artifact": str(source_path),
        "candidate_id": str(row.get("candidate_id")),
        "source_candidate_id": row.get("source_candidate_id"),
        "case_id": case_id,
        "decoder": row.get("decoder"),
        "target_region": row.get("target_region"),
        "route_class": row.get("route_class") or row.get("actual_locality_class"),
        "closure_size_bucket": row.get("closure_size_bucket")
        or block_count_bucket(int_value(row.get("block_count") or row.get("macro_closure_size"))),
        "metric_confidence": confidence,
        "prior_agreement_score": prior["score"],
        "prior_region_score": prior["region_score"],
        "prior_route_score": prior["route_score"],
        "prior_closure_score": prior["closure_score"],
        "hard_feasible_non_noop": bool_value(
            row.get("hard_feasible_non_noop") or row.get("hard_feasible_nonnoop")
        ),
        "non_original_non_noop": bool_value(row.get("non_original_non_noop")),
        "quality_gate_pass": bool_value(row.get("quality_gate_pass")),
        "official_like_improving": official_like_improving,
        "metric_regressing": bool_value(row.get("metric_regressing")) or official_like_regression,
        "dominated_by_original": bool_value(row.get("dominated_by_original")),
        "hpwl_delta": objective["hpwl_delta"],
        "bbox_area_delta": objective["bbox_area_delta"],
        "soft_constraint_delta": objective["soft_constraint_delta"],
        "official_like_cost_delta": objective["official_like_cost_delta"],
        "all_vector_nonregressing": all_vector_nonregressing,
        "bbox_regression": confidence == "exact_component_comparable"
        and component_gt(objective, "bbox_area_delta", EPS),
        "soft_regression": confidence == "exact_component_comparable"
        and component_gt(objective, "soft_constraint_delta", EPS),
        "official_like_regression": official_like_regression,
        "hpwl_gain_but_official_like_loss": confidence == "exact_component_comparable"
        and component_lt(objective, "hpwl_delta", -EPS)
        and official_like_regression,
        "selected_in_top_budget": False,
    }


def select_top_budget(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    comparable = [
        row
        for row in rows
        if row["metric_confidence"] == "exact_component_comparable"
        and row["source_model"] in {"step7ml_i", "step7ml_k"}
    ]
    ordered = sorted(
        comparable,
        key=lambda row: (
            -float_value(row["prior_agreement_score"]),
            -int(bool(row["all_vector_nonregressing"])),
            -int(bool(row["hard_feasible_non_noop"])),
            float_value(row["official_like_cost_delta"]),
            str(row["candidate_id"]),
        ),
    )
    selected: list[dict[str, Any]] = []
    by_case: Counter[str] = Counter()
    for row in ordered:
        if len(selected) >= TOP_BUDGET_SIZE:
            break
        if by_case[str(row["case_id"])] >= MAX_BUDGET_ROWS_PER_CASE:
            continue
        selected.append(row)
        by_case[str(row["case_id"])] += 1
    if len(selected) < TOP_BUDGET_SIZE:
        seen = {row["candidate_id"] for row in selected}
        for row in ordered:
            if len(selected) >= TOP_BUDGET_SIZE:
                break
            if row["candidate_id"] in seen:
                continue
            selected.append(row)
            seen.add(row["candidate_id"])
    for row in selected:
        row["selected_in_top_budget"] = True
    return selected


def summarize_calibration(
    rows: list[dict[str, Any]],
    top_budget: list[dict[str, Any]],
    *,
    atlas_path: Path,
    rows_out_path: Path,
    summary_out_path: Path,
    markdown_out_path: Path,
    step7ml_j_quality_path: Path | None,
    step7ml_k_quality_path: Path | None,
) -> dict[str, Any]:
    exact = [row for row in rows if row["metric_confidence"] == "exact_component_comparable"]
    step7ml_rows = [row for row in rows if row["source_model"] in {"step7ml_i", "step7ml_k"}]
    step7ml_winner_signatures = {
        winner_signature(row)
        for row in step7ml_rows
        if bool(row.get("official_like_improving")) or bool(row.get("quality_gate_pass"))
    }
    top_signatures = {winner_signature(row) for row in top_budget}
    top_winner_rows = [row for row in top_budget if row["official_like_improving"]]
    top_case_counts = Counter(str(row["case_id"]) for row in top_budget)
    step7n_rows = [row for row in rows if row["source_model"] == "step7n_i_archive"]
    step7n_winner_preserved = sum(
        int(bool(row.get("step7n_official_like_winner_preservation"))) for row in step7n_rows
    )
    step7n_pareto_preserved = sum(
        int(bool(row.get("step7n_non_anchor_pareto_preservation"))) for row in step7n_rows
    )
    k_baseline = baseline_summary(step7ml_k_quality_path)
    j_baseline = baseline_summary(step7ml_j_quality_path)
    top_bbox_regressions = sum(int(bool(row["bbox_regression"])) for row in top_budget)
    top_soft_regressions = sum(int(bool(row["soft_regression"])) for row in top_budget)
    winner_preservation_pass = STEP7ML_WINNER_BASELINE <= len(
        step7ml_winner_signatures.intersection(top_signatures)
    )
    archive_preservation_pass = (
        step7n_winner_preserved >= STEP7N_ARCHIVE_WINNER_BASELINE
        and step7n_pareto_preserved >= STEP7N_NON_ANCHOR_PARETO_BASELINE
    )
    regression_reduction_pass = (
        top_bbox_regressions <= int(0.7 * int_value(k_baseline.get("bbox_regression_count")))
        or top_soft_regressions <= int(0.7 * int_value(k_baseline.get("soft_regression_count")))
    )
    concentration = concentration_summary(top_budget, top_winner_rows)
    concentration_pass = not concentration["top_budget_winners_only_case024_or_case025"]
    phase3_gate_open = (
        winner_preservation_pass
        and archive_preservation_pass
        and regression_reduction_pass
        and concentration_pass
    )
    decision = "promote_to_masked_replay" if phase3_gate_open else "keep_prior_report_only"
    if not winner_preservation_pass:
        decision = "stop_prior_calibration_missing_winner_preservation"
    return {
        "schema": "step7o_phase2_prior_calibration_summary_v1",
        "decision": decision,
        "atlas_path": str(atlas_path),
        "rows_path": str(rows_out_path),
        "summary_path": str(summary_out_path),
        "markdown_path": str(markdown_out_path),
        "calibration_row_count": len(rows),
        "step7ml_candidate_row_count": len(step7ml_rows),
        "step7n_archive_report_row_count": len(step7n_rows),
        "metric_confidence_counts": dict(Counter(str(row["metric_confidence"]) for row in rows)),
        "exact_component_comparable_count": len(exact),
        "top_budget_count": len(top_budget),
        "top_budget_size_contract": TOP_BUDGET_SIZE,
        "top_budget_max_rows_per_case_contract": MAX_BUDGET_ROWS_PER_CASE,
        "top_budget_candidate_ids": [row["candidate_id"] for row in top_budget],
        "top_budget_case_counts": dict(top_case_counts),
        "top_budget_case024_share": case_share(top_case_counts, "24"),
        "top_budget_case025_share": case_share(top_case_counts, "25"),
        "top_budget_largest_case_id": largest_case(top_case_counts)[0],
        "top_budget_largest_case_share": largest_case(top_case_counts)[1]
        / max(sum(top_case_counts.values()), 1),
        "top_budget_official_like_improving_count": len(top_winner_rows),
        "top_budget_bbox_regression_count": top_bbox_regressions,
        "top_budget_soft_regression_count": top_soft_regressions,
        "top_budget_official_like_regression_count": sum(
            int(bool(row["official_like_regression"])) for row in top_budget
        ),
        "step7ml_winner_baseline": STEP7ML_WINNER_BASELINE,
        "step7ml_winner_signature_count": len(step7ml_winner_signatures),
        "step7ml_top_budget_winner_signature_preservation_count": len(
            step7ml_winner_signatures.intersection(top_signatures)
        ),
        "step7n_archive_official_like_winner_preservation": {
            "preserved": step7n_winner_preserved,
            "baseline": STEP7N_ARCHIVE_WINNER_BASELINE,
        },
        "step7n_non_anchor_pareto_preservation": {
            "preserved": step7n_pareto_preserved,
            "baseline": STEP7N_NON_ANCHOR_PARETO_BASELINE,
        },
        "step7ml_j_selected_budget_baseline": j_baseline,
        "step7ml_k_full_invariant_baseline": k_baseline,
        "winner_preservation_pass": winner_preservation_pass,
        "archive_preservation_pass": archive_preservation_pass,
        "regression_reduction_pass": regression_reduction_pass,
        "concentration_gate": concentration,
        "concentration_pass": concentration_pass,
        "phase3_gate_open": phase3_gate_open,
        "gnn_rl_gate_open": False,
        "forbidden_phase3_artifact_created": False,
        "next_recommendation": next_recommendation(decision),
    }


def concentration_summary(
    top_budget: list[dict[str, Any]], top_winner_rows: list[dict[str, Any]]
) -> dict[str, Any]:
    case_counts = Counter(str(row["case_id"]) for row in top_budget)
    winner_cases = {str(row["case_id"]) for row in top_winner_rows}
    largest_id, largest_count = largest_case(case_counts)
    return {
        "top_budget_case_count": len(case_counts),
        "top_budget_largest_case_id": largest_id,
        "top_budget_largest_case_share": largest_count / max(sum(case_counts.values()), 1),
        "top_budget_winner_cases": sorted(winner_cases),
        "top_budget_winners_only_case024_or_case025": bool(winner_cases)
        and winner_cases.issubset({"24", "25"}),
        "non_case024_non_case025_winner_count": sum(
            int(str(row["case_id"]) not in {"24", "25"}) for row in top_winner_rows
        ),
    }


def prior_agreement(row: dict[str, Any], atlas_prior: dict[str, Any]) -> dict[str, float]:
    case_id = str(row.get("case_id"))
    region = normalize_region(str(row.get("target_region", "")))
    route = str(row.get("route_class") or row.get("actual_locality_class") or "")
    bucket = str(
        row.get("closure_size_bucket")
        or block_count_bucket(int_value(row.get("block_count") or row.get("macro_closure_size")))
    )
    region_score = atlas_prior["region_scores"].get(case_id, {}).get(region, 0.0)
    route_score = atlas_prior["route_scores"].get(case_id, {}).get(route, 0.0)
    closure_score = float(bucket in atlas_prior["closure_buckets"].get(case_id, set()))
    return {
        "region_score": region_score,
        "route_score": route_score,
        "closure_score": closure_score,
        "score": 0.5 * region_score + 0.3 * route_score + 0.2 * closure_score,
    }


def objective_vector(row: dict[str, Any]) -> dict[str, float | None]:
    nested = row.get("objective_vector")
    vector = nested if isinstance(nested, dict) else {}
    return {
        "hpwl_delta": float_or_none(row.get("hpwl_delta", vector.get("hpwl_delta"))),
        "bbox_area_delta": float_or_none(row.get("bbox_area_delta", vector.get("bbox_area_delta"))),
        "soft_constraint_delta": float_or_none(
            row.get("soft_constraint_delta", vector.get("soft_constraint_delta"))
        ),
        "official_like_cost_delta": float_or_none(
            row.get("official_like_cost_delta", vector.get("official_like_cost_delta"))
        ),
    }


def metric_confidence(
    row: dict[str, Any], source_model: str, objective: dict[str, float | None]
) -> str:
    if source_model == "step7n_i_archive":
        return "archive_preservation_only"
    if all(value is not None for value in objective.values()):
        return "exact_component_comparable"
    if bool_value(row.get("hard_feasible_non_noop")) or bool_value(
        row.get("official_like_improving")
    ):
        return "exact_component_partial"
    return "proxy_only"


def candidate_rows(payload: Any, *, key: str) -> list[dict[str, Any]]:
    if isinstance(payload, dict) and isinstance(payload.get(key), list):
        return [row for row in payload[key] if isinstance(row, dict)]
    if isinstance(payload, list):
        return [row for row in payload if isinstance(row, dict)]
    return []


def normalize_case_region_scores(
    region_scores: dict[str, dict[str, float]]
) -> dict[str, dict[str, float]]:
    normalized: dict[str, dict[str, float]] = {}
    for case_id, scores in region_scores.items():
        max_score = max(scores.values(), default=1.0)
        normalized[case_id] = {
            region_id: value / max(max_score, EPS) for region_id, value in scores.items()
        }
    return normalized


def baseline_summary(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {}
    payload = load_json(path)
    if not isinstance(payload, dict):
        return {}
    summary = payload.get("summary")
    return summary if isinstance(summary, dict) else payload


def next_recommendation(decision: str) -> str:
    if decision == "promote_to_masked_replay":
        return "ralplan_before_phase3_masked_replay"
    if decision == "stop_prior_calibration_missing_winner_preservation":
        return "stop_step7o_prior_lane_or_rework_prior_score"
    return "keep_prior_as_report_only_do_not_open_phase3"


def calibration_markdown(summary: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# Step7O Phase2 Prior Calibration Summary",
            "",
            f"Decision: `{summary['decision']}`",
            "",
            f"- calibration_row_count: {summary['calibration_row_count']}",
            f"- exact_component_comparable_count: {summary['exact_component_comparable_count']}",
            f"- top_budget_count: {summary['top_budget_count']}",
            f"- top_budget_official_like_improving_count: "
            f"{summary['top_budget_official_like_improving_count']}",
            f"- top_budget_bbox_regression_count: "
            f"{summary['top_budget_bbox_regression_count']}",
            f"- top_budget_soft_regression_count: "
            f"{summary['top_budget_soft_regression_count']}",
            f"- winner_preservation_pass: {summary['winner_preservation_pass']}",
            f"- archive_preservation_pass: {summary['archive_preservation_pass']}",
            f"- regression_reduction_pass: {summary['regression_reduction_pass']}",
            f"- concentration_pass: {summary['concentration_pass']}",
            f"- phase3_gate_open: {summary['phase3_gate_open']}",
            f"- gnn_rl_gate_open: {summary['gnn_rl_gate_open']}",
            f"- next_recommendation: {summary['next_recommendation']}",
            "",
        ]
    )


def winner_signature(row: dict[str, Any]) -> str:
    return str(row.get("source_candidate_id") or row.get("candidate_id"))


def normalize_region(region: str) -> str:
    return region.replace("_", "c")


def block_count_bucket(count: int) -> str:
    if count <= 10:
        return "small_<=10"
    if count <= 20:
        return "medium_11_20"
    return "large_21_plus"


def case_share(case_counts: Counter[str], case_id: str) -> float:
    return case_counts.get(case_id, 0) / max(sum(case_counts.values()), 1)


def largest_case(case_counts: Counter[str]) -> tuple[str | None, int]:
    if not case_counts:
        return None, 0
    return max(case_counts.items(), key=lambda item: (item[1], item[0]))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
            count += 1
    return count


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def int_value(value: Any) -> int:
    if isinstance(value, bool) or value is None:
        return 0
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def float_value(value: Any) -> float:
    parsed = float_or_none(value)
    return 0.0 if parsed is None else parsed


def float_or_none(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def component_le(objective: dict[str, float | None], key: str, threshold: float) -> bool:
    value = objective.get(key)
    return value is not None and value <= threshold


def component_lt(objective: dict[str, float | None], key: str, threshold: float) -> bool:
    value = objective.get(key)
    return value is not None and value < threshold


def component_gt(objective: dict[str, float | None], key: str, threshold: float) -> bool:
    value = objective.get(key)
    return value is not None and value > threshold


def bool_value(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    if isinstance(value, str):
        return value.lower() in {"1", "true", "yes", "y"}
    return False
