"""Step7M Phase 3 corridor ablation over replayed exact-target rows."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

from puzzleplace.experiments.step7l_learning_guided_replay import read_jsonl
from puzzleplace.ml.floorset_training_corpus import write_json

EPS = 1e-9
MEANINGFUL_COST_EPS = 1e-7

AblationFilter = dict[str, Any]


def run_corridor_ablation(
    replay_rows_path: Path,
    rows_out_path: Path,
    summary_path: Path,
    *,
    families: list[str] | None = None,
) -> dict[str, Any]:
    replay_rows = read_jsonl(replay_rows_path)
    selected_families = families or default_ablation_families()
    rows = [ablation_row(name, replay_rows) for name in selected_families]
    rows_out_path.parent.mkdir(parents=True, exist_ok=True)
    with rows_out_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    summary = summarize_ablation(
        rows, replay_rows_path=replay_rows_path, rows_out_path=rows_out_path
    )
    write_json(summary_path, summary)
    summary_path.with_suffix(".md").write_text(
        ablation_summary_markdown(summary, rows), encoding="utf-8"
    )
    return summary


def default_ablation_families() -> list[str]:
    return [
        "all_phase2",
        "hpwl_only",
        "bbox_only",
        "soft_only",
        "hpwl_bbox",
        "hpwl_soft",
        "bbox_soft",
        "hpwl_bbox_soft",
        "wire_safe_gate",
        "soft_budgeted_gate",
        "micro_axis_source",
        "heatmap_supported",
    ]


def ablation_row(name: str, replay_rows: list[dict[str, Any]]) -> dict[str, Any]:
    rows = [row for row in replay_rows if row_matches_ablation(row, name)]
    hard = [row for row in rows if row.get("hard_feasible_nonnoop")]
    meaningful = [row for row in hard if official_delta(row) < -MEANINGFUL_COST_EPS]
    micro_improving = [row for row in hard if official_delta(row) < -EPS]
    all_nonregressing = [
        row
        for row in hard
        if hpwl_delta(row) <= EPS and bbox_delta(row) <= EPS and soft_delta(row) <= EPS
    ]
    status_counts = Counter(str(row.get("quality_gate_status")) for row in rows)
    return {
        "schema": "step7m_phase3_ablation_row_v1",
        "family": name,
        "request_count": len(rows),
        "represented_case_count": len({str(row.get("case_id")) for row in rows}),
        "hard_feasible_nonnoop_count": len(hard),
        "metric_regressing_count": sum(int(bool(row.get("metric_regressing"))) for row in hard),
        "metric_regression_rate": sum(int(bool(row.get("metric_regressing"))) for row in hard)
        / max(len(hard), 1),
        "actual_all_vector_nonregressing_count": len(all_nonregressing),
        "actual_all_vector_nonregressing_rate": len(all_nonregressing) / max(len(hard), 1),
        "meaningful_official_like_improving_count": len(meaningful),
        "micro_official_like_improving_count": len(micro_improving),
        "actual_hpwl_regression_count": sum(int(hpwl_delta(row) > EPS) for row in hard),
        "actual_bbox_regression_count": sum(int(bbox_delta(row) > EPS) for row in hard),
        "actual_soft_regression_count": sum(int(soft_delta(row) > EPS) for row in hard),
        "min_official_like_cost_delta": min((official_delta(row) for row in hard), default=0.0),
        "max_official_like_cost_delta": max((official_delta(row) for row in hard), default=0.0),
        "case025_share": case_share(rows, "25"),
        "gate_mode_counts": dict(Counter(str(row.get("gate_mode")) for row in rows)),
        "source_family_counts": dict(Counter(str(row.get("source_family")) for row in rows)),
        "status_counts": dict(status_counts),
        "proxy_actual_sign_precision": proxy_actual_precision(hard),
    }


def row_matches_ablation(row: dict[str, Any], name: str) -> bool:
    proxy = proxy_vector(row)
    hpwl = proxy_float(proxy, "hpwl_delta_proxy") <= EPS
    bbox = proxy_float(proxy, "bbox_area_delta_proxy") <= EPS
    soft = proxy_soft(proxy) <= EPS
    if name == "all_phase2":
        return True
    if name == "hpwl_only":
        return hpwl
    if name == "bbox_only":
        return bbox
    if name == "soft_only":
        return soft
    if name == "hpwl_bbox":
        return hpwl and bbox
    if name == "hpwl_soft":
        return hpwl and soft
    if name == "bbox_soft":
        return bbox and soft
    if name == "hpwl_bbox_soft":
        return hpwl and bbox and soft
    if name == "wire_safe_gate":
        return row.get("gate_mode") == "wire_safe"
    if name == "soft_budgeted_gate":
        return row.get("gate_mode") == "soft_repair_budgeted"
    if name == "micro_axis_source":
        return row.get("source_family") == "micro_axis_corridor"
    if name == "heatmap_supported":
        return row.get("heatmap_score") is not None
    raise ValueError(f"unknown ablation family: {name}")


def summarize_ablation(
    rows: list[dict[str, Any]], *, replay_rows_path: Path, rows_out_path: Path
) -> dict[str, Any]:
    non_empty = [row for row in rows if row["hard_feasible_nonnoop_count"] > 0]
    best = min(
        non_empty,
        key=lambda row: (row["metric_regression_rate"], -row["request_count"]),
        default=None,
    )
    strict = next((row for row in rows if row["family"] == "hpwl_bbox_soft"), None)
    all_phase2 = next((row for row in rows if row["family"] == "all_phase2"), None)
    decision = "fix_empty_phase3_ablation"
    if best is not None:
        decision = "promote_to_step7m_multiblock_corridor_v1"
        if best["meaningful_official_like_improving_count"] > 0:
            decision = "promote_to_archive_integration_review"
        elif (
            strict
            and all_phase2
            and strict["metric_regression_rate"] < all_phase2["metric_regression_rate"]
        ):
            decision = "tighten_to_hpwl_bbox_soft_then_multiblock_v1"
    return {
        "schema": "step7m_phase3_ablation_summary_v1",
        "decision": decision,
        "replay_rows_path": str(replay_rows_path),
        "ablation_rows_path": str(rows_out_path),
        "family_count": len(rows),
        "non_empty_family_count": len(non_empty),
        "best_family": best["family"] if best else None,
        "best_metric_regression_rate": best["metric_regression_rate"] if best else None,
        "best_hard_feasible_nonnoop_count": best["hard_feasible_nonnoop_count"] if best else 0,
        "hpwl_bbox_soft_metric_regression_rate": strict["metric_regression_rate"]
        if strict
        else None,
        "all_phase2_metric_regression_rate": all_phase2["metric_regression_rate"]
        if all_phase2
        else None,
        "soft_budgeted_gate_metric_regression_rate": _family_rate(rows, "soft_budgeted_gate"),
        "wire_safe_gate_metric_regression_rate": _family_rate(rows, "wire_safe_gate"),
        "meaningful_official_like_improving_total": sum(
            row["meaningful_official_like_improving_count"] for row in rows
        ),
        "heatmap_supported_request_count": next(
            (row["request_count"] for row in rows if row["family"] == "heatmap_supported"), 0
        ),
        "gnn_rl_gate_open": False,
        "next_recommendation": next_recommendation(best, strict, all_phase2),
    }


def next_recommendation(
    best: dict[str, Any] | None,
    strict: dict[str, Any] | None,
    all_phase2: dict[str, Any] | None,
) -> str:
    if best is None:
        return "rebuild_phase2_replay_rows"
    if best["meaningful_official_like_improving_count"] > 0:
        return "review_archive_candidates_before_any_model_work"
    if (
        strict
        and all_phase2
        and strict["metric_regression_rate"] < all_phase2["metric_regression_rate"]
    ):
        return (
            "drop hpwl-budgeted soft repairs, keep hpwl-bbox-soft strict gate, "
            "then test deterministic paired/block-shift corridors"
        )
    return "test deterministic multiblock corridors before GNN/RL"


def ablation_summary_markdown(summary: dict[str, Any], rows: list[dict[str, Any]]) -> str:
    lines = [
        "# Step7M Phase 3 Corridor Ablation",
        "",
        f"Decision: `{summary['decision']}`",
        "",
        f"- best_family: {summary['best_family']}",
        f"- best_metric_regression_rate: {summary['best_metric_regression_rate']}",
        "- hpwl_bbox_soft_metric_regression_rate: "
        f"{summary['hpwl_bbox_soft_metric_regression_rate']}",
        f"- all_phase2_metric_regression_rate: {summary['all_phase2_metric_regression_rate']}",
        "- wire_safe_gate_metric_regression_rate: "
        f"{summary['wire_safe_gate_metric_regression_rate']}",
        "- soft_budgeted_gate_metric_regression_rate: "
        f"{summary['soft_budgeted_gate_metric_regression_rate']}",
        "- meaningful_official_like_improving_total: "
        f"{summary['meaningful_official_like_improving_total']}",
        f"- heatmap_supported_request_count: {summary['heatmap_supported_request_count']}",
        f"- gnn_rl_gate_open: {summary['gnn_rl_gate_open']}",
        f"- next_recommendation: {summary['next_recommendation']}",
        "",
        "## Family rows",
        "",
        "| family | n | cases | regression_rate | nonreg_rate | hpwl_reg | soft_reg | meaningful |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| {family} | {request_count} | {represented_case_count} | "
            "{metric_regression_rate:.3f} | {actual_all_vector_nonregressing_rate:.3f} | "
            "{actual_hpwl_regression_count} | "
            "{actual_soft_regression_count} | {meaningful_official_like_improving_count} |".format(
                **row
            )
        )
    return "\n".join(lines) + "\n"


def proxy_actual_precision(rows: list[dict[str, Any]]) -> dict[str, float | int]:
    total = 0
    matched = 0
    for row in rows:
        signs = row.get("proxy_actual_signs")
        if not isinstance(signs, dict):
            continue
        for component in ("hpwl", "bbox", "soft"):
            total += 1
            matched += int(bool(signs.get(f"{component}_nonregression_match")))
    return {
        "component_match_count": matched,
        "component_total": total,
        "component_precision": matched / max(total, 1),
    }


def proxy_vector(row: dict[str, Any]) -> dict[str, Any]:
    value = row.get("proxy_objective_vector")
    return value if isinstance(value, dict) else {}


def proxy_float(proxy: dict[str, Any], key: str) -> float:
    return to_float(proxy.get(key))


def proxy_soft(proxy: dict[str, Any]) -> float:
    return (
        proxy_float(proxy, "boundary_delta_proxy")
        + proxy_float(proxy, "group_delta_proxy")
        + proxy_float(proxy, "mib_delta_proxy")
    )


def official_delta(row: dict[str, Any]) -> float:
    return to_float(row.get("official_like_cost_delta"))


def hpwl_delta(row: dict[str, Any]) -> float:
    return to_float(row.get("hpwl_delta"))


def bbox_delta(row: dict[str, Any]) -> float:
    return to_float(row.get("bbox_area_delta"))


def soft_delta(row: dict[str, Any]) -> float:
    return to_float(row.get("soft_constraint_delta"))


def case_share(rows: list[dict[str, Any]], case_id: str) -> float:
    if not rows:
        return 0.0
    return sum(int(str(row.get("case_id")) == case_id) for row in rows) / len(rows)


def _family_rate(rows: list[dict[str, Any]], family: str) -> float | None:
    row = next((item for item in rows if item["family"] == family), None)
    return None if row is None else float(row["metric_regression_rate"])


def to_float(value: Any, default: float = 0.0) -> float:
    if value is None:
        return default
    if isinstance(value, int | float):
        return float(value)
    try:
        return float(value)
    except (TypeError, ValueError):
        return default
