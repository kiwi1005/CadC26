"""Step7O training-derived topology/wire-demand prior sidecar.

Phase0 records that Step7N-ALR stopped before reservoir widening. Phase1 builds
an atlas from Step7ML-G training-prior artifacts only. It deliberately does not
read Step7 candidate outcomes, Step7N archives, validation target labels, or any
runtime/finalizer path.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import Any

FORBIDDEN_TERMS = (
    "target_positions",
    "fp_sol",
    "tree_sol",
    "supervised_target",
    "label_target",
)
SCHEMA_REPORT = "artifacts/research/step7ml_g_schema_report.json"
DENIED_SOURCE_HINTS = (
    "step7ml_g_candidate_quality_examples.json",
    "step7ml_i_decoded_candidates.json",
    "step7ml_j_ranked_candidates.json",
    "step7ml_k_invariant_candidates.json",
    "step7n_i_quality_filtered_candidates.json",
    "step7n_phase1_reservoir_atlas.jsonl",
    "step7n_phase2_reservoir_requests.jsonl",
)


def build_input_inventory(
    step7n_phase0_summary_path: Path,
    step7ml_g_inventory_path: Path,
    step7ml_i_quality_path: Path,
    step7ml_j_quality_path: Path,
    step7ml_k_quality_path: Path,
    inventory_out_path: Path,
    guard_markdown_path: Path,
    *,
    step7ml_g_schema_path: Path = Path(SCHEMA_REPORT),
) -> dict[str, Any]:
    """Write Phase0 inventory and stop guard."""

    step7n = load_json(step7n_phase0_summary_path)
    data_inventory = load_json(step7ml_g_inventory_path)
    schema = load_json(step7ml_g_schema_path) if step7ml_g_schema_path.exists() else {}
    quality_reports = {
        "step7ml_i": load_json(step7ml_i_quality_path),
        "step7ml_j": load_json(step7ml_j_quality_path),
        "step7ml_k": load_json(step7ml_k_quality_path),
    }
    metrics = dict_value(data_inventory.get("metrics"))
    label_rule = str(schema.get("label_separation_rule", ""))
    phase0_gate = (
        step7n.get("decision") == "stop_no_archive_signal"
        and int_value(step7n.get("strict_archive_candidate_count")) == 0
        and step7n.get("phase1_gate_open") is False
        and int_value(metrics.get("layout_prior_example_count")) >= 40000
        and int_value(metrics.get("region_heatmap_example_count")) >= 10000
        and bool(label_rule)
    )
    inventory = {
        "schema": "step7o_phase0_input_inventory_v1",
        "decision": "promote_to_training_demand_atlas"
        if phase0_gate
        else "stop_missing_training_prior_contract",
        "step7n_phase0": {
            "summary_path": str(step7n_phase0_summary_path),
            "decision": step7n.get("decision"),
            "strict_archive_candidate_count": step7n.get("strict_archive_candidate_count"),
            "strict_meaningful_non_micro_winner_count": step7n.get(
                "strict_meaningful_non_micro_winner_count"
            ),
            "phase1_gate_open": step7n.get("phase1_gate_open"),
        },
        "step7ml_g": {
            "inventory_path": str(step7ml_g_inventory_path),
            "schema_path": str(step7ml_g_schema_path),
            "layout_prior_example_count": int_value(metrics.get("layout_prior_example_count")),
            "region_heatmap_example_count": int_value(metrics.get("region_heatmap_example_count")),
            "candidate_quality_example_count": int_value(
                metrics.get("candidate_quality_example_count")
            ),
            "label_separation_rule": label_rule,
        },
        "quality_report_paths": {
            "step7ml_i": str(step7ml_i_quality_path),
            "step7ml_j": str(step7ml_j_quality_path),
            "step7ml_k": str(step7ml_k_quality_path),
        },
        "quality_report_summaries": summarize_quality_reports(quality_reports),
        "phase1_gate_open": phase0_gate,
        "forbidden_to_open": [
            "step7n_reservoir_phase1_or_phase2",
            "phase2_prior_calibration",
            "phase3_replay",
            "gnn_rl_model_training",
        ],
    }
    write_json(inventory_out_path, inventory)
    guard_markdown_path.parent.mkdir(parents=True, exist_ok=True)
    guard_markdown_path.write_text(input_inventory_markdown(inventory), encoding="utf-8")
    return inventory


def build_training_demand_atlas(
    step7ml_g_inventory_path: Path,
    step7ml_g_schema_path: Path,
    layout_prior_path: Path,
    region_heatmap_path: Path,
    validation_cases: Iterable[int],
    atlas_out_path: Path,
    summary_out_path: Path,
    markdown_out_path: Path,
) -> dict[str, Any]:
    """Build a Phase1 atlas from Step7ML-G prior artifacts only."""

    cases = [int(case_id) for case_id in validation_cases]
    inventory = load_json(step7ml_g_inventory_path)
    schema = load_json(step7ml_g_schema_path)
    layout_rows = list_rows(load_json(layout_prior_path))
    heatmap_rows = list_rows(load_json(region_heatmap_path))
    closure_stats = closure_prior_stats(layout_rows)
    region_stats = region_prior_stats(heatmap_rows)
    atlas_rows = atlas_rows_for_cases(cases, closure_stats, region_stats)
    write_jsonl(atlas_out_path, atlas_rows)
    forbidden_count = forbidden_term_count(atlas_rows)
    source_ledger = phase1_source_ledger(
        layout_prior_path,
        region_heatmap_path,
        step7ml_g_schema_path,
        step7ml_g_inventory_path,
    )
    represented_cases = {row["case_id"] for row in atlas_rows}
    summary = {
        "schema": "step7o_phase1_training_demand_summary_v1",
        "decision": "promote_to_prior_calibration"
        if atlas_rows and len(represented_cases) >= len(cases) and forbidden_count == 0
        else "stop_no_training_prior_signal",
        "atlas_path": str(atlas_out_path),
        "summary_path": str(summary_out_path),
        "markdown_path": str(markdown_out_path),
        "atlas_row_count": len(atlas_rows),
        "represented_case_count": len(represented_cases),
        "validation_cases": cases,
        "source_ledger": source_ledger,
        "rejected_sources": source_ledger["rejected_sources"],
        "forbidden_validation_label_term_count": forbidden_count,
        "closure_size_bucket_counts": closure_stats["bucket_counts"],
        "route_locality_bucket_counts": dict(
            Counter(str(row["route_locality_proxy"]) for row in atlas_rows)
        ),
        "region_prior_top": region_stats["top_regions"],
        "net_terminal_pressure_summary": closure_stats["net_terminal_pressure_summary"],
        "label_separation_rule_present": bool(schema.get("label_separation_rule")),
        "training_prior_counts": {
            "layout_prior_rows_read": len(layout_rows),
            "region_heatmap_rows_read": len(heatmap_rows),
            "inventory_layout_prior_count": int_value(
                dict_value(inventory.get("metrics")).get("layout_prior_example_count")
            ),
            "inventory_region_heatmap_count": int_value(
                dict_value(inventory.get("metrics")).get("region_heatmap_example_count")
            ),
        },
        "phase2_gate_open": bool(
            atlas_rows and len(represented_cases) >= len(cases) and forbidden_count == 0
        ),
        "gnn_rl_gate_open": False,
    }
    write_json(summary_out_path, summary)
    markdown_out_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_out_path.write_text(training_demand_markdown(summary), encoding="utf-8")
    return summary


def summarize_quality_reports(reports: dict[str, Any]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for name, report in reports.items():
        if not isinstance(report, dict):
            summary[name] = {}
            continue
        metrics = dict_value(report.get("summary"))
        if not metrics:
            metrics = report
        summary[name] = {
            "quality_gate_pass_count": metrics.get("quality_gate_pass_count")
            or metrics.get("quality_gate_pass"),
            "official_like_improving_count": metrics.get("official_like_improving_count")
            or metrics.get("official_like_improving"),
            "hard_feasible_nonnoop_count": metrics.get("hard_feasible_nonnoop_count")
            or metrics.get("hard_feasible_non_noop"),
            "bbox_regression_count": metrics.get("bbox_regression_count"),
            "soft_regression_count": metrics.get("soft_regression_count"),
        }
    return summary


def closure_prior_stats(rows: list[dict[str, Any]]) -> dict[str, Any]:
    groups: defaultdict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        closure_type = str(row.get("closure_type", "unknown"))
        bucket = block_count_bucket(int_value(row.get("block_count")))
        groups[(closure_type, bucket)].append(row)
    priors: list[dict[str, Any]] = []
    bucket_counts: Counter[str] = Counter()
    pressure_values: list[float] = []
    for (closure_type, bucket), group in sorted(groups.items()):
        bucket_counts[f"{closure_type}:{bucket}"] = len(group)
        pressure = mean(
            [
                float_value(row.get("b2b_edge_count"))
                + float_value(row.get("p2b_edge_count"))
                + float_value(row.get("pin_count"))
                for row in group
            ]
        )
        pressure_values.append(pressure)
        priors.append(
            {
                "closure_type": closure_type,
                "closure_size_bucket": bucket,
                "training_example_count": len(group),
                "mean_block_count": mean([float_value(row.get("block_count")) for row in group]),
                "mean_closure_area": mean([float_value(row.get("closure_area")) for row in group]),
                "mean_closure_aspect": mean(
                    [float_value(row.get("closure_aspect")) for row in group]
                ),
                "mean_wire_pressure_proxy": pressure,
                "route_locality_proxy": route_proxy(closure_type, bucket),
            }
        )
    return {
        "priors": sorted(
            priors,
            key=lambda row: (-row["training_example_count"], row["closure_type"]),
        ),
        "bucket_counts": dict(bucket_counts),
        "net_terminal_pressure_summary": {
            "mean": mean(pressure_values),
            "max": max(pressure_values, default=0.0),
            "min": min(pressure_values, default=0.0),
            "bucket_count": len(pressure_values),
        },
    }


def region_prior_stats(rows: list[dict[str, Any]]) -> dict[str, Any]:
    totals: defaultdict[str, dict[str, float]] = defaultdict(
        lambda: {"region": 0.0, "area": 0.0, "free": 0.0, "count": 0.0}
    )
    for row in rows:
        region_matrix = matrix(row.get("region_distribution"))
        area_matrix = matrix(row.get("area_distribution"))
        free_matrix = matrix(row.get("free_space_proxy"))
        for r_index, values in enumerate(region_matrix):
            for c_index, region_value in enumerate(values):
                key = f"r{r_index}c{c_index}"
                totals[key]["region"] += float_value(region_value)
                totals[key]["area"] += matrix_value(area_matrix, r_index, c_index)
                totals[key]["free"] += matrix_value(free_matrix, r_index, c_index)
                totals[key]["count"] += 1.0
    top_regions: list[dict[str, Any]] = []
    for region_id, stats in totals.items():
        count = max(stats["count"], 1.0)
        top_regions.append(
            {
                "region_id": region_id,
                "mean_region_prior": stats["region"] / count,
                "mean_area_prior": stats["area"] / count,
                "mean_free_space_proxy": stats["free"] / count,
                "training_example_count": int(stats["count"]),
            }
        )
    top_regions.sort(
        key=lambda row: (
            -row["mean_region_prior"],
            -row["mean_free_space_proxy"],
            row["region_id"],
        )
    )
    return {"top_regions": top_regions[:8]}


def atlas_rows_for_cases(
    cases: list[int], closure_stats: dict[str, Any], region_stats: dict[str, Any]
) -> list[dict[str, Any]]:
    atlas: list[dict[str, Any]] = []
    closure_priors = closure_stats["priors"][:8]
    region_priors = region_stats["top_regions"]
    for case_id in cases:
        for index, prior in enumerate(closure_priors):
            atlas.append(
                {
                    "schema": "step7o_phase1_training_demand_atlas_row_v1",
                    "case_id": case_id,
                    "atlas_id": f"case{case_id:03d}:closure:{index:02d}",
                    "feature_family": "training_closure_topology_prior",
                    "source_provenance": "step7ml_g_layout_prior_examples",
                    "route_locality_proxy": prior["route_locality_proxy"],
                    **prior,
                }
            )
        for index, prior in enumerate(region_priors):
            atlas.append(
                {
                    "schema": "step7o_phase1_training_demand_atlas_row_v1",
                    "case_id": case_id,
                    "atlas_id": f"case{case_id:03d}:region:{index:02d}",
                    "feature_family": "training_region_heatmap_prior",
                    "source_provenance": "step7ml_g_region_heatmap_examples",
                    "route_locality_proxy": "static_region_prior",
                    **prior,
                }
            )
    return atlas


def phase1_source_ledger(
    layout_prior_path: Path,
    region_heatmap_path: Path,
    schema_path: Path,
    inventory_path: Path,
) -> dict[str, Any]:
    accepted = [
        {"path": str(layout_prior_path), "status": "accepted_allowlist"},
        {"path": str(region_heatmap_path), "status": "accepted_allowlist"},
        {"path": str(schema_path), "status": "accepted_allowlist"},
        {"path": str(inventory_path), "status": "accepted_allowlist"},
        {"path": "static_validation_geometry_provenance", "status": "accepted_allowlist"},
    ]
    rejected = [
        {"path": hint, "status": "rejected_denylist", "reason": "not_read_by_phase1_atlas"}
        for hint in DENIED_SOURCE_HINTS
    ]
    return {
        "accepted_sources": accepted,
        "rejected_sources": rejected,
        "accepted_source_count": len(accepted),
        "rejected_source_count": len(rejected),
    }


def input_inventory_markdown(inventory: dict[str, Any]) -> str:
    step7n = inventory["step7n_phase0"]
    step7ml = inventory["step7ml_g"]
    return "\n".join(
        [
            "# Step7O Phase0 Input Inventory / Stop Guard",
            "",
            f"Decision: `{inventory['decision']}`",
            "",
            f"- Step7N decision: {step7n['decision']}",
            f"- Step7N strict_archive_candidate_count: {step7n['strict_archive_candidate_count']}",
            f"- Step7N phase1_gate_open: {step7n['phase1_gate_open']}",
            f"- layout_prior_example_count: {step7ml['layout_prior_example_count']}",
            f"- region_heatmap_example_count: {step7ml['region_heatmap_example_count']}",
            f"- candidate_quality_example_count: {step7ml['candidate_quality_example_count']}",
            f"- Phase1 gate open: {inventory['phase1_gate_open']}",
            "",
        ]
    )


def training_demand_markdown(summary: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# Step7O Phase1 Training-Demand Atlas Summary",
            "",
            f"Decision: `{summary['decision']}`",
            "",
            f"- atlas_row_count: {summary['atlas_row_count']}",
            f"- represented_case_count: {summary['represented_case_count']}",
            "- forbidden_validation_label_term_count: "
            f"{summary['forbidden_validation_label_term_count']}",
            f"- phase2_gate_open: {summary['phase2_gate_open']}",
            f"- gnn_rl_gate_open: {summary['gnn_rl_gate_open']}",
            f"- accepted_source_count: {summary['source_ledger']['accepted_source_count']}",
            f"- rejected_source_count: {summary['source_ledger']['rejected_source_count']}",
            "",
        ]
    )


def block_count_bucket(count: int) -> str:
    if count <= 4:
        return "small_<=4"
    if count <= 10:
        return "medium_5_10"
    return "large_11_plus"


def route_proxy(closure_type: str, bucket: str) -> str:
    if bucket.startswith("small"):
        return "local"
    if closure_type == "mib":
        return "macro"
    if bucket.startswith("large"):
        return "global_report_only"
    return "regional"


def forbidden_term_count(rows: list[dict[str, Any]]) -> int:
    text = json.dumps(rows, sort_keys=True).lower()
    return sum(text.count(term) for term in FORBIDDEN_TERMS)


def matrix(value: Any) -> list[list[Any]]:
    return value if isinstance(value, list) else []


def matrix_value(values: list[list[Any]], row: int, col: int) -> float:
    try:
        return float_value(values[row][col])
    except (IndexError, TypeError):
        return 0.0


def list_rows(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, dict) and isinstance(payload.get("rows"), list):
        return [row for row in payload["rows"] if isinstance(row, dict)]
    if isinstance(payload, list):
        return [row for row in payload if isinstance(row, dict)]
    return []


def dict_value(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def int_value(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def float_value(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def mean(values: Iterable[float]) -> float:
    items = list(values)
    return sum(items) / max(len(items), 1)


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
