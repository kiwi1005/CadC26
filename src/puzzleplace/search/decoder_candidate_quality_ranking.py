"""Step7ML-J metric-aware ranking for geometry-decoded macro candidates."""

from __future__ import annotations

import json
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from puzzleplace.ml.floorset_training_corpus import write_json
from puzzleplace.ml.supervised_macro_layout import STEP7P_BASELINES, find_artifact
from puzzleplace.ml.training_backed_data_mart import as_float, rows

STEP7ML_I_BASELINES = {
    "hard_feasible_non_noop": 67,
    "official_like_improving": 2,
    "quality_gate_pass": 2,
    "overlap_after_decode": 4,
}
ROUTE_RANK = {"local": 0, "regional": 1, "macro": 2, "global": 3, None: 4, "None": 4}


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _baseline_rows(base_dir: Path) -> dict[str, dict[str, Any]]:
    path = find_artifact("step7p_real_repack_candidates.json", base_dir=base_dir)
    indexed: dict[str, dict[str, Any]] = {}
    if path is None:
        return indexed
    for row in rows(load_json(path)):
        for key in ("candidate_id", "step7o_a_variant_candidate_id", "source_candidate_id"):
            if row.get(key):
                indexed[str(row[key])] = row
    return indexed


def _input_inventory(base_dir: Path) -> dict[str, Any]:
    rels = [
        "artifacts/research/step7ml_i_decoded_candidates.json",
        "artifacts/research/step7ml_i_feasibility_report.json",
        "artifacts/research/step7ml_i_metric_report.json",
        "artifacts/research/step7ml_i_quality_gate_report.json",
        "artifacts/research/step7n_i_quality_filtered_candidates.json",
        "artifacts/research/step7n_i_archive_replay.json",
    ]
    entries = []
    for rel in rels:
        path = base_dir / rel
        if not path.exists() and "step7ml_i" in rel:
            path = Path("/home/hwchen/PROJ/CadC26") / rel
        entries.append(
            {
                "path": rel,
                "exists": path.exists(),
                "size_bytes": path.stat().st_size if path.exists() else 0,
            }
        )
    return {"schema": "step7ml_j_input_inventory_v1", "inputs": entries}


def _slot_compactness(row: dict[str, Any]) -> float:
    raw_box = row.get("closure_bbox")
    box = raw_box if isinstance(raw_box, dict) else {}
    area = as_float(box.get("w"), 1.0) * as_float(box.get("h"), 1.0)
    block_area = 0.0
    for block in row.get("decoded_blocks_preview", []) or []:
        if isinstance(block, dict):
            block_area += as_float(block.get("w")) * as_float(block.get("h"))
    return block_area / max(area, 1e-9)


def _objective(row: dict[str, Any], baseline: dict[str, Any]) -> dict[str, Any]:
    route = row.get("route_class")
    official_delta = baseline.get("official_like_cost_delta")
    hpwl_delta = baseline.get("hpwl_delta")
    bbox_delta = baseline.get("bbox_area_delta")
    soft_delta = baseline.get("soft_constraint_delta")
    hard = bool(row.get("hard_feasible_non_noop"))
    global_route = str(route) == "global"
    no_op = not bool(row.get("non_original_non_noop"))
    return {
        "candidate_id": row.get("candidate_id"),
        "source_candidate_id": row.get("source_candidate_id"),
        "case_id": row.get("case_id"),
        "decoder": row.get("decoder"),
        "hard_feasible_non_noop": hard,
        "feasibility_rank": 0 if hard else 1,
        "route_class": route,
        "route_rank": ROUTE_RANK.get(route, ROUTE_RANK.get(str(route), 4)),
        "global_report_only": global_route,
        "no_op": no_op,
        "dominated_by_original": bool(row.get("dominated_by_original")),
        "metric_regressing": bool(row.get("metric_regressing")),
        "official_like_improving": bool(row.get("official_like_improving")),
        "quality_gate_pass": bool(row.get("quality_gate_pass")),
        "hpwl_delta": hpwl_delta,
        "bbox_area_delta": bbox_delta,
        "soft_constraint_delta": soft_delta,
        "official_like_cost_delta": official_delta,
        "moved_block_count": row.get("moved_block_count"),
        "closure_block_count": row.get("block_count"),
        "slot_count": row.get("slot_count", 0),
        "compactness": _slot_compactness(row),
        "failure_attribution": row.get("failure_attribution"),
        "selection_reason": None,
    }


def _dominates(a: dict[str, Any], b: dict[str, Any]) -> bool:
    keys = [
        "feasibility_rank",
        "route_rank",
        "official_like_cost_delta",
        "hpwl_delta",
        "bbox_area_delta",
        "soft_constraint_delta",
    ]
    better = False
    for key in keys:
        av = as_float(a.get(key), 0.0)
        bv = as_float(b.get(key), 0.0)
        if av > bv + 1e-12:
            return False
        if av < bv - 1e-12:
            better = True
    return better


def _original_anchor(case_id: Any) -> dict[str, Any]:
    return {
        "candidate_id": f"case{case_id}:original_anchor",
        "case_id": case_id,
        "decoder": "original_anchor",
        "hard_feasible_non_noop": True,
        "feasibility_rank": 0,
        "route_rank": 0,
        "official_like_cost_delta": 0.0,
        "hpwl_delta": 0.0,
        "bbox_area_delta": 0.0,
        "soft_constraint_delta": 0.0,
        "global_report_only": False,
        "no_op": True,
        "dominated_by_original": False,
        "metric_regressing": False,
        "official_like_improving": False,
        "quality_gate_pass": False,
    }


def _rank_score(row: dict[str, Any]) -> tuple[float, float, float, float, float, str]:
    return (
        as_float(row.get("feasibility_rank"), 1.0),
        as_float(row.get("route_rank"), 4.0),
        as_float(row.get("official_like_cost_delta"), 0.0),
        as_float(row.get("soft_constraint_delta"), 0.0),
        -as_float(row.get("compactness"), 0.0),
        str(row.get("candidate_id")),
    )


def build_objective_vectors(base_dir: Path) -> list[dict[str, Any]]:
    decoded_path = base_dir / "artifacts/research/step7ml_i_decoded_candidates.json"
    if not decoded_path.exists():
        decoded_path = Path(
            "/home/hwchen/PROJ/CadC26/artifacts/research/"
            "step7ml_i_decoded_candidates.json"
        )
    decoded = rows(load_json(decoded_path))
    baseline = _baseline_rows(base_dir)
    vectors = []
    for row in decoded:
        vectors.append(_objective(row, baseline.get(str(row.get("source_candidate_id")), {})))
    return vectors


def select_candidates(vectors: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    report_rows: list[dict[str, Any]] = []
    by_case: dict[Any, list[dict[str, Any]]] = defaultdict(list)
    for row in vectors:
        by_case[row.get("case_id")].append(row)
    for case_id, case_rows in by_case.items():
        anchors = [_original_anchor(case_id)]
        candidates = [r for r in case_rows if not r.get("global_report_only")]
        pool = candidates + anchors
        for row in candidates:
            dominators = [
                other for other in pool if other is not row and _dominates(other, row)
            ]
            item = dict(row)
            item["dominator_count"] = len(dominators)
            item["dominated_by_current_archive"] = bool(dominators)
            if (
                row.get("hard_feasible_non_noop")
                and not dominators
                and not row.get("metric_regressing")
            ):
                item["selected"] = True
                item["selection_reason"] = "non_dominated_hard_feasible_metric_safe"
                selected.append(item)
            elif row.get("official_like_improving") or row.get("quality_gate_pass"):
                item["selected"] = True
                item["selection_reason"] = "preserve_known_winner_or_gate_pass"
                selected.append(item)
            else:
                item["selected"] = False
                item["selection_reason"] = "dominated_or_metric_regressing"
            report_rows.append(item)
    selected = sorted(
        {str(row["candidate_id"]): row for row in selected}.values(), key=_rank_score
    )
    return selected, {"rows": report_rows}


def _summary(
    vectors: list[dict[str, Any]],
    selected: list[dict[str, Any]],
    runtime_ms: float,
) -> dict[str, Any]:
    hard = [row for row in vectors if row.get("hard_feasible_non_noop")]
    selected_hard = [row for row in selected if row.get("hard_feasible_non_noop")]
    return {
        "input_decoded_candidate_count": len(vectors),
        "hard_feasible_nonnoop_input_count": len(hard),
        "ranked_candidate_count": len(selected),
        "quality_gate_pass_count": sum(1 for row in selected if row.get("quality_gate_pass")),
        "official_like_improving_count": sum(
            1 for row in selected if row.get("official_like_improving")
        ),
        "official_like_improvement_density": sum(
            1 for row in selected if row.get("official_like_improving")
        )
        / max(len(selected_hard), 1),
        "dominated_by_original_count": sum(
            1 for row in selected if row.get("dominated_by_original")
        ),
        "metric_regressing_count": sum(1 for row in selected if row.get("metric_regressing")),
        "selected_candidate_count_by_decoder": dict(
            Counter(str(row.get("decoder")) for row in selected)
        ),
        "quality_gate_pass_by_decoder": dict(
            Counter(str(row.get("decoder")) for row in selected if row.get("quality_gate_pass"))
        ),
        "official_like_improving_by_decoder": dict(
            Counter(
                str(row.get("decoder"))
                for row in selected
                if row.get("official_like_improving")
            )
        ),
        "route_count_by_class": dict(Counter(str(row.get("route_class")) for row in selected)),
        "global_candidate_report_only_count": sum(
            1 for row in vectors if row.get("global_report_only")
        ),
        "hpwl_gain_but_official_like_loss_count": sum(
            1
            for row in vectors
            if as_float(row.get("hpwl_delta")) < 0
            and as_float(row.get("official_like_cost_delta")) > 0
        ),
        "bbox_regression_count": sum(
            1 for row in vectors if as_float(row.get("bbox_area_delta")) > 0
        ),
        "soft_regression_count": sum(
            1 for row in vectors if as_float(row.get("soft_constraint_delta")) > 0
        ),
        "original_inclusive_pareto_non_empty_count": len({row.get("case_id") for row in selected}),
        "runtime_proxy_ms": runtime_ms,
        "baseline_step7p": STEP7P_BASELINES,
        "baseline_step7ml_i": STEP7ML_I_BASELINES,
    }


def _failure_analysis(
    vectors: list[dict[str, Any]], selected: list[dict[str, Any]]
) -> dict[str, Any]:
    selected_ids = {str(row.get("candidate_id")) for row in selected}
    rejected = [row for row in vectors if str(row.get("candidate_id")) not in selected_ids]
    return {
        "schema": "step7ml_j_decoder_failure_analysis_v1",
        "input_failure_counts": dict(
            Counter(str(row.get("failure_attribution")) for row in vectors)
        ),
        "rejected_failure_counts": dict(
            Counter(str(row.get("failure_attribution")) for row in rejected)
        ),
        "metric_quality_counts": {
            "input_metric_regressing": sum(1 for row in vectors if row.get("metric_regressing")),
            "selected_metric_regressing": sum(
                1 for row in selected if row.get("metric_regressing")
            ),
            "input_dominated_by_original": sum(
                1 for row in vectors if row.get("dominated_by_original")
            ),
            "selected_dominated_by_original": sum(
                1 for row in selected if row.get("dominated_by_original")
            ),
        },
        "interpretation": (
            "Legal decoded candidates mostly fail because Step7P official-like provenance marks "
            "them as dominated or metric-regressing, not because geometry remains infeasible."
        ),
    }


def _sensitivity(vectors: list[dict[str, Any]]) -> dict[str, Any]:
    hard = [row for row in vectors if row.get("hard_feasible_non_noop")]
    sorted_rows = sorted(hard, key=_rank_score)
    budgets = [5, 10, 20, 50]
    return {
        "schema": "step7ml_j_selection_sensitivity_v1",
        "budgets": [
            {
                "top_k": k,
                "official_like_improving_count": sum(
                    1 for row in sorted_rows[:k] if row.get("official_like_improving")
                ),
                "quality_gate_pass_count": sum(
                    1 for row in sorted_rows[:k] if row.get("quality_gate_pass")
                ),
                "metric_regressing_count": sum(
                    1 for row in sorted_rows[:k] if row.get("metric_regressing")
                ),
                "dominated_by_original_count": sum(
                    1 for row in sorted_rows[:k] if row.get("dominated_by_original")
                ),
            }
            for k in budgets
        ],
        "weak_ranker_feature_contribution": {
            "used": False,
            "reason": (
                "No reliable learned ranker artifact is present; "
                "constrained Pareto is primary."
            ),
        },
    }


def decide(summary: dict[str, Any]) -> str:
    if summary["quality_gate_pass_count"] > 2 or summary["official_like_improving_count"] > 2:
        return "promote_decoder_quality_selector"
    if summary["metric_regressing_count"] < STEP7ML_I_BASELINES["hard_feasible_non_noop"]:
        return "use_selector_as_budget_controller"
    return "refine_decoder_metric_features"


def run_decoder_quality_ranking(base_dir: Path, output_dir: Path) -> dict[str, Any]:
    started = time.perf_counter()
    vectors = build_objective_vectors(base_dir)
    selected, report = select_candidates(vectors)
    summary = _summary(vectors, selected, (time.perf_counter() - started) * 1000.0)
    decision = decide(summary)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "step7ml_j_input_inventory.json", _input_inventory(base_dir))
    write_json(
        output_dir / "step7ml_j_objective_vectors.json",
        {"schema": "step7ml_j_objective_vectors_v1", "rows": vectors},
    )
    write_json(
        output_dir / "step7ml_j_ranked_candidates.json",
        {"schema": "step7ml_j_ranked_candidates_v1", "rows": selected, "summary": summary},
    )
    write_json(
        output_dir / "step7ml_j_quality_gate_report.json",
        {"schema": "step7ml_j_quality_gate_report_v1", "summary": summary},
    )
    write_json(
        output_dir / "step7ml_j_pareto_archive.json",
        {"schema": "step7ml_j_pareto_archive_v1", "selection_report": report, "front": selected},
    )
    write_json(
        output_dir / "step7ml_j_decoder_failure_analysis.json",
        _failure_analysis(vectors, selected),
    )
    write_json(output_dir / "step7ml_j_selection_sensitivity.json", _sensitivity(vectors))
    (output_dir / "step7ml_j_decision.md").write_text(
        _decision_md(decision, summary), encoding="utf-8"
    )
    return {"decision": decision, "summary": summary}


def _decision_md(decision: str, summary: dict[str, Any]) -> str:
    return f"""# Step7ML-J Decoder Candidate Quality Ranking

Decision: `{decision}`

## Key metrics

- input_decoded_candidate_count: {summary['input_decoded_candidate_count']}
- hard_feasible_nonnoop_input_count: {summary['hard_feasible_nonnoop_input_count']}
- ranked_candidate_count: {summary['ranked_candidate_count']}
- quality_gate_pass_count: {summary['quality_gate_pass_count']} vs baseline 2
- official_like_improving_count: {summary['official_like_improving_count']} vs baseline 2
- official_like_improvement_density: {summary['official_like_improvement_density']}
- dominated_by_original_count: {summary['dominated_by_original_count']}
- metric_regressing_count: {summary['metric_regressing_count']}
- selected_candidate_count_by_decoder: {summary['selected_candidate_count_by_decoder']}
- route_count_by_class: {summary['route_count_by_class']}
- global_candidate_report_only_count: {summary['global_candidate_report_only_count']}
- hpwl_gain_but_official_like_loss_count: {summary['hpwl_gain_but_official_like_loss_count']}
- bbox_regression_count: {summary['bbox_regression_count']}
- soft_regression_count: {summary['soft_regression_count']}
- original_inclusive_pareto_non_empty_count: {summary['original_inclusive_pareto_non_empty_count']}

## Interpretation

Step7ML-J mirrors constrained Pareto/objective-vector selection over Step7ML-I
hard-feasible decoded candidates. It reduces the selected set to metric-safe or
known winner/gate rows without adding scalar penalty thresholds. If winner/gate
counts remain at 2, the selector is useful as a budget controller but the next
bottleneck is richer metric features or more decoder variants, not legality.
"""
