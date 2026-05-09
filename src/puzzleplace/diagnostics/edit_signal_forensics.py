from __future__ import annotations

import math
from collections import Counter, defaultdict
from statistics import median
from typing import Any

SOURCE_B = "step7c_real_b"
SOURCE_C = "step7c_real_c"

DECISIONS = {
    "refine_slot_scoring",
    "expand_slack_fit_local_lane",
    "build_route_specific_legalizers",
    "pivot_to_coarse_region_planner",
    "pivot_to_macro_closure_generator",
    "build_training_retrieval_prior",
    "revisit_official_metric_model",
    "inconclusive_due_to_artifact_gaps",
}


def unified_candidate_table(
    step7c_real_b_rows: list[dict[str, Any]],
    step7c_real_c_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    rows.extend(_normalize_row(row, SOURCE_B) for row in step7c_real_b_rows)
    rows.extend(_normalize_row(row, SOURCE_C) for row in step7c_real_c_rows)
    return rows


def _normalize_row(row: dict[str, Any], source_step: str) -> dict[str, Any]:
    soft = row.get("mib_group_boundary_soft_delta") or {}
    hard_feasible = bool(row.get("after_route_official_like_hard_feasible", False))
    official_delta = _float(row.get("official_like_cost_delta"))
    hpwl_delta = _float(row.get("hpwl_delta"))
    bbox_delta = _float(row.get("bbox_area_delta"))
    soft_delta = _float(soft.get("total_delta"))
    internal_best = _maybe_float(row.get("internal_best_official_like_cost_delta"))
    window_count = _maybe_int(row.get("window_block_count"))
    changed_count = _int(row.get("changed_block_count"))
    slot_assignment_raw = row.get("slot_assignment")
    slot_assignment: dict[Any, Any] = (
        slot_assignment_raw if isinstance(slot_assignment_raw, dict) else {}
    )
    class_label = _candidate_class(row)
    return {
        "source_step": source_step,
        "case_id": _int(row.get("case_id")),
        "candidate_id": str(row.get("candidate_id", "")),
        "strategy": str(row.get("strategy", row.get("family", "unknown"))),
        "route_class": str(row.get("actual_locality_class", "unknown")),
        "descriptor_class": str(row.get("descriptor_locality_class", "unknown")),
        "route_lane": str(row.get("route_lane", "unknown")),
        "route_collapsed": bool(
            row.get("descriptor_locality_class") != "global"
            and row.get("actual_locality_class") == "global"
        ),
        "report_only": bool(row.get("report_only", False)),
        "feasible": hard_feasible,
        "official_eval_available": bool(row.get("official_eval_available", False)),
        "no_op": bool(row.get("no_op", False)),
        "safe_improvement": bool(row.get("safe_improvement", False)),
        "official_like_improving": bool(row.get("official_like_cost_improving", False)),
        "candidate_class": class_label,
        "changed_block_count": changed_count,
        "changed_block_fraction": _float(row.get("changed_block_fraction")),
        "window_block_count": window_count,
        "macro_closure_size": _int(row.get("macro_closure_size")),
        "affected_region_count": _int(row.get("affected_region_count")),
        "free_space_fit_ratio": _float(row.get("free_space_fit_ratio")),
        "internal_trial_count": _int(row.get("internal_trial_count")),
        "internal_feasible_trial_count": _int(row.get("internal_feasible_trial_count")),
        "internal_feasible_rate": _rate(
            _int(row.get("internal_feasible_trial_count")), _int(row.get("internal_trial_count"))
        ),
        "construction_score_delta": internal_best,
        "predicted_pressure_score": -internal_best if internal_best is not None else None,
        "slot_assignment_count": len(slot_assignment),
        "has_slot_assignment": bool(slot_assignment),
        "hpwl_delta": hpwl_delta,
        "bbox_area_delta": bbox_delta,
        "soft_constraint_delta": soft_delta,
        "official_like_cost_delta": official_delta,
        "official_like_hpwl_gap_delta": _float(row.get("official_like_hpwl_gap_delta")),
        "official_like_area_gap_delta": _float(row.get("official_like_area_gap_delta")),
        "official_like_violation_delta": _float(row.get("official_like_violation_delta")),
        "hpwl_gain": hpwl_delta < -1e-9,
        "bbox_penalty": bbox_delta > 1e-9,
        "soft_penalty": soft_delta > 1e-9,
        "failure_attribution": str(row.get("failure_attribution", "unknown")),
        "failure_family": _failure_family(str(row.get("failure_attribution", "unknown")), row),
        "mib_group_boundary_involved": bool(
            _int(soft.get("mib_delta"))
            or _int(soft.get("grouping_delta"))
            or _int(soft.get("boundary_delta"))
            or _int(row.get("macro_closure_size"))
        ),
        "raw_missing_fields": [],
    }


def _candidate_class(row: dict[str, Any]) -> str:
    if not bool(row.get("after_route_official_like_hard_feasible", False)):
        return "infeasible"
    if bool(row.get("no_op", False)):
        return "no_op"
    if (
        row.get("descriptor_locality_class") != "global"
        and row.get("actual_locality_class") == "global"
    ):
        return "route_collapsed"
    if bool(row.get("official_like_cost_improving", False)):
        return "official_like_improving"
    if bool(row.get("safe_improvement", False)):
        return "safe_proxy_improving"
    return "feasible_non_improving"


def _failure_family(name: str, row: dict[str, Any]) -> str:
    if name == "none":
        return "none"
    if "global_report" in name:
        return "global_report_only"
    if "route_collapse" in name or "collapsed_to_global" in name:
        return "route_collapse_globality"
    if "no_feasible" in name or "slack_slot" in name:
        return "no_feasible_slack"
    if "repack" in name:
        return "repack_or_placement_limitation"
    if "legalizer" in name or "hard_infeasible" in name or "overlap" in name:
        return "legalizer_limitation"
    if "tradeoff" in name or (
        bool(row.get("safe_improvement", False))
        and not bool(row.get("official_like_cost_improving", False))
    ):
        return "metric_tradeoff_failure"
    if bool(row.get("no_op", False)):
        return "poor_target_block_selection"
    if "target" in name or "non_improving" in name:
        return "poor_target_block_selection"
    return "unknown_or_artifact_specific"


def summary_counts(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "candidate_count_by_source_step": dict(Counter(row["source_step"] for row in rows)),
        "candidate_count_by_strategy": _counter_by(rows, "strategy"),
        "feasible_count_by_strategy": _counter_by(
            [row for row in rows if row["feasible"]], "strategy"
        ),
        "improving_count_by_strategy": _counter_by(
            [row for row in rows if row["safe_improvement"]], "strategy"
        ),
        "official_like_improving_count_by_strategy": _counter_by(
            [row for row in rows if row["official_like_improving"]], "strategy"
        ),
        "no_op_count_by_strategy": _counter_by([row for row in rows if row["no_op"]], "strategy"),
        "infeasible_count_by_strategy": _counter_by(
            [row for row in rows if not row["feasible"]], "strategy"
        ),
        "improvement_density_by_strategy": _density_by_strategy(rows, "safe_improvement"),
        "official_like_improvement_density_by_strategy": _density_by_strategy(
            rows, "official_like_improving"
        ),
        "route_class_vs_improvement_summary": _route_class_summary(rows),
        "changed_block_count_vs_improvement_summary": _bucket_summary(
            rows, "changed_block_count", buckets=(0, 1, 2, 4, 8, 999999)
        ),
        "window_size_vs_improvement_summary": _bucket_summary(
            [row for row in rows if row["window_block_count"] is not None],
            "window_block_count",
            buckets=(0, 1, 2, 4, 8, 999999),
        ),
    }


def feature_delta_correlation(rows: list[dict[str, Any]]) -> dict[str, Any]:
    features = [
        "hpwl_delta",
        "bbox_area_delta",
        "soft_constraint_delta",
        "predicted_pressure_score",
        "construction_score_delta",
        "internal_feasible_rate",
        "changed_block_count",
        "window_block_count",
        "slot_assignment_count",
    ]
    overall = {
        feature: _correlation(rows, feature, "official_like_cost_delta") for feature in features
    }
    by_source: dict[str, dict[str, Any]] = {}
    for source, group in _group_by(rows, "source_step").items():
        by_source[source] = {
            feature: _correlation(group, feature, "official_like_cost_delta")
            for feature in features
        }
    return {
        "hpwl_delta_vs_official_like_delta_summary": overall["hpwl_delta"],
        "bbox_delta_vs_official_like_delta_summary": overall["bbox_area_delta"],
        "soft_delta_vs_official_like_delta_summary": overall["soft_constraint_delta"],
        "predicted_pressure_vs_actual_delta_summary": overall["predicted_pressure_score"],
        "slot_score_vs_actual_delta_summary": overall["construction_score_delta"],
        "all_feature_correlations": overall,
        "by_source_step": by_source,
        "interpretation": _correlation_interpretation(overall),
    }


def winner_loser_examples(rows: list[dict[str, Any]], *, limit: int = 8) -> dict[str, Any]:
    candidates = [row for row in rows if row["strategy"] != "original_layout"]
    positives = sorted(
        candidates, key=lambda row: (row["official_like_cost_delta"], row["candidate_id"])
    )
    negatives = sorted(
        candidates,
        key=lambda row: (-row["official_like_cost_delta"], row["candidate_id"]),
    )
    by_strategy: dict[str, dict[str, Any]] = {}
    for strategy, group in _group_by(candidates, "strategy").items():
        by_strategy[strategy] = {
            "best": [
                _example_row(row)
                for row in sorted(group, key=lambda r: r["official_like_cost_delta"])[:2]
            ],
            "worst": [
                _example_row(row)
                for row in sorted(group, key=lambda r: -r["official_like_cost_delta"])[:2]
            ],
        }
    return {
        "top_positive_examples": [_example_row(row) for row in positives[:limit]],
        "top_negative_examples": [_example_row(row) for row in negatives[:limit]],
        "per_strategy_examples": by_strategy,
    }


def tradeoff_report(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "overall": _tradeoff_summary(rows),
        "by_source_step": {
            source: _tradeoff_summary(group)
            for source, group in _group_by(rows, "source_step").items()
        },
        "by_strategy": {
            strategy: _tradeoff_summary(group)
            for strategy, group in _group_by(rows, "strategy").items()
        },
        "hpwl_gain_but_not_official_improving_examples": [
            _example_row(row)
            for row in sorted(
                [row for row in rows if row["hpwl_gain"] and not row["official_like_improving"]],
                key=lambda r: (r["official_like_cost_delta"], r["candidate_id"]),
            )[:10]
        ],
    }


def slot_scoring_calibration(rows: list[dict[str, Any]]) -> dict[str, Any]:
    scored = [row for row in rows if row["construction_score_delta"] is not None]
    by_source = {
        source: _slot_score_summary(group)
        for source, group in _group_by(scored, "source_step").items()
    }
    by_strategy = {
        strategy: _slot_score_summary(group)
        for strategy, group in _group_by(scored, "strategy").items()
    }
    return {
        "scored_candidate_count": len(scored),
        "missing_slot_score_count": len(rows) - len(scored),
        "overall": _slot_score_summary(scored),
        "by_source_step": by_source,
        "by_strategy": by_strategy,
        "calibration_notes": [
            "construction_score_delta is the sidecar internal best official-like score delta, "
            "not an independent learned predictor.",
            "Step7C-real-B lacks explicit slot_assignment/window margin fields; Step7C-real-C "
            "has slot_assignment_count but not exact free-space margin.",
        ],
    }


def failure_taxonomy(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "failure_family_counts": dict(Counter(row["failure_family"] for row in rows)),
        "failure_family_by_source_step": {
            source: dict(Counter(row["failure_family"] for row in group))
            for source, group in _group_by(rows, "source_step").items()
        },
        "failure_family_by_strategy": {
            strategy: dict(Counter(row["failure_family"] for row in group))
            for strategy, group in _group_by(rows, "strategy").items()
        },
        "candidate_class_counts": dict(Counter(row["candidate_class"] for row in rows)),
        "decision_relevant_counts": {
            "legalizer_limited_count": sum(
                int(row["failure_family"] == "legalizer_limitation") for row in rows
            ),
            "metric_tradeoff_failure_count": sum(
                int(row["failure_family"] == "metric_tradeoff_failure") for row in rows
            ),
            "poor_target_block_selection_count": sum(
                int(row["failure_family"] == "poor_target_block_selection") for row in rows
            ),
            "macro_or_region_rows": sum(
                int(row["route_class"] in {"macro", "regional"}) for row in rows
            ),
            "macro_or_region_official_improving": sum(
                int(row["route_class"] in {"macro", "regional"} and row["official_like_improving"])
                for row in rows
            ),
        },
    }


def missing_field_report(rows: list[dict[str, Any]]) -> dict[str, Any]:
    required = [
        "source_step",
        "case_id",
        "strategy",
        "route_class",
        "feasible",
        "hpwl_delta",
        "bbox_area_delta",
        "soft_constraint_delta",
        "official_like_cost_delta",
        "failure_attribution",
    ]
    optional = [
        "window_block_count",
        "slot_assignment_count",
        "construction_score_delta",
        "predicted_pressure_score",
        "free_space_fit_ratio",
        "macro_closure_size",
    ]
    return {
        "required_missing_counts": _missing_counts(rows, required),
        "optional_missing_counts": _missing_counts(rows, optional),
        "source_specific_optional_missing_counts": {
            source: _missing_counts(group, optional)
            for source, group in _group_by(rows, "source_step").items()
        },
        "known_artifact_gaps": [
            "B/C route artifacts do not include before/after coordinates, "
            "so displacement magnitude "
            "is approximated only by changed_block_count/fraction.",
            "Step7C-real-B artifacts do not include slot assignments or window block ids.",
            "Neither B nor C artifacts include exact free-space margin for selected slots.",
            "Predicted pressure is reconstructed conservatively from "
            "construction_score_delta when present.",
        ],
    }


def decision_for_step7c_real_d(
    rows: list[dict[str, Any]],
    summaries: dict[str, Any],
    correlations: dict[str, Any],
    failures: dict[str, Any],
    missing: dict[str, Any],
) -> str:
    if any(missing["required_missing_counts"].values()):
        return "inconclusive_due_to_artifact_gaps"
    if sum(int(row["official_eval_available"]) for row in rows) < len(rows) * 0.95:
        return "revisit_official_metric_model"
    decision_counts = failures["decision_relevant_counts"]
    non_global = [row for row in rows if row["route_class"] != "global"]
    legalizer_rate = _rate(int(decision_counts["legalizer_limited_count"]), len(non_global))
    if legalizer_rate > 0.25:
        return "build_route_specific_legalizers"
    macro_rows = int(decision_counts["macro_or_region_rows"])
    macro_improving = int(decision_counts["macro_or_region_official_improving"])
    if macro_rows >= 20 and macro_improving >= 3:
        return "pivot_to_macro_closure_generator"
    source_counts = summaries["candidate_count_by_source_step"]
    b_rows = [row for row in rows if row["source_step"] == SOURCE_B]
    c_rows = [row for row in rows if row["source_step"] == SOURCE_C]
    b_official = sum(int(row["official_like_improving"]) for row in b_rows)
    c_official = sum(int(row["official_like_improving"]) for row in c_rows)
    b_density = _rate(b_official, sum(int(row["feasible"]) for row in b_rows))
    c_density = _rate(c_official, sum(int(row["feasible"]) for row in c_rows))
    b_local_winners = [
        row
        for row in b_rows
        if row["official_like_improving"]
        and row["strategy"] in {"slack_fit_insertion", "hpwl_directed_local_nudge"}
    ]
    hpwl_corr = correlations["hpwl_delta_vs_official_like_delta_summary"].get("pearson", 0.0)
    if len(b_local_winners) >= 2 and c_density < b_density:
        return "expand_slack_fit_local_lane"
    if abs(float(hpwl_corr or 0.0)) >= 0.25 and c_density < b_density:
        return "refine_slot_scoring"
    if (
        source_counts.get(SOURCE_B, 0)
        and source_counts.get(SOURCE_C, 0)
        and b_official == c_official == 0
    ):
        return "pivot_to_coarse_region_planner"
    if abs(float(hpwl_corr or 0.0)) < 0.10 and b_official <= 1 and c_official <= 1:
        return "build_training_retrieval_prior"
    return "refine_slot_scoring"


def _tradeoff_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    hpwl_gain = [row for row in rows if row["hpwl_gain"]]
    official = [row for row in rows if row["official_like_improving"]]
    cancelled_bbox = [
        row for row in hpwl_gain if not row["official_like_improving"] and row["bbox_penalty"]
    ]
    cancelled_soft = [
        row for row in hpwl_gain if not row["official_like_improving"] and row["soft_penalty"]
    ]
    return {
        "candidate_count": len(rows),
        "hpwl_gain_count": len(hpwl_gain),
        "official_like_improving_count": len(official),
        "hpwl_gain_to_official_improve_rate": _rate(
            sum(int(row["official_like_improving"]) for row in hpwl_gain), len(hpwl_gain)
        ),
        "hpwl_gain_cancelled_by_bbox_count": len(cancelled_bbox),
        "hpwl_gain_cancelled_by_soft_count": len(cancelled_soft),
        "hpwl_gain_but_official_worse_or_flat_count": len(hpwl_gain)
        - sum(int(row["official_like_improving"]) for row in hpwl_gain),
        "bbox_penalty_count": sum(int(row["bbox_penalty"]) for row in rows),
        "soft_penalty_count": sum(int(row["soft_penalty"]) for row in rows),
        "median_official_like_cost_delta": _median(row["official_like_cost_delta"] for row in rows),
        "median_hpwl_delta": _median(row["hpwl_delta"] for row in rows),
    }


def _slot_score_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"candidate_count": 0, "top_quartile_hit_rate": 0.0, "pearson": None}
    sorted_rows = sorted(
        rows,
        key=lambda row: (
            float("inf")
            if row["construction_score_delta"] is None
            else row["construction_score_delta"],
            row["candidate_id"],
        ),
    )
    k = max(1, math.ceil(len(sorted_rows) * 0.25))
    top = sorted_rows[:k]
    return {
        "candidate_count": len(rows),
        "top_quartile_size": k,
        "top_quartile_official_like_improving_count": sum(
            int(row["official_like_improving"]) for row in top
        ),
        "top_quartile_hit_rate": _rate(sum(int(row["official_like_improving"]) for row in top), k),
        "overall_hit_rate": _rate(
            sum(int(row["official_like_improving"]) for row in rows), len(rows)
        ),
        "correlation": _correlation(rows, "construction_score_delta", "official_like_cost_delta"),
    }


def _correlation(rows: list[dict[str, Any]], x_key: str, y_key: str) -> dict[str, Any]:
    pairs = []
    for row in rows:
        x = row.get(x_key)
        y = row.get(y_key)
        if x is None or y is None:
            continue
        pairs.append((float(x), float(y)))
    if len(pairs) < 2:
        return {
            "n": len(pairs),
            "pearson": None,
            "spearman": None,
            "missing": len(rows) - len(pairs),
        }
    xs = [x for x, _y in pairs]
    ys = [y for _x, y in pairs]
    return {
        "n": len(pairs),
        "pearson": _pearson(xs, ys),
        "spearman": _pearson(_ranks(xs), _ranks(ys)),
        "missing": len(rows) - len(pairs),
        "x_distribution": _distribution(xs),
        "y_distribution": _distribution(ys),
    }


def _correlation_interpretation(overall: dict[str, Any]) -> list[str]:
    notes = []
    hpwl = overall["hpwl_delta"].get("pearson")
    bbox = overall["bbox_area_delta"].get("pearson")
    soft = overall["soft_constraint_delta"].get("pearson")
    if hpwl is not None:
        notes.append(
            f"HPWL delta has pearson={hpwl:.3f} vs official-like delta; "
            "lower HPWL generally helps when negative correlation appears."
        )
    if bbox is not None:
        notes.append(
            f"bbox delta has pearson={bbox:.3f}; positive correlation means "
            "bbox growth cancels local gains."
        )
    if soft is not None:
        notes.append(
            f"soft delta has pearson={soft:.3f}; positive correlation means "
            "MIB/group/boundary penalties matter."
        )
    return notes


def _route_class_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    out = {}
    for route_class, group in _group_by(rows, "route_class").items():
        feasible = [row for row in group if row["feasible"]]
        out[route_class] = {
            "candidate_count": len(group),
            "feasible_count": len(feasible),
            "official_like_improving_count": sum(
                int(row["official_like_improving"]) for row in feasible
            ),
            "official_like_improvement_density": _rate(
                sum(int(row["official_like_improving"]) for row in feasible), len(feasible)
            ),
            "median_official_like_cost_delta": _median(
                row["official_like_cost_delta"] for row in group
            ),
        }
    return out


def _bucket_summary(
    rows: list[dict[str, Any]], key: str, *, buckets: tuple[int, ...]
) -> dict[str, Any]:
    labels: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        value = row.get(key)
        if value is None:
            labels["missing"].append(row)
            continue
        value_i = int(value)
        prev = 0
        placed = False
        for boundary in buckets:
            if value_i <= boundary:
                label = f"{prev}..{boundary}" if prev != boundary else str(boundary)
                labels[label].append(row)
                placed = True
                break
            prev = boundary + 1
        if not placed:
            labels[f">{buckets[-1]}"].append(row)
    return {
        label: {
            "candidate_count": len(group),
            "official_like_improving_count": sum(
                int(row["official_like_improving"]) for row in group
            ),
            "median_official_like_cost_delta": _median(
                row["official_like_cost_delta"] for row in group
            ),
        }
        for label, group in sorted(labels.items())
    }


def _density_by_strategy(rows: list[dict[str, Any]], key: str) -> dict[str, float]:
    out: dict[str, float] = {}
    for strategy, group in _group_by(rows, "strategy").items():
        feasible = [row for row in group if row["feasible"]]
        out[strategy] = _rate(sum(int(row[key]) for row in feasible), len(feasible))
    return out


def _counter_by(rows: list[dict[str, Any]], key: str) -> dict[str, int]:
    return dict(Counter(str(row.get(key, "missing")) for row in rows))


def _group_by(rows: list[dict[str, Any]], key: str) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        out[str(row.get(key, "missing"))].append(row)
    return dict(sorted(out.items()))


def _example_row(row: dict[str, Any]) -> dict[str, Any]:
    keys = [
        "source_step",
        "case_id",
        "candidate_id",
        "strategy",
        "route_class",
        "candidate_class",
        "official_like_cost_delta",
        "hpwl_delta",
        "bbox_area_delta",
        "soft_constraint_delta",
        "changed_block_count",
        "window_block_count",
        "construction_score_delta",
        "failure_family",
        "failure_attribution",
    ]
    return {key: row.get(key) for key in keys}


def _missing_counts(rows: list[dict[str, Any]], keys: list[str]) -> dict[str, int]:
    return {key: sum(int(row.get(key) is None) for row in rows) for key in keys}


def _pearson(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) != len(ys) or len(xs) < 2:
        return None
    mx = sum(xs) / len(xs)
    my = sum(ys) / len(ys)
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys, strict=False))
    den_x = math.sqrt(sum((x - mx) ** 2 for x in xs))
    den_y = math.sqrt(sum((y - my) ** 2 for y in ys))
    if den_x <= 1e-12 or den_y <= 1e-12:
        return None
    return num / (den_x * den_y)


def _ranks(values: list[float]) -> list[float]:
    ordered = sorted((value, idx) for idx, value in enumerate(values))
    ranks = [0.0] * len(values)
    i = 0
    while i < len(ordered):
        j = i
        while j + 1 < len(ordered) and ordered[j + 1][0] == ordered[i][0]:
            j += 1
        rank = (i + j) / 2.0 + 1.0
        for _value, idx in ordered[i : j + 1]:
            ranks[idx] = rank
        i = j + 1
    return ranks


def _distribution(values: list[float]) -> dict[str, Any]:
    vals = [float(v) for v in values]
    if not vals:
        return {"count": 0, "min": 0.0, "median": 0.0, "max": 0.0, "mean": 0.0}
    return {
        "count": len(vals),
        "min": min(vals),
        "median": median(vals),
        "max": max(vals),
        "mean": sum(vals) / len(vals),
    }


def _median(values: Any) -> float:
    vals = [float(v) for v in values]
    return median(vals) if vals else 0.0


def _float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _maybe_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _maybe_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _rate(count: int, total: int) -> float:
    return count / total if total > 0 else 0.0
