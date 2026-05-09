"""Step7Q-B constrained operator-source risk/ranking smoke.

This module scores only request-time feature fields from the Step7Q-A data mart.
Replay labels are used exclusively after selection to evaluate the selected deck
against the Branch C risk profile.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

from puzzleplace.ml.step7q_operator_learning import (
    LABEL_LEAKAGE_FEATURE_KEYS,
    direct_coordinate_field_count,
    finite_action_schema,
    largest_share,
    load_json,
    read_jsonl,
    validate_no_label_leakage,
    write_json,
    write_jsonl,
)

POLICY_SCHEMA = "step7q_operator_policy_smoke_v1"
SCORE_INPUT_POLICY = "features and masks only; labels are evaluation-only"
DEFAULT_REQUEST_COUNT = 96
DEFAULT_MIN_PER_CASE = 5
DEFAULT_MAX_CASE_SHARE = 0.25


def run_operator_policy_smoke(
    examples_path: Path,
    branch_summary_path: Path,
    scores_out_path: Path,
    deck_out_path: Path,
    summary_out_path: Path,
    markdown_out_path: Path,
    *,
    request_count: int = DEFAULT_REQUEST_COUNT,
    min_per_case: int = DEFAULT_MIN_PER_CASE,
    max_case_share: float = DEFAULT_MAX_CASE_SHARE,
) -> dict[str, Any]:
    examples = read_jsonl(examples_path)
    branch_summary = load_json(branch_summary_path)
    scores = score_operator_examples(examples)
    deck = select_source_deck(
        examples,
        scores,
        request_count=request_count,
        min_per_case=min_per_case,
        max_case_share=max_case_share,
    )
    write_jsonl(scores_out_path, scores)
    write_jsonl(deck_out_path, deck_rows(deck, scores))
    summary = summarize_policy_smoke(
        examples,
        scores,
        deck,
        branch_summary,
        paths={
            "examples_path": examples_path,
            "branch_summary_path": branch_summary_path,
            "scores_path": scores_out_path,
            "deck_path": deck_out_path,
            "summary_path": summary_out_path,
            "markdown_path": markdown_out_path,
        },
        request_count=request_count,
        min_per_case=min_per_case,
        max_case_share=max_case_share,
    )
    write_json(summary_out_path, summary)
    markdown_out_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_out_path.write_text(policy_smoke_markdown(summary), encoding="utf-8")
    return summary


def score_operator_examples(examples: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for example in examples:
        score, components = score_example_features(
            example.get("features", {}), example.get("masks", {})
        )
        rows.append(
            {
                "schema": "step7q_operator_policy_score_v1",
                "example_id": str(example.get("example_id")),
                "case_id": str(example.get("case_id")),
                "source_subproblem_id": str(example.get("source_subproblem_id")),
                "source_candidate_id": str(example.get("source_candidate_id")),
                "eligible_for_policy_selection": bool(
                    example.get("masks", {}).get("eligible_for_training")
                ),
                "policy_score": score,
                "score_components": components,
                "score_input_policy": SCORE_INPUT_POLICY,
            }
        )
    return rows


def score_example_features(
    features: dict[str, Any], masks: dict[str, Any] | None = None
) -> tuple[float, dict[str, float]]:
    """Score an operator source using feature/mask fields only."""

    masks = masks or {}
    components: dict[str, float] = {}
    components["eligible_for_training"] = 0.0 if masks.get("eligible_for_training") else -1000.0
    components["metric_confidence"] = {
        "exact_component_comparable": 2.0,
        "proxy_only": -1.0,
    }.get(str(features.get("metric_confidence")), 0.0)
    components["seed_source"] = {
        "step7m_phase2": 2.0,
        "step7m_phase4": 1.7,
        "step7ml_k": -1.2,
        "step7ml_i": 0.0,
    }.get(str(features.get("seed_source")), 0.0)
    components["intent_family"] = {
        "order_preserving_row_repack": 2.5,
        "hpwl_hull_shrink": 2.0,
        "closure_translate_with_repair": 0.8,
        "mib_shape_preserve_repack": -0.7,
        "soft_guarded_repair": -4.0,
        "blocker_chain_unblock": -5.0,
    }.get(str(features.get("intent_family")), 0.0)
    components["seed_failure_bucket"] = {
        "fb_unknown": 2.0,
        "fb_dominated": 1.2,
        "fb_hpwl_tradeoff": 1.0,
        "fb_soft_risk": -4.0,
        "fb_overlap_risk": -5.0,
        "fb_bbox_risk": -4.0,
    }.get(str(features.get("seed_failure_bucket")), 0.0)
    components["affected_size_penalty"] = -0.015 * min(
        numeric(features.get("affected_block_count")), 100.0
    )
    components["blocker_size_penalty"] = -0.1 * min(
        numeric(features.get("blocker_block_count")), 20.0
    )
    components["soft_link_size_penalty"] = -0.08 * min(
        numeric(features.get("soft_linked_block_count")), 20.0
    )
    components["move_size_penalty"] = -0.01 * min(numeric(features.get("moved_block_count")), 100.0)
    return round(sum(components.values()), 6), components


def select_source_deck(
    examples: list[dict[str, Any]],
    scores: list[dict[str, Any]],
    *,
    request_count: int,
    min_per_case: int,
    max_case_share: float,
) -> list[dict[str, Any]]:
    score_by_id = {str(row["example_id"]): row for row in scores}
    eligible = [
        example
        for example in examples
        if score_by_id[str(example.get("example_id"))]["eligible_for_policy_selection"]
    ]
    ranked = sorted(
        eligible,
        key=lambda example: (
            float(score_by_id[str(example.get("example_id"))]["policy_score"]),
            str(example.get("example_id")),
        ),
        reverse=True,
    )
    case_cap = max(1, int(request_count * max_case_share))
    selected: list[dict[str, Any]] = []
    selected_ids: set[str] = set()
    case_counts: Counter[str] = Counter()
    cases = sorted({str(example.get("case_id")) for example in eligible})

    for case_id in cases:
        need = min_per_case
        for example in ranked:
            example_id = str(example.get("example_id"))
            if str(example.get("case_id")) != case_id or example_id in selected_ids:
                continue
            if need <= 0 or case_counts[case_id] >= case_cap:
                break
            selected.append(example)
            selected_ids.add(example_id)
            case_counts[case_id] += 1
            need -= 1

    for example in ranked:
        if len(selected) >= request_count:
            break
        example_id = str(example.get("example_id"))
        case_id = str(example.get("case_id"))
        if example_id in selected_ids or case_counts[case_id] >= case_cap:
            continue
        selected.append(example)
        selected_ids.add(example_id)
        case_counts[case_id] += 1
    return selected


def deck_rows(deck: list[dict[str, Any]], scores: list[dict[str, Any]]) -> list[dict[str, Any]]:
    score_by_id = {str(row["example_id"]): row for row in scores}
    rows = []
    for index, example in enumerate(deck, start=1):
        score = score_by_id[str(example.get("example_id"))]
        rows.append(
            {
                "schema": "step7q_selected_source_deck_v1",
                "deck_rank": index,
                "example_id": str(example.get("example_id")),
                "case_id": str(example.get("case_id")),
                "source_subproblem_id": str(example.get("source_subproblem_id")),
                "source_candidate_id": str(example.get("source_candidate_id")),
                "seed_source": str(example.get("seed_source")),
                "intent_family": str(example.get("intent_family")),
                "seed_failure_bucket": str(example.get("seed_failure_bucket")),
                "metric_confidence": str(example.get("metric_confidence")),
                "finite_action_schema_version": str(example.get("finite_action_schema_version")),
                "policy_score": score["policy_score"],
                "score_input_policy": SCORE_INPUT_POLICY,
            }
        )
    return rows


def summarize_policy_smoke(
    examples: list[dict[str, Any]],
    scores: list[dict[str, Any]],
    deck: list[dict[str, Any]],
    branch_summary: dict[str, Any],
    *,
    paths: dict[str, Path],
    request_count: int,
    min_per_case: int,
    max_case_share: float,
) -> dict[str, Any]:
    branch_metrics = branch_summary.get("best_branch_metrics", {})
    case_counts = Counter(str(example.get("case_id")) for example in deck)
    n = len(deck)
    forbidden_count = sum(
        int(example.get("masks", {}).get("forbidden_request_term")) for example in deck
    )
    overlap_count = sum(
        int(example.get("labels", {}).get("overlap_after_splice", 0) > 0)
        for example in deck
    )
    soft_count = sum(
        int(bool(example.get("labels", {}).get("soft_regression"))) for example in deck
    )
    bbox_count = sum(
        int(bool(example.get("labels", {}).get("bbox_regression"))) for example in deck
    )
    hard_count = sum(
        int(bool(example.get("labels", {}).get("hard_feasible_nonnoop"))) for example in deck
    )
    all_vector_count = sum(
        int(bool(example.get("labels", {}).get("actual_all_vector_nonregressing")))
        for example in deck
    )
    strict_count = sum(
        int(bool(example.get("labels", {}).get("strict_meaningful_winner"))) for example in deck
    )
    soft_rate = ratio(soft_count, n)
    bbox_rate = ratio(bbox_count, n)
    score_leakage = score_label_leakage_count(scores)
    no_feature_leakage = len(validate_no_label_leakage(examples)) == 0
    representation_pass = (
        n >= request_count
        and len(case_counts) == 8
        and largest_share(case_counts) <= max_case_share
        and forbidden_count == 0
    )
    risk_profile_pass = (
        representation_pass
        and score_leakage == 0
        and no_feature_leakage
        and overlap_count < int(branch_metrics.get("overlap_after_splice_count", 10**9))
        and soft_rate < float(branch_metrics.get("soft_regression_rate", 1.0))
        and bbox_rate == 0.0
    )
    strict_gate_pass = strict_count >= 3
    if risk_profile_pass and strict_gate_pass:
        decision = "promote_to_phase3_replay_candidate"
        allowed_next_phase: str | None = "step7q_replay_selected_policy_deck"
        next_recommendation = "replay_selected_deck_then_recheck_phase4_gate"
    elif risk_profile_pass:
        decision = "risk_ranking_smoke_pass_strict_gate_closed"
        allowed_next_phase = "step7q_constrained_operator_parameter_expansion"
        next_recommendation = "use_risk_filter_as_guard_then_generate_new_strict-winner candidates"
    else:
        decision = "stop_policy_smoke_gate"
        allowed_next_phase = None
        next_recommendation = "revise_feature_scorer_or_data_mart_before_replay"

    return {
        "schema": POLICY_SCHEMA,
        "decision": decision,
        **{name: str(path) for name, path in paths.items()},
        "score_input_policy": SCORE_INPUT_POLICY,
        "request_count_target": request_count,
        "min_per_case": min_per_case,
        "max_case_share_target": max_case_share,
        "selected_request_count": n,
        "represented_case_count": len(case_counts),
        "case_counts": dict(case_counts),
        "largest_case_share": largest_share(case_counts),
        "forbidden_request_term_count": forbidden_count,
        "score_feature_label_leakage_count": score_leakage,
        "data_mart_feature_label_leakage_count": 0
        if no_feature_leakage
        else len(validate_no_label_leakage(examples)),
        "direct_coordinate_field_count": direct_coordinate_field_count(finite_action_schema()),
        "hard_feasible_nonnoop_count": hard_count,
        "overlap_after_splice_count": overlap_count,
        "soft_regression_count": soft_count,
        "soft_regression_rate": soft_rate,
        "bbox_regression_count": bbox_count,
        "bbox_regression_rate": bbox_rate,
        "actual_all_vector_nonregressing_count": all_vector_count,
        "strict_meaningful_winner_count": strict_count,
        "branch_c_baseline_name": branch_summary.get("best_branch_name"),
        "branch_c_baseline_metrics": branch_metrics,
        "representation_pass": representation_pass,
        "risk_profile_pass": risk_profile_pass,
        "strict_gate_pass": strict_gate_pass,
        "phase4_gate_open": bool(risk_profile_pass and strict_gate_pass),
        "gnn_rl_gate_open": True,
        "allowed_next_phase": allowed_next_phase,
        "next_recommendation": next_recommendation,
    }


def score_label_leakage_count(scores: list[dict[str, Any]]) -> int:
    leakage = 0
    for row in scores:
        text = json.dumps(
            {
                "score_components": row.get("score_components"),
                "score_input_policy": row.get("score_input_policy"),
            },
            sort_keys=True,
        )
        leakage += sum(int(key in text) for key in LABEL_LEAKAGE_FEATURE_KEYS)
    return leakage


def policy_smoke_markdown(summary: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# Step7Q Operator Policy Smoke Summary",
            "",
            f"Decision: `{summary['decision']}`",
            "",
            f"- selected_request_count: {summary['selected_request_count']}",
            f"- represented_case_count: {summary['represented_case_count']}",
            f"- largest_case_share: {summary['largest_case_share']}",
            f"- forbidden_request_term_count: {summary['forbidden_request_term_count']}",
            f"- score_feature_label_leakage_count: "
            f"{summary['score_feature_label_leakage_count']}",
            f"- hard_feasible_nonnoop_count: {summary['hard_feasible_nonnoop_count']}",
            f"- overlap_after_splice_count: {summary['overlap_after_splice_count']}",
            f"- soft_regression_rate: {summary['soft_regression_rate']}",
            f"- bbox_regression_rate: {summary['bbox_regression_rate']}",
            f"- actual_all_vector_nonregressing_count: "
            f"{summary['actual_all_vector_nonregressing_count']}",
            f"- strict_meaningful_winner_count: {summary['strict_meaningful_winner_count']}",
            f"- representation_pass: {summary['representation_pass']}",
            f"- risk_profile_pass: {summary['risk_profile_pass']}",
            f"- strict_gate_pass: {summary['strict_gate_pass']}",
            f"- phase4_gate_open: {summary['phase4_gate_open']}",
            f"- allowed_next_phase: {summary['allowed_next_phase']}",
            "",
        ]
    )


def numeric(value: Any) -> float:
    if isinstance(value, bool) or value is None:
        return 0.0
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def ratio(count: int, total: int) -> float:
    return 0.0 if total == 0 else count / total


__all__ = [
    "run_operator_policy_smoke",
    "score_example_features",
    "score_operator_examples",
    "select_source_deck",
]
