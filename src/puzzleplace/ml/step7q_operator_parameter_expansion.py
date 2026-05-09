"""Step7Q-C constrained operator-parameter expansion.

This phase expands the Step7Q-B low-risk source deck into finite, guarded
operator-action variants.  It does not execute geometry replay and therefore
cannot claim strict winners; it only prepares a non-leaky bounded deck for the
next fresh-metric executor/replay step.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

from puzzleplace.ml.step7q_operator_learning import (
    direct_coordinate_field_count,
    finite_action_schema,
    largest_share,
    load_json,
    read_jsonl,
    write_json,
    write_jsonl,
)
from puzzleplace.ml.step7q_operator_policy_smoke import SCORE_INPUT_POLICY

EXPANSION_SCHEMA = "step7q_operator_parameter_expansion_v1"
LABEL_USAGE_POLICY = "parent replay labels used only for aggregate gap diagnostics"
FRESH_METRIC_STATUS = "not_executed_requires_fresh_metric_replay"
DEFAULT_TARGET_COUNT = 96
DEFAULT_MIN_PER_CASE = 5
DEFAULT_MAX_CASE_SHARE = 0.25
STRICT_COST_EPS = 1e-7
FORBIDDEN_ACTION_TERMS = ("micro_axis_corridor", "soft_repair_budgeted", "hpwl_only")


def run_operator_parameter_expansion(
    examples_path: Path,
    selected_deck_path: Path,
    policy_summary_path: Path,
    candidates_out_path: Path,
    deck_out_path: Path,
    summary_out_path: Path,
    markdown_out_path: Path,
    *,
    target_count: int = DEFAULT_TARGET_COUNT,
    min_per_case: int = DEFAULT_MIN_PER_CASE,
    max_case_share: float = DEFAULT_MAX_CASE_SHARE,
) -> dict[str, Any]:
    examples = {row["example_id"]: row for row in read_jsonl(examples_path)}
    selected_deck = read_jsonl(selected_deck_path)
    policy_summary = load_json(policy_summary_path)
    candidates = expand_candidates(examples, selected_deck)
    expansion_deck = select_expansion_deck(
        candidates,
        target_count=target_count,
        min_per_case=min_per_case,
        max_case_share=max_case_share,
    )
    write_jsonl(candidates_out_path, candidates)
    write_jsonl(deck_out_path, expansion_deck)
    summary = summarize_expansion(
        examples,
        selected_deck,
        candidates,
        expansion_deck,
        policy_summary,
        paths={
            "examples_path": examples_path,
            "selected_source_deck_path": selected_deck_path,
            "policy_summary_path": policy_summary_path,
            "candidates_path": candidates_out_path,
            "deck_path": deck_out_path,
            "summary_path": summary_out_path,
            "markdown_path": markdown_out_path,
        },
        target_count=target_count,
        min_per_case=min_per_case,
        max_case_share=max_case_share,
    )
    write_json(summary_out_path, summary)
    markdown_out_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_out_path.write_text(expansion_markdown(summary), encoding="utf-8")
    return summary


def expand_candidates(
    examples: dict[str, dict[str, Any]], selected_deck: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for parent in selected_deck:
        example = examples[str(parent["example_id"])]
        features = example.get("features", {})
        for action_index, action in enumerate(action_variants(features)):
            action_score, components = score_action_variant(action, features)
            parent_score = float(parent.get("policy_score", 0.0))
            total_score = round(parent_score + action_score, 6)
            candidates.append(
                {
                    "schema": "step7q_parameter_expansion_candidate_v1",
                    "candidate_id": f"step7q_cand_{len(candidates):05d}",
                    "parent_rank": int(parent.get("deck_rank", 0)),
                    "parent_example_id": str(parent["example_id"]),
                    "case_id": str(parent["case_id"]),
                    "source_subproblem_id": str(parent["source_subproblem_id"]),
                    "source_candidate_id": str(parent["source_candidate_id"]),
                    "operator_action": action,
                    "action_variant_index": action_index,
                    "parent_policy_score": parent_score,
                    "action_score": action_score,
                    "expansion_score": total_score,
                    "score_components": components,
                    "score_input_policy": SCORE_INPUT_POLICY,
                    "fresh_metric_status": FRESH_METRIC_STATUS,
                }
            )
    return candidates


def action_variants(features: dict[str, Any]) -> list[dict[str, Any]]:
    family = str(features.get("intent_family"))
    closure_policy = closure_policy_for(features)
    bbox_guard = (
        "shrink_only"
        if family in {"order_preserving_row_repack", "hpwl_hull_shrink"}
        else "nonexpand"
    )
    vector_guard = "all_vector_nonregress"
    directions = direction_bins_for(features)
    magnitudes = magnitude_bins_for(features)
    variants = []
    for direction in directions:
        for magnitude in magnitudes:
            variants.append(
                {
                    "operator_family": family,
                    "closure_policy": closure_policy,
                    "blocker_chain_depth_bin": blocker_depth_bin(features),
                    "direction_bin": direction,
                    "magnitude_bin": magnitude,
                    "bbox_guard_mode": bbox_guard,
                    "vector_guard_mode": vector_guard,
                    "direct_coordinate_fields": [],
                }
            )
    return variants


def closure_policy_for(features: dict[str, Any]) -> str:
    blockers = int_value(features.get("blocker_block_count"))
    soft = int_value(features.get("soft_linked_block_count"))
    if blockers > 0 and soft > 0:
        return "affected_plus_blockers_plus_soft_linked"
    if blockers > 0:
        return "affected_plus_blockers"
    if soft > 0:
        return "affected_plus_soft_linked"
    return "affected_only"


def blocker_depth_bin(features: dict[str, Any]) -> str:
    blockers = int_value(features.get("blocker_block_count"))
    if blockers <= 0:
        return "0"
    if blockers == 1:
        return "1"
    if blockers == 2:
        return "2"
    return "bounded_all"


def direction_bins_for(features: dict[str, Any]) -> list[str]:
    family = str(features.get("intent_family"))
    failure = str(features.get("seed_failure_bucket"))
    if family == "order_preserving_row_repack":
        return ["slack_fill_left", "slack_fill_right", "bbox_shrink_x"]
    if family == "closure_translate_with_repair":
        return ["hpwl_sink_toward_pins", "slack_fill_left", "slack_fill_right"]
    if family == "mib_shape_preserve_repack":
        return ["bbox_shrink_x", "bbox_shrink_y"]
    if family == "soft_guarded_repair" or failure == "fb_soft_risk":
        return ["soft_release", "hpwl_sink_toward_pins"]
    if family == "blocker_chain_unblock":
        return ["blocker_unblock", "slack_fill_up"]
    return ["hpwl_sink_toward_pins", "bbox_shrink_x"]


def magnitude_bins_for(features: dict[str, Any]) -> list[str]:
    affected = int_value(features.get("affected_block_count"))
    if affected >= 20:
        return ["tiny", "small"]
    return ["small", "medium"]


def score_action_variant(
    action: dict[str, Any], features: dict[str, Any]
) -> tuple[float, dict[str, float]]:
    components: dict[str, float] = {}
    components["vector_guard"] = (
        1.0 if action["vector_guard_mode"] == "all_vector_nonregress" else 0.0
    )
    components["bbox_guard"] = (
        0.4 if action["bbox_guard_mode"] in {"nonexpand", "shrink_only"} else 0.0
    )
    components["magnitude"] = {"tiny": 0.3, "small": 0.2, "medium": -0.1, "slack_limited": 0.1}.get(
        str(action["magnitude_bin"]), 0.0
    )
    components["direction"] = {
        "hpwl_sink_toward_pins": 0.5,
        "bbox_shrink_x": 0.35,
        "bbox_shrink_y": 0.35,
        "slack_fill_left": 0.25,
        "slack_fill_right": 0.25,
        "slack_fill_up": 0.15,
        "slack_fill_down": 0.15,
        "soft_release": -0.2,
        "blocker_unblock": -0.3,
    }.get(str(action["direction_bin"]), 0.0)
    components["closure_scope"] = -0.15 * max(
        0, len(str(action["closure_policy"]).split("plus")) - 1
    )
    components["large_affected_penalty"] = -0.005 * min(
        int_value(features.get("affected_block_count")), 80
    )
    return round(sum(components.values()), 6), components


def select_expansion_deck(
    candidates: list[dict[str, Any]],
    *,
    target_count: int,
    min_per_case: int,
    max_case_share: float,
) -> list[dict[str, Any]]:
    ranked = sorted(
        candidates,
        key=lambda row: (float(row["expansion_score"]), str(row["candidate_id"])),
        reverse=True,
    )
    case_cap = max(1, int(target_count * max_case_share))
    selected: list[dict[str, Any]] = []
    selected_ids: set[str] = set()
    parent_counts: Counter[str] = Counter()
    case_counts: Counter[str] = Counter()
    cases = sorted({str(row["case_id"]) for row in candidates})
    for case_id in cases:
        need = min_per_case
        for row in ranked:
            if need <= 0 or case_counts[case_id] >= case_cap:
                break
            if str(row["case_id"]) != case_id or str(row["candidate_id"]) in selected_ids:
                continue
            if parent_counts[str(row["parent_example_id"])] >= 2:
                continue
            selected.append(row)
            selected_ids.add(str(row["candidate_id"]))
            parent_counts[str(row["parent_example_id"])] += 1
            case_counts[case_id] += 1
            need -= 1
    for row in ranked:
        if len(selected) >= target_count:
            break
        candidate_id = str(row["candidate_id"])
        case_id = str(row["case_id"])
        parent_id = str(row["parent_example_id"])
        if candidate_id in selected_ids or case_counts[case_id] >= case_cap:
            continue
        if parent_counts[parent_id] >= 2:
            continue
        selected.append(row)
        selected_ids.add(candidate_id)
        parent_counts[parent_id] += 1
        case_counts[case_id] += 1
    return selected


def summarize_expansion(
    examples: dict[str, dict[str, Any]],
    selected_deck: list[dict[str, Any]],
    candidates: list[dict[str, Any]],
    expansion_deck: list[dict[str, Any]],
    policy_summary: dict[str, Any],
    *,
    paths: dict[str, Path],
    target_count: int,
    min_per_case: int,
    max_case_share: float,
) -> dict[str, Any]:
    case_counts = Counter(str(row["case_id"]) for row in expansion_deck)
    unique_parent_count = len({str(row["parent_example_id"]) for row in expansion_deck})
    forbidden_count = forbidden_action_term_count(expansion_deck)
    coordinate_field_count = coordinate_field_count_in_actions(expansion_deck)
    action_signature_count = len(
        {action_signature(row["operator_action"]) for row in expansion_deck}
    )
    parent_gaps = parent_strict_gap_summary(examples, selected_deck)
    representation_pass = (
        len(expansion_deck) >= target_count
        and len(case_counts) == 8
        and largest_share(case_counts) <= max_case_share
        and unique_parent_count >= target_count // 2
        and forbidden_count == 0
        and coordinate_field_count == 0
    )
    risk_guard_pass = bool(policy_summary.get("risk_profile_pass"))
    fresh_replay_required = True
    strict_winner_evidence_count = 0
    ready = representation_pass and risk_guard_pass
    return {
        "schema": EXPANSION_SCHEMA,
        "decision": "parameter_expansion_deck_ready_for_fresh_replay"
        if ready
        else "stop_parameter_expansion_gate",
        **{name: str(path) for name, path in paths.items()},
        "label_usage_policy": LABEL_USAGE_POLICY,
        "fresh_metric_status": FRESH_METRIC_STATUS,
        "target_count": target_count,
        "min_per_case": min_per_case,
        "max_case_share_target": max_case_share,
        "parent_source_deck_count": len(selected_deck),
        "candidate_count": len(candidates),
        "selected_expansion_count": len(expansion_deck),
        "represented_case_count": len(case_counts),
        "case_counts": dict(case_counts),
        "largest_case_share": largest_share(case_counts),
        "unique_parent_source_count": unique_parent_count,
        "unique_action_signature_count": action_signature_count,
        "forbidden_action_term_count": forbidden_count,
        "direct_coordinate_field_count": direct_coordinate_field_count(finite_action_schema())
        + coordinate_field_count,
        "policy_risk_guard_decision": policy_summary.get("decision"),
        "policy_risk_profile_pass": risk_guard_pass,
        "representation_pass": representation_pass,
        "fresh_replay_required": fresh_replay_required,
        "strict_winner_evidence_count": strict_winner_evidence_count,
        "phase4_gate_open": False,
        "allowed_next_phase": "step7q_fresh_metric_replay_executor" if ready else None,
        "next_recommendation": "execute_geometry_or_fresh_metric_replay_for_expansion_deck"
        if ready
        else "fix_expansion_deck_before_replay",
        "parent_strict_gap_summary": parent_gaps,
    }


def parent_strict_gap_summary(
    examples: dict[str, dict[str, Any]], selected_deck: list[dict[str, Any]]
) -> dict[str, Any]:
    gaps = []
    exact_count = 0
    near_count = 0
    for parent in selected_deck:
        labels = examples[str(parent["example_id"])].get("labels", {})
        official = labels.get("official_like_cost_delta")
        if official is None:
            continue
        exact_count += 1
        gap = max(0.0, float(official) + STRICT_COST_EPS)
        gaps.append(gap)
        if 0.0 < gap <= 2.0e-7:
            near_count += 1
    return {
        "exact_parent_metric_count": exact_count,
        "near_strict_margin_parent_count": near_count,
        "min_required_official_delta_improvement": min(gaps) if gaps else None,
        "median_required_official_delta_improvement": median(gaps),
        "label_usage_policy": LABEL_USAGE_POLICY,
    }


def forbidden_action_term_count(rows: list[dict[str, Any]]) -> int:
    count = 0
    for row in rows:
        text = json.dumps(row.get("operator_action", {}), sort_keys=True).lower()
        count += sum(int(term in text) for term in FORBIDDEN_ACTION_TERMS)
    return count


def coordinate_field_count_in_actions(rows: list[dict[str, Any]]) -> int:
    forbidden = {"x", "y", "final_x", "final_y", "absolute_x", "absolute_y"}
    count = 0
    for row in rows:
        action = row.get("operator_action", {})
        if isinstance(action, dict):
            count += sum(int(key in forbidden) for key in action)
            count += len(action.get("direct_coordinate_fields", []))
    return count


def action_signature(action: dict[str, Any]) -> tuple[str, ...]:
    keys = [
        "operator_family",
        "closure_policy",
        "blocker_chain_depth_bin",
        "direction_bin",
        "magnitude_bin",
        "bbox_guard_mode",
        "vector_guard_mode",
    ]
    return tuple(str(action.get(key)) for key in keys)


def expansion_markdown(summary: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# Step7Q Operator Parameter Expansion Summary",
            "",
            f"Decision: `{summary['decision']}`",
            "",
            f"- parent_source_deck_count: {summary['parent_source_deck_count']}",
            f"- candidate_count: {summary['candidate_count']}",
            f"- selected_expansion_count: {summary['selected_expansion_count']}",
            f"- represented_case_count: {summary['represented_case_count']}",
            f"- largest_case_share: {summary['largest_case_share']}",
            f"- unique_parent_source_count: {summary['unique_parent_source_count']}",
            f"- unique_action_signature_count: {summary['unique_action_signature_count']}",
            f"- forbidden_action_term_count: {summary['forbidden_action_term_count']}",
            f"- direct_coordinate_field_count: {summary['direct_coordinate_field_count']}",
            f"- policy_risk_profile_pass: {summary['policy_risk_profile_pass']}",
            f"- representation_pass: {summary['representation_pass']}",
            f"- fresh_replay_required: {summary['fresh_replay_required']}",
            f"- strict_winner_evidence_count: {summary['strict_winner_evidence_count']}",
            f"- phase4_gate_open: {summary['phase4_gate_open']}",
            f"- allowed_next_phase: {summary['allowed_next_phase']}",
            "",
        ]
    )


def int_value(value: Any) -> int:
    if isinstance(value, bool) or value is None:
        return 0
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def median(values: list[float]) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return (ordered[middle - 1] + ordered[middle]) / 2.0


__all__ = [
    "action_variants",
    "expand_candidates",
    "run_operator_parameter_expansion",
    "select_expansion_deck",
]
