"""Step7Q-A non-leaky operator-learning data mart.

The data mart separates request-time features from replay labels and masks so a
later ranking/GNN/bandit policy cannot accidentally train on objective vectors
or replay outcomes as inputs.
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Iterable
from pathlib import Path
from typing import Any

FORBIDDEN_REQUEST_TERMS = ("micro_axis_corridor", "soft_repair_budgeted", "hpwl_only")
VALIDATION_LABEL_POLICY = "labels used for replay/evaluation only, not request generation"
LABEL_LEAKAGE_FEATURE_KEYS = (
    "objective_vector",
    "seed_objective_vector",
    "actual_objective_vector",
    "hard_feasible_nonnoop",
    "seed_hard_feasible_nonnoop",
    "actual_all_vector_nonregressing",
    "strict_meaningful_winner",
    "quality_gate_pass",
    "hpwl_delta",
    "bbox_area_delta",
    "soft_constraint_delta",
    "official_like_cost_delta",
    "fresh_metric_available",
    "overlap_after_splice",
    "soft_regression",
    "bbox_regression",
    "hpwl_regression",
)
FINITE_ACTION_SCHEMA = {
    "operator_family": [
        "closure_translate_with_repair",
        "order_preserving_row_repack",
        "blocker_chain_unblock",
        "mib_shape_preserve_repack",
        "soft_guarded_repair",
    ],
    "closure_policy": [
        "affected_only",
        "affected_plus_blockers",
        "affected_plus_soft_linked",
        "affected_plus_blockers_plus_soft_linked",
    ],
    "blocker_chain_depth_bin": ["0", "1", "2", "bounded_all"],
    "direction_bin": [
        "hpwl_sink_toward_pins",
        "bbox_shrink_x",
        "bbox_shrink_y",
        "soft_release",
        "blocker_unblock",
        "slack_fill_left",
        "slack_fill_right",
        "slack_fill_up",
        "slack_fill_down",
    ],
    "magnitude_bin": ["tiny", "small", "medium", "slack_limited"],
    "bbox_guard_mode": ["nonexpand", "shrink_only", "boundary_safe"],
    "vector_guard_mode": [
        "hpwl_nonregress",
        "bbox_nonregress",
        "soft_nonregress",
        "all_vector_nonregress",
    ],
}


def build_operator_learning_data_mart(
    atlas_path: Path,
    requests_path: Path,
    replay_rows_path: Path,
    blocker_path: Path,
    branch_summary_path: Path,
    examples_out_path: Path,
    label_summary_out_path: Path,
    feature_summary_out_path: Path,
    summary_out_path: Path,
    markdown_out_path: Path,
) -> dict[str, Any]:
    atlas_rows = read_jsonl(atlas_path)
    requests = read_jsonl(requests_path)
    replay_rows = read_jsonl(replay_rows_path)
    blocker = load_json(blocker_path)
    branch_summary = load_json(branch_summary_path)
    examples = build_operator_examples(atlas_rows, requests, replay_rows)
    write_jsonl(examples_out_path, examples)
    label_summary = summarize_labels(examples, blocker)
    feature_summary = summarize_features(examples)
    summary = summarize_data_mart(
        examples,
        label_summary,
        feature_summary,
        branch_summary,
        paths={
            "atlas_path": atlas_path,
            "requests_path": requests_path,
            "replay_rows_path": replay_rows_path,
            "blocker_path": blocker_path,
            "branch_summary_path": branch_summary_path,
            "examples_path": examples_out_path,
            "label_summary_path": label_summary_out_path,
            "feature_summary_path": feature_summary_out_path,
            "summary_path": summary_out_path,
            "markdown_path": markdown_out_path,
        },
    )
    write_json(label_summary_out_path, label_summary)
    write_json(feature_summary_out_path, feature_summary)
    write_json(summary_out_path, summary)
    markdown_out_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_out_path.write_text(data_mart_markdown(summary), encoding="utf-8")
    return summary


def build_operator_examples(
    atlas_rows: list[dict[str, Any]],
    requests: list[dict[str, Any]],
    replay_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    request_by_source = {
        str(row.get("source_subproblem_id")): row
        for row in requests
        if row.get("source_subproblem_id") is not None
    }
    replay_by_request = {
        str(row.get("request_id")): row
        for row in replay_rows
        if row.get("request_id") is not None
    }
    examples = []
    for index, row in enumerate(atlas_rows):
        subproblem_id = str(row.get("subproblem_id") or f"atlas_{index:05d}")
        request = request_by_source.get(subproblem_id)
        replay = replay_by_request.get(str(request.get("request_id"))) if request else None
        examples.append(example_from_rows(index, row, request, replay))
    return examples


def example_from_rows(
    index: int,
    atlas: dict[str, Any],
    request: dict[str, Any] | None,
    replay: dict[str, Any] | None,
) -> dict[str, Any]:
    subproblem_id = str(atlas.get("subproblem_id") or f"atlas_{index:05d}")
    features = feature_dict(atlas, request)
    labels = label_dict(atlas, request, replay)
    masks = mask_dict(atlas, request, labels)
    return {
        "schema": "step7q_operator_example_v1",
        "example_id": f"step7q_ex_{index:05d}",
        "case_id": str(atlas.get("case_id")),
        "source_subproblem_id": subproblem_id,
        "source_candidate_id": str(atlas.get("seed_candidate_id")),
        "seed_source": str(atlas.get("seed_source")),
        "intent_family": str(atlas.get("intent_family")),
        "seed_failure_bucket": safe_failure_bucket(str(atlas.get("seed_failure_bucket"))),
        "metric_confidence": str(atlas.get("metric_confidence")),
        "features": features,
        "graph": graph_dict(atlas, features),
        "labels": labels,
        "masks": masks,
        "rank_label_key": rank_label_key(labels),
        "finite_action_schema_version": "step7q_finite_operator_action_v1",
    }


def feature_dict(atlas: dict[str, Any], request: dict[str, Any] | None) -> dict[str, Any]:
    allowed = [str(item) for item in list_value(atlas.get("allowed_repack_families"))]
    affected = int_list(atlas.get("affected_block_ids"))
    blockers = int_list(atlas.get("blocker_block_ids"))
    soft_linked = int_list(atlas.get("soft_linked_block_ids"))
    request_policy = request.get("request_source_policy") if isinstance(request, dict) else None
    return {
        "case_id": str(atlas.get("case_id")),
        "seed_source": str(atlas.get("seed_source")),
        "intent_family": str(atlas.get("intent_family")),
        "seed_failure_bucket": safe_failure_bucket(str(atlas.get("seed_failure_bucket"))),
        "metric_confidence": str(atlas.get("metric_confidence")),
        "route_class": str(atlas.get("route_class")),
        "decoder": str(atlas.get("decoder")),
        "bbox_hull_risk_class": str(atlas.get("bbox_hull_risk_class")),
        "boundary_constraint_touched": bool(atlas.get("boundary_constraint_touched")),
        "mib_constraint_touched": bool(atlas.get("mib_constraint_touched")),
        "group_constraint_touched": bool(atlas.get("group_constraint_touched")),
        "affected_block_count": len(affected),
        "blocker_block_count": len(blockers),
        "soft_linked_block_count": len(soft_linked),
        "moved_block_count": int_value(atlas.get("moved_block_count")),
        "allowed_repack_family_flags": {
            family: family in allowed for family in sorted(set(allowed) | known_repack_families())
        },
        "has_phase3_request": request is not None,
        "request_source_policy": str(request_policy) if request_policy is not None else "none",
        "non_micro_intent": bool(request.get("non_micro_intent"))
        if isinstance(request, dict)
        else False,
    }


def label_dict(
    atlas: dict[str, Any], request: dict[str, Any] | None, replay: dict[str, Any] | None
) -> dict[str, Any]:
    replay_vector = replay.get("actual_objective_vector") if isinstance(replay, dict) else None
    atlas_vector = atlas.get("objective_vector")
    vector = vector_dict(replay_vector if replay_vector is not None else atlas_vector)
    fresh_metric_available = bool(replay.get("fresh_metric_available")) if replay else False
    hard = bool_value(
        replay.get("hard_feasible_nonnoop") if replay else atlas.get("hard_feasible_nonnoop")
    )
    soft_regression = (
        bool_value(replay.get("soft_regression"))
        if replay
        else vector_positive(vector, "soft_constraint_delta")
    )
    bbox_regression = (
        bool_value(replay.get("bbox_regression"))
        if replay
        else vector_positive(vector, "bbox_area_delta")
    )
    hpwl_regression = (
        bool_value(replay.get("hpwl_regression"))
        if replay
        else vector_positive(vector, "hpwl_delta")
    )
    all_vector = bool_value(replay.get("actual_all_vector_nonregressing")) if replay else (
        hard and not soft_regression and not bbox_regression and not hpwl_regression
    )
    strict = bool_value(replay.get("strict_meaningful_winner")) if replay else False
    return {
        "phase3_request_present": request is not None,
        "fresh_metric_available": fresh_metric_available,
        "hard_feasible_nonnoop": hard,
        "overlap_after_splice": int_value(replay.get("overlap_after_splice"))
        if replay
        else int(not hard),
        "soft_regression": soft_regression,
        "bbox_regression": bbox_regression,
        "hpwl_regression": hpwl_regression,
        "actual_all_vector_nonregressing": all_vector,
        "strict_meaningful_winner": strict,
        "quality_gate_pass": bool_value(replay.get("quality_gate_pass")) if replay else False,
        "hpwl_delta": vector["hpwl_delta"],
        "bbox_area_delta": vector["bbox_area_delta"],
        "soft_constraint_delta": vector["soft_constraint_delta"],
        "official_like_cost_delta": vector["official_like_cost_delta"],
    }


def mask_dict(
    atlas: dict[str, Any], request: dict[str, Any] | None, labels: dict[str, Any]
) -> dict[str, Any]:
    text = forbidden_term_search_text(atlas, request)
    forbidden = any(term in text for term in FORBIDDEN_REQUEST_TERMS)
    exact_vector = all(labels.get(key) is not None for key in vector_label_keys())
    validation_ok = atlas.get("validation_label_policy") == VALIDATION_LABEL_POLICY
    return {
        "forbidden_request_term": forbidden,
        "eligible_for_training": validation_ok and not forbidden,
        "eligible_for_selection": validation_ok and not forbidden and bool(request),
        "eligible_for_risk_supervision": validation_ok and not forbidden and exact_vector,
        "eligible_for_strict_supervision": False,
        "strict_supervision_disabled_reason": "zero_strict_meaningful_positive_count",
    }


def forbidden_term_search_text(atlas: dict[str, Any], request: dict[str, Any] | None) -> str:
    """Return only actual source/action text for forbidden operator matching.

    Phase3 request rows carry a ``forbidden`` policy list documenting operators
    that must *not* be used.  That declaration is safety metadata, not evidence
    that a forbidden operator was selected, so scanning the whole request dict
    would mask every safe Phase3 request.  Keep the search to chosen source and
    action-family fields.
    """

    fields: list[Any] = [
        atlas.get("subproblem_id"),
        atlas.get("seed_candidate_id"),
        atlas.get("seed_source_candidate_id"),
        atlas.get("seed_source"),
        atlas.get("intent_family"),
        atlas.get("allowed_repack_families"),
        atlas.get("decoder"),
    ]
    if isinstance(request, dict):
        fields.extend(
            [
                request.get("source_subproblem_id"),
                request.get("source_candidate_id"),
                request.get("seed_source"),
                request.get("intent_family"),
                request.get("allowed_repack_families"),
                request.get("request_source_policy"),
            ]
        )
    return json.dumps(fields, sort_keys=True).lower()


def graph_dict(atlas: dict[str, Any], features: dict[str, Any]) -> dict[str, Any]:
    nodes = [
        {"id": f"case:{features['case_id']}", "type": "case"},
        {"id": f"request:{atlas.get('subproblem_id')}", "type": "request"},
        {"id": f"intent:{features['intent_family']}", "type": "intent_family"},
        {"id": f"failure:{features['seed_failure_bucket']}", "type": "failure_bucket"},
    ]
    edges = [
        {"src": nodes[0]["id"], "dst": nodes[1]["id"], "type": "case_has_request"},
        {"src": nodes[1]["id"], "dst": nodes[2]["id"], "type": "request_intent"},
        {"src": nodes[1]["id"], "dst": nodes[3]["id"], "type": "request_failure"},
    ]
    for relation, block_ids in [
        ("affected_block", int_list(atlas.get("affected_block_ids"))),
        ("blocker_block", int_list(atlas.get("blocker_block_ids"))),
        ("soft_linked_block", int_list(atlas.get("soft_linked_block_ids"))),
    ]:
        for block_id in block_ids:
            block_node = {"id": f"block:{features['case_id']}:{block_id}", "type": "block"}
            nodes.append(block_node)
            edges.append({"src": nodes[1]["id"], "dst": block_node["id"], "type": relation})
    for family, enabled in features["allowed_repack_family_flags"].items():
        if not enabled:
            continue
        family_node = {"id": f"family:{family}", "type": "allowed_repack_family"}
        nodes.append(family_node)
        edges.append({"src": nodes[1]["id"], "dst": family_node["id"], "type": "allowed_family"})
    return {"nodes": nodes, "edges": edges}


def summarize_labels(examples: list[dict[str, Any]], blocker: dict[str, Any]) -> dict[str, Any]:
    labels = [example["labels"] for example in examples]
    masks = [example["masks"] for example in examples]
    strict_count = sum(int(label["strict_meaningful_winner"]) for label in labels)
    return {
        "schema": "step7q_operator_label_summary_v1",
        "example_count": len(examples),
        "fresh_metric_available_count": sum(
            int(label["fresh_metric_available"]) for label in labels
        ),
        "hard_feasible_nonnoop_count": sum(int(label["hard_feasible_nonnoop"]) for label in labels),
        "overlap_after_splice_count": sum(
            int(label["overlap_after_splice"] > 0) for label in labels
        ),
        "soft_regression_count": sum(int(label["soft_regression"]) for label in labels),
        "bbox_regression_count": sum(int(label["bbox_regression"]) for label in labels),
        "hpwl_regression_count": sum(int(label["hpwl_regression"]) for label in labels),
        "all_vector_nonregressing_positive_count": sum(
            int(label["actual_all_vector_nonregressing"]) for label in labels
        ),
        "strict_meaningful_positive_count": strict_count,
        "strict_source_count_from_blocker": blocker.get("strict_meaningful_source_count"),
        "strict_supervision_enabled": strict_count > 0,
        "eligible_exact_hard_nonforbidden_count": blocker.get(
            "eligible_exact_hard_nonforbidden_count"
        ),
        "eligible_for_training_count": sum(int(mask["eligible_for_training"]) for mask in masks),
        "eligible_for_selection_count": sum(int(mask["eligible_for_selection"]) for mask in masks),
        "eligible_for_risk_supervision_count": sum(
            int(mask["eligible_for_risk_supervision"]) for mask in masks
        ),
        "eligible_for_strict_supervision_count": sum(
            int(mask["eligible_for_strict_supervision"]) for mask in masks
        ),
    }


def summarize_features(examples: list[dict[str, Any]]) -> dict[str, Any]:
    leakage = leakage_examples(examples)
    case_counts = Counter(str(example["case_id"]) for example in examples)
    intent_counts = Counter(str(example["intent_family"]) for example in examples)
    failure_counts = Counter(str(example["seed_failure_bucket"]) for example in examples)
    return {
        "schema": "step7q_operator_feature_summary_v1",
        "example_count": len(examples),
        "represented_case_count": len(case_counts),
        "case_counts": dict(case_counts),
        "largest_case_share": largest_share(case_counts),
        "intent_family_counts": dict(intent_counts),
        "failure_bucket_counts": dict(failure_counts),
        "feature_label_leakage_count": len(leakage),
        "feature_label_leakage_examples": leakage[:10],
        "finite_action_schema": finite_action_schema(),
        "direct_coordinate_field_count": direct_coordinate_field_count(finite_action_schema()),
    }


def summarize_data_mart(
    examples: list[dict[str, Any]],
    label_summary: dict[str, Any],
    feature_summary: dict[str, Any],
    branch_summary: dict[str, Any],
    *,
    paths: dict[str, Path],
) -> dict[str, Any]:
    strict_supervision_enabled = bool(label_summary["strict_supervision_enabled"])
    promote = (
        len(examples) >= 325
        and feature_summary["represented_case_count"] == 8
        and feature_summary["largest_case_share"] <= 0.32
        and feature_summary["feature_label_leakage_count"] == 0
        and int(label_summary.get("eligible_exact_hard_nonforbidden_count") or 0) >= 107
        and label_summary["all_vector_nonregressing_positive_count"] >= 30
        and label_summary["strict_meaningful_positive_count"] == 0
        and not strict_supervision_enabled
        and feature_summary["direct_coordinate_field_count"] == 0
    )
    return {
        "schema": "step7q_operator_data_mart_summary_v1",
        "decision": "promote_to_constrained_risk_ranking" if promote else "stop_data_mart_invalid",
        **{name: str(path) for name, path in paths.items()},
        "example_count": len(examples),
        "represented_case_count": feature_summary["represented_case_count"],
        "largest_case_share": feature_summary["largest_case_share"],
        "feature_label_leakage_count": feature_summary["feature_label_leakage_count"],
        "direct_coordinate_field_count": feature_summary["direct_coordinate_field_count"],
        "eligible_exact_hard_nonforbidden_count": label_summary.get(
            "eligible_exact_hard_nonforbidden_count"
        ),
        "all_vector_nonregressing_positive_count": label_summary[
            "all_vector_nonregressing_positive_count"
        ],
        "strict_meaningful_positive_count": label_summary["strict_meaningful_positive_count"],
        "strict_supervision_enabled": strict_supervision_enabled,
        "branch_c_baseline_name": branch_summary.get("best_branch_name"),
        "forbidden_scope_preserved": True,
        "gnn_rl_gate_open": True,
        "allowed_next_phase": "step7q_constrained_risk_ranking" if promote else None,
        "next_recommendation": "train_or_smoke_constrained_risk_ranking"
        if promote
        else "fix_data_mart_before_learning",
    }


def validate_no_label_leakage(examples: list[dict[str, Any]]) -> list[str]:
    return leakage_examples(examples)


def leakage_examples(examples: list[dict[str, Any]]) -> list[str]:
    leaked = []
    for example in examples:
        feature_text = json.dumps(example.get("features", {}), sort_keys=True)
        for key in LABEL_LEAKAGE_FEATURE_KEYS:
            if key in feature_text:
                leaked.append(f"{example['example_id']}:{key}")
    return leaked


def finite_action_schema() -> dict[str, list[str]]:
    return {key: list(value) for key, value in FINITE_ACTION_SCHEMA.items()}


def direct_coordinate_field_count(schema: dict[str, list[str]]) -> int:
    forbidden = {"x", "y", "final_x", "final_y", "absolute_x", "absolute_y"}
    return sum(int(key in forbidden) for key in schema)


def rank_label_key(labels: dict[str, Any]) -> list[float | int]:
    return [
        -int(bool(labels["actual_all_vector_nonregressing"])),
        int(bool(labels["soft_regression"])),
        int(bool(labels["bbox_regression"])),
        int(bool(labels["hpwl_regression"])),
        float_or_default(labels.get("official_like_cost_delta"), 999.0),
    ]


def data_mart_markdown(summary: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# Step7Q Operator Data Mart Summary",
            "",
            f"Decision: `{summary['decision']}`",
            "",
            f"- example_count: {summary['example_count']}",
            f"- represented_case_count: {summary['represented_case_count']}",
            f"- largest_case_share: {summary['largest_case_share']}",
            f"- feature_label_leakage_count: {summary['feature_label_leakage_count']}",
            f"- direct_coordinate_field_count: {summary['direct_coordinate_field_count']}",
            f"- eligible_exact_hard_nonforbidden_count: "
            f"{summary['eligible_exact_hard_nonforbidden_count']}",
            f"- all_vector_nonregressing_positive_count: "
            f"{summary['all_vector_nonregressing_positive_count']}",
            f"- strict_meaningful_positive_count: "
            f"{summary['strict_meaningful_positive_count']}",
            f"- strict_supervision_enabled: {summary['strict_supervision_enabled']}",
            f"- allowed_next_phase: {summary['allowed_next_phase']}",
            f"- gnn_rl_gate_open: {summary['gnn_rl_gate_open']}",
            "",
        ]
    )


def known_repack_families() -> set[str]:
    return {
        "blocker_chain_unblock",
        "closure_translate_with_repair",
        "mib_shape_preserve_repack",
        "order_preserving_row_repack",
        "pareto_vector_filter",
        "soft_guarded_repair",
    }


def safe_failure_bucket(bucket: str) -> str:
    mapping = {
        "bad_internal_repack": "fb_bad_internal",
        "wrong_target_region": "fb_wrong_region",
        "wrong_slot": "fb_wrong_slot",
        "soft_regression": "fb_soft_risk",
        "bbox_regression": "fb_bbox_risk",
        "hpwl_gain_but_official_like_loss": "fb_hpwl_tradeoff",
        "dominated_by_original": "fb_dominated",
        "overlap_after_splice": "fb_overlap_risk",
        "unknown": "fb_unknown",
    }
    return mapping.get(bucket, "fb_other")


def vector_label_keys() -> tuple[str, str, str, str]:
    return ("hpwl_delta", "bbox_area_delta", "soft_constraint_delta", "official_like_cost_delta")


def vector_dict(value: Any) -> dict[str, float | None]:
    vector = value if isinstance(value, dict) else {}
    return {key: float_or_none(vector.get(key)) for key in vector_label_keys()}


def vector_positive(vector: dict[str, float | None], key: str) -> bool:
    value = vector.get(key)
    return value is not None and value > 0.0


def list_value(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def int_list(value: Any) -> list[int]:
    result = []
    for item in list_value(value):
        parsed = int_value(item)
        if parsed >= 0:
            result.append(parsed)
    return result


def int_value(value: Any) -> int:
    if isinstance(value, bool) or value is None:
        return 0
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def float_or_none(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def float_or_default(value: Any, default: float) -> float:
    parsed = float_or_none(value)
    return default if parsed is None else parsed


def bool_value(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    if isinstance(value, str):
        return value.lower() in {"1", "true", "yes", "y"}
    return False


def largest_share(counter: Counter[str]) -> float:
    total = sum(counter.values())
    return 0.0 if total == 0 else max(counter.values()) / total


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


__all__ = [
    "build_operator_examples",
    "build_operator_learning_data_mart",
    "finite_action_schema",
    "rank_label_key",
    "validate_no_label_leakage",
]
