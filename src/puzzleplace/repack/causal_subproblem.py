"""Step7P causal subproblem attribution primitives."""

from __future__ import annotations

from typing import Any

FORBIDDEN_LABEL_TERMS = (
    "target_positions",
    "fp_sol",
    "tree_sol",
    "supervised_target",
    "label_target",
)

INTENT_FAMILIES = (
    "hpwl_hull_shrink",
    "bbox_hull_compaction",
    "blocker_chain_unblock",
    "soft_guarded_repair",
    "mib_shape_preserve_repack",
    "boundary_contact_guarded",
    "order_preserving_row_repack",
    "closure_translate_with_repair",
)

FAILURE_BUCKETS = (
    "bad_internal_repack",
    "wrong_target_region",
    "wrong_slot",
    "soft_regression",
    "bbox_regression",
    "hpwl_gain_but_official_like_loss",
    "dominated_by_original",
    "overlap_after_splice",
    "unknown",
)

EPS = 1e-9


def infer_failure_bucket(row: dict[str, Any]) -> str:
    """Map heterogeneous Step7 rows to a causal failure bucket."""

    explicit = str(row.get("failure_attribution") or row.get("target_failure_bucket") or "")
    if "bad_internal_repack" in explicit:
        return "bad_internal_repack"
    if "wrong_target_region" in explicit:
        return "wrong_target_region"
    if "wrong_slot" in explicit or bool_value(row.get("no_slot_available")):
        return "wrong_slot"
    if "overlap" in explicit or int_value(row.get("overlap_pair_count")) > 0:
        return "overlap_after_splice"
    if float_value(row.get("soft_constraint_delta")) > EPS:
        return "soft_regression"
    if float_value(row.get("bbox_area_delta")) > EPS:
        return "bbox_regression"
    if float_value(row.get("hpwl_delta")) < -EPS and float_value(
        row.get("official_like_cost_delta")
    ) > EPS:
        return "hpwl_gain_but_official_like_loss"
    if bool_value(row.get("dominated_by_original")):
        return "dominated_by_original"
    if bool_value(row.get("metric_regressing")):
        return "soft_regression"
    return "unknown"


def infer_intent_family(row: dict[str, Any], failure_bucket: str) -> str:
    if failure_bucket == "soft_regression":
        return "soft_guarded_repair"
    if failure_bucket == "bbox_regression":
        return "bbox_hull_compaction"
    if failure_bucket == "hpwl_gain_but_official_like_loss":
        return "hpwl_hull_shrink"
    if failure_bucket in {"wrong_slot", "overlap_after_splice"}:
        return "blocker_chain_unblock"
    if failure_bucket == "wrong_target_region":
        return "boundary_contact_guarded"
    if failure_bucket == "bad_internal_repack":
        return "order_preserving_row_repack"
    if has_mib_member(row):
        return "mib_shape_preserve_repack"
    if int_value(row.get("moved_block_count") or row.get("block_count")) > 1:
        return "closure_translate_with_repair"
    if has_boundary_member(row) or str(row.get("route_class")) == "global":
        return "boundary_contact_guarded"
    return "order_preserving_row_repack"


def metric_confidence(row: dict[str, Any], source: str) -> str:
    if all(row.get(key) is not None for key in objective_keys()):
        return "exact_component_comparable"
    if source.startswith("step7m") and row.get("actual_objective_vector") is not None:
        return "exact_component_partial"
    if bool_value(row.get("hard_feasible_non_noop") or row.get("hard_feasible_nonnoop")):
        return "exact_component_partial"
    return "proxy_only"


def objective_keys() -> tuple[str, str, str, str]:
    return ("hpwl_delta", "bbox_area_delta", "soft_constraint_delta", "official_like_cost_delta")


def extract_block_ids(row: dict[str, Any]) -> list[int]:
    moved = row.get("moved_block_ids")
    if isinstance(moved, list):
        return sorted({int_value(item) for item in moved if int_value(item) >= 0})
    preview = row.get("decoded_blocks_preview")
    if isinstance(preview, list):
        block_ids = []
        for item in preview:
            if isinstance(item, dict) and item.get("block_id") is not None:
                block_ids.append(int_value(item.get("block_id")))
        return sorted({block_id for block_id in block_ids if block_id >= 0})
    block_id = row.get("block_id")
    return [] if block_id is None else [int_value(block_id)]


def has_boundary_member(row: dict[str, Any]) -> bool:
    preview = row.get("decoded_blocks_preview")
    return isinstance(preview, list) and any(
        isinstance(item, dict) and int_value(item.get("boundary")) != 0 for item in preview
    )


def has_mib_member(row: dict[str, Any]) -> bool:
    preview = row.get("decoded_blocks_preview")
    return isinstance(preview, list) and any(
        isinstance(item, dict) and int_value(item.get("mib")) != 0 for item in preview
    )


def bbox_hull_risk_class(row: dict[str, Any]) -> str:
    if float_value(row.get("bbox_area_delta")) > EPS:
        return "risky"
    if row.get("bbox_area_delta") is not None:
        return "safe"
    if row.get("closure_bbox") is not None:
        return "unknown"
    return "unknown"


def forbidden_term_count(rows: list[dict[str, Any]]) -> int:
    text = "\n".join(str(row) for row in rows).lower()
    return sum(text.count(term) for term in FORBIDDEN_LABEL_TERMS)


def int_value(value: Any) -> int:
    if isinstance(value, bool) or value is None:
        return 0
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def float_value(value: Any) -> float:
    if isinstance(value, bool) or value is None:
        return 0.0
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def bool_value(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    if isinstance(value, str):
        return value.lower() in {"1", "true", "yes", "y"}
    return False
