from __future__ import annotations

from collections import Counter
from typing import Any

from puzzleplace.research.move_library import improvement_score

STEP6O_GUARD_THRESHOLDS: dict[str, float] = {
    "hpwl_regression_per_boundary_gain": 40.0,
    "spatial_balance_worsening": 0.10,
}

SUSPICIOUS_CASE_IDS = {8, 10, 14, 15, 21, 23, 28, 31, 32, 33, 37, 38}


def hpwl_regression_per_boundary_gain(row: dict[str, Any]) -> float:
    gain = max(float(row.get("boundary_delta", 0.0)), 1e-6)
    return max(float(row.get("hpwl_delta", 0.0)), 0.0) / gain


def spatial_balance_worsening(pathology_delta: dict[str, Any]) -> float:
    return max(
        float(pathology_delta.get("left_right_balance_delta", 0.0)),
        float(pathology_delta.get("top_bottom_balance_delta", 0.0)),
    )


def step6o_guard_reasons(
    row: dict[str, Any],
    pathology_delta: dict[str, Any],
    *,
    thresholds: dict[str, float] | None = None,
) -> list[str]:
    """Return Step6O sidecar guard reasons without changing runtime selector rules."""
    active = thresholds or STEP6O_GUARD_THRESHOLDS
    if str(row.get("move_type")) != "simple_compaction":
        return []
    reasons: list[str] = []
    if (
        hpwl_regression_per_boundary_gain(row)
        > active["hpwl_regression_per_boundary_gain"]
    ):
        reasons.append("hpwl_regression_per_boundary_gain_gt_40")
    if spatial_balance_worsening(pathology_delta) > active["spatial_balance_worsening"]:
        reasons.append("spatial_balance_worsening_gt_0.10")
    return reasons


def select_guarded_case_alternative(
    rows: list[dict[str, Any]],
    *,
    mode: str = "safe",
) -> dict[str, Any]:
    """Replay Step6M selector after filtering accepted rows rejected by Step6O guards."""
    accepted = [row for row in rows if row.get("accepted") and not row.get("guard_rejected")]
    original = original_selection_payload(rows)
    if not accepted:
        return {**original, "selection_reason": "step6o_no_guarded_accepted_move"}

    def key(row: dict[str, Any]) -> tuple[float, float, float, float, float]:
        return (
            -float(row["boundary_delta"]),
            float(row["bbox_delta"]),
            float(row["hpwl_delta"]),
            float(row["soft_delta"]),
            float(row["generation_time_ms"] + row["eval_time_ms"]),
        )

    selected = min(accepted, key=key)
    if mode == "safe" and improvement_score(selected) <= 0.0:
        return {**original, "selection_reason": "step6o_guarded_moves_not_net_positive"}
    return {
        "case_id": selected["case_id"],
        "selected_move_type": selected["move_type"],
        "selected_target_blocks": selected["target_blocks"],
        "selection_reason": selected.get("selection_reason", "accepted_by_safe_gate"),
        "boundary_delta": selected["boundary_delta"],
        "bbox_delta": selected["bbox_delta"],
        "hpwl_delta": selected["hpwl_delta"],
        "soft_delta": selected["soft_delta"],
        "grouping_delta": selected["grouping_delta"],
        "mib_delta": selected["mib_delta"],
        "hard_feasible": selected["hard_feasible"],
        "frame_protrusion": selected["frame_protrusion"],
        "runtime_ms": selected["generation_time_ms"]
        + selected.get("repair_time_ms", 0.0)
        + selected["eval_time_ms"],
        "guard_rejected_alternatives": selected.get("case_guard_rejected_count", 0),
    }


def original_selection_payload(rows: list[dict[str, Any]]) -> dict[str, Any]:
    case_id = rows[0]["case_id"] if rows else "unknown"
    return {
        "case_id": case_id,
        "selected_move_type": "original",
        "selected_target_blocks": [],
        "selection_reason": "original_fallback",
        "boundary_delta": 0.0,
        "bbox_delta": 0.0,
        "hpwl_delta": 0.0,
        "soft_delta": 0.0,
        "grouping_delta": 0.0,
        "mib_delta": 0.0,
        "hard_feasible": True,
        "frame_protrusion": 0.0,
        "runtime_ms": 0.0,
        "guard_rejected_alternatives": 0,
    }


def compare_selection_sets(
    before: list[dict[str, Any]],
    after: list[dict[str, Any]],
    *,
    suspicious_case_ids: set[int] | None = None,
) -> dict[str, Any]:
    suspicious = suspicious_case_ids or SUSPICIOUS_CASE_IDS
    before_by_case = {int(row["case_id"]): row for row in before}
    after_by_case = {int(row["case_id"]): row for row in after}
    case_rows: list[dict[str, Any]] = []
    for case_id in sorted(before_by_case):
        old = before_by_case[case_id]
        new = after_by_case[case_id]
        case_rows.append(
            {
                "case_id": case_id,
                "was_suspicious": case_id in suspicious,
                "before_move": old["selected_move_type"],
                "after_move": new["selected_move_type"],
                "changed": old["selected_move_type"] != new["selected_move_type"]
                or list(old.get("selected_target_blocks", []))
                != list(new.get("selected_target_blocks", [])),
                "boundary_delta_change": float(new["boundary_delta"])
                - float(old["boundary_delta"]),
                "bbox_delta_change": float(new["bbox_delta"]) - float(old["bbox_delta"]),
                "hpwl_delta_change": float(new["hpwl_delta"]) - float(old["hpwl_delta"]),
                "soft_delta_change": float(new["soft_delta"]) - float(old["soft_delta"]),
            }
        )
    return {
        "selected_move_counts_before": dict(
            Counter(str(row["selected_move_type"]) for row in before)
        ),
        "selected_move_counts_after": dict(
            Counter(str(row["selected_move_type"]) for row in after)
        ),
        "suspicious_selected_count_before": sum(
            int(int(row["case_id"]) in suspicious and row["selected_move_type"] != "original")
            for row in before
        ),
        "suspicious_selected_count_after": sum(
            int(int(row["case_id"]) in suspicious and row["selected_move_type"] != "original")
            for row in after
        ),
        "suspicious_simple_compaction_count_before": sum(
            int(
                int(row["case_id"]) in suspicious
                and row["selected_move_type"] == "simple_compaction"
            )
            for row in before
        ),
        "suspicious_simple_compaction_count_after": sum(
            int(
                int(row["case_id"]) in suspicious
                and row["selected_move_type"] == "simple_compaction"
            )
            for row in after
        ),
        "original_fallback_count_before": sum(
            int(row["selected_move_type"] == "original") for row in before
        ),
        "original_fallback_count_after": sum(
            int(row["selected_move_type"] == "original") for row in after
        ),
        "mean_deltas_before": mean_selection_deltas(before),
        "mean_deltas_after": mean_selection_deltas(after),
        "mean_delta_change": {
            key: mean_selection_deltas(after)[key] - mean_selection_deltas(before)[key]
            for key in ("boundary_delta", "bbox_delta", "hpwl_delta", "soft_delta")
        },
        "case_comparisons": case_rows,
    }


def mean_selection_deltas(rows: list[dict[str, Any]]) -> dict[str, float]:
    return {
        key: sum(float(row.get(key, 0.0)) for row in rows) / max(len(rows), 1)
        for key in ("boundary_delta", "bbox_delta", "hpwl_delta", "soft_delta")
    }
