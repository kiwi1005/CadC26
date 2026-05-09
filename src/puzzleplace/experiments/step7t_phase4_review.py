"""Step7T Phase4 review / integration-comparison gate.

This module is intentionally artifact-driven.  It does not rerun FloorSet and it
must not touch contest runtime/finalizer paths.  It checks whether the Step7T
active-soft sidecar evidence is strong enough to become the next integration
candidate, while separately tracking whether it is already runtime-integratable.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

EPS = 1e-9
DEFAULT_MEANINGFUL_COST_EPS = 1e-7
DELTA_KEYS = (
    "official_like_cost_delta",
    "hpwl_delta",
    "bbox_area_delta",
    "soft_constraint_delta",
)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def delta_gate(delta: dict[str, float], meaningful_cost_eps: float) -> dict[str, Any]:
    checks = {
        "official_strict_decrease": delta["official_like_cost_delta"] < -meaningful_cost_eps,
        "hpwl_nonregressing": delta["hpwl_delta"] <= EPS,
        "bbox_nonregressing": delta["bbox_area_delta"] <= EPS,
        "soft_nonregressing": delta["soft_constraint_delta"] <= EPS,
    }
    return {"pass": all(checks.values()), "checks": checks}


def _record_delta(record: dict[str, Any]) -> dict[str, float]:
    raw = record.get("delta_exact", record)
    return {key: float(raw[key]) for key in DELTA_KEYS}


def _stored_delta(record: dict[str, Any]) -> dict[str, float]:
    raw = record.get("stored_delta") or {}
    return {key: float(raw.get(key, record.get(key, 0.0))) for key in DELTA_KEYS}


def review_step7t_phase4(
    active_summary: dict[str, Any],
    visual_sanity: dict[str, Any],
    *,
    step7s_summary: dict[str, Any] | None = None,
    step7q_summary: dict[str, Any] | None = None,
    step7r_decision: dict[str, Any] | None = None,
    source_summary_path: str | None = None,
    visual_sanity_path: str | None = None,
) -> dict[str, Any]:
    meaningful_cost_eps = float(
        active_summary.get("meaningful_cost_eps", DEFAULT_MEANINGFUL_COST_EPS)
    )
    records = list(visual_sanity.get("records", []))
    winner_records: list[dict[str, Any]] = []
    for record in records:
        delta = _record_delta(record)
        gate = delta_gate(delta, meaningful_cost_eps)
        winner_records.append(
            {
                "case_id": int(record["case_id"]),
                "candidate_id": str(record["candidate_id"]),
                "hard_feasible": bool(record.get("hard_feasible")),
                "all_vector_nonregressing": bool(record.get("all_vector_nonregressing")),
                "strict_meaningful_winner": bool(record.get("strict_meaningful_winner")),
                "delta_exact": delta,
                "stored_delta": _stored_delta(record),
                "stored_vs_exact_max_abs_delta_error": float(
                    record.get("stored_vs_exact_max_abs_delta_error", 0.0)
                ),
                "delta_gate": gate,
                "moves": record.get("moves", []),
                "repaired_component": record.get("repaired_component"),
                "visualization_png": record.get("visualization_png"),
            }
        )

    unique_strict_cases = sorted(
        {
            row["case_id"]
            for row in winner_records
            if row["hard_feasible"]
            and row["all_vector_nonregressing"]
            and row["strict_meaningful_winner"]
            and row["delta_gate"]["pass"]
        }
    )
    max_stored_err = max(
        (row["stored_vs_exact_max_abs_delta_error"] for row in winner_records),
        default=0.0,
    )

    blocker_counts = Counter(str(row.get("blocker")) for row in active_summary.get("per_case", []))
    source_policy_notes = [
        "Step7T evidence is replay-sidecar evidence, not contest-runtime behavior.",
        "Current implementation reconstructs validation target positions for replay/evaluation.",
        "Runtime integration requires a live-layout adapter that audits active soft violations "
        "from optimizer-produced candidate positions rather than validation labels.",
    ]
    runtime_blockers = [
        "live_layout_adapter_missing",
        "validation_label_replay_baseline_not_runtime_input",
        "contest_runtime_finalizer_unchanged",
    ]

    sidecar_review_pass = (
        bool(active_summary.get("phase4_gate_open"))
        and str(visual_sanity.get("decision")) == "strict_winner_visual_sanity_pass"
        and int(visual_sanity.get("exact_strict_winner_count", 0)) >= 3
        and len(unique_strict_cases) >= 3
        and max_stored_err <= EPS
    )
    runtime_integration_gate_open = sidecar_review_pass and not runtime_blockers
    decision = (
        "phase4_review_pass_runtime_adapter_required"
        if sidecar_review_pass and runtime_blockers
        else "phase4_review_pass_runtime_integration_ready"
        if sidecar_review_pass
        else "phase4_review_failed"
    )

    comparison = {
        "step7q_objective_slot": {
            "strict_meaningful_winner_count": int(
                (step7q_summary or {}).get("strict_meaningful_winner_count", 0)
            ),
            "actual_all_vector_nonregressing_count": int(
                (step7q_summary or {}).get("actual_all_vector_nonregressing_count", 0)
            ),
            "phase4_gate_open": bool((step7q_summary or {}).get("phase4_gate_open", False)),
        },
        "step7r_close": {
            "decision": (step7r_decision or {}).get("decision"),
            "phase4_gate_open": bool((step7r_decision or {}).get("phase4_gate_open", False)),
            "strict_meaningful_winner_count": 0,
        },
        "step7s_certificate": {
            "terminal_result": (step7s_summary or {}).get("terminal_result"),
            "strict_winner_count": int((step7s_summary or {}).get("strict_winner_count", 0)),
            "kkt_stationary_count": int((step7s_summary or {}).get("kkt_stationary_count", 0)),
            "kkt_with_hinge_cap_count": int(
                (step7s_summary or {}).get("kkt_with_hinge_cap_count", 0)
            ),
        },
        "step7t_active_soft": {
            "candidate_count": int(active_summary.get("candidate_count", 0)),
            "strict_winner_count": int(active_summary.get("strict_winner_count", 0)),
            "strict_winner_case_count": int(active_summary.get("strict_winner_case_count", 0)),
            "phase4_gate_open": bool(active_summary.get("phase4_gate_open")),
            "exact_visual_strict_winner_count": int(
                visual_sanity.get("exact_strict_winner_count", 0)
            ),
        },
    }

    return {
        "schema": "step7t_phase4_review_v1",
        "decision": decision,
        "sidecar_phase4_review_pass": sidecar_review_pass,
        "runtime_integration_gate_open": runtime_integration_gate_open,
        "meaningful_cost_eps": meaningful_cost_eps,
        "eps_nonregression": EPS,
        "source_summary_path": source_summary_path,
        "visual_sanity_path": visual_sanity_path,
        "strict_winner_case_count": len(unique_strict_cases),
        "strict_winner_cases": unique_strict_cases,
        "strict_winner_replayed_count": len(winner_records),
        "max_stored_vs_exact_delta_error": max_stored_err,
        "winner_records": winner_records,
        "blocker_counts": dict(sorted(blocker_counts.items())),
        "comparison": comparison,
        "runtime_blockers": runtime_blockers,
        "source_policy_notes": source_policy_notes,
        "recommended_next_experiment": "step7v_live_layout_active_soft_adapter",
        "recommended_next_gate": (
            "Replay active-soft repair from live optimizer candidate positions and require "
            ">=3 exact strict meaningful winners without validation-label baseline dependence."
        ),
    }


def write_review_markdown(review: dict[str, Any]) -> str:
    lines = [
        "# Step7T Phase4 Review",
        "",
        f"Decision: `{review['decision']}`",
        "",
        f"- sidecar phase4 review pass: `{review['sidecar_phase4_review_pass']}`",
        f"- runtime integration gate open: `{review['runtime_integration_gate_open']}`",
        f"- strict winner cases: `{review['strict_winner_cases']}`",
        f"- max stored/exact delta error: `{review['max_stored_vs_exact_delta_error']}`",
        f"- next experiment: `{review['recommended_next_experiment']}`",
        "",
        "| case | candidate | ΔC | ΔH | ΔA | ΔS | hard | strict |",
        "|---:|---|---:|---:|---:|---:|:---:|:---:|",
    ]
    for row in review["winner_records"]:
        delta = row["delta_exact"]
        lines.append(
            (
                "| {case_id} | `{candidate_id}` | {dc:.8g} | {dh:.8g} | "
                "{da:.8g} | {ds:.8g} | {hard} | {strict} |"
            ).format(
                case_id=row["case_id"],
                candidate_id=row["candidate_id"],
                dc=delta["official_like_cost_delta"],
                dh=delta["hpwl_delta"],
                da=delta["bbox_area_delta"],
                ds=delta["soft_constraint_delta"],
                hard="yes" if row["hard_feasible"] else "no",
                strict="yes" if row["strict_meaningful_winner"] else "no",
            )
        )
    lines += [
        "",
        "## Runtime blockers",
        "",
        *[f"- `{item}`" for item in review["runtime_blockers"]],
        "",
        "## Next gate",
        "",
        review["recommended_next_gate"],
        "",
    ]
    return "\n".join(lines)
