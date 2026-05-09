"""Step7P Phase0 stagnation lock.

This module is intentionally read-only over prior Step7 research artifacts. It
records the negative/closed gates that justify starting Step7P-CCR and forbids
reopening Step7O Phase3, Step7N reservoirs, Step7M micro widening, or GNN/RL.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

FORBIDDEN_NEXT_STEPS = (
    "step7o_phase3_masked_replay",
    "step7n_reservoir_reopen",
    "step7m_micro_axis_widening",
    "gnn_rl_training",
)


def build_stagnation_lock(
    step7l_summary_path: Path,
    step7m_phase2_path: Path,
    step7m_phase3_path: Path,
    step7m_phase4_path: Path,
    step7n_phase0_path: Path,
    step7ml_i_quality_path: Path,
    step7ml_j_quality_path: Path,
    step7ml_k_quality_path: Path,
    step7n_i_target_quality_path: Path,
    step7o_phase2_path: Path,
    out_path: Path,
    markdown_out_path: Path,
    candidate_audit_out_path: Path,
) -> dict[str, Any]:
    """Write Step7P Phase0 lock, markdown, and candidate-universe audit."""

    inputs = {
        "step7l_summary": load_json(step7l_summary_path),
        "step7m_phase2": load_json(step7m_phase2_path),
        "step7m_phase3": load_json(step7m_phase3_path),
        "step7m_phase4": load_json(step7m_phase4_path),
        "step7n_phase0": load_json(step7n_phase0_path),
        "step7ml_i_quality": load_json(step7ml_i_quality_path),
        "step7ml_j_quality": load_json(step7ml_j_quality_path),
        "step7ml_k_quality": load_json(step7ml_k_quality_path),
        "step7n_i_target_quality": load_json(step7n_i_target_quality_path),
        "step7o_phase2": load_json(step7o_phase2_path),
    }
    paths = {
        "step7l_summary": step7l_summary_path,
        "step7m_phase2": step7m_phase2_path,
        "step7m_phase3": step7m_phase3_path,
        "step7m_phase4": step7m_phase4_path,
        "step7n_phase0": step7n_phase0_path,
        "step7ml_i_quality": step7ml_i_quality_path,
        "step7ml_j_quality": step7ml_j_quality_path,
        "step7ml_k_quality": step7ml_k_quality_path,
        "step7n_i_target_quality": step7n_i_target_quality_path,
        "step7o_phase2": step7o_phase2_path,
    }
    audit_rows = candidate_universe_audit(inputs, paths)
    write_jsonl(candidate_audit_out_path, audit_rows)
    summary = summarize_lock(
        inputs,
        paths,
        audit_rows,
        out_path,
        markdown_out_path,
        candidate_audit_out_path,
    )
    write_json(out_path, summary)
    markdown_out_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_out_path.write_text(stagnation_markdown(summary), encoding="utf-8")
    return summary


def summarize_lock(
    inputs: dict[str, Any],
    paths: dict[str, Path],
    audit_rows: list[dict[str, Any]],
    out_path: Path,
    markdown_out_path: Path,
    candidate_audit_out_path: Path,
) -> dict[str, Any]:
    step7o = dict_value(inputs["step7o_phase2"])
    step7n = dict_value(inputs["step7n_phase0"])
    step7m_phase2 = dict_value(inputs["step7m_phase2"])
    step7m_phase3 = dict_value(inputs["step7m_phase3"])
    step7m_phase4 = dict_value(inputs["step7m_phase4"])
    step7ml_i = dict_value(inputs["step7ml_i_quality"])
    step7ml_k = dict_value(dict_value(inputs["step7ml_k_quality"]).get("summary"))
    step7ml_winner_baseline = max(
        int_value(step7ml_i.get("official_like_improving_count")),
        int_value(step7ml_k.get("official_like_improving_count")),
        int_value(step7o.get("step7ml_winner_baseline")),
    )
    step7m_meaningful_winner_count = (
        int_value(step7m_phase2.get("meaningful_official_like_improving_count"))
        + int_value(step7m_phase3.get("meaningful_official_like_improving_total"))
        + int_value(step7m_phase4.get("meaningful_official_like_improving_count"))
    )
    missing_inputs = [name for name, path in paths.items() if not path.exists()]
    gate_reasons = {
        "inputs_present": not missing_inputs,
        "step7o_phase3_closed": step7o.get("phase3_gate_open") is False,
        "step7o_concentration_failed": step7o.get("concentration_pass") is False,
        "step7n_archive_closed": int_value(step7n.get("strict_archive_candidate_count")) == 0,
        "step7n_phase1_closed": step7n.get("phase1_gate_open") is False,
        "step7m_meaningful_winners_zero": step7m_meaningful_winner_count == 0,
        "step7ml_winner_baseline_is_two": step7ml_winner_baseline == 2,
    }
    phase1_gate_open = all(gate_reasons.values())
    return {
        "schema": "step7p_phase0_stagnation_lock_v1",
        "decision": "start_causal_closure_repack"
        if phase1_gate_open
        else "stop_missing_stagnation_evidence",
        "summary_path": str(out_path),
        "markdown_path": str(markdown_out_path),
        "candidate_audit_path": str(candidate_audit_out_path),
        "input_paths": {name: str(path) for name, path in paths.items()},
        "missing_inputs": missing_inputs,
        "gate_reasons": gate_reasons,
        "phase1_gate_open": phase1_gate_open,
        "step7o_decision": step7o.get("decision"),
        "step7o_phase3_gate_open": step7o.get("phase3_gate_open"),
        "step7o_concentration_pass": step7o.get("concentration_pass"),
        "step7o_top_budget_official_like_improving_count": step7o.get(
            "top_budget_official_like_improving_count"
        ),
        "step7ml_winner_baseline": step7ml_winner_baseline,
        "step7n_strict_archive_candidate_count": int_value(
            step7n.get("strict_archive_candidate_count")
        ),
        "step7n_strict_meaningful_non_micro_winner_count": int_value(
            step7n.get("strict_meaningful_non_micro_winner_count")
        ),
        "step7m_meaningful_winner_count": step7m_meaningful_winner_count,
        "existing_candidate_universe_has_new_winner_signal": step7ml_winner_baseline > 2,
        "candidate_audit_row_count": len(audit_rows),
        "forbidden_next_steps": list(FORBIDDEN_NEXT_STEPS),
        "gnn_rl_gate_open": False,
        "next_recommendation": "build_step7p_phase1_causal_subproblem_atlas"
        if phase1_gate_open
        else "repair_missing_or_contradictory_stagnation_evidence",
    }


def candidate_universe_audit(
    inputs: dict[str, Any], paths: dict[str, Path]
) -> list[dict[str, Any]]:
    step7ml_j = dict_value(dict_value(inputs["step7ml_j_quality"]).get("summary"))
    step7ml_k = dict_value(dict_value(inputs["step7ml_k_quality"]).get("summary"))
    step7n_target = dict_value(inputs["step7n_i_target_quality"])
    return [
        audit_row(
            "step7ml_i_quality",
            paths["step7ml_i_quality"],
            "decoded_candidate_quality",
            inputs["step7ml_i_quality"],
            official_like_improving_count=int_value(
                dict_value(inputs["step7ml_i_quality"]).get("official_like_improving_count")
            ),
            quality_gate_pass_count=int_value(
                dict_value(inputs["step7ml_i_quality"]).get("quality_gate_pass_count")
            ),
        ),
        audit_row(
            "step7ml_j_quality",
            paths["step7ml_j_quality"],
            "ranked_candidate_budget",
            step7ml_j,
            official_like_improving_count=int_value(step7ml_j.get("official_like_improving_count")),
            quality_gate_pass_count=int_value(step7ml_j.get("quality_gate_pass_count")),
            bbox_regression_count=int_value(step7ml_j.get("bbox_regression_count")),
            soft_regression_count=int_value(step7ml_j.get("soft_regression_count")),
            hpwl_gain_but_official_like_loss_count=int_value(
                step7ml_j.get("hpwl_gain_but_official_like_loss_count")
            ),
        ),
        audit_row(
            "step7ml_k_quality",
            paths["step7ml_k_quality"],
            "invariant_candidate_universe",
            step7ml_k,
            official_like_improving_count=int_value(step7ml_k.get("official_like_improving_count")),
            quality_gate_pass_count=int_value(step7ml_k.get("quality_gate_pass_count")),
            bbox_regression_count=int_value(step7ml_k.get("bbox_regression_count")),
            soft_regression_count=int_value(step7ml_k.get("soft_regression_count")),
            metric_regressing_count=int_value(step7ml_k.get("metric_regressing_count")),
            dominated_by_original_count=int_value(step7ml_k.get("dominated_by_original_count")),
        ),
        audit_row(
            "step7n_i_target_quality",
            paths["step7n_i_target_quality"],
            "target_failure_diagnostics",
            step7n_target,
            baseline_winner_count=int_value(
                dict_value(step7n_target.get("target_screening")).get("baseline_winner_count")
            ),
            baseline_regressing_hard_feasible_non_noop_count=int_value(
                dict_value(step7n_target.get("target_screening")).get(
                    "baseline_regressing_hard_feasible_non_noop_count"
                )
            ),
        ),
        audit_row(
            "step7o_phase2",
            paths["step7o_phase2"],
            "prior_filter_report_only",
            inputs["step7o_phase2"],
            official_like_improving_count=int_value(
                dict_value(inputs["step7o_phase2"]).get("top_budget_official_like_improving_count")
            ),
            bbox_regression_count=int_value(
                dict_value(inputs["step7o_phase2"]).get("top_budget_bbox_regression_count")
            ),
            soft_regression_count=int_value(
                dict_value(inputs["step7o_phase2"]).get("top_budget_soft_regression_count")
            ),
        ),
    ]


def audit_row(
    name: str,
    path: Path,
    artifact_role: str,
    payload: Any,
    **metrics: int,
) -> dict[str, Any]:
    return {
        "schema": "step7p_phase0_candidate_universe_audit_row_v1",
        "source_name": name,
        "source_artifact": str(path),
        "artifact_role": artifact_role,
        "source_schema": dict_value(payload).get("schema"),
        "metrics": metrics,
        "supports_generation": False,
        "diagnostic_only": True,
    }


def stagnation_markdown(summary: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# Step7P Phase0 Stagnation Lock",
            "",
            f"Decision: `{summary['decision']}`",
            "",
            f"- phase1_gate_open: {summary['phase1_gate_open']}",
            f"- step7o_phase3_gate_open: {summary['step7o_phase3_gate_open']}",
            f"- step7ml_winner_baseline: {summary['step7ml_winner_baseline']}",
            f"- step7n_strict_archive_candidate_count: "
            f"{summary['step7n_strict_archive_candidate_count']}",
            f"- step7m_meaningful_winner_count: {summary['step7m_meaningful_winner_count']}",
            f"- forbidden_next_steps: {', '.join(summary['forbidden_next_steps'])}",
            f"- next_recommendation: {summary['next_recommendation']}",
            "",
        ]
    )


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


def int_value(value: Any) -> int:
    if isinstance(value, bool) or value is None:
        return 0
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def dict_value(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}
