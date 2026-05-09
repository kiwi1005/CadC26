"""Step7P gated RL/GNN readiness for causal-operator learning.

This gate does not promote contest runtime, Phase4 ablation, or a final solver.
It opens only a constrained learning phase when hand-designed Step7P operators
have produced enough broad, labeled failure/improvement data to justify learning
a causal operator policy or graph scorer.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

from puzzleplace.experiments.step7p_causal_request_replay import read_jsonl, write_json

MIN_ATLAS_ROWS = 80
MIN_CASES = 8
MIN_BRANCH_ROWS = 3
MIN_EFFECTIVE_BRANCHES = 1
MIN_ELIGIBLE_EXACT_HARD = 96
MIN_ALL_VECTOR = 30
MAX_LARGEST_CASE_SHARE = 0.35


def evaluate_gnn_rl_readiness(
    atlas_path: Path,
    branch_summary_path: Path,
    blocker_diagnosis_path: Path,
    replay_summary_path: Path,
    summary_out_path: Path,
    markdown_out_path: Path,
) -> dict[str, Any]:
    atlas_rows = read_jsonl(atlas_path)
    branch_summary = load_json(branch_summary_path)
    blocker = load_json(blocker_diagnosis_path)
    replay = load_json(replay_summary_path)
    case_counts = Counter(str(row.get("case_id")) for row in atlas_rows)
    failure_counts = Counter(str(row.get("seed_failure_bucket")) for row in atlas_rows)
    intent_counts = Counter(str(row.get("intent_family")) for row in atlas_rows)
    best_branch = branch_summary.get("best_branch_metrics")
    best = best_branch if isinstance(best_branch, dict) else {}
    criteria = {
        "atlas_broad_enough": len(atlas_rows) >= MIN_ATLAS_ROWS,
        "atlas_case_coverage_ok": len(case_counts) >= MIN_CASES,
        "atlas_not_case_concentrated": largest_share(case_counts) <= MAX_LARGEST_CASE_SHARE,
        "failure_labels_diverse": sum(1 for value in failure_counts.values() if value > 0) >= 4,
        "intent_labels_diverse": sum(1 for value in intent_counts.values() if value > 0) >= 4,
        "three_branch_experiment_done": int(branch_summary.get("branch_count", 0))
        >= MIN_BRANCH_ROWS,
        "partial_operator_branch_found": int(
            branch_summary.get("effective_partial_branch_count", 0)
        )
        >= MIN_EFFECTIVE_BRANCHES,
        "best_branch_reduces_overlap": float(best.get("overlap_after_splice_count", 9999))
        < float(replay.get("overlap_after_splice_count", 9999)),
        "best_branch_reduces_soft_regression": float(best.get("soft_regression_rate", 1.0))
        < float(replay.get("soft_regression_rate", 1.0)),
        "best_branch_reduces_bbox_regression": float(best.get("bbox_regression_rate", 1.0))
        < float(replay.get("bbox_regression_rate", 1.0)),
        "manual_operator_still_not_phase4": branch_summary.get("phase4_open_branch_count") == 0
        and replay.get("phase4_gate_open") is False,
        "strict_source_absent": int(blocker.get("strict_meaningful_source_count", -1)) == 0,
        "eligible_exact_hard_corpus_ok": int(
            blocker.get("eligible_exact_hard_nonforbidden_count", 0)
        )
        >= MIN_ELIGIBLE_EXACT_HARD,
        "all_vector_signal_ok": int(blocker.get("all_vector_nonregressing_source_count", 0))
        >= MIN_ALL_VECTOR,
    }
    gnn_rl_gate_open = all(criteria.values())
    summary = {
        "schema": "step7p_gnn_rl_readiness_summary_v1",
        "decision": "open_constrained_operator_learning_phase"
        if gnn_rl_gate_open
        else "keep_gnn_rl_closed",
        "atlas_path": str(atlas_path),
        "branch_summary_path": str(branch_summary_path),
        "blocker_diagnosis_path": str(blocker_diagnosis_path),
        "replay_summary_path": str(replay_summary_path),
        "summary_path": str(summary_out_path),
        "markdown_path": str(markdown_out_path),
        "gnn_rl_gate_open": gnn_rl_gate_open,
        "allowed_phase": "step7q_constrained_operator_learning" if gnn_rl_gate_open else None,
        "forbidden_scope": [
            "contest_runtime_integration",
            "finalizer_semantic_changes",
            "phase4_ablation_without_phase4_gate",
            "rl_direct_coordinate_solver",
            "gnn_without_vector_replay_gate",
        ],
        "required_learning_targets": [
            "overlap_after_splice_risk",
            "soft_regression_risk",
            "bbox_regression_risk",
            "all_vector_nonregressing_policy",
            "strict_winner_candidate_prior",
        ],
        "criteria": criteria,
        "atlas_row_count": len(atlas_rows),
        "represented_case_count": len(case_counts),
        "largest_case_share": largest_share(case_counts),
        "failure_bucket_counts": dict(failure_counts),
        "intent_family_counts": dict(intent_counts),
        "best_branch_name": branch_summary.get("best_branch_name"),
        "best_branch_metrics": best,
        "next_recommendation": "start_step7q_gnn_rl_operator_learning_prd"
        if gnn_rl_gate_open
        else "collect_more_operator_signal_before_gnn_rl",
    }
    write_json(summary_out_path, summary)
    markdown_out_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_out_path.write_text(readiness_markdown(summary), encoding="utf-8")
    return summary


def largest_share(counter: Counter[str]) -> float:
    total = sum(counter.values())
    if total == 0:
        return 0.0
    return max(counter.values()) / total


def readiness_markdown(summary: dict[str, Any]) -> str:
    lines = ["# Step7P GNN/RL Readiness", "", f"Decision: `{summary['decision']}`", ""]
    lines.append(f"- gnn_rl_gate_open: {summary['gnn_rl_gate_open']}")
    lines.append(f"- allowed_phase: {summary['allowed_phase']}")
    lines.append(f"- best_branch_name: {summary['best_branch_name']}")
    lines.append(f"- atlas_row_count: {summary['atlas_row_count']}")
    lines.append(f"- represented_case_count: {summary['represented_case_count']}")
    lines.append(f"- largest_case_share: {summary['largest_case_share']}")
    lines.append("")
    lines.append("## Criteria")
    for name, passed in summary["criteria"].items():
        lines.append(f"- {name}: {passed}")
    lines.append("")
    return "\n".join(lines)


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


__all__ = ["evaluate_gnn_rl_readiness"]
