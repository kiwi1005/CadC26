"""Step7P causal-operator branch comparison.

This sidecar tries three bounded request/operator-source strategies without
changing contest runtime.  The branch rows still use the Phase3 bounded replay
contract, so any improvement is evidence for the operator-source direction, not
a final official win claim.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from collections.abc import Callable
from pathlib import Path
from typing import Any

from puzzleplace.experiments.step7p_causal_request_replay import (
    FORBIDDEN_REQUEST_TERMS,
    VALIDATION_LABEL_POLICY,
    replay_request,
    request_from_row,
    write_json,
    write_jsonl,
)

MIN_BRANCH_REQUESTS = 96
MAX_CASE_SHARE = 0.25
BALANCED_HARD_TARGET = 60


def run_operator_branch_experiment(
    atlas_path: Path,
    baseline_replay_summary_path: Path,
    branch_rows_out_path: Path,
    summary_out_path: Path,
    markdown_out_path: Path,
) -> dict[str, Any]:
    rows = read_jsonl(atlas_path)
    baseline = load_json(baseline_replay_summary_path)
    eligible = [row for row in rows if eligible_source(row)]
    branches = [
        evaluate_branch(
            "branch_a_overlap_zero_hard_only",
            select_overlap_zero_hard_only(eligible, atlas_path),
            baseline,
        ),
        evaluate_branch(
            "branch_b_vector_guarded_narrow",
            select_vector_guarded_narrow(eligible, atlas_path),
            baseline,
        ),
        evaluate_branch(
            "branch_c_balanced_failure_budget",
            select_balanced_failure_budget(eligible, atlas_path),
            baseline,
        ),
    ]
    write_jsonl(branch_rows_out_path, branches)
    summary = summarize_branches(
        branches,
        baseline,
        atlas_path,
        baseline_replay_summary_path,
        branch_rows_out_path,
        summary_out_path,
        markdown_out_path,
    )
    write_json(summary_out_path, summary)
    markdown_out_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_out_path.write_text(branch_markdown(summary), encoding="utf-8")
    return summary


def eligible_source(row: dict[str, Any]) -> bool:
    if row.get("validation_label_policy") != VALIDATION_LABEL_POLICY:
        return False
    return not any(term in str(row).lower() for term in FORBIDDEN_REQUEST_TERMS)


def select_overlap_zero_hard_only(
    rows: list[dict[str, Any]], source_path: Path
) -> list[dict[str, Any]]:
    hard_rows = [row for row in rows if replay_source(row, source_path)["hard_feasible_nonnoop"]]
    return build_requests(
        balanced_fill(
            hard_rows,
            target_count=MIN_BRANCH_REQUESTS,
            min_per_case=3,
            max_per_case=24,
            key=hard_quality_key(source_path),
        ),
        source_path,
    )


def select_vector_guarded_narrow(
    rows: list[dict[str, Any]], source_path: Path
) -> list[dict[str, Any]]:
    guarded = [
        row
        for row in rows
        if replay_source(row, source_path)["actual_all_vector_nonregressing"]
    ]
    return build_requests(sorted(guarded, key=hard_quality_key(source_path)), source_path)


def select_balanced_failure_budget(
    rows: list[dict[str, Any]], source_path: Path
) -> list[dict[str, Any]]:
    hard_rows = [row for row in rows if replay_source(row, source_path)["hard_feasible_nonnoop"]]
    nonhard_rows = [
        row for row in rows if not replay_source(row, source_path)["hard_feasible_nonnoop"]
    ]
    selected = balanced_fill(
        hard_rows,
        target_count=BALANCED_HARD_TARGET,
        min_per_case=3,
        max_per_case=24,
        key=hard_quality_key(source_path),
    )
    selected_ids = {str(row["subproblem_id"]) for row in selected}
    case_counts = Counter(str(row["case_id"]) for row in selected)
    for row in sorted(nonhard_rows, key=nonhard_budget_key(source_path)):
        if len(selected) >= MIN_BRANCH_REQUESTS:
            break
        subproblem_id = str(row["subproblem_id"])
        case_id = str(row["case_id"])
        if subproblem_id in selected_ids or case_counts[case_id] >= 24:
            continue
        selected.append(row)
        selected_ids.add(subproblem_id)
        case_counts[case_id] += 1
    return build_requests(selected, source_path)


def balanced_fill(
    rows: list[dict[str, Any]],
    *,
    target_count: int,
    min_per_case: int,
    max_per_case: int,
    key: Callable[[dict[str, Any]], tuple[Any, ...]],
) -> list[dict[str, Any]]:
    by_case: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_case[str(row["case_id"])].append(row)
    selected: list[dict[str, Any]] = []
    selected_ids: set[str] = set()
    case_counts: Counter[str] = Counter()
    for case_id in sorted(by_case, key=lambda value: int(value)):
        for row in sorted(by_case[case_id], key=key)[:min_per_case]:
            selected.append(row)
            selected_ids.add(str(row["subproblem_id"]))
            case_counts[case_id] += 1
    for row in sorted(rows, key=key):
        if len(selected) >= target_count:
            break
        subproblem_id = str(row["subproblem_id"])
        case_id = str(row["case_id"])
        if subproblem_id in selected_ids or case_counts[case_id] >= max_per_case:
            continue
        selected.append(row)
        selected_ids.add(subproblem_id)
        case_counts[case_id] += 1
    return selected


def build_requests(rows: list[dict[str, Any]], source_path: Path) -> list[dict[str, Any]]:
    return [request_from_row(row, index, source_path) for index, row in enumerate(rows)]


def hard_quality_key(source_path: Path) -> Callable[[dict[str, Any]], tuple[Any, ...]]:
    def key(row: dict[str, Any]) -> tuple[Any, ...]:
        replay = replay_source(row, source_path)
        vector = replay["actual_objective_vector"]
        official = vector.get("official_like_cost_delta")
        return (
            not bool(replay["actual_all_vector_nonregressing"]),
            bool(replay["soft_regression"]),
            bool(replay["bbox_regression"]),
            999.0 if official is None else float(official),
            str(row["case_id"]),
            str(row["subproblem_id"]),
        )

    return key


def nonhard_budget_key(source_path: Path) -> Callable[[dict[str, Any]], tuple[Any, ...]]:
    def key(row: dict[str, Any]) -> tuple[Any, ...]:
        replay = replay_source(row, source_path)
        return (
            bool(replay["soft_regression"]),
            bool(replay["bbox_regression"]),
            str(row["case_id"]),
            str(row["subproblem_id"]),
        )

    return key


def replay_source(row: dict[str, Any], source_path: Path) -> dict[str, Any]:
    return replay_request(request_from_row(row, 0, source_path))


def evaluate_branch(
    branch_name: str, requests: list[dict[str, Any]], baseline: dict[str, Any]
) -> dict[str, Any]:
    replay_rows = [replay_request(row) for row in requests]
    request_count = len(replay_rows)
    case_counts = Counter(str(row["case_id"]) for row in replay_rows)
    hard = sum(int(row["hard_feasible_nonnoop"]) for row in replay_rows)
    overlap = sum(int(row["overlap_after_splice"]) for row in replay_rows)
    soft = sum(int(row["soft_regression"]) for row in replay_rows)
    bbox = sum(int(row["bbox_regression"]) for row in replay_rows)
    all_vector = sum(int(row["actual_all_vector_nonregressing"]) for row in replay_rows)
    strict = sum(int(row["strict_meaningful_winner"]) for row in replay_rows)
    soft_rate = soft / max(request_count, 1)
    bbox_rate = bbox / max(request_count, 1)
    largest_case_share = max(case_counts.values(), default=0) / max(request_count, 1)
    effective_partial = (
        request_count >= MIN_BRANCH_REQUESTS
        and len(case_counts) >= 8
        and largest_case_share <= MAX_CASE_SHARE
        and hard >= BALANCED_HARD_TARGET
        and overlap < int(baseline.get("overlap_after_splice_count", 0))
        and soft_rate < float(baseline.get("soft_regression_rate", 1.0))
        and bbox_rate < float(baseline.get("bbox_regression_rate", 1.0))
    )
    phase4_gate_open = (
        request_count >= MIN_BRANCH_REQUESTS
        and hard >= 60
        and overlap == 0
        and len({str(row["case_id"]) for row in replay_rows if row["hard_feasible_nonnoop"]}) >= 6
        and all_vector >= 30
        and soft_rate <= 0.10
        and bbox_rate <= 0.10
        and strict >= 3
    )
    return {
        "schema": "step7p_operator_branch_row_v1",
        "branch_name": branch_name,
        "request_count": request_count,
        "represented_case_count": len(case_counts),
        "case_counts": dict(case_counts),
        "largest_case_share": largest_case_share,
        "hard_feasible_nonnoop_count": hard,
        "overlap_after_splice_count": overlap,
        "overlap_after_splice_delta_vs_baseline": overlap
        - int(baseline.get("overlap_after_splice_count", 0)),
        "soft_regression_rate": soft_rate,
        "soft_regression_rate_delta_vs_baseline": soft_rate
        - float(baseline.get("soft_regression_rate", 0.0)),
        "bbox_regression_rate": bbox_rate,
        "bbox_regression_rate_delta_vs_baseline": bbox_rate
        - float(baseline.get("bbox_regression_rate", 0.0)),
        "actual_all_vector_nonregressing_count": all_vector,
        "strict_meaningful_winner_count": strict,
        "effective_partial_operator_source": effective_partial,
        "phase4_gate_open": phase4_gate_open,
        "gnn_rl_gate_open": False,
    }


def summarize_branches(
    branches: list[dict[str, Any]],
    baseline: dict[str, Any],
    atlas_path: Path,
    baseline_replay_summary_path: Path,
    branch_rows_out_path: Path,
    summary_out_path: Path,
    markdown_out_path: Path,
) -> dict[str, Any]:
    partial = [row for row in branches if row["effective_partial_operator_source"]]
    phase4 = [row for row in branches if row["phase4_gate_open"]]
    best = min(
        branches,
        key=lambda row: (
            not row["effective_partial_operator_source"],
            row["overlap_after_splice_count"],
            row["soft_regression_rate"],
            row["bbox_regression_rate"],
        ),
    )
    return {
        "schema": "step7p_operator_branch_summary_v1",
        "decision": "promote_branch_to_phase3_replay"
        if phase4
        else "partial_operator_source_found_phase4_still_closed"
        if partial
        else "stop_no_effective_operator_branch",
        "atlas_path": str(atlas_path),
        "baseline_replay_summary_path": str(baseline_replay_summary_path),
        "branch_rows_path": str(branch_rows_out_path),
        "summary_path": str(summary_out_path),
        "markdown_path": str(markdown_out_path),
        "baseline_metrics": {
            "request_count": baseline.get("request_count"),
            "overlap_after_splice_count": baseline.get("overlap_after_splice_count"),
            "soft_regression_rate": baseline.get("soft_regression_rate"),
            "bbox_regression_rate": baseline.get("bbox_regression_rate"),
            "strict_meaningful_winner_count": baseline.get("strict_meaningful_winner_count"),
            "phase4_gate_open": baseline.get("phase4_gate_open"),
        },
        "branch_count": len(branches),
        "effective_partial_branch_count": len(partial),
        "phase4_open_branch_count": len(phase4),
        "best_branch_name": best["branch_name"],
        "best_branch_metrics": {
            key: best[key]
            for key in [
                "request_count",
                "represented_case_count",
                "largest_case_share",
                "hard_feasible_nonnoop_count",
                "overlap_after_splice_count",
                "soft_regression_rate",
                "bbox_regression_rate",
                "strict_meaningful_winner_count",
                "effective_partial_operator_source",
                "phase4_gate_open",
            ]
        },
        "branches": branches,
        "gnn_rl_gate_open": False,
        "next_recommendation": (
            "use_best_partial_branch_as_operator_source_then_redesign_for_strict_winners"
        )
        if partial and not phase4
        else "run_phase4_for_open_branch"
        if phase4
        else "redesign_real_causal_operator",
    }


def branch_markdown(summary: dict[str, Any]) -> str:
    lines = ["# Step7P Operator Branch Experiment", "", f"Decision: `{summary['decision']}`", ""]
    lines.append(f"- branch_count: {summary['branch_count']}")
    lines.append(f"- effective_partial_branch_count: {summary['effective_partial_branch_count']}")
    lines.append(f"- phase4_open_branch_count: {summary['phase4_open_branch_count']}")
    lines.append(f"- best_branch_name: {summary['best_branch_name']}")
    lines.append(f"- gnn_rl_gate_open: {summary['gnn_rl_gate_open']}")
    lines.append("")
    for row in summary["branches"]:
        lines.append(f"## {row['branch_name']}")
        lines.append(f"- request_count: {row['request_count']}")
        lines.append(f"- represented_case_count: {row['represented_case_count']}")
        lines.append(f"- hard_feasible_nonnoop_count: {row['hard_feasible_nonnoop_count']}")
        lines.append(f"- overlap_after_splice_count: {row['overlap_after_splice_count']}")
        lines.append(f"- soft_regression_rate: {row['soft_regression_rate']}")
        lines.append(f"- bbox_regression_rate: {row['bbox_regression_rate']}")
        lines.append(f"- strict_meaningful_winner_count: {row['strict_meaningful_winner_count']}")
        lines.append(
            f"- effective_partial_operator_source: {row['effective_partial_operator_source']}"
        )
        lines.append(f"- phase4_gate_open: {row['phase4_gate_open']}")
        lines.append("")
    return "\n".join(lines)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


__all__ = [
    "run_operator_branch_experiment",
]
