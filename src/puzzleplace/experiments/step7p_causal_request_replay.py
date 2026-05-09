"""Step7P Phase3/4 causal request, bounded replay, and family ablation.

Phase3 remains sidecar-only. The replay step evaluates causal move-intent
requests against the exact component metrics already attached to their seed
rows; rows without exact objective vectors are fail-closed as proxy-only. This
prevents pretending that a validation geometry repacker exists before the repo
has implemented one.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import Any

MEANINGFUL_COST_EPS = 1e-7
MAX_REQUESTS_PER_CASE = 16
MIN_REQUESTS_PER_CASE = 12
VALIDATION_LABEL_POLICY = "labels used for replay/evaluation only, not request generation"
FORBIDDEN_REQUEST_TERMS = ("micro_axis_corridor", "soft_repair_budgeted", "hpwl_only")


def generate_causal_requests(
    subproblem_atlas_path: Path,
    operator_contract_path: Path,
    requests_out_path: Path,
    summary_out_path: Path,
    markdown_out_path: Path,
) -> dict[str, Any]:
    contract = load_json(operator_contract_path)
    rows = read_jsonl(subproblem_atlas_path)
    if contract.get("phase3_gate_open") is not True:
        requests: list[dict[str, Any]] = []
    else:
        requests = select_requests(rows, subproblem_atlas_path)
    write_jsonl(requests_out_path, requests)
    summary = summarize_requests(
        requests,
        contract,
        subproblem_atlas_path=subproblem_atlas_path,
        operator_contract_path=operator_contract_path,
        requests_out_path=requests_out_path,
        summary_out_path=summary_out_path,
        markdown_out_path=markdown_out_path,
    )
    write_json(summary_out_path, summary)
    markdown_out_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_out_path.write_text(request_markdown(summary), encoding="utf-8")
    return summary


def select_requests(rows: list[dict[str, Any]], source_path: Path) -> list[dict[str, Any]]:
    eligible = [row for row in rows if request_eligible(row)]
    by_case: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in sorted(eligible, key=request_sort_key):
        by_case[str(row["case_id"])].append(row)
    selected: list[dict[str, Any]] = []
    for case_id in sorted(by_case, key=lambda value: int(value)):
        for row in by_case[case_id][:MAX_REQUESTS_PER_CASE]:
            selected.append(request_from_row(row, len(selected), source_path))
    # Keep the deck bounded while preserving balanced per-case coverage.
    if len(selected) > 192:
        selected = selected[:192]
    return selected


def request_eligible(row: dict[str, Any]) -> bool:
    if row.get("validation_label_policy") != VALIDATION_LABEL_POLICY:
        return False
    if any(term in str(row).lower() for term in FORBIDDEN_REQUEST_TERMS):
        return False
    if str(row.get("seed_failure_bucket")) == "unknown":
        return non_forbidden_case_coverage_fallback(row)
    return True


def non_forbidden_case_coverage_fallback(row: dict[str, Any]) -> bool:
    """Allow exact non-forbidden closure seeds when a case has no labeled source.

    Case 51 currently has no non-forbidden non-unknown rows after the Step7M
    micro-axis and soft-budgeted sources are removed.  Its Step7M Phase4
    multiblock closure rows are still useful request-source coverage probes:
    they are exact, hard-feasible, non-noop, and carry no forbidden micro terms.
    They must remain unknown for attribution, but the request deck can include
    them so the *next* gate tests whether replay evidence is broad enough.
    """

    vector = objective_vector(row.get("objective_vector"))
    exact_vector = all(value is not None for value in vector.values())
    return (
        exact_vector
        and bool(row.get("hard_feasible_nonnoop"))
        and str(row.get("intent_family")) == "closure_translate_with_repair"
        and str(row.get("seed_source")) == "step7m_phase4"
    )


def request_sort_key(row: dict[str, Any]) -> tuple[int, int, str, str]:
    priority = {
        "soft_regression": 0,
        "bbox_regression": 1,
        "hpwl_gain_but_official_like_loss": 2,
        "wrong_slot": 3,
        "overlap_after_splice": 4,
        "bad_internal_repack": 5,
    }.get(str(row.get("seed_failure_bucket")), 9)
    return (
        priority,
        -int(bool(row.get("hard_feasible_nonnoop"))),
        str(row["case_id"]),
        str(row["subproblem_id"]),
    )


def request_from_row(row: dict[str, Any], index: int, source_path: Path) -> dict[str, Any]:
    return {
        "schema": "step7p_phase3_causal_request_v1",
        "request_id": f"step7p_req_{index:04d}",
        "case_id": str(row["case_id"]),
        "source_atlas_path": str(source_path),
        "source_subproblem_id": row.get("subproblem_id"),
        "source_candidate_id": row.get("seed_candidate_id"),
        "intent_family": row.get("intent_family"),
        "seed_failure_bucket": row.get("seed_failure_bucket"),
        "seed_source": row.get("seed_source"),
        "seed_metric_confidence": row.get("metric_confidence"),
        "seed_objective_vector": row.get("objective_vector"),
        "seed_hard_feasible_nonnoop": row.get("hard_feasible_nonnoop"),
        "affected_block_ids": row.get("affected_block_ids", []),
        "blocker_block_ids": row.get("blocker_block_ids", []),
        "soft_linked_block_ids": row.get("soft_linked_block_ids", []),
        "allowed_repack_families": row.get("allowed_repack_families", []),
        "vector_guard_policy": {
            "hpwl": "nonregress_or_improve",
            "bbox": "nonregress",
            "soft": "nonregress",
        },
        "forbidden": list(FORBIDDEN_REQUEST_TERMS),
        "non_micro_intent": True,
        "validation_label_policy": VALIDATION_LABEL_POLICY,
        "request_source_policy": "direct_causal_attribution"
        if str(row.get("seed_failure_bucket")) != "unknown"
        else "non_forbidden_exact_closure_coverage_fallback",
    }


def summarize_requests(
    requests: list[dict[str, Any]],
    contract: dict[str, Any],
    *,
    subproblem_atlas_path: Path,
    operator_contract_path: Path,
    requests_out_path: Path,
    summary_out_path: Path,
    markdown_out_path: Path,
) -> dict[str, Any]:
    case_counts = Counter(str(row["case_id"]) for row in requests)
    request_count = len(requests)
    largest_case_share = largest_case(case_counts)[1] / max(request_count, 1)
    case024_025_share = (case_counts.get("24", 0) + case_counts.get("25", 0)) / max(
        request_count, 1
    )
    unique_signatures = {request_signature(row) for row in requests}
    non_micro_share = sum(int(bool(row["non_micro_intent"])) for row in requests) / max(
        request_count, 1
    )
    forbidden_count = forbidden_term_count(requests)
    replay_gate_open = (
        contract.get("phase3_gate_open") is True
        and 96 <= request_count <= 192
        and len(case_counts) >= 8
        and len(unique_signatures) >= 96
        and non_micro_share >= 0.80
        and largest_case_share <= 0.25
        and case024_025_share <= 0.40
        and forbidden_count == 0
    )
    return {
        "schema": "step7p_phase3_causal_request_summary_v1",
        "decision": "promote_to_bounded_causal_replay" if replay_gate_open else "stop_request_gate",
        "subproblem_atlas_path": str(subproblem_atlas_path),
        "operator_contract_path": str(operator_contract_path),
        "requests_path": str(requests_out_path),
        "summary_path": str(summary_out_path),
        "markdown_path": str(markdown_out_path),
        "request_count": request_count,
        "represented_case_count": len(case_counts),
        "case_counts": dict(case_counts),
        "largest_case_share": largest_case_share,
        "case024_case025_request_share": case024_025_share,
        "unique_request_signature_count": len(unique_signatures),
        "non_micro_intent_share": non_micro_share,
        "forbidden_request_term_count": forbidden_count,
        "phase3_replay_gate_open": replay_gate_open,
        "gnn_rl_gate_open": False,
        "next_recommendation": "run_bounded_causal_replay"
        if replay_gate_open
        else "rework_request_deck",
    }


def replay_causal_requests(
    requests_path: Path,
    replay_rows_out_path: Path,
    summary_out_path: Path,
    markdown_out_path: Path,
    failures_out_path: Path,
) -> dict[str, Any]:
    requests = read_jsonl(requests_path)
    replay_rows = [replay_request(row) for row in requests]
    write_jsonl(replay_rows_out_path, replay_rows)
    failures = failures_by_case(replay_rows)
    write_json(failures_out_path, failures)
    summary = summarize_replay(
        replay_rows,
        requests_path=requests_path,
        replay_rows_out_path=replay_rows_out_path,
        summary_out_path=summary_out_path,
        markdown_out_path=markdown_out_path,
        failures_out_path=failures_out_path,
    )
    write_json(summary_out_path, summary)
    markdown_out_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_out_path.write_text(replay_markdown(summary), encoding="utf-8")
    return summary


def replay_request(request: dict[str, Any]) -> dict[str, Any]:
    vector = objective_vector(request.get("seed_objective_vector"))
    exact = all(value is not None for value in vector.values())
    hard_feasible = bool(request.get("seed_hard_feasible_nonnoop")) and exact
    soft_regression = exact and float(vector["soft_constraint_delta"] or 0.0) > 0
    bbox_regression = exact and float(vector["bbox_area_delta"] or 0.0) > 0
    hpwl_regression = exact and float(vector["hpwl_delta"] or 0.0) > 0
    official_delta = vector["official_like_cost_delta"]
    strict_winner = (
        hard_feasible
        and official_delta is not None
        and official_delta < -MEANINGFUL_COST_EPS
        and not soft_regression
        and not bbox_regression
        and not hpwl_regression
    )
    all_vector_nonregressing = (
        hard_feasible and not soft_regression and not bbox_regression and not hpwl_regression
    )
    return {
        "schema": "step7p_phase3_causal_replay_row_v1",
        "request_id": request.get("request_id"),
        "case_id": request.get("case_id"),
        "candidate_id": f"{request.get('request_id')}:bounded_seed_metric_replay",
        "intent_family": request.get("intent_family"),
        "source_candidate_id": request.get("source_candidate_id"),
        "seed_failure_bucket": request.get("seed_failure_bucket"),
        "replay_scope": "bounded_existing_metric_replay_no_validation_geometry_repack",
        "fresh_metric_available": exact,
        "hard_feasible_nonnoop": hard_feasible,
        "overlap_after_splice": 0 if hard_feasible else 1,
        "actual_objective_vector": vector,
        "hpwl_delta": vector["hpwl_delta"],
        "bbox_area_delta": vector["bbox_area_delta"],
        "soft_constraint_delta": vector["soft_constraint_delta"],
        "official_like_cost_delta": official_delta,
        "actual_all_vector_nonregressing": all_vector_nonregressing,
        "soft_regression": soft_regression,
        "bbox_regression": bbox_regression,
        "hpwl_regression": hpwl_regression,
        "strict_meaningful_winner": strict_winner,
        "quality_gate_pass": strict_winner,
        "validation_label_policy": VALIDATION_LABEL_POLICY,
    }


def summarize_replay(
    rows: list[dict[str, Any]],
    *,
    requests_path: Path,
    replay_rows_out_path: Path,
    summary_out_path: Path,
    markdown_out_path: Path,
    failures_out_path: Path,
) -> dict[str, Any]:
    request_count = len(rows)
    hard = [row for row in rows if row["hard_feasible_nonnoop"]]
    strict = [row for row in rows if row["strict_meaningful_winner"]]
    all_vector = [row for row in rows if row["actual_all_vector_nonregressing"]]
    bbox_regressions = sum(int(row["bbox_regression"]) for row in rows)
    soft_regressions = sum(int(row["soft_regression"]) for row in rows)
    strict_cases = Counter(str(row["case_id"]) for row in strict)
    phase4_gate_open = (
        len(hard) >= 60
        and sum(int(row["overlap_after_splice"]) for row in rows) == 0
        and len({str(row["case_id"]) for row in hard}) >= 6
        and len(all_vector) >= 30
        and soft_regressions / max(request_count, 1) <= 0.10
        and bbox_regressions / max(request_count, 1) <= 0.10
        and len(strict) >= 3
        and sum(int(str(row["case_id"]) not in {"24", "25"}) for row in strict) >= 1
        and len({row["source_candidate_id"] for row in strict}) >= 3
    )
    return {
        "schema": "step7p_phase3_causal_replay_summary_v1",
        "decision": "promote_to_causal_family_ablation"
        if phase4_gate_open
        else "stop_phase3_replay_gate",
        "request_path": str(requests_path),
        "replay_rows_path": str(replay_rows_out_path),
        "summary_path": str(summary_out_path),
        "markdown_path": str(markdown_out_path),
        "failures_by_case_path": str(failures_out_path),
        "request_count": request_count,
        "hard_feasible_nonnoop_count": len(hard),
        "overlap_after_splice_count": sum(int(row["overlap_after_splice"]) for row in rows),
        "represented_hard_feasible_case_count": len({str(row["case_id"]) for row in hard}),
        "actual_all_vector_nonregressing_count": len(all_vector),
        "soft_regression_rate": soft_regressions / max(request_count, 1),
        "bbox_regression_rate": bbox_regressions / max(request_count, 1),
        "strict_meaningful_winner_count": len(strict),
        "strict_winner_case_counts": dict(strict_cases),
        "non_case024_non_case025_strict_winner_count": sum(
            int(str(row["case_id"]) not in {"24", "25"}) for row in strict
        ),
        "unique_strict_winner_signature_count": len({row["source_candidate_id"] for row in strict}),
        "phase4_gate_open": phase4_gate_open,
        "gnn_rl_gate_open": False,
        "next_recommendation": "run_family_ablation"
        if phase4_gate_open
        else "stop_step7p_or_rework_causal_operator",
    }


def run_family_ablation(
    replay_rows_path: Path,
    rows_out_path: Path,
    summary_out_path: Path,
    markdown_out_path: Path,
) -> dict[str, Any]:
    replay_summary_path = replay_rows_path.with_name("step7p_phase3_causal_replay_summary.json")
    if replay_summary_path.exists():
        replay_summary = load_json(replay_summary_path)
        if replay_summary.get("phase4_gate_open") is not True:
            rows: list[dict[str, Any]] = []
            write_jsonl(rows_out_path, rows)
            summary = blocked_ablation_summary(
                replay_summary,
                rows_out_path,
                summary_out_path,
                markdown_out_path,
            )
            write_json(summary_out_path, summary)
            markdown_out_path.parent.mkdir(parents=True, exist_ok=True)
            markdown_out_path.write_text(ablation_markdown(summary), encoding="utf-8")
            return summary
    replay_rows = read_jsonl(replay_rows_path)
    families = sorted({str(row.get("intent_family")) for row in replay_rows})
    rows = [ablate_family(family, replay_rows) for family in families]
    write_jsonl(rows_out_path, rows)
    summary = summarize_ablation(rows, rows_out_path, summary_out_path, markdown_out_path)
    write_json(summary_out_path, summary)
    markdown_out_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_out_path.write_text(ablation_markdown(summary), encoding="utf-8")
    return summary


def blocked_ablation_summary(
    replay_summary: dict[str, Any],
    rows_out_path: Path,
    summary_out_path: Path,
    markdown_out_path: Path,
) -> dict[str, Any]:
    return {
        "schema": "step7p_phase4_causal_ablation_summary_v1",
        "decision": "blocked_by_phase3_replay_gate",
        "replay_summary_path": str(replay_summary.get("summary_path", "")),
        "rows_path": str(rows_out_path),
        "summary_path": str(summary_out_path),
        "markdown_path": str(markdown_out_path),
        "family_count": 0,
        "passed_family_count": 0,
        "passed_families": [],
        "gnn_rl_gate_open": False,
        "blocker": replay_summary.get("decision"),
        "phase4_gate_open": False,
        "next_recommendation": "stop_step7p_or_rework_causal_operator",
    }


def ablate_family(family: str, replay_rows: list[dict[str, Any]]) -> dict[str, Any]:
    rows = [row for row in replay_rows if str(row.get("intent_family")) == family]
    replayed_count = len(rows)
    hard_count = sum(int(row["hard_feasible_nonnoop"]) for row in rows)
    all_vector = sum(int(row["actual_all_vector_nonregressing"]) for row in rows)
    soft_regressions = sum(int(row["soft_regression"]) for row in rows)
    strict = [row for row in rows if row["strict_meaningful_winner"]]
    winner_case_counts = Counter(str(row["case_id"]) for row in strict)
    largest_winner_share = largest_case(winner_case_counts)[1] / max(len(strict), 1)
    family_pass = (
        replayed_count >= 20
        and hard_count / max(replayed_count, 1) >= 0.80
        and all_vector / max(replayed_count, 1) >= 0.50
        and soft_regressions / max(replayed_count, 1) <= 0.10
        and len(strict) >= 1
        and largest_winner_share <= 0.50
    )
    return {
        "schema": "step7p_phase4_causal_ablation_row_v1",
        "intent_family": family,
        "replayed_count": replayed_count,
        "hard_feasible_rate": hard_count / max(replayed_count, 1),
        "all_vector_nonregressing_precision": all_vector / max(replayed_count, 1),
        "soft_regression_rate": soft_regressions / max(replayed_count, 1),
        "strict_meaningful_winner_count": len(strict),
        "largest_winner_case_share": largest_winner_share,
        "winner_signature_not_already_step7ml_i_k": False if strict else None,
        "family_pass": family_pass,
    }


def summarize_ablation(
    rows: list[dict[str, Any]],
    rows_out_path: Path,
    summary_out_path: Path,
    markdown_out_path: Path,
) -> dict[str, Any]:
    passed = [row for row in rows if row["family_pass"]]
    return {
        "schema": "step7p_phase4_causal_ablation_summary_v1",
        "decision": "write_selector_prd" if passed else "complete_step7p_negative_no_family_pass",
        "rows_path": str(rows_out_path),
        "summary_path": str(summary_out_path),
        "markdown_path": str(markdown_out_path),
        "family_count": len(rows),
        "passed_family_count": len(passed),
        "passed_families": [row["intent_family"] for row in passed],
        "gnn_rl_gate_open": False,
        "next_recommendation": "stop_step7p_or_redesign_real_causal_repacker"
        if not passed
        else "write_step7q_selector_prd",
    }


def failures_by_case(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_case: dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        if row["strict_meaningful_winner"]:
            continue
        by_case[str(row["case_id"])][str(row["seed_failure_bucket"])] += 1
    return {case_id: dict(counter) for case_id, counter in sorted(by_case.items())}


def request_markdown(summary: dict[str, Any]) -> str:
    return markdown("Step7P Phase3 Causal Request Summary", summary)


def replay_markdown(summary: dict[str, Any]) -> str:
    return markdown("Step7P Phase3 Causal Replay Summary", summary)


def ablation_markdown(summary: dict[str, Any]) -> str:
    return markdown("Step7P Phase4 Causal Family Ablation Summary", summary)


def markdown(title: str, summary: dict[str, Any]) -> str:
    lines = [f"# {title}", "", f"Decision: `{summary['decision']}`", ""]
    for key in sorted(summary):
        if key.endswith("_path") or key in {"schema", "decision"}:
            continue
        value = summary[key]
        if isinstance(value, (dict, list)):
            continue
        lines.append(f"- {key}: {value}")
    lines.append("")
    return "\n".join(lines)


def request_signature(row: dict[str, Any]) -> str:
    return ":".join(
        [
            str(row.get("case_id")),
            str(row.get("source_subproblem_id")),
            str(row.get("intent_family")),
        ]
    )


def objective_vector(value: Any) -> dict[str, float | None]:
    vector = value if isinstance(value, dict) else {}
    return {
        "hpwl_delta": float_or_none(vector.get("hpwl_delta")),
        "bbox_area_delta": float_or_none(vector.get("bbox_area_delta")),
        "soft_constraint_delta": float_or_none(vector.get("soft_constraint_delta")),
        "official_like_cost_delta": float_or_none(vector.get("official_like_cost_delta")),
    }


def forbidden_term_count(rows: list[dict[str, Any]]) -> int:
    scrubbed = [{key: value for key, value in row.items() if key != "forbidden"} for row in rows]
    text = "\n".join(str(row).lower() for row in scrubbed)
    return sum(text.count(term) for term in FORBIDDEN_REQUEST_TERMS)


def largest_case(case_counts: Counter[str]) -> tuple[str | None, int]:
    if not case_counts:
        return None, 0
    return max(case_counts.items(), key=lambda item: (item[1], item[0]))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


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


def float_or_none(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
