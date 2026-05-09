"""Synthetic Step7P causal closure repacker.

The implementation is deliberately fixture-driven. It proves operator contracts
(no overlap, area preservation, fixed/preplaced preservation, guard rejection)
before any validation replay work is allowed.
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from puzzleplace.repack.vector_gate import pareto_front, reject_reason, vector_nonregressing

OPERATOR_FAMILIES = (
    "blocker_chain_unblock",
    "order_preserving_row_repack",
    "closure_translate_with_repair",
    "mib_shape_preserve_repack",
    "boundary_contact_guarded",
    "pareto_vector_filter",
)


def run_synthetic_repacker(
    fixtures_dir: Path,
    report_out_path: Path,
    markdown_out_path: Path,
    operator_contract_out_path: Path,
) -> dict[str, Any]:
    fixture_paths = sorted(fixtures_dir.glob("*.json"))
    fixture_reports = [evaluate_fixture(load_json(path), path) for path in fixture_paths]
    summary = summarize_reports(
        fixture_reports,
        fixtures_dir=fixtures_dir,
        report_out_path=report_out_path,
        markdown_out_path=markdown_out_path,
        operator_contract_out_path=operator_contract_out_path,
    )
    write_json(report_out_path, summary)
    write_json(operator_contract_out_path, operator_contract(summary))
    markdown_out_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_out_path.write_text(synthetic_markdown(summary), encoding="utf-8")
    return summary


def evaluate_fixture(fixture: dict[str, Any], path: Path) -> dict[str, Any]:
    base_blocks = block_map(fixture.get("blocks"))
    candidate_reports = [
        evaluate_candidate(candidate, base_blocks, fixture) for candidate in candidate_rows(fixture)
    ]
    accepted = [row for row in candidate_reports if row["accepted"]]
    rejected = [row for row in candidate_reports if not row["accepted"]]
    front = pareto_front(candidate_reports)
    guard_expectations = dict_value(fixture.get("expectations"))
    return {
        "schema": "step7p_phase2_fixture_report_v1",
        "fixture_path": str(path),
        "fixture_id": str(fixture.get("fixture_id") or path.stem),
        "intent_family": fixture.get("intent_family"),
        "candidate_count": len(candidate_reports),
        "accepted_count": len(accepted),
        "rejected_count": len(rejected),
        "pareto_front_count": len(front),
        "emits_legal_non_noop": any(row["legal_non_noop"] for row in accepted),
        "no_overlap_pass": all(row["no_overlap"] for row in accepted),
        "area_preserved_pass": all(row["area_preserved"] for row in accepted),
        "fixed_preplaced_unchanged_pass": all(
            row["fixed_preplaced_unchanged"] for row in accepted
        ),
        "mib_equal_shape_guard_pass": guard_pass(
            candidate_reports, guard_expectations, "expect_mib_guard_pass"
        ),
        "boundary_guard_pass": guard_pass(
            candidate_reports, guard_expectations, "expect_boundary_guard_pass"
        ),
        "hpwl_only_soft_regression_rejected": any(
            row["rejection_reason"] == "reject_soft_regression" for row in rejected
        )
        if guard_expectations.get("expect_soft_regression_rejection")
        else True,
        "candidate_reports": candidate_reports,
        "pareto_candidate_ids": [str(row["candidate_id"]) for row in front],
    }


def evaluate_candidate(
    candidate: dict[str, Any], base_blocks: dict[int, dict[str, Any]], fixture: dict[str, Any]
) -> dict[str, Any]:
    after_blocks = apply_moves(base_blocks, candidate.get("moves"))
    vector = objective_vector(candidate)
    rejection = reject_reason(vector)
    no_overlap = not has_overlap(list(after_blocks.values()))
    area_preserved = total_area(base_blocks.values()) == total_area(after_blocks.values())
    fixed_preplaced_unchanged = fixed_preplaced_unchanged_check(base_blocks, after_blocks)
    guard_ok = guard_checks(candidate, after_blocks, fixture)
    non_noop = after_blocks != base_blocks
    accepted = (
        rejection is None
        and no_overlap
        and area_preserved
        and fixed_preplaced_unchanged
        and guard_ok
        and non_noop
    )
    return {
        "schema": "step7p_phase2_candidate_report_v1",
        "candidate_id": str(candidate.get("candidate_id")),
        "operator_family": candidate.get("operator_family"),
        "accepted": accepted,
        "legal_non_noop": accepted and non_noop,
        "no_overlap": no_overlap,
        "area_preserved": area_preserved,
        "fixed_preplaced_unchanged": fixed_preplaced_unchanged,
        "guard_ok": guard_ok,
        "vector_nonregressing": vector_nonregressing(vector),
        "rejection_reason": rejection,
        "objective_vector": vector,
    }


def apply_moves(
    blocks: dict[int, dict[str, Any]], moves: Any
) -> dict[int, dict[str, Any]]:
    moved = {block_id: dict(block) for block_id, block in blocks.items()}
    if not isinstance(moves, list):
        return moved
    for move in moves:
        if not isinstance(move, dict):
            continue
        block_id = int(move.get("block_id", -1))
        if block_id not in moved:
            continue
        if moved[block_id].get("fixed") or moved[block_id].get("preplaced"):
            continue
        moved[block_id]["x"] = float(move.get("x", moved[block_id]["x"]))
        moved[block_id]["y"] = float(move.get("y", moved[block_id]["y"]))
    return moved


def guard_checks(
    candidate: dict[str, Any], after_blocks: dict[int, dict[str, Any]], fixture: dict[str, Any]
) -> bool:
    if fixture.get("intent_family") == "mib_shape_preserve_repack":
        return mib_shape_preserved(after_blocks)
    if fixture.get("intent_family") == "boundary_contact_guarded":
        return boundary_blocks_in_bounds(after_blocks, dict_value(fixture.get("bounds")))
    return bool(candidate.get("guard_ok", True))


def mib_shape_preserved(blocks: dict[int, dict[str, Any]]) -> bool:
    by_mib: dict[int, set[tuple[float, float]]] = {}
    for block in blocks.values():
        mib = int(block.get("mib", 0))
        if mib == 0:
            continue
        by_mib.setdefault(mib, set()).add((float(block["w"]), float(block["h"])))
    return all(len(shapes) <= 1 for shapes in by_mib.values())


def boundary_blocks_in_bounds(blocks: dict[int, dict[str, Any]], bounds: dict[str, Any]) -> bool:
    if not bounds:
        return True
    width = float(bounds.get("w", 0.0))
    height = float(bounds.get("h", 0.0))
    for block in blocks.values():
        if int(block.get("boundary", 0)) == 0:
            continue
        if block["x"] < 0 or block["y"] < 0:
            return False
        if block["x"] + block["w"] > width or block["y"] + block["h"] > height:
            return False
    return True


def guard_pass(
    candidate_reports: list[dict[str, Any]], expectations: dict[str, Any], key: str
) -> bool:
    if not expectations.get(key):
        return True
    return any(row["accepted"] and row["guard_ok"] for row in candidate_reports)


def summarize_reports(
    fixture_reports: list[dict[str, Any]],
    *,
    fixtures_dir: Path,
    report_out_path: Path,
    markdown_out_path: Path,
    operator_contract_out_path: Path,
) -> dict[str, Any]:
    fixture_count = len(fixture_reports)
    fixture_with_three_pareto = sum(int(row["pareto_front_count"] >= 3) for row in fixture_reports)
    all_fixture_legal_non_noop = all(row["emits_legal_non_noop"] for row in fixture_reports)
    all_no_overlap = all(row["no_overlap_pass"] for row in fixture_reports)
    all_area_preserved = all(row["area_preserved_pass"] for row in fixture_reports)
    all_fixed_preplaced = all(row["fixed_preplaced_unchanged_pass"] for row in fixture_reports)
    mib_guard = any(row["mib_equal_shape_guard_pass"] for row in fixture_reports)
    boundary_guard = any(row["boundary_guard_pass"] for row in fixture_reports)
    soft_reject = any(row["hpwl_only_soft_regression_rejected"] for row in fixture_reports)
    phase3_gate_open = (
        fixture_count >= 5
        and all_fixture_legal_non_noop
        and all_no_overlap
        and all_area_preserved
        and all_fixed_preplaced
        and mib_guard
        and boundary_guard
        and soft_reject
        and fixture_with_three_pareto >= 3
    )
    return {
        "schema": "step7p_phase2_synthetic_repacker_report_v1",
        "decision": "promote_to_causal_request_deck"
        if phase3_gate_open
        else "stop_synthetic_repacker",
        "fixtures_dir": str(fixtures_dir),
        "report_path": str(report_out_path),
        "markdown_path": str(markdown_out_path),
        "operator_contract_path": str(operator_contract_out_path),
        "fixture_count": fixture_count,
        "operator_families": list(OPERATOR_FAMILIES),
        "operator_family_counts": dict(
            Counter(str(row["intent_family"]) for row in fixture_reports)
        ),
        "all_fixture_legal_non_noop": all_fixture_legal_non_noop,
        "all_no_overlap_pass": all_no_overlap,
        "all_area_preserved_pass": all_area_preserved,
        "all_fixed_preplaced_unchanged_pass": all_fixed_preplaced,
        "mib_equal_shape_guard_pass": mib_guard,
        "boundary_guard_pass": boundary_guard,
        "hpwl_only_soft_regression_rejected": soft_reject,
        "fixtures_with_three_or_more_pareto": fixture_with_three_pareto,
        "phase3_gate_open": phase3_gate_open,
        "gnn_rl_gate_open": False,
        "fixture_reports": fixture_reports,
        "next_recommendation": "generate_phase3_causal_request_deck"
        if phase3_gate_open
        else "rework_synthetic_repacker_before_replay",
    }


def operator_contract(summary: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema": "step7p_phase2_operator_contract_v1",
        "phase3_gate_open": summary["phase3_gate_open"],
        "operator_families": list(OPERATOR_FAMILIES),
        "requires_vector_nonregression": True,
        "rejects_hpwl_only_soft_regression": summary["hpwl_only_soft_regression_rejected"],
        "requires_no_overlap": summary["all_no_overlap_pass"],
        "requires_area_preservation": summary["all_area_preserved_pass"],
        "requires_fixed_preplaced_preservation": summary["all_fixed_preplaced_unchanged_pass"],
        "validation_replay_allowed": bool(summary["phase3_gate_open"]),
        "gnn_rl_gate_open": False,
    }


def synthetic_markdown(summary: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# Step7P Phase2 Synthetic Causal Repacker Report",
            "",
            f"Decision: `{summary['decision']}`",
            "",
            f"- fixture_count: {summary['fixture_count']}",
            f"- all_fixture_legal_non_noop: {summary['all_fixture_legal_non_noop']}",
            f"- all_no_overlap_pass: {summary['all_no_overlap_pass']}",
            f"- all_area_preserved_pass: {summary['all_area_preserved_pass']}",
            f"- all_fixed_preplaced_unchanged_pass: "
            f"{summary['all_fixed_preplaced_unchanged_pass']}",
            f"- mib_equal_shape_guard_pass: {summary['mib_equal_shape_guard_pass']}",
            f"- boundary_guard_pass: {summary['boundary_guard_pass']}",
            f"- hpwl_only_soft_regression_rejected: "
            f"{summary['hpwl_only_soft_regression_rejected']}",
            f"- fixtures_with_three_or_more_pareto: "
            f"{summary['fixtures_with_three_or_more_pareto']}",
            f"- phase3_gate_open: {summary['phase3_gate_open']}",
            f"- gnn_rl_gate_open: {summary['gnn_rl_gate_open']}",
            "",
        ]
    )


def has_overlap(blocks: list[dict[str, Any]]) -> bool:
    for index, a in enumerate(blocks):
        for b in blocks[index + 1 :]:
            if boxes_overlap(a, b):
                return True
    return False


def boxes_overlap(a: dict[str, Any], b: dict[str, Any]) -> bool:
    return not (
        a["x"] + a["w"] <= b["x"]
        or b["x"] + b["w"] <= a["x"]
        or a["y"] + a["h"] <= b["y"]
        or b["y"] + b["h"] <= a["y"]
    )


def total_area(blocks: Iterable[dict[str, Any]]) -> float:
    return sum(float(block["w"]) * float(block["h"]) for block in blocks)


def fixed_preplaced_unchanged_check(
    before: dict[int, dict[str, Any]], after: dict[int, dict[str, Any]]
) -> bool:
    for block_id, block in before.items():
        if not (block.get("fixed") or block.get("preplaced")):
            continue
        if block != after.get(block_id):
            return False
    return True


def block_map(raw_blocks: Any) -> dict[int, dict[str, Any]]:
    blocks: dict[int, dict[str, Any]] = {}
    if not isinstance(raw_blocks, list):
        return blocks
    for block in raw_blocks:
        if not isinstance(block, dict):
            continue
        block_id = int(block["block_id"])
        blocks[block_id] = {
            **block,
            "x": float(block["x"]),
            "y": float(block["y"]),
            "w": float(block["w"]),
            "h": float(block["h"]),
        }
    return blocks


def candidate_rows(fixture: dict[str, Any]) -> list[dict[str, Any]]:
    rows = fixture.get("candidates")
    return [row for row in rows if isinstance(row, dict)] if isinstance(rows, list) else []


def objective_vector(candidate: dict[str, Any]) -> dict[str, float]:
    raw = candidate.get("objective_vector")
    vector = raw if isinstance(raw, dict) else {}
    return {
        "hpwl_delta": float(vector.get("hpwl_delta", 0.0)),
        "bbox_area_delta": float(vector.get("bbox_area_delta", 0.0)),
        "soft_constraint_delta": float(vector.get("soft_constraint_delta", 0.0)),
    }


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def dict_value(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}
