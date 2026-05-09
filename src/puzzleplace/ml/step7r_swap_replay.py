"""Step7R Phase 1.5 fresh-metric replay for k=2 swap candidates."""

from __future__ import annotations

import time
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from itertools import repeat
from pathlib import Path
from typing import Any

from puzzleplace.data.schema import FloorSetCase
from puzzleplace.geometry.legality import positions_from_case_targets, summarize_hard_legality
from puzzleplace.ml.step7q_fresh_metric_replay import (
    EPS,
    actual_delta,
    all_vector_nonregressing,
    failure_attribution,
    load_validation_cases,
    official_like_evaluator,
    quality_gate_status,
    same_box,
    strict_meaningful_winner,
)
from puzzleplace.ml.step7q_operator_learning import read_jsonl, write_json, write_jsonl

Box = tuple[float, float, float, float]
Center = tuple[float, float]
STEP7Q_F_ALL_VECTOR_BASELINE = 27


def replay_swap_deck(
    base_dir: Path,
    examples_path: Path,
    swap_deck_path: Path,
    replay_rows_out: Path,
    summary_out: Path,
    failures_out: Path,
    *,
    floorset_root: Path | None = None,
    max_candidates: int | None = None,
    auto_download: bool = False,
    n_workers: int | None = None,
) -> dict[str, Any]:
    """Replay a Step7R k=2 swap deck with fresh official-like metrics."""

    started = time.perf_counter()
    rows = read_jsonl(swap_deck_path)
    if max_candidates is not None:
        rows = rows[:max_candidates]
    examples = read_jsonl(examples_path)
    requested_workers = n_workers or min(48, len(rows))
    n_workers_used = max(1, requested_workers) if rows else 0
    ctx: dict[str, Any] = {
        "base_dir": str(base_dir),
        "floorset_root": str(floorset_root) if floorset_root is not None else None,
        "auto_download": auto_download,
    }

    if rows:
        with ProcessPoolExecutor(max_workers=n_workers_used) as executor:
            replay_rows = list(executor.map(_replay_one_swap_row, rows, repeat(ctx)))
    else:
        replay_rows = []

    write_jsonl(replay_rows_out, replay_rows)
    summary = summarize_swap_replay(
        replay_rows,
        request_count=len(rows),
        example_count=len(examples),
        swap_deck_path=swap_deck_path,
        replay_rows_path=replay_rows_out,
        n_workers_used=n_workers_used,
        runtime_proxy_ms=(time.perf_counter() - started) * 1000.0,
    )
    write_json(summary_out, summary)
    write_json(failures_out, failures_by_case(replay_rows))
    return summary


def _replay_one_swap_row(row: dict[str, Any], ctx: dict[str, Any]) -> dict[str, Any]:
    """Replay one independent swap row.  This function is process-picklable."""

    case_id = _case_id(row)
    case = _load_case_for_row(case_id, ctx)
    baseline = positions_from_case_targets(case)
    before_eval = official_like_evaluator(case, baseline)
    pair = _swap_pair(row)
    base = _base_row(row, pair)
    invalid_reason = _invalid_pair_reason(pair, case)
    if invalid_reason is not None:
        return _rejected_row(base, invalid_reason)

    after = _swap_positions(baseline, pair)
    after_eval = official_like_evaluator(case, after)
    legality = summarize_hard_legality(case, after)
    delta = actual_delta(before_eval, after_eval)
    nonnoop = any(
        not same_box(before, after_box) for before, after_box in zip(baseline, after, strict=False)
    )
    hard_feasible = bool(legality.is_feasible) and bool(after_eval["quality"].get("feasible"))
    hard_feasible_nonnoop = hard_feasible and nonnoop
    status = quality_gate_status(delta, hard_feasible_nonnoop)
    all_vector = all_vector_nonregressing(delta, hard_feasible_nonnoop)
    strict = strict_meaningful_winner(delta, hard_feasible_nonnoop)
    reason = failure_attribution(status, delta, legality.overlap_violations)
    rejection_reason = None if reason == "none" else reason
    return {
        **base,
        "post_swap_centers": _post_swap_centers(after, pair),
        "hard_feasible": hard_feasible,
        "hard_feasible_nonnoop": hard_feasible_nonnoop,
        "all_vector_nonregressing": all_vector,
        "actual_all_vector_nonregressing": all_vector,
        "strict_meaningful_winner": strict,
        "hpwl_delta": delta["hpwl_delta"],
        "bbox_delta": delta["bbox_area_delta"],
        "bbox_area_delta": delta["bbox_area_delta"],
        "soft_delta": delta["soft_constraint_delta"],
        "soft_constraint_delta": delta["soft_constraint_delta"],
        "official_like_cost_delta": delta["official_like_cost_delta"],
        "overlap_after_swap": legality.overlap_violations,
        "rejection_reason": rejection_reason,
        "quality_gate_status": status,
        "actual_objective_vector": delta,
        "legality": {
            "is_feasible": legality.is_feasible,
            "overlap_violations": legality.overlap_violations,
            "area_violations": legality.area_violations,
            "dimension_violations": legality.dimension_violations,
        },
    }


def summarize_swap_replay(
    rows: list[dict[str, Any]],
    *,
    request_count: int,
    example_count: int,
    swap_deck_path: Path,
    replay_rows_path: Path,
    n_workers_used: int,
    runtime_proxy_ms: float,
) -> dict[str, Any]:
    hard = [row for row in rows if row.get("hard_feasible_nonnoop")]
    strict = [row for row in hard if row.get("strict_meaningful_winner")]
    represented_case_count = len({str(row.get("case_id")) for row in rows})
    largest_case_share = _largest_case_share(rows, request_count)
    overlap_count = sum(int(_int_value(row.get("overlap_after_swap")) > 0) for row in rows)
    soft_count = sum(int(_float_value(row.get("soft_delta")) > EPS) for row in hard)
    bbox_count = sum(int(_float_value(row.get("bbox_delta")) > EPS) for row in hard)
    hpwl_count = sum(int(_float_value(row.get("hpwl_delta")) > EPS) for row in hard)
    all_vector_count = sum(int(bool(row.get("all_vector_nonregressing"))) for row in hard)
    phase2_gate_open = bool(
        len(strict) >= 1 or all_vector_count > STEP7Q_F_ALL_VECTOR_BASELINE
    )
    return {
        "schema": "step7r_swap_replay_summary_v1",
        "decision": "swap_replay_strict_gate_open"
        if phase2_gate_open
        else "swap_replay_strict_gate_closed",
        "request_count": request_count,
        "represented_case_count": represented_case_count,
        "largest_case_share": largest_case_share,
        "fresh_hard_feasible_nonnoop_count": len(hard),
        "overlap_after_splice_count": overlap_count,
        "soft_regression_rate": soft_count / max(request_count, 1),
        "bbox_regression_rate": bbox_count / max(request_count, 1),
        "hpwl_regression_rate": hpwl_count / max(request_count, 1),
        "actual_all_vector_nonregressing_count": all_vector_count,
        "strict_meaningful_winner_count": len(strict),
        "phase2_gate_open": phase2_gate_open,
        "n_workers_used": n_workers_used,
        "allowed_next_phase": "step7r_phase2_ripple_chain"
        if phase2_gate_open
        else "close_step7r",
        "replay_row_count": len(rows),
        "example_count": example_count,
        "swap_deck_path": str(swap_deck_path),
        "replay_rows_path": str(replay_rows_path),
        "soft_regression_count": soft_count,
        "bbox_regression_count": bbox_count,
        "hpwl_regression_count": hpwl_count,
        "status_counts": dict(Counter(str(row.get("quality_gate_status")) for row in rows)),
        "failure_counts": dict(
            Counter(str(row.get("rejection_reason") or "none") for row in rows)
        ),
        "runtime_proxy_ms": runtime_proxy_ms,
        "runtime_proxy_per_candidate_ms": runtime_proxy_ms / max(request_count, 1),
        "step7q_f_baseline_all_vector_nonregressing_count": STEP7Q_F_ALL_VECTOR_BASELINE,
    }


def failures_by_case(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_case: dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        by_case[str(row.get("case_id"))][str(row.get("rejection_reason") or "none")] += 1
    return {
        "schema": "step7r_swap_replay_failures_by_case_v1",
        "failures_by_case": {
            case_id: dict(counter) for case_id, counter in sorted(by_case.items())
        },
    }


def _load_case_for_row(case_id: int, ctx: dict[str, Any]) -> FloorSetCase:
    floorset_root = ctx.get("floorset_root")
    cases = load_validation_cases(
        Path(str(ctx["base_dir"])),
        [case_id],
        floorset_root=Path(str(floorset_root)) if floorset_root else None,
        auto_download=bool(ctx.get("auto_download")),
    )
    case = cases.get(case_id)
    if case is None:
        raise FileNotFoundError(f"Missing validation case {case_id} for Step7R swap replay")
    return case


def _case_id(row: dict[str, Any]) -> int:
    try:
        return int(str(row.get("case_id")))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid Step7R swap case_id: {row.get('case_id')!r}") from exc


def _swap_pair(row: dict[str, Any]) -> list[int]:
    raw = row.get("swap_pair")
    if not isinstance(raw, list | tuple) or len(raw) != 2:
        return []
    try:
        return [int(raw[0]), int(raw[1])]
    except (TypeError, ValueError):
        return []


def _invalid_pair_reason(pair: list[int], case: FloorSetCase) -> str | None:
    if len(pair) != 2:
        return "invalid_swap_pair"
    left, right = pair
    if left == right:
        return "invalid_swap_pair"
    if left < 0 or right < 0 or left >= case.block_count or right >= case.block_count:
        return "missing_swap_block"
    return None


def _base_row(row: dict[str, Any], pair: list[int]) -> dict[str, Any]:
    return {
        "schema": "step7r_swap_replay_row_v1",
        "case_id": str(row.get("case_id")),
        "deck_rank": row.get("deck_rank"),
        "parent_example_id": row.get("parent_example_id"),
        "source_candidate_id": row.get("source_candidate_id"),
        "swap_pair": pair,
        "source_legal": row.get("legal"),
        "source_rejection_reason": row.get("rejection_reason"),
    }


def _rejected_row(base: dict[str, Any], reason: str) -> dict[str, Any]:
    return {
        **base,
        "post_swap_centers": {},
        "hard_feasible": False,
        "hard_feasible_nonnoop": False,
        "all_vector_nonregressing": False,
        "actual_all_vector_nonregressing": False,
        "strict_meaningful_winner": False,
        "hpwl_delta": None,
        "bbox_delta": None,
        "bbox_area_delta": None,
        "soft_delta": None,
        "soft_constraint_delta": None,
        "official_like_cost_delta": None,
        "overlap_after_swap": 0,
        "rejection_reason": reason,
        "quality_gate_status": reason,
        "actual_objective_vector": None,
        "legality": None,
    }


def _swap_positions(positions: list[Box], pair: list[int]) -> list[Box]:
    left, right = pair
    left_center = _center(positions[left])
    right_center = _center(positions[right])
    after = list(positions)
    after[left] = _box_at_center(positions[left], right_center)
    after[right] = _box_at_center(positions[right], left_center)
    return after


def _post_swap_centers(positions: list[Box], pair: list[int]) -> dict[str, list[float]]:
    return {str(block_id): list(_center(positions[block_id])) for block_id in pair}


def _center(box: Box) -> Center:
    return box[0] + box[2] / 2.0, box[1] + box[3] / 2.0


def _box_at_center(box: Box, center: Center) -> Box:
    return (center[0] - box[2] / 2.0, center[1] - box[3] / 2.0, box[2], box[3])


def _largest_case_share(rows: list[dict[str, Any]], request_count: int) -> float:
    counts = Counter(str(row.get("case_id")) for row in rows)
    return max(counts.values(), default=0) / max(request_count, 1)


def _float_value(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _int_value(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


__all__ = [
    "_replay_one_swap_row",
    "failures_by_case",
    "replay_swap_deck",
    "summarize_swap_replay",
]
