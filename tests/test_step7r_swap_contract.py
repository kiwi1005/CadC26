from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any

from puzzleplace.alternatives.chain_move import propose_swap_candidates


def _layout(
    fixture_id: str,
    blocks: list[dict[str, Any]],
    *,
    mib_groups: dict[str, list[int]] | None = None,
    boundary: dict[str, float] | None = None,
) -> dict[str, Any]:
    return {
        "case_id": fixture_id,
        "blocks": blocks,
        "boundary": boundary or {"x": 0.0, "y": 0.0, "w": 100.0, "h": 100.0},
        "mib_groups": mib_groups or {},
    }


def _candidate(
    layout: dict[str, Any],
    pair: tuple[int, int],
    *,
    fixed_block_ids: set[int] | None = None,
) -> dict[str, Any]:
    before = deepcopy(layout)
    candidates = propose_swap_candidates(
        layout,
        seed_pairs=[pair],
        fixed_block_ids=fixed_block_ids or set(),
    )
    assert layout == before
    assert len(candidates) == 1
    candidate = candidates[0]
    assert candidate["schema"] == "step7r_swap_candidate_v1"
    assert candidate["area_preserving"] is True
    return candidate


def test_two_free_blocks_non_overlapping_swap_is_legal() -> None:
    layout = _layout(
        "free_swap",
        [
            {"block_id": 0, "x": 0.0, "y": 0.0, "w": 10.0, "h": 10.0, "fixed": False},
            {"block_id": 1, "x": 40.0, "y": 0.0, "w": 10.0, "h": 10.0, "fixed": False},
        ],
    )

    candidate = _candidate(layout, (0, 1))

    assert candidate["legal"] is True
    assert candidate["rejection_reason"] is None
    assert candidate["mib_guard_pass"] is True
    assert candidate["boundary_guard_pass"] is True
    assert candidate["post_swap_centers"] == {"0": [45.0, 5.0], "1": [5.0, 5.0]}


def test_swap_involving_fixed_block_is_rejected() -> None:
    layout = _layout(
        "fixed_reject",
        [
            {"block_id": 0, "x": 0.0, "y": 0.0, "w": 10.0, "h": 10.0, "fixed": True},
            {"block_id": 1, "x": 40.0, "y": 0.0, "w": 10.0, "h": 10.0, "fixed": False},
        ],
    )

    candidate = _candidate(layout, (0, 1), fixed_block_ids={0})

    assert candidate["legal"] is False
    assert candidate["rejection_reason"] == "fixed_block"


def test_swap_creating_neighbor_overlap_is_rejected() -> None:
    layout = _layout(
        "neighbor_overlap",
        [
            {"block_id": 0, "x": 0.0, "y": 0.0, "w": 20.0, "h": 20.0, "fixed": False},
            {"block_id": 1, "x": 50.0, "y": 15.0, "w": 10.0, "h": 10.0, "fixed": False},
            {"block_id": 2, "x": 62.0, "y": 10.0, "w": 5.0, "h": 20.0, "fixed": False},
        ],
    )

    candidate = _candidate(layout, (0, 1))

    assert candidate["legal"] is False
    assert candidate["rejection_reason"] == "neighbor_overlap"


def test_swap_crossing_mib_equal_shape_group_is_rejected() -> None:
    layout = _layout(
        "mib_reject",
        [
            {"block_id": 0, "x": 0.0, "y": 0.0, "w": 10.0, "h": 10.0, "fixed": False},
            {"block_id": 1, "x": 40.0, "y": 0.0, "w": 10.0, "h": 10.0, "fixed": False},
        ],
        mib_groups={"left": [0], "right": [1]},
    )

    candidate = _candidate(layout, (0, 1))

    assert candidate["legal"] is False
    assert candidate["rejection_reason"] == "mib_violation"
    assert candidate["mib_guard_pass"] is False


def test_swap_pushing_block_past_boundary_is_rejected() -> None:
    layout = _layout(
        "boundary_reject",
        [
            {"block_id": 0, "x": 40.0, "y": 40.0, "w": 20.0, "h": 20.0, "fixed": False},
            {"block_id": 1, "x": 0.0, "y": 0.0, "w": 10.0, "h": 10.0, "fixed": False},
        ],
    )

    candidate = _candidate(layout, (0, 1))

    assert candidate["legal"] is False
    assert candidate["rejection_reason"] == "boundary_violation"
    assert candidate["boundary_guard_pass"] is False


def test_emitted_step7r_swap_deck_summary_contract() -> None:
    summary_path = Path("artifacts/research/step7r_swap_source_deck_summary.json")
    assert summary_path.exists()
    summary = json.loads(summary_path.read_text(encoding="utf-8"))

    assert summary["legal_count"] >= 30
    assert summary["represented_case_count"] >= 8
    assert summary["largest_case_share"] <= 0.25
    assert summary["forbidden_action_term_count"] == 0
