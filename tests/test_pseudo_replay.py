from __future__ import annotations

import torch

from puzzleplace.data.schema import FloorSetCase
from puzzleplace.trajectory import compare_positions, generate_negative_actions, generate_pseudo_traces, replay_trace


def _make_case() -> FloorSetCase:
    return FloorSetCase(
        case_id="pseudo-1",
        block_count=4,
        area_targets=torch.tensor([6.0, 6.0, 4.0, 8.0]),
        b2b_edges=torch.tensor([[0.0, 1.0, 3.0], [0.0, 3.0, 2.0], [2.0, 3.0, 1.0]]),
        p2b_edges=torch.tensor([[0.0, 0.0, 1.0], [1.0, 3.0, 1.0]]),
        pins_pos=torch.empty((0, 2)),
        constraints=torch.tensor([
            [1.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0],
        ]),
        target_positions=torch.tensor([
            [0.0, 0.0, 2.0, 3.0],
            [2.0, 0.0, 3.0, 2.0],
            [0.0, 3.0, 2.0, 2.0],
            [2.0, 2.0, 2.0, 4.0],
        ]),
        metrics=torch.tensor([20.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0]),
    )


def test_generate_multiple_distinct_pseudo_traces() -> None:
    traces = generate_pseudo_traces(_make_case(), max_traces=4)
    assert len(traces) >= 2
    assert len({tuple(trace.ordered_blocks) for trace in traces}) >= 2


def test_pseudo_trace_replay_reconstructs_final_layout() -> None:
    case = _make_case()
    trace = generate_pseudo_traces(case, max_traces=1)[0]
    replayed = replay_trace(case, trace)
    comparison = compare_positions(case, replayed)
    assert comparison["all_blocks_present"] is True
    assert comparison["reconstruction_rate"] == 1.0


def test_negative_actions_are_generated_for_absolute_actions() -> None:
    trace = generate_pseudo_traces(_make_case(), max_traces=1)[0]
    negatives = generate_negative_actions(trace.actions)
    assert negatives
    assert all(action.metadata.get("negative") for action in negatives)
