from __future__ import annotations

import torch

from puzzleplace.actions.schema import ActionPrimitive
from puzzleplace.data.schema import FloorSetCase
from puzzleplace.models.policy import DecoderOutput
from puzzleplace.rollout import beam_rollout, greedy_rollout


class FakePolicy:
    def __call__(self, case, *, role_evidence=None, placements=None):
        block_count = case.block_count
        primitive_logits = torch.full((len(ActionPrimitive),), -10.0)
        primitive_logits[list(ActionPrimitive).index(ActionPrimitive.PLACE_ABSOLUTE)] = 10.0
        block_logits = torch.arange(block_count, dtype=torch.float32) * -1.0
        target_logits = torch.zeros((block_count, block_count), dtype=torch.float32)
        boundary_logits = torch.zeros((block_count, 5), dtype=torch.float32)
        geometry = (
            case.target_positions.clone()
            if case.target_positions is not None
            else torch.zeros((block_count, 4))
        )
        return DecoderOutput(
            primitive_logits=primitive_logits,
            block_logits=block_logits,
            target_logits=target_logits,
            boundary_logits=boundary_logits,
            geometry=geometry,
        )


def _make_case() -> FloorSetCase:
    return FloorSetCase(
        case_id="rollout-1",
        block_count=2,
        area_targets=torch.tensor([6.0, 6.0]),
        b2b_edges=torch.tensor([[0.0, 1.0, 1.0]]),
        p2b_edges=torch.empty((0, 3)),
        pins_pos=torch.empty((0, 2)),
        constraints=torch.zeros((2, 5)),
        target_positions=torch.tensor([[0.0, 0.0, 2.0, 3.0], [2.0, 0.0, 3.0, 2.0]]),
        metrics=None,
    )


def test_greedy_rollout_smoke_places_at_least_one_block() -> None:
    result = greedy_rollout(_make_case(), FakePolicy())
    assert result.placed_count >= 1
    assert result.stopped_reason in {"completed", "step_limit", "no_legal_candidates"}


def test_beam_rollout_smoke_places_at_least_as_many_blocks_as_greedy() -> None:
    case = _make_case()
    greedy = greedy_rollout(case, FakePolicy())
    beam = beam_rollout(case, FakePolicy(), beam_width=2, per_state_candidates=2)
    assert beam.placed_count >= greedy.placed_count
