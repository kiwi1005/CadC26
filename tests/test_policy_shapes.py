from __future__ import annotations

import torch

from puzzleplace.data.schema import FloorSetCase
from puzzleplace.models import GraphStateEncoder, TypedActionPolicy
from puzzleplace.roles import label_case_roles


def _make_case() -> FloorSetCase:
    return FloorSetCase(
        case_id="policy-1",
        block_count=4,
        area_targets=torch.tensor([6.0, 6.0, 4.0, 8.0]),
        b2b_edges=torch.tensor([[0.0, 1.0, 3.0], [0.0, 3.0, 2.0], [2.0, 3.0, 1.0]]),
        p2b_edges=torch.tensor([[0.0, 0.0, 1.0], [1.0, 3.0, 1.0]]),
        pins_pos=torch.empty((0, 2)),
        constraints=torch.tensor([
            [1.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0, 1.0],
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


def test_graph_state_encoder_shapes() -> None:
    case = _make_case()
    roles = label_case_roles(case)
    encoder = GraphStateEncoder(hidden_dim=64)
    encoded = encoder(case, role_evidence=roles, placements={0: (0.0, 0.0, 2.0, 3.0)})
    assert encoded.block_embeddings.shape == (case.block_count, 64)
    assert encoded.graph_embedding.shape == (64,)
    assert encoded.block_mask.shape == (case.block_count,)


def test_typed_action_policy_output_shapes() -> None:
    case = _make_case()
    roles = label_case_roles(case)
    policy = TypedActionPolicy(hidden_dim=64)
    output = policy(case, role_evidence=roles, placements={0: (0.0, 0.0, 2.0, 3.0)})
    assert output.primitive_logits.shape[0] >= 1
    assert output.block_logits.shape == (case.block_count,)
    assert output.target_logits.shape == (case.block_count, case.block_count)
    assert output.boundary_logits.shape == (case.block_count, 5)
    assert output.geometry.shape == (case.block_count, 4)
