from __future__ import annotations

import inspect

import pytest
import torch

from puzzleplace.actions.schema import ActionPrimitive, TypedAction
from puzzleplace.data.schema import FloorSetCase
from puzzleplace.research.transition_payload import (
    ALLOWED_TRANSITION_PAYLOAD_FIELDS,
    DENIED_TRANSITION_PAYLOAD_FIELDS,
    SharedEncoderTransitionComparator,
    build_transition_payload,
    validate_transition_payload,
)


def _make_case() -> FloorSetCase:
    constraints = torch.zeros((3, 5), dtype=torch.float32)
    constraints[0, 0] = 1.0  # fixed/preplaced anchor relation channel
    constraints[1, 3] = 7.0  # cluster relation channel
    constraints[2, 3] = 7.0
    constraints[1, 4] = 2.0  # boundary relation channel
    constraints[2, 4] = 2.0
    return FloorSetCase(
        case_id="step6f-hidden-case-id",
        block_count=3,
        area_targets=torch.tensor([6.0, 6.0, 4.0], dtype=torch.float32),
        b2b_edges=torch.tensor([[0.0, 1.0, 3.0], [1.0, 2.0, 1.0]], dtype=torch.float32),
        p2b_edges=torch.tensor([[0.0, 2.0, 2.0]], dtype=torch.float32),
        pins_pos=torch.tensor([[1.0, 2.0]], dtype=torch.float32),
        constraints=constraints,
        target_positions=torch.tensor(
            [[0.0, 0.0, 2.0, 3.0], [2.0, 0.0, 3.0, 2.0], [0.0, 3.0, 2.0, 2.0]],
            dtype=torch.float32,
        ),
        metrics=None,
    )


def _legal_payload() -> dict[str, object]:
    return build_transition_payload(
        _make_case(),
        TypedAction(
            ActionPrimitive.PLACE_ABSOLUTE,
            block_index=1,
            x=2.0,
            y=0.0,
            w=3.0,
            h=2.0,
        ),
        placements={0: (0.0, 0.0, 2.0, 3.0)},
        state_step=1,
        pairwise_majority_target=1,
    )


def test_transition_payload_schema_contains_only_legal_fields() -> None:
    payload = _legal_payload()

    assert set(payload) == ALLOWED_TRANSITION_PAYLOAD_FIELDS
    assert set(payload).isdisjoint(DENIED_TRANSITION_PAYLOAD_FIELDS)
    validate_transition_payload(payload)

    pre_block = payload["pre_block_features"]
    post_block = payload["post_block_features"]
    assert isinstance(pre_block, torch.Tensor)
    assert isinstance(post_block, torch.Tensor)
    assert pre_block.shape == post_block.shape == (3, 16)
    assert not torch.equal(pre_block, post_block)

    pre_edges = payload["pre_typed_edges"]
    post_edges = payload["post_typed_edges"]
    assert isinstance(pre_edges, torch.Tensor)
    assert isinstance(post_edges, torch.Tensor)
    assert pre_edges.shape[1] == post_edges.shape[1] == 4
    assert torch.equal(pre_edges[:, 0], post_edges[:, 0])

    action_token = payload["action_token"]
    assert isinstance(action_token, torch.Tensor)
    assert action_token.tolist() == [1.0, -1.0, 0.0, 0.0]


@pytest.mark.parametrize(
    "field",
    sorted(DENIED_TRANSITION_PAYLOAD_FIELDS),
)
def test_payload_validator_fails_closed_for_denied_fields(field: str) -> None:
    payload = _legal_payload()
    payload[field] = torch.zeros(1)

    with pytest.raises(ValueError, match="denied transition payload fields"):
        validate_transition_payload(payload)


def test_payload_validator_fails_for_missing_and_unknown_fields() -> None:
    payload = _legal_payload()
    payload.pop("post_typed_edges")
    with pytest.raises(ValueError, match="missing required transition payload fields: post_typed_edges"):
        validate_transition_payload(payload)

    payload = _legal_payload()
    payload["debug_feature"] = torch.zeros(1)
    with pytest.raises(ValueError, match="unsupported transition payload fields: debug_feature"):
        validate_transition_payload(payload)


def test_payload_validator_recurses_into_nested_shortcut_fields() -> None:
    payload = _legal_payload()
    payload["action_token"] = {"primitive_id": 0, "case_id": "forbidden"}

    with pytest.raises(ValueError, match="case_id"):
        validate_transition_payload(payload)


def test_transition_builder_rejects_invalid_candidate_without_pre_graph_fallback() -> None:
    invalid_move = TypedAction(ActionPrimitive.MOVE, block_index=2, dx=1.0, dy=0.0)

    with pytest.raises(ValueError, match="invalid transition candidate"):
        build_transition_payload(
            _make_case(),
            invalid_move,
            placements={0: (0.0, 0.0, 2.0, 3.0)},
            state_step=1,
            pairwise_majority_target=0,
        )


@pytest.mark.parametrize(
    "metadata",
    [
        {"candidate_features": [1.0]},
        {"nested": {"target_positions": [[0.0, 0.0, 1.0, 1.0]]}},
        {"manual_delta_rows": [[0.1, -0.1]]},
    ],
)
def test_transition_builder_rejects_denied_candidate_metadata(metadata: dict[str, object]) -> None:
    action = TypedAction(
        ActionPrimitive.PLACE_ABSOLUTE,
        block_index=1,
        x=2.0,
        y=0.0,
        w=3.0,
        h=2.0,
        metadata=metadata,
    )

    with pytest.raises(ValueError, match="denied transition action metadata fields"):
        build_transition_payload(
            _make_case(),
            action,
            placements={0: (0.0, 0.0, 2.0, 3.0)},
            pairwise_majority_target=1,
        )


def test_shared_encoder_comparator_has_no_manual_delta_input() -> None:
    payload = _legal_payload()
    block_features = payload["pre_block_features"]
    assert isinstance(block_features, torch.Tensor)
    model = SharedEncoderTransitionComparator(block_feature_dim=block_features.shape[1], hidden_dim=16)

    assert model.pre_encoder is model.post_encoder
    assert "manual_delta" not in inspect.signature(model.forward).parameters
    assert not any("delta" in name for name, _module in model.named_modules())

    score = model(payload)
    assert score.shape == torch.Size([])

    denied_payload = dict(payload)
    denied_payload["manual_delta_rows"] = torch.zeros((3, 2))
    with pytest.raises(ValueError, match="manual_delta_rows"):
        model(denied_payload)
