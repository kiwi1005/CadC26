from __future__ import annotations

import pytest
import torch

from puzzleplace.actions import ActionPrimitive, ExecutionState, TypedAction, canonical_action_key
from puzzleplace.research.puzzle_candidate_payload import (
    ALLOWED_PUZZLE_INFERENCE_FIELDS,
    DENIED_PUZZLE_INFERENCE_FIELDS,
    PuzzleCandidateDescriptor,
    build_puzzle_candidate_descriptors,
    descriptor_signature,
    masked_anchor_xywh,
    validate_puzzle_candidate_payload,
)
from puzzleplace.research.step6g_synthetic import make_step6g_synthetic_case


def test_masked_anchor_hides_soft_target_positions() -> None:
    case = make_step6g_synthetic_case(block_count=6)
    anchors = masked_anchor_xywh(case)
    assert case.target_positions is not None

    assert torch.equal(anchors[0], case.target_positions[0])
    assert torch.equal(anchors[1], case.target_positions[1])
    for idx in range(2, case.block_count):
        assert anchors[idx].tolist() == [-1.0, -1.0, -1.0, -1.0]


def test_descriptor_payload_contains_only_allowed_inference_fields() -> None:
    case = make_step6g_synthetic_case(block_count=8)
    descriptors = build_puzzle_candidate_descriptors(
        case,
        ExecutionState(),
        remaining_blocks=[0, 1, 2],
        max_shape_bins=3,
    )
    assert descriptors
    payload = descriptors[0].inference_payload()

    assert set(payload) == ALLOWED_PUZZLE_INFERENCE_FIELDS
    validate_puzzle_candidate_payload(payload)
    assert payload["legality_status"] == "legal"


def test_descriptor_separates_action_token_from_semantic_identity() -> None:
    action = TypedAction(
        ActionPrimitive.PLACE_ABSOLUTE,
        block_index=2,
        x=0.0,
        y=0.0,
        w=2.0,
        h=2.0,
        metadata={"debug": "ignored-by-canonical-key"},
    )
    first = PuzzleCandidateDescriptor(
        block_index=2,
        shape_bin_id=7,
        exact_shape_flag=False,
        site_id=1,
        contact_mode="right",
        anchor_kind="placed_block",
        candidate_family="shape_bin:bin|anchor:placed_block|contact:right",
        legality_status="legal",
        action_token=action,
        normalized_features=torch.zeros(16),
    )
    second = PuzzleCandidateDescriptor(
        block_index=2,
        shape_bin_id=7,
        exact_shape_flag=False,
        site_id=2,
        contact_mode="top",
        anchor_kind="placed_block",
        candidate_family="shape_bin:bin|anchor:placed_block|contact:top",
        legality_status="legal",
        action_token=TypedAction(
            ActionPrimitive.PLACE_ABSOLUTE,
            block_index=2,
            x=0.0,
            y=0.0,
            w=2.0,
            h=2.0,
            metadata={"other": "also-ignored"},
        ),
        normalized_features=torch.zeros(16),
    )

    assert canonical_action_key(first.action_token) == canonical_action_key(second.action_token)
    assert descriptor_signature(first) != descriptor_signature(second)


@pytest.mark.parametrize("field", sorted(DENIED_PUZZLE_INFERENCE_FIELDS))
def test_payload_validator_rejects_every_denied_field(field: str) -> None:
    case = make_step6g_synthetic_case(block_count=6)
    payload = build_puzzle_candidate_descriptors(case, ExecutionState(), remaining_blocks=[2])[
        0
    ].inference_payload()
    payload[field] = torch.zeros(1)
    with pytest.raises(ValueError, match="denied puzzle candidate payload fields"):
        validate_puzzle_candidate_payload(payload)


def test_payload_validator_rejects_nested_and_action_metadata_denied_fields() -> None:
    case = make_step6g_synthetic_case(block_count=6)
    payload = build_puzzle_candidate_descriptors(case, ExecutionState(), remaining_blocks=[2])[
        0
    ].inference_payload()
    payload["normalized_features"] = {"nested": {"target_positions": [[0.0, 0.0, 1.0, 1.0]]}}
    with pytest.raises(ValueError, match="target_positions"):
        validate_puzzle_candidate_payload(payload)

    payload = build_puzzle_candidate_descriptors(case, ExecutionState(), remaining_blocks=[2])[
        0
    ].inference_payload()
    payload["action_token"].metadata["manual_delta_rows"] = [[1.0]]
    with pytest.raises(ValueError, match="manual_delta_rows"):
        validate_puzzle_candidate_payload(payload)


def test_payload_validator_rejects_missing_unknown_and_wrong_feature_shape() -> None:
    case = make_step6g_synthetic_case(block_count=6)
    payload = build_puzzle_candidate_descriptors(case, ExecutionState(), remaining_blocks=[2])[
        0
    ].inference_payload()
    payload.pop("site_id")
    with pytest.raises(ValueError, match="missing required puzzle candidate fields: site_id"):
        validate_puzzle_candidate_payload(payload)

    payload = build_puzzle_candidate_descriptors(case, ExecutionState(), remaining_blocks=[2])[
        0
    ].inference_payload()
    payload["debug_feature"] = 1
    with pytest.raises(ValueError, match="unsupported puzzle candidate fields: debug_feature"):
        validate_puzzle_candidate_payload(payload)

    payload = build_puzzle_candidate_descriptors(case, ExecutionState(), remaining_blocks=[2])[
        0
    ].inference_payload()
    payload["normalized_features"] = torch.zeros((1, 16))
    with pytest.raises(ValueError, match="rank-1"):
        validate_puzzle_candidate_payload(payload)
