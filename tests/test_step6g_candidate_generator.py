from __future__ import annotations

from puzzleplace.actions import ExecutionState
from puzzleplace.research.puzzle_candidate_payload import (
    build_puzzle_candidate_descriptors,
    choose_expert_descriptor,
    empty_mask_reason_buckets,
    masked_anchor_xywh,
)
from puzzleplace.research.step6g_synthetic import make_step6g_synthetic_case
from puzzleplace.trajectory import generate_pseudo_traces


def test_shape_aware_generator_emits_required_families_without_soft_anchor_leak() -> None:
    case = make_step6g_synthetic_case(block_count=10)
    assert case.target_positions is not None
    first_box = tuple(float(v) for v in case.target_positions[0].tolist())
    assert len(first_box) == 4
    descriptors = build_puzzle_candidate_descriptors(
        case,
        ExecutionState(placements={0: first_box}),
        remaining_blocks=[2, 3, 4],
        max_shape_bins=5,
    )
    assert descriptors
    families = "\n".join(desc.candidate_family for desc in descriptors)
    assert "shape_bin:bin" in families
    assert "anchor:placed_block" in families or "anchor:boundary" in families
    assert "anchor:free_rect|contact:free_rect" in families
    assert "contact:" in families
    assert all(desc.legality_status == "legal" for desc in descriptors)

    anchors = masked_anchor_xywh(case)
    assert anchors[2].tolist() == [-1.0, -1.0, -1.0, -1.0]


def test_expert_descriptor_selection_buckets_training_label_without_requiring_exact_candidate() -> (
    None
):
    case = make_step6g_synthetic_case(block_count=10)
    trace = generate_pseudo_traces(case, max_traces=1)[0]
    state = ExecutionState()
    action = trace.actions[0]
    descriptors = build_puzzle_candidate_descriptors(
        case, state, remaining_blocks=list(range(case.block_count))
    )
    chosen = choose_expert_descriptor(descriptors, action)
    assert chosen is not None
    assert descriptors[chosen].block_index == action.block_index



def test_mib_exact_shape_and_mask_reason_buckets_are_reported() -> None:
    case = make_step6g_synthetic_case(block_count=12)
    assert case.target_positions is not None
    mib_row = case.target_positions[8].tolist()
    mib_mate = (
        float(mib_row[0]),
        float(mib_row[1]),
        float(mib_row[2]),
        float(mib_row[3]),
    )
    state = ExecutionState(placements={8: mib_mate})
    buckets = empty_mask_reason_buckets()

    descriptors = build_puzzle_candidate_descriptors(
        case,
        state,
        remaining_blocks=[9],
        max_shape_bins=6,
        max_descriptors_per_block=32,
        mask_reason_buckets=buckets,
    )

    assert any(desc.exact_shape_flag and desc.shape_bin_id == -2 for desc in descriptors)
    assert any(desc.anchor_kind == "free_rect" for desc in descriptors)
    assert set(buckets) >= {"overlap", "non_positive", "area_tolerance", "fixed_preplaced"}
    assert buckets["overlap"] > 0
