from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest
import torch

from hcfp.floorset_lite import sample_with_source_from_lite_tensors
from hcfp.repair.dataset import (
    audit_clean_sample,
    source_split,
    summarize_clean_pool,
    validate_training_root,
)


def _clean_source(*, preplace_last_group_member: bool = False):
    area_constraints = torch.tensor(
        (
            (1.0, 0, 1, 1, 1, 1),
            (1.0, 1, 0, 1, 1, 0),
            (1.0, 1, int(preplace_last_group_member), 0, 1, 0),
            (1.0, 0, 0, 0, 0, 2),
        )
    )
    b2b = torch.tensor(((0.0, 1.0, 1.0), (1.0, 2.0, 1.0)))
    padded = torch.tensor(((-1.0, -1.0, -1.0),))
    pins = torch.tensor(((-1.0, -1.0),))
    tree = torch.tensor(((0, 1, 0), (1, 2, 0), (2, 3, 0)))
    fp_sol = torch.tensor(
        ((1.0, 1.0, 0.0, 0.0), (1.0, 1.0, 1.0, 0.0), (1.0, 1.0, 2.0, 0.0), (1.0, 1.0, 3.0, 0.0))
    )
    metrics = torch.tensor((4.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.0, 0.0))
    return sample_with_source_from_lite_tensors(
        "worker_0/layouts_0.th:0",
        area_constraints,
        b2b,
        padded,
        pins,
        fp_sol,
        metrics,
        tree,
    )


def test_source_split_is_stable_and_disjoint() -> None:
    ids = [f"worker_{index}/layouts_0.th:0" for index in range(100)]
    first = {sample_id: source_split(sample_id) for sample_id in ids}
    second = {sample_id: source_split(sample_id) for sample_id in reversed(ids)}
    train = {sample_id for sample_id, (split, _) in first.items() if split == "train"}
    heldout = {sample_id for sample_id, (split, _) in first.items() if split == "heldout"}

    assert first == second
    assert train and heldout
    assert train.isdisjoint(heldout)
    assert source_split(ids[0]) == source_split(f"{ids[0]}:c0:3")


@pytest.mark.parametrize("name", ("LiteTensorDataTest", "training_test", "visible", "validation"))
def test_training_root_rejects_leakage_tokens(tmp_path: Path, name: str) -> None:
    with pytest.raises(ValueError, match="forbidden"):
        validate_training_root(tmp_path / name)


def test_clean_audit_uses_exact_raw_fp_sol_and_fixed_blocks_remain_position_movable() -> None:
    sample, source = _clean_source()

    record = audit_clean_sample(sample, source)

    assert torch.equal(
        source["fp_sol_xywh"],
        torch.tensor(((0.0, 0.0, 1.0, 1.0), (1.0, 0.0, 1.0, 1.0), (2.0, 0.0, 1.0, 1.0), (3.0, 0.0, 1.0, 1.0))),
    )
    assert record["hard"]["feasible"]
    assert record["eligibility"] == {
        "contact_clean": True,
        "contact_c0_structural": True,
        "contact_c1_structural": True,
        "contact_c2_structural": True,
        "boundary": True,
        "mib": True,
        "topology": True,
    }
    assert record["counts"] == {"fixed": 2, "preplaced": 1, "constrained_blocks": 4}


def test_preplaced_leaf_is_not_a_contact_corruption_target() -> None:
    sample, source = _clean_source(preplace_last_group_member=True)

    record = audit_clean_sample(sample, source)

    assert record["eligibility"]["contact_clean"]
    assert not record["eligibility"]["contact_c0_structural"]
    assert record["eligibility"]["contact_c1_structural"]
    assert not record["eligibility"]["contact_c2_structural"]


def test_invalid_tree_is_reported_and_summary_is_order_stable() -> None:
    sample, source = _clean_source()
    invalid = replace(sample, tree_edges=torch.tensor(((0, 1, 0), (0, 2, 0), (2, 3, 0))))
    invalid_record = audit_clean_sample(invalid, source)
    records = [
        audit_clean_sample(replace(sample, sample_id=f"worker_{index}/layouts_0.th:0"), source)
        for index in range(20)
    ]

    assert not invalid_record["tree_valid"]
    assert summarize_clean_pool(records) == summarize_clean_pool(list(reversed(records)))
    assert summarize_clean_pool(records)["split_overlap_count"] == 0
    with pytest.raises(ValueError, match="duplicate source ids"):
        summarize_clean_pool(records + records[:1])
