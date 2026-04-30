from __future__ import annotations

from puzzleplace.ml.training_backed_data_mart import (
    _closure_groups,
    _region_heatmap_example,
    stable_split,
)


def test_stable_split_is_deterministic() -> None:
    assert stable_split("train_0000001") == stable_split("train_0000001")
    assert stable_split("train_0000001") in {"train", "validation", "test"}


def test_closure_groups_keep_geometry_and_missing_flags() -> None:
    blocks = [
        {
            "block_id": 0,
            "x": 0.0,
            "y": 0.0,
            "w": 2.0,
            "h": 1.0,
            "area": 2.0,
            "fixed": False,
            "preplaced": False,
            "mib": 1,
            "cluster": 0,
            "boundary": 0,
        },
        {
            "block_id": 1,
            "x": 2.0,
            "y": 0.0,
            "w": 1.0,
            "h": 1.0,
            "area": 1.0,
            "fixed": True,
            "preplaced": False,
            "mib": 1,
            "cluster": 2,
            "boundary": 0,
        },
        {
            "block_id": 2,
            "x": 0.0,
            "y": 1.0,
            "w": 1.0,
            "h": 1.0,
            "area": 1.0,
            "fixed": False,
            "preplaced": True,
            "mib": 0,
            "cluster": 2,
            "boundary": 0,
        },
    ]
    groups = _closure_groups(blocks)
    assert {group["closure_type"] for group in groups} == {"mib", "cluster"}
    mib = next(group for group in groups if group["closure_type"] == "mib")
    assert mib["fixed_count"] == 1
    assert mib["block_geometry"][0]["w"] == 2.0
    assert "fixed_shape" in mib["missing_fields"]


def test_region_heatmap_assigns_blocks_to_regions() -> None:
    blocks = [
        {
            "block_id": 0,
            "x": 0.0,
            "y": 0.0,
            "w": 1.0,
            "h": 1.0,
            "area": 1.0,
            "fixed": False,
            "preplaced": False,
        },
        {
            "block_id": 1,
            "x": 3.0,
            "y": 3.0,
            "w": 1.0,
            "h": 1.0,
            "area": 1.0,
            "fixed": True,
            "preplaced": False,
        },
    ]
    example = _region_heatmap_example("train_0000000", 0, blocks)
    assert example["grid"] == {"rows": 4, "cols": 4}
    assert len(example["block_region_labels"]) == 2
    assert sum(sum(row) for row in example["region_distribution"]) == 1.0
