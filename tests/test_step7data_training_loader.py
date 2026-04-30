from __future__ import annotations

from pathlib import Path

import torch

from puzzleplace.ml.floorset_training_corpus import (
    _closure_groups,
    _valid_block_mask,
    decide,
    probe_to_json,
)


def test_valid_block_mask_requires_area_and_fp_sol() -> None:
    area = torch.tensor([1.0, -1.0, 2.0])
    fp = torch.tensor(
        [
            [1.0, 1.0, 0.0, 0.0],
            [-1.0, -1.0, -1.0, -1.0],
            [-1.0, -1.0, -1.0, -1.0],
        ]
    )
    assert _valid_block_mask(area, fp).tolist() == [True, False, False]


def test_closure_groups_extract_mib_and_cluster_labels() -> None:
    blocks = [
        {"block_id": 0, "x": 0.0, "y": 0.0, "w": 1.0, "h": 1.0, "mib": 1, "cluster": 0},
        {"block_id": 1, "x": 1.0, "y": 0.0, "w": 1.0, "h": 1.0, "mib": 1, "cluster": 2},
        {"block_id": 2, "x": 2.0, "y": 0.0, "w": 1.0, "h": 1.0, "mib": 0, "cluster": 2},
    ]
    groups = _closure_groups(blocks)
    assert {group["closure_type"] for group in groups} == {"mib", "cluster"}
    assert all(
        group["label_contract"] == "exact training label fp_sol provides [w,h,x,y]"
        for group in groups
    )


def test_decide_requires_download_when_loader_not_ready() -> None:
    inventory = probe_to_json(None, root=Path("/tmp/floorset"))
    assert decide(inventory, None) == "download_required"


def test_decide_training_loader_validated_with_macro_labels() -> None:
    inventory = {"training_loader_ready": True}
    report = {"loader_status": "ok", "macro_label_count": 2}
    assert decide(inventory, report) == "training_loader_validated"
