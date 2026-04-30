from __future__ import annotations

from puzzleplace.ml.supervised_macro_layout import (
    _overlap_count,
    bucket_block_count,
    feature_vector,
)


def test_bucket_block_count() -> None:
    assert bucket_block_count(2) == "2-3"
    assert bucket_block_count(5) == "4-5"
    assert bucket_block_count(8) == "6-8"
    assert bucket_block_count(10) == "9+"


def test_feature_vector_has_expected_length() -> None:
    block = {
        "target_rel_w": 0.2,
        "target_rel_h": 0.3,
        "fixed": True,
        "preplaced": False,
        "boundary": 1,
    }
    features = feature_vector(
        block,
        closure_type="mib",
        block_count=4,
        closure_aspect=2.0,
        area_rank=1,
    )
    assert len(features) == 11
    assert features[3] == 1.0
    assert features[8] == 1.0


def test_overlap_count_detects_pair_overlap() -> None:
    blocks = [
        {"x": 0.0, "y": 0.0, "w": 2.0, "h": 2.0},
        {"x": 1.0, "y": 1.0, "w": 2.0, "h": 2.0},
        {"x": 4.0, "y": 4.0, "w": 1.0, "h": 1.0},
    ]
    assert _overlap_count(blocks) == 1
