from __future__ import annotations

import pytest

from puzzleplace.ml.step7l_schema import (
    Step7LRecordFamily,
    assert_no_validation_label_leakage,
    has_forbidden_validation_targets,
    sidecar_candidate_quality_record,
    training_layout_prior_record,
    validation_inference_record,
)


def test_training_record_allows_prior_but_forbids_quality_gate_label() -> None:
    record = training_layout_prior_record(
        sample_id="train_0000000",
        sample_index=0,
        block_count=3,
        feature_summary={"b2b_edge_count": 2},
        label_summary={"fp_sol_shape": [3, 4]},
    )
    assert record.family == Step7LRecordFamily.TRAINING_LAYOUT_PRIOR
    assert record.has_target_label is True
    assert "supervised_target_prior" in record.allowed_uses
    assert "candidate_quality_gate_label" in record.forbidden_uses


def test_validation_record_has_no_target_label() -> None:
    record = validation_inference_record(
        case_id="24",
        block_count=42,
        feature_summary={"b2b_edge_count": 100},
        requested_case_id="24",
    )
    assert record.family == Step7LRecordFamily.VALIDATION_INFERENCE
    assert record.has_target_label is False
    assert "polygons" in record.label_summary["discarded_loader_label_fields"]
    assert "supervised_target_prior" in record.forbidden_uses
    assert_no_validation_label_leakage([record])


def test_validation_forbidden_target_key_detection_is_recursive() -> None:
    assert has_forbidden_validation_targets({"features": {"fp_sol": [1, 2, 3]}})
    assert has_forbidden_validation_targets({"rows": [{"target_positions": []}]})
    assert not has_forbidden_validation_targets({"features": {"area": [1.0]}})


def test_validation_label_leakage_raises() -> None:
    leaked = validation_inference_record(case_id="25", block_count=10, feature_summary={})
    object.__setattr__(leaked, "has_target_label", True)
    with pytest.raises(ValueError, match="validation inference records carry target labels"):
        assert_no_validation_label_leakage([leaked])


def test_sidecar_quality_record_forbids_layout_prior_target() -> None:
    record = sidecar_candidate_quality_record(
        candidate_id="case024:demo",
        case_id="24",
        block_count=2,
        source_artifact="artifacts/research/demo.json",
        label_summary={"official_like_improving": True},
    )
    assert record.family == Step7LRecordFamily.SIDECAR_CANDIDATE_QUALITY
    assert record.target_label_kind == "candidate_quality_vector"
    assert "layout_prior_target" in record.forbidden_uses
