from __future__ import annotations

import json
from pathlib import Path

import torch

from puzzleplace.ml.learning_data_audit import (
    audit_training_batch,
    audit_validation_batch,
    load_candidate_quality_records,
)
from puzzleplace.ml.step7l_schema import Step7LRecordFamily


def _training_batch() -> tuple[torch.Tensor, ...]:
    area = torch.tensor([[1.0, 2.0, -1.0]])
    b2b = torch.tensor([[[0.0, 1.0, 1.0], [-1.0, -1.0, -1.0]]])
    p2b = torch.tensor([[[0.0, 0.0, 1.0], [-1.0, -1.0, -1.0]]])
    pins = torch.tensor([[[0.1, 0.2], [-1.0, -1.0]]])
    constraints = torch.tensor(
        [
            [
                [0.0, 0.0, 1.0, 0.0, 0.0],
                [1.0, 0.0, 1.0, 0.0, 0.0],
                [-1.0, -1.0, -1.0, -1.0, -1.0],
            ]
        ]
    )
    tree_sol = torch.zeros((1, 3, 4))
    fp_sol = torch.tensor(
        [[[1.0, 1.0, 0.0, 0.0], [2.0, 1.0, 1.0, 0.0], [-1.0, -1.0, -1.0, -1.0]]]
    )
    metrics = torch.tensor([[0.0, 0.0, 0.0]])
    return area, b2b, p2b, pins, constraints, tree_sol, fp_sol, metrics


def test_audit_training_batch_builds_prior_record() -> None:
    records, summary = audit_training_batch(_training_batch())
    assert summary["fp_sol_contract_valid_count"] == 1
    assert summary["invalid_or_malformed_sample_count"] == 0
    assert len(records) == 1
    record = records[0]
    assert record.family == Step7LRecordFamily.TRAINING_LAYOUT_PRIOR
    assert record.block_count == 2
    assert record.label_summary["fp_sol_label_order"] == "[w, h, x, y]"


def test_audit_training_batch_rejects_bad_fp_sol_contract() -> None:
    batch = list(_training_batch())
    batch[6] = torch.zeros((1, 3, 3))
    records, summary = audit_training_batch(tuple(batch))
    assert records == []
    assert summary["invalid_or_malformed_sample_count"] == 1


def test_audit_validation_batch_discards_labels() -> None:
    inputs = _training_batch()[:5]
    labels = (torch.ones((1, 3, 4, 2)), torch.ones((1, 3)))
    record = audit_validation_batch((inputs, labels), case_id="24", requested_case_id="24")
    assert record.family == Step7LRecordFamily.VALIDATION_INFERENCE
    assert record.has_target_label is False
    assert record.block_count == 2
    assert record.label_summary == {"discarded_loader_label_fields": ["polygons", "metrics"]}


def test_load_candidate_quality_records_keeps_quality_separate(tmp_path: Path) -> None:
    artifact = tmp_path / "artifacts/research/step7ml_g_candidate_quality_examples.json"
    artifact.parent.mkdir(parents=True)
    artifact.write_text(
        json.dumps(
            {
                "rows": [
                    {
                        "candidate_id": "case024:demo",
                        "case_id": 24,
                        "hard_feasible": True,
                        "official_like_improving": True,
                        "dominated_by_original": False,
                        "official_like_cost_delta": -0.1,
                        "changed_block_count": 2,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    records, report = load_candidate_quality_records(tmp_path)
    assert report["candidate_quality_source_row_count"] == 1
    assert records[0].family == Step7LRecordFamily.SIDECAR_CANDIDATE_QUALITY
    assert "layout_prior_target" in records[0].forbidden_uses
    assert records[0].label_summary["official_like_improving"] is True
