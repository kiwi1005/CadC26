"""Step7L learning-guided target schema.

The Step7L learning lane deliberately separates three record families:

* FloorSet training ``fp_sol`` labels for supervised layout/target priors.
* Visible validation inference records, which must not carry target labels.
* Step7 sidecar candidate-quality labels for ranking / gate diagnostics only.

These helpers are intentionally small and JSON-serializable so later heatmap,
GNN, or offline-RL experiments can share the same leakage checks.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any


class Step7LRecordFamily(str, Enum):
    """Label family boundary for Step7L learning artifacts."""

    TRAINING_LAYOUT_PRIOR = "training_layout_prior"
    VALIDATION_INFERENCE = "validation_inference"
    SIDECAR_CANDIDATE_QUALITY = "sidecar_candidate_quality"


TRAINING_TARGET_KEYS = frozenset(
    {
        "fp_sol",
        "target_positions",
        "target_boxes",
        "target_heatmap",
        "block_target_regions",
        "closure_target_regions",
    }
)
VALIDATION_FORBIDDEN_TARGET_KEYS = TRAINING_TARGET_KEYS | frozenset(
    {
        "polygons",
        "metrics",
        "official_solution",
        "oracle_layout",
    }
)


@dataclass(frozen=True, slots=True)
class Step7LRecord:
    """Minimal JSON schema for one learning/audit record."""

    schema: str
    family: Step7LRecordFamily
    record_id: str
    source: str
    split: str
    case_id: str | None
    block_count: int
    has_target_label: bool
    target_label_kind: str | None = None
    allowed_uses: list[str] = field(default_factory=list)
    forbidden_uses: list[str] = field(default_factory=list)
    feature_summary: dict[str, Any] = field(default_factory=dict)
    label_summary: dict[str, Any] = field(default_factory=dict)
    missing_fields: list[str] = field(default_factory=list)

    def to_json(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["family"] = self.family.value
        return payload


def training_layout_prior_record(
    *,
    sample_id: str,
    sample_index: int,
    block_count: int,
    feature_summary: dict[str, Any],
    label_summary: dict[str, Any],
    missing_fields: list[str] | None = None,
) -> Step7LRecord:
    return Step7LRecord(
        schema="step7l_record_v1",
        family=Step7LRecordFamily.TRAINING_LAYOUT_PRIOR,
        record_id=sample_id,
        source="floorset_lite_training_fp_sol",
        split="floorset_lite_training",
        case_id=None,
        block_count=block_count,
        has_target_label=True,
        target_label_kind="fp_sol_[w,h,x,y]",
        allowed_uses=["supervised_target_prior", "heatmap_prior", "masked_bc_teacher"],
        forbidden_uses=["candidate_quality_gate_label", "visible_validation_win_claim"],
        feature_summary={"sample_index": sample_index, **feature_summary},
        label_summary=label_summary,
        missing_fields=missing_fields or [],
    )


def validation_inference_record(
    *,
    case_id: str,
    block_count: int,
    feature_summary: dict[str, Any],
    requested_case_id: str | None = None,
    missing_fields: list[str] | None = None,
) -> Step7LRecord:
    return Step7LRecord(
        schema="step7l_record_v1",
        family=Step7LRecordFamily.VALIDATION_INFERENCE,
        record_id=f"validation_{case_id}",
        source="floorset_lite_visible_validation_inputs_only",
        split="visible_validation_inference",
        case_id=case_id,
        block_count=block_count,
        has_target_label=False,
        target_label_kind=None,
        allowed_uses=["fresh_replay_input", "candidate_generation_input", "metric_replay_case"],
        forbidden_uses=["supervised_target_prior", "bc_teacher", "heatmap_target_label"],
        feature_summary={"requested_case_id": requested_case_id, **feature_summary},
        label_summary={"discarded_loader_label_fields": ["polygons", "metrics"]},
        missing_fields=missing_fields or [],
    )


def sidecar_candidate_quality_record(
    *,
    candidate_id: str,
    case_id: str | None,
    block_count: int,
    source_artifact: str,
    label_summary: dict[str, Any],
    feature_summary: dict[str, Any] | None = None,
    missing_fields: list[str] | None = None,
) -> Step7LRecord:
    return Step7LRecord(
        schema="step7l_record_v1",
        family=Step7LRecordFamily.SIDECAR_CANDIDATE_QUALITY,
        record_id=candidate_id,
        source="step7_sidecar_candidate_replay",
        split="sidecar_artifact_labels",
        case_id=case_id,
        block_count=block_count,
        has_target_label=True,
        target_label_kind="candidate_quality_vector",
        allowed_uses=["candidate_ranker", "quality_gate_diagnostic", "leave_case_out_ranker"],
        forbidden_uses=["layout_prior_target", "direct_generator_teacher_without_split"],
        feature_summary={"source_artifact": source_artifact, **(feature_summary or {})},
        label_summary=label_summary,
        missing_fields=missing_fields or [],
    )


def has_forbidden_validation_targets(payload: dict[str, Any]) -> bool:
    """Return True if a validation-inference payload contains target labels."""

    keys = set(payload)
    if keys & VALIDATION_FORBIDDEN_TARGET_KEYS:
        return True
    for value in payload.values():
        if isinstance(value, dict) and has_forbidden_validation_targets(value):
            return True
        if isinstance(value, list):
            for item in value:
                if isinstance(item, dict) and has_forbidden_validation_targets(item):
                    return True
    return False


def assert_no_validation_label_leakage(records: list[Step7LRecord]) -> None:
    """Raise if any visible-validation inference record carries target labels."""

    leaked = [
        record.record_id
        for record in records
        if record.family == Step7LRecordFamily.VALIDATION_INFERENCE and record.has_target_label
    ]
    if leaked:
        raise ValueError(f"validation inference records carry target labels: {leaked}")
