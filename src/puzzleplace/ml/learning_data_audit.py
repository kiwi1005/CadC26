"""Step7L learning-data audit utilities.

This audit is the first gate before target heatmaps, GNNs, or offline RL. It
proves that FloorSet training labels, visible-validation inference inputs, and
Step7 sidecar quality labels are separate artifacts with explicit allowed uses.
"""

from __future__ import annotations

import time
from collections import Counter
from pathlib import Path
from typing import Any

import torch

from puzzleplace.ml.floorset_training_corpus import (
    _auto_yes_download,
    _import_official_evaluator,
    _valid_block_mask,
    load_json,
    probe_to_json,
    probe_training_corpus,
    resolve_floorset_root,
    write_json,
)
from puzzleplace.ml.step7l_schema import (
    Step7LRecord,
    Step7LRecordFamily,
    assert_no_validation_label_leakage,
    sidecar_candidate_quality_record,
    training_layout_prior_record,
    validation_inference_record,
)


def _shape(value: Any) -> list[int] | None:
    shape = getattr(value, "shape", None)
    if shape is None:
        return None
    return [int(dim) for dim in shape]


def _rows(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, dict):
        rows = payload.get("rows", [])
        return [row for row in rows if isinstance(row, dict)] if isinstance(rows, list) else []
    if isinstance(payload, list):
        return [row for row in payload if isinstance(row, dict)]
    return []


def _safe_float(value: Any, default: float = 0.0) -> float:
    if value is None:
        return default
    if isinstance(value, int | float):
        return float(value)
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _valid_count_from_area(area: torch.Tensor) -> int:
    valid = area >= 0
    if valid.ndim > 1:
        valid = valid.reshape(-1)
    return int(valid.sum().item())


def _edge_count(edges: torch.Tensor) -> int:
    if edges.ndim == 0:
        return 0
    if edges.ndim == 1:
        return int((edges >= 0).sum().item())
    return int((edges[..., 0] >= 0).sum().item())


def audit_training_batch(
    batch: tuple[torch.Tensor, ...],
    *,
    sample_offset: int = 0,
    max_records: int | None = None,
) -> tuple[list[Step7LRecord], dict[str, Any]]:
    area, b2b, p2b, pins, constraints, _tree_sol, fp_sol, metrics = batch
    records: list[Step7LRecord] = []
    malformed = 0
    fp_sol_valid = 0
    batch_size = int(area.shape[0]) if area.ndim else 0
    limit = min(batch_size, max_records if max_records is not None else batch_size)
    for row_idx in range(limit):
        sample_index = sample_offset + row_idx
        if fp_sol.shape[-1] != 4:
            malformed += 1
            continue
        mask = _valid_block_mask(area[row_idx], fp_sol[row_idx])
        block_count = int(mask.sum().item())
        if block_count <= 0:
            malformed += 1
            continue
        fp_sol_valid += 1
        sample_id = f"train_{sample_index:07d}"
        valid_fp = fp_sol[row_idx][mask]
        valid_constraints = constraints[row_idx][mask]
        record = training_layout_prior_record(
            sample_id=sample_id,
            sample_index=sample_index,
            block_count=block_count,
            feature_summary={
                "area_shape": _shape(area[row_idx]),
                "b2b_edge_count": _edge_count(b2b[row_idx]) if b2b.ndim >= 2 else 0,
                "p2b_edge_count": _edge_count(p2b[row_idx]) if p2b.ndim >= 2 else 0,
                "pin_count": _edge_count(pins[row_idx]) if pins.ndim >= 2 else 0,
                "constraint_shape": _shape(valid_constraints),
            },
            label_summary={
                "fp_sol_shape": _shape(valid_fp),
                "fp_sol_label_order": "[w, h, x, y]",
                "metrics_shape": _shape(metrics[row_idx]) if metrics.ndim else _shape(metrics),
            },
            missing_fields=["true_route_demand_heatmap", "pseudo_trajectory_actions"],
        )
        records.append(record)
    return records, {
        "batch_size": batch_size,
        "record_count": len(records),
        "fp_sol_contract_valid_count": fp_sol_valid,
        "invalid_or_malformed_sample_count": malformed,
    }


def audit_validation_batch(
    batch: tuple[Any, Any],
    *,
    case_id: str,
    requested_case_id: str | None = None,
) -> Step7LRecord:
    inputs, _labels = batch
    area, b2b, p2b, pins, constraints = inputs
    area0 = area[0] if getattr(area, "ndim", 0) > 1 else area
    block_count = _valid_count_from_area(area0)
    return validation_inference_record(
        case_id=case_id,
        requested_case_id=requested_case_id,
        block_count=block_count,
        feature_summary={
            "area_shape": _shape(area),
            "b2b_shape": _shape(b2b),
            "p2b_shape": _shape(p2b),
            "pins_shape": _shape(pins),
            "constraints_shape": _shape(constraints),
            "b2b_edge_count": (
                _edge_count(b2b[0]) if getattr(b2b, "ndim", 0) >= 3 else _edge_count(b2b)
            ),
            "p2b_edge_count": (
                _edge_count(p2b[0]) if getattr(p2b, "ndim", 0) >= 3 else _edge_count(p2b)
            ),
        },
        missing_fields=["fresh_candidate_requests", "fresh_replay_metrics"],
    )


def load_candidate_quality_records(base_dir: Path) -> tuple[list[Step7LRecord], dict[str, Any]]:
    candidates = [
        base_dir / "artifacts/research/step7ml_g_candidate_quality_examples.json",
        base_dir / "artifacts/research/step7n_i_annotated_candidates.json",
    ]
    source_path = next((path for path in candidates if path.exists()), None)
    if source_path is None:
        return [], {
            "candidate_quality_source_artifact": None,
            "missing_artifacts": [str(p) for p in candidates],
        }
    rows = _rows(load_json(source_path))
    records: list[Step7LRecord] = []
    for idx, row in enumerate(rows):
        candidate_id = str(
            row.get("candidate_id") or row.get("unified_id") or f"candidate_{idx:06d}"
        )
        official_delta = row.get("official_like_cost_delta")
        if official_delta is None and isinstance(row.get("objective_vector"), dict):
            official_delta = row["objective_vector"].get("official_like_cost_delta")
        block_count = int(row.get("changed_block_count") or row.get("macro_closure_size") or 0)
        records.append(
            sidecar_candidate_quality_record(
                candidate_id=candidate_id,
                case_id=str(row.get("case_id")) if row.get("case_id") is not None else None,
                block_count=block_count,
                source_artifact=str(source_path),
                feature_summary={
                    "route_class": row.get("route_class") or row.get("actual_locality_class"),
                    "source_step": row.get("source_step") or row.get("source_branch"),
                    "target_region": row.get("target_region"),
                },
                label_summary={
                    "hard_feasible": bool(row.get("hard_feasible")),
                    "official_like_improving": bool(
                        row.get("official_like_improving")
                        or row.get("official_like_cost_improving")
                    ),
                    "passes_quality_gate": bool(
                        row.get("passes_step7n_i_quality_gate") or row.get("step7n_h_c_retained")
                    ),
                    "dominated_by_original": bool(row.get("dominated_by_original")),
                    "metric_regressing": bool(row.get("metric_regressing"))
                    or (
                        _safe_float(official_delta) > 0
                        and not bool(row.get("official_like_improving"))
                    ),
                    "official_like_cost_delta": official_delta,
                },
                missing_fields=[
                    field
                    for field, value in {
                        "route_class": row.get("route_class") or row.get("actual_locality_class"),
                        "official_like_cost_delta": official_delta,
                    }.items()
                    if value is None
                ],
            )
        )
    return records, {
        "candidate_quality_source_artifact": str(source_path),
        "candidate_quality_source_row_count": len(rows),
    }


def _count_by_family(records: list[Step7LRecord]) -> dict[str, int]:
    return dict(Counter(record.family.value for record in records))


def _decision(metrics: dict[str, Any]) -> str:
    if metrics.get("validation_inference_records_with_fp_sol", 0) > 0:
        return "fix_validation_label_leakage"
    if metrics.get("loaded_training_sample_count", 0) <= 0:
        return "fix_training_loader_or_schema"
    if not metrics.get("candidate_quality_records_separate", False):
        return "fix_candidate_quality_boundary"
    return "promote_to_step7l_heatmap_baseline"


def decision_markdown(result: dict[str, Any]) -> str:
    metrics = result["metrics"]
    decision = result["decision"]
    return f"""# Step7L Phase 0 Learning Data Audit

Decision: `{decision}`

## Key metrics

- loaded_training_sample_count: {metrics['loaded_training_sample_count']}
- fp_sol_contract_valid_count: {metrics['fp_sol_contract_valid_count']}
- invalid_or_malformed_sample_count: {metrics['invalid_or_malformed_sample_count']}
- validation_inference_record_count: {metrics['validation_inference_record_count']}
- validation_inference_records_with_fp_sol: {metrics['validation_inference_records_with_fp_sol']}
- candidate_quality_record_count: {metrics['candidate_quality_record_count']}
- candidate_quality_records_separate: {metrics['candidate_quality_records_separate']}
- record_count_by_family: {metrics['record_count_by_family']}
- runtime_proxy_ms: {metrics['runtime_proxy_ms']:.3f}

## Interpretation

Step7L Phase 0 is a leakage/audit gate. FloorSet training `fp_sol` records are
allowed only for supervised target/heatmap priors; visible-validation records are
stored as inputs-only inference records; Step7 sidecar replay rows are candidate
quality labels only. The next safe branch is the deterministic topology / wire-
demand heatmap baseline if the decision is `promote_to_step7l_heatmap_baseline`.
"""


def run_learning_data_audit(
    base_dir: Path,
    output_path: Path,
    *,
    floorset_root: Path | None = None,
    train_samples: int = 128,
    validation_case_ids: list[str] | None = None,
    grid: int = 16,
    batch_size: int = 32,
    auto_download: bool = False,
) -> dict[str, Any]:
    started = time.perf_counter()
    resolved = resolve_floorset_root(base_dir, floorset_root)
    if resolved is None:
        raise FileNotFoundError("Could not resolve external/FloorSet official checkout")
    evaluator = _import_official_evaluator(resolved)
    probe = probe_training_corpus(resolved)
    training_records: list[Step7LRecord] = []
    training_summary: Counter[str] = Counter()
    with _auto_yes_download(auto_download):
        train_loader = evaluator.get_training_dataloader(
            data_path=str(resolved),
            batch_size=batch_size,
            num_samples=train_samples,
            shuffle=False,
        )
        sample_offset = 0
        for batch in train_loader:
            remaining = train_samples - len(training_records)
            if remaining <= 0:
                break
            records, summary = audit_training_batch(
                batch, sample_offset=sample_offset, max_records=remaining
            )
            training_records.extend(records)
            training_summary.update(summary)
            sample_offset += int(summary["batch_size"])
            if len(training_records) >= train_samples:
                break
        validation_records: list[Step7LRecord] = []
        requested_ids = validation_case_ids or []
        validation_count = len(requested_ids) if requested_ids else 0
        if validation_count > 0:
            validation_loader = evaluator.get_validation_dataloader(
                data_path=str(resolved), batch_size=1
            )
            for idx, batch in enumerate(validation_loader):
                if idx >= validation_count:
                    break
                requested = requested_ids[idx] if idx < len(requested_ids) else None
                validation_records.append(
                    audit_validation_batch(
                        batch,
                        case_id=requested or f"loader_index_{idx:03d}",
                        requested_case_id=requested,
                    )
                )
    candidate_records, candidate_report = load_candidate_quality_records(base_dir)
    all_records = training_records + validation_records + candidate_records
    assert_no_validation_label_leakage(validation_records)
    validation_leaks = sum(1 for record in validation_records if record.has_target_label)
    candidate_separate = all(
        record.family == Step7LRecordFamily.SIDECAR_CANDIDATE_QUALITY
        and "layout_prior_target" in record.forbidden_uses
        for record in candidate_records
    )
    metrics = {
        "grid": grid,
        "loaded_training_sample_count": len(training_records),
        "fp_sol_contract_valid_count": int(training_summary["fp_sol_contract_valid_count"]),
        "invalid_or_malformed_sample_count": int(
            training_summary["invalid_or_malformed_sample_count"]
        ),
        "validation_inference_record_count": len(validation_records),
        "validation_inference_records_with_fp_sol": validation_leaks,
        "candidate_quality_record_count": len(candidate_records),
        "candidate_quality_records_separate": candidate_separate,
        "record_count_by_family": _count_by_family(all_records),
        "candidate_quality_source_artifact": candidate_report.get(
            "candidate_quality_source_artifact"
        ),
        "floorset_training_loader_ready": bool(probe.training_loader_ready),
        "runtime_proxy_ms": (time.perf_counter() - started) * 1000.0,
    }
    result = {
        "schema": "step7l_phase0_audit_v1",
        "decision": _decision(metrics),
        "floorset_inventory": probe_to_json(probe),
        "metrics": metrics,
        "records": [record.to_json() for record in all_records],
    }
    write_json(output_path, result)
    md_path = output_path.with_suffix(".md")
    md_path.write_text(decision_markdown(result), encoding="utf-8")
    return result
