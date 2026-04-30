"""Step7ML-G training-backed macro layout data mart.

Sidecar-only data builder that keeps official FloorSet training layout labels
separate from Step7 candidate-quality labels. It uses the official FloorSet-Lite
training loader for placement priors and existing Step7N-I artifacts for
sidecar candidate quality labels.
"""

from __future__ import annotations

import hashlib
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import torch

from puzzleplace.ml.floorset_training_corpus import (
    _auto_yes_download,
    _block_payload,
    _import_official_evaluator,
    _summary,
    _valid_block_mask,
    load_json,
    probe_to_json,
    probe_training_corpus,
    resolve_floorset_root,
    write_json,
)

REGION_ROWS = 4
REGION_COLS = 4
DECISIONS = {
    "promote_to_supervised_macro_layout_training",
    "promote_to_masked_region_heatmap_training",
    "promote_to_candidate_ranker_training",
    "expand_data_mart_to_100k",
    "build_missing_feature_extractors",
    "fix_training_loader_or_schema",
    "inconclusive_due_to_data_quality",
}


def rows(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, dict):
        value = payload.get("rows", [])
        return [row for row in value if isinstance(row, dict)] if isinstance(value, list) else []
    if isinstance(payload, list):
        return [row for row in payload if isinstance(row, dict)]
    return []


def as_float(value: Any, default: float = 0.0) -> float:
    if value is None:
        return default
    if isinstance(value, int | float):
        return float(value)
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def stable_split(key: str) -> str:
    bucket = int(hashlib.sha1(key.encode("utf-8")).hexdigest()[:8], 16) % 100
    if bucket < 80:
        return "train"
    if bucket < 90:
        return "validation"
    return "test"


def _bbox(blocks: list[dict[str, Any]]) -> dict[str, float]:
    x0 = min(as_float(block["x"]) for block in blocks)
    y0 = min(as_float(block["y"]) for block in blocks)
    x1 = max(as_float(block["x"]) + as_float(block["w"]) for block in blocks)
    y1 = max(as_float(block["y"]) + as_float(block["h"]) for block in blocks)
    return {"x": x0, "y": y0, "w": x1 - x0, "h": y1 - y0, "area": (x1 - x0) * (y1 - y0)}


def _aspect(box: dict[str, float]) -> float | None:
    height = box.get("h", 0.0)
    if abs(height) <= 1e-12:
        return None
    return box.get("w", 0.0) / height


def _region_id_for_point(x: float, y: float, frame: dict[str, float]) -> str:
    width = max(frame["w"], 1e-9)
    height = max(frame["h"], 1e-9)
    col = min(max(int(((x - frame["x"]) / width) * REGION_COLS), 0), REGION_COLS - 1)
    row = min(max(int(((y - frame["y"]) / height) * REGION_ROWS), 0), REGION_ROWS - 1)
    return f"r{row}c{col}"


def _region_index(region_id: str) -> tuple[int, int]:
    row = int(region_id[1])
    col = int(region_id[3])
    return row, col


def _empty_grid(value: float = 0.0) -> list[list[float]]:
    return [[value for _ in range(REGION_COLS)] for _ in range(REGION_ROWS)]


def _sample_frame(blocks: list[dict[str, Any]]) -> dict[str, float]:
    box = _bbox(blocks)
    return {"x": box["x"], "y": box["y"], "w": max(box["w"], 1.0), "h": max(box["h"], 1.0)}


def _closure_groups(blocks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: list[dict[str, Any]] = []
    for closure_type, field in (("mib", "mib"), ("cluster", "cluster")):
        by_group: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for block in blocks:
            group_id = int(block.get(field, 0))
            if group_id > 0:
                by_group[group_id].append(block)
        for group_id, members in sorted(by_group.items()):
            if len(members) < 2:
                continue
            box = _bbox(members)
            groups.append(
                {
                    "closure_type": closure_type,
                    "group_id": group_id,
                    "member_block_ids": [int(block["block_id"]) for block in members],
                    "block_count": len(members),
                    "closure_bbox": box,
                    "closure_aspect": _aspect(box),
                    "closure_area": box["area"],
                    "fixed_count": sum(1 for block in members if block.get("fixed")),
                    "preplaced_count": sum(1 for block in members if block.get("preplaced")),
                    "fixed_shape_count": None,
                    "missing_fields": ["fixed_shape", "closure_net_features"],
                    "block_geometry": [
                        {
                            "block_id": int(block["block_id"]),
                            "w": as_float(block["w"]),
                            "h": as_float(block["h"]),
                            "x": as_float(block["x"]),
                            "y": as_float(block["y"]),
                            "fixed": bool(block.get("fixed")),
                            "preplaced": bool(block.get("preplaced")),
                            "fixed_shape": None,
                            "boundary": int(block.get("boundary", 0)),
                        }
                        for block in members
                    ],
                    "normalized_internal_coordinates": [
                        {
                            "block_id": int(block["block_id"]),
                            "rel_x": (as_float(block["x"]) - box["x"]) / max(box["w"], 1e-9),
                            "rel_y": (as_float(block["y"]) - box["y"]) / max(box["h"], 1e-9),
                            "rel_w": as_float(block["w"]) / max(box["w"], 1e-9),
                            "rel_h": as_float(block["h"]) / max(box["h"], 1e-9),
                        }
                        for block in members
                    ],
                }
            )
    return groups


def _region_heatmap_example(
    sample_id: str,
    sample_index: int,
    blocks: list[dict[str, Any]],
) -> dict[str, Any]:
    frame = _sample_frame(blocks)
    count_grid = _empty_grid()
    area_grid = _empty_grid()
    fixed_grid = _empty_grid()
    preplaced_grid = _empty_grid()
    block_regions: list[dict[str, Any]] = []
    total_area = sum(as_float(block["area"]) for block in blocks) or 1.0
    for block in blocks:
        cx = as_float(block["x"]) + as_float(block["w"]) / 2.0
        cy = as_float(block["y"]) + as_float(block["h"]) / 2.0
        region_id = _region_id_for_point(cx, cy, frame)
        row, col = _region_index(region_id)
        count_grid[row][col] += 1.0
        area_grid[row][col] += as_float(block["area"]) / total_area
        if block.get("fixed"):
            fixed_grid[row][col] += 1.0
        if block.get("preplaced"):
            preplaced_grid[row][col] += 1.0
        block_regions.append(
            {
                "block_id": int(block["block_id"]),
                "region_id": region_id,
                "center_x_norm": (cx - frame["x"]) / max(frame["w"], 1e-9),
                "center_y_norm": (cy - frame["y"]) / max(frame["h"], 1e-9),
            }
        )
    total_blocks = max(len(blocks), 1)
    occupancy = [[value / total_blocks for value in row] for row in count_grid]
    free_space = [[max(0.0, 1.0 - value) for value in row] for row in occupancy]
    return {
        "schema": "step7ml_g_region_heatmap_example_v1",
        "sample_id": sample_id,
        "sample_index": sample_index,
        "source": "floorset_lite_training_fp_sol",
        "grid": {"rows": REGION_ROWS, "cols": REGION_COLS},
        "frame": frame,
        "block_count": len(blocks),
        "block_region_labels": block_regions,
        "region_distribution": occupancy,
        "occupancy_mask": occupancy,
        "free_space_proxy": free_space,
        "fixed_mask": [[value / total_blocks for value in row] for row in fixed_grid],
        "preplaced_mask": [[value / total_blocks for value in row] for row in preplaced_grid],
        "area_distribution": area_grid,
        "missing_fields": ["true_route_demand_heatmap", "pin_density_heatmap"],
        "mart_split": stable_split(sample_id),
    }


def _build_training_rows(
    loader: Any,
    *,
    requested_samples: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    layout_examples: list[dict[str, Any]] = []
    region_examples: list[dict[str, Any]] = []
    malformed = 0
    fp_sol_contract_validation_count = 0
    started = time.perf_counter()
    sample_index = 0
    for batch in loader:
        area, b2b, p2b, pins, constraints, _tree_sol, fp_sol, metrics = batch
        if fp_sol.shape[-1] != 4:
            malformed += int(area.shape[0])
            sample_index += int(area.shape[0])
            continue
        for row_idx in range(area.shape[0]):
            if sample_index >= requested_samples:
                runtime_ms = (time.perf_counter() - started) * 1000.0
                return layout_examples, region_examples, {
                    "invalid_or_malformed_sample_count": malformed,
                    "fp_sol_contract_validation_count": fp_sol_contract_validation_count,
                    "runtime_ms": runtime_ms,
                }
            try:
                mask = _valid_block_mask(area[row_idx], fp_sol[row_idx])
                valid_ids = [
                    int(idx) for idx in torch.nonzero(mask, as_tuple=False).flatten().tolist()
                ]
                if not valid_ids:
                    malformed += 1
                    sample_index += 1
                    continue
                blocks = [
                    _block_payload(idx, fp_sol[row_idx, idx], constraints[row_idx])
                    for idx in valid_ids
                ]
                sample_id = f"train_{sample_index:07d}"
                closures = _closure_groups(blocks)
                sample_metrics = [float(value) for value in metrics[row_idx].tolist()]
                for closure_index, closure in enumerate(closures):
                    layout_examples.append(
                        {
                            "schema": "step7ml_g_layout_prior_example_v1",
                            "sample_id": sample_id,
                            "sample_index": sample_index,
                            "closure_index": closure_index,
                            "source": "floorset_lite_training_fp_sol",
                            "source_split": "floorset_lite_training",
                            "mart_split": stable_split(sample_id),
                            "label_contract": "fp_sol[..., 0:4] is [w, h, x, y]",
                            "block_count_in_sample": len(valid_ids),
                            "pin_count": int((pins[row_idx, :, 0] >= 0).sum().item())
                            if pins.ndim == 3
                            else 0,
                            "b2b_edge_count": int((b2b[row_idx, :, 0] >= 0).sum().item())
                            if b2b.ndim == 3
                            else 0,
                            "p2b_edge_count": int((p2b[row_idx, :, 0] >= 0).sum().item())
                            if p2b.ndim == 3
                            else 0,
                            "training_metrics": sample_metrics,
                            **closure,
                        }
                    )
                region_examples.append(_region_heatmap_example(sample_id, sample_index, blocks))
                fp_sol_contract_validation_count += 1
            except Exception:  # pragma: no cover - environment/data defensive path.
                malformed += 1
            sample_index += 1
    runtime_ms = (time.perf_counter() - started) * 1000.0
    return layout_examples, region_examples, {
        "invalid_or_malformed_sample_count": malformed,
        "fp_sol_contract_validation_count": fp_sol_contract_validation_count,
        "runtime_ms": runtime_ms,
    }


def _quality_filtered_ids(base_dir: Path) -> set[str]:
    path = base_dir / "artifacts/research/step7n_i_quality_filtered_candidates.json"
    if not path.exists():
        return set()
    return {
        str(row.get("candidate_id"))
        for row in rows(load_json(path))
        if row.get("candidate_id")
    }


def _archive_ids(base_dir: Path) -> set[str]:
    path = base_dir / "artifacts/research/step7n_i_archive_replay.json"
    if not path.exists():
        return set()
    payload = load_json(path)
    ids: set[str] = set()
    per_case = payload.get("per_case", {}) if isinstance(payload, dict) else {}
    if isinstance(per_case, dict):
        for case_payload in per_case.values():
            for row in case_payload.get("front_rows", []) if isinstance(case_payload, dict) else []:
                if isinstance(row, dict) and row.get("candidate_id"):
                    ids.add(str(row["candidate_id"]))
    return ids


def _candidate_quality_examples(base_dir: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    annotated_path = base_dir / "artifacts/research/step7n_i_annotated_candidates.json"
    if not annotated_path.exists():
        return [], {"artifact_label_join_coverage": 0.0, "missing_artifacts": [str(annotated_path)]}
    retained_ids = _quality_filtered_ids(base_dir)
    archive_ids = _archive_ids(base_dir)
    annotated = rows(load_json(annotated_path))
    examples: list[dict[str, Any]] = []
    for row in annotated:
        candidate_id = str(row.get("candidate_id") or row.get("unified_id") or "")
        raw_vector = row.get("objective_vector")
        vector = raw_vector if isinstance(raw_vector, dict) else {}
        official_delta = row.get(
            "official_like_cost_delta", vector.get("official_like_cost_delta")
        )
        examples.append(
            {
                "schema": "step7ml_g_candidate_quality_example_v1",
                "candidate_id": candidate_id,
                "source_step": row.get("source_branch") or row.get("source_artifact"),
                "source_artifact": row.get("source_artifact"),
                "case_id": row.get("case_id"),
                "source_case": row.get("source_case"),
                "mart_split": stable_split(f"step7_case_{row.get('case_id')}")
                if row.get("case_id") is not None
                else "unknown",
                "route_class": row.get("actual_locality_class"),
                "lane": row.get("lane"),
                "hard_feasible": bool(row.get("hard_feasible")),
                "non_noop": bool(not row.get("no_op")),
                "non_original_non_noop": bool(row.get("non_original_non_noop")),
                "hard_feasible_non_noop": bool(row.get("hard_feasible_non_noop")),
                "official_like_improving": bool(row.get("official_like_cost_improving")),
                "passes_step7n_i_quality_gate": candidate_id in retained_ids
                or bool(row.get("step7n_h_c_retained")),
                "selected_for_step7n_i_archive": candidate_id in archive_ids
                or bool(row.get("step7n_i_selected_for_archive")),
                "dominated_by_original": bool(row.get("dominated_by_original")),
                "dominated_by_current": row.get("dominated_by_current"),
                "metric_regressing": as_float(official_delta) > 0
                and not bool(row.get("official_like_cost_improving")),
                "hpwl_delta": row.get("hpwl_delta", vector.get("hpwl_delta")),
                "bbox_area_delta": row.get("bbox_area_delta", vector.get("bbox_area_delta")),
                "soft_constraint_delta": row.get(
                    "soft_constraint_delta", vector.get("soft_constraint_delta")
                ),
                "official_like_cost_delta": official_delta,
                "macro_closure_size": row.get("macro_closure_size"),
                "changed_block_count": row.get("changed_block_count"),
                "slot_repack_method": row.get("slot_repack_method"),
                "target_origin": row.get("target_origin"),
                "target_region": row.get("target_region"),
                "failure_attribution": row.get("failure_attribution"),
                "metric_regression_reason": row.get("metric_regression_reason"),
                "missing_fields": [
                    name
                    for name, value in {
                        "dominated_by_current": row.get("dominated_by_current"),
                        "soft_constraint_delta": row.get("soft_constraint_delta"),
                        "target_region": row.get("target_region"),
                    }.items()
                    if value is None
                ],
            }
        )
    coverage = len(examples) / max(len(annotated), 1)
    return examples, {
        "artifact_label_join_coverage": coverage,
        "annotated_candidate_count": len(annotated),
        "quality_filtered_candidate_count": len(retained_ids),
        "archive_candidate_count": len(archive_ids),
    }


def _count_by(rows_: list[dict[str, Any]], key: str) -> dict[str, int]:
    return dict(Counter(str(row.get(key)) for row in rows_))


def _split_report(
    layout_examples: list[dict[str, Any]],
    region_examples: list[dict[str, Any]],
    candidate_examples: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "schema": "step7ml_g_split_report_v1",
        "layout_prior_split_counts": _count_by(layout_examples, "mart_split"),
        "region_heatmap_split_counts": _count_by(region_examples, "mart_split"),
        "candidate_quality_split_counts": _count_by(candidate_examples, "mart_split"),
        "split_rule": (
            "stable sha1 hash; training samples split by sample_id, "
            "Step7 artifacts by case_id"
        ),
    }


def _missing_report(
    layout_examples: list[dict[str, Any]],
    region_examples: list[dict[str, Any]],
    candidate_examples: list[dict[str, Any]],
) -> dict[str, Any]:
    groups = {
        "layout_prior": layout_examples,
        "region_heatmap": region_examples,
        "candidate_quality": candidate_examples,
    }
    report: dict[str, Any] = {"schema": "step7ml_g_missing_field_report_v1", "groups": {}}
    for name, group_rows in groups.items():
        missing_counter: Counter[str] = Counter()
        for row in group_rows:
            for field in row.get("missing_fields", []) or []:
                missing_counter[str(field)] += 1
        denom = max(len(group_rows), 1)
        report["groups"][name] = {
            "row_count": len(group_rows),
            "missing_field_counts": dict(missing_counter),
            "missing_field_rates": {
                field: count / denom for field, count in sorted(missing_counter.items())
            },
        }
    return report


def _schema_report() -> dict[str, Any]:
    return {
        "schema": "step7ml_g_schema_report_v1",
        "families": {
            "layout_prior_examples": {
                "source": "FloorSet-Lite training fp_sol labels",
                "target": "closure internal geometry and normalized coordinates",
                "not_for": "Step7N-I quality gate labels",
            },
            "region_heatmap_examples": {
                "source": "FloorSet-Lite training fp_sol labels",
                "target": "block/supernode coarse target region distributions and masks",
                "not_for": "candidate pass/fail labels",
            },
            "candidate_quality_examples": {
                "source": "Step7N-I sidecar artifacts",
                "target": "candidate quality labels such as gate pass/fail and metric regression",
                "not_for": "general placement imitation labels",
            },
        },
        "label_separation_rule": (
            "FloorSet training labels teach layout priors; Step7 artifacts teach "
            "sidecar candidate quality. They are not collapsed into a single target."
        ),
    }


def _decide(metrics: dict[str, Any]) -> str:
    if (
        not metrics.get("training_loader_ready")
        or metrics.get("invalid_or_malformed_sample_count", 0) > 0
    ):
        return "fix_training_loader_or_schema"
    if metrics.get("extracted_macro_closure_count", 0) >= 10000:
        return "promote_to_supervised_macro_layout_training"
    if metrics.get("region_heatmap_example_count", 0) >= 10000:
        return "promote_to_masked_region_heatmap_training"
    if metrics.get("candidate_quality_example_count", 0) > 0:
        return "promote_to_candidate_ranker_training"
    return "inconclusive_due_to_data_quality"


def decision_markdown(metrics: dict[str, Any], decision: str) -> str:
    return f"""# Step7ML-G Training-Backed Macro Layout Data Mart

Decision: `{decision}`

## Key metrics

- requested_training_sample_count: {metrics['requested_training_sample_count']}
- loaded_training_sample_count: {metrics['loaded_training_sample_count']}
- extracted_macro_closure_count: {metrics['extracted_macro_closure_count']}
- closure_count_by_type: {metrics['closure_count_by_type']}
- layout_prior_example_count: {metrics['layout_prior_example_count']}
- region_heatmap_example_count: {metrics['region_heatmap_example_count']}
- candidate_quality_example_count: {metrics['candidate_quality_example_count']}
- unique_training_case_count: {metrics['unique_training_case_count']}
- unique_step7_artifact_case_count: {metrics['unique_step7_artifact_case_count']}
- fp_sol_contract_validation_count: {metrics['fp_sol_contract_validation_count']}
- invalid_or_malformed_sample_count: {metrics['invalid_or_malformed_sample_count']}
- artifact_label_join_coverage: {metrics['artifact_label_join_coverage']}
- runtime_proxy_ms: {metrics['runtime_proxy_ms']:.3f}

## Interpretation

Step7ML-G separates real FloorSet training layout-prior labels from Step7
candidate-quality labels. The training corpus now provides enough macro closure
geometry labels for supervised macro layout experiments. Candidate quality labels
remain a separate sidecar table for Step7N-I gate/ranker training.

## Next branch

Use `layout_prior_examples` for supervised macro closure layout training first.
Use `candidate_quality_examples` as a separate weak ranker/gate dataset; do not
train a generator from Step7 candidate labels alone.
"""


def build_training_backed_data_mart(
    base_dir: Path,
    output_dir: Path,
    *,
    floorset_root: Path | None = None,
    requested_training_sample_count: int = 10_000,
    batch_size: int = 64,
    auto_download: bool = False,
) -> dict[str, Any]:
    started = time.perf_counter()
    resolved = resolve_floorset_root(base_dir, floorset_root)
    if resolved is None:
        raise FileNotFoundError("Could not resolve external/FloorSet official checkout")
    probe = probe_training_corpus(resolved)
    inventory = probe_to_json(probe)
    evaluator = _import_official_evaluator(resolved)
    with _auto_yes_download(auto_download):
        loader = evaluator.get_training_dataloader(
            data_path=str(resolved),
            batch_size=batch_size,
            num_samples=requested_training_sample_count,
            shuffle=False,
        )
        layout_examples, region_examples, training_report = _build_training_rows(
            loader, requested_samples=requested_training_sample_count
        )
    candidate_examples, candidate_report = _candidate_quality_examples(base_dir)
    split_report = _split_report(layout_examples, region_examples, candidate_examples)
    missing_report = _missing_report(layout_examples, region_examples, candidate_examples)
    metrics = {
        "requested_training_sample_count": requested_training_sample_count,
        "loaded_training_sample_count": int(training_report["fp_sol_contract_validation_count"]),
        "extracted_macro_closure_count": len(layout_examples),
        "closure_count_by_type": _count_by(layout_examples, "closure_type"),
        "closure_block_count_distribution": _summary(
            [int(row["block_count"]) for row in layout_examples]
        ),
        "layout_prior_example_count": len(layout_examples),
        "region_heatmap_example_count": len(region_examples),
        "candidate_quality_example_count": len(candidate_examples),
        "unique_training_case_count": len({row["sample_id"] for row in region_examples}),
        "unique_step7_artifact_case_count": len(
            {row.get("case_id") for row in candidate_examples if row.get("case_id") is not None}
        ),
        "train_validation_test_split_counts": split_report,
        "missing_field_rate_by_feature_group": missing_report["groups"],
        "fp_sol_contract_validation_count": int(
            training_report["fp_sol_contract_validation_count"]
        ),
        "invalid_or_malformed_sample_count": int(
            training_report["invalid_or_malformed_sample_count"]
        ),
        "artifact_label_join_coverage": float(
            candidate_report.get("artifact_label_join_coverage", 0.0)
        ),
        "runtime_proxy_ms": (time.perf_counter() - started) * 1000.0,
        "training_loader_ready": bool(inventory.get("training_loader_ready")),
    }
    decision = _decide(metrics)
    data_inventory = {
        "schema": "step7ml_g_data_inventory_v1",
        "decision": decision,
        "floorset_inventory": inventory,
        "candidate_artifact_report": candidate_report,
        "metrics": metrics,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "step7ml_g_data_inventory.json", data_inventory)
    write_json(
        output_dir / "step7ml_g_layout_prior_examples.json",
        {"schema": "step7ml_g_layout_prior_examples_v1", "rows": layout_examples},
    )
    write_json(
        output_dir / "step7ml_g_region_heatmap_examples.json",
        {"schema": "step7ml_g_region_heatmap_examples_v1", "rows": region_examples},
    )
    write_json(
        output_dir / "step7ml_g_candidate_quality_examples.json",
        {"schema": "step7ml_g_candidate_quality_examples_v1", "rows": candidate_examples},
    )
    write_json(output_dir / "step7ml_g_schema_report.json", _schema_report())
    write_json(output_dir / "step7ml_g_split_report.json", split_report)
    write_json(output_dir / "step7ml_g_missing_field_report.json", missing_report)
    (output_dir / "step7ml_g_decision.md").write_text(
        decision_markdown(metrics, decision), encoding="utf-8"
    )
    return {"decision": decision, "metrics": metrics}
