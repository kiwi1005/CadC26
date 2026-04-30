"""Step7DATA FloorSet-Lite training corpus loader sidecar.

This module validates the official FloorSet training dataloader and converts a
small bounded sample into JSON-serializable macro/closure labels for Step7ML.
It deliberately treats the official FloorSet checkout as read-only code: data is
loaded through the official dataloader and no contest runtime/finalizer behavior
is modified.
"""

from __future__ import annotations

import builtins
import importlib
import json
import sys
import time
from collections import Counter
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

DATASET_URL = "https://huggingface.co/datasets/IntelLabs/FloorSet/resolve/main/LiteTensorData_v2.tar.gz"
LOADER_CONTRACT = (
    "get_training_dataloader(...)-> "
    "area,b2b,p2b,pins,constraints,tree_sol,fp_sol,metrics"
)
TRAIN_ARCHIVE_NAMES = ("LiteTensorData_v2.tar.gz", "data/LiteTensorData_v2.tar.gz")
DECISIONS = {
    "training_loader_validated",
    "download_required",
    "download_attempt_failed",
    "loader_valid_but_macro_labels_sparse",
    "inconclusive_due_to_loader_error",
}


@dataclass(frozen=True)
class CorpusProbe:
    floorset_root: Path
    validation_case_count: int
    training_archive_available: bool
    training_unpacked_worker_count: int
    training_loader_ready: bool
    training_archive_candidates: list[str]


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def resolve_floorset_root(base_dir: Path, explicit_root: Path | None = None) -> Path | None:
    candidates = []
    if explicit_root is not None:
        candidates.append(explicit_root)
    candidates.extend(
        [
            base_dir / "external" / "FloorSet",
            Path("/home/hwchen/PROJ/CadC26/external/FloorSet"),
            base_dir.parent / "CadC26" / "external" / "FloorSet",
        ]
    )
    for candidate in candidates:
        if (candidate / "iccad2026contest" / "iccad2026_evaluate.py").exists():
            return candidate.resolve()
    return None


def probe_training_corpus(floorset_root: Path) -> CorpusProbe:
    validation_cases = list((floorset_root / "LiteTensorDataTest").glob("config_*"))
    archives = [floorset_root / name for name in TRAIN_ARCHIVE_NAMES]
    workers = list((floorset_root / "floorset_lite").glob("worker_*"))
    return CorpusProbe(
        floorset_root=floorset_root,
        validation_case_count=len(validation_cases),
        training_archive_available=any(path.exists() for path in archives),
        training_unpacked_worker_count=len(workers),
        training_loader_ready=len(workers) >= 100,
        training_archive_candidates=[str(path) for path in archives],
    )


def probe_to_json(probe: CorpusProbe | None, *, root: Path | None = None) -> dict[str, Any]:
    if probe is None:
        return {
            "resolved_root": str(root) if root is not None else None,
            "validation_case_count": 0,
            "training_archive_available": False,
            "training_unpacked_worker_count": 0,
            "training_loader_ready": False,
            "training_archive_candidates": [],
            "dataset_url": DATASET_URL,
        }
    return {
        "resolved_root": str(probe.floorset_root),
        "validation_case_count": probe.validation_case_count,
        "training_archive_available": probe.training_archive_available,
        "training_unpacked_worker_count": probe.training_unpacked_worker_count,
        "training_loader_ready": probe.training_loader_ready,
        "training_archive_candidates": probe.training_archive_candidates,
        "dataset_url": DATASET_URL,
        "expected_layout": {
            "official_loader_data_path": "FloorSet root directory",
            "training_unpacked_dir": str(probe.floorset_root / "floorset_lite"),
            "loader_contract": LOADER_CONTRACT,
            "lite_label_contract": "fp_sol[..., 0:4] is [w, h, x, y] for valid blocks",
        },
    }


@contextmanager
def _auto_yes_download(enabled: bool) -> Iterator[None]:
    original = builtins.input
    if enabled:
        builtins.input = lambda _prompt="": "y"
    try:
        yield
    finally:
        builtins.input = original


def _import_official_evaluator(floorset_root: Path) -> Any:
    for path in (floorset_root / "iccad2026contest", floorset_root):
        path_str = str(path)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)
    return importlib.import_module("iccad2026_evaluate")


def _valid_block_mask(area_target: torch.Tensor, fp_sol: torch.Tensor) -> torch.Tensor:
    area_valid = area_target >= 0
    fp_valid = ~(fp_sol == -1).all(dim=-1)
    return area_valid & fp_valid


def _tensor_shape(value: torch.Tensor) -> list[int]:
    return [int(dim) for dim in value.shape]


def _float(value: Any) -> float:
    if isinstance(value, torch.Tensor):
        return float(value.item())
    return float(value)


def _block_payload(
    block_id: int,
    sol_row: torch.Tensor,
    constraints: torch.Tensor,
) -> dict[str, Any]:
    w, h, x, y = [_float(value) for value in sol_row[:4]]
    row = constraints[block_id]
    return {
        "block_id": int(block_id),
        "w": w,
        "h": h,
        "x": x,
        "y": y,
        "area": w * h,
        "fixed": bool(row[0].item()),
        "preplaced": bool(row[1].item()),
        "mib": int(row[2].item()) if row.numel() > 2 else 0,
        "cluster": int(row[3].item()) if row.numel() > 3 else 0,
        "boundary": int(row[4].item()) if row.numel() > 4 else 0,
    }


def _bbox(blocks: list[dict[str, Any]]) -> dict[str, float]:
    x0 = min(block["x"] for block in blocks)
    y0 = min(block["y"] for block in blocks)
    x1 = max(block["x"] + block["w"] for block in blocks)
    y1 = max(block["y"] + block["h"] for block in blocks)
    return {"x": x0, "y": y0, "w": x1 - x0, "h": y1 - y0, "area": (x1 - x0) * (y1 - y0)}


def _closure_groups(blocks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: list[dict[str, Any]] = []
    for key in ("mib", "cluster"):
        by_group: dict[int, list[dict[str, Any]]] = {}
        for block in blocks:
            group_id = int(block.get(key, 0))
            if group_id > 0:
                by_group.setdefault(group_id, []).append(block)
        for group_id, members in sorted(by_group.items()):
            if len(members) < 2:
                continue
            box = _bbox(members)
            groups.append(
                {
                    "closure_type": key,
                    "group_id": group_id,
                    "block_ids": [int(block["block_id"]) for block in members],
                    "block_count": len(members),
                    "bbox": box,
                    "normalized_blocks": [
                        {
                            "block_id": int(block["block_id"]),
                            "rel_x": (block["x"] - box["x"]) / max(box["w"], 1e-9),
                            "rel_y": (block["y"] - box["y"]) / max(box["h"], 1e-9),
                            "rel_w": block["w"] / max(box["w"], 1e-9),
                            "rel_h": block["h"] / max(box["h"], 1e-9),
                        }
                        for block in members
                    ],
                    "label_source": "floorset_lite_training_fp_sol",
                    "label_contract": "exact training label fp_sol provides [w,h,x,y]",
                }
            )
    return groups


def _summarize_batch(batch: tuple[torch.Tensor, ...]) -> dict[str, Any]:
    area, b2b, p2b, pins, constraints, tree_sol, fp_sol, metrics = batch
    valid_counts = [
        _valid_block_mask(area[i], fp_sol[i]).sum().item() for i in range(area.shape[0])
    ]
    return {
        "batch_size": int(area.shape[0]),
        "shapes": {
            "area_target": _tensor_shape(area),
            "b2b_connectivity": _tensor_shape(b2b),
            "p2b_connectivity": _tensor_shape(p2b),
            "pins_pos": _tensor_shape(pins),
            "constraints": _tensor_shape(constraints),
            "tree_sol": _tensor_shape(tree_sol),
            "fp_sol": _tensor_shape(fp_sol),
            "metrics": _tensor_shape(metrics),
        },
        "valid_block_counts": [int(value) for value in valid_counts],
        "fp_sol_last_dim": int(fp_sol.shape[-1]),
        "fp_sol_label_order": "[w, h, x, y]",
    }


def collect_training_examples(
    loader: Any,
    *,
    max_examples: int,
    max_macro_labels: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    examples: list[dict[str, Any]] = []
    macro_labels: list[dict[str, Any]] = []
    first_batch_summary: dict[str, Any] = {}
    sample_index = 0
    for batch in loader:
        if not first_batch_summary:
            first_batch_summary = _summarize_batch(batch)
        area, b2b, p2b, pins, constraints, _tree_sol, fp_sol, metrics = batch
        for row_idx in range(area.shape[0]):
            if sample_index >= max_examples:
                return examples, macro_labels, first_batch_summary
            mask = _valid_block_mask(area[row_idx], fp_sol[row_idx])
            valid_ids = [int(idx) for idx in torch.nonzero(mask, as_tuple=False).flatten().tolist()]
            blocks = [
                _block_payload(idx, fp_sol[row_idx, idx], constraints[row_idx])
                for idx in valid_ids
            ]
            closures = _closure_groups(blocks)
            example = {
                "sample_index": sample_index,
                "block_count": len(valid_ids),
                "pin_count": (
                    int((pins[row_idx, :, 0] >= 0).sum().item()) if pins.ndim == 3 else 0
                ),
                "b2b_edge_count": (
                    int((b2b[row_idx, :, 0] >= 0).sum().item()) if b2b.ndim == 3 else 0
                ),
                "p2b_edge_count": (
                    int((p2b[row_idx, :, 0] >= 0).sum().item()) if p2b.ndim == 3 else 0
                ),
                "fixed_count": sum(1 for block in blocks if block["fixed"]),
                "preplaced_count": sum(1 for block in blocks if block["preplaced"]),
                "mib_group_count": len({block["mib"] for block in blocks if block["mib"] > 0}),
                "cluster_group_count": len(
                    {block["cluster"] for block in blocks if block["cluster"] > 0}
                ),
                "macro_closure_count": len(closures),
                "metrics": [float(value) for value in metrics[row_idx].tolist()],
                "label_contract": "FloorSet-Lite fp_sol rows are [w,h,x,y] for each valid block",
            }
            examples.append(example)
            for closure_index, closure in enumerate(closures):
                if len(macro_labels) >= max_macro_labels:
                    break
                macro_labels.append(
                    {
                        "sample_index": sample_index,
                        "closure_index": closure_index,
                        **closure,
                    }
                )
            sample_index += 1
    return examples, macro_labels, first_batch_summary


def validate_training_loader(
    floorset_root: Path,
    *,
    auto_download: bool,
    batch_size: int,
    num_samples: int,
    max_examples: int,
    max_macro_labels: int,
) -> dict[str, Any]:
    evaluator = _import_official_evaluator(floorset_root)
    started = time.perf_counter()
    with _auto_yes_download(auto_download):
        loader = evaluator.get_training_dataloader(
            data_path=str(floorset_root),
            batch_size=batch_size,
            num_samples=num_samples,
            shuffle=False,
        )
        examples, macro_labels, first_batch = collect_training_examples(
            loader,
            max_examples=max_examples,
            max_macro_labels=max_macro_labels,
        )
    return {
        "loader_status": "ok",
        "auto_download_enabled": auto_download,
        "requested_num_samples": num_samples,
        "collected_example_count": len(examples),
        "macro_label_count": len(macro_labels),
        "first_batch": first_batch,
        "example_block_count_summary": _summary([row["block_count"] for row in examples]),
        "macro_label_block_count_summary": _summary([row["block_count"] for row in macro_labels]),
        "runtime_ms": (time.perf_counter() - started) * 1000.0,
        "examples": examples,
        "macro_labels": macro_labels,
    }


def _summary(values: list[int | float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "min": None, "max": None, "mean": None}
    return {
        "count": len(values),
        "min": min(values),
        "max": max(values),
        "mean": sum(values) / len(values),
    }


def decide(inventory: dict[str, Any], loader_report: dict[str, Any] | None) -> str:
    if loader_report is None:
        if not inventory.get("training_loader_ready"):
            return "download_required"
        return "inconclusive_due_to_loader_error"
    if loader_report.get("loader_status") != "ok":
        if loader_report.get("auto_download_enabled"):
            return "download_attempt_failed"
        return "inconclusive_due_to_loader_error"
    if int(loader_report.get("macro_label_count", 0)) == 0:
        return "loader_valid_but_macro_labels_sparse"
    return "training_loader_validated"


def run_step7data(
    base_dir: Path,
    output_dir: Path,
    *,
    floorset_root: Path | None = None,
    auto_download: bool = False,
    num_samples: int = 1000,
    batch_size: int = 16,
    max_examples: int = 1000,
    max_macro_labels: int = 5000,
) -> dict[str, Any]:
    resolved_root = resolve_floorset_root(base_dir, floorset_root)
    probe = probe_training_corpus(resolved_root) if resolved_root is not None else None
    inventory = probe_to_json(probe, root=resolved_root)
    loader_report: dict[str, Any] | None = None
    loader_error: dict[str, Any] | None = None
    should_try_loader = bool(auto_download or inventory.get("training_loader_ready"))
    if resolved_root is not None and should_try_loader:
        try:
            loader_report = validate_training_loader(
                resolved_root,
                auto_download=auto_download,
                batch_size=batch_size,
                num_samples=num_samples,
                max_examples=max_examples,
                max_macro_labels=max_macro_labels,
            )
            # Re-probe after possible download/extract.
            inventory = probe_to_json(probe_training_corpus(resolved_root))
        except Exception as exc:  # pragma: no cover - exercised by environment failures.
            loader_error = {"type": type(exc).__name__, "message": str(exc)}
            loader_report = {
                "loader_status": "error",
                "auto_download_enabled": auto_download,
                "error": loader_error,
            }

    decision = decide(inventory, loader_report)
    metrics = {
        "decision": decision,
        "training_loader_ready": bool(inventory.get("training_loader_ready")),
        "training_unpacked_worker_count": int(inventory.get("training_unpacked_worker_count", 0)),
        "validation_case_count": int(inventory.get("validation_case_count", 0)),
        "download_attempted": bool(auto_download),
        "loader_status": None if loader_report is None else loader_report.get("loader_status"),
        "sample_count": (
            0 if loader_report is None else loader_report.get("collected_example_count", 0)
        ),
        "macro_label_count": (
            0 if loader_report is None else loader_report.get("macro_label_count", 0)
        ),
        "loader_error": loader_error,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(
        output_dir / "step7data_inventory.json",
        {"schema": "step7data_inventory_v1", **inventory},
    )
    write_json(
        output_dir / "step7data_loader_smoke.json",
        {
            "schema": "step7data_loader_smoke_v1",
            "metrics": metrics,
            "loader_report": _strip_rows(loader_report),
        },
    )
    write_json(
        output_dir / "step7data_training_samples.json",
        {
            "schema": "step7data_training_samples_v1",
            "rows": [] if loader_report is None else loader_report.get("examples", []),
        },
    )
    write_json(
        output_dir / "step7data_macro_labels.json",
        {
            "schema": "step7data_macro_labels_v1",
            "rows": [] if loader_report is None else loader_report.get("macro_labels", []),
            "label_summary": _label_summary(
                [] if loader_report is None else loader_report.get("macro_labels", [])
            ),
        },
    )
    (output_dir / "step7data_decision.md").write_text(decision_markdown(metrics), encoding="utf-8")
    return {"decision": decision, "metrics": metrics, "inventory": inventory}


def _strip_rows(loader_report: dict[str, Any] | None) -> dict[str, Any] | None:
    if loader_report is None:
        return None
    return {
        key: value
        for key, value in loader_report.items()
        if key not in {"examples", "macro_labels"}
    }


def _label_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "closure_type_counts": dict(Counter(str(row.get("closure_type")) for row in rows)),
        "block_count_summary": _summary([int(row.get("block_count", 0)) for row in rows]),
        "sample_count_with_macro_label": len({row.get("sample_index") for row in rows}),
    }


def decision_markdown(metrics: dict[str, Any]) -> str:
    return f"""# Step7DATA FloorSet-Lite Training Corpus Loader Validation

Decision: `{metrics['decision']}`

## Key metrics

- training_loader_ready: {metrics['training_loader_ready']}
- training_unpacked_worker_count: {metrics['training_unpacked_worker_count']}
- validation_case_count: {metrics['validation_case_count']}
- download_attempted: {metrics['download_attempted']}
- loader_status: {metrics['loader_status']}
- sample_count: {metrics['sample_count']}
- macro_label_count: {metrics['macro_label_count']}
- loader_error: {metrics['loader_error']}

## Interpretation

Step7DATA validates the official FloorSet `get_training_dataloader` path before
any further Step7ML training claims.  Labels are accepted only from the official
FloorSet-Lite training `fp_sol` tensor, whose contract is `[w, h, x, y]` for each
valid rectangular block.

If this decision is `download_required`, the next run should use `--auto-download`
with enough disk space.  If it is `training_loader_validated`, Step7ML-F can be
extended from sparse Step7 artifact labels to real FloorSet training examples.
"""
