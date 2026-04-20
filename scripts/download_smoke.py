#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
FLOORSET_ROOT = ROOT / "external" / "FloorSet"
CONTEST_ROOT = FLOORSET_ROOT / "iccad2026contest"


def _ensure_import_paths() -> None:
    if not FLOORSET_ROOT.is_dir():
        raise FileNotFoundError(
            "Missing external/FloorSet checkout. Clone https://github.com/IntelLabs/FloorSet.git "
            "into external/FloorSet first."
        )
    for path in (CONTEST_ROOT, FLOORSET_ROOT):
        path_str = str(path)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)


def _auto_approve_downloads() -> None:
    import lite_dataset
    import lite_dataset_test

    lite_dataset.decide_download = lambda url: True
    lite_dataset_test.decide_download = lambda url: True



def _shape(value: Any) -> list[int] | None:
    shape = getattr(value, "shape", None)
    if shape is None:
        return None
    return [int(dim) for dim in shape]



def _count_valid_blocks(area_target: Any) -> int | None:
    if not hasattr(area_target, "ne"):
        return None
    valid = area_target.ne(-1)
    if valid.ndim > 1:
        valid = valid[0]
    return int(valid.sum().item())



def _print_json(title: str, payload: dict[str, Any]) -> None:
    print(f"\n=== {title} ===")
    print(json.dumps(payload, indent=2, sort_keys=True))



def main() -> None:
    _ensure_import_paths()
    _auto_approve_downloads()

    from iccad2026_evaluate import get_training_dataloader, get_validation_dataloader

    print(f"Repo root: {ROOT}")
    print(f"FloorSet root: {FLOORSET_ROOT}")
    print("Downloads are auto-approved for this smoke script.")

    validation_loader = get_validation_dataloader(data_path=str(FLOORSET_ROOT), batch_size=1)
    validation_inputs, validation_labels = next(iter(validation_loader))
    area_target, b2b_connectivity, p2b_connectivity, pins_pos, placement_constraints = validation_inputs
    polygons, metrics = validation_labels
    _print_json(
        "validation_first_batch",
        {
            "fields": [
                "area_target",
                "b2b_connectivity",
                "p2b_connectivity",
                "pins_pos",
                "placement_constraints",
                "polygons",
                "metrics",
            ],
            "block_count": _count_valid_blocks(area_target),
            "shapes": {
                "area_target": _shape(area_target),
                "b2b_connectivity": _shape(b2b_connectivity),
                "p2b_connectivity": _shape(p2b_connectivity),
                "pins_pos": _shape(pins_pos),
                "placement_constraints": _shape(placement_constraints),
                "polygons": _shape(polygons),
                "metrics": _shape(metrics),
            },
        },
    )

    training_loader = get_training_dataloader(
        data_path=str(FLOORSET_ROOT),
        batch_size=1,
        num_samples=10,
        shuffle=False,
    )
    training_batch = next(iter(training_loader))
    (
        train_area_target,
        train_b2b_connectivity,
        train_p2b_connectivity,
        train_pins_pos,
        train_constraints,
        tree_sol,
        fp_sol,
        train_metrics,
    ) = training_batch
    _print_json(
        "training_first_batch",
        {
            "requested_num_samples": 10,
            "fields": [
                "area_target",
                "b2b_connectivity",
                "p2b_connectivity",
                "pins_pos",
                "placement_constraints",
                "tree_sol",
                "fp_sol",
                "metrics",
            ],
            "block_count": _count_valid_blocks(train_area_target),
            "loader_length": len(training_loader),
            "shapes": {
                "area_target": _shape(train_area_target),
                "b2b_connectivity": _shape(train_b2b_connectivity),
                "p2b_connectivity": _shape(train_p2b_connectivity),
                "pins_pos": _shape(train_pins_pos),
                "placement_constraints": _shape(train_constraints),
                "tree_sol": _shape(tree_sol),
                "fp_sol": _shape(fp_sol),
                "metrics": _shape(train_metrics),
            },
        },
    )


if __name__ == "__main__":
    main()
