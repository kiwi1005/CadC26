"""Step7L heatmap dataset construction from FloorSet training labels."""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import torch

from puzzleplace.ml.floorset_training_corpus import (
    _auto_yes_download,
    _import_official_evaluator,
    resolve_floorset_root,
)
from puzzleplace.ml.topology_maps import (
    build_block_heatmaps,
    frame_from_fp_sol,
    target_cell_for_fp_row,
    top_cells,
    valid_block_mask,
    with_grid_size,
)

TOP_K = 16


def _safe_int(value: torch.Tensor) -> int:
    return int(float(value.item()))


def examples_from_training_batch(
    batch: tuple[torch.Tensor, ...],
    *,
    sample_offset: int = 0,
    grid_size: int = 16,
) -> list[dict[str, Any]]:
    area, _b2b, p2b, pins, constraints, _tree_sol, fp_sol, _metrics = batch
    examples: list[dict[str, Any]] = []
    for row_idx in range(int(area.shape[0])):
        mask = valid_block_mask(area[row_idx], fp_sol[row_idx])
        if int(mask.sum().item()) <= 0:
            continue
        frame = with_grid_size(frame_from_fp_sol(fp_sol[row_idx], mask), grid_size)
        valid_ids = torch.nonzero(mask, as_tuple=False).flatten().tolist()
        sample_id = f"train_{sample_offset + row_idx:07d}"
        for block_id in valid_ids:
            block_idx = int(block_id)
            target_row, target_col = target_cell_for_fp_row(fp_sol[row_idx, block_idx], frame)
            boundary_code = (
                _safe_int(constraints[row_idx, block_idx, 4])
                if constraints.shape[-1] > 4
                else 0
            )
            maps = build_block_heatmaps(
                block_id=block_idx,
                boundary_code=boundary_code,
                p2b_edges=p2b[row_idx],
                pins_pos=pins[row_idx],
                grid=frame,
            )
            examples.append(
                {
                    "schema": "step7l_phase1_heatmap_example_v1",
                    "source": "floorset_lite_training_fp_sol",
                    "sample_id": sample_id,
                    "sample_index": sample_offset + row_idx,
                    "block_id": block_idx,
                    "block_count": int(mask.sum().item()),
                    "grid": {"rows": grid_size, "cols": grid_size},
                    "frame": {"x": frame.x, "y": frame.y, "w": frame.w, "h": frame.h},
                    "target_cell": {"row": target_row, "col": target_col},
                    "boundary_code": boundary_code,
                    "has_terminal_demand": any(
                        int(edge[1].item()) == block_idx and float(edge[0].item()) >= 0
                        for edge in p2b[row_idx]
                    ),
                    "topology_top_cells": top_cells(maps["topology"], k=TOP_K),
                    "terminal_top_cells": top_cells(maps["terminal"], k=TOP_K),
                    "center_top_cells": top_cells(maps["center"], k=TOP_K),
                    "label_contract": "target_cell derived only from FloorSet training fp_sol",
                }
            )
    return examples


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
            count += 1
    return count


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def build_heatmap_dataset(
    base_dir: Path,
    out_path: Path,
    *,
    floorset_root: Path | None = None,
    train_samples: int = 256,
    grid_size: int = 16,
    batch_size: int = 32,
    auto_download: bool = False,
) -> dict[str, Any]:
    resolved = resolve_floorset_root(base_dir, floorset_root)
    if resolved is None:
        raise FileNotFoundError("Could not resolve external/FloorSet official checkout")
    evaluator = _import_official_evaluator(resolved)
    all_examples: list[dict[str, Any]] = []
    with _auto_yes_download(auto_download):
        loader = evaluator.get_training_dataloader(
            data_path=str(resolved),
            batch_size=batch_size,
            num_samples=train_samples,
            shuffle=False,
        )
        sample_offset = 0
        for batch in loader:
            all_examples.extend(
                examples_from_training_batch(
                    batch, sample_offset=sample_offset, grid_size=grid_size
                )
            )
            sample_offset += int(batch[0].shape[0])
            if sample_offset >= train_samples:
                break
    row_count = write_jsonl(out_path, all_examples)
    return {
        "schema": "step7l_phase1_heatmap_dataset_report_v1",
        "example_count": row_count,
        "sample_count": train_samples,
        "grid": grid_size,
        "out_path": str(out_path),
    }
