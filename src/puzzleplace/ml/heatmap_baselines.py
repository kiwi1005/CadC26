"""Evaluation helpers for Step7L deterministic heatmap baselines."""

from __future__ import annotations

import math
from collections import defaultdict
from pathlib import Path
from typing import Any

from puzzleplace.ml.floorset_training_corpus import write_json
from puzzleplace.ml.heatmap_dataset import read_jsonl

KS = (1, 3, 5, 10)


def _cell_tuple(cell: dict[str, Any]) -> tuple[int, int]:
    return int(cell["row"]), int(cell["col"])


def _distance(a: tuple[int, int], b: tuple[int, int]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _eval_top_cells(rows: list[dict[str, Any]], key: str) -> dict[str, Any]:
    recall = {k: 0 for k in KS}
    min_dist = {k: 0.0 for k in KS}
    count = 0
    by_bucket: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        target = _cell_tuple(row["target_cell"])
        cells = [_cell_tuple(cell) for cell in row.get(key, [])]
        if not cells:
            continue
        count += 1
        bucket = "terminal" if row.get("has_terminal_demand") else "no_terminal"
        for k in KS:
            topk = cells[:k]
            hit = target in topk
            recall[k] += int(hit)
            dist = min(_distance(target, cell) for cell in topk)
            min_dist[k] += dist
            if k == 5:
                by_bucket[bucket].append(dist)
    denom = max(count, 1)
    return {
        "count": count,
        "recall_at_k": {str(k): recall[k] / denom for k in KS},
        "mean_min_grid_distance_at_k": {str(k): min_dist[k] / denom for k in KS},
        "mean_min_grid_distance_at_5_by_terminal_bucket": {
            bucket: sum(values) / max(len(values), 1) for bucket, values in by_bucket.items()
        },
    }


def evaluate_heatmap_baselines(examples_path: Path, out_path: Path | None = None) -> dict[str, Any]:
    rows = read_jsonl(examples_path)
    report = {
        "schema": "step7l_phase1_heatmap_metrics_v1",
        "example_count": len(rows),
        "source_examples": str(examples_path),
        "baselines": {
            "topology": _eval_top_cells(rows, "topology_top_cells"),
            "terminal": _eval_top_cells(rows, "terminal_top_cells"),
            "center": _eval_top_cells(rows, "center_top_cells"),
        },
        "decision": (
            "promote_heatmap_to_candidate_request_dry_run" if rows else "fix_heatmap_dataset"
        ),
    }
    if out_path is not None:
        write_json(out_path, report)
    return report
