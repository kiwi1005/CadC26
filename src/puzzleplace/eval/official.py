from __future__ import annotations

import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch

from puzzleplace.data.floorset_adapter import adapt_validation_batch
from puzzleplace.data.schema import FloorSetCase
from puzzleplace.geometry.legality import positions_from_case_targets, summarize_hard_legality

ROOT = Path(__file__).resolve().parents[3]
FLOORSET_ROOT = ROOT / "external" / "FloorSet"
CONTEST_ROOT = FLOORSET_ROOT / "iccad2026contest"

for path in (CONTEST_ROOT, FLOORSET_ROOT):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from iccad2026_evaluate import evaluate_solution  # noqa: E402


def _boxes_from_polygons(polygons: torch.Tensor) -> list[tuple[float, float, float, float]]:
    case = adapt_validation_batch(
        ((torch.empty(1, 0),) * 5, (polygons.unsqueeze(0), torch.empty(1, 8)))
    )
    return positions_from_case_targets(case)


def extract_validation_baseline_metrics(case: FloorSetCase) -> dict[str, float]:
    if case.target_positions is None:
        raise ValueError("validation baseline extraction requires target_positions")
    positions = positions_from_case_targets(case)
    bbox_x0 = min(x for x, _, _, _ in positions)
    bbox_y0 = min(y for _, y, _, _ in positions)
    bbox_x1 = max(x + w for x, _, w, _ in positions)
    bbox_y1 = max(y + h for _, y, _, h in positions)
    bbox_area = float((bbox_x1 - bbox_x0) * (bbox_y1 - bbox_y0))
    hpwl_baseline = 0.0
    if case.metrics is not None and case.metrics.numel() >= 8:
        hpwl_baseline = float(case.metrics[-2].item() + case.metrics[-1].item())
        if case.metrics[0].item() > 0:
            bbox_area = float(case.metrics[0].item())
    return {"area_baseline": bbox_area, "hpwl_baseline": hpwl_baseline}


def evaluate_positions(
    case: FloorSetCase,
    positions: list[tuple[float, float, float, float]],
    *,
    runtime: float = 1.0,
    median_runtime: float = 1.0,
) -> dict[str, Any]:
    baseline = extract_validation_baseline_metrics(case)
    metrics = evaluate_solution(
        {"positions": positions, "runtime": runtime},
        baseline,
        case.constraints,
        case.b2b_edges,
        case.p2b_edges,
        case.pins_pos,
        case.area_targets,
        target_positions=(
            positions_from_case_targets(case) if case.target_positions is not None else None
        ),
        median_runtime=median_runtime,
    )
    legality = summarize_hard_legality(case, positions)
    official = asdict(metrics)
    quality = {
        "quality_cost_runtime1": float(metrics.cost),
        "cost": float(metrics.cost),
        "official_cost_raw": float(metrics.cost),
        "HPWLgap": float(metrics.hpwl_gap),
        "Areagap_bbox": float(metrics.area_gap),
        "Violationsrelative": float(metrics.violations_relative),
        "feasible": bool(metrics.is_feasible),
    }
    return {
        "official": official,
        "legality": asdict(legality),
        "baseline": baseline,
        "quality": quality,
    }


class OfficialEvaluatorWrapper:
    def evaluate_validation_batch(
        self,
        batch: tuple[list[torch.Tensor], list[torch.Tensor]]
        | tuple[tuple[torch.Tensor, ...], tuple[torch.Tensor, ...]],
        *,
        case_id: str | int = "validation-case",
        runtime: float = 1.0,
    ) -> dict[str, Any]:
        case = adapt_validation_batch(batch, case_id=case_id)
        if case.target_positions is None:
            raise ValueError("validation batch must provide target positions")
        positions = positions_from_case_targets(case)
        return evaluate_positions(case, positions, runtime=runtime)
