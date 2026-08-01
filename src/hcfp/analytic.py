"""Safe analytic HCFP solver: population dynamics followed by BDP."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import torch

from hcfp.case import FloorplanCase, from_official
from hcfp.dynamics import DynamicsConfig, relax
from hcfp.fallback import safe_shelf
from hcfp.geometry import bbox_area_tensor, centers_from_xywh, denormalize_xywh, hpwl_tensor
from hcfp.projection import project_disjunctive
from hcfp.verify import soft_violation_normalized, verify_feasible


Tensor = torch.Tensor


@dataclass(frozen=True)
class AnalyticConfig:
    dynamics: DynamicsConfig = DynamicsConfig()
    projection_iterations: int = 24
    direction_beam: int = 4

    def __post_init__(self) -> None:
        if self.projection_iterations <= 0:
            raise ValueError("projection_iterations must be positive")
        if self.direction_beam <= 0:
            raise ValueError("direction_beam must be positive")


def select_device(requested: str | torch.device | None = None) -> torch.device:
    choice = str(requested or os.environ.get("HCFP_DEVICE", "auto"))
    if choice == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if choice.startswith("cuda") and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(choice)


def solve_case(case: FloorplanCase, config: AnalyticConfig | None = None) -> Tensor:
    """Return the best verified normalized candidate, never worse than fallback."""

    cfg = config or AnalyticConfig()
    cpu_case = case.to(device="cpu", dtype=torch.float32)
    fallback = safe_shelf(cpu_case).to(dtype=torch.float32)
    best = fallback
    best_key = _candidate_key(cpu_case, fallback)

    result = relax(case, cfg.dynamics, initial_xywh=fallback.to(case.area.device))
    candidates = torch.cat((fallback.to(case.area.device).unsqueeze(0), result.boxes), dim=0)
    projected = project_disjunctive(
        candidates,
        problem=case,
        iterations=cfg.projection_iterations,
        beam=cfg.direction_beam,
    ).xywh

    for candidate in projected.detach().to(device="cpu", dtype=torch.float32):
        if not verify_feasible(cpu_case, candidate):
            continue
        key = _candidate_key(cpu_case, candidate)
        if key < best_key:
            best = candidate
            best_key = key
    return best


def solve(
    case: Any,
    config: AnalyticConfig | None = None,
    *,
    device: str | torch.device | None = None,
) -> list[tuple[float, float, float, float]]:
    """Solve a runtime ``SolveCase`` and return raw official coordinates."""

    selected_device = select_device(device)
    normalized = from_official(
        int(_field(case, "block_count")),
        _field(case, "area_targets"),
        _field(case, "b2b_connectivity"),
        _field(case, "p2b_connectivity"),
        _field(case, "pins_pos"),
        _field(case, "constraints"),
        _field(case, "target_positions"),
        device=selected_device,
    )
    normalized_solution = solve_case(normalized, config)
    raw = denormalize_xywh(normalized.to(device="cpu"), normalized_solution).to(torch.float64)
    _copy_raw_hard_targets(case, normalized.to(device="cpu"), raw)
    return [tuple(float(value) for value in row) for row in raw.tolist()]


def _candidate_key(case: FloorplanCase, boxes: Tensor) -> tuple[float, float]:
    soft = soft_violation_normalized(case, boxes).total
    quality = float(bbox_area_tensor(boxes)) + 0.05 * float(hpwl_tensor(case, centers_from_xywh(boxes)))
    return soft, quality


def _copy_raw_hard_targets(case: Any, normalized: FloorplanCase, raw: Tensor) -> None:
    target_source = _field(case, "target_positions")
    if target_source is None:
        return
    target = torch.as_tensor(target_source, dtype=raw.dtype, device="cpu")[: normalized.n]
    preplaced = normalized.preplaced_mask
    hard_shape = normalized.fixed_mask | preplaced
    raw[preplaced] = target[preplaced]
    raw[hard_shape, 2:4] = target[hard_shape, 2:4]


def _field(source: Any, name: str) -> Any:
    if isinstance(source, dict):
        return source.get(name)
    return getattr(source, name)
