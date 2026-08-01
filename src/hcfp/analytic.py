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
from hcfp.incumbent import IncumbentManager
from hcfp.projection import ProjectionResult, project_disjunctive
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


@dataclass(frozen=True)
class CandidateTelemetry:
    hard_feasible: Tensor
    raw_overlap: Tensor
    projected_overlap: Tensor
    overlap_components: Tensor
    projection_ok: Tensor
    projection_active_pairs: Tensor
    hpwl: Tensor
    bbox_area: Tensor
    soft_violation: Tensor
    projection_displacement: Tensor
    projection_failure_reasons: tuple[str, ...]


@dataclass(frozen=True)
class AnalyticResult:
    selected: Tensor
    raw_candidates: Tensor
    projected_candidates: Tensor
    telemetry: CandidateTelemetry
    energy_history: Tensor
    projection_status: str
    incumbent_snapshot: dict[str, object]


def select_device(requested: str | torch.device | None = None) -> torch.device:
    choice = str(requested or os.environ.get("HCFP_DEVICE", "auto"))
    if choice == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if choice.startswith("cuda") and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(choice)


def solve_case(case: FloorplanCase, config: AnalyticConfig | None = None) -> Tensor:
    """Return the best verified normalized candidate, never worse than fallback."""

    return _solve_candidates(case, config)[0]


def solve_case_with_telemetry(case: FloorplanCase, config: AnalyticConfig | None = None) -> AnalyticResult:
    """Return best candidate plus per-candidate telemetry after projection."""

    best, cpu_case, candidates, projection, energy_history, incumbent = _solve_candidates(case, config)
    return AnalyticResult(
        selected=best,
        raw_candidates=candidates.detach(),
        projected_candidates=projection.xywh.detach(),
        telemetry=_telemetry(cpu_case, candidates.detach(), projection),
        energy_history=energy_history.detach(),
        projection_status=projection.status,
        incumbent_snapshot=incumbent,
    )


def _solve_candidates(
    case: FloorplanCase,
    config: AnalyticConfig | None,
) -> tuple[Tensor, FloorplanCase, Tensor, ProjectionResult, Tensor, dict[str, object]]:
    cfg = config or AnalyticConfig()
    cpu_case = case.to(device="cpu", dtype=torch.float32)
    fallback = safe_shelf(cpu_case).to(dtype=torch.float32)
    manager = IncumbentManager(cpu_case, fallback)

    result = relax(case, cfg.dynamics, initial_xywh=fallback.to(case.area.device))
    candidates = torch.cat(
        (fallback.to(case.area.device).unsqueeze(0), result.initial_boxes, result.boxes),
        dim=0,
    )
    projection = project_disjunctive(
        candidates,
        problem=case,
        iterations=cfg.projection_iterations,
        beam=cfg.direction_beam,
    )
    projected = projection.xywh

    ok_mask = projection.ok_mask.detach().to(device="cpu", dtype=torch.bool).reshape(-1)
    for idx, candidate in enumerate(projected.detach().to(device="cpu", dtype=torch.float32)):
        manager.consider(candidate, source=f"candidate_{idx}", fast_feasible=bool(ok_mask[idx]))
    return manager.best_exact.xywh, cpu_case, candidates, projection, result.state.energy_history, manager.snapshot()


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


def _overlap_sum(boxes: Tensor) -> Tensor:
    from hcfp.geometry import overlap_area_matrix

    overlap = overlap_area_matrix(boxes)
    return torch.triu(overlap, diagonal=1).sum(dim=(-2, -1))


def _telemetry(case: FloorplanCase, raw: Tensor, projection: ProjectionResult) -> CandidateTelemetry:
    projected = projection.xywh.detach()
    cpu_projected = projected.detach().to(device="cpu", dtype=torch.float32)
    hard = torch.tensor([verify_feasible(case, candidate) for candidate in cpu_projected], dtype=torch.bool)
    soft = torch.tensor([soft_violation_normalized(case, candidate).total for candidate in cpu_projected], dtype=torch.float32)
    projected_cpu_case = case.to(device=projected.device, dtype=torch.float32)
    centers = centers_from_xywh(projected)
    displacement = torch.linalg.vector_norm(projected[..., :2] - raw[..., :2], dim=-1).sum(dim=1)
    return CandidateTelemetry(
        hard_feasible=hard.to(device=projected.device),
        raw_overlap=_overlap_sum(raw),
        projected_overlap=_overlap_sum(projected),
        overlap_components=_overlap_components(projected),
        projection_ok=projection.ok_mask.detach(),
        projection_active_pairs=projection.active_pair_count.detach(),
        hpwl=hpwl_tensor(projected_cpu_case, centers),
        bbox_area=bbox_area_tensor(projected),
        soft_violation=soft.to(device=projected.device),
        projection_displacement=displacement,
        projection_failure_reasons=projection.failure_reasons,
    )


def _overlap_components(boxes: Tensor) -> Tensor:
    from hcfp.geometry import overlap_area_matrix

    adjacency = overlap_area_matrix(boxes).detach().to(device="cpu") > 0.0
    counts = []
    for graph in adjacency:
        remaining = set(torch.nonzero(graph.any(dim=1), as_tuple=False).reshape(-1).tolist())
        components = 0
        while remaining:
            components += 1
            stack = [remaining.pop()]
            while stack:
                node = stack.pop()
                neighbors = set(torch.nonzero(graph[node], as_tuple=False).reshape(-1).tolist()) & remaining
                remaining -= neighbors
                stack.extend(neighbors)
        counts.append(components)
    return torch.tensor(counts, dtype=torch.long, device=boxes.device)


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
