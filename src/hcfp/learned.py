"""Checkpoint-gated learned initializer with an exact-safe analytic tail."""

from __future__ import annotations

from dataclasses import dataclass, fields, replace
import os
from pathlib import Path
from typing import Any

import torch

from hcfp.analytic import (
    AnalyticResult,
    AnalyticConfig,
    CandidateTelemetry,
    select_device,
    solve as solve_analytic,
    solve_case_with_telemetry,
    solve_case_from_population_with_telemetry,
    to_official_placements,
)
from hcfp.case import FloorplanCase, from_official
from hcfp.candidates import candidate_features
from hcfp.checkpoint import RUNTIME_NORMALIZATION, load_checkpoint
from hcfp.dynamics import initialize_population
from hcfp.fallback import safe_fallback, safe_shelf
from hcfp.geometry import xywh_from_state
from hcfp.verify import bbox_area, soft_violation_normalized, total_hpwl, verify_feasible


Tensor = torch.Tensor


@dataclass(frozen=True)
class LearnedResult:
    selected: Tensor
    used_checkpoint: bool
    checkpoint_hash: str | None
    failure_reason: str | None
    flow_steps: int = 0
    candidate_count: int = 0


@dataclass(frozen=True)
class LearnedAnalysis:
    result: LearnedResult
    analytic: AnalyticResult


@dataclass(frozen=True)
class LearnedConfig:
    analytic: AnalyticConfig = AnalyticConfig()
    flow_steps: int = 6
    flow_fraction: float = 0.50
    flow_noise_scale: float = 1.0
    max_position_residual: float = 0.50
    max_aspect_residual: float = 1.0
    tail_topk: int | None = None
    seed: int | None = None

    def __post_init__(self) -> None:
        if self.flow_steps < 0:
            raise ValueError("flow_steps must be non-negative")
        if not 0.0 <= self.flow_fraction <= 1.0:
            raise ValueError("flow_fraction must be in [0, 1]")
        if self.flow_noise_scale < 0.0:
            raise ValueError("flow_noise_scale must be non-negative")
        if self.max_position_residual <= 0.0 or self.max_aspect_residual <= 0.0:
            raise ValueError("flow residual limits must be positive")
        if self.tail_topk is not None and self.tail_topk <= 0:
            raise ValueError("tail_topk must be positive")


def solve_case_with_checkpoint(
    case: FloorplanCase,
    checkpoint: str | Path,
    config: AnalyticConfig | LearnedConfig | None = None,
) -> LearnedResult:
    """Use learned population residuals only behind hash and verifier gates."""

    return analyze_case_with_checkpoint(case, checkpoint, config).result


def analyze_case_with_checkpoint(
    case: FloorplanCase,
    checkpoint: str | Path,
    config: AnalyticConfig | LearnedConfig | None = None,
) -> LearnedAnalysis:
    """Return the learned result plus per-candidate exact-tail telemetry."""

    learned_cfg = _learned_config(config)
    cfg = learned_cfg.analytic
    try:
        model, metadata = load_checkpoint(
            checkpoint,
            expected_normalization=RUNTIME_NORMALIZATION,
            map_location="cpu",
        )
        model = model.to(device=case.area.device).eval()
        learned_population = _learned_population(
            case,
            model,
            learned_cfg,
            seed=(
                int(str(metadata["state_hash"])[:8], 16)
                if learned_cfg.seed is None
                else learned_cfg.seed
            ),
        )
        analytic_analysis = solve_case_with_telemetry(case, cfg)
        learned_tail_cfg = replace(
            cfg,
            dynamics=replace(cfg.dynamics, population=int(learned_population.shape[0])),
        )
        learned_analysis = solve_case_from_population_with_telemetry(
            case,
            learned_population,
            learned_tail_cfg,
        )
        analysis = _merge_tail_analyses(case, analytic_analysis, learned_analysis)
        candidate_count = cfg.dynamics.population + int(learned_population.shape[0])
        return LearnedAnalysis(
            LearnedResult(
                analysis.selected,
                True,
                str(metadata["state_hash"]),
                None,
                learned_cfg.flow_steps,
                candidate_count,
            ),
            analysis,
        )
    except Exception as exc:
        analysis = solve_case_with_telemetry(case, cfg)
        return LearnedAnalysis(
            LearnedResult(
                analysis.selected,
                False,
                None,
                f"{type(exc).__name__}: {exc}",
            ),
            analysis,
        )


def solve(
    source: Any,
    *,
    checkpoint: str | Path,
    config: AnalyticConfig | LearnedConfig | None = None,
    device: str | torch.device | None = None,
    require_checkpoint: bool = False,
) -> list[tuple[float, float, float, float]]:
    """Official-contract learned lane; any checkpoint failure uses analytic P1."""

    try:
        selected_device = select_device(device)
        case = from_official(
            int(_field(source, "block_count")),
            _field(source, "area_targets"),
            _field(source, "b2b_connectivity"),
            _field(source, "p2b_connectivity"),
            _field(source, "pins_pos"),
            _field(source, "constraints"),
            _field(source, "target_positions"),
            device=selected_device,
        )
        analysis = analyze_case_with_checkpoint(case, checkpoint, config)
        result = analysis.result
        if require_checkpoint and not result.used_checkpoint:
            raise RuntimeError(result.failure_reason or "checkpoint was not used")
        return select_official_from_analysis(
            source,
            case,
            analysis,
            config=config,
            device=device,
        )
    except Exception:
        if require_checkpoint:
            raise
        return solve_analytic(source, _learned_config(config).analytic, device=device)


def select_official_from_analysis(
    source: Any,
    case: FloorplanCase,
    analysis: LearnedAnalysis,
    *,
    config: AnalyticConfig | LearnedConfig | None = None,
    device: str | torch.device | None = None,
) -> list[tuple[float, float, float, float]]:
    """Apply the exact runtime raw-selection chain to an existing analysis."""

    placements = to_official_placements(source, case, analysis.result.selected)
    if verify_feasible(source, placements):
        return _raw_analytic_pareto_guard(
            source,
            case,
            analysis,
            placements,
        )
    telemetry = analysis.analytic.telemetry
    candidates = analysis.analytic.projected_candidates.detach().to(device="cpu", dtype=torch.float32)
    hard_feasible = telemetry.hard_feasible.detach().to(device="cpu", dtype=torch.bool)
    soft_violation = telemetry.soft_violation.detach().to(device="cpu", dtype=torch.float32)
    quality = (telemetry.bbox_area + 0.05 * telemetry.hpwl).detach().to(device="cpu", dtype=torch.float32)
    order = sorted(
        (index for index in range(len(candidates)) if bool(hard_feasible[index])),
        key=lambda index: (float(soft_violation[index]), float(quality[index]), index),
    )
    for index in order:
        candidate = to_official_placements(source, case, candidates[index])
        if verify_feasible(source, candidate):
            if index == 0:
                break
            return candidate
    analytic = solve_analytic(source, _learned_config(config).analytic, device=device)
    if verify_feasible(source, analytic):
        return analytic
    fallback = safe_fallback(source)
    return [tuple(float(value) for value in row) for row in fallback]


def _field(source: Any, name: str) -> Any:
    if isinstance(source, dict):
        return source.get(name)
    return getattr(source, name)


def _learned_config(config: AnalyticConfig | LearnedConfig | None) -> LearnedConfig:
    if config is None:
        tail = os.environ.get("HCFP_TAIL_TOPK")
        seed = os.environ.get("HCFP_FLOW_SEED")
        return LearnedConfig(
            flow_steps=int(os.environ.get("HCFP_FLOW_STEPS", "6")),
            tail_topk=int(tail) if tail else None,
            seed=int(seed) if seed is not None else None,
        )
    if isinstance(config, AnalyticConfig):
        return LearnedConfig(analytic=config)
    return config


def _merge_tail_analyses(
    case: FloorplanCase,
    analytic: AnalyticResult,
    learned: AnalyticResult,
) -> AnalyticResult:
    analytic_count = (analytic.projected_candidates.shape[0] - 1) // 2
    learned_count = (learned.projected_candidates.shape[0] - 1) // 2

    def merge_tensor(first: Tensor, second: Tensor) -> Tensor:
        return torch.cat(
            (
                first[:1],
                first[1 : 1 + analytic_count],
                second[1 : 1 + learned_count],
                first[1 + analytic_count :],
                second[1 + learned_count :],
            ),
            dim=0,
        )

    def merge_tuple(first: tuple[str, ...], second: tuple[str, ...]) -> tuple[str, ...]:
        return (
            first[:1]
            + first[1 : 1 + analytic_count]
            + second[1 : 1 + learned_count]
            + first[1 + analytic_count :]
            + second[1 + learned_count :]
        )

    projected = merge_tensor(analytic.projected_candidates, learned.projected_candidates)
    telemetry_values = {}
    for field in fields(CandidateTelemetry):
        first = getattr(analytic.telemetry, field.name)
        second = getattr(learned.telemetry, field.name)
        telemetry_values[field.name] = (
            merge_tuple(first, second)
            if field.name == "projection_failure_reasons"
            else merge_tensor(first, second)
        )
    telemetry = CandidateTelemetry(**telemetry_values)
    ok_mask = telemetry.projection_ok.detach().to(device="cpu", dtype=torch.bool)
    hard_mask = telemetry.hard_feasible.detach().to(device="cpu", dtype=torch.bool)
    exact_mask = hard_mask & ok_mask
    exact_mask[0] = True
    area = telemetry.bbox_area.detach().to(device="cpu", dtype=torch.float64)
    hpwl = telemetry.hpwl.detach().to(device="cpu", dtype=torch.float64)
    soft = telemetry.soft_violation.detach().to(device="cpu", dtype=torch.float64)

    def best_index(mask: Tensor) -> int | None:
        indices = torch.nonzero(mask, as_tuple=False).reshape(-1).tolist()
        return (
            min(
                indices,
                key=lambda index: (
                    float(soft[index]),
                    float(area[index]) + 0.05 * float(hpwl[index]),
                    index,
                ),
            )
            if indices
            else None
        )

    fast_index = best_index(ok_mask)
    exact_index = best_index(exact_mask)
    if exact_index is None:
        raise RuntimeError("merged tail lost the safe fallback")
    rejections = {
        reason: count
        for reason, count in (
            ("fast_infeasible", int((~ok_mask).sum().item())),
            ("exact_infeasible", int((ok_mask & ~hard_mask).sum().item())),
        )
        if count
    }
    snapshot = {
        "safe_source": "fallback",
        "fast_source": f"candidate_{fast_index}" if fast_index is not None else None,
        "exact_source": "fallback" if exact_index == 0 else f"candidate_{exact_index}",
        "analytic_exact_source": _merged_analytic_source(
            analytic.incumbent_snapshot.get("exact_source"),
            analytic_count,
            learned_count,
        ),
        "analytic_fast_source": _merged_analytic_source(
            analytic.incumbent_snapshot.get("fast_source"),
            analytic_count,
            learned_count,
        ),
        "rejections": rejections,
    }
    status = analytic.projection_status
    if learned.projection_status != status:
        status = f"analytic={status};learned={learned.projection_status}"
    return AnalyticResult(
        selected=(
            analytic.selected
            if exact_index == 0
            else projected[exact_index].detach().to(device="cpu", dtype=torch.float32)
        ),
        raw_candidates=merge_tensor(analytic.raw_candidates, learned.raw_candidates),
        projected_candidates=projected,
        telemetry=telemetry,
        energy_history=torch.cat((analytic.energy_history, learned.energy_history), dim=0),
        projection_status=status,
        incumbent_snapshot=snapshot,
    )


def _raw_analytic_pareto_guard(
    source: Any,
    case: FloorplanCase,
    analysis: LearnedAnalysis,
    current: list[tuple[float, float, float, float]],
) -> list[tuple[float, float, float, float]]:
    """Admit only raw-feasible analytic candidates that dominate the incumbent."""

    snapshot = getattr(analysis.analytic, "incumbent_snapshot", {})
    selected_index = _candidate_index(snapshot.get("exact_source"))
    projected = analysis.analytic.projected_candidates
    analytic_indices = tuple(
        dict.fromkeys(
            index
            for index in (
                _candidate_index(snapshot.get("analytic_exact_source")),
                _candidate_index(snapshot.get("analytic_fast_source")),
            )
            if index is not None and index != selected_index
        )
    )
    if selected_index is None or not 0 <= selected_index < projected.shape[0]:
        return current
    analytic_indices = tuple(index for index in analytic_indices if 0 <= index < projected.shape[0])
    if not analytic_indices:
        return current
    try:
        current_metrics = _raw_quality(source, case, current)
    except (TypeError, ValueError):
        return current

    admitted = []
    for index in analytic_indices:
        protected = analysis.analytic.raw_candidates[0] if index == 0 else projected[index]
        try:
            placement = to_official_placements(
                source,
                case,
                protected.detach().to(device="cpu", dtype=torch.float32),
            )
            if not verify_feasible(source, placement):
                continue
            candidate_metrics = _raw_quality(source, case, placement)
        except (TypeError, ValueError):
            continue
        if _dominates(candidate_metrics, current_metrics):
            admitted.append((candidate_metrics, index, placement))
    return (
        min(
            admitted,
            key=lambda item: (item[0][0], item[0][1] + 0.05 * item[0][2], item[1]),
        )[2]
        if admitted
        else current
    )


def _raw_quality(
    source: Any,
    case: FloorplanCase,
    placement: list[tuple[float, float, float, float]],
) -> tuple[float, float, float]:
    soft_case = {
        "normalized": False,
        "boundary_bits": case.boundary_bits.detach().to(device="cpu"),
        "group_membership": case.group_membership.detach().to(device="cpu"),
        "mib_membership": case.mib_membership.detach().to(device="cpu"),
    }
    return (
        soft_violation_normalized(soft_case, placement).total,
        bbox_area(placement),
        total_hpwl(source, placement),
    )


def _candidate_index(source: object) -> int | None:
    value = str(source)
    if value == "fallback":
        return 0
    if not value.startswith("candidate_"):
        return None
    try:
        return int(value.removeprefix("candidate_"))
    except ValueError:
        return None


def _merged_analytic_source(source: object, analytic_count: int, learned_count: int) -> str | None:
    index = _candidate_index(source)
    if index is None:
        return None
    if index == 0:
        return "fallback"
    if index <= analytic_count:
        return f"candidate_{index}"
    if index <= 2 * analytic_count:
        merged = 1 + analytic_count + learned_count + index - analytic_count - 1
        return f"candidate_{merged}"
    return None


def _dominates(candidate: tuple[float, ...], incumbent: tuple[float, ...]) -> bool:
    tolerance = tuple(1.0e-6 * max(1.0, abs(value)) for value in incumbent)
    return all(value <= base for value, base in zip(candidate, incumbent)) and any(
        value < base - tol for value, base, tol in zip(candidate, incumbent, tolerance)
    )


def _learned_population(
    case: FloorplanCase,
    model,
    config: LearnedConfig,
    *,
    seed: int,
) -> Tensor:
    population = config.analytic.dynamics.population
    fallback = safe_shelf(case).to(device=case.area.device, dtype=torch.float32)
    base = initialize_population(case, config.analytic.dynamics, fallback)
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    noise = torch.randn((population, case.n, 3), generator=generator, dtype=torch.float32)
    residual = noise.to(device=case.area.device) * config.flow_noise_scale
    residual[:, case.preplaced_mask, :2] = 0.0
    residual[:, case.fixed_mask | case.preplaced_mask, 2] = 0.0

    with torch.inference_mode():
        output = model(case, population=population)
        flow_count = round(population * config.flow_fraction)
        if flow_count and config.flow_steps:
            for step in range(config.flow_steps):
                time = (step + 0.5) / config.flow_steps
                device_type = "cuda" if case.area.is_cuda else "cpu"
                with torch.autocast(
                    device_type=device_type,
                    dtype=torch.bfloat16,
                    enabled=model.config.compute_dtype == "bfloat16",
                ):
                    velocity = model.flow(case, output.embedding, population, residual, time)
                residual = residual + velocity.float() / config.flow_steps
                residual[..., :2].clamp_(
                    -config.max_position_residual,
                    config.max_position_residual,
                )
                residual[..., 2].clamp_(
                    -config.max_aspect_residual,
                    config.max_aspect_residual,
                )
                residual[:, case.preplaced_mask, :2] = 0.0
                residual[:, case.fixed_mask | case.preplaced_mask, 2] = 0.0
            output.center_residual[-flow_count:] = residual[-flow_count:, :, :2]
            output.log_aspect_residual[-flow_count:] = residual[-flow_count:, :, 2]

    center = base.center + output.center_residual
    log_aspect = (base.log_aspect + output.log_aspect_residual).clamp(-4.0, 4.0)
    learned_boxes = xywh_from_state(case, center, log_aspect)
    if config.tail_topk is not None and config.tail_topk < population:
        features = candidate_features(case, learned_boxes, fallback)
        with torch.inference_mode():
            scores = model.ranker(output.embedding, population, features)
        keep = torch.argsort(scores, stable=True)[: config.tail_topk]
        learned_boxes = learned_boxes[keep]
    return learned_boxes
