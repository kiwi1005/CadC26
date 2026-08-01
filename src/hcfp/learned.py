"""Checkpoint-gated learned initializer with an exact-safe analytic tail."""

from __future__ import annotations

from dataclasses import dataclass, replace
import os
from pathlib import Path
from typing import Any

import torch

from hcfp.analytic import (
    AnalyticResult,
    AnalyticConfig,
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
from hcfp.verify import verify_feasible


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
        population = _learned_population(
            case,
            model,
            learned_cfg,
            seed=int(str(metadata["state_hash"])[:8], 16),
        )
        if population.shape[0] != cfg.dynamics.population:
            cfg = replace(cfg, dynamics=replace(cfg.dynamics, population=int(population.shape[0])))
        analysis = solve_case_from_population_with_telemetry(case, population, cfg)
        return LearnedAnalysis(
            LearnedResult(
                analysis.selected,
                True,
                str(metadata["state_hash"]),
                None,
                learned_cfg.flow_steps,
                int(population.shape[0]),
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
        result = solve_case_with_checkpoint(case, checkpoint, config)
        if require_checkpoint and not result.used_checkpoint:
            raise RuntimeError(result.failure_reason or "checkpoint was not used")
        placements = to_official_placements(source, case, result.selected)
        if not verify_feasible(source, placements):
            analytic = solve_analytic(source, _learned_config(config).analytic, device=device)
            if verify_feasible(source, analytic):
                return analytic
            fallback = safe_fallback(source)
            return [tuple(float(value) for value in row) for row in fallback]
        return placements
    except Exception:
        if require_checkpoint:
            raise
        return solve_analytic(source, _learned_config(config).analytic, device=device)


def _field(source: Any, name: str) -> Any:
    if isinstance(source, dict):
        return source.get(name)
    return getattr(source, name)


def _learned_config(config: AnalyticConfig | LearnedConfig | None) -> LearnedConfig:
    if config is None:
        tail = os.environ.get("HCFP_TAIL_TOPK")
        return LearnedConfig(
            flow_steps=int(os.environ.get("HCFP_FLOW_STEPS", "6")),
            tail_topk=int(tail) if tail else None,
        )
    if isinstance(config, AnalyticConfig):
        return LearnedConfig(analytic=config)
    return config


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
    analytic_boxes = xywh_from_state(case, base.center, base.log_aspect)
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
    return torch.cat((analytic_boxes, learned_boxes), dim=0)
