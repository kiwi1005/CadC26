"""Checkpoint-gated learned initializer with an exact-safe analytic tail."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from hcfp.analytic import (
    AnalyticConfig,
    select_device,
    solve as solve_analytic,
    solve_case,
    solve_case_from_population,
    to_official_placements,
)
from hcfp.case import FloorplanCase, from_official
from hcfp.checkpoint import RUNTIME_NORMALIZATION, load_checkpoint
from hcfp.dynamics import initialize_population
from hcfp.fallback import safe_shelf
from hcfp.geometry import xywh_from_state


Tensor = torch.Tensor


@dataclass(frozen=True)
class LearnedResult:
    selected: Tensor
    used_checkpoint: bool
    checkpoint_hash: str | None
    failure_reason: str | None


def solve_case_with_checkpoint(
    case: FloorplanCase,
    checkpoint: str | Path,
    config: AnalyticConfig | None = None,
) -> LearnedResult:
    """Use learned population residuals only behind hash and verifier gates."""

    cfg = config or AnalyticConfig()
    try:
        model, metadata = load_checkpoint(
            checkpoint,
            expected_normalization=RUNTIME_NORMALIZATION,
            map_location="cpu",
        )
        model = model.to(device=case.area.device).eval()
        fallback = safe_shelf(case).to(device=case.area.device, dtype=torch.float32)
        state = initialize_population(case, cfg.dynamics, fallback)
        with torch.inference_mode():
            output = model(case, population=cfg.dynamics.population)
        center = state.center + output.center_residual
        log_aspect = (state.log_aspect + output.log_aspect_residual).clamp(-4.0, 4.0)
        population = xywh_from_state(case, center, log_aspect)
        selected = solve_case_from_population(case, population, cfg)
        return LearnedResult(selected, True, str(metadata["state_hash"]), None)
    except Exception as exc:
        selected = solve_case(case, cfg)
        return LearnedResult(selected, False, None, f"{type(exc).__name__}: {exc}")


def solve(
    source: Any,
    *,
    checkpoint: str | Path,
    config: AnalyticConfig | None = None,
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
        return to_official_placements(source, case, result.selected)
    except Exception:
        if require_checkpoint:
            raise
        return solve_analytic(source, config, device=device)


def _field(source: Any, name: str) -> Any:
    if isinstance(source, dict):
        return source.get(name)
    return getattr(source, name)
