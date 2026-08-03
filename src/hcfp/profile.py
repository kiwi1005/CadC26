"""Synthetic runtime profiling for the HCFP analytic lane."""

from __future__ import annotations

from dataclasses import dataclass
import json
import time
from pathlib import Path
from typing import Any

import torch

from hcfp.analytic import AnalyticConfig, select_device, solve_case, solve_case_with_telemetry
from hcfp.benchmark import percentile
from hcfp.case import FloorplanCase, from_official
from hcfp.dynamics import DynamicsConfig
from hcfp.projection import ComponentBDPConfig
from hcfp.verify import verify_feasible


SYNTHETIC_BUCKETS = (32, 64, 96, 120)


@dataclass(frozen=True)
class ProfileConfig:
    block_count: int = 120
    candidates: int = 32
    steps: int = 4
    repeats: int = 3
    warmups: int = 1
    projection_iterations: int = 8
    direction_beam: int = 2
    component_bdp: bool = False
    device: str = "auto"

    def __post_init__(self) -> None:
        if self.block_count not in SYNTHETIC_BUCKETS:
            raise ValueError(f"block_count must be one of {SYNTHETIC_BUCKETS}")
        if self.candidates <= 0:
            raise ValueError("candidates must be positive")
        if self.steps < 0 or self.repeats <= 0 or self.warmups < 0:
            raise ValueError("steps/repeats/warmups must be non-negative, with repeats positive")
        if self.projection_iterations <= 0 or self.direction_beam <= 0:
            raise ValueError("projection_iterations and direction_beam must be positive")


def synthetic_case(block_count: int, *, device: str | torch.device | None = None) -> FloorplanCase:
    """Build a deterministic FloorSet-like synthetic case without official data."""

    if block_count not in SYNTHETIC_BUCKETS:
        raise ValueError(f"block_count must be one of {SYNTHETIC_BUCKETS}")
    areas = [float(1.0 + (i % 7) * 0.17) for i in range(block_count)]
    b2b = []
    for i in range(block_count):
        b2b.append([i, (i + 1) % block_count, 1.0 + (i % 5) * 0.25])
        if i + 8 < block_count:
            b2b.append([i, i + 8, 0.35])
    pins = [[float(i % 12) * 1.7, float(i // 12) * 1.3] for i in range(max(4, block_count // 8))]
    p2b = [[i % len(pins), i, 0.20 + (i % 3) * 0.10] for i in range(block_count)]
    constraints = [[0, 0, 0, 0, 0] for _ in range(block_count)]
    targets = [[-1.0, -1.0, -1.0, -1.0] for _ in range(block_count)]

    side0 = areas[0] ** 0.5
    constraints[0] = [0, 1, 0, 0, 1]
    targets[0] = [0.0, 0.0, side0, side0]
    if block_count >= 64:
        side1 = areas[1] ** 0.5
        constraints[1] = [1, 0, 0, 0, 2]
        targets[1] = [-1.0, -1.0, side1 * 1.25, side1 / 1.25]
    for i in range(2, min(block_count, 10), 2):
        constraints[i][3] = 1
    for i in range(10, min(block_count, 16)):
        constraints[i][2] = 1

    return from_official(
        block_count,
        areas,
        b2b,
        p2b,
        pins,
        constraints,
        targets,
        device=device,
    )


def run_profile(config: ProfileConfig) -> dict[str, Any]:
    """Profile one synthetic bucket and return a JSON-serializable report."""

    device = select_device(config.device)
    case = synthetic_case(config.block_count, device=device)
    analytic_config = AnalyticConfig(
        dynamics=DynamicsConfig(population=config.candidates, steps=config.steps),
        projection_iterations=config.projection_iterations,
        direction_beam=config.direction_beam,
        component_bdp=ComponentBDPConfig(enabled=config.component_bdp),
    )
    for _ in range(config.warmups):
        _profile_once(case, analytic_config, device)

    timings = []
    peaks = []
    selected = None
    for _ in range(config.repeats):
        sample = _profile_once(case, analytic_config, device)
        timings.append(sample["seconds"])
        peaks.append(sample["cuda_peak_bytes"])
        selected = sample["selected"]

    assert selected is not None
    analysis = solve_case_with_telemetry(case, analytic_config)
    timing_sorted = sorted(timings)
    feasible = verify_feasible(case.to(device="cpu", dtype=torch.float32), selected.to(device="cpu"))
    telemetry = analysis.telemetry
    report = {
        "schema_version": 1,
        "config": {
            "block_count": config.block_count,
            "candidates": config.candidates,
            "steps": config.steps,
            "repeats": config.repeats,
            "warmups": config.warmups,
            "projection_iterations": config.projection_iterations,
            "direction_beam": config.direction_beam,
            "component_bdp": config.component_bdp,
            "requested_device": config.device,
            "actual_device": str(device),
        },
        "timing_seconds": {
            "samples": timings,
            "p50": percentile(timing_sorted, 0.50),
            "p95": percentile(timing_sorted, 0.95),
            "p99": percentile(timing_sorted, 0.99),
            "max": max(timing_sorted),
        },
        "cuda": {
            "available": torch.cuda.is_available(),
            "peak_bytes_per_repeat": peaks,
            "peak_bytes_max": max(peaks, default=0),
        },
        "phases": {
            "case_builder": "synthetic_case",
            "solver": "solve_case",
            "telemetry": "collected_once_outside_timing",
            "synchronized_timing": device.type == "cuda",
        },
        "candidate_metadata": {
            "raw_candidates": int(analysis.raw_candidates.shape[0]),
            "projected_candidates": int(analysis.projected_candidates.shape[0]),
            "blocks": int(analysis.projected_candidates.shape[1]),
            "energy_history_shape": list(analysis.energy_history.shape),
            "projection_status": analysis.projection_status,
            "projection_ok": int(torch.count_nonzero(telemetry.projection_ok).item()),
            "hard_feasible_candidates": int(torch.count_nonzero(telemetry.hard_feasible).item()),
            "max_active_pairs": int(telemetry.projection_active_pairs.max().item()),
            "max_raw_overlap": float(telemetry.raw_overlap.max().item()),
            "max_projected_overlap": float(telemetry.projected_overlap.max().item()),
            "component_rebuilds": int(
                telemetry.projection_component_rebuilds.sum().item()
            ),
            "new_pairs_detected": int(telemetry.projection_new_pairs.sum().item()),
            "resets": int(telemetry.projection_resets.sum().item()),
            "beam_states_evaluated": int(
                telemetry.projection_beam_states.sum().item()
            ),
            "max_component_size": int(
                telemetry.projection_max_component_size.max().item()
            ),
        },
        "incumbent": {
            "feasible": bool(feasible),
            "source": analysis.incumbent_snapshot.get("exact_source"),
            "snapshot": analysis.incumbent_snapshot,
        },
    }
    return report


def write_profile(report: dict[str, Any], output: str | Path | None) -> None:
    text = json.dumps(report, indent=2, sort_keys=True)
    if output is None:
        print(text)
        return
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text + "\n", encoding="utf-8")


def _profile_once(case: FloorplanCase, config: AnalyticConfig, device: torch.device) -> dict[str, Any]:
    _sync(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    start = time.perf_counter()
    selected = solve_case(case, config)
    _sync(device)
    seconds = time.perf_counter() - start
    peak = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0
    return {"seconds": seconds, "cuda_peak_bytes": int(peak), "selected": selected}


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
