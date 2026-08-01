"""Synthetic, synchronized training throughput profile."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import time
from typing import Any

import torch

from hcfp.analytic import select_device
from hcfp.benchmark import percentile
from hcfp.data import DataSample, extract_labels
from hcfp.fallback import safe_shelf
from hcfp.model import HCFPModel, ModelConfig
from hcfp.profile import SYNTHETIC_BUCKETS, synthetic_case
from hcfp.training import TRAINING_STAGES, supervised_loss


@dataclass(frozen=True)
class TrainingProfileConfig:
    block_count: int = 120
    population: int = 8
    hidden_dim: int = 128
    encoder_layers: int = 3
    stage: str = "all"
    compute_dtype: str = "float32"
    warmups: int = 3
    steps: int = 20
    device: str = "auto"

    def __post_init__(self) -> None:
        if self.block_count not in SYNTHETIC_BUCKETS:
            raise ValueError(f"block_count must be one of {SYNTHETIC_BUCKETS}")
        if self.population <= 0 or self.hidden_dim <= 0 or self.encoder_layers <= 0:
            raise ValueError("model/profile dimensions must be positive")
        if self.stage not in TRAINING_STAGES:
            raise ValueError(f"stage must be one of {TRAINING_STAGES}")
        if self.compute_dtype not in {"float32", "bfloat16"}:
            raise ValueError("compute_dtype must be float32 or bfloat16")
        if self.warmups < 0 or self.steps <= 0:
            raise ValueError("warmups must be non-negative and steps positive")


def run_training_profile(config: TrainingProfileConfig) -> dict[str, Any]:
    device = select_device(config.device)
    cpu_case = synthetic_case(config.block_count, device="cpu")
    sample = DataSample(
        "training-profile",
        cpu_case,
        extract_labels(cpu_case, safe_shelf(cpu_case), normalized=True),
    )
    model = HCFPModel(
        ModelConfig(
            hidden_dim=config.hidden_dim,
            encoder_layers=config.encoder_layers,
            compute_dtype=config.compute_dtype,
        )
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3.0e-4)
    for index in range(config.warmups):
        _step(model, sample, optimizer, config, index)
    _sync(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    timings = []
    last = None
    for index in range(config.steps):
        start = time.perf_counter()
        last = _step(model, sample, optimizer, config, config.warmups + index)
        _sync(device)
        timings.append(time.perf_counter() - start)
    ordered = sorted(timings)
    return {
        "schema_version": 1,
        "config": {**asdict(config), "actual_device": str(device)},
        "model_parameters": sum(parameter.numel() for parameter in model.parameters()),
        "forward_calls_per_step": 1,
        "timing_seconds": {
            "samples": timings,
            "p50": percentile(ordered, 0.50),
            "p95": percentile(ordered, 0.95),
            "p99": percentile(ordered, 0.99),
            "max": max(ordered),
            "steps_per_second": config.steps / sum(timings),
        },
        "cuda_peak_bytes": int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0,
        "last_loss": last.scalars() if last is not None else {},
    }


def _step(model, sample, optimizer, config, seed):
    optimizer.zero_grad(set_to_none=True)
    report = supervised_loss(
        model,
        sample,
        population=config.population,
        stage=config.stage,
        seed=seed,
    )
    report.total.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
    optimizer.step()
    return report


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
