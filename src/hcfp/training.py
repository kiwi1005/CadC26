"""Minimal supervised training loop for structure, initializer, and flow heads."""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Callable, Iterable, Iterator

import torch
from torch.nn import functional as F

from hcfp.data import DataSample, SolutionLabels
from hcfp.dynamics import DynamicsConfig, initialize_population
from hcfp.fallback import safe_shelf
from hcfp.model import HCFPModel


Tensor = torch.Tensor
TRAINING_STAGES = ("structure", "initializer", "flow", "all")


@dataclass(frozen=True)
class LossReport:
    total: Tensor
    structure: Tensor
    initializer: Tensor
    flow: Tensor

    def scalars(self) -> dict[str, float]:
        return {
            "total": float(self.total.detach()),
            "structure": float(self.structure.detach()),
            "initializer": float(self.initializer.detach()),
            "flow": float(self.flow.detach()),
        }


def supervised_loss(
    model: HCFPModel,
    sample: DataSample,
    *,
    population: int,
    stage: str = "all",
    seed: int = 0,
) -> LossReport:
    """Compute stage-gated losses from one normalized, auditable sample."""

    if stage not in TRAINING_STAGES:
        raise ValueError(f"stage must be one of {TRAINING_STAGES}")
    device = next(model.parameters()).device
    case = sample.case.to(device=device, dtype=torch.float32)
    labels = _labels_to(sample.labels, device)
    base = initialize_population(
        case,
        DynamicsConfig(population=population, steps=0),
        safe_shelf(case).to(device=device),
    )
    output = model(case, population=population)
    zero = output.embedding.sum() * 0.0

    structure = zero
    if stage in {"structure", "all"}:
        n = case.n
        valid = ~labels.precedence_tie_mask
        valid &= ~torch.eye(n, dtype=torch.bool, device=device)
        precedence = F.cross_entropy(output.precedence_logits[valid], labels.pairwise_precedence[valid])
        outline = F.smooth_l1_loss(output.outline, labels.outline)
        structure = precedence + outline

    target_center = labels.centers.unsqueeze(0) - base.center
    target_aspect = labels.log_aspect.unsqueeze(0) - base.log_aspect
    target_center = target_center.clamp(-model.config.residual_bound, model.config.residual_bound)
    target_aspect = target_aspect.clamp(-model.config.aspect_residual_bound, model.config.aspect_residual_bound)
    target_center[:, case.preplaced_mask] = 0.0
    target_aspect[:, case.fixed_mask | case.preplaced_mask] = 0.0

    initializer = zero
    if stage in {"initializer", "all"}:
        initializer = F.smooth_l1_loss(output.center_residual, target_center)
        initializer += F.smooth_l1_loss(output.log_aspect_residual, target_aspect)

    flow = zero
    if stage in {"flow", "all"}:
        qstar = torch.cat((target_center, target_aspect.unsqueeze(-1)), dim=-1)
        generator = torch.Generator(device="cpu").manual_seed(int(seed))
        noise = torch.randn(qstar.shape, generator=generator, dtype=torch.float32).to(device=device)
        noise[:, case.preplaced_mask, :2] = 0.0
        noise[:, case.fixed_mask | case.preplaced_mask, 2] = 0.0
        time = torch.rand(population, generator=generator, dtype=torch.float32).to(device=device)
        qt = (1.0 - time[:, None, None]) * noise + time[:, None, None] * qstar
        prediction = model(case, population=population, flow_state=qt, flow_time=time).flow_velocity
        flow = F.mse_loss(prediction, qstar - noise)

    return LossReport(structure + initializer + flow, structure, initializer, flow)


def train_steps(
    model: HCFPModel,
    samples: Iterable[DataSample] | Callable[[], Iterable[DataSample]],
    optimizer: torch.optim.Optimizer,
    *,
    steps: int,
    population: int,
    stage: str = "all",
    seed: int = 0,
) -> list[dict[str, float]]:
    """Train for a bounded number of deterministic sample-cycling steps."""

    if steps <= 0:
        raise ValueError("steps must be positive")
    if callable(samples):
        sample_factory = samples
    else:
        materialized = list(samples)
        if not materialized:
            raise ValueError("training requires at least one sample")

        def sample_factory() -> Iterable[DataSample]:
            return iter(materialized)
    iterator: Iterator[DataSample] = iter(sample_factory())
    history = []
    model.train()
    for index in range(steps):
        try:
            sample = next(iterator)
        except StopIteration:
            iterator = iter(sample_factory())
            try:
                sample = next(iterator)
            except StopIteration as exc:
                raise ValueError("training requires at least one sample") from exc
        optimizer.zero_grad(set_to_none=True)
        report = supervised_loss(
            model,
            sample,
            population=population,
            stage=stage,
            seed=seed + index,
        )
        report.total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        history.append(report.scalars())
    return history


def _labels_to(labels: SolutionLabels, device: torch.device) -> SolutionLabels:
    return SolutionLabels(
        rectangles=labels.rectangles.to(device=device),
        centers=labels.centers.to(device=device),
        log_aspect=labels.log_aspect.to(device=device),
        pairwise_precedence=labels.pairwise_precedence.to(device=device),
        precedence_tie_mask=labels.precedence_tie_mask.to(device=device),
        outline=labels.outline.to(device=device),
    )
