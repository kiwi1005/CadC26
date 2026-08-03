"""Minimal supervised training loop for structure, initializer, and flow heads."""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Callable, Iterable, Iterator

import torch
from torch.nn import functional as F

from hcfp.data import DataSample, SolutionLabels
from hcfp.dynamics import DynamicsConfig, initialize_population
from hcfp.fallback import safe_shelf
from hcfp.model import HCFPModel, soft_sequence_pair_relation_logits
from hcfp.constraints.contact_tree import BOTTOM, LEFT, RIGHT, TOP, extract_contacts
from hcfp.topology import antisymmetry_loss, partial_label_nll, relation_mask_from_rectangles


Tensor = torch.Tensor
TRAINING_STAGES = ("structure", "initializer", "flow", "all")


@dataclass(frozen=True)
class LossReport:
    total: Tensor
    structure: Tensor
    initializer: Tensor
    flow: Tensor
    constraint: Tensor

    def scalars(self) -> dict[str, float]:
        return {
            "total": float(self.total.detach()),
            "structure": float(self.structure.detach()),
            "initializer": float(self.initializer.detach()),
            "flow": float(self.flow.detach()),
            "constraint": float(self.constraint.detach()),
        }


class ExponentialMovingAverage:
    """Small in-device EMA used only while training."""

    def __init__(self, model: HCFPModel, decay: float = 0.999) -> None:
        if not 0.0 < decay < 1.0:
            raise ValueError("EMA decay must be in (0, 1)")
        self.decay = float(decay)
        self.shadow = {
            name: value.detach().clone()
            for name, value in model.state_dict().items()
            if torch.is_floating_point(value)
        }

    @torch.no_grad()
    def update(self, model: HCFPModel) -> None:
        for name, value in model.state_dict().items():
            if name in self.shadow:
                self.shadow[name].lerp_(value.detach(), 1.0 - self.decay)

    @torch.no_grad()
    def copy_to(self, model: HCFPModel) -> None:
        state = model.state_dict()
        for name, value in self.shadow.items():
            state[name].copy_(value)


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
    target_center = labels.centers.unsqueeze(0) - base.center
    target_aspect = labels.log_aspect.unsqueeze(0) - base.log_aspect
    target_center = target_center.clamp(-model.config.residual_bound, model.config.residual_bound)
    target_aspect = target_aspect.clamp(-model.config.aspect_residual_bound, model.config.aspect_residual_bound)
    target_center[:, case.preplaced_mask] = 0.0
    target_aspect[:, case.fixed_mask | case.preplaced_mask] = 0.0

    flow_state = None
    flow_time: float | Tensor = 0.0
    flow_target = None
    if stage in {"flow", "all"}:
        qstar = torch.cat((target_center, target_aspect.unsqueeze(-1)), dim=-1)
        generator = torch.Generator(device="cpu").manual_seed(int(seed))
        noise = torch.randn(qstar.shape, generator=generator, dtype=torch.float32).to(device=device)
        noise[:, case.preplaced_mask, :2] = 0.0
        noise[:, case.fixed_mask | case.preplaced_mask, 2] = 0.0
        flow_time = torch.rand(population, generator=generator, dtype=torch.float32).to(device=device)
        flow_state = (1.0 - flow_time[:, None, None]) * noise + flow_time[:, None, None] * qstar
        flow_target = qstar - noise

    output = model(
        case,
        population=population,
        flow_state=flow_state,
        flow_time=flow_time,
    )
    zero = output.embedding.sum() * 0.0

    structure = zero
    constraint = zero
    if stage in {"structure", "all"}:
        allowed = relation_mask_from_rectangles(
            labels.rectangles,
            valid_mask=case.block_mask,
        )
        valid = allowed.any(dim=-1)
        relation_logits = output.precedence_logits[..., :4]
        precedence = partial_label_nll(relation_logits, allowed, pair_mask=valid)
        precedence += antisymmetry_loss(relation_logits, pair_mask=valid)
        if output.positive_permutation is not None and output.negative_permutation is not None:
            topology_logits = soft_sequence_pair_relation_logits(
                output.positive_permutation,
                output.negative_permutation,
            )
            precedence += partial_label_nll(topology_logits, allowed, pair_mask=valid)
        outline = F.smooth_l1_loss(output.outline, labels.outline)
        if output.contact_logits is not None:
            contact_target, contact_mask = _contact_targets(case, labels)
            if bool(contact_mask.any()):
                constraint = constraint + F.cross_entropy(
                    output.contact_logits[contact_mask],
                    contact_target[contact_mask],
                )
        if output.boundary_order_scores is not None:
            boundary_target, boundary_mask = _boundary_order_targets(case, labels)
            if bool(boundary_mask.any()):
                constraint = constraint + F.smooth_l1_loss(
                    torch.sigmoid(output.boundary_order_scores[boundary_mask]),
                    boundary_target[boundary_mask],
                )
        if output.mib_log_aspect is not None:
            mib_target, mib_mask = _mib_log_aspect_targets(case, labels)
            if bool(mib_mask.any()):
                constraint = constraint + F.smooth_l1_loss(
                    output.mib_log_aspect[mib_mask],
                    mib_target[mib_mask],
                )
        structure = precedence + outline + constraint

    initializer = zero
    if stage in {"initializer", "all"}:
        initializer = F.smooth_l1_loss(output.center_residual, target_center)
        initializer += F.smooth_l1_loss(output.log_aspect_residual, target_aspect)

    flow = zero
    if flow_target is not None:
        flow = F.mse_loss(output.flow_velocity, flow_target)

    return LossReport(structure + initializer + flow, structure, initializer, flow, constraint)


def train_steps(
    model: HCFPModel,
    samples: Iterable[DataSample] | Callable[[], Iterable[DataSample]],
    optimizer: torch.optim.Optimizer,
    *,
    steps: int,
    population: int,
    stage: str = "all",
    seed: int = 0,
    ema: ExponentialMovingAverage | None = None,
    on_step: Callable[[int, LossReport], None] | None = None,
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
        if ema is not None:
            ema.update(model)
        if on_step is not None:
            on_step(index + 1, report)
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
        baseline_area=labels.baseline_area.to(device=device),
        baseline_hpwl=labels.baseline_hpwl.to(device=device),
    )


def _contact_targets(case, labels: SolutionLabels) -> tuple[Tensor, Tensor]:
    membership = case.group_membership.to(device=labels.rectangles.device, dtype=torch.bool)
    n = case.n
    if membership.numel():
        active = membership.to(dtype=labels.rectangles.dtype)
        same_group = (active.transpose(0, 1) @ active) > 0
    else:
        same_group = torch.zeros((n, n), dtype=torch.bool, device=labels.rectangles.device)
    diagonal = torch.eye(n, dtype=torch.bool, device=labels.rectangles.device)
    mask = same_group & ~diagonal & case.block_mask[:, None] & case.block_mask[None, :]
    target = torch.zeros((n, n), dtype=torch.long, device=labels.rectangles.device)
    if bool(mask.any()):
        contacts = extract_contacts(
            labels.rectangles.detach().to(device="cpu"),
            net_weight=case.b2b_weight.detach().to(device="cpu"),
            tolerance=1.0e-6,
        )
        side_class = {LEFT: 1, RIGHT: 2, TOP: 3, BOTTOM: 4}
        inverse = {LEFT: RIGHT, RIGHT: LEFT, TOP: BOTTOM, BOTTOM: TOP}
        for contact in contacts:
            first = int(contact.first)
            second = int(contact.second)
            if bool(mask[first, second]):
                target[first, second] = side_class[contact.first_side]
            if bool(mask[second, first]):
                target[second, first] = side_class[inverse[contact.first_side]]
    return target, mask


def _boundary_order_targets(case, labels: SolutionLabels) -> tuple[Tensor, Tensor]:
    bits = case.boundary_bits.to(device=labels.rectangles.device, dtype=torch.bool)
    rects = labels.rectangles
    centers = labels.centers
    left = rects[:, 0].amin()
    bottom = rects[:, 1].amin()
    right = (rects[:, 0] + rects[:, 2]).amax()
    top = (rects[:, 1] + rects[:, 3]).amax()
    width = (right - left).clamp_min(1.0e-12)
    height = (top - bottom).clamp_min(1.0e-12)
    x_norm = (centers[:, 0] - left) / width
    y_norm = (centers[:, 1] - bottom) / height
    target = torch.stack((y_norm, y_norm, x_norm, x_norm), dim=1).to(dtype=torch.float32)
    mask = bits & case.block_mask[:, None]
    return target, mask


def _mib_log_aspect_targets(case, labels: SolutionLabels) -> tuple[Tensor, Tensor]:
    membership = case.mib_membership.to(device=labels.rectangles.device, dtype=torch.bool)
    if membership.shape[0] == 0:
        return labels.log_aspect.new_empty((0,)), torch.zeros((0,), dtype=torch.bool, device=labels.rectangles.device)
    mask = membership & case.block_mask[None, :]
    counts = mask.sum(dim=1)
    target = labels.log_aspect.new_zeros((membership.shape[0],))
    active = counts > 0
    if bool(active.any()):
        weights = mask[active].to(dtype=labels.log_aspect.dtype)
        target[active] = (weights @ labels.log_aspect) / counts[active].to(dtype=labels.log_aspect.dtype)
    return target, active
