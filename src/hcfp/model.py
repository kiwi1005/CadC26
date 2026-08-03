"""Small trainable HCFP model components.

The module is intentionally inference-neutral: random weights are trainable
building blocks, not a claim of learned quality until a checkpoint is supplied.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from hcfp.case import FloorplanCase
from hcfp.topology import DualPermutationHead


Tensor = torch.Tensor
_NODE_FEATURES = 22


@dataclass(frozen=True)
class ModelConfig:
    hidden_dim: int = 128
    encoder_layers: int = 3
    population_embed_dim: int = 8
    residual_bound: float = 0.10
    aspect_residual_bound: float = 0.25
    force_channels: int = 7
    candidate_metric_dim: int = 8
    compute_dtype: str = "float32"
    topology_enabled: bool = False

    def __post_init__(self) -> None:
        if self.hidden_dim <= 0 or self.encoder_layers <= 0:
            raise ValueError("hidden_dim and encoder_layers must be positive")
        if self.population_embed_dim <= 0 or self.force_channels <= 0 or self.candidate_metric_dim <= 0:
            raise ValueError("model dimensions must be positive")
        if self.residual_bound <= 0 or self.aspect_residual_bound <= 0:
            raise ValueError("residual bounds must be positive")
        if self.compute_dtype not in {"float32", "bfloat16"}:
            raise ValueError("compute_dtype must be float32 or bfloat16")


@dataclass(frozen=True)
class ModelOutput:
    embedding: Tensor
    precedence_logits: Tensor
    outline: Tensor
    center_residual: Tensor
    log_aspect_residual: Tensor
    flow_velocity: Tensor
    force_gates: Tensor
    rank_score: Tensor
    positive_permutation: Tensor | None = None
    negative_permutation: Tensor | None = None


def soft_sequence_pair_relation_logits(positive: Tensor, negative: Tensor) -> Tensor:
    """Convert two soft rank assignments into differentiable L/R/U/D logits."""

    plus = torch.as_tensor(positive)
    minus = torch.as_tensor(negative, device=plus.device)
    if plus.shape != minus.shape or plus.ndim < 2 or plus.shape[-1] != plus.shape[-2]:
        raise ValueError("soft permutations must have matching [...,N,N] shapes")
    if not torch.is_floating_point(plus) or not torch.is_floating_point(minus):
        raise ValueError("soft permutations must be floating point")
    n = plus.shape[-1]
    before_rank = torch.triu(
        torch.ones((n, n), dtype=plus.dtype, device=plus.device),
        diagonal=1,
    )
    plus_before = plus @ before_rank @ plus.transpose(-1, -2)
    minus_before = minus @ before_rank @ minus.transpose(-1, -2)
    plus_after = plus_before.transpose(-1, -2)
    minus_after = minus_before.transpose(-1, -2)
    probabilities = torch.stack(
        (
            plus_before * minus_before,
            plus_after * minus_after,
            plus_before * minus_after,
            plus_after * minus_before,
        ),
        dim=-1,
    )
    return probabilities.clamp_min(torch.finfo(probabilities.dtype).tiny).log()


def _mlp(width: int, layers: int) -> nn.Sequential:
    blocks: list[nn.Module] = []
    for _ in range(layers):
        blocks.extend((nn.LayerNorm(width), nn.Linear(width, width), nn.SiLU()))
    return nn.Sequential(*blocks)


class SceneEncoder(nn.Module):
    """Dense static encoder over one normalized FloorplanCase."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.input = nn.Linear(_NODE_FEATURES, config.hidden_dim)
        self.message = nn.Linear(config.hidden_dim, config.hidden_dim, bias=False)
        if config.topology_enabled:
            self.group_message = nn.Linear(config.hidden_dim, config.hidden_dim, bias=False)
            self.mib_message = nn.Linear(config.hidden_dim, config.hidden_dim, bias=False)
        self.layers = _mlp(config.hidden_dim, config.encoder_layers)

    def forward(self, case: FloorplanCase) -> Tensor:
        features = self._features(case)
        hidden = self.input(features)
        weights = case.b2b_weight.to(device=hidden.device, dtype=hidden.dtype)
        weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1.0)
        if self.config.topology_enabled:
            typed_weights = []
            for membership in (case.group_membership, case.mib_membership):
                adjacency = torch.zeros_like(weights)
                if membership.numel():
                    active = membership.to(device=hidden.device, dtype=hidden.dtype)
                    adjacency = active.transpose(0, 1) @ active
                    adjacency.fill_diagonal_(0.0)
                    adjacency = (adjacency > 0.0).to(dtype=hidden.dtype)
                typed_weights.append(
                    adjacency / adjacency.sum(dim=1, keepdim=True).clamp_min(1.0)
                )
        for layer in self.layers:
            if isinstance(layer, nn.LayerNorm):
                message = self.message(weights @ hidden)
                if self.config.topology_enabled:
                    message = message + self.group_message(typed_weights[0] @ hidden)
                    message = message + self.mib_message(typed_weights[1] @ hidden)
                hidden = hidden + message
            hidden = layer(hidden)
        return hidden.float()

    @staticmethod
    def _features(case: FloorplanCase) -> Tensor:
        device = case.area.device
        degree = case.b2b_weight.sum(dim=1)
        max_weight = case.b2b_weight.max(dim=1).values
        pin_weight = torch.zeros(case.n, dtype=torch.float32, device=device)
        pin_centroid = torch.zeros(case.n, 2, dtype=torch.float32, device=device)
        pin_spread = torch.zeros(case.n, 2, dtype=torch.float32, device=device)
        if case.p2b_edges.numel():
            edges = case.p2b_edges.to(device=device)
            pins = case.pins.to(device=device, dtype=torch.float32)
            pin_index = edges[:, 0].long()
            block_index = edges[:, 1].long()
            weight = edges[:, 2].float()
            weighted_pins = pins[pin_index] * weight[:, None]
            pin_weight.index_add_(0, block_index, weight)
            pin_centroid.index_add_(0, block_index, weighted_pins)
            pin_centroid = pin_centroid / pin_weight.clamp_min(1.0e-6)[:, None]
            pin_spread.index_add_(0, block_index, (pins[pin_index] - pin_centroid[block_index]).abs() * weight[:, None])
            pin_spread = pin_spread / pin_weight.clamp_min(1.0e-6)[:, None]

        group_present = case.group_membership.to(device=device).any(dim=0) if case.group_membership.numel() else torch.zeros(case.n, dtype=torch.bool, device=device)
        mib_present = case.mib_membership.to(device=device).any(dim=0) if case.mib_membership.numel() else torch.zeros(case.n, dtype=torch.bool, device=device)
        return torch.cat(
            (
                torch.log(case.area).unsqueeze(1),
                torch.sqrt(case.area).unsqueeze(1),
                degree.unsqueeze(1),
                max_weight.unsqueeze(1),
                case.fixed_mask.float().unsqueeze(1),
                case.preplaced_mask.float().unsqueeze(1),
                case.target_valid_mask.float().unsqueeze(1),
                case.target.float(),
                case.boundary_bits.float(),
                group_present.float().unsqueeze(1),
                mib_present.float().unsqueeze(1),
                pin_weight.unsqueeze(1),
                pin_centroid,
                pin_spread,
            ),
            dim=1,
        )


class StructureHeads(nn.Module):
    """Pairwise precedence and temporary-outline predictions."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.left = nn.Linear(config.hidden_dim, config.hidden_dim, bias=False)
        self.right = nn.Linear(config.hidden_dim, config.hidden_dim, bias=False)
        self.precedence = nn.Linear(config.hidden_dim, 5)
        self.outline = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.SiLU(),
            nn.Linear(config.hidden_dim, 2),
        )

    def forward(self, case: FloorplanCase, embedding: Tensor) -> tuple[Tensor, Tensor]:
        pair = torch.tanh(self.left(embedding)[:, None, :] + self.right(embedding)[None, :, :])
        precedence = self.precedence(pair).float()
        raw_outline = self.outline(embedding.mean(dim=0)).float()
        utilization = 0.45 + 0.50 * torch.sigmoid(raw_outline[0])
        ratio = torch.exp(2.0 * torch.tanh(raw_outline[1]))
        envelope_area = case.area.sum().float() / utilization
        width = torch.sqrt(envelope_area * ratio)
        height = envelope_area / width
        outline = torch.stack((width, height, utilization, ratio))
        return precedence, outline


class ResidualInitializer(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.population = nn.Embedding(256, config.population_embed_dim)
        self.head = nn.Sequential(
            nn.Linear(config.hidden_dim + config.population_embed_dim, config.hidden_dim),
            nn.SiLU(),
            nn.Linear(config.hidden_dim, 3),
        )

    def forward(self, case: FloorplanCase, embedding: Tensor, population: int) -> tuple[Tensor, Tensor]:
        if population <= 0 or population > self.population.num_embeddings:
            raise ValueError("population must be in [1, 256]")
        ids = torch.arange(population, device=embedding.device)
        pop = self.population(ids)[:, None, :].expand(population, case.n, -1)
        emb = embedding[None, :, :].expand(population, case.n, -1)
        raw = self.head(torch.cat((emb, pop), dim=-1)).float()
        center = torch.tanh(raw[..., :2]) * self.config.residual_bound
        aspect = torch.tanh(raw[..., 2]) * self.config.aspect_residual_bound
        center[:, case.preplaced_mask.to(device=center.device)] = 0.0
        aspect[:, (case.fixed_mask | case.preplaced_mask).to(device=aspect.device)] = 0.0
        return center, aspect


class RectifiedFlowHead(nn.Module):
    """Velocity field over residual ``(cx, cy, log-aspect)`` states."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(config.hidden_dim + 4, config.hidden_dim),
            nn.SiLU(),
            nn.Linear(config.hidden_dim, 3),
        )

    def forward(
        self,
        case: FloorplanCase,
        embedding: Tensor,
        population: int,
        state: Tensor | None,
        time: float | Tensor,
    ) -> Tensor:
        if state is None:
            work = torch.zeros(population, case.n, 3, dtype=embedding.dtype, device=embedding.device)
        else:
            work = torch.as_tensor(state, dtype=embedding.dtype, device=embedding.device)
            if work.shape != (population, case.n, 3):
                raise ValueError("flow_state must have shape [population, N, 3]")
        step = torch.as_tensor(time, dtype=embedding.dtype, device=embedding.device)
        if step.numel() == 1:
            step = step.reshape(1, 1, 1).expand(population, case.n, 1)
        elif step.shape == (population,):
            step = step.reshape(population, 1, 1).expand(population, case.n, 1)
        else:
            raise ValueError("flow_time must be scalar or shape [population]")
        emb = embedding[None, :, :].expand(population, case.n, -1)
        velocity = self.net(torch.cat((emb, work, step), dim=-1)).float()
        velocity[:, case.preplaced_mask.to(device=velocity.device), :2] = 0.0
        velocity[:, (case.fixed_mask | case.preplaced_mask).to(device=velocity.device), 2] = 0.0
        return velocity


class TypedForceGateController(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(config.hidden_dim + 1, config.hidden_dim),
            nn.SiLU(),
            nn.Linear(config.hidden_dim, config.force_channels),
        )

    def forward(self, embedding: Tensor, population: int, step_fraction: float | Tensor = 0.0) -> Tensor:
        step = torch.as_tensor(step_fraction, dtype=embedding.dtype, device=embedding.device).reshape(1, 1, 1)
        step = step.expand(population, embedding.shape[0], 1)
        emb = embedding[None, :, :].expand(population, -1, -1)
        return torch.nn.functional.softplus(self.net(torch.cat((emb, step), dim=-1))).float() + 1.0e-6


class RepairAwareRanker(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(config.hidden_dim + config.candidate_metric_dim, config.hidden_dim),
            nn.SiLU(),
            nn.Linear(config.hidden_dim, 1),
        )

    def forward(self, embedding: Tensor, population: int, candidate_metrics: Tensor | None = None) -> Tensor:
        pooled = embedding.mean(dim=0, keepdim=True).expand(population, -1)
        if candidate_metrics is None:
            metrics = torch.zeros(population, self.net[0].in_features - embedding.shape[1], device=embedding.device)
        else:
            metrics = torch.as_tensor(candidate_metrics, dtype=embedding.dtype, device=embedding.device)
            if metrics.shape != (population, self.net[0].in_features - embedding.shape[1]):
                raise ValueError("candidate_metrics shape does not match [population, candidate_metric_dim]")
        return self.net(torch.cat((pooled, metrics), dim=1)).squeeze(1).float()


class HCFPModel(nn.Module):
    def __init__(self, config: ModelConfig | None = None):
        super().__init__()
        self.config = config or ModelConfig()
        self.encoder = SceneEncoder(self.config)
        self.structure = StructureHeads(self.config)
        self.initializer = ResidualInitializer(self.config)
        self.flow = RectifiedFlowHead(self.config)
        self.gates = TypedForceGateController(self.config)
        self.ranker = RepairAwareRanker(self.config)
        if self.config.topology_enabled:
            self.topology = DualPermutationHead(self.config.hidden_dim)

    def forward(
        self,
        case: FloorplanCase,
        *,
        population: int,
        candidate_metrics: Tensor | None = None,
        step_fraction: float | Tensor = 0.0,
        flow_state: Tensor | None = None,
        flow_time: float | Tensor = 0.0,
    ) -> ModelOutput:
        device_type = "cuda" if case.area.is_cuda else "cpu"
        enabled = self.config.compute_dtype == "bfloat16"
        with torch.autocast(device_type=device_type, dtype=torch.bfloat16, enabled=enabled):
            embedding = self.encoder(case)
            precedence, outline = self.structure(case, embedding)
            positive: Tensor | None = None
            negative: Tensor | None = None
            if self.config.topology_enabled:
                positive, negative = self.topology(embedding, case.block_mask)
            center, aspect = self.initializer(case, embedding, population)
            flow_velocity = self.flow(case, embedding, population, flow_state, flow_time)
            gates = self.gates(embedding, population, step_fraction)
            score = self.ranker(embedding, population, candidate_metrics)
        return ModelOutput(
            embedding=embedding.float(),
            precedence_logits=precedence.float(),
            outline=outline.float(),
            center_residual=center.float(),
            log_aspect_residual=aspect.float(),
            flow_velocity=flow_velocity.float(),
            force_gates=gates.float(),
            rank_score=score.float(),
            positive_permutation=(positive.float() if positive is not None else None),
            negative_permutation=(negative.float() if negative is not None else None),
        )
