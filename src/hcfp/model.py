"""Small trainable HCFP model components.

The module is intentionally inference-neutral: random weights are trainable
building blocks, not a claim of learned quality until a checkpoint is supplied.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import nn

from hcfp.case import FloorplanCase
from hcfp.collective import PAIR_FEATURES
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
    initializer_absolute: bool = False
    force_channels: int = 7
    candidate_metric_dim: int = 8
    ranker_feature_mean: tuple[float, ...] = ()
    ranker_feature_scale: tuple[float, ...] = ()
    ranker_feature_version: str = "stored_candidate_features_v1"
    ranker_use_scene_embedding: bool = True
    compute_dtype: str = "float32"
    topology_enabled: bool = False
    btree_enabled: bool = False
    constraint_enabled: bool = False
    collective_enabled: bool = False
    baseline_enabled: bool = False
    collective_message_dim: int = 128
    collective_passes: int = 3
    collective_position_bound: float = 0.05
    collective_aspect_bound: float = 0.05
    collective_gate_delta: float = 0.50

    def __post_init__(self) -> None:
        if self.hidden_dim <= 0 or self.encoder_layers <= 0:
            raise ValueError("hidden_dim and encoder_layers must be positive")
        if (
            self.population_embed_dim <= 0
            or self.force_channels <= 0
            or self.candidate_metric_dim <= 0
        ):
            raise ValueError("model dimensions must be positive")
        if self.residual_bound <= 0 or self.aspect_residual_bound <= 0:
            raise ValueError("residual bounds must be positive")
        if type(self.initializer_absolute) is not bool:
            raise ValueError("initializer_absolute must be boolean")
        if self.compute_dtype not in {"float32", "bfloat16"}:
            raise ValueError("compute_dtype must be float32 or bfloat16")
        if type(self.ranker_use_scene_embedding) is not bool:
            raise ValueError("ranker_use_scene_embedding must be boolean")
        if type(self.btree_enabled) is not bool:
            raise ValueError("btree_enabled must be boolean")
        if type(self.baseline_enabled) is not bool:
            raise ValueError("baseline_enabled must be boolean")
        if (
            not isinstance(self.ranker_feature_version, str)
            or not self.ranker_feature_version
        ):
            raise ValueError("ranker_feature_version must be a non-empty string")
        if self.collective_message_dim <= 0 or self.collective_passes <= 0:
            raise ValueError("collective dimensions and passes must be positive")
        if self.collective_position_bound <= 0.0 or self.collective_aspect_bound <= 0.0:
            raise ValueError("collective update bounds must be positive")
        if not 0.0 < self.collective_gate_delta < 1.0:
            raise ValueError("collective_gate_delta must be in (0, 1)")
        if self.collective_enabled and self.force_channels != 7:
            raise ValueError(
                "collective dynamics require the canonical seven force channels"
            )
        mean = tuple(float(value) for value in self.ranker_feature_mean)
        scale = tuple(float(value) for value in self.ranker_feature_scale)
        object.__setattr__(self, "ranker_feature_mean", mean)
        object.__setattr__(self, "ranker_feature_scale", scale)
        if bool(mean) != bool(scale):
            raise ValueError(
                "ranker feature mean and scale must be configured together"
            )
        if mean and (
            len(mean) != self.candidate_metric_dim
            or len(scale) != self.candidate_metric_dim
        ):
            raise ValueError(
                "ranker feature normalization must match candidate_metric_dim"
            )
        if any(not math.isfinite(value) for value in (*mean, *scale)) or any(
            value <= 0.0 for value in scale
        ):
            raise ValueError(
                "ranker feature normalization must be finite with positive scales"
            )


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
    contact_logits: Tensor | None = None
    boundary_order_scores: Tensor | None = None
    mib_log_aspect: Tensor | None = None
    btree_root_logits: Tensor | None = None
    btree_edge_logits: Tensor | None = None
    baseline_log_area: Tensor | None = None
    baseline_log_hpwl: Tensor | None = None


@dataclass(frozen=True)
class CollectiveStepOutput:
    velocity: Tensor
    force_gates: Tensor


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
            self.group_message = nn.Linear(
                config.hidden_dim, config.hidden_dim, bias=False
            )
            self.mib_message = nn.Linear(
                config.hidden_dim, config.hidden_dim, bias=False
            )
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
            pin_spread.index_add_(
                0,
                block_index,
                (pins[pin_index] - pin_centroid[block_index]).abs() * weight[:, None],
            )
            pin_spread = pin_spread / pin_weight.clamp_min(1.0e-6)[:, None]

        group_present = (
            case.group_membership.to(device=device).any(dim=0)
            if case.group_membership.numel()
            else torch.zeros(case.n, dtype=torch.bool, device=device)
        )
        mib_present = (
            case.mib_membership.to(device=device).any(dim=0)
            if case.mib_membership.numel()
            else torch.zeros(case.n, dtype=torch.bool, device=device)
        )
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
        pair = torch.tanh(
            self.left(embedding)[:, None, :] + self.right(embedding)[None, :, :]
        )
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
            nn.Linear(
                config.hidden_dim + config.population_embed_dim, config.hidden_dim
            ),
            nn.SiLU(),
            nn.Linear(config.hidden_dim, 3),
        )

    def forward(
        self, case: FloorplanCase, embedding: Tensor, population: int
    ) -> tuple[Tensor, Tensor]:
        if population <= 0 or population > self.population.num_embeddings:
            raise ValueError("population must be in [1, 256]")
        ids = torch.arange(population, device=embedding.device)
        pop = self.population(ids)[:, None, :].expand(population, case.n, -1)
        emb = embedding[None, :, :].expand(population, case.n, -1)
        raw = self.head(torch.cat((emb, pop), dim=-1)).float()
        center = torch.tanh(raw[..., :2]) * self.config.residual_bound
        aspect = torch.tanh(raw[..., 2]) * self.config.aspect_residual_bound
        center[:, case.preplaced_mask.to(device=center.device)] = 0.0
        aspect[:, (case.fixed_mask | case.preplaced_mask).to(device=aspect.device)] = (
            0.0
        )
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
            work = torch.zeros(
                population, case.n, 3, dtype=embedding.dtype, device=embedding.device
            )
        else:
            work = torch.as_tensor(
                state, dtype=embedding.dtype, device=embedding.device
            )
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
        velocity[
            :, (case.fixed_mask | case.preplaced_mask).to(device=velocity.device), 2
        ] = 0.0
        return velocity


class TypedForceGateController(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(config.hidden_dim + 1, config.hidden_dim),
            nn.SiLU(),
            nn.Linear(config.hidden_dim, config.force_channels),
        )

    def forward(
        self, embedding: Tensor, population: int, step_fraction: float | Tensor = 0.0
    ) -> Tensor:
        step = torch.as_tensor(
            step_fraction, dtype=embedding.dtype, device=embedding.device
        ).reshape(1, 1, 1)
        step = step.expand(population, embedding.shape[0], 1)
        emb = embedding[None, :, :].expand(population, -1, -1)
        return (
            torch.nn.functional.softplus(
                self.net(torch.cat((emb, step), dim=-1))
            ).float()
            + 1.0e-6
        )


class GeometryAwareCollectiveHead(nn.Module):
    """Permutation-equivariant update from the current rectangle geometry."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        width = config.collective_message_dim
        self.static_node = nn.Linear(config.hidden_dim, width)
        self.geometry_node = nn.Linear(3, width, bias=False)
        self.step_node = nn.Linear(1, width, bias=False)
        self.pair = nn.Linear(len(PAIR_FEATURES), width, bias=False)
        self.sender = nn.Linear(width, width, bias=False)
        self.receiver = nn.Linear(width, width, bias=False)
        self.update_norm = nn.LayerNorm(width)
        self.update = nn.Sequential(
            nn.Linear(width, width),
            nn.SiLU(),
            nn.Linear(width, width),
        )
        self.velocity = nn.Sequential(
            nn.LayerNorm(width),
            nn.Linear(width, width),
            nn.SiLU(),
            nn.Linear(width, 3),
        )
        self.force_gates = nn.Sequential(
            nn.LayerNorm(width),
            nn.Linear(width, width),
            nn.SiLU(),
            nn.Linear(width, config.force_channels),
        )
        nn.init.zeros_(self.velocity[-1].weight)
        nn.init.zeros_(self.velocity[-1].bias)
        nn.init.zeros_(self.force_gates[-1].weight)
        nn.init.zeros_(self.force_gates[-1].bias)

    def forward(
        self,
        case: FloorplanCase,
        embedding: Tensor,
        node_geometry: Tensor,
        pair_features: Tensor,
        pair_mask: Tensor,
        step_fraction: float | Tensor,
    ) -> CollectiveStepOutput:
        if embedding.ndim != 2 or embedding.shape != (case.n, self.config.hidden_dim):
            raise ValueError("embedding must have shape [N, hidden_dim]")
        geometry = torch.as_tensor(
            node_geometry, device=embedding.device, dtype=embedding.dtype
        )
        if geometry.ndim != 3 or geometry.shape[1:] != (case.n, 3):
            raise ValueError("node_geometry must have shape [K,N,3]")
        population = geometry.shape[0]
        pairs = torch.as_tensor(
            pair_features, device=embedding.device, dtype=embedding.dtype
        )
        if pairs.shape != (population, case.n, case.n, len(PAIR_FEATURES)):
            raise ValueError("pair_features must have shape [K,N,N,19]")
        mask = torch.as_tensor(pair_mask, device=embedding.device)
        if mask.dtype != torch.bool or mask.shape != (case.n, case.n):
            raise ValueError("pair_mask must be boolean with shape [N,N]")
        if not bool(torch.isfinite(geometry).all() and torch.isfinite(pairs).all()):
            raise ValueError("collective inputs must be finite")

        fraction = torch.as_tensor(
            step_fraction, device=embedding.device, dtype=embedding.dtype
        )
        if fraction.numel() == 1:
            fraction = fraction.reshape(1, 1, 1).expand(population, case.n, 1)
        elif fraction.shape == (population,):
            fraction = fraction.reshape(population, 1, 1).expand(population, case.n, 1)
        else:
            raise ValueError("step_fraction must be scalar or shape [K]")

        hidden = self.static_node(embedding).unsqueeze(0)
        hidden = hidden + self.geometry_node(geometry) + self.step_node(fraction)
        pair_hidden = self.pair(pairs)
        active = mask.reshape(1, case.n, case.n, 1)
        denominator = mask.sum(dim=1).clamp_min(1).reshape(1, case.n, 1)
        for _ in range(self.config.collective_passes):
            messages = torch.nn.functional.silu(
                pair_hidden
                + self.receiver(hidden).unsqueeze(2)
                + self.sender(hidden).unsqueeze(1)
            )
            aggregate = (messages * active).sum(dim=2) / denominator
            hidden = hidden + self.update(self.update_norm(hidden + aggregate))

        raw_velocity = torch.tanh(self.velocity(hidden)).float()
        scale = raw_velocity.new_tensor(
            (
                self.config.collective_position_bound,
                self.config.collective_position_bound,
                self.config.collective_aspect_bound,
            )
        )
        velocity = raw_velocity * scale
        velocity[:, case.preplaced_mask.to(device=velocity.device), :2] = 0.0
        velocity[
            :, (case.fixed_mask | case.preplaced_mask).to(device=velocity.device), 2
        ] = 0.0
        force_gates = (
            1.0
            + self.config.collective_gate_delta
            * torch.tanh(self.force_gates(hidden)).float()
        )
        return CollectiveStepOutput(velocity=velocity, force_gates=force_gates)


class RepairAwareRanker(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.metric_dim = config.candidate_metric_dim
        self.use_scene_embedding = config.ranker_use_scene_embedding
        mean = config.ranker_feature_mean or (0.0,) * config.candidate_metric_dim
        scale = config.ranker_feature_scale or (1.0,) * config.candidate_metric_dim
        self.register_buffer(
            "feature_mean",
            torch.tensor(mean, dtype=torch.float32),
            persistent=False,
        )
        self.register_buffer(
            "feature_scale",
            torch.tensor(scale, dtype=torch.float32),
            persistent=False,
        )
        self.net = nn.Sequential(
            nn.Linear(
                config.candidate_metric_dim
                + (config.hidden_dim if config.ranker_use_scene_embedding else 0),
                config.hidden_dim,
            ),
            nn.SiLU(),
            nn.Linear(config.hidden_dim, 1),
        )

    def forward(
        self,
        embedding: Tensor,
        population: int,
        candidate_metrics: Tensor | None = None,
    ) -> Tensor:
        context = (
            embedding.mean(dim=0, keepdim=True).expand(population, -1)
            if self.use_scene_embedding
            else embedding.new_empty((population, 0))
        )
        if candidate_metrics is None:
            metrics = torch.zeros(population, self.metric_dim, device=embedding.device)
        else:
            metrics = torch.as_tensor(
                candidate_metrics, dtype=embedding.dtype, device=embedding.device
            )
            if metrics.shape != (population, self.metric_dim):
                raise ValueError(
                    "candidate_metrics shape does not match [population, candidate_metric_dim]"
                )
        metrics = (
            metrics - self.feature_mean.to(dtype=metrics.dtype)
        ) / self.feature_scale.to(dtype=metrics.dtype)
        return self.net(torch.cat((context, metrics), dim=1)).squeeze(1).float()


class ConstraintHeads(nn.Module):
    """Candidate-independent learned Q2 soft-constraint predictions."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.left = nn.Linear(config.hidden_dim, config.hidden_dim, bias=False)
        self.right = nn.Linear(config.hidden_dim, config.hidden_dim, bias=False)
        self.contact = nn.Linear(config.hidden_dim, 5)
        self.boundary = nn.Linear(config.hidden_dim, 4)
        self.mib = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.SiLU(),
            nn.Linear(config.hidden_dim, 1),
        )

    def forward(
        self,
        case: FloorplanCase,
        embedding: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        pair = torch.tanh(
            self.left(embedding)[:, None, :] + self.right(embedding)[None, :, :]
        )
        contact = self.contact(pair).float()
        boundary = self.boundary(embedding).float()
        if case.mib_membership.numel():
            membership = case.mib_membership.to(
                device=embedding.device, dtype=embedding.dtype
            )
            counts = membership.sum(dim=1, keepdim=True).clamp_min(1.0)
            pooled = membership @ embedding / counts
            mib = self.mib(pooled).squeeze(-1).float()
        else:
            mib = embedding.new_empty(
                (case.mib_membership.shape[0],), dtype=torch.float32
            )
        return contact, boundary, mib


class BTreeHeads(nn.Module):
    """Root and joint parent/branch scores for a hard B*-Tree decoder."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.child = nn.Linear(config.hidden_dim, config.hidden_dim, bias=False)
        self.parent = nn.Linear(config.hidden_dim, config.hidden_dim, bias=False)
        self.root = nn.Linear(config.hidden_dim, 1)
        self.edge = nn.Linear(config.hidden_dim, 2)

    def forward(self, embedding: Tensor) -> tuple[Tensor, Tensor]:
        pair = torch.tanh(
            self.child(embedding)[:, None, :] + self.parent(embedding)[None, :, :]
        )
        edge = self.edge(pair).float()
        diagonal = torch.eye(
            embedding.shape[0], dtype=torch.bool, device=embedding.device
        )
        edge[diagonal] = torch.finfo(edge.dtype).min
        return self.root(embedding).squeeze(-1).float(), edge


class BaselineHeads(nn.Module):
    """Case-level baseline area/HPWL estimates in normalized log space."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        aggregate_features = 10
        self.net = nn.Sequential(
            nn.Linear(config.hidden_dim + aggregate_features, config.hidden_dim),
            nn.SiLU(),
            nn.Linear(config.hidden_dim, 2),
        )

    def forward(self, case: FloorplanCase, embedding: Tensor) -> tuple[Tensor, Tensor]:
        device = embedding.device
        b2b = case.b2b_weight.to(device=device, dtype=embedding.dtype)
        p2b = case.p2b_edges.to(device=device, dtype=embedding.dtype)
        group = case.group_membership.to(device=device)
        mib = case.mib_membership.to(device=device)
        pin_count = float(case.pins.shape[0])
        features = embedding.new_tensor(
            (
                math.log1p(case.n),
                math.log1p(float(b2b.sum() * 0.5)),
                math.log1p(float(p2b[:, 2].sum())) if p2b.numel() else 0.0,
                math.log1p(float((b2b > 0).sum() * 0.5)),
                math.log1p(pin_count),
                float(case.fixed_mask.float().mean()),
                float(case.preplaced_mask.float().mean()),
                float(case.boundary_bits.any(dim=1).float().mean()),
                float(group.any(dim=0).float().mean()) if group.numel() else 0.0,
                float(mib.any(dim=0).float().mean()) if mib.numel() else 0.0,
            )
        )
        prediction = self.net(torch.cat((embedding.mean(dim=0), features)))
        return prediction[0].float(), prediction[1].float()


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
        if self.config.btree_enabled:
            self.btree = BTreeHeads(self.config)
        if self.config.constraint_enabled:
            self.constraints = ConstraintHeads(self.config)
        if self.config.collective_enabled:
            self.collective = GeometryAwareCollectiveHead(self.config)
        if self.config.baseline_enabled:
            self.baseline = BaselineHeads(self.config)

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
        with torch.autocast(
            device_type=device_type, dtype=torch.bfloat16, enabled=enabled
        ):
            embedding = self.encoder(case)
            precedence, outline = self.structure(case, embedding)
            positive: Tensor | None = None
            negative: Tensor | None = None
            if self.config.topology_enabled:
                positive, negative = self.topology(embedding, case.block_mask)
            btree_root: Tensor | None = None
            btree_edge: Tensor | None = None
            if self.config.btree_enabled:
                btree_root, btree_edge = self.btree(embedding)
            contact_logits: Tensor | None = None
            boundary_scores: Tensor | None = None
            mib_log_aspect: Tensor | None = None
            if self.config.constraint_enabled:
                contact_logits, boundary_scores, mib_log_aspect = self.constraints(
                    case, embedding
                )
            center, aspect = self.initializer(case, embedding, population)
            flow_velocity = self.flow(
                case, embedding, population, flow_state, flow_time
            )
            gates = self.gates(embedding, population, step_fraction)
            score = self.ranker(embedding, population, candidate_metrics)
            baseline_area: Tensor | None = None
            baseline_hpwl: Tensor | None = None
            if self.config.baseline_enabled:
                baseline_area, baseline_hpwl = self.baseline(case, embedding)
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
            contact_logits=(
                contact_logits.float() if contact_logits is not None else None
            ),
            boundary_order_scores=(
                boundary_scores.float() if boundary_scores is not None else None
            ),
            mib_log_aspect=(
                mib_log_aspect.float() if mib_log_aspect is not None else None
            ),
            btree_root_logits=(btree_root.float() if btree_root is not None else None),
            btree_edge_logits=(btree_edge.float() if btree_edge is not None else None),
            baseline_log_area=(
                baseline_area.float() if baseline_area is not None else None
            ),
            baseline_log_hpwl=(
                baseline_hpwl.float() if baseline_hpwl is not None else None
            ),
        )
