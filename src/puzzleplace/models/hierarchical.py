from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from puzzleplace.actions import ActionPrimitive
from puzzleplace.data import FloorSetCase
from puzzleplace.roles import WeakRoleEvidence

from .encoders import GraphStateEncoder, RelationAwareGraphStateEncoder, TypedConstraintGraphStateEncoder


@dataclass(slots=True)
class HierarchicalDecoderOutput:
    block_logits: torch.Tensor
    primitive_logits_by_block: torch.Tensor
    target_logits: torch.Tensor
    geometry: torch.Tensor


class HierarchicalSetPolicy(nn.Module):
    """Block-first policy prototype for Step6C learning-formulation tests.

    This intentionally does not replace the contest policy path. It gives the
    learnability audits a staged decision model: select a block first, then
    predict the primitive conditioned on that block embedding and the graph
    state.
    """

    def __init__(self, hidden_dim: int = 128, encoder_kind: str = "graph"):
        super().__init__()
        if encoder_kind == "relation_aware":
            self.encoder = RelationAwareGraphStateEncoder(hidden_dim=hidden_dim)
        elif encoder_kind == "typed_constraint_graph":
            self.encoder = TypedConstraintGraphStateEncoder(hidden_dim=hidden_dim)
        elif encoder_kind == "typed_constraint_graph_no_anchor":
            self.encoder = TypedConstraintGraphStateEncoder(
                hidden_dim=hidden_dim, enabled_relation_ids={0, 1, 2, 3, 4}
            )
        elif encoder_kind == "typed_constraint_graph_no_boundary":
            self.encoder = TypedConstraintGraphStateEncoder(
                hidden_dim=hidden_dim, enabled_relation_ids={0, 1, 2, 3, 5}
            )
        elif encoder_kind == "typed_constraint_graph_no_groups":
            self.encoder = TypedConstraintGraphStateEncoder(
                hidden_dim=hidden_dim, enabled_relation_ids={0, 1, 5}
            )
        elif encoder_kind == "graph":
            self.encoder = GraphStateEncoder(hidden_dim=hidden_dim)
        else:
            raise ValueError(f"unknown encoder_kind: {encoder_kind}")
        self.encoder_kind = encoder_kind
        self.block_head = nn.Linear(hidden_dim, 1)
        self.primitive_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, len(ActionPrimitive)),
        )
        self.target_query = nn.Linear(hidden_dim, hidden_dim)
        self.target_key = nn.Linear(hidden_dim, hidden_dim)
        self.geometry_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 4),
        )

    def forward(
        self,
        case: FloorSetCase,
        *,
        role_evidence: list[WeakRoleEvidence] | None = None,
        placements: dict[int, tuple[float, float, float, float]] | None = None,
        state_step: int | None = None,
    ) -> HierarchicalDecoderOutput:
        encoded = self.encoder(
            case,
            role_evidence=role_evidence,
            placements=placements,
            state_step=state_step,
        )
        block_embeddings = encoded.block_embeddings
        graph_embedding = encoded.graph_embedding
        graph_tiled = graph_embedding.unsqueeze(0).expand(block_embeddings.shape[0], -1)
        conditioned = torch.cat([block_embeddings, graph_tiled], dim=-1)
        return HierarchicalDecoderOutput(
            block_logits=self.block_head(block_embeddings).squeeze(-1),
            primitive_logits_by_block=self.primitive_head(conditioned),
            target_logits=self.target_query(block_embeddings)
            @ self.target_key(block_embeddings).transpose(0, 1),
            geometry=self.geometry_head(conditioned),
        )


class CandidateQualityRanker(nn.Module):
    """Small action-value head for Step6C candidate-pool quality experiments."""

    def __init__(self, feature_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features).squeeze(-1)


class CandidateComponentRanker(nn.Module):
    """Action-Q head with auxiliary objective-component logits.

    The overall head is still used for action selection. The component heads are
    trained against HPWL/area/violation pool oracles so the shared representation
    is not forced to compress all quality evidence into a single scalar target.
    """

    def __init__(self, feature_dim: int, hidden_dim: int = 64, component_count: int = 3):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.overall_head = nn.Linear(hidden_dim, 1)
        self.component_head = nn.Linear(hidden_dim, component_count)

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.trunk(features)
        return self.overall_head(hidden).squeeze(-1), self.component_head(hidden)


class CandidateSetPairwiseRanker(nn.Module):
    """Set-attentive pairwise comparator for candidate-pool ranking.

    This is an audit-sidecar model for Step6C. Candidate feature rows already
    include state/case geometry. The model first contextualizes all candidates
    in the same pool with self-attention, then predicts pairwise preferences.
    This avoids requiring absolute score calibration across unrelated cases.
    """

    def __init__(self, feature_dim: int, hidden_dim: int = 64, num_heads: int = 4):
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads")
        self.input_proj = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.self_attention = nn.MultiheadAttention(
            hidden_dim,
            num_heads=num_heads,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(hidden_dim)
        self.pair_head = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.scalar_head = nn.Linear(hidden_dim, 1)

    def encode(self, features: torch.Tensor) -> torch.Tensor:
        hidden = self.input_proj(features).unsqueeze(0)
        attended, _weights = self.self_attention(hidden, hidden, hidden, need_weights=False)
        return self.norm(hidden + attended).squeeze(0)

    def pair_logits(self, features: torch.Tensor) -> torch.Tensor:
        encoded = self.encode(features)
        lhs = encoded.unsqueeze(1).expand(-1, encoded.shape[0], -1)
        rhs = encoded.unsqueeze(0).expand(encoded.shape[0], -1, -1)
        pair_features = torch.cat([lhs, rhs, lhs - rhs, (lhs - rhs).abs()], dim=-1)
        return self.pair_head(pair_features).squeeze(-1)

    def scalar_scores(self, features: torch.Tensor) -> torch.Tensor:
        return self.scalar_head(self.encode(features)).squeeze(-1)

    def score_candidates(self, features: torch.Tensor) -> torch.Tensor:
        logits = self.pair_logits(features)
        if logits.shape[0] <= 1:
            return logits.sum(dim=1)
        mask = ~torch.eye(logits.shape[0], dtype=torch.bool, device=logits.device)
        win_probability = torch.sigmoid(logits)
        return (win_probability * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)

    def hybrid_scores(self, features: torch.Tensor, *, pairwise_weight: float = 0.5) -> torch.Tensor:
        scalar = self.scalar_scores(features)
        pairwise = self.score_candidates(features)
        scalar = (scalar - scalar.mean()) / scalar.std().clamp_min(1e-6)
        pairwise = (pairwise - pairwise.mean()) / pairwise.std().clamp_min(1e-6)
        weight = float(max(0.0, min(1.0, pairwise_weight)))
        return (1.0 - weight) * scalar + weight * pairwise


class CandidateRelationalActionQRanker(nn.Module):
    """Relation-aware action-Q ranker for candidate pools.

    This sidecar ranker is intentionally pool-local: it projects each candidate
    state/action feature row, contextualizes candidates with self-attention, and
    then performs typed pairwise message passing before producing an action-Q
    score. The pairwise messages make the score depend on how a candidate
    compares with the alternatives in the same state rather than on an absolute
    feature/logit scale across unrelated cases.
    """

    def __init__(self, feature_dim: int, hidden_dim: int = 64, num_heads: int = 4):
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads")
        self.input_proj = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.self_attention = nn.MultiheadAttention(
            hidden_dim,
            num_heads=num_heads,
            batch_first=True,
        )
        self.attn_norm = nn.LayerNorm(hidden_dim)
        pair_dim = hidden_dim * 4
        self.pair_gate = nn.Sequential(
            nn.Linear(pair_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.pair_value = nn.Sequential(
            nn.Linear(pair_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.pair_logit_head = nn.Sequential(
            nn.Linear(pair_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.context_norm = nn.LayerNorm(hidden_dim)
        self.score_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.component_head = nn.Linear(hidden_dim * 2, 3)

    def _pair_features(self, encoded: torch.Tensor) -> torch.Tensor:
        lhs = encoded.unsqueeze(1).expand(-1, encoded.shape[0], -1)
        rhs = encoded.unsqueeze(0).expand(encoded.shape[0], -1, -1)
        return torch.cat([lhs, rhs, lhs - rhs, (lhs - rhs).abs()], dim=-1)

    def encode(self, features: torch.Tensor) -> torch.Tensor:
        hidden = self.input_proj(features).unsqueeze(0)
        attended, _weights = self.self_attention(hidden, hidden, hidden, need_weights=False)
        encoded = self.attn_norm(hidden + attended).squeeze(0)
        if encoded.shape[0] <= 1:
            return encoded
        pair_features = self._pair_features(encoded)
        mask = ~torch.eye(encoded.shape[0], dtype=torch.bool, device=encoded.device)
        gates = self.pair_gate(pair_features).squeeze(-1).masked_fill(~mask, -1e9)
        weights = torch.softmax(gates, dim=1)
        values = self.pair_value(pair_features)
        pair_context = (weights.unsqueeze(-1) * values).sum(dim=1)
        return self.context_norm(encoded + pair_context)

    def pair_logits(self, features: torch.Tensor) -> torch.Tensor:
        encoded = self.encode(features)
        return self.pair_logit_head(self._pair_features(encoded)).squeeze(-1)

    def score_candidates(self, features: torch.Tensor) -> torch.Tensor:
        encoded = self.encode(features)
        graph_context = encoded.mean(dim=0, keepdim=True).expand(encoded.shape[0], -1)
        return self.score_head(torch.cat([encoded, graph_context], dim=-1)).squeeze(-1)

    def component_logits(self, features: torch.Tensor) -> torch.Tensor:
        encoded = self.encode(features)
        graph_context = encoded.mean(dim=0, keepdim=True).expand(encoded.shape[0], -1)
        return self.component_head(torch.cat([encoded, graph_context], dim=-1))


class CandidateConstraintTokenRanker(nn.Module):
    """Candidate ranker that treats constraint extras as typed relation tokens.

    The previous constraint-relation trial appended scalar features directly to
    each candidate row and regressed LOCO transfer. This sidecar keeps the same
    evidence surface but gives the final constraint-relation feature segment its
    own type embeddings and token attention before candidate-set attention.
    """

    def __init__(
        self,
        feature_dim: int,
        hidden_dim: int = 64,
        num_heads: int = 4,
        constraint_feature_count: int = 16,
    ):
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads")
        if feature_dim <= constraint_feature_count:
            raise ValueError("feature_dim must exceed constraint_feature_count")
        self.constraint_feature_count = constraint_feature_count
        base_dim = feature_dim - constraint_feature_count
        self.base_proj = nn.Sequential(
            nn.Linear(base_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.constraint_type_embedding = nn.Embedding(constraint_feature_count, hidden_dim)
        self.constraint_value_proj = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.constraint_attention = nn.MultiheadAttention(
            hidden_dim,
            num_heads=num_heads,
            batch_first=True,
        )
        self.constraint_norm = nn.LayerNorm(hidden_dim)
        self.candidate_proj = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.candidate_attention = nn.MultiheadAttention(
            hidden_dim,
            num_heads=num_heads,
            batch_first=True,
        )
        self.candidate_norm = nn.LayerNorm(hidden_dim)
        self.score_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.pair_head = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def encode(self, features: torch.Tensor) -> torch.Tensor:
        base = features[:, : -self.constraint_feature_count]
        constraint_values = features[:, -self.constraint_feature_count :]
        base_hidden = self.base_proj(base)
        type_ids = torch.arange(
            self.constraint_feature_count,
            dtype=torch.long,
            device=features.device,
        )
        type_tokens = self.constraint_type_embedding(type_ids).unsqueeze(0).expand(
            features.shape[0], -1, -1
        )
        value_tokens = self.constraint_value_proj(constraint_values.unsqueeze(-1))
        constraint_tokens = type_tokens + value_tokens
        constraint_context, _weights = self.constraint_attention(
            constraint_tokens,
            constraint_tokens,
            constraint_tokens,
            need_weights=False,
        )
        constraint_summary = self.constraint_norm(
            constraint_tokens + constraint_context
        ).mean(dim=1)
        candidate_hidden = self.candidate_proj(torch.cat([base_hidden, constraint_summary], dim=-1))
        attended, _weights = self.candidate_attention(
            candidate_hidden.unsqueeze(0),
            candidate_hidden.unsqueeze(0),
            candidate_hidden.unsqueeze(0),
            need_weights=False,
        )
        return self.candidate_norm(candidate_hidden + attended.squeeze(0))

    def score_candidates(self, features: torch.Tensor) -> torch.Tensor:
        encoded = self.encode(features)
        graph_context = encoded.mean(dim=0, keepdim=True).expand(encoded.shape[0], -1)
        return self.score_head(torch.cat([encoded, graph_context], dim=-1)).squeeze(-1)

    def pair_logits(self, features: torch.Tensor) -> torch.Tensor:
        encoded = self.encode(features)
        lhs = encoded.unsqueeze(1).expand(-1, encoded.shape[0], -1)
        rhs = encoded.unsqueeze(0).expand(encoded.shape[0], -1, -1)
        pair_features = torch.cat([lhs, rhs, lhs - rhs, (lhs - rhs).abs()], dim=-1)
        return self.pair_head(pair_features).squeeze(-1)


class CandidateLateFusionRanker(nn.Module):
    """Late-fuse independent scalar and pairwise candidate scorers."""

    def __init__(self, feature_dim: int, hidden_dim: int = 64, num_heads: int = 4):
        super().__init__()
        self.scalar_ranker = CandidateQualityRanker(feature_dim=feature_dim, hidden_dim=hidden_dim)
        self.pairwise_ranker = CandidateSetPairwiseRanker(
            feature_dim=feature_dim,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
        )

    def scalar_scores(self, features: torch.Tensor) -> torch.Tensor:
        return self.scalar_ranker(features)

    def pairwise_scores(self, features: torch.Tensor) -> torch.Tensor:
        return self.pairwise_ranker.score_candidates(features)

    def hybrid_scores(self, features: torch.Tensor, *, pairwise_weight: float = 0.5) -> torch.Tensor:
        scalar = self.scalar_scores(features)
        pairwise = self.pairwise_scores(features)
        scalar = (scalar - scalar.mean()) / scalar.std().clamp_min(1e-6)
        pairwise = (pairwise - pairwise.mean()) / pairwise.std().clamp_min(1e-6)
        weight = float(max(0.0, min(1.0, pairwise_weight)))
        return (1.0 - weight) * scalar + weight * pairwise
