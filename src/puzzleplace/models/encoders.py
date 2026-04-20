from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from puzzleplace.data import FloorSetCase
from puzzleplace.roles import RoleLabel, WeakRoleEvidence


ROLE_TO_ID = {
    RoleLabel.FIXED_ANCHOR: 0,
    RoleLabel.PREPLACED_ANCHOR: 1,
    RoleLabel.BOUNDARY_SEEKER: 2,
    RoleLabel.CONNECTIVITY_HUB: 3,
    RoleLabel.CLUSTER_MEMBER: 4,
    RoleLabel.MULTI_INSTANCE: 5,
    RoleLabel.AREA_FOLLOWER: 6,
}


@dataclass(slots=True)
class EncoderOutput:
    block_embeddings: torch.Tensor
    graph_embedding: torch.Tensor
    block_mask: torch.Tensor


def _weighted_degrees(case: FloorSetCase) -> torch.Tensor:
    degrees = torch.zeros(case.block_count, dtype=torch.float32)
    for src, dst, weight in case.b2b_edges.tolist():
        i, j = int(src), int(dst)
        if i < case.block_count:
            degrees[i] += float(weight)
        if j < case.block_count:
            degrees[j] += float(weight)
    for _pin, block, weight in case.p2b_edges.tolist():
        j = int(block)
        if j < case.block_count:
            degrees[j] += float(weight)
    return degrees


def build_block_features(
    case: FloorSetCase,
    *,
    role_evidence: list[WeakRoleEvidence] | None = None,
    placements: dict[int, tuple[float, float, float, float]] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    placements = placements or {}
    degrees = _weighted_degrees(case)
    role_ids = torch.full((case.block_count,), ROLE_TO_ID[RoleLabel.AREA_FOLLOWER], dtype=torch.long)
    if role_evidence:
        for item in role_evidence:
            role_ids[item.block_index] = ROLE_TO_ID[item.role]

    placement_feats = torch.zeros((case.block_count, 5), dtype=torch.float32)
    for idx in range(case.block_count):
        if idx in placements:
            x, y, w, h = placements[idx]
            placement_feats[idx] = torch.tensor([1.0, float(x), float(y), float(w), float(h)])

    block_features = torch.cat(
        [
            case.area_targets.unsqueeze(-1).to(torch.float32),
            case.constraints.to(torch.float32),
            degrees.unsqueeze(-1),
            placement_feats,
        ],
        dim=-1,
    )
    return block_features, role_ids


class GraphStateEncoder(nn.Module):
    def __init__(self, hidden_dim: int = 128, role_vocab_size: int = 8, role_dim: int = 16):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.role_embedding = nn.Embedding(role_vocab_size, role_dim)
        self.input_proj = nn.Sequential(
            nn.Linear(12 + role_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.message_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def _aggregate_messages(self, case: FloorSetCase, block_embeddings: torch.Tensor) -> torch.Tensor:
        agg = torch.zeros_like(block_embeddings)
        for src, dst, weight in case.b2b_edges.tolist():
            i, j = int(src), int(dst)
            w = float(weight)
            if i < case.block_count and j < case.block_count:
                agg[i] += block_embeddings[j] * w
                agg[j] += block_embeddings[i] * w
        for _pin, block, weight in case.p2b_edges.tolist():
            j = int(block)
            w = float(weight)
            if j < case.block_count:
                agg[j] += block_embeddings[j] * w
        return agg

    def forward(
        self,
        case: FloorSetCase,
        *,
        role_evidence: list[WeakRoleEvidence] | None = None,
        placements: dict[int, tuple[float, float, float, float]] | None = None,
    ) -> EncoderOutput:
        block_features, role_ids = build_block_features(case, role_evidence=role_evidence, placements=placements)
        role_embed = self.role_embedding(role_ids)
        x = torch.cat([block_features, role_embed], dim=-1)
        block_embeddings = self.input_proj(x)
        messages = self._aggregate_messages(case, block_embeddings)
        block_embeddings = self.norm(block_embeddings + self.message_proj(messages))
        graph_embedding = block_embeddings.mean(dim=0)
        block_mask = torch.ones(case.block_count, dtype=torch.bool)
        return EncoderOutput(block_embeddings=block_embeddings, graph_embedding=graph_embedding, block_mask=block_mask)
