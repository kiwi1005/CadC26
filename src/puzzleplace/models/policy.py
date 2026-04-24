from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from puzzleplace.data import FloorSetCase
from puzzleplace.roles import WeakRoleEvidence

from ..actions import ActionPrimitive
from .encoders import EncoderOutput, GraphStateEncoder


PRIMITIVE_TO_ID = {primitive: idx for idx, primitive in enumerate(ActionPrimitive)}
BOUNDARY_CLASS_COUNT = 5  # none, left, right, top, bottom


@dataclass(slots=True)
class DecoderOutput:
    primitive_logits: torch.Tensor
    block_logits: torch.Tensor
    target_logits: torch.Tensor
    boundary_logits: torch.Tensor
    geometry: torch.Tensor


class TypedActionDecoder(nn.Module):
    def __init__(self, hidden_dim: int = 128):
        super().__init__()
        self.primitive_head = nn.Linear(hidden_dim, len(ActionPrimitive))
        self.block_head = nn.Linear(hidden_dim, 1)
        self.target_query = nn.Linear(hidden_dim, hidden_dim)
        self.target_key = nn.Linear(hidden_dim, hidden_dim)
        self.boundary_head = nn.Linear(hidden_dim, BOUNDARY_CLASS_COUNT)
        self.geometry_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 4),
        )

    def forward(self, encoder_output: EncoderOutput) -> DecoderOutput:
        block_embeddings = encoder_output.block_embeddings
        graph_embedding = encoder_output.graph_embedding
        primitive_logits = self.primitive_head(graph_embedding)
        block_logits = self.block_head(block_embeddings).squeeze(-1)
        target_logits = self.target_query(block_embeddings) @ self.target_key(block_embeddings).transpose(0, 1)
        boundary_logits = self.boundary_head(block_embeddings)
        graph_tiled = graph_embedding.unsqueeze(0).expand(block_embeddings.shape[0], -1)
        geometry = self.geometry_head(torch.cat([block_embeddings, graph_tiled], dim=-1))
        return DecoderOutput(
            primitive_logits=primitive_logits,
            block_logits=block_logits,
            target_logits=target_logits,
            boundary_logits=boundary_logits,
            geometry=geometry,
        )


class TypedActionPolicy(nn.Module):
    def __init__(self, hidden_dim: int = 128):
        super().__init__()
        self.encoder = GraphStateEncoder(hidden_dim=hidden_dim)
        self.decoder = TypedActionDecoder(hidden_dim=hidden_dim)

    def forward(
        self,
        case: FloorSetCase,
        *,
        role_evidence: list[WeakRoleEvidence] | None = None,
        placements: dict[int, tuple[float, float, float, float]] | None = None,
        state_step: int | None = None,
    ) -> DecoderOutput:
        encoded = self.encoder(
            case,
            role_evidence=role_evidence,
            placements=placements,
            state_step=state_step,
        )
        return self.decoder(encoded)
