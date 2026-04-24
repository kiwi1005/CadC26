from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from puzzleplace.data import ConstraintColumns, FloorSetCase
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
    state_step: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    placements = placements or {}
    degrees = _weighted_degrees(case)
    block_indices = torch.arange(case.block_count, dtype=torch.float32)
    if case.block_count > 1:
        normalized_indices = block_indices / float(case.block_count - 1)
    else:
        normalized_indices = torch.zeros_like(block_indices)
    role_ids = torch.full((case.block_count,), ROLE_TO_ID[RoleLabel.AREA_FOLLOWER], dtype=torch.long)
    if role_evidence:
        for item in role_evidence:
            role_ids[item.block_index] = ROLE_TO_ID[item.role]

    placement_feats = torch.zeros((case.block_count, 5), dtype=torch.float32)
    for idx in range(case.block_count):
        if idx in placements:
            x, y, w, h = placements[idx]
            placement_feats[idx] = torch.tensor([1.0, float(x), float(y), float(w), float(h)])
    max_steps = max(case.block_count * 4, 1)
    normalized_step = 0.0 if state_step is None else max(0.0, float(state_step) / float(max_steps))
    step_features = torch.full((case.block_count, 1), normalized_step, dtype=torch.float32)

    block_features = torch.cat(
        [
            case.area_targets.unsqueeze(-1).to(torch.float32),
            case.constraints.to(torch.float32),
            degrees.unsqueeze(-1),
            normalized_indices.unsqueeze(-1),
            placement_feats,
            step_features,
        ],
        dim=-1,
    )
    return block_features, role_ids


class GraphStateEncoder(nn.Module):
    def __init__(self, hidden_dim: int = 128, role_vocab_size: int = 8, role_dim: int = 16):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.block_index_vocab_size = 1024
        self.block_index_dim = 8
        self.role_embedding = nn.Embedding(role_vocab_size, role_dim)
        self.block_index_embedding = nn.Embedding(self.block_index_vocab_size, self.block_index_dim)
        self.input_proj = nn.Sequential(
            nn.Linear(14 + self.block_index_dim + role_dim, hidden_dim),
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
        state_step: int | None = None,
    ) -> EncoderOutput:
        block_features, role_ids = build_block_features(
            case,
            role_evidence=role_evidence,
            placements=placements,
            state_step=state_step,
        )
        role_embed = self.role_embedding(role_ids)
        block_index = torch.arange(
            case.block_count, dtype=torch.long, device=block_features.device
        ) % self.block_index_vocab_size
        index_embed = self.block_index_embedding(block_index)
        x = torch.cat([block_features, role_embed, index_embed], dim=-1)
        block_embeddings = self.input_proj(x)
        messages = self._aggregate_messages(case, block_embeddings)
        block_embeddings = self.norm(block_embeddings + self.message_proj(messages))
        graph_embedding = block_embeddings.mean(dim=0)
        block_mask = torch.ones(case.block_count, dtype=torch.bool)
        return EncoderOutput(block_embeddings=block_embeddings, graph_embedding=graph_embedding, block_mask=block_mask)


def build_relation_aware_block_features(
    case: FloorSetCase,
    *,
    role_evidence: list[WeakRoleEvidence] | None = None,
    placements: dict[int, tuple[float, float, float, float]] | None = None,
    state_step: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build per-case/per-scale normalized state features for relation-aware encoders.

    The legacy encoder intentionally kept raw coordinates/areas as a minimal
    baseline. Step6C LOCO diagnostics showed that cross-case scale drift is now
    a first-order blocker, so this feature builder expresses geometry relative
    to each case's total-area scale before relation message passing.
    """

    placements = placements or {}
    total_area = max(float(case.area_targets.sum().item()), 1e-6)
    case_scale = max(total_area**0.5, 1e-6)
    degrees = _weighted_degrees(case)
    max_degree = max(float(degrees.abs().max().item()), 1.0)
    block_indices = torch.arange(case.block_count, dtype=torch.float32)
    if case.block_count > 1:
        normalized_indices = block_indices / float(case.block_count - 1)
    else:
        normalized_indices = torch.zeros_like(block_indices)
    role_ids = torch.full((case.block_count,), ROLE_TO_ID[RoleLabel.AREA_FOLLOWER], dtype=torch.long)
    if role_evidence:
        for item in role_evidence:
            role_ids[item.block_index] = ROLE_TO_ID[item.role]

    placement_feats = torch.zeros((case.block_count, 6), dtype=torch.float32)
    for idx in range(case.block_count):
        if idx in placements:
            x, y, w, h = placements[idx]
            placement_feats[idx] = torch.tensor(
                [
                    1.0,
                    float(x) / case_scale,
                    float(y) / case_scale,
                    float(w) / case_scale,
                    float(h) / case_scale,
                    float(w) * float(h) / total_area,
                ],
                dtype=torch.float32,
            )
    max_steps = max(case.block_count * 4, 1)
    normalized_step = 0.0 if state_step is None else max(0.0, float(state_step) / float(max_steps))
    step_features = torch.full((case.block_count, 1), normalized_step, dtype=torch.float32)
    block_features = torch.cat(
        [
            (case.area_targets / total_area).unsqueeze(-1).to(torch.float32),
            torch.sqrt(case.area_targets.clamp_min(0.0) / total_area).unsqueeze(-1).to(torch.float32),
            case.constraints.to(torch.float32),
            (degrees / max_degree).unsqueeze(-1),
            normalized_indices.unsqueeze(-1),
            placement_feats,
            step_features,
        ],
        dim=-1,
    )
    return block_features, role_ids


class RelationAwareGraphStateEncoder(nn.Module):
    """Typed relation message-passing encoder for Step6C learnability audits.

    This sidecar encoder keeps the public policy API unchanged while adding:
    - per-case/per-scale normalized block and placement features;
    - separate block-block and pin-block relation channels;
    - relation-aware graph pooling with placed/unplaced context preserved in
      node embeddings.
    """

    def __init__(self, hidden_dim: int = 128, role_vocab_size: int = 8, role_dim: int = 16):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.block_index_vocab_size = 1024
        self.block_index_dim = 8
        self.role_embedding = nn.Embedding(role_vocab_size, role_dim)
        self.block_index_embedding = nn.Embedding(self.block_index_vocab_size, self.block_index_dim)
        self.input_proj = nn.Sequential(
            nn.Linear(16 + self.block_index_dim + role_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.b2b_message_proj = nn.Sequential(
            nn.Linear(hidden_dim * 2 + 1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.p2b_message_proj = nn.Sequential(
            nn.Linear(hidden_dim + 3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.self_proj = nn.Linear(hidden_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.graph_proj = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def _aggregate_messages(self, case: FloorSetCase, block_embeddings: torch.Tensor) -> torch.Tensor:
        agg = torch.zeros_like(block_embeddings)
        max_b2b_weight = 1.0
        if case.b2b_edges.numel() > 0:
            max_b2b_weight = max(float(case.b2b_edges[:, 2].abs().max().item()), 1.0)
        for src, dst, weight in case.b2b_edges.tolist():
            i, j = int(src), int(dst)
            if not (0 <= i < case.block_count and 0 <= j < case.block_count):
                continue
            w = float(weight) / max_b2b_weight
            ij = torch.cat(
                [block_embeddings[i], block_embeddings[j], torch.tensor([w], dtype=torch.float32)]
            )
            ji = torch.cat(
                [block_embeddings[j], block_embeddings[i], torch.tensor([w], dtype=torch.float32)]
            )
            agg[i] += self.b2b_message_proj(ij)
            agg[j] += self.b2b_message_proj(ji)

        total_area = max(float(case.area_targets.sum().item()), 1e-6)
        case_scale = max(total_area**0.5, 1e-6)
        max_p2b_weight = 1.0
        if case.p2b_edges.numel() > 0:
            max_p2b_weight = max(float(case.p2b_edges[:, 2].abs().max().item()), 1.0)
        for pin, block, weight in case.p2b_edges.tolist():
            pin_idx, block_idx = int(pin), int(block)
            if not (0 <= block_idx < case.block_count and 0 <= pin_idx < case.pins_pos.shape[0]):
                continue
            px, py = [float(v) for v in case.pins_pos[pin_idx].tolist()]
            pin_features = torch.tensor(
                [px / case_scale, py / case_scale, float(weight) / max_p2b_weight],
                dtype=torch.float32,
            )
            agg[block_idx] += self.p2b_message_proj(
                torch.cat([block_embeddings[block_idx], pin_features])
            )
        return agg

    def forward(
        self,
        case: FloorSetCase,
        *,
        role_evidence: list[WeakRoleEvidence] | None = None,
        placements: dict[int, tuple[float, float, float, float]] | None = None,
        state_step: int | None = None,
    ) -> EncoderOutput:
        block_features, role_ids = build_relation_aware_block_features(
            case,
            role_evidence=role_evidence,
            placements=placements,
            state_step=state_step,
        )
        role_embed = self.role_embedding(role_ids)
        block_index = torch.arange(
            case.block_count, dtype=torch.long, device=block_features.device
        ) % self.block_index_vocab_size
        index_embed = self.block_index_embedding(block_index)
        x = torch.cat([block_features, role_embed, index_embed], dim=-1)
        block_embeddings = self.input_proj(x)
        messages = self._aggregate_messages(case, block_embeddings)
        block_embeddings = self.norm(self.self_proj(block_embeddings) + messages)
        graph_embedding = self.graph_proj(
            torch.cat([block_embeddings.mean(dim=0), block_embeddings.max(dim=0).values], dim=-1)
        )
        block_mask = torch.ones(case.block_count, dtype=torch.bool)
        return EncoderOutput(
            block_embeddings=block_embeddings,
            graph_embedding=graph_embedding,
            block_mask=block_mask,
        )

class TypedConstraintGraphStateEncoder(nn.Module):
    """Heterogeneous typed-constraint graph encoder for Step6E research.

    RelationAwareGraphStateEncoder already separates b2b and p2b edges, but it
    still leaves fixed/preplaced, boundary, cluster, and MIB constraints as flat
    per-block scalars. Step6E diagnostics showed scalar appends and tokenized
    scalar extras do not transfer. This encoder constructs explicit typed
    relation messages from the case/state graph: connectivity, pin anchors,
    same-cluster, same-MIB, boundary-side, and fixed/preplaced-anchor channels.
    """

    def __init__(
        self,
        hidden_dim: int = 128,
        role_vocab_size: int = 8,
        role_dim: int = 16,
        enabled_relation_ids: set[int] | None = None,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.enabled_relation_ids = set(range(6)) if enabled_relation_ids is None else set(enabled_relation_ids)
        self.block_index_vocab_size = 1024
        self.block_index_dim = 8
        self.role_embedding = nn.Embedding(role_vocab_size, role_dim)
        self.block_index_embedding = nn.Embedding(self.block_index_vocab_size, self.block_index_dim)
        self.input_proj = nn.Sequential(
            nn.Linear(16 + self.block_index_dim + role_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.relation_embedding = nn.Embedding(6, hidden_dim)
        self.edge_message_proj = nn.Sequential(
            nn.Linear(hidden_dim * 3 + 1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.node_update = nn.GRUCell(hidden_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.graph_proj = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    @staticmethod
    def _constraint_value(case: FloorSetCase, block_index: int, column: ConstraintColumns) -> float:
        return float(case.constraints[block_index, column].item())

    def _add_message(
        self,
        agg: torch.Tensor,
        block_embeddings: torch.Tensor,
        *,
        src: int,
        dst: int,
        relation_id: int,
        weight: float,
    ) -> None:
        if relation_id not in self.enabled_relation_ids:
            return
        relation = self.relation_embedding(
            torch.tensor(relation_id, dtype=torch.long, device=block_embeddings.device)
        )
        w = torch.tensor([float(weight)], dtype=torch.float32, device=block_embeddings.device)
        payload = torch.cat([block_embeddings[src], block_embeddings[dst], relation, w], dim=-1)
        agg[dst] += self.edge_message_proj(payload)

    def _aggregate_messages(self, case: FloorSetCase, block_embeddings: torch.Tensor) -> torch.Tensor:
        agg = torch.zeros_like(block_embeddings)
        max_b2b_weight = 1.0
        if case.b2b_edges.numel() > 0:
            max_b2b_weight = max(float(case.b2b_edges[:, 2].abs().max().item()), 1.0)
        for src, dst, weight in case.b2b_edges.tolist():
            i, j = int(src), int(dst)
            if 0 <= i < case.block_count and 0 <= j < case.block_count:
                w = float(weight) / max_b2b_weight
                self._add_message(agg, block_embeddings, src=i, dst=j, relation_id=0, weight=w)
                self._add_message(agg, block_embeddings, src=j, dst=i, relation_id=0, weight=w)

        max_p2b_weight = 1.0
        if case.p2b_edges.numel() > 0:
            max_p2b_weight = max(float(case.p2b_edges[:, 2].abs().max().item()), 1.0)
        for _pin, block, weight in case.p2b_edges.tolist():
            j = int(block)
            if 0 <= j < case.block_count:
                self._add_message(
                    agg,
                    block_embeddings,
                    src=j,
                    dst=j,
                    relation_id=1,
                    weight=float(weight) / max_p2b_weight,
                )

        cluster_groups: dict[float, list[int]] = {}
        mib_groups: dict[float, list[int]] = {}
        boundary_groups: dict[float, list[int]] = {}
        anchor_blocks: list[int] = []
        for idx in range(case.block_count):
            cluster = self._constraint_value(case, idx, ConstraintColumns.CLUSTER)
            mib = self._constraint_value(case, idx, ConstraintColumns.MIB)
            boundary = self._constraint_value(case, idx, ConstraintColumns.BOUNDARY)
            if cluster > 0:
                cluster_groups.setdefault(cluster, []).append(idx)
            if mib > 0:
                mib_groups.setdefault(mib, []).append(idx)
            if boundary > 0:
                boundary_groups.setdefault(boundary, []).append(idx)
            if (
                self._constraint_value(case, idx, ConstraintColumns.FIXED) > 0
                or self._constraint_value(case, idx, ConstraintColumns.PREPLACED) > 0
            ):
                anchor_blocks.append(idx)

        for relation_id, groups in ((2, cluster_groups), (3, mib_groups), (4, boundary_groups)):
            for members in groups.values():
                if len(members) < 2:
                    continue
                denom = max(len(members) - 1, 1)
                for src in members:
                    for dst in members:
                        if src != dst:
                            self._add_message(
                                agg,
                                block_embeddings,
                                src=src,
                                dst=dst,
                                relation_id=relation_id,
                                weight=1.0 / denom,
                            )

        if anchor_blocks:
            denom = max(len(anchor_blocks), 1)
            for anchor in anchor_blocks:
                for dst in range(case.block_count):
                    if dst != anchor:
                        self._add_message(
                            agg,
                            block_embeddings,
                            src=anchor,
                            dst=dst,
                            relation_id=5,
                            weight=1.0 / denom,
                        )
        return agg

    def forward(
        self,
        case: FloorSetCase,
        *,
        role_evidence: list[WeakRoleEvidence] | None = None,
        placements: dict[int, tuple[float, float, float, float]] | None = None,
        state_step: int | None = None,
    ) -> EncoderOutput:
        block_features, role_ids = build_relation_aware_block_features(
            case,
            role_evidence=role_evidence,
            placements=placements,
            state_step=state_step,
        )
        role_embed = self.role_embedding(role_ids)
        block_index = torch.arange(
            case.block_count, dtype=torch.long, device=block_features.device
        ) % self.block_index_vocab_size
        index_embed = self.block_index_embedding(block_index)
        hidden = self.input_proj(torch.cat([block_features, role_embed, index_embed], dim=-1))
        messages = self._aggregate_messages(case, hidden)
        updated = self.node_update(messages, hidden)
        block_embeddings = self.norm(hidden + updated)
        placed_mask = block_features[:, 10] > 0.5
        placed_pool = (
            block_embeddings[placed_mask].mean(dim=0)
            if bool(placed_mask.any())
            else torch.zeros(self.hidden_dim, dtype=block_embeddings.dtype, device=block_embeddings.device)
        )
        unplaced_pool = (
            block_embeddings[~placed_mask].mean(dim=0)
            if bool((~placed_mask).any())
            else torch.zeros(self.hidden_dim, dtype=block_embeddings.dtype, device=block_embeddings.device)
        )
        graph_embedding = self.graph_proj(
            torch.cat([block_embeddings.mean(dim=0), placed_pool, unplaced_pool], dim=-1)
        )
        block_mask = torch.ones(case.block_count, dtype=torch.bool)
        return EncoderOutput(
            block_embeddings=block_embeddings,
            graph_embedding=graph_embedding,
            block_mask=block_mask,
        )

