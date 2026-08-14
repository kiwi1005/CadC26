"""Dynamic Contact-only repair policy used by the P11.4 smoke gates."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from hcfp.case import BOUNDARY_BOTTOM, BOUNDARY_LEFT, BOUNDARY_RIGHT, BOUNDARY_TOP
from hcfp.model import SCENE_NODE_FEATURES, scene_node_features
from hcfp.repair.actions import action_sha256
from hcfp.repair.schema import ExpertKind, RepairAction, RepairObligation, RepairState


Tensor = torch.Tensor
CONTACT_RELATIONS = ("LEFT", "RIGHT", "TOP", "BOTTOM")
PATCH_BUDGETS = (2, 4, 8, 16)
_PAIR_FEATURES = 12
_DYNAMIC_NODE_FEATURES = 18
_SIDE_INDEX = {
    BOUNDARY_LEFT: 0,
    BOUNDARY_RIGHT: 1,
    BOUNDARY_TOP: 2,
    BOUNDARY_BOTTOM: 3,
}


@dataclass(frozen=True)
class RepairModelConfig:
    hidden_dim: int = 64
    encoder_layers: int = 2
    attention_heads: int = 4
    transformer_ffn_multiplier: int = 4

    def __post_init__(self) -> None:
        if (
            min(
                self.hidden_dim,
                self.encoder_layers,
                self.attention_heads,
                self.transformer_ffn_multiplier,
            )
            <= 0
        ):
            raise ValueError("repair model dimensions must be positive")
        if self.hidden_dim % self.attention_heads:
            raise ValueError("hidden_dim must be divisible by attention_heads")


@dataclass(frozen=True)
class ContactActionMasks:
    target: Tensor
    anchor: Tensor
    side: Tensor
    patch_budget: Tensor


@dataclass(frozen=True)
class ContactActionOutput:
    embedding: Tensor
    target_logits: Tensor
    anchor_logits: Tensor
    side_logits: Tensor
    patch_budget_logits: Tensor
    success_logits: Tensor
    debt_delta: Tensor
    masks: ContactActionMasks


class _PairBiasedBlock(nn.Module):
    def __init__(self, config: RepairModelConfig) -> None:
        super().__init__()
        width = config.hidden_dim
        self.heads = config.attention_heads
        self.head_dim = width // self.heads
        self.norm1 = nn.LayerNorm(width)
        self.qkv = nn.Linear(width, 3 * width, bias=False)
        self.pair_bias = nn.Linear(_PAIR_FEATURES, self.heads, bias=False)
        self.projection = nn.Linear(width, width, bias=False)
        self.norm2 = nn.LayerNorm(width)
        self.ffn = nn.Sequential(
            nn.Linear(width, width * config.transformer_ffn_multiplier),
            nn.SiLU(),
            nn.Linear(width * config.transformer_ffn_multiplier, width),
        )

    def forward(self, hidden: Tensor, pair_features: Tensor) -> Tensor:
        n, width = hidden.shape
        qkv = self.qkv(self.norm1(hidden)).reshape(n, 3, self.heads, self.head_dim)
        query, key, value = qkv.unbind(dim=1)
        query = query.permute(1, 0, 2).unsqueeze(0)
        key = key.permute(1, 0, 2).unsqueeze(0)
        value = value.permute(1, 0, 2).unsqueeze(0)
        bias = self.pair_bias(pair_features).permute(2, 0, 1).unsqueeze(0)
        attended = F.scaled_dot_product_attention(
            query, key, value, attn_mask=bias, dropout_p=0.0
        )
        attended = attended.squeeze(0).permute(1, 0, 2).reshape(n, width)
        hidden = hidden + self.projection(attended)
        return hidden + self.ffn(self.norm2(hidden))


class ContactRepairModel(nn.Module):
    """Encode mutable repair state and score factorized Contact actions."""

    def __init__(self, config: RepairModelConfig | None = None) -> None:
        super().__init__()
        self.config = config or RepairModelConfig()
        self.input = nn.Linear(
            SCENE_NODE_FEATURES + _DYNAMIC_NODE_FEATURES, self.config.hidden_dim
        )
        self.blocks = nn.ModuleList(
            _PairBiasedBlock(self.config) for _ in range(self.config.encoder_layers)
        )
        self.output_norm = nn.LayerNorm(self.config.hidden_dim)
        self.target = nn.Linear(self.config.hidden_dim, 1)
        self.left = nn.Linear(
            self.config.hidden_dim, self.config.hidden_dim, bias=False
        )
        self.right = nn.Linear(
            self.config.hidden_dim, self.config.hidden_dim, bias=False
        )
        self.anchor = nn.Linear(self.config.hidden_dim, 1)
        self.side = nn.Linear(self.config.hidden_dim, len(CONTACT_RELATIONS))
        self.patch_budget = nn.Sequential(
            nn.Linear(2 * self.config.hidden_dim, self.config.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.config.hidden_dim, len(PATCH_BUDGETS)),
        )
        self.value = nn.Linear(self.config.hidden_dim, 2)

    def encode(self, state: RepairState) -> Tensor:
        device = self.input.weight.device
        dtype = self.input.weight.dtype
        hidden = self.input(_node_features(state, device=device, dtype=dtype))
        pair_features = _pair_features(state, device=device, dtype=dtype)
        for block in self.blocks:
            hidden = block(hidden, pair_features)
        return self.output_norm(hidden).float()

    def forward(
        self, state: RepairState, obligation: RepairObligation
    ) -> ContactActionOutput:
        embedding = self.encode(state)
        masks = contact_action_masks(state, obligation, device=embedding.device)
        pair = torch.tanh(
            self.left(embedding)[:, None, :] + self.right(embedding)[None, :, :]
        )
        pooled = embedding.mean(dim=0).expand(state.case.n, state.case.n, -1)
        value = self.value(pair)
        return ContactActionOutput(
            embedding=embedding,
            target_logits=_masked(self.target(embedding).squeeze(-1), masks.target),
            anchor_logits=_masked(self.anchor(pair).squeeze(-1), masks.anchor),
            side_logits=_masked(self.side(pair), masks.side),
            patch_budget_logits=_masked(
                self.patch_budget(torch.cat((pair, pooled), dim=-1)), masks.patch_budget
            ),
            success_logits=value[..., 0],
            debt_delta=value[..., 1],
            masks=masks,
        )


def contact_action_masks(
    state: RepairState,
    obligation: RepairObligation,
    *,
    device: torch.device | str | None = None,
) -> ContactActionMasks:
    """Derive decoder-free Contact validity masks from one dynamic state."""

    if obligation.expert != ExpertKind.CONTACT:
        raise ValueError("Contact action masks require a Contact obligation")
    n = state.case.n
    target_device = (
        torch.device(device) if device is not None else state.placement.device
    )
    members = torch.zeros(n, dtype=torch.bool, device=target_device)
    member_ids = torch.tensor(
        obligation.target_ids, dtype=torch.long, device=target_device
    )
    if member_ids.numel() < 2 or bool((member_ids >= n).any()):
        raise ValueError("Contact obligation must contain two in-range members")
    members[member_ids] = True
    movable = state.position_mobility.to(device=target_device, dtype=torch.bool)
    observed = state.geometry_observed.to(device=target_device, dtype=torch.bool)
    target = members & movable & observed
    anchor = target[:, None] & members[None, :] & observed[None, :]
    anchor &= ~torch.eye(n, dtype=torch.bool, device=target_device)
    # Exact component size is decoder-owned: float32 state contacts can differ at
    # a zero-gap boundary from the float64 placement the exact decoder verifies.
    patch = anchor[..., None].expand(-1, -1, len(PATCH_BUDGETS)).clone()
    target &= patch.any(dim=(1, 2))
    anchor &= target[:, None]
    patch &= target[:, None, None]
    side = anchor[..., None].expand(-1, -1, len(CONTACT_RELATIONS))
    return ContactActionMasks(
        target=target, anchor=anchor, side=side, patch_budget=patch
    )


def topk_contact_actions(
    output: ContactActionOutput,
    obligation: RepairObligation,
    *,
    k: int,
) -> tuple[RepairAction, ...]:
    """Enumerate bounded factorized actions without constructing geometry."""

    if k <= 0:
        return ()
    target_log = F.log_softmax(output.target_logits, dim=0)
    actions: list[RepairAction] = []
    for target in (
        torch.nonzero(output.masks.target, as_tuple=False).reshape(-1).tolist()
    ):
        anchor_log = F.log_softmax(output.anchor_logits[target], dim=0)
        for anchor in (
            torch.nonzero(output.masks.anchor[target], as_tuple=False)
            .reshape(-1)
            .tolist()
        ):
            side_log = F.log_softmax(output.side_logits[target, anchor], dim=0)
            patch_log = F.log_softmax(output.patch_budget_logits[target, anchor], dim=0)
            for side in (
                torch.nonzero(output.masks.side[target, anchor], as_tuple=False)
                .reshape(-1)
                .tolist()
            ):
                for budget in (
                    torch.nonzero(
                        output.masks.patch_budget[target, anchor], as_tuple=False
                    )
                    .reshape(-1)
                    .tolist()
                ):
                    score = float(
                        target_log[target]
                        + anchor_log[anchor]
                        + side_log[side]
                        + patch_log[budget]
                    )
                    actions.append(
                        RepairAction(
                            expert=ExpertKind.CONTACT,
                            obligation_id=obligation.obligation_id,
                            target_ids=(target,),
                            anchor_ids=(anchor,),
                            relation=CONTACT_RELATIONS[side],
                            patch_budget=PATCH_BUDGETS[budget],
                            score=score,
                        )
                    )
    return tuple(
        sorted(actions, key=lambda action: (-action.score, action_sha256(action)))[:k]
    )


def _node_features(
    state: RepairState, *, device: torch.device, dtype: torch.dtype
) -> Tensor:
    case = state.case
    boxes = state.placement.to(device=device, dtype=dtype)
    centers = boxes[:, :2] + 0.5 * boxes[:, 2:4]
    component = state.group_component_id.to(device=device, dtype=dtype)
    component_size = _component_sizes(state.group_component_id).to(
        device=device, dtype=dtype
    )
    contact_degree = _contact_degree(state, device=device, dtype=dtype)
    missing = state.boundary_missing.to(device=device, dtype=torch.long)
    boundary = torch.stack(
        tuple((missing & bit != 0).to(dtype=dtype) for bit in _SIDE_INDEX), dim=1
    )
    dynamic = torch.cat(
        (
            centers,
            boxes[:, 2:4].clamp_min(1.0e-12).log(),
            state.geometry_observed.to(device=device, dtype=dtype).unsqueeze(1),
            state.repair_target.to(device=device, dtype=dtype).unsqueeze(1),
            state.position_mobility.to(device=device, dtype=dtype).unsqueeze(1),
            state.shape_mobility.to(device=device, dtype=dtype).unsqueeze(1),
            component.unsqueeze(1) / max(case.n, 1),
            component_size.unsqueeze(1) / max(case.n, 1),
            contact_degree.unsqueeze(1) / max(case.n - 1, 1),
            boundary,
            state.mib_shape_class.to(device=device, dtype=dtype).unsqueeze(1)
            / max(case.n, 1),
            boxes.new_full((case.n, 1), float(state.round_index) / 16.0),
            boxes.new_full((case.n, 1), float(state.corruption_level) / 4.0),
        ),
        dim=1,
    )
    if dynamic.shape[1] != _DYNAMIC_NODE_FEATURES:
        raise AssertionError("repair dynamic feature dimension drifted")
    return torch.cat(
        (scene_node_features(case).to(device=device, dtype=dtype), dynamic), dim=1
    )


def _pair_features(
    state: RepairState, *, device: torch.device, dtype: torch.dtype
) -> Tensor:
    case = state.case
    n = case.n
    weights = torch.log1p(case.b2b_weight.to(device=device, dtype=dtype))
    weights = weights / weights.amax().clamp_min(1.0)
    groups = _same_membership(case.group_membership, n, device=device, dtype=dtype)
    mib = _same_membership(case.mib_membership, n, device=device, dtype=dtype)
    component = state.group_component_id.to(device=device)
    same_component = (
        (component[:, None] >= 0)
        & (component[None, :] >= 0)
        & (component[:, None] == component[None, :])
    ).to(dtype=dtype)
    contact = torch.zeros((n, n, len(CONTACT_RELATIONS)), dtype=dtype, device=device)
    for first, second, first_side, second_side in state.contact_edges.tolist():
        contact[first, second, _SIDE_INDEX[int(first_side)]] = 1.0
        contact[second, first, _SIDE_INDEX[int(second_side)]] = 1.0
    boxes = state.placement.to(device=device, dtype=dtype)
    centers = boxes[:, :2] + 0.5 * boxes[:, 2:4]
    delta = centers[None, :, :] - centers[:, None, :]
    gap = delta.abs() - 0.5 * (boxes[:, None, 2:4] + boxes[None, :, 2:4])
    return torch.cat(
        (
            weights.unsqueeze(-1),
            groups.unsqueeze(-1),
            mib.unsqueeze(-1),
            same_component.unsqueeze(-1),
            contact,
            delta,
            gap,
        ),
        dim=-1,
    )


def _same_membership(
    membership: Tensor, n: int, *, device: torch.device, dtype: torch.dtype
) -> Tensor:
    if not membership.numel():
        return torch.zeros((n, n), dtype=dtype, device=device)
    active = membership.to(device=device, dtype=dtype)
    return (active.transpose(0, 1) @ active > 0).to(dtype=dtype)


def _contact_degree(
    state: RepairState, *, device: torch.device, dtype: torch.dtype
) -> Tensor:
    degree = torch.zeros(state.case.n, dtype=dtype, device=device)
    for first, second in state.contact_edges[:, :2].tolist():
        degree[first] += 1.0
        degree[second] += 1.0
    return degree


def _component_sizes(component_id: Tensor) -> Tensor:
    result = torch.zeros_like(component_id, dtype=torch.long)
    for component in torch.unique(component_id[component_id >= 0]).tolist():
        mask = component_id == component
        result[mask] = int(mask.sum())
    return result


def _masked(logits: Tensor, mask: Tensor) -> Tensor:
    return logits.masked_fill(~mask, torch.finfo(logits.dtype).min)


__all__ = [
    "CONTACT_RELATIONS",
    "PATCH_BUDGETS",
    "ContactActionMasks",
    "ContactActionOutput",
    "ContactRepairModel",
    "RepairModelConfig",
    "contact_action_masks",
    "topk_contact_actions",
]
