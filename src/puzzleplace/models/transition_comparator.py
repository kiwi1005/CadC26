from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import torch
from torch import nn

from puzzleplace.actions import ActionExecutor, ActionPrimitive, ExecutionState, TypedAction
from puzzleplace.data import ConstraintColumns, FloorSetCase
from puzzleplace.models.encoders import build_relation_aware_block_features
from puzzleplace.roles import WeakRoleEvidence


ALLOWED_TRANSITION_PAYLOAD_FIELDS = frozenset(
    {
        "pre_block_features",
        "pre_typed_edges",
        "post_block_features",
        "post_typed_edges",
        "action_token",
        "pairwise_majority_target",
    }
)

DENIED_TRANSITION_PAYLOAD_FIELDS = frozenset(
    {
        "candidate_features",
        "heuristic_features",
        "heuristic_score",
        "heuristic_scores",
        "raw_logits",
        "policy_logits",
        "quality_decomposition",
        "official_score_components",
        "case_id",
        "case_index",
        "hard_case_selector",
        "target_positions",
        "target_position",
        "manual_delta_rows",
        "handcrafted_delta_rows",
        "action_delta_features",
        "delta_features",
    }
)

TRANSITION_RELATION_COUNT = 7


@dataclass(frozen=True, slots=True)
class Step6FTransitionPayload:
    """Legal Step6F candidate payload.

    The payload intentionally contains only graph state tensors, a compact
    action token, and the pairwise-majority supervision row.  It does not carry
    candidate feature rows, heuristic scores, target positions, case ids, raw
    logits, or hand-authored before/after deltas.
    """

    pre_block_features: list[list[float]]
    pre_typed_edges: list[list[float]]
    post_block_features: list[list[float]]
    post_typed_edges: list[list[float]]
    action_token: list[float]
    pairwise_majority_target: list[float]

    def to_mapping(self) -> dict[str, Any]:
        return {
            "pre_block_features": self.pre_block_features,
            "pre_typed_edges": self.pre_typed_edges,
            "post_block_features": self.post_block_features,
            "post_typed_edges": self.post_typed_edges,
            "action_token": self.action_token,
            "pairwise_majority_target": self.pairwise_majority_target,
        }

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "Step6FTransitionPayload":
        validated = validate_transition_payload(payload)
        return cls(
            pre_block_features=validated["pre_block_features"].tolist(),
            pre_typed_edges=validated["pre_typed_edges"].tolist(),
            post_block_features=validated["post_block_features"].tolist(),
            post_typed_edges=validated["post_typed_edges"].tolist(),
            action_token=validated["action_token"].tolist(),
            pairwise_majority_target=validated["pairwise_majority_target"].tolist(),
        )


def _clone_state(state: ExecutionState) -> ExecutionState:
    return ExecutionState(
        placements=dict(state.placements),
        frozen_blocks=set(state.frozen_blocks),
        proposed_positions=dict(state.proposed_positions),
        shape_assigned=set(state.shape_assigned),
        semantic_placed=set(state.semantic_placed),
        physically_placed=set(state.physically_placed),
        step=state.step,
        history=list(state.history),
        last_rollout_mode=state.last_rollout_mode,
    )


def _placements_for_features(state: ExecutionState) -> dict[int, tuple[float, float, float, float]]:
    return dict(state.proposed_positions or state.placements)


def _constraint_value(case: FloorSetCase, block_index: int, column: ConstraintColumns) -> float:
    return float(case.constraints[block_index, column].item())


def build_transition_typed_edges(
    case: FloorSetCase,
    placements: Mapping[int, tuple[float, float, float, float]] | None = None,
) -> list[list[float]]:
    """Build typed graph edges for a pre/post state without target-position access.

    Relation ids are stable:
    0 block-block connectivity, 1 pin-block self edge, 2 same cluster,
    3 same MIB, 4 same boundary class, 5 fixed/preplaced anchor context,
    6 currently placed block context.  The final relation lets the shared graph
    encoder observe the post-state graph structure after a candidate action
    without introducing a manually authored delta row.
    """

    placements = placements or {}
    edges: list[list[float]] = []

    if case.b2b_edges.numel() > 0:
        denom = max(float(case.b2b_edges[:, 2].abs().max().item()), 1.0)
        for src, dst, weight in case.b2b_edges.tolist():
            i, j = int(src), int(dst)
            if 0 <= i < case.block_count and 0 <= j < case.block_count:
                w = float(weight) / denom
                edges.append([float(i), float(j), 0.0, w])
                edges.append([float(j), float(i), 0.0, w])

    if case.p2b_edges.numel() > 0:
        denom = max(float(case.p2b_edges[:, 2].abs().max().item()), 1.0)
        for _pin, block, weight in case.p2b_edges.tolist():
            j = int(block)
            if 0 <= j < case.block_count:
                edges.append([float(j), float(j), 1.0, float(weight) / denom])

    grouped: dict[tuple[int, float], list[int]] = {}
    anchors: list[int] = []
    for idx in range(case.block_count):
        cluster = _constraint_value(case, idx, ConstraintColumns.CLUSTER)
        mib = _constraint_value(case, idx, ConstraintColumns.MIB)
        boundary = _constraint_value(case, idx, ConstraintColumns.BOUNDARY)
        if cluster > 0:
            grouped.setdefault((2, cluster), []).append(idx)
        if mib > 0:
            grouped.setdefault((3, mib), []).append(idx)
        if boundary > 0:
            grouped.setdefault((4, boundary), []).append(idx)
        if (
            _constraint_value(case, idx, ConstraintColumns.FIXED) > 0
            or _constraint_value(case, idx, ConstraintColumns.PREPLACED) > 0
        ):
            anchors.append(idx)

    for (relation_id, _value), members in grouped.items():
        if len(members) < 2:
            continue
        denom = max(len(members) - 1, 1)
        for src in members:
            for dst in members:
                if src != dst:
                    edges.append([float(src), float(dst), float(relation_id), 1.0 / denom])

    if anchors:
        denom = max(len(anchors), 1)
        for src in anchors:
            for dst in range(case.block_count):
                if src != dst:
                    edges.append([float(src), float(dst), 5.0, 1.0 / denom])

    placed = sorted(int(idx) for idx in placements if 0 <= int(idx) < case.block_count)
    if len(placed) >= 2:
        denom = max(len(placed) - 1, 1)
        for src in placed:
            for dst in placed:
                if src != dst:
                    edges.append([float(src), float(dst), 6.0, 1.0 / denom])

    return edges


def _action_token(action: TypedAction, block_count: int) -> list[float]:
    primitive_id = list(ActionPrimitive).index(action.primitive)
    target_index = -1 if action.target_index is None else int(action.target_index)
    return [
        float(int(action.block_index)),
        float(target_index),
        float(primitive_id),
        0.0 if action.target_index is None else 1.0,
    ]


def build_transition_payload(
    case: FloorSetCase,
    state: ExecutionState,
    action: TypedAction,
    *,
    role_evidence: list[WeakRoleEvidence] | None = None,
    pairwise_majority_target: list[float] | None = None,
) -> Step6FTransitionPayload:
    """Create the legal pre/post graph payload for one candidate action."""

    pre_placements = _placements_for_features(state)
    pre_features, _pre_role_ids = build_relation_aware_block_features(
        case,
        role_evidence=role_evidence,
        placements=pre_placements,
        state_step=state.step,
    )
    pre_edges = build_transition_typed_edges(case, pre_placements)

    trial = _clone_state(state)
    try:
        ActionExecutor(case).apply(trial, action)
    except Exception as exc:  # pragma: no cover - defensive context for research runner
        raise ValueError(f"candidate action cannot build a post graph: {exc}") from exc

    post_placements = _placements_for_features(trial)
    post_features, _post_role_ids = build_relation_aware_block_features(
        case,
        role_evidence=role_evidence,
        placements=post_placements,
        state_step=trial.step,
    )
    post_edges = build_transition_typed_edges(case, post_placements)
    if pre_features.shape != post_features.shape:
        raise ValueError(
            f"pre/post block feature shape mismatch: {tuple(pre_features.shape)} "
            f"!= {tuple(post_features.shape)}"
        )
    if torch.equal(pre_features, post_features) and pre_edges == post_edges:
        raise ValueError("post graph is identical to pre graph; refusing silent fallback")

    payload = Step6FTransitionPayload(
        pre_block_features=pre_features.tolist(),
        pre_typed_edges=pre_edges,
        post_block_features=post_features.tolist(),
        post_typed_edges=post_edges,
        action_token=_action_token(action, case.block_count),
        pairwise_majority_target=list(pairwise_majority_target or []),
    )
    # Round-trip through the fail-closed validator so caller mistakes are caught
    # before the payload reaches the model/training path.
    return Step6FTransitionPayload.from_mapping(payload.to_mapping())


def _tensor_2d(value: Any, *, name: str, width: int | None = None) -> torch.Tensor:
    tensor = torch.as_tensor(value, dtype=torch.float32)
    if tensor.ndim != 2:
        raise ValueError(f"{name} must be a 2D tensor-like value")
    if width is not None and tensor.shape[1] != width:
        raise ValueError(f"{name} must have width {width}, got {tensor.shape[1]}")
    return tensor


def _tensor_1d(value: Any, *, name: str) -> torch.Tensor:
    tensor = torch.as_tensor(value, dtype=torch.float32)
    if tensor.ndim == 0:
        tensor = tensor.unsqueeze(0)
    if tensor.ndim != 1:
        raise ValueError(f"{name} must be a 1D tensor-like value")
    return tensor


def validate_transition_payload(payload: Mapping[str, Any]) -> dict[str, torch.Tensor]:
    keys = set(payload)
    denied = sorted(keys & DENIED_TRANSITION_PAYLOAD_FIELDS)
    if denied:
        raise ValueError(f"transition payload contains denied field(s): {', '.join(denied)}")
    missing = sorted(ALLOWED_TRANSITION_PAYLOAD_FIELDS - keys)
    extra = sorted(keys - ALLOWED_TRANSITION_PAYLOAD_FIELDS)
    if missing or extra:
        details = []
        if missing:
            details.append(f"missing={missing}")
        if extra:
            details.append(f"extra={extra}")
        raise ValueError("transition payload schema mismatch: " + "; ".join(details))

    pre_features = _tensor_2d(payload["pre_block_features"], name="pre_block_features")
    post_features = _tensor_2d(payload["post_block_features"], name="post_block_features")
    if pre_features.shape != post_features.shape:
        raise ValueError(
            f"pre/post block feature shape mismatch: {tuple(pre_features.shape)} "
            f"!= {tuple(post_features.shape)}"
        )
    pre_edges = _tensor_2d(payload["pre_typed_edges"], name="pre_typed_edges", width=4)
    post_edges = _tensor_2d(payload["post_typed_edges"], name="post_typed_edges", width=4)
    action = _tensor_1d(payload["action_token"], name="action_token")
    target = _tensor_1d(payload["pairwise_majority_target"], name="pairwise_majority_target")
    return {
        "pre_block_features": pre_features,
        "pre_typed_edges": pre_edges,
        "post_block_features": post_features,
        "post_typed_edges": post_edges,
        "action_token": action,
        "pairwise_majority_target": target,
    }


class TransitionGraphEncoder(nn.Module):
    """Tensor-native typed graph encoder used with shared pre/post weights."""

    def __init__(
        self,
        block_feature_dim: int,
        hidden_dim: int = 64,
        relation_count: int = TRANSITION_RELATION_COUNT,
    ):
        super().__init__()
        self.block_feature_dim = int(block_feature_dim)
        self.hidden_dim = int(hidden_dim)
        self.relation_count = int(relation_count)
        self.node_proj = nn.Sequential(
            nn.Linear(self.block_feature_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.relation_embedding = nn.Embedding(relation_count, hidden_dim)
        self.message_proj = nn.Sequential(
            nn.Linear(hidden_dim * 3 + 1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.node_norm = nn.LayerNorm(hidden_dim)
        self.graph_proj = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, block_features: torch.Tensor, typed_edges: torch.Tensor) -> torch.Tensor:
        nodes = self.node_proj(block_features)
        if typed_edges.numel() > 0:
            agg = torch.zeros_like(nodes)
            for src, dst, relation_id, weight in typed_edges.tolist():
                i, j, rel = int(src), int(dst), int(relation_id)
                if not (0 <= i < nodes.shape[0] and 0 <= j < nodes.shape[0]):
                    continue
                if not 0 <= rel < self.relation_count:
                    raise ValueError(f"relation id {rel} outside relation_count={self.relation_count}")
                relation = self.relation_embedding(
                    torch.tensor(rel, dtype=torch.long, device=nodes.device)
                )
                w = torch.tensor([float(weight)], dtype=torch.float32, device=nodes.device)
                message = self.message_proj(torch.cat([nodes[i], nodes[j], relation, w], dim=-1))
                agg[j] += message
            nodes = self.node_norm(nodes + agg)
        return self.graph_proj(torch.cat([nodes.mean(dim=0), nodes.max(dim=0).values], dim=-1))


class SharedEncoderTransitionComparator(nn.Module):
    """Step6F shared-encoder pre/post transition comparator.

    A single ``TransitionGraphEncoder`` instance is reused for both pre and post
    graphs.  The model interface is intentionally restricted to the legal
    transition payload schema; candidate feature rows, target positions, raw
    logits, heuristic scores, quality components, case ids, and manual delta
    rows are rejected before scoring.
    """

    allowed_payload_fields = ALLOWED_TRANSITION_PAYLOAD_FIELDS
    denied_payload_fields = DENIED_TRANSITION_PAYLOAD_FIELDS

    def __init__(
        self,
        *,
        block_feature_dim: int,
        action_token_dim: int = 4,
        hidden_dim: int = 64,
        relation_count: int = TRANSITION_RELATION_COUNT,
    ):
        super().__init__()
        self.encoder = TransitionGraphEncoder(
            block_feature_dim=block_feature_dim,
            hidden_dim=hidden_dim,
            relation_count=relation_count,
        )
        self.action_proj = nn.Sequential(
            nn.Linear(action_token_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.transition_proj = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.pair_head = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    @staticmethod
    def model_input_keys() -> tuple[str, ...]:
        return (
            "pre_block_features",
            "pre_typed_edges",
            "post_block_features",
            "post_typed_edges",
            "action_token",
        )

    def encode_payload(self, payload: Mapping[str, Any] | Step6FTransitionPayload) -> torch.Tensor:
        raw_payload = payload.to_mapping() if isinstance(payload, Step6FTransitionPayload) else payload
        validated = validate_transition_payload(raw_payload)
        pre_graph = self.encoder(
            validated["pre_block_features"],
            validated["pre_typed_edges"],
        )
        post_graph = self.encoder(
            validated["post_block_features"],
            validated["post_typed_edges"],
        )
        action = self.action_proj(validated["action_token"])
        return self.transition_proj(torch.cat([pre_graph, post_graph, action], dim=-1))

    def encode_pool(
        self, payloads: list[Mapping[str, Any] | Step6FTransitionPayload]
    ) -> torch.Tensor:
        if not payloads:
            raise ValueError("transition comparator requires at least one payload")
        return torch.stack([self.encode_payload(payload) for payload in payloads], dim=0)

    def pair_logits(
        self, payloads: list[Mapping[str, Any] | Step6FTransitionPayload]
    ) -> torch.Tensor:
        encoded = self.encode_pool(payloads)
        lhs = encoded.unsqueeze(1).expand(-1, encoded.shape[0], -1)
        rhs = encoded.unsqueeze(0).expand(encoded.shape[0], -1, -1)
        pair_features = torch.cat([lhs, rhs, lhs - rhs, (lhs - rhs).abs()], dim=-1)
        return self.pair_head(pair_features).squeeze(-1)

    def score_candidates(
        self, payloads: list[Mapping[str, Any] | Step6FTransitionPayload]
    ) -> torch.Tensor:
        logits = self.pair_logits(payloads)
        if logits.shape[0] <= 1:
            return logits.sum(dim=1)
        mask = ~torch.eye(logits.shape[0], dtype=torch.bool, device=logits.device)
        wins = torch.sigmoid(logits).masked_fill(~mask, 0.0)
        return wins.sum(dim=1) / mask.sum(dim=1).clamp_min(1)
