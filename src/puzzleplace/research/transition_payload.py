from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import torch
from torch import nn

from puzzleplace.actions.executor import ActionExecutor, ExecutionState
from puzzleplace.actions.schema import ActionPrimitive, TypedAction
from puzzleplace.data import ConstraintColumns, FloorSetCase
from puzzleplace.models.encoders import build_relation_aware_block_features

TransitionPayload = dict[str, Any]

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
        "heuristic_scores",
        "heuristics",
        "raw_logits",
        "logits",
        "quality_decomposition",
        "quality_components",
        "official_score_components",
        "case_id",
        "case_index",
        "hard_case_selector",
        "hard_case_id",
        "hard_case_ids",
        "target_positions",
        "manual_delta_rows",
        "handcrafted_delta_rows",
        "relation_delta_rows",
        "delta_rows",
        "before_after_delta",
    }
)


def _scan_denied_keys(value: Any, *, path: str = "payload") -> list[str]:
    denied: list[str] = []
    if isinstance(value, Mapping):
        for key, nested in value.items():
            key_text = str(key)
            nested_path = f"{path}.{key_text}"
            if key_text in DENIED_TRANSITION_PAYLOAD_FIELDS:
                denied.append(nested_path)
            denied.extend(_scan_denied_keys(nested, path=nested_path))
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for index, nested in enumerate(value):
            denied.extend(_scan_denied_keys(nested, path=f"{path}[{index}]"))
    return denied


def validate_transition_payload(payload: Mapping[str, Any]) -> None:
    """Fail closed on Step6F transition-comparator payload contract violations."""

    keys = set(payload.keys())
    denied = sorted(keys & DENIED_TRANSITION_PAYLOAD_FIELDS)
    nested_denied = sorted(_scan_denied_keys(payload))
    if denied or nested_denied:
        all_denied = sorted(set(denied + nested_denied))
        raise ValueError(f"denied transition payload fields: {', '.join(all_denied)}")

    missing = sorted(ALLOWED_TRANSITION_PAYLOAD_FIELDS - keys)
    if missing:
        raise ValueError(f"missing required transition payload fields: {', '.join(missing)}")

    extras = sorted(keys - ALLOWED_TRANSITION_PAYLOAD_FIELDS)
    if extras:
        raise ValueError(f"unsupported transition payload fields: {', '.join(extras)}")

    pre_block = payload["pre_block_features"]
    post_block = payload["post_block_features"]
    if not isinstance(pre_block, torch.Tensor) or not isinstance(post_block, torch.Tensor):
        raise TypeError("pre_block_features and post_block_features must be torch.Tensor")
    if pre_block.ndim != 2 or post_block.ndim != 2:
        raise ValueError("pre/post block features must be rank-2 tensors")
    if pre_block.shape != post_block.shape:
        raise ValueError("pre/post block features must have identical shape")

    pre_edges = payload["pre_typed_edges"]
    post_edges = payload["post_typed_edges"]
    if not isinstance(pre_edges, torch.Tensor) or not isinstance(post_edges, torch.Tensor):
        raise TypeError("pre_typed_edges and post_typed_edges must be torch.Tensor")
    if pre_edges.ndim != 2 or post_edges.ndim != 2 or pre_edges.shape[1] != 4 or post_edges.shape[1] != 4:
        raise ValueError("pre/post typed edges must be rank-2 tensors with columns relation_id, src, dst, weight")
    if pre_edges.shape[1] != post_edges.shape[1]:
        raise ValueError("pre/post typed edge schemas must match")

    action_token = payload["action_token"]
    if not isinstance(action_token, torch.Tensor):
        raise TypeError("action_token must be a torch.Tensor")
    if action_token.numel() != 4:
        raise ValueError("action_token must contain block_index, target_index, primitive_id, has_target")


def build_typed_edges(case: FloorSetCase) -> torch.Tensor:
    """Build stable typed-relation edge rows without target-position or case-id data."""

    rows: list[tuple[float, float, float, float]] = []
    max_b2b_weight = 1.0
    if case.b2b_edges.numel() > 0:
        max_b2b_weight = max(float(case.b2b_edges[:, 2].abs().max().item()), 1.0)
    for src, dst, weight in case.b2b_edges.tolist():
        i, j = int(src), int(dst)
        if 0 <= i < case.block_count and 0 <= j < case.block_count:
            w = float(weight) / max_b2b_weight
            rows.append((0.0, float(i), float(j), w))
            rows.append((0.0, float(j), float(i), w))

    max_p2b_weight = 1.0
    if case.p2b_edges.numel() > 0:
        max_p2b_weight = max(float(case.p2b_edges[:, 2].abs().max().item()), 1.0)
    for _pin, block, weight in case.p2b_edges.tolist():
        j = int(block)
        if 0 <= j < case.block_count:
            rows.append((1.0, float(j), float(j), float(weight) / max_p2b_weight))

    cluster_groups: dict[float, list[int]] = {}
    mib_groups: dict[float, list[int]] = {}
    boundary_groups: dict[float, list[int]] = {}
    anchor_blocks: list[int] = []
    for idx in range(case.block_count):
        cluster = float(case.constraints[idx, ConstraintColumns.CLUSTER].item())
        mib = float(case.constraints[idx, ConstraintColumns.MIB].item())
        boundary = float(case.constraints[idx, ConstraintColumns.BOUNDARY].item())
        if cluster > 0:
            cluster_groups.setdefault(cluster, []).append(idx)
        if mib > 0:
            mib_groups.setdefault(mib, []).append(idx)
        if boundary > 0:
            boundary_groups.setdefault(boundary, []).append(idx)
        if (
            float(case.constraints[idx, ConstraintColumns.FIXED].item()) > 0
            or float(case.constraints[idx, ConstraintColumns.PREPLACED].item()) > 0
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
                        rows.append((float(relation_id), float(src), float(dst), 1.0 / denom))

    if anchor_blocks:
        denom = max(len(anchor_blocks), 1)
        for anchor in anchor_blocks:
            for dst in range(case.block_count):
                if dst != anchor:
                    rows.append((5.0, float(anchor), float(dst), 1.0 / denom))

    if not rows:
        return torch.empty((0, 4), dtype=torch.float32)
    return torch.tensor(rows, dtype=torch.float32)


def _action_token(action: TypedAction) -> torch.Tensor:
    primitive_id = list(ActionPrimitive).index(action.primitive)
    target_index = -1 if action.target_index is None else int(action.target_index)
    return torch.tensor(
        [
            int(action.block_index),
            target_index,
            primitive_id,
            1 if action.target_index is not None else 0,
        ],
        dtype=torch.float32,
    )


def build_transition_payload(
    case: FloorSetCase,
    action: TypedAction,
    *,
    placements: Mapping[int, tuple[float, float, float, float]] | None = None,
    state_step: int = 0,
    pairwise_majority_target: int | float | torch.Tensor = 0,
) -> TransitionPayload:
    """Build the Step6F legal pre/post payload for one candidate action.

    The builder applies the candidate through the existing action executor. If a
    candidate cannot be applied, it raises instead of silently reusing the pre
    graph as the post graph.
    """

    if action.metadata:
        denied = _scan_denied_keys(action.metadata, path="action.metadata")
        if denied:
            raise ValueError(f"denied transition action metadata fields: {', '.join(sorted(denied))}")

    pre_placements = dict(placements or {})
    pre_block_features, _roles = build_relation_aware_block_features(
        case,
        placements=pre_placements,
        state_step=state_step,
    )
    pre_typed_edges = build_typed_edges(case)

    state = ExecutionState(placements=dict(pre_placements), step=int(state_step))
    try:
        post_state = ActionExecutor(case).apply(state, action)
    except Exception as exc:  # noqa: BLE001 - convert all executor failures into contract failure.
        raise ValueError(f"invalid transition candidate: {exc}") from exc

    post_block_features, _post_roles = build_relation_aware_block_features(
        case,
        placements=post_state.placements,
        state_step=post_state.step,
    )
    post_typed_edges = build_typed_edges(case)

    target = (
        pairwise_majority_target.detach().clone()
        if isinstance(pairwise_majority_target, torch.Tensor)
        else torch.tensor(pairwise_majority_target, dtype=torch.float32)
    )
    payload: TransitionPayload = {
        "pre_block_features": pre_block_features,
        "pre_typed_edges": pre_typed_edges,
        "post_block_features": post_block_features,
        "post_typed_edges": post_typed_edges,
        "action_token": _action_token(action),
        "pairwise_majority_target": target,
    }
    validate_transition_payload(payload)
    return payload


class SharedEncoderTransitionComparator(nn.Module):
    """Minimal shared-encoder comparator for Step6F contract and smoke tests."""

    def __init__(self, block_feature_dim: int, hidden_dim: int = 64, action_token_dim: int = 4):
        super().__init__()
        self.graph_encoder = nn.Sequential(
            nn.Linear(block_feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.pre_encoder = self.graph_encoder
        self.post_encoder = self.graph_encoder
        self.action_encoder = nn.Sequential(
            nn.Linear(action_token_dim, hidden_dim),
            nn.ReLU(),
        )
        self.comparator = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def _encode_graph(self, block_features: torch.Tensor, typed_edges: torch.Tensor) -> torch.Tensor:
        del typed_edges  # The contract keeps typed edges available; this smoke seam only pools node features.
        return self.graph_encoder(block_features.to(torch.float32)).mean(dim=0)

    def forward(self, payload: Mapping[str, Any]) -> torch.Tensor:
        validate_transition_payload(payload)
        pre_graph = self._encode_graph(payload["pre_block_features"], payload["pre_typed_edges"])
        post_graph = self._encode_graph(payload["post_block_features"], payload["post_typed_edges"])
        action = self.action_encoder(payload["action_token"].to(torch.float32).reshape(1, -1)).squeeze(0)
        return self.comparator(torch.cat([pre_graph, post_graph, action], dim=-1)).squeeze(-1)
