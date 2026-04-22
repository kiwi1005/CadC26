from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from puzzleplace.actions import ActionPrimitive
from puzzleplace.models import TypedActionPolicy

from .dataset_bc import BCStepRecord, action_to_targets


@dataclass(slots=True)
class BCLossBreakdown:
    total: torch.Tensor
    primitive: torch.Tensor
    block: torch.Tensor
    target: torch.Tensor
    boundary: torch.Tensor
    geometry: torch.Tensor


@dataclass(slots=True)
class BCTrainingSummary:
    initial_loss: float
    final_loss: float
    primitive_accuracy: float
    block_accuracy: float
    dataset_size: int
    epochs: int


def compute_bc_loss(policy: TypedActionPolicy, record: BCStepRecord) -> BCLossBreakdown:
    output = policy(record.case, role_evidence=record.role_evidence, placements=record.placements)
    targets = action_to_targets(record.action)
    ce = nn.CrossEntropyLoss()
    primitive_id = targets["primitive_id"]
    block_index = targets["block_index"]
    target_index = targets["target_index"]
    geometry_target = targets["geometry"]
    if not isinstance(primitive_id, int) or not isinstance(block_index, int):
        raise TypeError("BC targets must provide integer primitive and block ids")

    primitive_target = torch.tensor([primitive_id], dtype=torch.long)
    primitive_loss = ce(output.primitive_logits.unsqueeze(0), primitive_target)

    block_target = torch.tensor([block_index], dtype=torch.long)
    block_loss = ce(output.block_logits.unsqueeze(0), block_target)

    if target_index is None:
        target_loss = torch.zeros((), dtype=torch.float32)
    else:
        row = output.target_logits[block_index].unsqueeze(0)
        target_loss = ce(row, torch.tensor([target_index], dtype=torch.long))

    boundary_class_raw = targets["boundary_class"]
    if not isinstance(boundary_class_raw, int):
        raise TypeError("BC targets must provide an integer boundary class")
    boundary_class = boundary_class_raw
    if record.action.primitive is ActionPrimitive.ALIGN_BOUNDARY:
        row = output.boundary_logits[block_index].unsqueeze(0)
        boundary_loss = ce(row, torch.tensor([boundary_class], dtype=torch.long))
    else:
        boundary_loss = torch.zeros((), dtype=torch.float32)

    if geometry_target is None:
        geometry_loss = torch.zeros((), dtype=torch.float32)
    else:
        if not isinstance(geometry_target, torch.Tensor):
            raise TypeError("BC targets must provide a tensor geometry target")
        pred_geometry = output.geometry[block_index]
        geometry_loss = nn.functional.mse_loss(pred_geometry, geometry_target)

    total = primitive_loss + block_loss + target_loss + boundary_loss + geometry_loss
    return BCLossBreakdown(
        total=total,
        primitive=primitive_loss,
        block=block_loss,
        target=target_loss,
        boundary=boundary_loss,
        geometry=geometry_loss,
    )


def _accuracy(policy: TypedActionPolicy, dataset: list[BCStepRecord]) -> tuple[float, float]:
    primitive_hits = 0
    block_hits = 0
    for record in dataset:
        output = policy(
            record.case,
            role_evidence=record.role_evidence,
            placements=record.placements,
        )
        targets = action_to_targets(record.action)
        primitive_id = targets["primitive_id"]
        block_index = targets["block_index"]
        if not isinstance(primitive_id, int) or not isinstance(block_index, int):
            raise TypeError("BC accuracy targets must provide integer primitive and block ids")
        primitive_hits += int(int(output.primitive_logits.argmax().item()) == primitive_id)
        block_hits += int(int(output.block_logits.argmax().item()) == block_index)
    denom = max(len(dataset), 1)
    return primitive_hits / denom, block_hits / denom


def run_bc_overfit(
    dataset: list[BCStepRecord],
    *,
    hidden_dim: int = 128,
    lr: float = 1e-3,
    epochs: int = 50,
    seed: int = 0,
) -> tuple[TypedActionPolicy, BCTrainingSummary]:
    torch.manual_seed(seed)
    policy = TypedActionPolicy(hidden_dim=hidden_dim)
    optimizer = torch.optim.Adam(policy.parameters(), lr=lr)

    initial_loss = None
    for _epoch in range(epochs):
        epoch_loss = 0.0
        for record in dataset:
            optimizer.zero_grad(set_to_none=True)
            breakdown = compute_bc_loss(policy, record)
            breakdown.total.backward()
            optimizer.step()
            epoch_loss += float(breakdown.total.item())
        if initial_loss is None:
            initial_loss = epoch_loss / max(len(dataset), 1)

    final_loss = 0.0
    for record in dataset:
        final_loss += float(compute_bc_loss(policy, record).total.item())
    final_loss /= max(len(dataset), 1)
    primitive_acc, block_acc = _accuracy(policy, dataset)
    return policy, BCTrainingSummary(
        initial_loss=float(initial_loss or 0.0),
        final_loss=float(final_loss),
        primitive_accuracy=float(primitive_acc),
        block_accuracy=float(block_acc),
        dataset_size=len(dataset),
        epochs=epochs,
    )
