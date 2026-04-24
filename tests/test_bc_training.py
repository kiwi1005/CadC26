from __future__ import annotations

import torch

from puzzleplace.actions.schema import ActionPrimitive, TypedAction
from puzzleplace.data.schema import FloorSetCase
from puzzleplace.feedback import load_policy_checkpoint, save_policy_checkpoint
from puzzleplace.roles import label_case_roles
from puzzleplace.train import BCStepRecord, compute_bc_loss, run_bc_overfit


def _make_case() -> FloorSetCase:
    return FloorSetCase(
        case_id="bc-1",
        block_count=2,
        area_targets=torch.tensor([6.0, 6.0]),
        b2b_edges=torch.tensor([[0.0, 1.0, 1.0]]),
        p2b_edges=torch.empty((0, 3)),
        pins_pos=torch.empty((0, 2)),
        constraints=torch.zeros((2, 5)),
        target_positions=torch.tensor([[0.0, 0.0, 2.0, 3.0], [2.0, 0.0, 3.0, 2.0]]),
        metrics=None,
    )


def _dataset() -> list[BCStepRecord]:
    case = _make_case()
    roles = label_case_roles(case)
    return [
        BCStepRecord(
            case=case,
            role_evidence=roles,
            placements={},
            action=TypedAction(
                ActionPrimitive.PLACE_ABSOLUTE,
                block_index=0,
                x=0.0,
                y=0.0,
                w=2.0,
                h=3.0,
            ),
        ),
        BCStepRecord(
            case=case,
            role_evidence=roles,
            placements={0: (0.0, 0.0, 2.0, 3.0)},
            action=TypedAction(
                ActionPrimitive.PLACE_ABSOLUTE,
                block_index=1,
                x=2.0,
                y=0.0,
                w=3.0,
                h=2.0,
            ),
        ),
    ]


def test_bc_loss_computes_on_single_record() -> None:
    dataset = _dataset()
    from puzzleplace.models import TypedActionPolicy

    policy = TypedActionPolicy(hidden_dim=32)
    breakdown = compute_bc_loss(policy, dataset[0])
    assert breakdown.total.item() > 0


def test_bc_overfit_reduces_loss_on_tiny_dataset() -> None:
    dataset = _dataset()
    _policy, summary = run_bc_overfit(dataset, hidden_dim=32, lr=1e-2, epochs=30, seed=0)
    assert summary.final_loss < summary.initial_loss


def test_bc_checkpoint_round_trip_keeps_action_scores(tmp_path) -> None:
    dataset = _dataset()
    policy, _summary = run_bc_overfit(dataset, hidden_dim=32, lr=1e-2, epochs=10, seed=0)
    checkpoint_path = tmp_path / "bc_policy.pt"
    save_policy_checkpoint(policy, checkpoint_path, metadata={"split": "training"})
    reloaded = load_policy_checkpoint(checkpoint_path)

    original = policy(
        dataset[0].case,
        role_evidence=dataset[0].role_evidence,
        placements=dataset[0].placements,
    )
    restored = reloaded(
        dataset[0].case,
        role_evidence=dataset[0].role_evidence,
        placements=dataset[0].placements,
    )
    assert torch.allclose(original.primitive_logits, restored.primitive_logits)
    assert torch.allclose(original.block_logits, restored.block_logits)


def test_bc_checkpoint_load_tolerates_input_dim_extension(tmp_path) -> None:
    policy, _summary = run_bc_overfit(
        _dataset(),
        hidden_dim=32,
        lr=1e-2,
        epochs=3,
        seed=0,
    )
    checkpoint_path = tmp_path / "legacy_policy.pt"
    case = _make_case()
    state_dict = policy.state_dict()
    state_dict["encoder.input_proj.0.weight"] = state_dict[
        "encoder.input_proj.0.weight"
    ][:, : state_dict["encoder.input_proj.0.weight"].shape[1] - 1]
    torch.save(
        {
            "hidden_dim": 32,
            "state_dict": state_dict,
            "metadata": {},
        },
        checkpoint_path,
    )
    reloaded = load_policy_checkpoint(checkpoint_path)
    replay = reloaded(
        case,
        role_evidence=label_case_roles(case),
        placements={},
    )
    assert replay.block_logits.shape == (2,)
