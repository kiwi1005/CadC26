from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import pytest
import torch

from hcfp.case import from_official
from hcfp.checkpoint import RUNTIME_NORMALIZATION, load_checkpoint, save_checkpoint
from hcfp.data import DataSample, extract_labels, pairwise_precedence, write_shard
from hcfp.model import HCFPModel, ModelConfig
from hcfp.profile import synthetic_case
from hcfp.training import ExponentialMovingAverage, supervised_loss, train_steps


def _sample() -> DataSample:
    case = synthetic_case(32, device="cpu")
    from hcfp.fallback import safe_shelf

    return DataSample("train-0", case, extract_labels(case, safe_shelf(case), normalized=True))


def _constraint_sample() -> DataSample:
    case = from_official(
        4,
        [1.0, 1.0, 1.0, 1.0],
        [[0, 1, 2.0], [1, 2, 1.0], [2, 3, 1.0]],
        [[0, 0, 0.0]],
        [[0.0, 0.0]],
        [
            [0, 0, 1, 1, 9],
            [0, 0, 1, 1, 1],
            [0, 0, 0, 1, 6],
            [0, 0, 0, 0, 2],
        ],
    )
    rectangles = torch.tensor(
        [
            [0.0, 0.0, 0.5, 0.5],
            [0.5, 0.0, 0.5, 0.5],
            [1.0, 0.0, 0.5, 0.5],
            [1.5, 0.0, 0.5, 0.5],
        ],
        dtype=torch.float32,
    )
    return DataSample(
        "train-constraints",
        case,
        extract_labels(case, rectangles, normalized=True),
    )


def _empty_constraint_sample() -> DataSample:
    case = from_official(
        3,
        [1.0, 1.0, 1.0],
        [],
        [[0, 0, 0.0]],
        [[0.0, 0.0]],
        [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]],
    )
    side = torch.sqrt(case.area[0])
    rectangles = torch.tensor(
        [
            [0.0, 0.0, side, side],
            [side, 0.0, side, side],
            [2.0 * side, 0.0, side, side],
        ],
        dtype=torch.float32,
    )
    return DataSample("train-empty-constraints", case, extract_labels(case, rectangles, normalized=True))


def test_all_supervised_heads_train_with_finite_losses() -> None:
    torch.manual_seed(4)
    model = HCFPModel(ModelConfig(hidden_dim=16, encoder_layers=1))
    report = supervised_loss(model, _sample(), population=2, stage="all", seed=8)

    assert torch.isfinite(report.total)
    assert float(report.structure.detach()) > 0.0
    assert float(report.flow.detach()) > 0.0

    optimizer = torch.optim.AdamW(model.parameters(), lr=1.0e-3)
    history = train_steps(model, [_sample()], optimizer, steps=2, population=2, seed=8)
    assert len(history) == 2
    assert all(torch.isfinite(torch.tensor(step["total"])) for step in history)


def test_constraint_supervision_contributes_finite_gradients() -> None:
    torch.manual_seed(6)
    model = HCFPModel(ModelConfig(hidden_dim=16, encoder_layers=1, constraint_enabled=True))
    report = supervised_loss(model, _constraint_sample(), population=2, stage="structure", seed=9)

    assert torch.isfinite(report.total)
    assert torch.isfinite(report.constraint)
    assert float(report.constraint.detach()) > 0.0

    report.total.backward()
    grads = [
        parameter.grad
        for name, parameter in model.named_parameters()
        if name.startswith("constraints.")
    ]
    assert grads
    assert any(grad is not None and float(grad.detach().abs().sum()) > 0.0 for grad in grads)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_constraint_supervision_runs_on_cuda() -> None:
    model = HCFPModel(
        ModelConfig(hidden_dim=16, encoder_layers=1, constraint_enabled=True)
    ).cuda()

    report = supervised_loss(
        model,
        _constraint_sample(),
        population=2,
        stage="structure",
        seed=9,
    )

    assert report.total.is_cuda
    assert torch.isfinite(report.total)


def test_constraint_supervision_handles_empty_constraint_sets() -> None:
    torch.manual_seed(7)
    sample = _empty_constraint_sample()
    model = HCFPModel(ModelConfig(hidden_dim=16, encoder_layers=1, constraint_enabled=True))
    report = supervised_loss(model, sample, population=2, stage="structure", seed=10)

    assert torch.isfinite(report.total)
    assert torch.isfinite(report.constraint)
    assert float(report.constraint.detach()) == 0.0


def test_collective_stage_requires_enabled_head() -> None:
    model = HCFPModel(ModelConfig(hidden_dim=16, encoder_layers=1))

    with pytest.raises(ValueError, match="collective_enabled"):
        supervised_loss(
            model,
            _constraint_sample(),
            population=2,
            stage="collective",
            seed=12,
        )


def test_collective_rollout_overfits_one_corruption_with_finite_gradients() -> None:
    torch.manual_seed(12)
    sample = _constraint_sample()
    model = HCFPModel(
        ModelConfig(
            hidden_dim=16,
            encoder_layers=1,
            topology_enabled=True,
            constraint_enabled=True,
            collective_enabled=True,
            collective_message_dim=12,
            collective_passes=2,
        )
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.0e-2)
    initial = supervised_loss(
        model,
        sample,
        population=2,
        stage="collective",
        seed=13,
    )

    report = initial
    for _ in range(20):
        optimizer.zero_grad(set_to_none=True)
        report = supervised_loss(
            model,
            sample,
            population=2,
            stage="collective",
            seed=13,
        )
        report.total.backward()
        optimizer.step()

    assert torch.isfinite(report.collective)
    assert float(report.collective.detach()) < float(initial.collective.detach())
    gradients = {
        name: parameter.grad
        for name, parameter in model.named_parameters()
        if name.startswith("collective.")
    }
    assert gradients
    assert all(gradient is not None and torch.isfinite(gradient).all() for gradient in gradients.values())
    assert gradients["collective.pair.weight"].abs().sum() > 0.0
    assert gradients["collective.force_gates.3.weight"].abs().sum() > 0.0


def test_train_steps_restarts_stream_factory_without_materializing() -> None:
    model = HCFPModel(ModelConfig(hidden_dim=16, encoder_layers=1))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.0e-3)
    calls = 0

    def samples():
        nonlocal calls
        calls += 1
        yield _sample()

    history = train_steps(model, samples, optimizer, steps=3, population=2)

    assert len(history) == 3
    assert calls == 3


def test_all_stage_uses_one_model_forward_and_updates_ema() -> None:
    model = HCFPModel(ModelConfig(hidden_dim=16, encoder_layers=1))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.0e-3)
    ema = ExponentialMovingAverage(model, decay=0.9)
    calls = 0
    original = model.forward

    def counted(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    model.forward = counted  # type: ignore[method-assign]
    train_steps(model, [_sample()], optimizer, steps=1, population=2, ema=ema)

    assert calls == 1
    assert ema.shadow


def test_vectorized_precedence_preserves_unique_relation_semantics() -> None:
    boxes = torch.tensor(
        [
            [0.0, 0.0, 1.0, 1.0],
            [2.0, 0.0, 1.0, 1.0],
            [0.0, 2.0, 1.0, 1.0],
            [2.0, 2.0, 1.0, 1.0],
        ]
    )

    relation, tie = pairwise_precedence(boxes)

    assert relation[0, 1].item() == 0
    assert relation[1, 0].item() == 1
    assert relation[2, 0].item() == 2
    assert relation[0, 2].item() == 3
    assert relation[0, 3].item() == 4
    assert tie[0, 3]
    assert torch.all(torch.diag(tie))


def test_vectorized_precedence_matches_scalar_reference() -> None:
    boxes = torch.rand(20, 4)
    boxes[:, 2:4] += 0.05
    relation, tie = pairwise_precedence(boxes)
    expected = torch.full((20, 20), 4, dtype=torch.long)
    expected_tie = torch.eye(20, dtype=torch.bool)
    for i in range(20):
        for j in range(20):
            if i == j:
                continue
            gaps = (
                boxes[j, 0] - boxes[i, 0] - boxes[i, 2],
                boxes[i, 0] - boxes[j, 0] - boxes[j, 2],
                boxes[i, 1] - boxes[j, 1] - boxes[j, 3],
                boxes[j, 1] - boxes[i, 1] - boxes[i, 3],
            )
            valid = [index for index, gap in enumerate(gaps) if float(gap) >= -1.0e-7]
            if len(valid) == 1:
                expected[i, j] = valid[0]
                expected_tie[i, j] = False
            elif len(valid) > 1:
                expected_tie[i, j] = True

    assert torch.equal(relation, expected)
    assert torch.equal(tie, expected_tie)


def test_training_cli_emits_checkpoint_and_audit_report(tmp_path: Path) -> None:
    shard = tmp_path / "train.tar"
    checkpoint = tmp_path / "model.pt"
    write_shard(
        [_sample()],
        shard,
        provenance={
            "source": "FloorSet-train",
            "source_version": "fixture-v1",
            "split": "train",
            "denylist_sha256": "fixture-denylist",
        },
    )

    subprocess.run(
        [
            sys.executable,
            "scripts/train_hcfp.py",
            str(shard),
            "-o",
            str(checkpoint),
            "--steps",
            "1",
            "--population",
            "2",
            "--hidden-dim",
            "16",
            "--encoder-layers",
            "1",
            "--device",
            "cpu",
        ],
        cwd=Path(__file__).resolve().parents[1],
        check=True,
        text=True,
        capture_output=True,
    )

    report = json.loads(Path(f"{checkpoint}.training.json").read_text(encoding="utf-8"))
    _, metadata = load_checkpoint(checkpoint, expected_normalization=RUNTIME_NORMALIZATION)
    assert checkpoint.is_file()
    assert report["sample_count"] == 1
    assert report["steps"] == 1
    assert report["constraint_enabled"] is False
    assert len(report["checkpoint_hash"]) == 64
    assert metadata["capabilities"] == {"flow": True}
    assert metadata["trained_heads"] == ["encoder", "flow", "initializer", "structure"]
    assert metadata["training_objective_version"] == "supervised_loss_v1"
    assert metadata["parent_state_hash"] is None


def test_training_cli_enables_constraints_from_legacy_checkpoint(tmp_path: Path) -> None:
    shard = tmp_path / "train.tar"
    legacy = tmp_path / "legacy.pt"
    checkpoint = tmp_path / "constraint-model.pt"
    save_checkpoint(
        HCFPModel(ModelConfig(hidden_dim=16, encoder_layers=1)),
        legacy,
        RUNTIME_NORMALIZATION,
    )
    write_shard(
        [_constraint_sample()],
        shard,
        provenance={
            "source": "FloorSet-train",
            "source_version": "fixture-v1",
            "split": "train",
            "denylist_sha256": "fixture-denylist",
        },
    )

    subprocess.run(
        [
            sys.executable,
            "scripts/train_hcfp.py",
            str(shard),
            "-o",
            str(checkpoint),
            "--steps",
            "1",
            "--population",
            "2",
            "--device",
            "cpu",
            "--amp",
            "off",
            "--constraints",
            "--init-checkpoint",
            str(legacy),
        ],
        cwd=Path(__file__).resolve().parents[1],
        check=True,
        text=True,
        capture_output=True,
    )

    report = json.loads(Path(f"{checkpoint}.training.json").read_text(encoding="utf-8"))
    loaded, metadata = load_checkpoint(checkpoint, expected_normalization=RUNTIME_NORMALIZATION)
    assert report["constraint_enabled"] is True
    assert report["model_config"]["constraint_enabled"] is True
    assert loaded.config.constraint_enabled is True
    assert metadata["capabilities"] == {"flow": True}
    assert metadata["trained_heads"] == [
        "constraints",
        "encoder",
        "flow",
        "initializer",
        "structure",
    ]


def test_training_cli_warm_starts_collective_head_and_declares_capability(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.pt"
    source_hash = save_checkpoint(
        HCFPModel(
            ModelConfig(
                hidden_dim=16,
                encoder_layers=1,
                topology_enabled=True,
                constraint_enabled=True,
            )
        ),
        source,
        RUNTIME_NORMALIZATION,
        metadata={
            "capabilities": {"flow": False},
            "trained_heads": ["encoder", "structure", "topology", "constraints"],
            "training_objective_version": "structure_q2_v1",
        },
    )
    shard = tmp_path / "collective.tar"
    output = tmp_path / "collective.pt"
    write_shard(
        [_constraint_sample()],
        shard,
        provenance={
            "source": "FloorSet-train",
            "source_version": "fixture-v1",
            "split": "train",
            "denylist_sha256": "fixture-denylist",
        },
    )

    subprocess.run(
        [
            sys.executable,
            "scripts/train_hcfp.py",
            str(shard),
            "--output",
            str(output),
            "--steps",
            "1",
            "--population",
            "2",
            "--stage",
            "collective",
            "--collective",
            "--init-checkpoint",
            str(source),
            "--device",
            "cpu",
            "--amp",
            "off",
        ],
        cwd=Path(__file__).resolve().parents[1],
        check=True,
        text=True,
        capture_output=True,
    )

    source_model, _ = load_checkpoint(source)
    loaded, metadata = load_checkpoint(
        output,
        expected_normalization=RUNTIME_NORMALIZATION,
    )
    assert loaded.config.collective_enabled is True
    assert metadata["capabilities"] == {"collective": True, "flow": False}
    assert "collective" in metadata["trained_heads"]
    assert metadata["training_objective_version"] == "collective_rollout_v1"
    assert metadata["parent_state_hash"] == source_hash
    for name, value in source_model.state_dict().items():
        assert torch.equal(value, loaded.state_dict()[name])
    report = json.loads(Path(f"{output}.training.json").read_text(encoding="utf-8"))
    assert report["trainable_parameter_names"]
    assert all(
        name.startswith("collective.")
        for name in report["trainable_parameter_names"]
    )


def test_training_cli_preserves_checkpoint_constraints_when_unspecified(tmp_path: Path) -> None:
    shard = tmp_path / "train.tar"
    source = tmp_path / "constraint-source.pt"
    checkpoint = tmp_path / "constraint-preserved.pt"
    save_checkpoint(
        HCFPModel(ModelConfig(hidden_dim=16, encoder_layers=1, constraint_enabled=True)),
        source,
        RUNTIME_NORMALIZATION,
        metadata={
            "capabilities": {"flow": True},
            "trained_heads": ["encoder", "flow"],
            "training_objective_version": "supervised_loss_v1",
        },
    )
    write_shard(
        [_constraint_sample()],
        shard,
        provenance={
            "source": "FloorSet-train",
            "source_version": "fixture-v1",
            "split": "train",
            "denylist_sha256": "fixture-denylist",
        },
    )

    subprocess.run(
        [
            sys.executable,
            "scripts/train_hcfp.py",
            str(shard),
            "-o",
            str(checkpoint),
            "--steps",
            "1",
            "--population",
            "2",
            "--device",
            "cpu",
            "--amp",
            "off",
            "--stage",
            "structure",
            "--init-checkpoint",
            str(source),
        ],
        cwd=Path(__file__).resolve().parents[1],
        check=True,
        text=True,
        capture_output=True,
    )

    report = json.loads(Path(f"{checkpoint}.training.json").read_text(encoding="utf-8"))
    loaded, metadata = load_checkpoint(checkpoint, expected_normalization=RUNTIME_NORMALIZATION)
    assert report["constraint_enabled"] is True
    assert report["model_config"]["constraint_enabled"] is True
    assert loaded.config.constraint_enabled is True
    assert metadata["capabilities"] == {"flow": True}
    assert metadata["trained_heads"] == ["constraints", "encoder", "flow", "structure"]
    assert metadata["parent_state_hash"] == load_checkpoint(source)[1]["state_hash"]
