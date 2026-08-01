from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import torch

from hcfp.data import DataSample, extract_labels, write_shard
from hcfp.model import HCFPModel, ModelConfig
from hcfp.profile import synthetic_case
from hcfp.training import supervised_loss, train_steps


def _sample() -> DataSample:
    case = synthetic_case(32, device="cpu")
    from hcfp.fallback import safe_shelf

    return DataSample("train-0", case, extract_labels(case, safe_shelf(case), normalized=True))


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
    assert checkpoint.is_file()
    assert report["sample_count"] == 1
    assert report["steps"] == 1
    assert len(report["checkpoint_hash"]) == 64
