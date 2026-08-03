from __future__ import annotations

import importlib.util
from pathlib import Path
import json
import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch

from hcfp.analytic import AnalyticConfig, solve_case_with_telemetry
from hcfp.checkpoint import RUNTIME_NORMALIZATION, load_checkpoint, save_checkpoint
from hcfp.data import DataSample, extract_labels, sample_to_payload
from hcfp.dynamics import DynamicsConfig
from hcfp.fallback import safe_shelf
from hcfp.model import HCFPModel, ModelConfig
from hcfp.profile import synthetic_case
from hcfp.replay import (
    OFFICIAL_TARGET_KIND,
    iter_replay,
    official_replay_scores,
    record_from_analysis,
    train_ranker_steps,
    write_replay,
)


ROOT = Path(__file__).resolve().parents[1]
REPLAY_SCRIPT = ROOT / "scripts/generate_hcfp_replay.py"
REPLAY_SPEC = importlib.util.spec_from_file_location("generate_hcfp_replay_test", REPLAY_SCRIPT)
assert REPLAY_SPEC is not None and REPLAY_SPEC.loader is not None
generate_replay = importlib.util.module_from_spec(REPLAY_SPEC)
REPLAY_SPEC.loader.exec_module(generate_replay)
ACTIVATION_REPLAY_SCRIPT = ROOT / "scripts/generate_hcfp_activation_replay.py"
ACTIVATION_REPLAY_SPEC = importlib.util.spec_from_file_location(
    "generate_hcfp_activation_replay_test",
    ACTIVATION_REPLAY_SCRIPT,
)
assert ACTIVATION_REPLAY_SPEC is not None and ACTIVATION_REPLAY_SPEC.loader is not None
generate_activation_replay = importlib.util.module_from_spec(ACTIVATION_REPLAY_SPEC)
ACTIVATION_REPLAY_SPEC.loader.exec_module(generate_activation_replay)


def test_replay_roundtrip_and_ranker_update(tmp_path: Path) -> None:
    case = synthetic_case(32, device="cpu")
    sample = DataSample("train-0", case, extract_labels(case, safe_shelf(case), normalized=True))
    config = AnalyticConfig(
        dynamics=DynamicsConfig(population=2, steps=0),
        projection_iterations=2,
        direction_beam=1,
    )
    analysis = solve_case_with_telemetry(case, config)
    record = record_from_analysis(
        sample,
        "a" * 64,
        analysis.raw_candidates,
        analysis.telemetry,
        population=2,
    )
    path = tmp_path / "replay.jsonl"
    assert write_replay([record], path) == 1
    loaded = list(iter_replay(path))

    assert len(loaded) == 1
    assert loaded[0].candidate_features.shape == (2, 8)
    assert loaded[0].target_score.shape == (2,)
    assert loaded[0].target_kind == OFFICIAL_TARGET_KIND
    assert torch.isfinite(record.target_score).all()

    model = HCFPModel(ModelConfig(hidden_dim=16, encoder_layers=1))
    before = {name: value.detach().clone() for name, value in model.ranker.named_parameters()}
    optimizer = torch.optim.AdamW(model.ranker.parameters(), lr=1.0e-3)
    history = train_ranker_steps(model, loaded, optimizer, steps=2)

    assert len(history) == 2
    assert all(torch.isfinite(torch.tensor(value)) for value in history)
    assert any(not torch.equal(before[name], value.detach()) for name, value in model.ranker.named_parameters())


def test_legacy_proxy_replay_remains_readable_but_cannot_train(tmp_path: Path) -> None:
    case = synthetic_case(32, device="cpu")
    sample = DataSample("legacy-0", case, extract_labels(case, safe_shelf(case), normalized=True))
    path = tmp_path / "legacy.jsonl"
    payload = {
        "schema_version": 1,
        "checkpoint_hash": "b" * 64,
        "sample": sample_to_payload(sample),
        "candidate_features": [[0.0] * 8],
        "target_score": [1.0],
    }
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    record = next(iter_replay(path))

    assert record.target_kind == "legacy_proxy_v1"
    model = HCFPModel(ModelConfig(hidden_dim=16, encoder_layers=1))
    optimizer = torch.optim.AdamW(model.ranker.parameters(), lr=1.0e-3)
    with pytest.raises(ValueError, match="official v10 replay targets"):
        train_ranker_steps(model, [record], optimizer, steps=1)


def test_official_replay_scores_are_lexicographic_without_cost_cap_ties() -> None:
    telemetry = SimpleNamespace(
        hpwl=torch.tensor([5.0, 10.0, 5.0]),
        bbox_area=torch.tensor([1.0, 2.0, 1.0]),
        soft_violation=torch.tensor([0.0, 0.0, 0.0]),
        hard_feasible=torch.tensor([True, True, False]),
        projected_overlap=torch.tensor([0.0, 0.0, 0.5]),
        overlap_components=torch.tensor([0, 0, 1]),
        projection_ok=torch.tensor([True, True, False]),
        projection_displacement=torch.tensor([0.0, 0.0, 2.0]),
    )
    case = SimpleNamespace(scale=10.0)

    scores = official_replay_scores(case, telemetry, baseline_area=100.0, baseline_hpwl=50.0)

    assert float(scores[0]) == pytest.approx(0.0)
    assert float(scores[1]) == pytest.approx(torch.log(torch.tensor(2.0)).item())
    assert float(scores[2]) > float(scores[:2].max())


def test_ranker_training_preserves_capabilities_and_declares_ranker(tmp_path: Path) -> None:
    case = synthetic_case(32, device="cpu")
    sample = DataSample("ranker-0", case, extract_labels(case, safe_shelf(case), normalized=True))
    checkpoint = tmp_path / "source.pt"
    source_hash = save_checkpoint(
        HCFPModel(ModelConfig(hidden_dim=16)),
        checkpoint,
        RUNTIME_NORMALIZATION,
        metadata={
            "capabilities": {"flow": True},
            "trained_heads": ["encoder", "flow"],
            "training_objective_version": "supervised_loss_v1",
        },
    )
    replay = tmp_path / "ranker.jsonl"
    replay.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "checkpoint_hash": source_hash,
                "target_kind": OFFICIAL_TARGET_KIND,
                "sample": sample_to_payload(sample),
                "candidate_features": [[0.0] * 8, [1.0] * 8],
                "target_score": [0.0, 1.0],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    ranked = tmp_path / "ranked.pt"

    subprocess.run(
        [
            sys.executable,
            "scripts/train_hcfp_ranker.py",
            str(replay),
            "--checkpoint",
            str(checkpoint),
            "--output",
            str(ranked),
            "--steps",
            "1",
            "--device",
            "cpu",
        ],
        cwd=Path(__file__).resolve().parents[1],
        check=True,
        text=True,
        capture_output=True,
    )

    _, metadata = load_checkpoint(ranked, expected_normalization=RUNTIME_NORMALIZATION)
    assert metadata["capabilities"] == {"flow": True}
    assert metadata["trained_heads"] == ["encoder", "flow", "ranker"]
    assert metadata["training_objective_version"] == "ranker_official_v10_v1"
    assert metadata["parent_state_hash"] == source_hash


def test_ranker_clis_reject_checkpoint_mismatch_and_split_leakage(tmp_path: Path) -> None:
    case = synthetic_case(32, device="cpu")
    sample = DataSample("guard-0", case, extract_labels(case, safe_shelf(case), normalized=True))
    replay = tmp_path / "guard.jsonl"
    replay.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "checkpoint_hash": "b" * 64,
                "target_kind": OFFICIAL_TARGET_KIND,
                "sample": sample_to_payload(sample),
                "candidate_features": [[0.0] * 8, [1.0] * 8],
                "target_score": [0.0, 1.0],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    checkpoint = tmp_path / "source.pt"
    save_checkpoint(HCFPModel(ModelConfig(hidden_dim=16)), checkpoint, RUNTIME_NORMALIZATION)
    root = Path(__file__).resolve().parents[1]

    train = subprocess.run(
        [
            sys.executable,
            "scripts/train_hcfp_ranker.py",
            str(replay),
            "--checkpoint",
            str(checkpoint),
            "--output",
            str(tmp_path / "ranked.pt"),
            "--steps",
            "1",
            "--device",
            "cpu",
        ],
        cwd=root,
        text=True,
        capture_output=True,
    )
    assert train.returncode != 0
    assert "replay checkpoint hash does not match" in train.stderr

    evaluate = subprocess.run(
        [
            sys.executable,
            "scripts/eval_hcfp_ranker.py",
            "--replay",
            f"first={replay}",
            "--replay",
            f"second={replay}",
            "--checkpoint",
            f"model={checkpoint}",
            "--output",
            str(tmp_path / "report.json"),
            "--device",
            "cpu",
        ],
        cwd=root,
        text=True,
        capture_output=True,
    )
    assert evaluate.returncode != 0
    assert "replay sample overlap" in evaluate.stderr


@pytest.mark.parametrize("module", (generate_replay, generate_activation_replay))
def test_replay_generators_reject_negative_collective_steps(module) -> None:
    with pytest.raises(module.argparse.ArgumentTypeError):
        module._non_negative_int("-1")
