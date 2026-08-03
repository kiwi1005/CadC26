from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from hcfp.checkpoint import RUNTIME_NORMALIZATION, save_checkpoint
from hcfp.model import HCFPModel, ModelConfig


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "artifacts/floorset-v10"
SCRIPT = ROOT / "scripts/benchmark_hcfp.py"
SPEC = importlib.util.spec_from_file_location("benchmark_hcfp_test", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
benchmark = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(benchmark)


@pytest.mark.skipif(not (DATA / "LiteTensorDataTest").is_dir(), reason="official validation cache is unavailable")
def test_official_benchmark_runs_both_lanes_on_one_case(tmp_path: Path) -> None:
    output = tmp_path / "official.json"
    subprocess.run(
        [
            sys.executable,
            "scripts/benchmark_hcfp.py",
            "--optimizer",
            "fallback=scripts/audit_fallback_optimizer.py",
            "--optimizer",
            "analytic=submission/optimizer.py",
            "--baseline",
            "fallback",
            "--data-path",
            str(DATA),
            "--cases",
            "0",
            "--device",
            "cpu",
            "--output",
            str(output),
        ],
        cwd=ROOT,
        check=True,
        text=True,
        capture_output=True,
    )

    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["lane_summary"]["fallback"]["feasible"] == 1
    assert report["lane_summary"]["analytic"]["feasible"] == 1
    assert report["provenance"]["evaluator_sha256"]
    assert len(report["case_metadata"]["0"]["constraints"]) == 21


@pytest.mark.skipif(not (DATA / "LiteTensorDataTest").is_dir(), reason="official validation cache is unavailable")
def test_official_benchmark_audits_valid_checkpoint_usage(tmp_path: Path) -> None:
    checkpoint = tmp_path / "model.pt"
    checkpoint_hash = save_checkpoint(
        HCFPModel(ModelConfig(hidden_dim=16, encoder_layers=1)),
        checkpoint,
        RUNTIME_NORMALIZATION,
        metadata={
            "capabilities": {"flow": True},
            "trained_heads": ["flow"],
            "training_objective_version": "supervised_loss_v1",
        },
    )
    output = tmp_path / "learned.json"
    subprocess.run(
        [
            sys.executable,
            "scripts/benchmark_hcfp.py",
            "--optimizer",
            "fallback=scripts/audit_fallback_optimizer.py",
            "--optimizer",
            "learned=scripts/audit_learned_optimizer.py",
            "--checkpoint",
            f"learned={checkpoint}",
            "--baseline",
            "fallback",
            "--data-path",
            str(DATA),
            "--cases",
            "0",
            "--device",
            "cpu",
            "--flow-steps",
            "2",
            "--flow-seed",
            "17",
            "--tail-topk",
            "1",
            "--output",
            str(output),
        ],
        cwd=ROOT,
        check=True,
        text=True,
        capture_output=True,
    )

    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["lane_summary"]["learned"]["feasible"] == 1
    assert report["lane_metadata"]["learned"]["checkpoint_hash"] == checkpoint_hash
    assert report["lane_metadata"]["learned"]["required"] is True
    assert report["lane_metadata"]["learned"]["requested_flow_steps"] == 2
    assert report["lane_metadata"]["learned"]["flow_steps"] == 2
    assert report["lane_metadata"]["learned"]["flow_seed"] == 17
    assert report["lane_metadata"]["learned"]["tail_topk"] == 1


def test_benchmark_optimizer_sets_collective_steps_per_lane(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    false_checkpoint = tmp_path / "false.pt"
    true_checkpoint = tmp_path / "true.pt"
    save_checkpoint(
        HCFPModel(ModelConfig(hidden_dim=16, collective_enabled=True)),
        false_checkpoint,
        RUNTIME_NORMALIZATION,
        metadata={
            "capabilities": {"collective": False},
            "trained_heads": ["collective"],
            "training_objective_version": "collective_loss_v1",
        },
    )
    true_hash = save_checkpoint(
        HCFPModel(ModelConfig(hidden_dim=16, collective_enabled=True)),
        true_checkpoint,
        RUNTIME_NORMALIZATION,
        metadata={
            "capabilities": {"collective": True},
            "trained_heads": ["collective"],
            "training_objective_version": "collective_loss_v1",
        },
    )
    calls: list[tuple[str, str | None, str | None]] = []

    class FakeEvaluator:
        def __init__(self, *_args, **_kwargs) -> None:
            pass

        def evaluate(self, optimizer: str, *, test_ids=None):
            del test_ids
            calls.append(
                (
                    optimizer,
                    benchmark.os.environ.get("HCFP_CHECKPOINT"),
                    benchmark.os.environ.get("HCFP_COLLECTIVE_STEPS"),
                )
            )
            return SimpleNamespace(test_results=[])

    monkeypatch.setattr(
        benchmark,
        "_load_evaluator",
        lambda _data_path: SimpleNamespace(ContestEvaluator=FakeEvaluator),
    )

    _, _, lane_metadata = benchmark._run_optimizers(
        {
            "disabled": Path("disabled.py"),
            "enabled": Path("enabled.py"),
            "plain": Path("plain.py"),
        },
        tmp_path,
        [0],
        "cpu",
        {"disabled": false_checkpoint, "enabled": true_checkpoint},
        flow_steps=0,
        collective_steps=3,
        flow_seed=0,
        tail_topk=None,
    )

    assert calls == [
        ("disabled.py", str(false_checkpoint), "0"),
        ("enabled.py", str(true_checkpoint), "3"),
        ("plain.py", None, "0"),
    ]
    assert lane_metadata["disabled"]["requested_collective_steps"] == 3
    assert lane_metadata["disabled"]["collective_steps"] == 0
    assert lane_metadata["enabled"]["checkpoint_hash"] == true_hash
    assert lane_metadata["enabled"]["capabilities"]["collective"] is True
    assert lane_metadata["enabled"]["trained_heads"] == ["collective"]
    assert lane_metadata["enabled"]["collective_steps"] == 3
    assert lane_metadata["plain"]["collective_steps"] == 0


def test_benchmark_optimizer_propagates_structural_search_metadata(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str | None, str | None, str | None]] = []

    class FakeEvaluator:
        def __init__(self, *_args, **_kwargs) -> None:
            pass

        def evaluate(self, _optimizer: str, *, test_ids=None):
            del test_ids
            calls.append(
                (
                    benchmark.os.environ.get("HCFP_TOPOLOGY_SEEDS"),
                    benchmark.os.environ.get("HCFP_CONSTRAINT_SEEDS"),
                    benchmark.os.environ.get("HCFP_RANKER_SELECTION_EXPERIMENT"),
                )
            )
            return SimpleNamespace(test_results=[])

    monkeypatch.setattr(
        benchmark,
        "_load_evaluator",
        lambda _data_path: SimpleNamespace(ContestEvaluator=FakeEvaluator),
    )

    _, _, lane_metadata = benchmark._run_optimizers(
        {"plain": Path("plain.py")},
        tmp_path,
        [0],
        "cpu",
        {},
        flow_steps=0,
        collective_steps=0,
        flow_seed=7,
        tail_topk=None,
        topology_seeds=16,
        constraint_seeds=8,
        ranker_selection_experiment=True,
    )

    assert calls == [("16", "8", "1")]
    assert lane_metadata["plain"]["topology_seeds"] == 16
    assert lane_metadata["plain"]["constraint_seeds"] == 8
    assert lane_metadata["plain"]["ranker_selection_experiment"] is False


def test_benchmark_provenance_binds_search_and_worktree(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    completed = iter(
        (
            SimpleNamespace(stdout="abc123\n"),
            SimpleNamespace(stdout=" M changed.py\n"),
        )
    )
    monkeypatch.setattr(benchmark.subprocess, "run", lambda *_args, **_kwargs: next(completed))

    provenance = benchmark._provenance(
        Path("dataset"),
        "cpu",
        "optimizer",
        search_config={"topology_seeds": 16},
    )

    assert provenance["git_commit"] == "abc123"
    assert provenance["git_clean"] is False
    assert provenance["git_status_sha256"]
    assert provenance["search_config"] == {"topology_seeds": 16}


def test_benchmark_collective_steps_defaults_to_zero() -> None:
    parser_value = benchmark._non_negative_int("0")

    assert parser_value == 0


def test_benchmark_rejects_negative_collective_steps() -> None:
    with pytest.raises(benchmark.argparse.ArgumentTypeError):
        benchmark._non_negative_int("-1")
