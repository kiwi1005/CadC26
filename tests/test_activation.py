from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import subprocess
import sys

import pytest
import torch

from hcfp.activation import (
    ACTIVATION_FEATURE_DIM,
    ACTIVATION_FEATURE_VERSION,
    ActivationOutcome,
    ActivationPolicy,
    ActivationRecord,
    activation_policy_metrics,
    activation_features,
    activation_outcome,
    fit_activation_policy,
    iter_activation_replay,
    load_activation_policy,
    save_activation_policy,
    write_activation_replay,
)
from hcfp.analytic import AnalyticConfig, solve_case_with_telemetry
from hcfp.dynamics import DynamicsConfig
from hcfp.profile import synthetic_case


def _analysis():
    case = synthetic_case(32, device="cpu")
    config = AnalyticConfig(
        dynamics=DynamicsConfig(population=2, steps=0),
        projection_iterations=2,
        direction_beam=1,
    )
    return case, solve_case_with_telemetry(case, config)


def _outcome(objective: float) -> ActivationOutcome:
    return ActivationOutcome(True, 0.25, 1.0, 2.0, objective, 0.1)


def _record(
    features: torch.Tensor | None = None,
    *,
    sample_id: str = "worker/layout:1",
    tail_needed: bool = True,
    block_count: int = 32,
    config_hash: str = "b" * 64,
) -> ActivationRecord:
    analytic = _outcome(3.0 if tail_needed else 2.5)
    learned = _outcome(2.5 if tail_needed else 3.0)
    return ActivationRecord(
        sample_id=sample_id,
        block_count=block_count,
        checkpoint_hash="a" * 64,
        config_hash=config_hash,
        features=torch.zeros(ACTIVATION_FEATURE_DIM) if features is None else features,
        tail_needed=tail_needed,
        quality_margin=-0.5 if tail_needed else 0.5,
        analytic=analytic,
        learned=learned,
    )


def test_activation_features_are_deterministic_fixed_cpu_fp32() -> None:
    case, analysis = _analysis()
    learned = analysis.raw_candidates[1:3]
    scores = torch.tensor([0.4, 0.1])

    first = activation_features(case, analysis, learned, scores)
    second = activation_features(case, analysis, learned.clone(), scores.clone())

    assert first.shape == (ACTIVATION_FEATURE_DIM,)
    assert first.dtype == torch.float32
    assert first.device.type == "cpu"
    assert torch.equal(first, second)
    assert torch.isfinite(first).all()


def test_activation_features_use_selected_analytic_telemetry() -> None:
    case, analysis = _analysis()
    exact_source = str(analysis.incumbent_snapshot["exact_source"])
    exact_index = 0 if exact_source == "fallback" else int(exact_source.removeprefix("candidate_"))
    hard = torch.ones_like(analysis.telemetry.hard_feasible)
    projection = torch.zeros_like(analysis.telemetry.projection_ok)
    hard[exact_index] = False
    projection[exact_index] = True
    selected_analysis = replace(
        analysis,
        telemetry=replace(
            analysis.telemetry,
            hard_feasible=hard,
            projection_ok=projection,
        ),
    )

    features = activation_features(
        case,
        selected_analysis,
        selected_analysis.raw_candidates[1:3],
        torch.tensor([0.4, 0.1]),
    )

    assert features[48].item() == 0.0
    assert features[49].item() == 1.0


def test_activation_replay_roundtrip(tmp_path: Path) -> None:
    path = tmp_path / "activation.jsonl"
    record = _record(torch.arange(ACTIVATION_FEATURE_DIM, dtype=torch.float32))

    assert write_activation_replay([record], path) == 1
    loaded = list(iter_activation_replay(path))

    assert len(loaded) == 1
    assert loaded[0].sample_id == record.sample_id
    assert loaded[0].block_count == record.block_count
    assert loaded[0].feature_version == ACTIVATION_FEATURE_VERSION
    assert loaded[0].tail_needed is True
    assert loaded[0].analytic == record.analytic
    assert torch.equal(loaded[0].features, record.features)


def test_activation_replay_preserves_auditable_learned_failure(tmp_path: Path) -> None:
    path = tmp_path / "activation-failure.jsonl"
    analytic = _outcome(2.0)
    learned = ActivationOutcome(False, 0.0, 0.0, 0.0, 10.0, 0.2)
    record = ActivationRecord(
        sample_id="worker/failure:1",
        block_count=32,
        checkpoint_hash="a" * 64,
        config_hash="b" * 64,
        features=torch.zeros(ACTIVATION_FEATURE_DIM),
        tail_needed=False,
        quality_margin=8.0,
        analytic=analytic,
        learned=learned,
        failure_reason="RuntimeError: learned tail failed",
    )

    write_activation_replay([record], path)
    loaded = list(iter_activation_replay(path))

    assert loaded[0].failure_reason == record.failure_reason
    assert loaded[0].learned.feasible is False
    assert loaded[0].tail_needed is False


def test_activation_failure_reason_requires_infeasible_learned_outcome() -> None:
    with pytest.raises(ValueError, match="infeasible learned outcome"):
        ActivationRecord(
            sample_id="worker/failure:2",
            block_count=32,
            checkpoint_hash="a" * 64,
            config_hash="b" * 64,
            features=torch.zeros(ACTIVATION_FEATURE_DIM),
            tail_needed=True,
            quality_margin=-0.5,
            analytic=_outcome(3.0),
            learned=_outcome(2.5),
            failure_reason="RuntimeError: impossible",
        )


def test_activation_split_schedule_interleaves_exact_counts() -> None:
    from scripts.generate_hcfp_activation_replay import _interleaved_split_names

    schedule = _interleaved_split_names({"train": 256, "calibration": 64, "heldout": 64})

    assert len(schedule) == 384
    assert schedule.count("train") == 256
    assert schedule.count("calibration") == 64
    assert schedule.count("heldout") == 64
    assert schedule[:6].count("train") == 4
    assert schedule[:6].count("calibration") == 1
    assert schedule[:6].count("heldout") == 1


def test_activation_outcome_uses_uncapped_runtime_independent_objective() -> None:
    case, analysis = _analysis()
    outcome = activation_outcome(
        case,
        analysis.selected,
        baseline_area=1.0,
        baseline_hpwl=1.0,
        runtime_seconds=123.0,
    )

    assert outcome.feasible
    assert outcome.runtime_seconds == 123.0
    assert outcome.objective > 0.0


@pytest.mark.parametrize(
    ("field", "value", "match"),
    (
        ("schema_version", 99, "schema mismatch"),
        ("feature_version", "future", "feature version mismatch"),
        ("tail_needed", 1, "tail_needed must be boolean"),
    ),
)
def test_activation_replay_rejects_incompatible_payloads(
    tmp_path: Path,
    field: str,
    value,
    match: str,
) -> None:
    path = tmp_path / "bad.jsonl"
    good = tmp_path / "good.jsonl"
    write_activation_replay([_record()], good)
    payload = json.loads(good.read_text(encoding="utf-8"))
    payload[field] = value
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match=match):
        list(iter_activation_replay(path))


def test_activation_record_rejects_nonfinite_or_wrong_feature_shape() -> None:
    with pytest.raises(ValueError, match="non-empty string"):
        _record(sample_id=123)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="shape"):
        _record(torch.zeros(ACTIVATION_FEATURE_DIM - 1))
    features = torch.zeros(ACTIVATION_FEATURE_DIM)
    features[0] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        _record(features)
    with pytest.raises(ValueError, match="SHA256"):
        ActivationRecord(
            "sample",
            32,
            "not-a-hash",
            "b" * 64,
            torch.zeros(ACTIVATION_FEATURE_DIM),
            False,
            0.0,
            _outcome(1.0),
            _outcome(1.0),
        )


def test_activation_policy_fit_calibrate_roundtrip_and_metrics(tmp_path: Path) -> None:
    def record(sample_id: str, value: float, tail_needed: bool) -> ActivationRecord:
        features = torch.zeros(ACTIVATION_FEATURE_DIM)
        features[0] = value
        return _record(features, sample_id=sample_id, tail_needed=tail_needed)

    train = [
        record("train-n0", -2.0, False),
        record("train-n1", -1.0, False),
        record("train-p0", 1.0, True),
        record("train-p1", 2.0, True),
    ]
    calibration = [
        record("cal-n0", -0.5, False),
        record("cal-p0", 0.5, True),
    ]

    policy, history = fit_activation_policy(
        train,
        calibration,
        steps=200,
        learning_rate=0.05,
    )
    metrics = activation_policy_metrics(policy, calibration, force_large_min=121)
    calibration_probabilities = policy.probability(
        torch.stack([record.features for record in calibration])
    )
    path = tmp_path / "activation-policy.json"
    saved_hash = save_activation_policy(policy, path)
    loaded = load_activation_policy(path)

    assert len(history) == 200
    assert history[-1] < history[0]
    assert metrics["recall"] == 1.0
    assert metrics["false_skip_sample_ids"] == []
    assert policy.threshold < float(calibration_probabilities[1])
    assert saved_hash
    assert torch.equal(policy.probability(train[0].features), loaded.probability(train[0].features))


def test_activation_policy_rejects_leakage_missing_positives_and_tampering(tmp_path: Path) -> None:
    negative = _record(sample_id="same", tail_needed=False)
    positive = _record(sample_id="positive", tail_needed=True)
    with pytest.raises(ValueError, match="sample overlap"):
        fit_activation_policy([negative, positive], [negative])
    with pytest.raises(ValueError, match="positive and negative"):
        fit_activation_policy(
            [_record(sample_id="n0", tail_needed=False)],
            [_record(sample_id="p0", tail_needed=True)],
        )

    policy = ActivationPolicy(
        "a" * 64,
        "b" * 64,
        torch.zeros(ACTIVATION_FEATURE_DIM),
        torch.ones(ACTIVATION_FEATURE_DIM),
        torch.zeros(ACTIVATION_FEATURE_DIM),
        0.0,
        0.5,
    )
    path = tmp_path / "policy.json"
    save_activation_policy(policy, path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["bias"] = 1.0
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="hash mismatch"):
        load_activation_policy(path)


def test_activation_training_and_evaluation_clis(tmp_path: Path) -> None:
    def record(sample_id: str, value: float, tail_needed: bool) -> ActivationRecord:
        features = torch.zeros(ACTIVATION_FEATURE_DIM)
        features[0] = value
        return _record(features, sample_id=sample_id, tail_needed=tail_needed)

    train_path = tmp_path / "train.jsonl"
    calibration_path = tmp_path / "calibration.jsonl"
    heldout_path = tmp_path / "heldout.jsonl"
    policy_path = tmp_path / "policy.json"
    report_path = tmp_path / "report.json"
    write_activation_replay(
        [
            record("train-n", -2.0, False),
            record("train-n2", -1.0, False),
            record("train-p", 1.0, True),
            record("train-p2", 2.0, True),
        ],
        train_path,
    )
    write_activation_replay(
        [record("cal-n", -0.5, False), record("cal-p", 0.5, True)],
        calibration_path,
    )
    write_activation_replay(
        [record("held-n", -0.25, False), record("held-p", 0.75, True)],
        heldout_path,
    )

    subprocess.run(
        [
            sys.executable,
            "scripts/train_hcfp_activation.py",
            "--train-replay",
            str(train_path),
            "--calibration-replay",
            str(calibration_path),
            "--output",
            str(policy_path),
            "--steps",
            "100",
            "--min-train-positives",
            "1",
            "--min-calibration-positives",
            "1",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run(
        [
            sys.executable,
            "scripts/eval_hcfp_activation.py",
            "--policy",
            str(policy_path),
            "--replay",
            f"heldout={heldout_path}",
            "--training-report",
            str(policy_path.with_suffix(".json.training.json")),
            "--output",
            str(report_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["results"]["heldout"]["recall"] == 1.0
    assert "not promotion evidence" in report["runtime_warning"]
    assert set(report["training_exclusions"]["splits"]) == {"train", "calibration"}
    assert policy_path.with_suffix(".json.training.json").is_file()

    leaked = subprocess.run(
        [
            sys.executable,
            "scripts/eval_hcfp_activation.py",
            "--policy",
            str(policy_path),
            "--replay",
            f"heldout={train_path}",
            "--training-report",
            str(policy_path.with_suffix(".json.training.json")),
            "--output",
            str(tmp_path / "leaked.json"),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert leaked.returncode != 0
    assert "sample overlap" in leaked.stderr
