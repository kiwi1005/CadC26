from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/eval_hcfp_q6.py"
SPEC = importlib.util.spec_from_file_location("eval_hcfp_q6_test", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
q6 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(q6)


def test_q6_aggregates_benchmark_ranker_and_checkpoint_probe(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    benchmark = tmp_path / "benchmark.json"
    ranker = tmp_path / "ranker.json"
    checkpoint = tmp_path / "ranker.pt"
    output = tmp_path / "q6.json"
    checkpoint.write_bytes(b"checkpoint")
    benchmark.write_text(json.dumps(_benchmark_report()), encoding="utf-8")
    ranker.write_text(json.dumps(_ranker_eval()), encoding="utf-8")
    monkeypatch.setattr(
        q6,
        "load_checkpoint",
        lambda *_args, **_kwargs: (
            SimpleNamespace(to=lambda device: SimpleNamespace(device=device)),
            {"state_hash": "a" * 64, "parent_state_hash": "b" * 64},
        ),
    )

    assert q6.main(
        [
            "--benchmark",
            f"seed0={benchmark}",
            "--ranker-eval",
            str(ranker),
            "--checkpoint",
            f"ranker={checkpoint}",
            "--device",
            "cpu",
            "--output",
            str(output),
        ]
    ) == 0

    report = json.loads(output.read_text(encoding="utf-8"))
    learned = report["benchmarks"]["seed0"]["lanes"]["learned"]
    assert report["training_performed"] is False
    assert learned["cases"] == 3
    assert learned["hard_feasible"] == 2
    assert learned["cap_cross_count"] == 1
    assert learned["runtime_p50"] == pytest.approx(2.0)
    assert learned["runtime_p95"] == pytest.approx(2.9)
    assert learned["subset_106_120"]["cases"] == 2
    assert learned["subset_106_120"]["hard_feasible"] == 1
    gate = report["benchmarks"]["seed0"]["strict_gates"]["learned"]
    assert gate["passed"] is False
    assert gate["hard_feasibility_100_met"] is False
    assert gate["regressed_cases_zero_met"] is False
    assert gate["large_subset_nonregression_met"] is False
    assert gate["regressed_cases"] == 2
    assert gate["runtime_p50_met"] is True
    assert gate["runtime_p95_met"] is True
    assert report["ranker_eval"]["oracle_at_1"] == 1
    assert report["ranker_eval"]["oracle_at_4"] == 2
    assert report["ranker_eval"]["by_stage"]["initial"]["cases"] == 1
    assert report["summary"]["ranker_quality_gate"]["passed"] is False
    assert "initial_stage_fewer_than_16_cases" in report["summary"]["ranker_quality_gate"]["blockers"]
    assert report["checkpoints"]["ranker"]["state_hash"] == "a" * 64
    assert report["checkpoints"]["ranker"]["device"] == "cpu"
    assert "cpu_file_load_seconds" in report["checkpoints"]["ranker"]
    assert "device_transfer_seconds" in report["checkpoints"]["ranker"]
    assert "cold_load_total_seconds" in report["checkpoints"]["ranker"]
    assert report["checkpoints"]["ranker"]["a100_profile_available"] is False
    assert report["provenance"]["command"]
    assert report["provenance"]["torch_version"]
    assert "hcfp_environment" in report["provenance"]
    assert report["summary"]["q6_shadow_validation_gate"]["passed"] is False
    assert report["summary"]["submission_freeze_gate"]["passed"] is False
    assert report["summary"]["submission_freeze_gate"]["active_ranker_selection_proven"] is False
    assert "active_ranker_selection_unproven" in report["summary"]["submission_freeze_gate"]["blockers"]


def test_q6_resume_reuses_matching_benchmark_entry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    benchmark = tmp_path / "benchmark.json"
    output = tmp_path / "q6.json"
    benchmark.write_text(json.dumps(_benchmark_report()), encoding="utf-8")

    q6.main(["--benchmark", f"seed0={benchmark}", "--output", str(output)])
    monkeypatch.setattr(
        q6,
        "_lane_summary",
        lambda _rows: pytest.fail("resume should not recompute matching benchmark"),
    )

    assert q6.main(
        [
            "--benchmark",
            f"seed0={benchmark}",
            "--output",
            str(output),
            "--resume",
        ]
    ) == 0

    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["benchmarks"]["seed0"]["resume_status"] == "reused"


def test_q6_rejects_empty_or_incomplete_benchmark_rows(tmp_path: Path) -> None:
    benchmark = tmp_path / "bad.json"
    benchmark.write_text(
        json.dumps({"lanes": {"learned": [{"test_id": 1}]}}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="missing"):
        q6.main(["--benchmark", f"bad={benchmark}", "--output", str(tmp_path / "q6.json")])


def test_q6_rejects_inconsistent_feasibility_and_regression_summary(tmp_path: Path) -> None:
    benchmark = tmp_path / "bad.json"
    output = tmp_path / "q6.json"
    payload = _benchmark_report()
    payload["lanes"]["learned"][0]["hard_feasible"] = False  # type: ignore[index]
    with pytest.raises(ValueError, match="feasibility fields disagree"):
        benchmark.write_text(json.dumps(payload), encoding="utf-8")
        q6.main(["--benchmark", f"bad={benchmark}", "--output", str(output)])

    payload = _benchmark_report()
    payload["comparisons"]["learned"]["regressed_cases"] = 0  # type: ignore[index]
    benchmark.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="regression count disagrees"):
        q6.main(["--benchmark", f"bad={benchmark}", "--output", str(output)])


def test_q6_rejects_duplicate_benchmark_and_checkpoint_names(tmp_path: Path) -> None:
    benchmark = tmp_path / "benchmark.json"
    checkpoint = tmp_path / "ranker.pt"
    output = tmp_path / "q6.json"
    benchmark.write_text(json.dumps(_benchmark_report()), encoding="utf-8")
    checkpoint.write_bytes(b"checkpoint")

    with pytest.raises(ValueError, match="duplicate benchmark"):
        q6.main(
            [
                "--benchmark",
                f"same={benchmark}",
                "--benchmark",
                f"same={benchmark}",
                "--output",
                str(output),
            ]
        )

    with pytest.raises(ValueError, match="duplicate checkpoint"):
        q6.main(
            [
                "--benchmark",
                f"seed0={benchmark}",
                "--checkpoint",
                f"same={checkpoint}",
                "--checkpoint",
                f"same={checkpoint}",
                "--output",
                str(output),
            ]
        )


def test_q6_handles_missing_large_subset_without_division(tmp_path: Path) -> None:
    benchmark = tmp_path / "small.json"
    output = tmp_path / "q6.json"
    benchmark.write_text(json.dumps(_small_benchmark_report()), encoding="utf-8")

    assert q6.main(["--benchmark", f"small={benchmark}", "--output", str(output)]) == 0

    report = json.loads(output.read_text(encoding="utf-8"))
    learned = report["benchmarks"]["small"]["lanes"]["learned"]
    gate = report["benchmarks"]["small"]["strict_gates"]["learned"]
    assert learned["subset_106_120"]["cases"] == 0
    assert learned["subset_106_120"]["hard_feasibility_rate"] is None
    assert gate["large_subset_nonregression_met"] is None
    assert gate["passed"] is False
    assert gate["reason"] == "no_106_120_subset_to_prove_large_nonregression"
    assert not (output.parent / f".{output.name}.tmp").exists()


def test_q6_shadow_gate_requires_and_accepts_three_distinct_passing_seeds(
    tmp_path: Path,
) -> None:
    args = []
    for seed in (7001, 7002, 7003):
        benchmark = tmp_path / f"seed-{seed}.json"
        benchmark.write_text(
            json.dumps(_passing_benchmark_report(seed)),
            encoding="utf-8",
        )
        args.extend(("--benchmark", f"seed{seed}={benchmark}"))
    output = tmp_path / "q6.json"

    assert q6.main([*args, "--output", str(output)]) == 0

    report = json.loads(output.read_text(encoding="utf-8"))
    gate = report["summary"]["q6_shadow_validation_gate"]
    assert gate["passed"] is True
    assert gate["distinct_execution_seeds"] == [7001, 7002, 7003]
    assert report["summary"]["submission_freeze_gate"]["passed"] is False


def test_q6_rejects_non_boolean_feasibility(tmp_path: Path) -> None:
    benchmark = tmp_path / "bad-bool.json"
    payload = _benchmark_report()
    payload["lanes"]["learned"][0]["is_feasible"] = "false"  # type: ignore[index]
    benchmark.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="feasibility must be boolean"):
        q6.main(["--benchmark", f"bad={benchmark}", "--output", str(tmp_path / "q6.json")])


def test_q6_ranker_gate_uses_initial_stage_only() -> None:
    ranker = {
        "oracle_at_1": 12,
        "oracle_at_4": 15,
        "by_stage": {
            "initial": {
                "cases": 16,
                "oracle_at_1": 12,
                "oracle_at_4": 15,
                "oracle_at_1_rate": 12.0 / 16.0,
                "oracle_at_4_rate": 15.0 / 16.0,
                "false_promotion": 0,
            },
            "post_relax": {
                "cases": 16,
                "oracle_at_1": 0,
                "oracle_at_4": 0,
                "oracle_at_1_rate": 0.0,
                "oracle_at_4_rate": 0.0,
                "false_promotion": 16,
            },
        },
    }

    summary = q6._overall_summary({}, ranker, {})

    assert summary["ranker_quality_gate"]["passed"] is True
    assert summary["ranker_quality_gate"]["observed"] == ranker["by_stage"]["initial"]


def test_q6_strict_gate_rejects_runtime_regression() -> None:
    payload = _passing_benchmark_report(7001)
    payload["lanes"]["learned"][0]["runtime_seconds"] = 2.0  # type: ignore[index]
    lanes = q6._extract_lanes(payload)

    gate = q6._strict_gates(lanes, payload["comparisons"], payload["baseline"])["learned"]

    assert gate["passed"] is False
    assert gate["runtime_p50_met"] is False
    assert gate["runtime_p95_met"] is False
    assert gate["reason"] == "runtime_p50_regression,runtime_p95_regression"


def _benchmark_report() -> dict[str, object]:
    return {
        "schema_version": 1,
        "baseline": "fallback",
        "lanes": {
            "fallback": [
                _row(0, 64, 2.0, feasible=True, runtime=1.0, soft=0.0),
                _row(1, 106, 3.0, feasible=True, runtime=2.0, soft=0.0),
                _row(2, 120, 4.0, feasible=True, runtime=3.0, soft=0.0),
            ],
            "learned": [
                _row(0, 64, 1.5, feasible=True, runtime=1.0, soft=0.0),
                _row(1, 106, 9.999999, feasible=True, runtime=2.0, soft=1.0),
                _row(2, 120, 10.0, feasible=False, runtime=3.0, soft=0.0),
            ],
        },
        "promotion_decisions": {"learned": "HOLD"},
        "comparisons": {
            "learned": {
                "regressed_cases": 2,
                "per_case_delta": [
                    {"test_id": 0, "cost": -0.5},
                    {"test_id": 1, "cost": 6.999999},
                    {"test_id": 2, "cost": 6.0},
                ],
            }
        },
    }


def _small_benchmark_report() -> dict[str, object]:
    return {
        "schema_version": 1,
        "baseline": "fallback",
        "lanes": {
            "fallback": [
                _row(0, 32, 2.0, feasible=True, runtime=1.0, soft=0.0),
                _row(1, 64, 3.0, feasible=True, runtime=2.0, soft=0.0),
            ],
            "learned": [
                _row(0, 32, 1.5, feasible=True, runtime=1.0, soft=0.0),
                _row(1, 64, 2.5, feasible=True, runtime=2.0, soft=0.0),
            ],
        },
        "comparisons": {"learned": {"regressed_cases": 0}},
    }


def _passing_benchmark_report(seed: int) -> dict[str, object]:
    baseline = [_row(0, 120, 2.0, feasible=True, runtime=1.0, soft=0.0)]
    learned = [_row(0, 120, 1.5, feasible=True, runtime=1.0, soft=0.0)]
    return {
        "schema_version": 1,
        "baseline": "fallback",
        "provenance": {"search_config": {"execution_seed": seed}},
        "lanes": {"fallback": baseline, "learned": learned},
        "comparisons": {"learned": {"regressed_cases": 0}},
    }


def _row(
    test_id: int,
    blocks: int,
    cost: float,
    *,
    feasible: bool,
    runtime: float,
    soft: float,
) -> dict[str, object]:
    return {
        "test_id": test_id,
        "block_count": blocks,
        "is_feasible": feasible,
        "hpwl_gap": max(0.0, cost / 10.0 - 0.1),
        "area_gap": 0.0,
        "violations_relative": soft,
        "runtime_seconds": runtime,
        "cost": cost,
        "positions": [[0.0, 0.0, 1.0, 1.0]],
    }


def _ranker_eval() -> dict[str, object]:
    return {
        "schema_version": 2,
        "results": {
            "dev": {
                "ranker": {
                    "cases": [
                        {
                            "sample_id": "case/a",
                            "candidate_stage": "initial",
                            "top1_exact_best": True,
                            "top4_oracle_recall": True,
                            "false_promotion": False,
                            "rank_regret": 0,
                            "score_regret": 0.0,
                        },
                        {
                            "sample_id": "case/b",
                            "candidate_stage": "post_relax",
                            "top1_exact_best": False,
                            "top4_oracle_recall": True,
                            "false_promotion": True,
                            "rank_regret": 2,
                            "score_regret": 0.25,
                        },
                    ]
                }
            }
        },
    }
