from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import pytest

from hcfp.benchmark import build_report, weighted_score


def _rows(costs=(2.0, 3.0), feasible=(True, True)):
    return [
        {
            "test_id": index,
            "block_count": block_count,
            "is_feasible": feasible[index],
            "hpwl_gap": cost / 10.0,
            "area_gap": cost / 20.0,
            "violations_relative": 0.0,
            "runtime_seconds": 0.1 + index,
            "cost": cost,
            "positions": [[0.0, 0.0, 1.0, 1.0]],
        }
        for index, (block_count, cost) in enumerate(zip((32, 120), costs))
    ]


def test_weighted_score_prioritizes_large_cases() -> None:
    assert weighted_score(_rows((1.0, 9.0))) > 8.9


def test_report_contains_pairwise_buckets_and_promotion_decision() -> None:
    report = build_report(
        {"fallback": _rows(), "analytic": _rows((1.5, 2.5))},
        baseline="fallback",
        provenance={"git_commit": "abc"},
    )

    assert report["schema_version"] == 1
    assert report["bucket_summary"]["analytic"]["116-120"]["cases"] == 1
    assert report["large_case_summary"]["analytic"]["106-120"]["cases"] == 1
    assert report["comparisons"]["analytic"]["improved_cases"] == 2
    assert report["comparisons"]["analytic"]["per_case_delta"][1]["test_id"] == 1
    assert report["promotion_decisions"]["analytic"] == "PROMOTE"


def test_capped_feasible_is_hold_and_infeasible_is_reject() -> None:
    capped = build_report(
        {"fallback": _rows((9.999999, 9.999999)), "analytic": _rows((9.999999, 9.999999))},
        baseline="fallback",
    )
    rejected = build_report(
        {"fallback": _rows(), "analytic": _rows((1.0, 2.0), (True, False))},
        baseline="fallback",
    )

    assert capped["lane_summary"]["analytic"]["capped_feasible_cases"] == 2
    assert capped["promotion_decisions"]["analytic"] == "HOLD"
    assert rejected["promotion_decisions"]["analytic"] == "REJECT"


def test_promotion_holds_on_runtime_or_large_case_regression() -> None:
    slow = _rows((1.5, 2.5))
    for row in slow:
        row["runtime_seconds"] *= 2.0
    large_regression = _rows((1.0, 3.5))

    slow_report = build_report({"fallback": _rows(), "analytic": slow}, baseline="fallback")
    large_report = build_report(
        {"fallback": _rows(), "analytic": large_regression}, baseline="fallback"
    )

    assert slow_report["promotion_decisions"]["analytic"] == "HOLD"
    assert large_report["promotion_decisions"]["analytic"] == "HOLD"


def test_result_cli_writes_report_and_html(tmp_path: Path) -> None:
    fallback = tmp_path / "fallback.json"
    analytic = tmp_path / "analytic.json"
    output = tmp_path / "report.json"
    plots = tmp_path / "plots"
    fallback.write_text(json.dumps({"test_results": _rows()}), encoding="utf-8")
    analytic.write_text(json.dumps({"test_results": _rows((1.5, 2.5))}), encoding="utf-8")

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/benchmark_hcfp.py",
            "--result",
            f"fallback={fallback}",
            "--result",
            f"analytic={analytic}",
            "--baseline",
            "fallback",
            "--output",
            str(output),
            "--visualize-dir",
            str(plots),
        ],
        cwd=Path(__file__).resolve().parents[1],
        check=True,
        text=True,
        capture_output=True,
    )

    assert str(output) in completed.stdout
    assert json.loads(output.read_text(encoding="utf-8"))["promotion_decisions"]["analytic"] == "PROMOTE"
    assert (plots / "case_0.html").read_text(encoding="utf-8").startswith("<!doctype html>")


def test_report_rejects_mismatched_case_ids() -> None:
    other = _rows()
    other[1]["test_id"] = 99
    with pytest.raises(ValueError, match="same test ids"):
        build_report({"fallback": _rows(), "analytic": other}, baseline="fallback")
