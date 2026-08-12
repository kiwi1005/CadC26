from __future__ import annotations

import json
import math
from pathlib import Path
import subprocess
import sys

import pytest

from hcfp.cap_margin import attribute_record, build_cap_report, render_markdown
from hcfp.score_attribution import (
    CAP_LOG,
    attribute_score,
    attribute_score_from_relative,
)
from hcfp.verify import compute_cost


def _metrics(*, hpwl_gap: float = 0.0, feasible: bool = True) -> dict[str, object]:
    return {
        "hard_feasible": feasible,
        "hpwl_gap": hpwl_gap,
        "area_gap": 0.0,
        "boundary_violations": 0,
        "grouping_violations": 0,
        "mib_violations": 0,
        "max_possible_violations": 0,
    }


def test_exact_components_reconstruct_pinned_official_uncapped_cost() -> None:
    result = attribute_score(
        0.4,
        0.2,
        boundary_violations=2,
        grouping_violations=1,
        mib_violations=3,
        max_possible_violations=10,
        runtime_factor=1.7,
    )
    expected_runtime = max(0.7, 1.7**0.3)
    expected = 1.3 * math.exp(1.2) * expected_runtime

    assert result.quality_factor == pytest.approx(1.3)
    assert result.boundary_contribution == pytest.approx(0.4)
    assert result.grouping_contribution == pytest.approx(0.2)
    assert result.mib_contribution == pytest.approx(0.6)
    assert result.boundary_contribution is not None
    assert result.grouping_contribution is not None
    assert result.mib_contribution is not None
    assert result.uncapped_cost == pytest.approx(expected)
    assert result.log_uncapped_cost == pytest.approx(math.log(expected))
    assert result.log_uncapped_cost == pytest.approx(
        result.quality_contribution
        + result.boundary_contribution
        + result.grouping_contribution
        + result.mib_contribution
        + result.runtime_contribution
    )
    assert result.cap_margin == pytest.approx(math.log(10.0) - math.log(expected))
    assert result.official_capped_cost == pytest.approx(
        compute_cost(0.4, 0.2, 0.6, 1.7, True)
    )


def test_counterfactual_fix_requirements_and_blockers() -> None:
    mixed = attribute_score(
        2.0,
        0.0,
        boundary_violations=4,
        grouping_violations=3,
        mib_violations=3,
        max_possible_violations=10,
    )
    quality = attribute_score(
        20.0,
        0.0,
        boundary_violations=0,
        grouping_violations=0,
        mib_violations=0,
        max_possible_violations=0,
    )
    hard = attribute_score(
        0.0,
        0.0,
        boundary_violations=0,
        grouping_violations=0,
        mib_violations=0,
        max_possible_violations=0,
        hard_feasible=False,
    )

    assert mixed.log_uncapped_cost > CAP_LOG
    assert mixed.required_soft_fixes_to_uncap == 2
    target_gap = (math.exp(CAP_LOG - 2.0) - 1.0) / 0.5
    assert mixed.required_quality_gap_to_uncap == pytest.approx(2.0 - target_gap)
    assert mixed.blocker_classification == "mixed"
    assert quality.required_soft_fixes_to_uncap is None
    assert not quality.soft_fixes_sufficient
    assert quality.required_quality_gap_to_uncap == pytest.approx(2.0)
    assert quality.blocker_classification == "quality"
    assert hard.blocker_classification == "hard"
    assert hard.required_quality_gap_to_uncap is None


def test_soft_fix_requirement_is_preserved_when_existing_fixes_are_insufficient() -> None:
    result = attribute_score(
        20.0,
        0.0,
        boundary_violations=1,
        grouping_violations=0,
        mib_violations=0,
        max_possible_violations=100,
    )

    assert result.max_possible_violations is not None
    assert result.total_soft_violations is not None
    expected = math.ceil(
        max(0.0, result.log_uncapped_cost - CAP_LOG)
        * result.max_possible_violations
        / 2.0
    )
    assert result.required_soft_fixes_to_uncap == expected
    assert expected > result.total_soft_violations
    assert not result.soft_fixes_sufficient


def test_relative_constructor_marks_unknown_soft_split() -> None:
    result = attribute_score_from_relative(0.3, -0.1, 0.25)

    assert result.violations_relative == 0.25
    assert result.soft_contribution == 0.5
    assert result.boundary_contribution is None
    assert result.grouping_contribution is None
    assert result.mib_contribution is None
    assert not result.soft_breakdown_available


def test_inconsistent_or_nonfinite_inputs_are_rejected() -> None:
    with pytest.raises(ValueError, match="cannot exceed"):
        attribute_score(
            0.0,
            0.0,
            boundary_violations=2,
            grouping_violations=0,
            mib_violations=0,
            max_possible_violations=1,
        )
    with pytest.raises(ValueError, match="inconsistent"):
        attribute_record(
            {
                **_metrics(),
                "boundary_violations": 1,
                "max_possible_violations": 2,
                "violations_relative": 0.75,
            }
        )
    with pytest.raises(ValueError, match="finite"):
        attribute_score_from_relative(math.inf, 0.0, 0.0)


def test_staged_report_marks_only_complete_cap_crossing_as_projection() -> None:
    report = build_cap_report(
        {
            "cases": [
                {
                    "test_id": 7,
                    "block_count": 64,
                    "source": "analytic_initial",
                    "repair_displacement": 3.5,
                    "raw": _metrics(),
                    "projected": _metrics(hpwl_gap=20.0),
                    "post": _metrics(hpwl_gap=20.0),
                }
            ]
        }
    )
    case = report["cases"][0]

    assert case["stages"]["raw"]["cap_margin"] > 0.0
    assert case["stages"]["projected"]["cap_margin"] < 0.0
    assert case["projection_dominated"]
    assert case["blocker_classification"] == "projection"
    assert report["summary"]["projection_dominated_cases"] == 1
    assert "analytic_initial" in report["summary"]["classification_counts_by_source"]
    assert "| 7 | 64 | analytic_initial |" in render_markdown(report)


def test_projection_classification_catches_j_regression_without_cap_crossing() -> None:
    report = build_cap_report(
        {
            "cases": [
                {
                    "test_id": 8,
                    "raw": _metrics(hpwl_gap=0.0),
                    "projected": _metrics(hpwl_gap=0.1),
                    "post": _metrics(hpwl_gap=0.2),
                }
            ]
        }
    )

    case = report["cases"][0]
    assert case["raw_cap_margin"] > 0.0
    assert case["post_repair_cap_margin"] == case["post_cap_margin"]
    assert case["post_cap_margin"] > 0.0
    assert case["projection_dominated"]
    assert case["blocker_classification"] == "projection"


def test_existing_oracle_schema_uses_paired_incumbent_candidate() -> None:
    raw = {
        "candidate_index": 2,
        "source": "learned_initial",
        "hard_feasible": True,
        "hpwl_gap": 0.0,
        "area_gap": 0.0,
        "violations_relative": 0.0,
    }
    projected = {**raw, "hpwl_gap": 20.0}
    report = build_cap_report(
        {
            "cases": [
                {
                    "test_id": 4,
                    "block_count": 21,
                    "raw": {"candidates": [raw]},
                    "post_bdp": {"candidates": [projected]},
                    "incumbent": projected,
                }
            ]
        }
    )

    case = report["cases"][0]
    assert report["input_schema"] == "oracle_report"
    assert case["candidate_index"] == 2
    assert case["projection_dominated"]
    assert case["boundary_contribution"] is None
    assert report["schema_limitations"]


def test_cli_writes_json_and_markdown_from_minimal_schema(tmp_path: Path) -> None:
    source = tmp_path / "input.json"
    output = tmp_path / "cap.json"
    source.write_text(
        json.dumps(
            {
                "cases": [
                    {
                        "test_id": 9,
                        "block_count": 96,
                        "source": "analytic",
                        **_metrics(hpwl_gap=20.0),
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/report_cap_sources.py",
            "--input",
            str(source),
            "--output",
            str(output),
        ],
        cwd=Path(__file__).resolve().parents[1],
        check=True,
        text=True,
        capture_output=True,
    )

    assert str(output) in completed.stdout
    assert (
        json.loads(output.read_text(encoding="utf-8"))["cases"][0][
            "blocker_classification"
        ]
        == "quality"
    )
    assert (
        output.with_suffix(".md")
        .read_text(encoding="utf-8")
        .startswith("# HCFP exact cap attribution")
    )


def test_cli_can_select_one_benchmark_lane(tmp_path: Path) -> None:
    source = tmp_path / "benchmark.json"
    output = tmp_path / "cap.json"
    source.write_text(
        json.dumps(
            {
                "lanes": {
                    "analytic": [{"test_id": 1, **_metrics()}],
                    "learned": [{"test_id": 1, **_metrics(hpwl_gap=0.2)}],
                }
            }
        ),
        encoding="utf-8",
    )

    subprocess.run(
        [
            sys.executable,
            "scripts/report_cap_sources.py",
            "--input",
            str(source),
            "--output",
            str(output),
            "--lane",
            "learned",
        ],
        cwd=Path(__file__).resolve().parents[1],
        check=True,
        text=True,
        capture_output=True,
    )

    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["summary"]["cases"] == 1
    assert report["cases"][0]["lane"] == "learned"


def test_cli_adds_large_case_evidence_aliases_and_primary_blocker(
    tmp_path: Path,
) -> None:
    source = tmp_path / "large15.json"
    output = tmp_path / "attribution.json"
    source.write_text(
        json.dumps(
            {
                "lanes": {
                    "learned": [
                        {
                            "test_id": 85,
                            "block_count": 106,
                            "source": "learned_initial",
                            "is_feasible": True,
                            "hpwl_gap": 0.0,
                            "area_gap": 20.0,
                            "boundary_violations": 0,
                            "grouping_violations": 0,
                            "mib_violations": 0,
                            "max_soft_violations": 0,
                            "positions": [
                                [0.0, 0.0, 2.0, 2.0],
                                [2.0, 0.0, 2.0, 2.0],
                            ],
                            "raw_cap_margin": 0.4,
                            "projected_cap_margin": -0.2,
                            "final_cap_margin": -0.2,
                            "projection_displacement": 1.5,
                        }
                    ]
                }
            }
        ),
        encoding="utf-8",
    )

    subprocess.run(
        [
            sys.executable,
            "scripts/report_cap_sources.py",
            "--input",
            str(source),
            "--output",
            str(output),
            "--lane",
            "learned",
        ],
        cwd=Path(__file__).resolve().parents[1],
        check=True,
        text=True,
        capture_output=True,
    )

    case = json.loads(output.read_text(encoding="utf-8"))["cases"][0]
    assert case["hard_feasible"]
    assert case["quality_factor"] == pytest.approx(11.0)
    assert case["boundary_violations"] == 0
    assert case["grouping_violations"] == 0
    assert case["mib_violations"] == 0
    assert case["max_possible_violations"] == 0
    assert case["uncapped_cost"] == pytest.approx(11.0)
    assert case["capped_cost"] == pytest.approx(9.999999)
    assert case["raw_cap_margin"] == pytest.approx(0.4)
    assert case["projected_cap_margin"] == pytest.approx(-0.2)
    assert case["final_cap_margin"] == pytest.approx(-0.2)
    assert case["projection_dominated"]
    assert case["utilization"] == pytest.approx(1.0)
    assert case["projection_displacement"] == pytest.approx(1.5)
    assert case["candidate_source"] == "learned_initial"
    assert case["primary_blocker"] == "area"
    assert case["primary_blocker_classification"] == "area"


def test_cli_reconstructs_large15_soft_breakdown_from_case_metadata(
    tmp_path: Path,
) -> None:
    source = (
        Path(__file__).resolve().parents[1]
        / "artifacts/benchmarks/"
        / "hcfp5090-q2-structure-large-s3000-seed6501-"
        "constraints16-official-large15-exact.json"
    )
    if not source.is_file():
        pytest.skip("large15 benchmark artifact is not present")
    output = tmp_path / "large15-attribution.json"

    subprocess.run(
        [
            sys.executable,
            "scripts/report_cap_sources.py",
            "--input",
            str(source),
            "--output",
            str(output),
            "--lane",
            "learned",
        ],
        cwd=Path(__file__).resolve().parents[1],
        check=True,
        text=True,
        capture_output=True,
    )

    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["summary"]["cases"] == 15
    assert report["summary"]["soft_breakdown_available_cases"] == 15
    assert not any(
        str(item).startswith("Some inputs expose only violations_relative")
        for item in report["schema_limitations"]
    )
    cases = {case["test_id"]: case for case in report["cases"]}
    case = cases[85]
    assert case["primary_blocker_classification"] == "none"
    assert (case["boundary_violations"], case["grouping_violations"], case["mib_violations"]) == (
        23,
        2,
        0,
    )
    assert case["max_possible_violations"] == 64
    assert case["violations_relative"] == pytest.approx(25 / 64)
    assert case["soft_breakdown_source"] == "case_metadata"
    assert case["candidate_source"] == "learned"
    assert case["utilization"] == pytest.approx(0.23511499, abs=1.0e-7)
