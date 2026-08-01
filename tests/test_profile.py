from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import pytest
import torch

from hcfp.profile import ProfileConfig, run_profile, synthetic_case
from hcfp.verify import verify_feasible


def test_synthetic_buckets_are_feasible_with_safe_fallback() -> None:
    for block_count in (32, 64, 96, 120):
        case = synthetic_case(block_count, device="cpu")
        from hcfp.fallback import safe_shelf

        assert case.n == block_count
        assert verify_feasible(case, safe_shelf(case))


def test_profile_reports_timing_candidates_and_feasible_incumbent() -> None:
    report = run_profile(
        ProfileConfig(
            block_count=32,
            candidates=3,
            steps=1,
            repeats=2,
            warmups=0,
            projection_iterations=4,
            direction_beam=1,
            device="cpu",
        )
    )

    assert report["schema_version"] == 1
    assert report["incumbent"]["feasible"] is True
    assert report["timing_seconds"]["p50"] <= report["timing_seconds"]["p95"]
    assert report["timing_seconds"]["p95"] <= report["timing_seconds"]["p99"]
    assert report["timing_seconds"]["p99"] <= report["timing_seconds"]["max"]
    assert len(report["timing_seconds"]["samples"]) == 2
    assert report["candidate_metadata"]["raw_candidates"] == 7
    assert report["candidate_metadata"]["projected_candidates"] == 7
    assert report["candidate_metadata"]["blocks"] == 32
    assert report["cuda"]["peak_bytes_max"] == 0
    assert report["phases"]["solver"] == "solve_case"
    assert report["phases"]["telemetry"] == "collected_once_outside_timing"


def test_profile_cli_writes_json_and_runs_n120_k32_smoke(tmp_path: Path) -> None:
    output = tmp_path / "profile.json"
    subprocess.run(
        [
            sys.executable,
            "scripts/profile_hcfp.py",
            "--blocks",
            "120",
            "--candidates",
            "32",
            "--steps",
            "0",
            "--repeats",
            "1",
            "--warmups",
            "0",
            "--projection-steps",
            "2",
            "--beam",
            "1",
            "--device",
            "cpu",
            "--output",
            str(output),
        ],
        cwd=Path(__file__).resolve().parents[1],
        check=True,
        text=True,
        capture_output=True,
    )
    payload = json.loads(output.read_text(encoding="utf-8"))

    assert payload["config"]["block_count"] == 120
    assert payload["config"]["candidates"] == 32
    assert payload["candidate_metadata"]["raw_candidates"] == 65
    assert payload["candidate_metadata"]["blocks"] == 120
    assert payload["incumbent"]["feasible"] is True


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_cuda_profile_records_peak_memory_and_synchronizes() -> None:
    report = run_profile(
        ProfileConfig(
            block_count=32,
            candidates=2,
            steps=1,
            repeats=1,
            warmups=0,
            projection_iterations=2,
            direction_beam=1,
            device="cuda",
        )
    )

    assert report["config"]["actual_device"].startswith("cuda")
    assert report["phases"]["synchronized_timing"] is True
    assert report["cuda"]["peak_bytes_max"] > 0
    assert report["incumbent"]["feasible"] is True
