from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "artifacts/floorset-v10"


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
