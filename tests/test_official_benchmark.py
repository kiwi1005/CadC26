from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import pytest

from hcfp.checkpoint import RUNTIME_NORMALIZATION, save_checkpoint
from hcfp.model import HCFPModel, ModelConfig


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
    assert len(report["case_metadata"]["0"]["constraints"]) == 21


@pytest.mark.skipif(not (DATA / "LiteTensorDataTest").is_dir(), reason="official validation cache is unavailable")
def test_official_benchmark_audits_valid_checkpoint_usage(tmp_path: Path) -> None:
    checkpoint = tmp_path / "model.pt"
    checkpoint_hash = save_checkpoint(
        HCFPModel(ModelConfig(hidden_dim=16, encoder_layers=1)),
        checkpoint,
        RUNTIME_NORMALIZATION,
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
    assert report["lane_metadata"]["learned"]["flow_steps"] == 2
    assert report["lane_metadata"]["learned"]["tail_topk"] == 1
