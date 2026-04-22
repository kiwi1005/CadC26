#!/usr/bin/env python3
# ruff: noqa: E402, I001
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any, cast

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
CONTEST_ROOT = ROOT / "external" / "FloorSet" / "iccad2026contest"

for path in (ROOT, SRC, CONTEST_ROOT):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from iccad2026_evaluate import validate_submission
from scripts.download_smoke import _auto_approve_downloads, _ensure_import_paths


def _run_command(command: list[str]) -> dict[str, object]:
    completed = subprocess.run(
        command,
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    return {
        "command": command,
        "returncode": completed.returncode,
        "stdout": completed.stdout[-4000:],
        "stderr": completed.stderr[-4000:],
        "passed": completed.returncode == 0,
    }


def main() -> None:
    _ensure_import_paths()
    _auto_approve_downloads()
    python = sys.executable

    steps: list[dict[str, Any]] = [
        _run_command(
            [
                python,
                "-m",
                "pytest",
                "-q",
                "tests/test_geometry.py",
                "tests/test_candidate_masks.py",
                "tests/test_bc_training.py",
                "tests/test_rollout_smoke.py",
                "tests/test_eval_reports.py",
                "tests/test_feedback_awbc.py",
                "tests/test_contest_optimizer.py",
                "tests/test_repair_finalizer.py",
            ]
        ),
        _run_command(
            [
                python,
                "-m",
                "ruff",
                "check",
                "src/puzzleplace/actions/candidates.py",
                "src/puzzleplace/rollout/greedy.py",
                "src/puzzleplace/rollout/beam.py",
                "src/puzzleplace/eval",
                "src/puzzleplace/feedback",
                "src/puzzleplace/optimizer",
                "scripts/train_awbc_small.py",
                "scripts/evaluate_contest_optimizer.py",
                "scripts/run_smoke_regression_matrix.py",
                "contest_optimizer.py",
                "tests/test_eval_reports.py",
                "tests/test_feedback_awbc.py",
                "tests/test_contest_optimizer.py",
            ]
        ),
        _run_command(
            [
                python,
                "-m",
                "mypy",
                "src/puzzleplace/eval",
                "src/puzzleplace/feedback",
                "src/puzzleplace/optimizer",
                "scripts/train_awbc_small.py",
                "scripts/evaluate_contest_optimizer.py",
                "contest_optimizer.py",
            ]
        ),
        _run_command(
            [
                python,
                "scripts/check_candidate_coverage.py",
                "--case-ids",
                "0",
                "1",
                "2",
                "3",
                "4",
                "--max-traces",
                "2",
                "--modes",
                "semantic",
                "relaxed",
                "strict",
            ]
        ),
        _run_command(
            [
                "env",
                "ROLLOUT_EPOCHS=5",
                python,
                "scripts/rollout_validate.py",
                "--case-ids",
                "0",
                "1",
                "2",
                "3",
                "4",
                "--rollout-mode",
                "semantic",
            ]
        ),
        _run_command([python, "scripts/repair_validate.py", "--case-ids", "0", "1", "2", "3", "4"]),
        _run_command([python, "scripts/train_awbc_small.py"]),
        _run_command([python, "scripts/evaluate_contest_optimizer.py"]),
    ]

    validation_ok = validate_submission(
        str(ROOT / "contest_optimizer.py"), quick=True, verbose=False
    )
    payload = {
        "all_passed": all(step["passed"] for step in steps) and bool(validation_ok),
        "quick_validate": bool(validation_ok),
        "steps": steps,
    }

    report_dir = ROOT / "artifacts" / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "agent13_regression_matrix.json").write_text(json.dumps(payload, indent=2))
    (report_dir / "agent13_regression_matrix.md").write_text(
        "\n".join(
            [
                "# Agent 13 Summary",
                "",
                f"- quick validator: `{validation_ok}`",
                f"- all matrix steps passed: `{payload['all_passed']}`",
                "",
                "## Steps",
            ]
            + [
                (
                    f"- `{' '.join(cast(list[str], step['command']))}` -> "
                    f"`{'PASS' if step['passed'] else 'FAIL'}`"
                )
                for step in steps
            ]
        )
    )
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
