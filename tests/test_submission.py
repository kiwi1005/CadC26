from __future__ import annotations

import json
import math
import os
import subprocess
import sys

import pytest

from hcfp.runtime import HCFPRuntime, SolveCase
from submission.optimizer import HCFPOptimizer, solve


def _case_payload():
    return {
        "block_count": 3,
        "area_targets": [4.0, 9.0, 16.0],
        "b2b_connectivity": [],
        "p2b_connectivity": [],
        "pins_pos": [],
        "constraints": [
            [0, 1, 0, 0, 0],
            [1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
        ],
        "target_positions": [
            [10.0, 5.0, 2.0, 2.0],
            [0.0, 0.0, 3.0, 3.0],
            [-1.0, -1.0, -1.0, -1.0],
        ],
    }


def test_official_solve_contract_preserves_hard_targets_and_float_tuples():
    payload = _case_payload()

    placements = solve(**payload)

    assert len(placements) == 3
    assert placements[0] == (10.0, 5.0, 2.0, 2.0)
    assert placements[1][2:] == (3.0, 3.0)
    assert all(isinstance(rect, tuple) for rect in placements)
    assert all(isinstance(value, float) for rect in placements for value in rect)
    assert all(math.isfinite(value) for rect in placements for value in rect)


def test_runtime_uses_injected_solver_when_valid():
    def solver(case: SolveCase):
        assert case.block_count == 2
        return [(0, 0, 1, 1), (1, 0, 2, 2)]

    runtime = HCFPRuntime(solver=solver)

    assert runtime.solve(2, [1, 4], [], [], [], []) == [(0.0, 0.0, 1.0, 1.0), (1.0, 0.0, 2.0, 2.0)]


def test_runtime_falls_back_on_exception_and_non_finite_output():
    def broken_solver(case: SolveCase):
        return [(0, 0, float("nan"), 1)]

    runtime = HCFPRuntime(solver=broken_solver)

    placements = runtime.solve(1, [25.0], [], [], [], [])

    assert len(placements) == 1
    assert placements[0][2:] == (5.0, 5.0)
    assert all(math.isfinite(value) for value in placements[0])


def test_runtime_rejects_finite_but_hard_infeasible_output():
    def overlapping_solver(case: SolveCase):
        return [(0.0, 0.0, 2.0, 2.0), (0.0, 0.0, 2.0, 2.0)]

    runtime = HCFPRuntime(solver=overlapping_solver)
    placements = runtime.solve(2, [4.0, 4.0], [], [], [], [[0, 0, 0, 0, 0]] * 2)

    assert placements[0][0] != placements[1][0]


def test_runtime_fails_closed_for_impossible_or_missing_hard_anchors():
    runtime = HCFPRuntime(solver=None)
    overlapping_targets = [[0.0, 0.0, 2.0, 2.0], [1.0, 1.0, 2.0, 2.0]]
    preplaced = [[0, 1, 0, 0, 0], [0, 1, 0, 0, 0]]

    with pytest.raises(RuntimeError, match="no hard-feasible fallback"):
        runtime.solve(2, [4.0, 4.0], [], [], [], preplaced, overlapping_targets)

    with pytest.raises(ValueError, match="target_positions is required"):
        runtime.solve(1, [4.0], [], [], [], [[0, 1, 0, 0, 0]])


def test_optimizer_class_accepts_optional_target_positions():
    payload = _case_payload()
    optimizer = HCFPOptimizer()

    placements = optimizer.solve(
        payload["block_count"],
        payload["area_targets"],
        payload["b2b_connectivity"],
        payload["p2b_connectivity"],
        payload["pins_pos"],
        payload["constraints"],
        payload["target_positions"],
    )

    assert placements[0] == (10.0, 5.0, 2.0, 2.0)


def test_binary_main_accepts_kwargs_json_and_emits_placements():
    payload = {"kwargs": _case_payload()}
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{os.getcwd()}/src:{os.getcwd()}:{env.get('PYTHONPATH', '')}"

    completed = subprocess.run(
        [sys.executable, "submission/binary_main.py"],
        input=json.dumps(payload),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env,
        check=True,
    )

    output = json.loads(completed.stdout)
    assert output["placements"][0] == [10.0, 5.0, 2.0, 2.0]
    assert len(output["placements"]) == 3


def test_optimizer_imports_when_cwd_is_submission_dir():
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            "import optimizer; print(callable(optimizer.solve))",
        ],
        cwd="submission",
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
    )

    assert completed.stdout.strip() == "True"
