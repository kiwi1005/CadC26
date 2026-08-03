from __future__ import annotations

import importlib.util
import math
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from hcfp.constraints.raw_repair import RawConstraintRepair
from hcfp.score_attribution import attribute_score


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/audit_hcfp_constraint_raw.py"
SPEC = importlib.util.spec_from_file_location("constraint_raw_audit", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit)


class _Evaluator:
    @staticmethod
    def evaluate_solution(solution, *_args, **_kwargs):
        positions = solution["positions"]
        hpwl_gap = float(positions[0][0]) / 100.0
        score = attribute_score(
            hpwl_gap,
            0.0,
            boundary_violations=1,
            grouping_violations=2,
            mib_violations=0,
            max_possible_violations=10,
        )
        return SimpleNamespace(
            is_feasible=True,
            overlap_violations=0,
            area_violations=0,
            dimension_violations=0,
            fixed_violations=0,
            preplaced_violations=0,
            hpwl_total=10.0,
            bbox_area=20.0,
            hpwl_gap=hpwl_gap,
            area_gap=0.0,
            boundary_violations=1,
            grouping_violations=2,
            mib_violations=0,
            total_soft_violations=3,
            max_possible_violations=10,
            violations_relative=0.3,
            cost=score.official_capped_cost,
        )


def test_candidate_pairs_classify_and_repair_constraint_sources(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw = torch.tensor(
        [
            [[0.0, 0.0, 1.0, 1.0]],
            [[10.0, 0.0, 1.0, 1.0]],
            [[20.0, 0.0, 1.0, 1.0]],
        ]
    )
    projected = raw.clone()
    projected[:, 0, 0] += torch.tensor([1.0, 2.0, 3.0])
    repair_calls: list[dict[str, object]] = []

    monkeypatch.setattr(
        audit,
        "to_official_placements",
        lambda _source, _case, boxes: boxes.tolist(),
    )

    def repair(_source, placements, record):
        repair_calls.append(record)
        rows = torch.as_tensor(placements, dtype=torch.float64).clone()
        rows[:, 1] += 1.0
        return RawConstraintRepair(
            tuple(tuple(float(value) for value in row) for row in rows.tolist()),
            group_edges_applied=1,
            group_edges_rejected=0,
            boundary_blocks_applied=0,
            boundary_blocks_rejected=1,
        )

    monkeypatch.setattr(audit, "repair_raw_constraints", repair)
    record = {"source": "candidate_2", "details": {}}

    raw_rows, projected_rows = audit._candidate_pair_records(
        _Evaluator,
        {},
        object(),
        raw,
        projected,
        ("fallback", "learned_initial", "learned_initial"),
        frozenset((1,)),
        frozenset((2,)),
        {2: record},
        (),
    )

    assert [row["candidate_type"] for row in raw_rows] == [
        "fallback",
        "topology",
        "constraint",
    ]
    assert repair_calls == [record, record]
    assert raw_rows[2]["raw_constraint_repair"]["group_edges_applied"] == 1
    assert projected_rows[2]["raw_constraint_repair"]["boundary_blocks_rejected"] == 1
    assert raw_rows[2]["projection_displacement"] == pytest.approx(3.0)
    assert projected_rows[2]["projection_displacement"] == pytest.approx(3.0)
    assert raw_rows[2]["boundary_violations"] == 1
    assert raw_rows[2]["grouping_violations"] == 2
    assert raw_rows[2]["constraint_kind"] == "unknown"
    assert len(raw_rows[2]["placement_sha256"]) == 64
    assert raw_rows[2]["cap_margin"] == pytest.approx(
        math.log(10.0) - raw_rows[2]["log_uncapped_cost"]
    )
    assert raw_rows[2]["log_uncapped_cost"] == pytest.approx(
        math.log(raw_rows[2]["uncapped_cost"])
    )


def test_projection_displacement_is_sum_of_raw_xy_l2_norms() -> None:
    raw = [(0.0, 0.0, 2.0, 2.0), (5.0, 5.0, 1.0, 1.0)]
    projected = [(3.0, 4.0, 2.0, 2.0), (5.0, 7.0, 1.0, 1.0)]

    assert audit._projection_displacement(raw, projected) == pytest.approx(7.0)


def test_placement_hash_is_stable_and_geometry_sensitive() -> None:
    first = [(0.0, 1.0, 2.0, 3.0)]
    second = [(0.0, 1.0, 2.0, 4.0)]

    assert audit._placement_sha256(first) == audit._placement_sha256(list(first))
    assert audit._placement_sha256(first) != audit._placement_sha256(second)


def test_analysis_and_constraint_provenance_fail_closed() -> None:
    analysis = SimpleNamespace(
        result=SimpleNamespace(
            used_checkpoint=True,
            checkpoint_hash="state-hash",
            failure_reason=None,
            topology_seed_count=2,
            constraint_seed_count=1,
        ),
        analytic=SimpleNamespace(
            incumbent_snapshot={
                "constraint_seed_failure_reason": "construction shortfall"
            }
        ),
    )

    with pytest.raises(
        RuntimeError,
        match=r"requested 2 constraint seeds, produced 1: construction shortfall",
    ):
        audit._validate_analysis("heldout/a", analysis, "state-hash", 2, 2)

    snapshot = {
        "constraint_seed_sources": ("candidate_3", "candidate_7"),
        "constraint_seed_provenance": (
            {"source": "candidate_3"},
            {"source": "candidate_8"},
        ),
    }
    indices = audit._seed_indices("heldout/a", snapshot, "constraint_seed_sources", 2)
    with pytest.raises(RuntimeError, match="do not match"):
        audit._constraint_records("heldout/a", snapshot, indices)


def _candidate(
    index: int,
    candidate_type: str,
    log_cost: float,
    displacement: float,
) -> dict[str, object]:
    return {
        "candidate_index": index,
        "source": "analytic_initial",
        "candidate_type": candidate_type,
        "hard_feasible": True,
        "boundary_violations": index,
        "grouping_violations": 1,
        "mib_violations": 0,
        "log_uncapped_cost": log_cost,
        "projection_displacement": displacement,
    }


def _case(
    test_id: int,
    blocks: int,
    *,
    analytic: float,
    topology: float,
    constraint: float,
    selected: float,
    topology_displacement: float,
    constraint_displacement: float,
) -> dict[str, object]:
    candidates = [
        _candidate(1, "analytic", analytic, 0.0),
        _candidate(2, "topology", topology, topology_displacement),
        _candidate(3, "constraint", constraint, constraint_displacement),
    ]
    stage = {"candidates": candidates, "oracles": audit._oracles(candidates)}
    return {
        "test_id": test_id,
        "block_count": blocks,
        "raw": stage,
        "post_bdp": stage,
        "selected": {
            "hard_feasible": True,
            "log_uncapped_cost": selected,
        },
    }


def test_summary_reports_q2_weighted_gates_and_selected_gain() -> None:
    cases = [
        _case(
            0,
            100,
            analytic=3.0,
            topology=2.0,
            constraint=1.0,
            selected=0.5,
            topology_displacement=4.0,
            constraint_displacement=2.0,
        ),
        _case(
            1,
            120,
            analytic=4.0,
            topology=3.0,
            constraint=3.5,
            selected=2.0,
            topology_displacement=8.0,
            constraint_displacement=5.0,
        ),
    ]

    summary = audit._summary(cases)
    gains = summary["topology_vs_constraint"]["post_bdp"]
    weight = math.exp((100 - 120) / 12.0)
    expected_weighted_gain = (1.0 * weight - 0.5) / (weight + 1.0)

    assert gains["comparable_cases"] == 2
    assert gains["constraint_better_cases"] == 1
    assert gains["topology_better_cases"] == 1
    assert gains["mean_constraint_j_gain"] == pytest.approx(0.25)
    assert gains["weighted_mean_constraint_j_gain"] == pytest.approx(
        expected_weighted_gain
    )
    assert summary["selected_vs_analytic"]["mean_selected_j_gain"] == pytest.approx(
        2.25
    )
    displacement = summary["projection_displacement"]
    assert displacement["topology"]["mean"] == pytest.approx(6.0)
    assert displacement["constraint"]["mean"] == pytest.approx(3.5)
    assert displacement["constraint_minus_topology"]["mean"] == pytest.approx(-2.5)
    assert displacement["constraint"]["weighted_mean"] == pytest.approx(
        (2.0 * weight + 5.0) / (weight + 1.0)
    )
    assert summary["oracle"]["post_bdp"]["constraint"]["total_boundary_violations"] == 6
    assert summary["hard_feasibility"]["raw"]["hard_feasible_by_type"] == {
        "fallback": 0,
        "analytic": 2,
        "learned_residual": 0,
        "topology": 2,
        "constraint": 2,
    }
