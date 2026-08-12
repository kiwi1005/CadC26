from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "audit_outline_recovery", ROOT / "scripts/audit_outline_recovery.py"
)
assert SPEC is not None and SPEC.loader is not None
AUDIT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(AUDIT)


def test_audit_layout_separates_gold_bbox_and_pin_perimeter() -> None:
    area_constraints = torch.tensor(
        [
            [4.0, 0, 1, 0, 0, 0],
            [4.0, 0, 0, 0, 0, 0],
        ]
    )
    pins = torch.tensor([[0.0, 0.0], [4.0, 0.0], [0.0, 4.0], [4.0, 4.0]])
    fp_sol = torch.tensor([[2.0, 2.0, 0.0, 0.0], [2.0, 2.0, 2.0, 0.0]])

    row = AUDIT.audit_layout("worker/layout:0", area_constraints, pins, fp_sol)

    assert row["hypotheses"] >= 4
    assert row["oracle"]["pin_perimeter_side_recovery"] == 1.0
    assert row["oracle"]["gold_outline_side_recovery"] >= 0.5
    assert row["oracle"]["area_relative_error"] < 0.06
    assert row["all_hypotheses_contain_preplaced"]


def test_report_gates_are_explicit_and_hash_sample_order() -> None:
    record = {
        "sample_id": "worker/layout:0",
        "block_count": 120,
        "pin_count": 12,
        "preplaced_density": 0.1,
        "boundary_density": 0.25,
        "hypotheses": 4,
        "all_hypotheses_contain_preplaced": True,
        "top1": {
            "area_relative_error": 0.006,
            "width_relative_error": 0.02,
            "height_relative_error": 0.02,
            "pin_perimeter_side_recovery": 0.75,
            "pin_side_coverage": 0.9,
            "side_coverage": 0.75,
            "gold_outline_side_recovery": 0.75,
            "gold_outside_block_ratio": 0.1,
        },
        "oracle": {
            "area_relative_error": 0.005,
            "width_relative_error": 0.01,
            "height_relative_error": 0.01,
            "max_dimension_relative_error": 0.01,
            "pin_perimeter_side_recovery": 1.0,
            "pin_side_coverage": 1.0,
            "side_coverage": 1.0,
            "gold_outline_side_recovery": 1.0,
            "gold_outside_block_ratio": 0.0,
        },
    }

    report = AUDIT.build_report([record], provenance={"git_commit": "abc"})

    assert report["gates"]["passed"]
    assert report["summary"]["cases"] == 1
    assert report["summary"]["top1_area_relative_error"]["median"] == 0.006
    assert report["provenance"]["sample_id_sha256"]
    second = AUDIT.build_report([record], provenance={"git_commit": "different"})
    assert report["provenance"]["summary_sha256"] == second["provenance"]["summary_sha256"]
    assert report == AUDIT.build_report([record], provenance={"git_commit": "abc"})
    assert report["definitions"]["oracle_at_k"].startswith("best hypothesis")
    assert report["buckets"]["block_count"]["116-120"]["cases"] == 1
    assert report["buckets"]["pin_count"]["9-32"]["cases"] == 1
    assert report["buckets"]["preplaced_density"]["(0.05,0.20]"]["cases"] == 1
    assert report["buckets"]["boundary_density"]["(0.20,1]"]["cases"] == 1

    incomplete = AUDIT.build_report(
        [record], provenance={"git_commit": "abc", "requested_cases": 2}
    )
    assert not incomplete["gates"]["audited_cases_eq_requested"]
    assert not incomplete["gates"]["passed"]


def test_layout_files_reject_visible_validation(tmp_path: Path) -> None:
    path = tmp_path / "visible" / "floorset_lite"
    path.mkdir(parents=True)

    with pytest.raises(ValueError, match="forbidden"):
        AUDIT._layout_files(path)


def test_git_provenance_binds_audit_sources() -> None:
    provenance = AUDIT._git_provenance()

    assert len(provenance["git_commit"]) == 40
    assert set(provenance["source_sha256"]) == {
        "scripts/audit_outline_recovery.py",
        "src/hcfp/outline_inference.py",
    }
    assert all(len(value) == 64 for value in provenance["source_sha256"].values())


def test_movable_gold_geometry_never_changes_inference_hypothesis() -> None:
    area_constraints = torch.tensor(
        [
            [4.0, 0, 1, 0, 0, 0],
            [4.0, 0, 0, 0, 0, 0],
        ]
    )
    pins = torch.tensor([[0.0, 0.0], [4.0, 0.0], [0.0, 4.0], [4.0, 4.0]])
    near = torch.tensor([[2.0, 2.0, 0.0, 0.0], [2.0, 2.0, 2.0, 0.0]])
    far = torch.tensor([[2.0, 2.0, 0.0, 0.0], [2.0, 2.0, 100.0, 100.0]])

    first = AUDIT.audit_layout("near", area_constraints, pins, near)
    second = AUDIT.audit_layout("far", area_constraints, pins, far)

    assert first["top1"]["hypothesis_id"] == second["top1"]["hypothesis_id"]
    assert first["oracle"]["area_relative_error"] != second["oracle"]["area_relative_error"]
