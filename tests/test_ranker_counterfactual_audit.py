from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch

from hcfp.analytic import AnalyticResult, CandidateTelemetry
from hcfp.case import from_official
from hcfp.data import DataSample, extract_labels


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/audit_hcfp_ranker_counterfactual.py"
SPEC = importlib.util.spec_from_file_location("ranker_counterfactual_audit", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit)


def test_counterfactual_audit_cli_imports_directly() -> None:
    completed = subprocess.run(
        [sys.executable, str(SCRIPT), "--help"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr


class _Evaluator:
    calls = 0

    @staticmethod
    def evaluate_solution(solution, *_args, **_kwargs):
        _Evaluator.calls += 1
        x = float(solution["positions"][0][0])
        return SimpleNamespace(
            is_feasible=x >= 0.0,
            cost=x,
            hpwl_gap=x,
            area_gap=0.0,
            boundary_violations=0,
            grouping_violations=0,
            mib_violations=0,
            total_soft_violations=0,
            max_possible_violations=1,
        )


def _case_payload(sample_id: str = "case/a") -> dict[str, object]:
    return {
        "sample_id": sample_id,
        "block_count": 2,
        "area_targets": [4.0, 4.0],
        "b2b_connectivity": [],
        "p2b_connectivity": [],
        "pins_pos": [],
        "constraints": [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0]],
        "target_positions": None,
        "solution": [[0.0, 0.0, 2.0, 2.0], [3.0, 0.0, 2.0, 2.0]],
    }


def _sample(sample_id: str = "case/a") -> DataSample:
    payload = _case_payload(sample_id)
    case = from_official(
        payload["block_count"],
        payload["area_targets"],
        payload["b2b_connectivity"],
        payload["p2b_connectivity"],
        payload["pins_pos"],
        payload["constraints"],
        payload["target_positions"],
    )
    return DataSample(
        sample_id,
        case,
        extract_labels(case, payload["solution"], normalized=False),
    )


def _analysis(*, used_checkpoint: bool = True) -> SimpleNamespace:
    rows = torch.tensor(
        [
            [[0.0, 0.0, 2.0, 2.0], [3.0, 0.0, 2.0, 2.0]],
            [[10.0, 0.0, 2.0, 2.0], [13.0, 0.0, 2.0, 2.0]],
        ],
        dtype=torch.float32,
    )
    zeros = torch.zeros(2)
    telemetry = CandidateTelemetry(
        hard_feasible=torch.ones(2, dtype=torch.bool),
        raw_overlap=zeros,
        projected_overlap=zeros,
        overlap_components=zeros,
        projection_ok=torch.ones(2, dtype=torch.bool),
        projection_active_pairs=zeros,
        hpwl=zeros,
        bbox_area=zeros,
        soft_violation=zeros,
        projection_displacement=zeros,
        projection_failure_reasons=("", ""),
        projection_initial_pairs=zeros,
        projection_final_pairs=zeros,
        projection_component_rebuilds=zeros,
        projection_new_pairs=zeros,
        projection_resets=zeros,
        projection_beam_states=zeros,
        projection_max_component_size=zeros,
        component_proposal_available=torch.zeros(2, dtype=torch.bool),
        component_proposal_xywh=torch.zeros((2, 2, 4)),
        component_proposal_hard_ok=torch.zeros(2, dtype=torch.bool),
        component_proposal_structure_ok=torch.zeros(2, dtype=torch.bool),
        component_proposal_final_pair_count=zeros,
        component_proposal_displacement=zeros,
        component_proposal_rollback_reason=("", ""),
    )
    analytic = AnalyticResult(
        selected=rows[0],
        raw_candidates=rows,
        projected_candidates=rows,
        telemetry=telemetry,
        energy_history=torch.zeros((1, 0, 3)),
        projection_status="ok",
        incumbent_snapshot={
            "exact_source": "candidate_0",
            "ranker_shadow_top4": (
                {"source": "candidate_1", "score": -1.0, "kind": "learned"},
            ),
            "ranker_selection_counterfactual": {
                "would_accept": True,
                "source": "candidate_1",
                "metrics": (0.0, 1.0, 1.0),
                "current_metrics": (1.0, 2.0, 2.0),
                "rejection_reason": None,
            },
            "ranker_selection_evaluated_top4": (
                {"source": "candidate_1", "rejection_reason": None},
            ),
        },
    )
    return SimpleNamespace(
        result=SimpleNamespace(
            used_checkpoint=used_checkpoint,
            selected=rows[0],
        ),
        analytic=analytic,
    )


def _install_fixture(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, *, used_checkpoint: bool = True) -> Path:
    data = tmp_path / "data"
    evaluator = data / audit.OFFICIAL_FLOORSET_V10.evaluator_path
    evaluator.parent.mkdir(parents=True)
    evaluator.write_text("fixture evaluator\n", encoding="utf-8")
    monkeypatch.setattr(
        audit,
        "OFFICIAL_FLOORSET_V10",
        SimpleNamespace(
            evaluator_path=audit.OFFICIAL_FLOORSET_V10.evaluator_path,
            evaluator_sha256=audit.file_sha256(evaluator),
            commit=audit.OFFICIAL_FLOORSET_V10.commit,
        ),
    )
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"checkpoint")
    monkeypatch.setattr(audit, "_load_evaluator", lambda _path: _Evaluator)
    monkeypatch.setattr(
        audit,
        "load_checkpoint",
        lambda *_args, **_kwargs: (
            object(),
            {
                "state_hash": "state",
                "normalization": audit.RUNTIME_NORMALIZATION,
                "capabilities": {"ranker": False},
                "trained_heads": [],
            },
        ),
    )
    monkeypatch.setattr(
        audit,
        "analyze_case_with_checkpoint",
        lambda *_args, **_kwargs: _analysis(used_checkpoint=used_checkpoint),
    )
    monkeypatch.setattr(
        audit,
        "select_official_from_analysis",
        lambda *_args, **_kwargs: [(0.0, 0.0, 2.0, 2.0), (3.0, 0.0, 2.0, 2.0)],
    )
    return checkpoint


def test_target_positions_reconstructed_like_evaluator() -> None:
    source = {
        "block_count": 2,
        "area_targets": [4.0, 9.0],
        "b2b_connectivity": [],
        "p2b_connectivity": [],
        "pins_pos": [],
        "constraints": [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0]],
        "target_positions": None,
    }

    target = audit._target_positions_from_case(source)

    assert target[0] == [-1.0, -1.0, -1.0, -1.0]
    assert target[1] == [-1.0, -1.0, -1.0, -1.0]


def test_official_case_loader_reconstructs_optimizer_targets() -> None:
    constraints = torch.tensor(
        [[1, 0, 0, 0, 0], [0, 1, 0, 0, 0]],
        dtype=torch.long,
    )
    targets = torch.tensor(
        [[0.0, 0.0, 2.0, 2.0], [3.0, 0.0, 2.0, 2.0]],
        dtype=torch.float32,
    )

    class Contest:
        def __init__(self, *_args, **_kwargs) -> None:
            self.dataset = [
                {
                    "input": (
                        torch.tensor([4.0, 4.0]),
                        [],
                        [],
                        [],
                        constraints,
                    ),
                    "label": object(),
                }
            ]

        def _load_dataset(self) -> None:
            return None

        def _extract_baseline(self, *_args, **_kwargs):
            return {"area_baseline": 10.0, "hpwl_baseline": 1.0}, targets

    cases = audit._official_case_sources(
        SimpleNamespace(ContestEvaluator=Contest),
        Path("data"),
        "all",
    )

    sample, source = cases[0]
    optimizer_targets = torch.as_tensor(source["target_positions"])
    assert sample.sample_id == "official_visible:0"
    torch.testing.assert_close(
        optimizer_targets,
        torch.tensor(
            [[-1.0, -1.0, 2.0, 2.0], [3.0, 0.0, 2.0, 2.0]]
        ),
    )


def test_audit_runner_writes_atomic_report_and_preserves_returned_placement(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    checkpoint = _install_fixture(monkeypatch, tmp_path)
    case_path = tmp_path / "case.json"
    case_path.write_text(json.dumps(_case_payload()), encoding="utf-8")
    output = tmp_path / "audit.json"

    assert audit.main(
        [
            "--checkpoint",
            str(checkpoint),
            "--data-path",
            str(tmp_path / "data"),
            "--case-json",
            str(case_path),
            "--seed",
            "7",
            "--output",
            str(output),
        ]
    ) == 0

    report = json.loads(output.read_text(encoding="utf-8"))
    row = report["cases"][0]
    assert report["production_output_altered"] is False
    assert report["status"] == "complete"
    assert row["case_id"] == "case/a"
    assert row["seed"] == 7
    assert row["selected_positions"][0][0] == 0.0
    assert row["ranker_selection_counterfactual"]["would_accept"] is True
    assert row["ranker_shadow_available"] is True
    assert len(row["selected_sha256"]) == 64
    assert report["summary"]["all_hard_feasible"] is True
    assert report["summary"]["rows"] == 1
    assert report["summary"]["expected_rows"] == 1
    assert report["summary"]["coverage_complete"] is True
    assert report["summary"]["would_accept"] == 1
    assert report["summary"]["counterfactual_audit_gate_passed"] is True
    assert report["summary"]["ranker_shadow_missing"] == 0
    assert report["summary"]["unique_selected_hashes"] == 1
    assert report["summary"]["output_hashes_by_case"]["case/a"] == [
        row["selected_sha256"]
    ]


def test_audit_resume_reuses_case_seed_config_hash(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    checkpoint = _install_fixture(monkeypatch, tmp_path)
    case_path = tmp_path / "case.json"
    case_path.write_text(json.dumps(_case_payload()), encoding="utf-8")
    output = tmp_path / "audit.json"
    args = [
        "--checkpoint",
        str(checkpoint),
        "--data-path",
        str(tmp_path / "data"),
        "--case-json",
        str(case_path),
        "--seed",
        "7",
        "--output",
        str(output),
    ]
    assert audit.main(args) == 0
    monkeypatch.setattr(
        audit,
        "analyze_case_with_checkpoint",
        lambda *_args, **_kwargs: pytest.fail("resume should not recompute"),
    )

    assert audit.main([*args, "--resume"]) == 0


def test_audit_fails_closed_on_checkpoint_fallback(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    checkpoint = _install_fixture(monkeypatch, tmp_path, used_checkpoint=False)
    case_path = tmp_path / "case.json"
    case_path.write_text(json.dumps(_case_payload()), encoding="utf-8")

    with pytest.raises(RuntimeError, match="checkpoint fallback"):
        audit.main(
            [
                "--checkpoint",
                str(checkpoint),
                "--data-path",
                str(tmp_path / "data"),
                "--case-json",
                str(case_path),
                "--seed",
                "7",
                "--output",
                str(tmp_path / "audit.json"),
            ]
        )


def test_audit_records_zero_eligible_ranker_shadow_as_coverage_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    checkpoint = _install_fixture(monkeypatch, tmp_path)
    analysis = _analysis()
    analysis.analytic.incumbent_snapshot["ranker_shadow_top4"] = ()
    analysis.analytic.incumbent_snapshot["ranker_shadow_eligible_count"] = 0
    analysis.analytic.incumbent_snapshot["ranker_shadow_empty_reason"] = (
        "no_exact_eligible_candidates"
    )
    analysis.analytic.incumbent_snapshot["ranker_selection_counterfactual"] = {
        "would_accept": False,
        "rejection_reason": "no_exact_eligible_ranker_candidates",
    }
    monkeypatch.setattr(
        audit,
        "analyze_case_with_checkpoint",
        lambda *_args, **_kwargs: analysis,
    )
    case_path = tmp_path / "case.json"
    case_path.write_text(json.dumps(_case_payload()), encoding="utf-8")
    output = tmp_path / "audit.json"

    assert audit.main(
        [
            "--checkpoint",
            str(checkpoint),
            "--data-path",
            str(tmp_path / "data"),
            "--case-json",
            str(case_path),
            "--seed",
            "7",
            "--output",
            str(output),
        ]
    ) == 0

    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["cases"][0]["ranker_shadow_available"] is False
    assert report["cases"][0]["ranker_shadow_empty_reason"] == "no_exact_eligible_candidates"
    assert report["summary"]["ranker_shadow_missing"] == 0
    assert report["summary"]["ranker_shadow_zero_eligible"] == 1
    assert report["summary"]["zero_eligible_ranker_shadow_rows"] == [
        {"case_id": "case/a", "seed": 7}
    ]
    assert report["summary"]["counterfactual_audit_gate_passed"] is False


def test_audit_records_missing_ranker_shadow_telemetry_separately() -> None:
    row = {
        "case_id": "case/a",
        "seed": 7,
        "hard_feasible": True,
        "ranker_shadow_available": False,
        "ranker_shadow_eligible_count": 0,
        "selected_sha256": "a" * 64,
        "ranker_selection_counterfactual": {
            "would_accept": False,
            "rejection_reason": "missing_shadow_top4",
        },
    }

    summary = audit._summary([row], expected_rows=1)

    assert summary["ranker_shadow_missing"] == 1
    assert summary["ranker_shadow_zero_eligible"] == 0
    assert summary["missing_ranker_shadow_rows"] == [{"case_id": "case/a", "seed": 7}]
    assert summary["counterfactual_audit_gate_passed"] is False


def test_partial_counterfactual_summary_never_passes_gate() -> None:
    row = {
        "case_id": "case/a",
        "seed": 7,
        "hard_feasible": True,
        "ranker_shadow_available": True,
        "selected_sha256": "a" * 64,
        "ranker_selection_counterfactual": {"would_accept": False},
    }

    summary = audit._summary([row], expected_rows=2)

    assert summary["coverage_complete"] is False
    assert summary["counterfactual_audit_gate_passed"] is False


def test_audit_fails_closed_on_duplicate_cases_or_seeds(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    checkpoint = _install_fixture(monkeypatch, tmp_path)
    case_path = tmp_path / "case.json"
    case_path.write_text(json.dumps([_case_payload(), _case_payload()]), encoding="utf-8")

    with pytest.raises(ValueError, match="duplicate case IDs"):
        audit.main(
            [
                "--checkpoint",
                str(checkpoint),
                "--data-path",
                str(tmp_path / "data"),
                "--case-json",
                str(case_path),
                "--seed",
                "7",
                "--output",
                str(tmp_path / "audit.json"),
            ]
        )
    case_path.write_text(json.dumps(_case_payload()), encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate seeds"):
        audit.main(
            [
                "--checkpoint",
                str(checkpoint),
                "--data-path",
                str(tmp_path / "data"),
                "--case-json",
                str(case_path),
                "--seed",
                "7",
                "--seed",
                "7",
                "--output",
                str(tmp_path / "audit.json"),
            ]
        )


def test_audit_rejects_empty_case_selection(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    checkpoint = _install_fixture(monkeypatch, tmp_path)
    case_path = tmp_path / "empty.json"
    case_path.write_text("[]\n", encoding="utf-8")

    with pytest.raises(ValueError, match="selected no cases"):
        audit.main(
            [
                "--checkpoint",
                str(checkpoint),
                "--data-path",
                str(tmp_path / "data"),
                "--case-json",
                str(case_path),
                "--seed",
                "7",
                "--output",
                str(tmp_path / "audit.json"),
            ]
        )


def test_audit_fails_closed_on_evaluator_hash_mismatch(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    checkpoint = _install_fixture(monkeypatch, tmp_path)
    case_path = tmp_path / "case.json"
    case_path.write_text(json.dumps(_case_payload()), encoding="utf-8")

    with pytest.raises(ValueError, match="evaluator hash mismatch"):
        audit.main(
            [
                "--checkpoint",
                str(checkpoint),
                "--data-path",
                str(tmp_path / "data"),
                "--case-json",
                str(case_path),
                "--seed",
                "7",
                "--evaluator-sha256",
                "wrong",
                "--output",
                str(tmp_path / "audit.json"),
            ]
        )
