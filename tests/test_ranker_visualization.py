from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest
import torch

from hcfp.candidates import candidate_features
from hcfp.data import DataSample, extract_labels
from hcfp.fallback import safe_shelf
from hcfp.profile import synthetic_case
from hcfp.replay import (
    CAP_LOG,
    OFFICIAL_TARGET_KIND,
    ReplayRecord,
    _candidate_geometry_hashes,
    _candidate_row_ids,
    _target_rank,
    write_replay_v3,
)


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/visualize_hcfp_ranker.py"
SPEC = importlib.util.spec_from_file_location("visualize_hcfp_ranker_test", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
ranker_visualize = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ranker_visualize)


def test_ranker_visualization_cli_writes_index_manifest_and_html(tmp_path: Path) -> None:
    replay, evaluation, record = _fixture_files(tmp_path)
    output = tmp_path / "visuals"

    assert ranker_visualize.main(
        [
            "--replay",
            str(replay),
            "--evaluation",
            str(evaluation),
            "--output-dir",
            str(output),
            "--split",
            "dev",
            "--checkpoint",
            "ranker",
            "--stage",
            "initial",
        ]
    ) == 0

    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    index = (output / "index.html").read_text(encoding="utf-8")
    page = (output / manifest["cases"][0]["file"]).read_text(encoding="utf-8")

    assert manifest["schema_version"] == 1
    assert manifest["inputs"]["split"] == "dev"
    assert manifest["inputs"]["checkpoint"] == "ranker"
    assert manifest["cases"][0]["sample_id"] == "sample/one"
    assert manifest["cases"][0]["selected_index"] == 1
    assert manifest["cases"][0]["oracle_index"] == 0
    assert manifest["cases"][0]["selected_row_id"] == record.candidate_row_ids[1]
    assert "sample/one" in index
    assert page.count("<svg ") == 6
    assert "ranker selected candidate 1 raw" in page
    assert "exact oracle candidate 0 post-repair" in page


def test_ranker_visualization_rejects_mismatched_row_id(tmp_path: Path) -> None:
    replay, evaluation, _record = _fixture_files(tmp_path)
    payload = json.loads(evaluation.read_text(encoding="utf-8"))
    payload["results"]["dev"]["ranker"]["cases"][0]["selected_row_id"] = "not-the-replay-row"
    evaluation.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="selected_row_id"):
        ranker_visualize.main(
            [
                "--replay",
                str(replay),
                "--evaluation",
                str(evaluation),
                "--output-dir",
                str(tmp_path / "visuals"),
            ]
        )


def test_ranker_visualization_requires_eval_row_identity(tmp_path: Path) -> None:
    replay, evaluation, _record = _fixture_files(tmp_path)
    payload = json.loads(evaluation.read_text(encoding="utf-8"))
    del payload["results"]["dev"]["ranker"]["cases"][0]["oracle_row_id"]
    evaluation.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="oracle_row_id"):
        ranker_visualize.main(
            [
                "--replay",
                str(replay),
                "--evaluation",
                str(evaluation),
                "--output-dir",
                str(tmp_path / "visuals"),
            ]
        )


def test_ranker_visualization_refuses_to_overwrite_without_force(tmp_path: Path) -> None:
    replay, evaluation, _record = _fixture_files(tmp_path)
    output = tmp_path / "visuals"
    args = [
        "--replay",
        str(replay),
        "--evaluation",
        str(evaluation),
        "--output-dir",
        str(output),
    ]

    ranker_visualize.main(args)

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        ranker_visualize.main(args)


def test_ranker_visualization_script_invocation(tmp_path: Path) -> None:
    replay, evaluation, _record = _fixture_files(tmp_path)
    output = tmp_path / "visuals"

    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--replay",
            str(replay),
            "--evaluation",
            str(evaluation),
            "--output-dir",
            str(output),
            "--limit",
            "1",
        ],
        check=True,
        cwd=ROOT,
        capture_output=True,
        text=True,
    )

    assert str(output / "index.html") in result.stdout
    assert (output / "manifest.json").is_file()


def _fixture_files(tmp_path: Path) -> tuple[Path, Path, ReplayRecord]:
    record = _record()
    replay = tmp_path / "replay.jsonl"
    evaluation = tmp_path / "evaluation.json"
    write_replay_v3([record], replay)
    evaluation.write_text(json.dumps(_evaluation(record), indent=2), encoding="utf-8")
    return replay, evaluation, record


def _record() -> ReplayRecord:
    case = synthetic_case(32, device="cpu")
    base = safe_shelf(case)
    sample = DataSample("sample/one", case, extract_labels(case, base, normalized=True))
    candidates = torch.stack(
        (
            base,
            _shift(base, block=0, dx=0.03, dy=0.0),
            _shift(base, block=1, dx=0.0, dy=0.04),
        )
    ).float()
    post_bdp = candidates.clone()
    post_bdp[1, :, 0] += 0.01
    post_repair = post_bdp.clone()
    post_repair[1, :, 1] += 0.02
    target = torch.tensor([0.0, 0.4, 0.2], dtype=torch.float32)
    tiers = torch.tensor([0, 1, 0], dtype=torch.long)
    hard = tiers == 0
    hashes = _candidate_geometry_hashes(candidates)
    kinds = ("learned", "topology", "constraint")
    source_types = kinds
    row_ids = _candidate_row_ids(
        sample_id=sample.sample_id,
        stage="initial",
        kinds=kinds,
        source_types=source_types,
        geometry_hashes=hashes,
    )
    teacher_delta = _centers(post_repair) - _centers(candidates)
    repair_displacement = torch.linalg.vector_norm(teacher_delta, dim=-1).sum(dim=1)
    return ReplayRecord(
        sample,
        "d" * 64,
        candidate_features(case, candidates, safe_shelf(case)).detach().cpu(),
        target,
        OFFICIAL_TARGET_KIND,
        candidate_row_ids=row_ids,
        candidate_source_indices=torch.arange(3, dtype=torch.long),
        candidate_kinds=kinds,
        candidate_source_types=source_types,
        candidate_geometry_sha256=hashes,
        feasibility_tier=tiers,
        target_rank=_target_rank(target, tiers, row_ids),
        candidate_stage="initial",
        candidate_population=3,
        population_seed=7,
        candidate_geometry=candidates,
        post_bdp_geometry=post_bdp,
        post_repair_geometry=post_repair,
        teacher_delta_xy=teacher_delta,
        repair_displacement=repair_displacement,
        post_repair_hard_feasible=hard,
        post_repair_log_uncapped_cost=target,
        post_repair_cap_margin=CAP_LOG - target,
        boundary_violations=torch.tensor([0, 1, 0], dtype=torch.long),
        grouping_violations=torch.tensor([0, 0, 1], dtype=torch.long),
        mib_violations=torch.tensor([0, 0, 0], dtype=torch.long),
    )


def _evaluation(record: ReplayRecord) -> dict[str, object]:
    return {
        "schema_version": 2,
        "target_kind": OFFICIAL_TARGET_KIND,
        "results": {
            "dev": {
                "ranker": {
                    "cases": [
                        {
                            "sample_id": record.sample.sample_id,
                            "candidate_stage": record.candidate_stage,
                            "selected_index": 1,
                            "oracle_index": 0,
                            "top4_indices": [1, 0, 2],
                            "selected_row_id": record.candidate_row_ids[1],
                            "oracle_row_id": record.candidate_row_ids[0],
                            "rank_regret": 2,
                            "score_regret": 0.4,
                            "top1_exact_best": False,
                            "top4_oracle_recall": True,
                            "prediction_selected_cost": -1.0,
                            "prediction_oracle_cost": 0.0,
                        }
                    ],
                    "summary": {"records": 1},
                }
            }
        },
    }


def _shift(boxes: torch.Tensor, *, block: int, dx: float, dy: float) -> torch.Tensor:
    out = boxes.clone()
    out[block, 0] += dx
    out[block, 1] += dy
    return out


def _centers(boxes: torch.Tensor) -> torch.Tensor:
    return boxes[:, :, :2] + 0.5 * boxes[:, :, 2:4]
