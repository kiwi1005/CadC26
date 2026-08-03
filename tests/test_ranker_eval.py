from __future__ import annotations

from dataclasses import replace
import importlib.util
import json
from pathlib import Path

import pytest
import torch

from hcfp.data import DataSample, extract_labels
from hcfp.fallback import safe_shelf
from hcfp.profile import synthetic_case
from hcfp.replay import OFFICIAL_TARGET_KIND, ReplayRecord


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/eval_hcfp_ranker.py"
SPEC = importlib.util.spec_from_file_location("eval_hcfp_ranker_test", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
ranker_eval = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ranker_eval)


class _FeatureCostModel:
    def to(self, *, device):
        self.device = device
        return self

    def eval(self):
        return self

    @staticmethod
    def encoder(_case):
        return torch.zeros(1)

    @staticmethod
    def ranker(_embedding, _population, features):
        return features[:, 0]


def _sample(sample_id: str, block_count: int = 32) -> DataSample:
    case = synthetic_case(block_count, device="cpu")
    return DataSample(sample_id, case, extract_labels(case, safe_shelf(case), normalized=True))


def _v3_record(
    sample_id: str,
    *,
    checkpoint_hash: str = "a" * 64,
    costs: tuple[float, ...] = (1.0, 0.0, 2.0, 3.0),
    target_score: tuple[float, ...] = (0.0, 0.2, 1.0, 0.1),
    target_rank: tuple[int, ...] = (0, 2, 3, 1),
    feasibility_tier: tuple[int, ...] = (0, 1, 1, 0),
) -> ReplayRecord:
    count = len(costs)
    features = torch.zeros((count, 8), dtype=torch.float32)
    features[:, 0] = torch.tensor(costs, dtype=torch.float32)
    return ReplayRecord(
        _sample(sample_id),
        checkpoint_hash,
        features,
        torch.tensor(target_score, dtype=torch.float32),
        OFFICIAL_TARGET_KIND,
        candidate_row_ids=tuple(f"{sample_id}:row-{index}" for index in range(count)),
        feasibility_tier=torch.tensor(feasibility_tier, dtype=torch.long),
        target_rank=torch.tensor(target_rank, dtype=torch.long),
    )


def _legacy_record(
    sample_id: str,
    *,
    checkpoint_hash: str = "a" * 64,
) -> ReplayRecord:
    features = torch.zeros((3, 8), dtype=torch.float32)
    features[:, 0] = torch.tensor([2.0, 0.0, 1.0])
    return ReplayRecord(
        _sample(sample_id),
        checkpoint_hash,
        features,
        torch.tensor([0.0, 0.5, 0.2], dtype=torch.float32),
        OFFICIAL_TARGET_KIND,
    )


def test_prediction_order_ties_by_stable_row_id_without_target_leakage() -> None:
    order = ranker_eval._prediction_order(
        torch.tensor([0.0, 0.0, 1.0]),
        ("c", "b", "a"),
    )

    assert order == [1, 0, 2]


def test_percentile_uses_nearest_rank_for_small_heldout_sets() -> None:
    assert ranker_eval._percentile([1.0, 2.0], 0.95) == 2.0


def test_v3_case_metrics_report_listwise_regret_and_false_promotion() -> None:
    record = _v3_record("case/a")
    metrics = ranker_eval._v3_case_metrics(
        record,
        torch.tensor([1.0, 0.0, 2.0, 3.0]),
        record.target_score,
    )

    assert metrics["selected_index"] == 1
    assert metrics["oracle_index"] == 0
    assert metrics["top1_exact_best"] is False
    assert metrics["top4_oracle_recall"] is True
    assert metrics["rank_regret"] == 2
    assert metrics["score_regret"] == pytest.approx(0.2)
    assert metrics["false_promotion"] is True


def test_cli_writes_split_and_overall_v3_metrics(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    replay = tmp_path / "heldout.jsonl"
    checkpoint = tmp_path / "ranker.pt"
    output = tmp_path / "report.json"
    replay.write_text("placeholder\n", encoding="utf-8")
    checkpoint.write_bytes(b"checkpoint")
    records = [
        replace(_v3_record("case/a"), candidate_stage="initial"),
        replace(
            _v3_record("case/a", costs=(0.0, 1.0, 2.0, 3.0)),
            candidate_stage="post_relax",
        ),
    ]
    monkeypatch.setattr(ranker_eval, "iter_replay", lambda _path: iter(records))
    monkeypatch.setattr(
        ranker_eval,
        "load_checkpoint",
        lambda *_args, **_kwargs: (
            _FeatureCostModel(),
            {"state_hash": "b" * 64, "parent_state_hash": "a" * 64},
        ),
    )

    assert ranker_eval.main(
        [
            "--replay",
            f"heldout={replay}",
            "--checkpoint",
            f"ranker={checkpoint}",
            "--output",
            str(output),
            "--device",
            "cpu",
        ]
    ) == 0

    report = json.loads(output.read_text(encoding="utf-8"))
    split = report["results"]["heldout"]["ranker"]["summary"]
    by_stage = report["results"]["heldout"]["ranker"]["by_stage"]
    overall = report["overall"]["ranker"]
    assert split["metric_mode"] == "schema_v3_listwise"
    assert split["records"] == 2
    assert split["top1_exact_best"] == 1
    assert split["top4_oracle_recall"] == 2
    assert split["false_promotion"] == 1
    assert split["promotion_gates"] == {
        "evaluable": False,
        "records": 2,
        "records_required": 16,
        "top1_12_of_16_met": None,
        "top1_exact_best": 1,
        "top1_exact_best_required": 12,
        "top4_15_of_16_met": None,
        "top4_oracle_recall": 2,
        "top4_oracle_recall_required": 15,
    }
    assert overall == split
    assert set(by_stage) == {"initial", "post_relax"}
    assert by_stage["initial"]["records"] == 1
    assert by_stage["post_relax"]["records"] == 1
    assert report["overall_by_stage"]["ranker"] == by_stage
    assert report["checkpoints"]["ranker"]["compatible_replay_hashes"] == [
        "a" * 64,
        "b" * 64,
    ]
    assert report["replays"]["heldout"]["samples"] == 1


def test_cli_rejects_replay_sample_overlap(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first = tmp_path / "first.jsonl"
    second = tmp_path / "second.jsonl"
    checkpoint = tmp_path / "ranker.pt"
    for path in (first, second, checkpoint):
        path.write_text("x", encoding="utf-8")
    monkeypatch.setattr(
        ranker_eval,
        "iter_replay",
        lambda _path: iter((_v3_record("case/a"),)),
    )

    with pytest.raises(ValueError, match="replay sample overlap"):
        ranker_eval.main(
            [
                "--replay",
                f"first={first}",
                "--replay",
                f"second={second}",
                "--checkpoint",
                f"ranker={checkpoint}",
                "--output",
                str(tmp_path / "report.json"),
                "--device",
                "cpu",
            ]
        )


def test_cli_rejects_checkpoint_replay_hash_mismatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    replay = tmp_path / "heldout.jsonl"
    checkpoint = tmp_path / "ranker.pt"
    replay.write_text("placeholder\n", encoding="utf-8")
    checkpoint.write_bytes(b"checkpoint")
    monkeypatch.setattr(
        ranker_eval,
        "iter_replay",
        lambda _path: iter((_v3_record("case/a", checkpoint_hash="c" * 64),)),
    )
    monkeypatch.setattr(
        ranker_eval,
        "load_checkpoint",
        lambda *_args, **_kwargs: (
            _FeatureCostModel(),
            {"state_hash": "b" * 64, "parent_state_hash": "a" * 64},
        ),
    )

    with pytest.raises(ValueError, match="not compatible"):
        ranker_eval.main(
            [
                "--replay",
                f"heldout={replay}",
                "--checkpoint",
                f"ranker={checkpoint}",
                "--output",
                str(tmp_path / "report.json"),
                "--device",
                "cpu",
            ]
        )


def test_cli_rejects_nonofficial_replay_target(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    replay = tmp_path / "proxy.jsonl"
    checkpoint = tmp_path / "ranker.pt"
    replay.write_text("placeholder\n", encoding="utf-8")
    checkpoint.write_bytes(b"checkpoint")
    proxy = replace(_legacy_record("case/a"), target_kind="legacy_proxy_v1")
    monkeypatch.setattr(ranker_eval, "iter_replay", lambda _path: iter((proxy,)))

    with pytest.raises(ValueError, match="official replay targets"):
        ranker_eval.main(
            [
                "--replay",
                f"proxy={replay}",
                "--checkpoint",
                f"ranker={checkpoint}",
                "--output",
                str(tmp_path / "report.json"),
                "--device",
                "cpu",
            ]
        )


def test_cli_preserves_legacy_v2_score_regret_mode(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    replay = tmp_path / "legacy.jsonl"
    checkpoint = tmp_path / "ranker.pt"
    output = tmp_path / "legacy-report.json"
    replay.write_text("placeholder\n", encoding="utf-8")
    checkpoint.write_bytes(b"checkpoint")
    monkeypatch.setattr(ranker_eval, "iter_replay", lambda _path: iter((_legacy_record("case/a"),)))
    monkeypatch.setattr(
        ranker_eval,
        "load_checkpoint",
        lambda *_args, **_kwargs: (
            _FeatureCostModel(),
            {"state_hash": "a" * 64, "parent_state_hash": None},
        ),
    )

    assert ranker_eval.main(
        [
            "--replay",
            f"legacy={replay}",
            "--checkpoint",
            f"ranker={checkpoint}",
            "--output",
            str(output),
            "--device",
            "cpu",
        ]
    ) == 0

    report = json.loads(output.read_text(encoding="utf-8"))
    summary = report["results"]["legacy"]["ranker"]["summary"]
    assert summary["metric_mode"] == "legacy_v2_score_regret"
    assert summary["top1_exact_best"] == 0
    assert summary["weighted_score_regret"] == pytest.approx(0.5)
