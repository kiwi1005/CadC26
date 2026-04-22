from __future__ import annotations

import sys
from types import SimpleNamespace

import torch

from puzzleplace.data import ConstraintColumns, adapt_training_batch, adapt_validation_batch
from puzzleplace.train.dataset_bc import load_training_cases


def test_adapt_training_batch_trims_padding_and_preserves_labels() -> None:
    batch = (
        torch.tensor([[10.0, 20.0, -1.0]]),
        torch.tensor([[[0.0, 1.0, 2.0], [-1.0, -1.0, -1.0]]]),
        torch.tensor([[[0.0, 0.0, 1.0], [1.0, 1.0, 2.0]]]),
        torch.tensor([[[0.0, 0.0], [1.0, 1.0], [-1.0, -1.0]]]),
        torch.tensor([[[1.0, 0.0, 0.0, 0.0, 1.0], [0.0, 1.0, 0.0, 1.0, 2.0], [-1.0, -1.0, -1.0, -1.0, -1.0]]]),
        torch.zeros((1, 2, 3)),
        torch.tensor([[[2.0, 3.0, 0.0, 0.0], [4.0, 5.0, 1.0, 1.0], [-1.0, -1.0, -1.0, -1.0]]]),
        torch.tensor([[100.0] * 8]),
    )
    case = adapt_training_batch(batch, case_id="train-1")
    assert case.case_id == "train-1"
    assert case.block_count == 2
    assert case.area_targets.tolist() == [10.0, 20.0]
    assert case.target_positions is not None
    assert case.target_positions.shape == (2, 4)
    assert torch.allclose(case.target_positions[0], torch.tensor([0.0, 0.0, 2.0, 3.0]))
    assert torch.allclose(case.target_positions[1], torch.tensor([1.0, 1.0, 4.0, 5.0]))
    assert int(case.constraints[0, ConstraintColumns.FIXED].item()) == 1
    assert case.b2b_edges.shape == (1, 3)


def test_adapt_validation_batch_converts_polygons_to_boxes() -> None:
    inputs = (
        torch.tensor([[10.0, 20.0]]),
        torch.tensor([[[0.0, 1.0, 1.0]]]),
        torch.tensor([[[0.0, 0.0, 1.0]]]),
        torch.tensor([[[0.0, 0.0], [1.0, 1.0]]]),
        torch.tensor([[[0.0, 0.0, 0.0, 0.0, 1.0], [0.0, 1.0, 0.0, 0.0, 2.0]]]),
    )
    polygons = torch.tensor([
        [
            [[0.0, 0.0], [2.0, 0.0], [2.0, 3.0], [0.0, 3.0]],
            [[5.0, 5.0], [8.0, 5.0], [8.0, 7.0], [5.0, 7.0]],
        ]
    ])
    metrics = torch.tensor([[50.0] * 8])
    case = adapt_validation_batch((inputs, (polygons, metrics)), case_id="val-1")
    assert case.block_count == 2
    assert case.target_positions is not None
    assert torch.allclose(case.target_positions[0], torch.tensor([0.0, 0.0, 2.0, 3.0]))
    assert torch.allclose(case.target_positions[1], torch.tensor([5.0, 5.0, 3.0, 2.0]))


def test_load_training_cases_splits_multi_sample_batches(monkeypatch) -> None:
    calls: list[dict[str, int | bool | str | None]] = []

    def _fake_loader(*, data_path: str, batch_size: int, num_samples: int | None, shuffle: bool):
        calls.append(
            {
                "data_path": data_path,
                "batch_size": batch_size,
                "num_samples": num_samples,
                "shuffle": shuffle,
            }
        )
        batch = (
            torch.tensor([[10.0, 20.0, -1.0], [30.0, 40.0, -1.0]]),
            torch.tensor(
                [
                    [[0.0, 1.0, 2.0], [-1.0, -1.0, -1.0]],
                    [[0.0, 1.0, 2.0], [-1.0, -1.0, -1.0]],
                ]
            ),
            torch.tensor(
                [
                    [[0.0, 0.0, 1.0], [1.0, 1.0, 2.0]],
                    [[0.0, 0.0, 1.0], [1.0, 1.0, 2.0]],
                ]
            ),
            torch.tensor(
                [
                    [[0.0, 0.0], [1.0, 1.0], [-1.0, -1.0]],
                    [[2.0, 2.0], [3.0, 3.0], [-1.0, -1.0]],
                ]
            ),
            torch.tensor(
                [
                    [[1.0, 0.0, 0.0, 0.0, 1.0], [0.0, 1.0, 0.0, 1.0, 2.0], [-1.0, -1.0, -1.0, -1.0, -1.0]],
                    [[0.0, 1.0, 0.0, 1.0, 2.0], [1.0, 0.0, 0.0, 0.0, 1.0], [-1.0, -1.0, -1.0, -1.0, -1.0]],
                ]
            ),
            torch.zeros((2, 2, 3)),
            torch.tensor(
                [
                    [[2.0, 3.0, 0.0, 0.0], [4.0, 5.0, 1.0, 1.0], [-1.0, -1.0, -1.0, -1.0]],
                    [[6.0, 7.0, 0.0, 0.0], [8.0, 9.0, 1.0, 1.0], [-1.0, -1.0, -1.0, -1.0]],
                ]
            ),
            torch.tensor([[100.0] * 8, [200.0] * 8]),
        )
        return [batch]

    monkeypatch.setitem(
        sys.modules,
        "iccad2026_evaluate",
        SimpleNamespace(get_training_dataloader=_fake_loader),
    )
    monkeypatch.setattr("puzzleplace.train.dataset_bc._ensure_import_paths", lambda: None)
    monkeypatch.setattr("puzzleplace.train.dataset_bc._auto_approve_downloads", lambda: None)

    cases = load_training_cases(case_limit=2, batch_size=2)

    assert calls == [
        {
            "data_path": calls[0]["data_path"],
            "batch_size": 2,
            "num_samples": 2,
            "shuffle": False,
        }
    ]
    assert [case.case_id for case in cases] == ["train-0", "train-1"]
    assert [case.block_count for case in cases] == [2, 2]
