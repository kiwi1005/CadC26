from __future__ import annotations

from pathlib import Path

import pytest
import torch

from hcfp.case import from_official
from hcfp.floorset_lite import (
    fp_sol_to_xywh,
    iter_floorset_lite,
    iter_floorset_lite_with_source,
    sample_from_lite_tensors,
    sample_with_source_from_lite_tensors,
    score_aware_acceptance,
    target_positions_from_solution,
)


def _layout_payload() -> list[torch.Tensor]:
    area_constraints = torch.tensor(
        [
            [4.0, 0, 1, 0, 0, 1],
            [9.0, 1, 0, 0, 0, 0],
            [16.0, 0, 0, 0, 0, 0],
        ]
    ).unsqueeze(0)
    b2b = torch.tensor([[[0.0, 1.0, 2.0], [1.0, 2.0, 1.0]]])
    p2b = torch.tensor([[[-1.0, -1.0, -1.0]]])
    pins = torch.tensor([[[-1.0, -1.0]]])
    tree = torch.tensor([[[0, 1, 0], [0, 2, 1]]])
    fp_sol = torch.tensor([[[2.0, 2.0, 0.0, 0.0], [3.0, 3.0, 3.0, 0.0], [4.0, 4.0, 0.0, 4.0]]])
    metrics = torch.tensor([[120.0, 0.0, 0.0, 0.0, 0.0, 0.0, 7.0, 3.0]])
    return [area_constraints, b2b, p2b, pins, tree, fp_sol, metrics]


def test_lite_rectangles_and_targets_match_official_semantics() -> None:
    payload = _layout_payload()
    rectangles = fp_sol_to_xywh(payload[5][0], 3)
    constraints = payload[0][0, :, 1:6]
    targets = target_positions_from_solution(constraints, rectangles)
    sample = sample_from_lite_tensors(
        "worker/layout:0",
        payload[0][0],
        payload[1][0],
        payload[2][0],
        payload[3][0],
        payload[5][0],
        payload[6][0],
    )

    assert torch.equal(rectangles[0], torch.tensor([0.0, 0.0, 2.0, 2.0]))
    assert torch.equal(targets[0], rectangles[0])
    assert torch.equal(targets[1], torch.tensor([-1.0, -1.0, 3.0, 3.0]))
    assert torch.equal(targets[2], torch.full((4,), -1.0))
    assert sample.case.preplaced_mask.tolist() == [True, False, False]
    assert sample.case.fixed_mask.tolist() == [False, True, False]
    assert sample.labels.baseline_area.item() == pytest.approx(120.0)
    assert sample.labels.baseline_hpwl.item() == pytest.approx(10.0)


def test_lite_runtime_source_preserves_exact_official_targets() -> None:
    payload = _layout_payload()
    sample, source = sample_with_source_from_lite_tensors(
        "worker/layout:0",
        payload[0][0],
        payload[1][0],
        payload[2][0],
        payload[3][0],
        payload[5][0],
        payload[6][0],
    )
    rebuilt = from_official(
        source["block_count"],
        source["area_targets"],
        source["b2b_connectivity"],
        source["p2b_connectivity"],
        source["pins_pos"],
        source["constraints"],
        source["target_positions"],
    )

    assert torch.equal(source["target_positions"][0], torch.tensor([0.0, 0.0, 2.0, 2.0]))
    assert torch.equal(source["target_positions"][2], torch.full((4,), -1.0))
    assert torch.equal(source["preplaced_mask"], sample.case.preplaced_mask)
    assert torch.equal(rebuilt.area, sample.case.area)
    assert torch.equal(rebuilt.target, sample.case.target)
    assert torch.equal(rebuilt.target_valid_mask, sample.case.target_valid_mask)


def test_direct_training_stream_loads_layouts_without_creating_shards(tmp_path: Path) -> None:
    layout = tmp_path / "floorset_lite/worker_0/layouts_0.pth"
    layout.parent.mkdir(parents=True)
    torch.save(_layout_payload(), layout)

    samples = list(iter_floorset_lite(tmp_path, limit=1))

    assert [sample.sample_id for sample in samples] == ["worker_0/layouts_0.pth:0"]
    assert samples[0].labels.baseline_area.item() == pytest.approx(120.0)
    assert samples[0].labels.baseline_hpwl.item() == pytest.approx(10.0)
    assert torch.equal(samples[0].tree_edges, payload := _layout_payload()[4][0].long())
    assert not list(tmp_path.rglob("*.tar"))

    sourced = list(iter_floorset_lite_with_source(tmp_path, limit=1))
    assert sourced[0][0].sample_id == samples[0].sample_id
    assert torch.equal(sourced[0][1]["tree_sol_edges"], payload)
    assert torch.equal(sourced[0][1]["target_positions"][0], torch.tensor([0.0, 0.0, 2.0, 2.0]))


def test_training_stream_can_cap_layouts_per_source_file(tmp_path: Path) -> None:
    first = tmp_path / "floorset_lite/worker_0/layouts_0.pth"
    second = tmp_path / "floorset_lite/worker_1/layouts_0.pth"
    first.parent.mkdir(parents=True)
    second.parent.mkdir(parents=True)
    doubled = [tensor.repeat((2,) + (1,) * (tensor.ndim - 1)) for tensor in _layout_payload()]
    torch.save(doubled, first)
    torch.save(_layout_payload(), second)

    samples = list(
        iter_floorset_lite(
            tmp_path,
            limit=2,
            max_layouts_per_file=1,
        )
    )

    assert [sample.sample_id.split("/")[0] for sample in samples] == ["worker_0", "worker_1"]


def test_training_stream_filters_block_count_before_sample_construction(tmp_path: Path) -> None:
    layout = tmp_path / "floorset_lite/worker_0/layouts_0.pth"
    layout.parent.mkdir(parents=True)
    mixed = [tensor.repeat((2,) + (1,) * (tensor.ndim - 1)) for tensor in _layout_payload()]
    mixed[0][1, 2, 0] = -1.0
    torch.save(mixed, layout)

    samples = list(iter_floorset_lite(tmp_path, min_blocks=3, max_blocks=3))

    assert [sample.case.n for sample in samples] == [3]


def test_file_cap_counts_score_aware_rejections(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    layout = tmp_path / "floorset_lite/worker_0/layouts_0.pth"
    layout.parent.mkdir(parents=True)
    mixed = [tensor.repeat((2,) + (1,) * (tensor.ndim - 1)) for tensor in _layout_payload()]
    mixed[0][1, 2, 0] = -1.0
    torch.save(mixed, layout)
    monkeypatch.setattr(
        "hcfp.floorset_lite.score_aware_acceptance",
        lambda block_count: 0.0 if block_count == 3 else 1.0,
    )

    samples = list(
        iter_floorset_lite(
            tmp_path,
            limit=1,
            score_aware=True,
            max_layouts_per_file=1,
        )
    )

    assert samples == []


@pytest.mark.parametrize(("index", "value"), [(0, -1.0), (6, float("nan"))])
def test_lite_metrics_reject_invalid_baselines(index: int, value: float) -> None:
    payload = _layout_payload()
    metrics = payload[6][0].clone()
    metrics[index] = value

    with pytest.raises(ValueError, match="finite and non-negative"):
        sample_from_lite_tensors(
            "worker/layout:0",
            payload[0][0],
            payload[1][0],
            payload[2][0],
            payload[3][0],
            payload[5][0],
            metrics,
        )


def test_visible_validation_path_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="forbidden"):
        list(iter_floorset_lite(tmp_path / "LiteTensorDataTest"))


def test_score_aware_sampling_prioritizes_large_cases() -> None:
    assert 0.30 < score_aware_acceptance(32) < score_aware_acceptance(80)
    assert score_aware_acceptance(80) < score_aware_acceptance(120)
    assert score_aware_acceptance(120) == pytest.approx(1.0)
