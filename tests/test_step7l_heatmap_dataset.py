from __future__ import annotations

from pathlib import Path

import torch

from puzzleplace.ml.heatmap_baselines import evaluate_heatmap_baselines
from puzzleplace.ml.heatmap_dataset import examples_from_training_batch, read_jsonl, write_jsonl


def _training_batch() -> tuple[torch.Tensor, ...]:
    area = torch.tensor([[1.0, 1.0, -1.0]])
    b2b = torch.tensor([[[-1.0, -1.0, -1.0]]])
    p2b = torch.tensor([[[0.0, 0.0, 1.0], [1.0, 1.0, 1.0]]])
    pins = torch.tensor([[[0.0, 0.0], [4.0, 4.0]]])
    constraints = torch.tensor(
        [[[0.0, 0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, 0.0, 2.0], [-1.0] * 5]]
    )
    tree_sol = torch.zeros((1, 2, 3))
    fp_sol = torch.tensor([[[1.0, 1.0, 0.0, 0.0], [1.0, 1.0, 4.0, 4.0], [-1.0] * 4]])
    metrics = torch.zeros((1, 8))
    return area, b2b, p2b, pins, constraints, tree_sol, fp_sol, metrics


def test_examples_from_training_batch_emits_per_block_records() -> None:
    examples = examples_from_training_batch(_training_batch(), grid_size=4)
    assert len(examples) == 2
    assert examples[0]["schema"] == "step7l_phase1_heatmap_example_v1"
    assert examples[0]["label_contract"].startswith("target_cell derived only")
    assert len(examples[0]["topology_top_cells"]) == 16


def test_heatmap_jsonl_roundtrip_and_eval(tmp_path: Path) -> None:
    examples = examples_from_training_batch(_training_batch(), grid_size=4)
    path = tmp_path / "examples.jsonl"
    assert write_jsonl(path, examples) == 2
    assert len(read_jsonl(path)) == 2
    out = tmp_path / "metrics.json"
    report = evaluate_heatmap_baselines(path, out)
    assert out.exists()
    assert report["example_count"] == 2
    assert "topology" in report["baselines"]
    assert "1" in report["baselines"]["topology"]["recall_at_k"]
