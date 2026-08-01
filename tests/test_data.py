from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import pytest
import torch

from hcfp.case import from_official
from hcfp.data import (
    DataSample,
    case_to_payload,
    corrupt_rectangles,
    extract_labels,
    file_sha256,
    inverse_transform,
    labels_to_payload,
    read_shard,
    sample_from_payload,
    sample_to_payload,
    transform_sample,
    write_shard,
)


def _case():
    return from_official(
        4,
        [4.0, 9.0, 16.0, 25.0],
        [[0, 1, 2.0], [1, 2, 3.0], [2, 3, 1.0]],
        [[0, 0, 1.0], [1, 3, 1.0]],
        [[0.0, 0.0], [8.0, 1.0]],
        [[0, 1, 0, 1, 1], [1, 0, 0, 1, 2], [0, 0, 7, 0, 4], [0, 0, 7, 0, 8]],
        [[0.0, 0.0, 2.0, 2.0], [4.0, 0.0, 3.0, 3.0], [-1.0, -1.0, -1.0, -1.0], [-1.0, -1.0, -1.0, -1.0]],
    )


def _solution():
    return torch.tensor(
        [
            [0.0, 0.0, 2.0, 2.0],
            [4.0, 0.0, 3.0, 3.0],
            [0.0, 4.0, 4.0, 4.0],
            [5.0, 5.0, 5.0, 5.0],
        ],
        dtype=torch.float32,
    )


def _sample() -> DataSample:
    case = _case()
    return DataSample("toy-0", case, extract_labels(case, _solution()))


def test_extract_labels_has_geometry_and_pairwise_precedence() -> None:
    labels = _sample().labels

    assert labels.rectangles.shape == (4, 4)
    assert labels.centers.shape == (4, 2)
    assert labels.log_aspect.shape == (4,)
    assert labels.pairwise_precedence.shape == (4, 4)
    assert labels.precedence_tie_mask.shape == (4, 4)
    assert labels.outline.shape == (4,)
    assert int(labels.pairwise_precedence[0, 1]) == 0
    assert int(labels.pairwise_precedence[2, 0]) == 2
    assert bool(labels.precedence_tie_mask[0, 0])


def test_d4_transform_roundtrip_preserves_area_and_remaps_boundary() -> None:
    sample = _sample()
    rotated = transform_sample(sample, "rot90")
    restored = transform_sample(rotated, inverse_transform("rot90"))

    assert torch.allclose(restored.labels.rectangles[:, 2:].prod(dim=1), sample.labels.rectangles[:, 2:].prod(dim=1))
    assert torch.equal(rotated.case.boundary_bits[0], torch.tensor([False, False, False, True]))
    assert torch.equal(rotated.case.boundary_bits[1], torch.tensor([False, False, True, False]))
    assert torch.equal(rotated.case.boundary_bits[2], torch.tensor([True, False, False, False]))
    assert torch.equal(rotated.case.boundary_bits[3], torch.tensor([False, True, False, False]))
    assert torch.allclose(restored.labels.rectangles, sample.labels.rectangles, atol=1.0e-6)
    assert torch.equal(restored.case.boundary_bits, sample.case.boundary_bits)


def test_diagonal_pair_is_ambiguous_instead_of_arbitrary_axis() -> None:
    sample = _sample()
    relation = sample.labels.pairwise_precedence
    tie = sample.labels.precedence_tie_mask

    assert int(relation[3, 0]) == 4
    assert bool(tie[3, 0])


def test_corruption_is_deterministic_and_preserves_hard_targets() -> None:
    sample = _sample()
    a = corrupt_rectangles(sample.case, sample.labels.rectangles, seed=13)
    b = corrupt_rectangles(sample.case, sample.labels.rectangles, seed=13)

    assert torch.equal(a, b)
    assert torch.equal(a[sample.case.preplaced_mask], sample.case.target[sample.case.preplaced_mask])
    assert torch.equal(a[sample.case.fixed_mask, 2:4], sample.case.target[sample.case.fixed_mask, 2:4])
    assert torch.allclose(a[:, 2:].prod(dim=1), sample.case.area, atol=1.0e-6)


def test_shard_roundtrip_and_checksum(tmp_path: Path) -> None:
    shard = tmp_path / "data.tar"
    manifest = write_shard([_sample()], shard)
    loaded = read_shard(shard)

    assert manifest["tar_sha256"] == file_sha256(shard)
    assert len(loaded) == 1
    assert loaded[0].sample_id == "toy-0"
    assert torch.allclose(loaded[0].labels.rectangles, _sample().labels.rectangles)


def test_build_shards_cli_skips_denylisted_validation_ids(tmp_path: Path) -> None:
    fixture = tmp_path / "fixtures.json"
    deny = tmp_path / "deny.txt"
    shard = tmp_path / "out.tar"
    sample = _sample()
    item = {
        "test_id": 99,
        "sample_id": "validation-99",
        "case": json.loads(json.dumps(case_to_payload(sample.case))),
        "labels": json.loads(json.dumps(labels_to_payload(sample.labels))),
    }
    fixture.write_text(json.dumps([item, {**item, "test_id": 1, "sample_id": "train-1"}]), encoding="utf-8")
    deny.write_text("99\n", encoding="utf-8")

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/build_hcfp_shards.py",
            str(fixture),
            "-o",
            str(shard),
            "--source",
            "FloorSet-train",
            "--source-version",
            "v10-fixture",
            "--split",
            "train",
            "--denylist",
            str(deny),
        ],
        cwd=Path(__file__).resolve().parents[1],
        check=True,
        text=True,
        capture_output=True,
    )

    assert '"tar_sha256"' in completed.stdout
    sidecar = json.loads(Path(f"{shard}.manifest.json").read_text(encoding="utf-8"))
    assert sidecar["tar_sha256"] == file_sha256(shard)
    assert sidecar["provenance"]["split"] == "train"
    assert sidecar["provenance"]["denylist_sha256"] == file_sha256(deny)
    assert [sample.sample_id for sample in read_shard(shard)] == ["train-1"]


def test_payload_decode_rejects_stale_hard_target_labels() -> None:
    payload = json.loads(json.dumps(sample_to_payload(_sample())))
    payload["labels"]["rectangles"][0][2] += 0.25

    with pytest.raises(ValueError, match="area does not match"):
        sample_from_payload(payload)
