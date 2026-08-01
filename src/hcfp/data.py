"""Auditable data labels, transforms, corruptions, and tar shards."""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields, replace
import hashlib
import io
import json
from pathlib import Path
import tarfile
from typing import Any, Iterable, Iterator

import torch

from hcfp.case import (
    BOUNDARY_BOTTOM,
    BOUNDARY_LEFT,
    BOUNDARY_RIGHT,
    BOUNDARY_TOP,
    FloorplanCase,
    from_official,
)
from hcfp.geometry import centers_from_xywh, log_aspect_from_xywh, normalize_xywh, xywh_from_state


REL_LEFT = 0
REL_RIGHT = 1
REL_ABOVE = 2
REL_BELOW = 3
REL_AMBIGUOUS = 4
D4_TRANSFORMS = (
    "identity",
    "hflip",
    "vflip",
    "rot90",
    "rot180",
    "rot270",
    "transpose",
    "antitranspose",
)


@dataclass(frozen=True)
class SolutionLabels:
    rectangles: torch.Tensor
    centers: torch.Tensor
    log_aspect: torch.Tensor
    pairwise_precedence: torch.Tensor
    precedence_tie_mask: torch.Tensor
    outline: torch.Tensor


@dataclass(frozen=True)
class DataSample:
    sample_id: str
    case: FloorplanCase
    labels: SolutionLabels


def extract_labels(case: FloorplanCase, solution_xywh: Any, *, normalized: bool = False) -> SolutionLabels:
    rects = torch.as_tensor(solution_xywh, dtype=torch.float32, device=case.area.device).reshape(case.n, 4)
    if not normalized:
        rects = normalize_xywh(case, rects)
    centers = centers_from_xywh(rects)
    log_aspect = log_aspect_from_xywh(rects)
    relation, tie = pairwise_precedence(rects)
    left = rects[:, 0].amin()
    bottom = rects[:, 1].amin()
    right = (rects[:, 0] + rects[:, 2]).amax()
    top = (rects[:, 1] + rects[:, 3]).amax()
    width = (right - left).clamp_min(1.0e-12)
    height = (top - bottom).clamp_min(1.0e-12)
    utilization = case.area.sum() / (width * height)
    outline = torch.stack((width, height, utilization, width / height)).float()
    return SolutionLabels(rects.float(), centers.float(), log_aspect.float(), relation, tie, outline)


def pairwise_precedence(rects: torch.Tensor, tol: float = 1.0e-7) -> tuple[torch.Tensor, torch.Tensor]:
    boxes = torch.as_tensor(rects, dtype=torch.float32)
    n = boxes.shape[0]
    left, bottom = boxes[:, 0], boxes[:, 1]
    right, top = left + boxes[:, 2], bottom + boxes[:, 3]
    gaps = torch.stack(
        (
            left[None, :] - right[:, None],
            left[:, None] - right[None, :],
            bottom[:, None] - top[None, :],
            bottom[None, :] - top[:, None],
        ),
        dim=-1,
    )
    valid = gaps >= -tol
    valid_count = valid.sum(dim=-1)
    unique = valid_count == 1
    relation = torch.where(
        unique,
        valid.to(torch.long).argmax(dim=-1),
        torch.full((n, n), REL_AMBIGUOUS, dtype=torch.long, device=boxes.device),
    )
    diagonal = torch.eye(n, dtype=torch.bool, device=boxes.device)
    relation[diagonal] = REL_AMBIGUOUS
    tie = (valid_count > 1) | diagonal
    return relation, tie


def transform_sample(sample: DataSample, name: str) -> DataSample:
    if name not in D4_TRANSFORMS:
        raise ValueError(f"unknown D4 transform {name!r}")
    bounds = _rect_bounds(sample.labels.rectangles)
    case = _transform_case(sample.case, name, bounds)
    rects = _transform_rectangles(sample.labels.rectangles, name, bounds)
    return DataSample(f"{sample.sample_id}:{name}", case, extract_labels(case, rects, normalized=True))


def inverse_transform(name: str) -> str:
    if name == "rot90":
        return "rot270"
    if name == "rot270":
        return "rot90"
    if name in D4_TRANSFORMS:
        return name
    raise ValueError(f"unknown D4 transform {name!r}")


def corrupt_rectangles(case: FloorplanCase, rectangles: Any, *, seed: int, shift: float = 0.03, aspect: float = 0.08) -> torch.Tensor:
    rects = torch.as_tensor(rectangles, dtype=torch.float32, device=case.area.device).reshape(case.n, 4)
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    center = centers_from_xywh(rects)
    log_aspect = log_aspect_from_xywh(rects)
    center_noise = (torch.rand((case.n, 2), generator=generator, dtype=torch.float32) - 0.5) * (2.0 * shift)
    aspect_noise = (torch.rand(case.n, generator=generator, dtype=torch.float32) - 0.5) * (2.0 * aspect)
    center = center + center_noise.to(device=center.device)
    log_aspect = log_aspect + aspect_noise.to(device=log_aspect.device)
    hard_shape = case.fixed_mask | case.preplaced_mask
    if hard_shape.any():
        log_aspect[hard_shape] = torch.log(case.target[hard_shape, 2] / case.target[hard_shape, 3])
    center[case.preplaced_mask] = centers_from_xywh(case.target)[case.preplaced_mask]
    out = xywh_from_state(case, center, log_aspect)
    out[case.preplaced_mask] = case.target[case.preplaced_mask]
    out[case.fixed_mask, 2:4] = case.target[case.fixed_mask, 2:4]
    return out.float()


def write_shard(
    samples: Iterable[DataSample],
    path: str | Path,
    *,
    provenance: dict[str, Any] | None = None,
) -> dict[str, Any]:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    entries = []
    with tarfile.open(path, "w") as tar:
        for index, sample in enumerate(samples):
            validate_sample(sample)
            name = f"{index:06d}.json"
            data = json.dumps(sample_to_payload(sample), sort_keys=True, separators=(",", ":")).encode()
            _add_bytes(tar, name, data)
            entries.append({"name": name, "sample_id": sample.sample_id, "sha256": hashlib.sha256(data).hexdigest()})
        manifest = {"schema_version": 1, "provenance": provenance or {}, "samples": entries}
        _add_bytes(tar, "manifest.json", json.dumps(manifest, sort_keys=True, indent=2).encode())
    manifest["tar_sha256"] = file_sha256(path)
    sidecar = Path(f"{path}.manifest.json")
    sidecar.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def read_shard(path: str | Path) -> list[DataSample]:
    return list(iter_shard(path))


def iter_shard(path: str | Path) -> Iterator[DataSample]:
    """Stream verified samples without materializing an entire shard."""

    path = Path(path)
    sidecar = Path(f"{path}.manifest.json")
    if sidecar.is_file():
        external = json.loads(sidecar.read_text(encoding="utf-8"))
        if external.get("tar_sha256") != file_sha256(path):
            raise ValueError("tar checksum does not match sidecar manifest")
    with tarfile.open(path, "r") as tar:
        manifest = json.loads(_read_member(tar, "manifest.json").decode())
        for entry in manifest["samples"]:
            data = _read_member(tar, entry["name"])
            if hashlib.sha256(data).hexdigest() != entry["sha256"]:
                raise ValueError(f"sample checksum mismatch: {entry['name']}")
            yield sample_from_payload(json.loads(data.decode()))


def read_shard_manifest(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    sidecar = Path(f"{path}.manifest.json")
    if not sidecar.is_file():
        raise ValueError(f"missing shard sidecar manifest: {sidecar}")
    manifest = json.loads(sidecar.read_text(encoding="utf-8"))
    if manifest.get("tar_sha256") != file_sha256(path):
        raise ValueError("tar checksum does not match sidecar manifest")
    return manifest


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sample_to_payload(sample: DataSample) -> dict[str, Any]:
    return {
        "sample_id": sample.sample_id,
        "case": case_to_payload(sample.case),
        "labels": labels_to_payload(sample.labels),
    }


def sample_from_payload(payload: dict[str, Any]) -> DataSample:
    sample = DataSample(
        str(payload["sample_id"]),
        case_from_payload(payload["case"]),
        labels_from_payload(payload["labels"]),
    )
    validate_sample(sample)
    return sample


def sample_from_fixture(payload: dict[str, Any]) -> DataSample:
    if "case" in payload and "labels" in payload:
        return sample_from_payload(payload)
    if "case" in payload:
        case = case_from_payload(payload["case"])
    else:
        case = from_official(
            payload["block_count"],
            payload["area_targets"],
            payload.get("b2b_connectivity", []),
            payload.get("p2b_connectivity", []),
            payload.get("pins_pos", []),
            payload.get("constraints", []),
            payload.get("target_positions"),
        )
    solution = payload.get("solution", payload.get("rectangles", payload.get("placement")))
    if solution is None:
        raise ValueError("fixture is missing solution/rectangles/placement")
    return DataSample(str(payload.get("sample_id", payload.get("test_id", "sample"))), case, extract_labels(case, solution))


def case_to_payload(case: FloorplanCase) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for field in fields(case):
        value = getattr(case, field.name)
        payload[field.name] = value.detach().cpu().tolist() if isinstance(value, torch.Tensor) else value
    return payload


def case_from_payload(payload: dict[str, Any]) -> FloorplanCase:
    values: dict[str, Any] = {"n": int(payload["n"]), "scale": float(payload["scale"])}
    bools = {"block_mask", "fixed_mask", "preplaced_mask", "target_valid_mask", "group_membership", "mib_membership", "boundary_bits"}
    longs = {"constraints", "cluster_id", "mib_id", "cluster_group_ids", "mib_group_ids"}
    for field in fields(FloorplanCase):
        if field.name in values:
            continue
        value = payload[field.name]
        if field.name in bools:
            values[field.name] = torch.as_tensor(value, dtype=torch.bool)
        elif field.name in longs:
            values[field.name] = torch.as_tensor(value, dtype=torch.long)
        else:
            values[field.name] = torch.as_tensor(value, dtype=torch.float32)
    return FloorplanCase(**values)


def labels_to_payload(labels: SolutionLabels) -> dict[str, Any]:
    return {key: value.detach().cpu().tolist() for key, value in asdict(labels).items()}


def labels_from_payload(payload: dict[str, Any]) -> SolutionLabels:
    return SolutionLabels(
        rectangles=torch.as_tensor(payload["rectangles"], dtype=torch.float32),
        centers=torch.as_tensor(payload["centers"], dtype=torch.float32),
        log_aspect=torch.as_tensor(payload["log_aspect"], dtype=torch.float32),
        pairwise_precedence=torch.as_tensor(payload["pairwise_precedence"], dtype=torch.long),
        precedence_tie_mask=torch.as_tensor(payload["precedence_tie_mask"], dtype=torch.bool),
        outline=torch.as_tensor(payload["outline"], dtype=torch.float32),
    )


def validate_sample(sample: DataSample) -> None:
    """Reject stale or malformed precomputed labels before training."""

    case, labels = sample.case, sample.labels
    if labels.rectangles.shape != (case.n, 4):
        raise ValueError("label rectangles must have shape [N,4]")
    if not bool(torch.isfinite(labels.rectangles).all()) or not bool((labels.rectangles[:, 2:4] > 0.0).all()):
        raise ValueError("label rectangles must be finite with positive dimensions")
    actual_area = labels.rectangles[:, 2:4].prod(dim=1)
    if not torch.allclose(actual_area, case.area, rtol=0.01, atol=1.0e-6):
        raise ValueError("label rectangle area does not match case area")
    if case.preplaced_mask.any() and not torch.equal(
        labels.rectangles[case.preplaced_mask], case.target[case.preplaced_mask]
    ):
        raise ValueError("preplaced label geometry must exactly match target")
    hard_shape = case.fixed_mask | case.preplaced_mask
    if hard_shape.any() and not torch.equal(labels.rectangles[hard_shape, 2:4], case.target[hard_shape, 2:4]):
        raise ValueError("fixed-shape label dimensions must exactly match target")
    rebuilt = extract_labels(case, labels.rectangles, normalized=True)
    if not torch.equal(labels.pairwise_precedence, rebuilt.pairwise_precedence):
        raise ValueError("pairwise precedence labels are stale")
    if not torch.equal(labels.precedence_tie_mask, rebuilt.precedence_tie_mask):
        raise ValueError("precedence tie labels are stale")
    if not torch.allclose(labels.outline, rebuilt.outline, rtol=1.0e-5, atol=1.0e-6):
        raise ValueError("outline labels are stale")


def _transform_case(case: FloorplanCase, name: str, bounds: tuple[float, float, float, float]) -> FloorplanCase:
    pins = _transform_points(case.pins, name, bounds) if case.pins.numel() else case.pins.clone()
    target = case.target.clone()
    if case.target_valid_mask.any():
        target[case.target_valid_mask] = _transform_rectangles(target[case.target_valid_mask], name, bounds)
    boundary_bits = _remap_boundary(case.boundary_bits, name)
    codes = torch.zeros(case.n, dtype=torch.long)
    for bit, column in zip((BOUNDARY_LEFT, BOUNDARY_RIGHT, BOUNDARY_TOP, BOUNDARY_BOTTOM), range(4)):
        codes = codes + boundary_bits[:, column].long() * bit
    constraints = case.constraints.clone()
    constraints[:, 4] = codes
    return replace(case, pins=pins.float(), target=target.float(), boundary_bits=boundary_bits, constraints=constraints)


def _transform_rectangles(rectangles: torch.Tensor, name: str, bounds: tuple[float, float, float, float]) -> torch.Tensor:
    boxes = torch.as_tensor(rectangles, dtype=torch.float32)
    p0 = boxes[:, :2]
    p1 = boxes[:, :2] + boxes[:, 2:4]
    corners = torch.stack((p0, torch.stack((p0[:, 0], p1[:, 1]), dim=1), p1, torch.stack((p1[:, 0], p0[:, 1]), dim=1)), dim=1)
    transformed = _transform_points(corners.reshape(-1, 2), name, bounds).reshape(-1, 4, 2)
    low = transformed.amin(dim=1)
    high = transformed.amax(dim=1)
    return torch.cat((low, high - low), dim=1).float()


def _transform_points(points: torch.Tensor, name: str, bounds: tuple[float, float, float, float]) -> torch.Tensor:
    left, bottom, right, top = bounds
    width, height = right - left, top - bottom
    p = torch.as_tensor(points, dtype=torch.float32)
    x, y = p[:, 0] - left, p[:, 1] - bottom
    if name == "identity":
        u, v = x, y
    elif name == "hflip":
        u, v = width - x, y
    elif name == "vflip":
        u, v = x, height - y
    elif name == "rot90":
        u, v = height - y, x
    elif name == "rot180":
        u, v = width - x, height - y
    elif name == "rot270":
        u, v = y, width - x
    elif name == "transpose":
        u, v = y, x
    elif name == "antitranspose":
        u, v = height - y, width - x
    else:
        raise ValueError(f"unknown D4 transform {name!r}")
    return torch.stack((u + left, v + bottom), dim=1)


def _remap_boundary(bits: torch.Tensor, name: str) -> torch.Tensor:
    order = {"L": 0, "R": 1, "T": 2, "B": 3}
    mapping = {
        "identity": {"L": "L", "R": "R", "T": "T", "B": "B"},
        "hflip": {"L": "R", "R": "L", "T": "T", "B": "B"},
        "vflip": {"L": "L", "R": "R", "T": "B", "B": "T"},
        "rot90": {"L": "B", "R": "T", "T": "L", "B": "R"},
        "rot180": {"L": "R", "R": "L", "T": "B", "B": "T"},
        "rot270": {"L": "T", "R": "B", "T": "R", "B": "L"},
        "transpose": {"L": "B", "R": "T", "T": "R", "B": "L"},
        "antitranspose": {"L": "T", "R": "B", "T": "L", "B": "R"},
    }[name]
    out = torch.zeros_like(bits)
    for source, target in mapping.items():
        out[:, order[target]] |= bits[:, order[source]]
    return out


def _rect_bounds(rectangles: torch.Tensor) -> tuple[float, float, float, float]:
    boxes = torch.as_tensor(rectangles, dtype=torch.float32)
    return (
        float(boxes[:, 0].amin()),
        float(boxes[:, 1].amin()),
        float((boxes[:, 0] + boxes[:, 2]).amax()),
        float((boxes[:, 1] + boxes[:, 3]).amax()),
    )


def _add_bytes(tar: tarfile.TarFile, name: str, data: bytes) -> None:
    info = tarfile.TarInfo(name)
    info.size = len(data)
    info.mtime = 0
    tar.addfile(info, io.BytesIO(data))


def _read_member(tar: tarfile.TarFile, name: str) -> bytes:
    stream = tar.extractfile(tar.getmember(name))
    if stream is None:
        raise ValueError(f"tar member is not a regular file: {name}")
    return stream.read()
