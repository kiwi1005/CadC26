"""Direct, no-copy streaming adapter for official FloorSet-Lite training files."""

from __future__ import annotations

import math
from pathlib import Path
import random
from typing import Iterator

import torch

from hcfp.case import from_official
from hcfp.data import DataSample, extract_labels


Tensor = torch.Tensor


def fp_sol_to_xywh(fp_sol: Tensor, block_count: int) -> Tensor:
    """Convert FloorSet ``fp_sol`` rectangles or polygons to ``(x,y,w,h)``."""

    solution = torch.as_tensor(fp_sol, dtype=torch.float32)[:block_count]
    if solution.ndim == 2 and solution.shape[1] == 4:
        width, height, x, y = solution.unbind(dim=1)
        rectangles = torch.stack((x, y, width, height), dim=1)
    elif solution.ndim == 3 and solution.shape[2] == 2:
        rows = []
        for polygon in solution:
            valid = polygon[polygon[:, 0] != -1]
            if not valid.numel():
                raise ValueError("fp_sol polygon is missing vertices")
            low = valid.amin(dim=0)
            high = valid.amax(dim=0)
            rows.append(torch.cat((low, high - low)))
        rectangles = torch.stack(rows)
    else:
        raise ValueError("fp_sol must have shape [N,4] or [N,V,2]")
    if not bool(torch.isfinite(rectangles).all()) or not bool((rectangles[:, 2:4] > 0.0).all()):
        raise ValueError("fp_sol contains invalid rectangles")
    return rectangles.float()


def target_positions_from_solution(constraints: Tensor, rectangles: Tensor) -> Tensor:
    """Rebuild the official fixed/preplaced target tensor from ground truth."""

    rules = torch.as_tensor(constraints, dtype=torch.long)
    targets = torch.full_like(rectangles, -1.0)
    fixed = rules[:, 0] != 0
    preplaced = rules[:, 1] != 0
    targets[fixed, 2:4] = rectangles[fixed, 2:4]
    targets[preplaced] = rectangles[preplaced]
    return targets


def sample_from_lite_tensors(
    sample_id: str,
    area_constraints: Tensor,
    b2b_connectivity: Tensor,
    p2b_connectivity: Tensor,
    pins_pos: Tensor,
    fp_sol: Tensor,
) -> DataSample:
    area_constraints = torch.as_tensor(area_constraints)
    if area_constraints.ndim != 2 or area_constraints.shape[1] < 6:
        raise ValueError("area/constraint tensor must have columns [area,fixed,preplaced,mib,cluster,boundary]")
    valid = area_constraints[:, 0] != -1
    block_count = int(valid.sum().item())
    area = area_constraints[:block_count, 0].float()
    constraints = area_constraints[:block_count, 1:6].long()
    rectangles = fp_sol_to_xywh(fp_sol, block_count)
    targets = target_positions_from_solution(constraints, rectangles)
    case = from_official(
        block_count,
        area,
        b2b_connectivity,
        p2b_connectivity,
        pins_pos,
        constraints,
        targets,
    )
    return DataSample(sample_id, case, extract_labels(case, rectangles))


def score_aware_acceptance(block_count: int) -> float:
    """Capped rejection probability for the contest's high-value large cases."""

    score_weight = min(math.exp((int(block_count) - 80) / 12.0), 8.0)
    return 0.30 + 0.70 * score_weight / 8.0


def iter_floorset_lite(
    root: str | Path,
    *,
    limit: int | None = None,
    seed: int | None = None,
    score_aware: bool = False,
) -> Iterator[DataSample]:
    """Yield training samples one source file at a time without copying 1M cases."""

    root = Path(root).resolve()
    if any(token in str(root).lower() for token in ("litetensordatatest", "validation", "visible")):
        raise ValueError("visible validation/test paths are forbidden for training")
    layout_root = root if root.name == "floorset_lite" else root / "floorset_lite"
    files = sorted(layout_root.glob("worker_*/layouts*"))
    if not files:
        raise FileNotFoundError(f"no FloorSet-Lite training layouts under {layout_root}")
    generator = random.Random(seed)
    if seed is not None:
        generator.shuffle(files)
    yielded = 0
    for path in files:
        payload = torch.load(path, map_location="cpu", weights_only=True)
        layout_indices = list(range(len(payload[0])))
        if seed is not None:
            generator.shuffle(layout_indices)
        for layout_index in layout_indices:
            block_count = int((payload[0][layout_index][:, 0] != -1).sum().item())
            if score_aware and generator.random() > score_aware_acceptance(block_count):
                continue
            yield sample_from_lite_tensors(
                f"{path.parent.name}/{path.name}:{layout_index}",
                payload[0][layout_index],
                payload[1][layout_index],
                payload[2][layout_index],
                payload[3][layout_index],
                payload[5][layout_index],
            )
            yielded += 1
            if limit is not None and yielded >= limit:
                return
