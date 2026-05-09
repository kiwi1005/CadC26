"""Step7L request-only learning-guided target proposals.

This module converts deterministic Step7L topology / terminal heatmaps into a
small sidecar request deck for later repacker work. It deliberately consumes
visible validation *inputs only* and emits unrouted target-window requests; it
never reads validation polygons, metrics, replay labels, or contest runtime
state.
"""

from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import torch

from puzzleplace.ml.floorset_training_corpus import (
    _auto_yes_download,
    _import_official_evaluator,
    resolve_floorset_root,
    write_json,
)
from puzzleplace.ml.heatmap_dataset import write_jsonl
from puzzleplace.ml.topology_maps import (
    GridSpec,
    build_block_heatmaps,
    terminal_weighted_centroid,
    top_cells,
)

CandidateSourceFamily = Literal["original_anchor", "topology", "terminal", "union_diversified"]
DEFAULT_VALIDATION_CASES = (19, 24, 25, 51, 76, 79, 91, 99)


@dataclass(frozen=True, slots=True)
class LearningGuidedTargetRequest:
    """One request-only target-window proposal for a future repacker."""

    schema: str
    case_id: str
    requested_case_id: str
    loader_index: int
    request_id: str
    source_family: CandidateSourceFamily
    block_id: int | None
    target_window: dict[str, Any] | None
    move_family: str
    route_class: str
    heatmap_score: float | None
    has_terminal_demand: bool
    boundary_code: int
    is_anchor: bool
    global_report_only: bool
    provenance: dict[str, Any]
    reason: str

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


def valid_block_ids(area_targets: torch.Tensor) -> list[int]:
    """Return non-padding block ids from visible area targets."""

    area = _squeeze_one(area_targets).reshape(-1)
    return [int(idx) for idx in torch.nonzero(area >= 0, as_tuple=False).flatten().tolist()]


def frame_from_visible_inputs(
    area_targets: torch.Tensor,
    pins_pos: torch.Tensor,
    *,
    grid_size: int = 16,
    pad_fraction: float = 0.12,
) -> GridSpec:
    """Build an inference frame from visible pins plus area scale.

    Validation labels are unavailable by contract, so Step7L uses the pin bbox as
    the observable coordinate frame and pads it by an area-derived scale. If an
    instance has no visible pins, the frame falls back to a square with side
    ``sqrt(total_visible_area)``.
    """

    area = _squeeze_one(area_targets).reshape(-1).to(torch.float32)
    visible_area = area[area >= 0]
    total_area = float(visible_area.sum().item()) if visible_area.numel() else 1.0
    area_side = max(math.sqrt(max(total_area, 1.0)), 1.0)

    pins = _squeeze_one(pins_pos).to(torch.float32)
    if pins.ndim != 2 or pins.shape[-1] < 2:
        return GridSpec(rows=grid_size, cols=grid_size, x=0.0, y=0.0, w=area_side, h=area_side)
    valid = pins[(pins[:, 0] >= 0) & (pins[:, 1] >= 0)]
    if valid.numel() == 0:
        return GridSpec(rows=grid_size, cols=grid_size, x=0.0, y=0.0, w=area_side, h=area_side)

    x0 = float(valid[:, 0].min().item())
    y0 = float(valid[:, 1].min().item())
    x1 = float(valid[:, 0].max().item())
    y1 = float(valid[:, 1].max().item())
    width = max(x1 - x0, 1.0)
    height = max(y1 - y0, 1.0)
    margin = max(width, height, area_side) * pad_fraction + area_side * 0.10
    return GridSpec(
        rows=grid_size,
        cols=grid_size,
        x=x0 - margin,
        y=y0 - margin,
        w=max(width + 2.0 * margin, 1.0),
        h=max(height + 2.0 * margin, 1.0),
    )


def requests_from_validation_batch(
    batch: tuple[Any, Any],
    *,
    loader_index: int,
    requested_case_id: str | int,
    grid_size: int = 16,
    top_blocks_per_family: int = 4,
    windows_per_block: int = 3,
    include_anchor: bool = True,
) -> list[LearningGuidedTargetRequest]:
    """Generate request-only Step7L target proposals from one validation batch.

    ``batch[1]`` may contain official validation polygons/metrics, but this
    function intentionally ignores it. The generated records preserve that
    leakage boundary in their provenance.
    """

    inputs, _discarded_labels = batch
    area, _b2b, p2b, pins, constraints = inputs
    area0 = _squeeze_one(area).to(torch.float32)
    p2b0 = _squeeze_one(p2b).to(torch.float32)
    pins0 = _squeeze_one(pins).to(torch.float32)
    constraints0 = _squeeze_one(constraints).to(torch.float32)
    case_id = str(requested_case_id)
    grid = frame_from_visible_inputs(area0, pins0, grid_size=grid_size)
    block_ids = valid_block_ids(area0)

    out: list[LearningGuidedTargetRequest] = []
    if include_anchor:
        out.append(_anchor_request(case_id, loader_index, block_ids, grid))

    heatmap_families: tuple[Literal["topology", "terminal"], ...] = ("topology", "terminal")
    ranked_by_family: dict[str, list[dict[str, Any]]] = {}
    for family in heatmap_families:
        ranked_by_family[family] = _rank_family_candidates(
            family=family,
            block_ids=block_ids,
            p2b_edges=p2b0,
            pins_pos=pins0,
            constraints=constraints0,
            grid=grid,
            top_blocks_per_family=top_blocks_per_family,
            windows_per_block=windows_per_block,
        )

    for family in heatmap_families:
        for rank, candidate in enumerate(ranked_by_family[family]):
            out.append(
                _request_from_candidate(
                    candidate,
                    source_family=family,
                    case_id=case_id,
                    loader_index=loader_index,
                    rank=rank,
                    grid=grid,
                )
            )

    union_limit = max(top_blocks_per_family * windows_per_block, windows_per_block)
    for rank, candidate in enumerate(_union_diversified(ranked_by_family, limit=union_limit)):
        out.append(
            _request_from_candidate(
                candidate,
                source_family="union_diversified",
                case_id=case_id,
                loader_index=loader_index,
                rank=rank,
                grid=grid,
            )
        )
    return out


def generate_learning_guided_requests(
    base_dir: Path,
    out_path: Path,
    *,
    floorset_root: Path | None = None,
    validation_case_ids: Iterable[int] = DEFAULT_VALIDATION_CASES,
    grid_size: int = 16,
    top_blocks_per_family: int = 4,
    windows_per_block: int = 3,
    auto_download: bool = False,
) -> dict[str, Any]:
    """Load visible validation inputs and write a Step7L Phase 2 request deck."""

    requested = sorted({int(value) for value in validation_case_ids})
    if not requested:
        raise ValueError("at least one validation case id is required")
    resolved = resolve_floorset_root(base_dir, floorset_root)
    if resolved is None:
        raise FileNotFoundError("Could not resolve external/FloorSet official checkout")
    evaluator = _import_official_evaluator(resolved)

    requests: list[dict[str, Any]] = []
    requested_set = set(requested)
    with _auto_yes_download(auto_download):
        loader = evaluator.get_validation_dataloader(data_path=str(resolved), batch_size=1)
        for loader_index, batch in enumerate(loader):
            if loader_index > requested[-1]:
                break
            if loader_index not in requested_set:
                continue
            rows = requests_from_validation_batch(
                batch,
                loader_index=loader_index,
                requested_case_id=loader_index,
                grid_size=grid_size,
                top_blocks_per_family=top_blocks_per_family,
                windows_per_block=windows_per_block,
            )
            requests.extend(row.to_json() for row in rows)

    row_count = write_jsonl(out_path, requests)
    report = summarize_requests(
        requests,
        request_path=out_path,
        requested_case_ids=requested,
        grid_size=grid_size,
        top_blocks_per_family=top_blocks_per_family,
        windows_per_block=windows_per_block,
    )
    report["request_count"] = row_count
    return report


def summarize_requests(
    rows: list[dict[str, Any]],
    *,
    request_path: Path | None = None,
    requested_case_ids: Iterable[int] | None = None,
    grid_size: int | None = None,
    top_blocks_per_family: int | None = None,
    windows_per_block: int | None = None,
) -> dict[str, Any]:
    families = Counter(str(row["source_family"]) for row in rows)
    per_case: dict[str, Counter[str]] = defaultdict(Counter)
    signatures: Counter[tuple[Any, ...]] = Counter()
    duplicate_rows = 0
    for row in rows:
        case_id = str(row["case_id"])
        family = str(row["source_family"])
        per_case[case_id][family] += 1
        signature = _request_signature(row)
        signatures[signature] += 1
        if signatures[signature] > 1:
            duplicate_rows += 1
    case_ids = sorted(per_case, key=_natural_case_sort_key)
    selected_case_count = len(case_ids)
    has_forbidden_label_terms = any(
        _payload_mentions_forbidden_validation_label(row) for row in rows
    )
    return {
        "schema": "step7l_phase2_candidate_request_summary_v1",
        "decision": _phase2_decision(rows, has_forbidden_label_terms),
        "request_path": str(request_path) if request_path is not None else None,
        "request_count": len(rows),
        "selected_case_count": selected_case_count,
        "requested_case_ids": [int(value) for value in requested_case_ids or []],
        "selected_case_ids": case_ids,
        "grid": grid_size,
        "top_blocks_per_family": top_blocks_per_family,
        "windows_per_block": windows_per_block,
        "request_count_by_family": dict(sorted(families.items())),
        "original_anchor_count": int(families.get("original_anchor", 0)),
        "global_report_only_count": sum(int(bool(row.get("global_report_only"))) for row in rows),
        "duplicate_signature_count": duplicate_rows,
        "unique_signature_count": len(signatures),
        "per_case_request_count_by_family": {
            case_id: dict(sorted(counter.items())) for case_id, counter in per_case.items()
        },
        "validation_label_policy": "visible validation inputs only; loader labels discarded",
        "uses_validation_target_labels": False,
        "has_forbidden_validation_label_terms": has_forbidden_label_terms,
        "route_class_contract": "request-only; no replay and no contest runtime/finalizer mutation",
    }


def summary_markdown(report: dict[str, Any]) -> str:
    return f"""# Step7L Phase 2 Learning-Guided Candidate Requests

Decision: `{report['decision']}`

## Key counts

- request_count: {report['request_count']}
- selected_case_count: {report['selected_case_count']}
- request_count_by_family: {report['request_count_by_family']}
- original_anchor_count: {report['original_anchor_count']}
- global_report_only_count: {report['global_report_only_count']}
- unique_signature_count: {report['unique_signature_count']}
- duplicate_signature_count: {report['duplicate_signature_count']}
- uses_validation_target_labels: {report['uses_validation_target_labels']}
- has_forbidden_validation_label_terms: {report['has_forbidden_validation_label_terms']}

## Contract

The JSONL deck is a request-only sidecar artifact. It compares `topology`,
`terminal`, and `union_diversified` target-window sources plus one
`original_anchor` row per selected validation case. It does not run replay,
read validation label tensors, or change contest runtime/finalizer code.
"""


def write_request_summary(report: dict[str, Any], path: Path) -> None:
    write_json(path, report)
    path.with_suffix(".md").write_text(summary_markdown(report), encoding="utf-8")


def _squeeze_one(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.squeeze(0) if tensor.ndim > 0 and int(tensor.shape[0]) == 1 else tensor


def _boundary_code(constraints: torch.Tensor, block_id: int) -> int:
    if constraints.ndim < 2 or block_id >= int(constraints.shape[0]) or constraints.shape[-1] <= 4:
        return 0
    return int(float(constraints[block_id, 4].item()))


def _has_terminal_demand(block_id: int, p2b_edges: torch.Tensor, pins_pos: torch.Tensor) -> bool:
    return terminal_weighted_centroid(block_id, p2b_edges, pins_pos) is not None


def _rank_family_candidates(
    *,
    family: Literal["topology", "terminal"],
    block_ids: list[int],
    p2b_edges: torch.Tensor,
    pins_pos: torch.Tensor,
    constraints: torch.Tensor,
    grid: GridSpec,
    top_blocks_per_family: int,
    windows_per_block: int,
) -> list[dict[str, Any]]:
    block_rows: list[dict[str, Any]] = []
    for block_id in block_ids:
        boundary_code = _boundary_code(constraints, block_id)
        terminal_demand = _has_terminal_demand(block_id, p2b_edges, pins_pos)
        if family == "terminal" and not terminal_demand:
            continue
        maps = build_block_heatmaps(
            block_id=block_id,
            boundary_code=boundary_code,
            p2b_edges=p2b_edges,
            pins_pos=pins_pos,
            grid=grid,
        )
        cells = top_cells(maps[family], k=max(windows_per_block, 1))
        best_score = float(cells[0]["score"]) if cells else 0.0
        block_rows.append(
            {
                "block_id": block_id,
                "boundary_code": boundary_code,
                "has_terminal_demand": terminal_demand,
                "cells": cells,
                "best_score": best_score,
                "source_model_family": family,
            }
        )
    block_rows.sort(
        key=lambda row: (
            -float(row["best_score"]),
            -int(bool(row["has_terminal_demand"])),
            -int(row["boundary_code"]),
            int(row["block_id"]),
        )
    )
    out: list[dict[str, Any]] = []
    for block_rank, row in enumerate(block_rows[: max(top_blocks_per_family, 0)]):
        for cell_rank, cell in enumerate(row["cells"][: max(windows_per_block, 0)]):
            out.append(
                {
                    "block_id": int(row["block_id"]),
                    "boundary_code": int(row["boundary_code"]),
                    "has_terminal_demand": bool(row["has_terminal_demand"]),
                    "cell": cell,
                    "heatmap_score": float(cell["score"]),
                    "source_model_family": family,
                    "block_rank": block_rank,
                    "cell_rank": cell_rank,
                }
            )
    return out


def _union_diversified(
    ranked_by_family: Mapping[str, list[dict[str, Any]]], *, limit: int
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    seen: set[tuple[int, int, int]] = set()
    families = ("topology", "terminal")
    max_len = max((len(ranked_by_family.get(family, [])) for family in families), default=0)
    for index in range(max_len):
        for family in families:
            rows = ranked_by_family.get(family, [])
            if index >= len(rows):
                continue
            row = dict(rows[index])
            cell = row["cell"]
            signature = (int(row["block_id"]), int(cell["row"]), int(cell["col"]))
            if signature in seen:
                continue
            seen.add(signature)
            row["source_model_family"] = family
            out.append(row)
            if len(out) >= limit:
                return out
    return out


def _target_window(grid: GridSpec, row: int, col: int) -> dict[str, Any]:
    cx, cy = grid.cell_center(row, col)
    return {
        "row": int(row),
        "col": int(col),
        "cx": cx,
        "cy": cy,
        "w": grid.w / max(grid.cols, 1),
        "h": grid.h / max(grid.rows, 1),
        "grid": {"rows": grid.rows, "cols": grid.cols},
        "frame": {"x": grid.x, "y": grid.y, "w": grid.w, "h": grid.h},
    }


def _move_family(boundary_code: int, has_terminal_demand: bool) -> str:
    if boundary_code > 0 and has_terminal_demand:
        return "boundary_terminal_target_window"
    if boundary_code > 0:
        return "boundary_aware_target_window"
    if has_terminal_demand:
        return "terminal_demand_target_window"
    return "topology_center_target_window"


def _anchor_request(
    case_id: str,
    loader_index: int,
    block_ids: list[int],
    grid: GridSpec,
) -> LearningGuidedTargetRequest:
    return LearningGuidedTargetRequest(
        schema="step7l_phase2_candidate_request_v1",
        case_id=case_id,
        requested_case_id=case_id,
        loader_index=loader_index,
        request_id=f"step7l_p2_case{case_id}_original_anchor",
        source_family="original_anchor",
        block_id=None,
        target_window=None,
        move_family="original_layout_anchor",
        route_class="original_anchor_sidecar",
        heatmap_score=None,
        has_terminal_demand=False,
        boundary_code=0,
        is_anchor=True,
        global_report_only=True,
        provenance={
            "source": "visible_validation_inputs_only",
            "block_count": len(block_ids),
            "grid_frame": {"x": grid.x, "y": grid.y, "w": grid.w, "h": grid.h},
            "validation_labels_discarded": True,
        },
        reason="baseline anchor for later replay comparison; no target label consumed",
    )


def _request_from_candidate(
    candidate: dict[str, Any],
    *,
    source_family: CandidateSourceFamily,
    case_id: str,
    loader_index: int,
    rank: int,
    grid: GridSpec,
) -> LearningGuidedTargetRequest:
    cell = candidate["cell"]
    block_id = int(candidate["block_id"])
    boundary_code = int(candidate["boundary_code"])
    has_terminal_demand = bool(candidate["has_terminal_demand"])
    request_id = (
        f"step7l_p2_case{case_id}_{source_family}_{rank:03d}"
        f"_b{block_id}_r{int(cell['row'])}c{int(cell['col'])}"
    )
    return LearningGuidedTargetRequest(
        schema="step7l_phase2_candidate_request_v1",
        case_id=case_id,
        requested_case_id=case_id,
        loader_index=loader_index,
        request_id=request_id,
        source_family=source_family,
        block_id=block_id,
        target_window=_target_window(grid, int(cell["row"]), int(cell["col"])),
        move_family=_move_family(boundary_code, has_terminal_demand),
        route_class="unrouted_request_sidecar",
        heatmap_score=float(candidate["heatmap_score"]),
        has_terminal_demand=has_terminal_demand,
        boundary_code=boundary_code,
        is_anchor=False,
        global_report_only=False,
        provenance={
            "source": "visible_validation_inputs_only",
            "source_model_family": candidate.get("source_model_family"),
            "block_rank": int(candidate.get("block_rank", rank)),
            "cell_rank": int(candidate.get("cell_rank", 0)),
            "validation_labels_discarded": True,
        },
        reason=(
            f"{source_family} heatmap target-window request; future repacker must legalize "
            "and replay before any quality claim"
        ),
    )


def _request_signature(row: dict[str, Any]) -> tuple[Any, ...]:
    window = row.get("target_window") or {}
    return (
        row.get("case_id"),
        row.get("source_family"),
        row.get("block_id"),
        window.get("row"),
        window.get("col"),
        row.get("is_anchor"),
    )


def _payload_mentions_forbidden_validation_label(row: dict[str, Any]) -> bool:
    text = json.dumps(row, sort_keys=True).lower()
    forbidden = ("polygons", "metrics", "fp_sol", "target_positions", "oracle_layout")
    return any(term in text for term in forbidden)


def _phase2_decision(rows: list[dict[str, Any]], has_forbidden_label_terms: bool) -> str:
    if has_forbidden_label_terms:
        return "fix_validation_label_leakage_before_replay"
    families = {str(row.get("source_family")) for row in rows}
    required = {"original_anchor", "topology", "terminal", "union_diversified"}
    if not required.issubset(families):
        return "fix_missing_request_family"
    if not rows:
        return "fix_empty_request_deck"
    return "promote_to_repacker_interface_design_not_replay"


def _natural_case_sort_key(value: str) -> tuple[int, str]:
    try:
        return int(value), value
    except ValueError:
        return 10**9, value
