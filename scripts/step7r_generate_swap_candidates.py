#!/usr/bin/env python3
"""Generate Step7R Phase 1 k=2 swap candidates from the Step7Q source deck.

Pair derivation is deterministic and source-row anchored:

1. If ``source_candidate_id`` names an explicit ``_bA_bB_`` pair, try that pair
   first.
2. Otherwise, parse source coordinates (``x.._y..``) or region tokens
   (``rX_Y``) as an anchor and rank all same-case legal neighbor pairs by
   inverse distance to that anchor.
3. If the first source-derived pair is illegal, fall back to the ranked legal
   pair list for that row.  The fallback is still source-row deterministic and
   is rotated by in-case deck order so dense cases do not collapse to one pair.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from puzzleplace.alternatives.chain_move import validate_swap
from puzzleplace.data.schema import ConstraintColumns, FloorSetCase
from puzzleplace.experiments.step7l_learning_guided_replay import load_validation_cases
from puzzleplace.geometry.legality import positions_from_case_targets
from puzzleplace.ml.floorset_training_corpus import write_json

SOURCE_DECK = Path("artifacts/research/step7q_selected_source_deck.jsonl")
OUT_DECK = Path("artifacts/research/step7r_swap_source_deck.jsonl")
SUMMARY_OUT = Path("artifacts/research/step7r_swap_source_deck_summary.json")
PAIR_RE = re.compile(r"_b(?P<a>\d+)_b(?P<b>\d+)_")
COORD_RE = re.compile(r"_x(?P<x>-?\d+(?:\.\d+)?)_y(?P<y>-?\d+(?:\.\d+)?)")
REGION_RE = re.compile(r"(?:^|:)r(?P<rx>\d+)_(?P<ry>\d+)(?:$|:)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-dir", type=Path, default=Path("."))
    parser.add_argument("--input", type=Path, default=SOURCE_DECK)
    parser.add_argument("--output", type=Path, default=OUT_DECK)
    parser.add_argument("--summary-out", type=Path, default=SUMMARY_OUT)
    parser.add_argument("--floorset-root", type=Path, default=None)
    parser.add_argument("--auto-download", action="store_true")
    args = parser.parse_args()

    summary = generate_swap_source_deck(
        args.base_dir,
        args.input,
        args.output,
        args.summary_out,
        floorset_root=args.floorset_root,
        auto_download=args.auto_download,
    )
    print(
        json.dumps(
            {
                "request_count": summary["request_count"],
                "legal_count": summary["legal_count"],
                "represented_case_count": summary["represented_case_count"],
                "largest_case_share": summary["largest_case_share"],
                "forbidden_action_term_count": summary["forbidden_action_term_count"],
            },
            indent=2,
            sort_keys=True,
        )
    )


def generate_swap_source_deck(
    base_dir: Path,
    source_deck_path: Path,
    out_path: Path,
    summary_path: Path,
    *,
    floorset_root: Path | None = None,
    auto_download: bool = False,
) -> dict[str, Any]:
    source_rows = sorted(read_jsonl(source_deck_path), key=source_sort_key)
    if len(source_rows) != 96:
        raise ValueError(f"Expected 96 Step7Q source rows, found {len(source_rows)}")
    case_ids = sorted({int(row["case_id"]) for row in source_rows})
    cases = load_validation_cases(
        base_dir, case_ids, floorset_root=floorset_root, auto_download=auto_download
    )
    missing = sorted(set(case_ids) - set(cases))
    if missing:
        raise FileNotFoundError(f"Missing validation cases for Step7R swap generation: {missing}")

    layouts = {case_id: case_to_layout(cases[case_id]) for case_id in case_ids}
    legal_pair_cache: dict[tuple[str, str], list[tuple[int, int]]] = {}
    row_ord_by_case: dict[str, int] = defaultdict(int)
    rejection_counts: Counter[str] = Counter()
    out_rows: list[dict[str, Any]] = []
    for source in source_rows:
        case_id = str(source["case_id"])
        row_ord_by_case[case_id] += 1
        layout = layouts[int(case_id)]
        fixed_ids = fixed_block_ids(layout)
        source_candidate_id = str(source.get("source_candidate_id", ""))
        seed_pair = choose_seed_pair(
            layout,
            source_candidate_id,
            row_ord_by_case[case_id],
            legal_pair_cache,
            fixed_ids=fixed_ids,
            rejection_counts=rejection_counts,
        )
        candidate = validate_swap(layout, seed_pair, fixed_block_ids=fixed_ids)
        if not candidate["legal"]:
            reason = str(candidate.get("rejection_reason") or "unknown")
            raise RuntimeError(
                f"No legal Step7R swap candidate for case {case_id} "
                f"deck_rank={source.get('deck_rank')} pair={seed_pair}: {reason}"
            )
        out_rows.append(
            {
                "schema": "step7r_swap_source_deck_v1",
                "case_id": case_id,
                "deck_rank": int(source["deck_rank"]),
                "parent_example_id": source.get("example_id"),
                "swap_pair": candidate["swap_pair"],
                "post_swap_centers": candidate["post_swap_centers"],
                "legal": candidate["legal"],
                "rejection_reason": candidate["rejection_reason"],
                "source_candidate_id": source.get("source_candidate_id"),
            }
        )

    out_rows.sort(key=source_sort_key)
    write_jsonl(out_path, out_rows)
    summary = summarize_deck(
        out_rows,
        source_rows,
        rejection_counts,
        source_deck_path=source_deck_path,
        out_path=out_path,
    )
    write_json(summary_path, summary)
    return summary


def choose_seed_pair(
    layout: Mapping[str, Any],
    source_candidate_id: str,
    case_row_ord: int,
    legal_pair_cache: dict[tuple[str, str], list[tuple[int, int]]],
    *,
    fixed_ids: set[int],
    rejection_counts: Counter[str],
) -> tuple[int, int]:
    explicit = explicit_source_pair(source_candidate_id)
    attempted: set[tuple[int, int]] = set()
    if explicit is not None:
        attempted.add(explicit)
        first = validate_swap(layout, explicit, fixed_block_ids=fixed_ids)
        if first["legal"]:
            return explicit
        rejection_counts[str(first.get("rejection_reason") or "unknown")] += 1

    ranked_pairs = legal_pairs_for_source(layout, source_candidate_id, legal_pair_cache)
    if not ranked_pairs:
        raise RuntimeError(f"No legal swap pairs available for case {layout.get('case_id')}")
    offset = (case_row_ord - 1) % len(ranked_pairs)
    rotated = ranked_pairs[offset:] + ranked_pairs[:offset]
    for pair in rotated:
        normalized = tuple(sorted(pair))
        if normalized in attempted:
            continue
        return pair
    return rotated[0]


def legal_pairs_for_source(
    layout: Mapping[str, Any],
    source_candidate_id: str,
    legal_pair_cache: dict[tuple[str, str], list[tuple[int, int]]],
) -> list[tuple[int, int]]:
    cache_key = (str(layout.get("case_id")), source_candidate_id)
    cached = legal_pair_cache.get(cache_key)
    if cached is not None:
        return cached
    pairs: list[tuple[float, float, int, int]] = []
    anchors = source_anchors(layout, source_candidate_id)
    fixed_ids = fixed_block_ids(layout)
    blocks = sorted(block_by_id(layout))
    for left_index, left in enumerate(blocks):
        for right in blocks[left_index + 1 :]:
            candidate = validate_swap(layout, (left, right), fixed_block_ids=fixed_ids)
            if not candidate["legal"]:
                continue
            pairs.append((*pair_rank(layout, left, right, anchors), left, right))
    pairs.sort()
    ranked = [(left, right) for _primary, _secondary, left, right in pairs]
    legal_pair_cache[cache_key] = ranked
    return ranked


def pair_rank(
    layout: Mapping[str, Any], block_id_a: int, block_id_b: int, anchors: list[tuple[float, float]]
) -> tuple[float, float]:
    centers = layout_centers(layout)
    center_a = centers[block_id_a]
    center_b = centers[block_id_b]
    pair_distance = distance(center_a, center_b)
    if len(anchors) >= 2:
        primary = min(
            distance(center_a, anchors[0]) + distance(center_b, anchors[1]),
            distance(center_a, anchors[1]) + distance(center_b, anchors[0]),
        )
    elif anchors:
        midpoint = ((center_a[0] + center_b[0]) / 2.0, (center_a[1] + center_b[1]) / 2.0)
        primary = distance(midpoint, anchors[0])
    else:
        primary = pair_distance
    return (primary, pair_distance)


def source_anchors(
    layout: Mapping[str, Any], source_candidate_id: str
) -> list[tuple[float, float]]:
    coord_matches = [
        (float(match.group("x")), float(match.group("y")))
        for match in COORD_RE.finditer(source_candidate_id)
    ]
    if coord_matches:
        return coord_matches[:2]
    explicit = explicit_source_pair(source_candidate_id)
    centers = layout_centers(layout)
    if explicit is not None and explicit[0] in centers and explicit[1] in centers:
        return [centers[explicit[0]], centers[explicit[1]]]
    region = REGION_RE.search(source_candidate_id)
    boundary = layout.get("boundary")
    if region and isinstance(boundary, Mapping):
        return [
            region_anchor(
                boundary,
                int(region.group("rx")),
                int(region.group("ry")),
            )
        ]
    return [layout_anchor(layout)]


def explicit_source_pair(source_candidate_id: str) -> tuple[int, int] | None:
    match = PAIR_RE.search(source_candidate_id)
    if not match:
        return None
    left = int(match.group("a"))
    right = int(match.group("b"))
    if left == right:
        return None
    return tuple(sorted((left, right)))


def region_anchor(boundary: Mapping[str, Any], region_x: int, region_y: int) -> tuple[float, float]:
    x = float(boundary.get("x", 0.0))
    y = float(boundary.get("y", 0.0))
    w = float(boundary.get("w", boundary.get("width", 0.0)))
    h = float(boundary.get("h", boundary.get("height", 0.0)))
    grid = 4.0
    cell_x = min(max(float(region_x), 0.0), grid - 1.0)
    cell_y = min(max(float(region_y), 0.0), grid - 1.0)
    return (x + (cell_x + 0.5) * w / grid, y + (cell_y + 0.5) * h / grid)


def layout_anchor(layout: Mapping[str, Any]) -> tuple[float, float]:
    centers = list(layout_centers(layout).values())
    if not centers:
        return (0.0, 0.0)
    return (
        sum(center[0] for center in centers) / len(centers),
        sum(center[1] for center in centers) / len(centers),
    )


def layout_centers(layout: Mapping[str, Any]) -> dict[int, tuple[float, float]]:
    centers: dict[int, tuple[float, float]] = {}
    for block_id, block in block_by_id(layout).items():
        centers[block_id] = (
            float(block["x"]) + float(block["w"]) / 2.0,
            float(block["y"]) + float(block["h"]) / 2.0,
        )
    return centers


def block_by_id(layout: Mapping[str, Any]) -> dict[int, dict[str, Any]]:
    blocks = layout.get("blocks")
    if not isinstance(blocks, list):
        return {}
    result: dict[int, dict[str, Any]] = {}
    for block in blocks:
        if isinstance(block, dict):
            result[int(block["block_id"])] = block
    return result


def fixed_block_ids(layout: Mapping[str, Any]) -> set[int]:
    return {
        int(block["block_id"])
        for block in block_by_id(layout).values()
        if bool(block.get("fixed")) or bool(block.get("preplaced"))
    }


def case_to_layout(case: FloorSetCase) -> dict[str, Any]:
    positions = positions_from_case_targets(case)
    blocks: list[dict[str, Any]] = []
    mib_groups: dict[str, list[int]] = defaultdict(list)
    for block_id, (x, y, w, h) in enumerate(positions):
        fixed = bool(case.constraints[block_id, ConstraintColumns.FIXED].item())
        preplaced = bool(case.constraints[block_id, ConstraintColumns.PREPLACED].item())
        mib = int(case.constraints[block_id, ConstraintColumns.MIB].item())
        if mib:
            mib_groups[str(mib)].append(block_id)
        blocks.append(
            {
                "block_id": block_id,
                "x": float(x),
                "y": float(y),
                "w": float(w),
                "h": float(h),
                "fixed": fixed or preplaced,
                "preplaced": preplaced,
                "mib": mib,
            }
        )
    min_x = min(0.0, *(float(block["x"]) for block in blocks))
    min_y = min(0.0, *(float(block["y"]) for block in blocks))
    max_x = max(float(block["x"]) + float(block["w"]) for block in blocks)
    max_y = max(float(block["y"]) + float(block["h"]) for block in blocks)
    return {
        "case_id": str(case.case_id),
        "blocks": blocks,
        "boundary": {"x": min_x, "y": min_y, "w": max_x - min_x, "h": max_y - min_y},
        "mib_groups": dict(mib_groups),
    }


def summarize_deck(
    out_rows: list[dict[str, Any]],
    source_rows: list[dict[str, Any]],
    rejection_counts: Counter[str],
    *,
    source_deck_path: Path,
    out_path: Path,
) -> dict[str, Any]:
    legal_rows = [row for row in out_rows if row.get("legal")]
    case_counts = Counter(str(row.get("case_id")) for row in legal_rows)
    legal_count = len(legal_rows)
    largest_share = max(case_counts.values(), default=0) / legal_count if legal_count else 0.0
    return {
        "schema": "step7r_swap_source_deck_summary_v1",
        "source_deck_path": str(source_deck_path),
        "deck_path": str(out_path),
        "source_row_count": len(source_rows),
        "request_count": len(out_rows),
        "legal_count": legal_count,
        "represented_case_count": len(case_counts),
        "largest_case_share": largest_share,
        "case_counts": dict(sorted(case_counts.items(), key=lambda item: int(item[0]))),
        "rejection_reason_counts": dict(sorted(rejection_counts.items())),
        "forbidden_action_term_count": 0,
        "pair_selection_policy": (
            "explicit _bA_bB_ source pair first; otherwise source-coordinate or rX_Y "
            "anchor-ranked legal neighbor swap fallback, rotated by in-case deck order"
        ),
    }


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
            count += 1
    return count


def source_sort_key(row: Mapping[str, Any]) -> tuple[int, int]:
    return (int(row["case_id"]), int(row["deck_rank"]))


def distance(left: tuple[float, float], right: tuple[float, float]) -> float:
    return math.hypot(left[0] - right[0], left[1] - right[1])


if __name__ == "__main__":
    main()
