"""Step7ML-I geometry-aware macro closure decoders.

The decoders are deterministic sidecar probes. They consume Step7P exact geometry
payloads, generate non-overlap placements by construction where possible, and
only then join Step7P/Step7N-I provenance for route/metric/gate comparison.
"""

from __future__ import annotations

import json
import math
import time
from collections import Counter
from pathlib import Path
from typing import Any

from puzzleplace.ml.floorset_training_corpus import write_json
from puzzleplace.ml.supervised_macro_layout import STEP7P_BASELINES, find_artifact
from puzzleplace.ml.training_backed_data_mart import as_float, rows

DECISIONS = {
    "promote_geometry_aware_decoder",
    "use_decoder_as_repack_initialization",
    "refine_slot_generation",
    "refine_shelf_or_order_decoder",
    "build_sequence_pair_micro_decoder",
    "combine_with_learned_ranker",
    "inconclusive_due_to_decoder_quality",
}
EPS = 1e-9


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def rect(block: dict[str, Any]) -> dict[str, float]:
    return {
        "x": as_float(block.get("x")),
        "y": as_float(block.get("y")),
        "w": as_float(block.get("w")),
        "h": as_float(block.get("h")),
    }


def bbox_for_blocks(blocks: list[dict[str, Any]]) -> dict[str, float]:
    x0 = min(as_float(block["x"]) for block in blocks)
    y0 = min(as_float(block["y"]) for block in blocks)
    x1 = max(as_float(block["x"]) + as_float(block["w"]) for block in blocks)
    y1 = max(as_float(block["y"]) + as_float(block["h"]) for block in blocks)
    return {"x": x0, "y": y0, "w": max(x1 - x0, 1.0), "h": max(y1 - y0, 1.0)}


def overlap_area(a: dict[str, Any], b: dict[str, Any]) -> float:
    ar = rect(a)
    br = rect(b)
    dx = min(ar["x"] + ar["w"], br["x"] + br["w"]) - max(ar["x"], br["x"])
    dy = min(ar["y"] + ar["h"], br["y"] + br["h"]) - max(ar["y"], br["y"])
    return max(0.0, dx) * max(0.0, dy)


def overlap_pair_count(blocks: list[dict[str, Any]]) -> int:
    count = 0
    for i in range(len(blocks)):
        for j in range(i + 1, len(blocks)):
            count += int(overlap_area(blocks[i], blocks[j]) > EPS)
    return count


def within_bbox(block: dict[str, Any], box: dict[str, float]) -> bool:
    r = rect(block)
    return (
        r["x"] >= box["x"] - EPS
        and r["y"] >= box["y"] - EPS
        and r["x"] + r["w"] <= box["x"] + box["w"] + EPS
        and r["y"] + r["h"] <= box["y"] + box["h"] + EPS
    )


def is_movable(block: dict[str, Any]) -> bool:
    return bool(block.get("movable", not (block.get("fixed") or block.get("preplaced"))))


def moved_count(original: list[dict[str, Any]], decoded: list[dict[str, Any]]) -> int:
    by_id = {int(block["block_id"]): block for block in original}
    count = 0
    for block in decoded:
        old = by_id.get(int(block["block_id"]))
        if old is None:
            continue
        count += int(abs(as_float(block["x"]) - as_float(old["x"])) > 1e-6)
        if abs(as_float(block["y"]) - as_float(old["y"])) > 1e-6:
            count += 0 if abs(as_float(block["x"]) - as_float(old["x"])) > 1e-6 else 1
    return count


def no_fixed_moved(original: list[dict[str, Any]], decoded: list[dict[str, Any]]) -> bool:
    by_id = {int(block["block_id"]): block for block in decoded}
    for block in original:
        if is_movable(block):
            continue
        new = by_id.get(int(block["block_id"]))
        if new is None:
            return False
        if abs(as_float(new["x"]) - as_float(block["x"])) > 1e-6:
            return False
        if abs(as_float(new["y"]) - as_float(block["y"])) > 1e-6:
            return False
    return True


def _ordered_blocks(blocks: list[dict[str, Any]], mode: str) -> list[dict[str, Any]]:
    if mode == "height_desc":
        return sorted(blocks, key=lambda b: (-as_float(b["h"]), as_float(b["x"]), as_float(b["y"])))
    if mode == "area_desc":
        return sorted(
            blocks,
            key=lambda b: (
                -(as_float(b["w"]) * as_float(b["h"])),
                as_float(b["x"]),
                as_float(b["y"]),
            ),
        )
    return sorted(
        blocks, key=lambda b: (as_float(b["y"]), as_float(b["x"]), int(b["block_id"]))
    )


def shelf_row_decoder(
    payload: dict[str, Any], *, order_mode: str = "height_desc"
) -> dict[str, Any]:
    blocks = [dict(block) for block in payload.get("blocks", [])]
    box = bbox_for_blocks(blocks)
    fixed = [dict(block) for block in blocks if not is_movable(block)]
    movable = _ordered_blocks([dict(block) for block in blocks if is_movable(block)], order_mode)
    placed = [dict(block) for block in fixed]
    x = box["x"]
    y = box["y"]
    row_h = 0.0
    failed = False
    for block in movable:
        w = as_float(block["w"])
        h = as_float(block["h"])
        if x + w > box["x"] + box["w"] + EPS:
            x = box["x"]
            y += row_h
            row_h = 0.0
        candidate = {**block, "x": x, "y": y}
        if y + h > box["y"] + box["h"] + EPS:
            failed = True
            candidate = dict(block)
        placed.append(candidate)
        x += w
        row_h = max(row_h, h)
    return _decoded_payload(payload, placed, "shelf_row_decoder", failed)


def generate_slots(box: dict[str, float], block: dict[str, Any]) -> list[dict[str, float]]:
    w = as_float(block["w"])
    h = as_float(block["h"])
    if w > box["w"] + EPS or h > box["h"] + EPS:
        return []
    step_x = max(1.0, min(w, box["w"] / 6.0))
    step_y = max(1.0, min(h, box["h"] / 6.0))
    xs = {box["x"], box["x"] + max(box["w"] - w, 0.0), as_float(block.get("x"))}
    ys = {box["y"], box["y"] + max(box["h"] - h, 0.0), as_float(block.get("y"))}
    n_x = int(max(1, math.floor(max(box["w"] - w, 0.0) / step_x)))
    n_y = int(max(1, math.floor(max(box["h"] - h, 0.0) / step_y)))
    for i in range(n_x + 1):
        xs.add(box["x"] + min(i * step_x, max(box["w"] - w, 0.0)))
    for j in range(n_y + 1):
        ys.add(box["y"] + min(j * step_y, max(box["h"] - h, 0.0)))
    slots = [
        {"x": float(x), "y": float(y), "w": w, "h": h}
        for y in sorted(ys)
        for x in sorted(xs)
        if x >= box["x"] - EPS
        and y >= box["y"] - EPS
        and x + w <= box["x"] + box["w"] + EPS
        and y + h <= box["y"] + box["h"] + EPS
    ]
    return slots


def slot_assignment_decoder(payload: dict[str, Any]) -> dict[str, Any]:
    blocks = [dict(block) for block in payload.get("blocks", [])]
    box = bbox_for_blocks(blocks)
    placed = [dict(block) for block in blocks if not is_movable(block)]
    movable = _ordered_blocks([dict(block) for block in blocks if is_movable(block)], "area_desc")
    slot_counts: list[int] = []
    failed = False
    for block in movable:
        slots = generate_slots(box, block)
        slot_counts.append(len(slots))
        best: dict[str, Any] | None = None
        best_score = float("inf")
        for slot in slots:
            candidate = {**block, "x": slot["x"], "y": slot["y"]}
            if any(overlap_area(candidate, other) > EPS for other in placed):
                continue
            # Separate cost components are intentionally simple and inspectable.
            displacement = abs(slot["x"] - as_float(block["x"])) + abs(
                slot["y"] - as_float(block["y"])
            )
            centroid_pull = abs((slot["x"] + slot["w"] / 2.0) - (box["x"] + box["w"] / 2.0))
            centroid_pull += abs((slot["y"] + slot["h"] / 2.0) - (box["y"] + box["h"] / 2.0))
            score = (displacement, centroid_pull, slot["y"], slot["x"])
            numeric_score = score[0] * 1_000_000.0 + score[1] * 1_000.0 + score[2] + score[3] * 1e-3
            if numeric_score < best_score:
                best = candidate
                best_score = numeric_score
        if best is None:
            failed = True
            placed.append(dict(block))
        else:
            placed.append(best)
    decoded = _decoded_payload(payload, placed, "slot_assignment_decoder", failed)
    decoded["slot_count"] = sum(slot_counts)
    decoded["slot_count_by_block"] = slot_counts
    return decoded


def _decoded_payload(
    payload: dict[str, Any],
    placed: list[dict[str, Any]],
    decoder: str,
    failed: bool,
) -> dict[str, Any]:
    original = [dict(block) for block in payload.get("blocks", [])]
    box = bbox_for_blocks(original)
    overlap = overlap_pair_count(placed)
    containment_fail = sum(1 for block in placed if not within_bbox(block, box))
    moved = moved_count(original, placed)
    return {
        "source_candidate_id": payload.get("candidate_id"),
        "candidate_id": f"{payload.get('candidate_id')}:{decoder}",
        "case_id": payload.get("case_id"),
        "target_region": payload.get("target_region"),
        "decoder": decoder,
        "decoded_blocks_preview": placed[:20],
        "block_count": len(placed),
        "moved_block_count": moved,
        "non_original_non_noop": moved > 0,
        "overlap_pair_count": overlap,
        "overlap_after_decode": overlap > 0,
        "bbox_containment_failure_count": containment_fail,
        "fixed_preplaced_blocked": not no_fixed_moved(original, placed),
        "no_slot_available": failed,
        "closure_bbox": box,
    }


def _baseline_rows(base_dir: Path) -> dict[str, dict[str, Any]]:
    path = find_artifact("step7p_real_repack_candidates.json", base_dir=base_dir)
    if path is None:
        return {}
    indexed: dict[str, dict[str, Any]] = {}
    for row in rows(load_json(path)):
        for key in ("candidate_id", "step7o_a_variant_candidate_id", "source_candidate_id"):
            if row.get(key):
                indexed[str(row[key])] = row
    return indexed


def _failure(row: dict[str, Any], baseline: dict[str, Any]) -> str:
    if row["no_slot_available"]:
        return "no_slot_available"
    if row["fixed_preplaced_blocked"]:
        return "fixed_preplaced_blocked"
    if row["bbox_containment_failure_count"]:
        return "bbox_containment_failure"
    if row["overlap_after_decode"]:
        return "overlap_after_decode"
    if not row["non_original_non_noop"]:
        return "no_op"
    if as_float(baseline.get("soft_constraint_delta")) > 0:
        return "soft_regression"
    if as_float(baseline.get("hpwl_delta")) > 0:
        return "hpwl_regression"
    if as_float(baseline.get("bbox_area_delta")) > 0:
        return "bbox_regression"
    if baseline.get("predicted_locality_class") == "global":
        return "route_global"
    return "none"


def run_geometry_aware_decoding(base_dir: Path, output_dir: Path) -> dict[str, Any]:
    started = time.perf_counter()
    payload_path = find_artifact("step7p_geometry_payloads.json", base_dir=base_dir)
    if payload_path is None:
        raise FileNotFoundError("Missing step7p_geometry_payloads.json")
    payloads = rows(load_json(payload_path))
    baseline_by_id = _baseline_rows(base_dir)
    decoded: list[dict[str, Any]] = []
    slot_rows: list[dict[str, Any]] = []
    for payload in payloads:
        variants = [slot_assignment_decoder(payload), shelf_row_decoder(payload)]
        for row in variants:
            baseline = baseline_by_id.get(str(row["source_candidate_id"]), {})
            hard_feasible = (
                row["non_original_non_noop"]
                and not row["overlap_after_decode"]
                and row["bbox_containment_failure_count"] == 0
                and not row["fixed_preplaced_blocked"]
            )
            official = bool(baseline.get("official_like_cost_improving")) and hard_feasible
            gate = bool(baseline.get("step7p_selected_for_archive")) and hard_feasible
            row.update(
                {
                    "hard_feasible_non_noop": hard_feasible,
                    "official_like_improving": official,
                    "quality_gate_pass": gate,
                    "route_class": baseline.get("source_actual_locality_class")
                    or baseline.get("predicted_locality_class"),
                    "dominated_by_original": baseline.get("dominated_by_original"),
                    "metric_regressing": as_float(baseline.get("official_like_cost_delta")) > 0
                    and not official,
                    "failure_attribution": _failure(row, baseline),
                    "baseline_quality_gate_status": baseline.get("quality_gate_status"),
                    "baseline_hard_feasible_non_noop": baseline.get("hard_feasible_non_noop"),
                }
            )
            decoded.append(row)
            if row["decoder"] == "slot_assignment_decoder":
                slot_rows.append(
                    {
                        "candidate_id": row["candidate_id"],
                        "source_candidate_id": row["source_candidate_id"],
                        "case_id": row["case_id"],
                        "slot_count": row.get("slot_count", 0),
                        "slot_count_by_block": row.get("slot_count_by_block", []),
                    }
                )
    summary = _summary(decoded, payloads, started, str(payload_path))
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "step7ml_i_decoder_config.json", _config(str(payload_path)))
    write_json(
        output_dir / "step7ml_i_slot_candidates.json",
        {"schema": "step7ml_i_slot_candidates_v1", "rows": slot_rows},
    )
    write_json(
        output_dir / "step7ml_i_decoded_candidates.json",
        {"schema": "step7ml_i_decoded_candidates_v1", "rows": decoded},
    )
    write_json(output_dir / "step7ml_i_route_report.json", _route_report(decoded))
    write_json(output_dir / "step7ml_i_feasibility_report.json", _feasibility_report(summary))
    write_json(output_dir / "step7ml_i_metric_report.json", _metric_report(summary))
    write_json(output_dir / "step7ml_i_quality_gate_report.json", _quality_gate_report(summary))
    write_json(output_dir / "step7ml_i_decoder_ablation.json", _ablation(decoded))
    write_json(output_dir / "step7ml_i_failure_attribution.json", _failure_report(decoded))
    decision = decide(summary)
    (output_dir / "step7ml_i_decision.md").write_text(
        _decision_md(decision, summary), encoding="utf-8"
    )
    return {"decision": decision, "summary": summary}


def _summary(
    decoded: list[dict[str, Any]],
    payloads: list[dict[str, Any]],
    started: float,
    payload_path: str,
) -> dict[str, Any]:
    overlap = sum(1 for row in decoded if row["overlap_after_decode"])
    hard = sum(1 for row in decoded if row["hard_feasible_non_noop"])
    official = sum(1 for row in decoded if row["official_like_improving"])
    gate = sum(1 for row in decoded if row["quality_gate_pass"])
    return {
        "schema": "step7ml_i_summary_v1",
        "payload_source": payload_path,
        "step7p_payload_count": len(payloads),
        "decoded_candidate_count": len(decoded),
        "candidate_count_by_decoder": dict(Counter(str(row["decoder"]) for row in decoded)),
        "slot_count_by_payload": {
            str(row["source_candidate_id"]): row.get("slot_count", 0)
            for row in decoded
            if row["decoder"] == "slot_assignment_decoder"
        },
        "non_original_non_noop_count": sum(1 for row in decoded if row["non_original_non_noop"]),
        "hard_feasible_non_noop_count": hard,
        "overlap_after_decode_count": overlap,
        "overlap_reduction_vs_step7p": STEP7P_BASELINES["overlap_after_repack"] - overlap,
        "overlap_reduction_vs_step7ml_h": 71 - overlap,
        "official_like_improving_count": official,
        "quality_gate_pass_count": gate,
        "dominated_by_original_count": sum(
            1 for row in decoded if row.get("dominated_by_original")
        ),
        "metric_regressing_count": sum(1 for row in decoded if row.get("metric_regressing")),
        "route_count_by_class": dict(Counter(str(row.get("route_class")) for row in decoded)),
        "no_slot_available_count": sum(1 for row in decoded if row["no_slot_available"]),
        "fixed_preplaced_blocked_count": sum(
            1 for row in decoded if row["fixed_preplaced_blocked"]
        ),
        "bbox_containment_failure_count": sum(
            1 for row in decoded if row["bbox_containment_failure_count"] > 0
        ),
        "no_op_count": sum(1 for row in decoded if not row["non_original_non_noop"]),
        "runtime_proxy_per_payload_ms": ((time.perf_counter() - started) * 1000.0)
        / max(len(payloads), 1),
        "comparison_baselines": {"step7p": STEP7P_BASELINES, "step7ml_h_overlap": 71},
    }


def _config(payload_path: str) -> dict[str, Any]:
    return {
        "schema": "step7ml_i_decoder_config_v1",
        "payload_source": payload_path,
        "decoders": ["slot_assignment_decoder", "shelf_row_decoder"],
        "learned_prior_usage": (
            "Step7ML-H motivated geometry-aware decoding; this probe uses "
            "deterministic geometry costs and Step7P provenance joins."
        ),
        "hard_constraints": [
            "no_overlap",
            "bbox_containment",
            "fixed_preplaced_preservation",
            "no_op_detection",
        ],
    }


def _route_report(decoded: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "schema": "step7ml_i_route_report_v1",
        "route_count_by_class": dict(Counter(str(row.get("route_class")) for row in decoded)),
        "route_semantics": (
            "Route class is joined from Step7P/Step7G provenance after geometry screening."
        ),
    }


def _feasibility_report(summary: dict[str, Any]) -> dict[str, Any]:
    return {"schema": "step7ml_i_feasibility_report_v1", "summary": summary}


def _metric_report(summary: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema": "step7ml_i_metric_report_v1",
        "summary": summary,
        "metric_semantics": (
            "Official-like metric labels are preserved from Step7P baselines only for "
            "decoded candidates that pass deterministic geometry screening."
        ),
    }


def _quality_gate_report(summary: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema": "step7ml_i_quality_gate_report_v1",
        "quality_gate_pass_count": summary["quality_gate_pass_count"],
        "official_like_improving_count": summary["official_like_improving_count"],
        "comparison_baselines": summary["comparison_baselines"],
    }


def _ablation(decoded: list[dict[str, Any]]) -> dict[str, Any]:
    rows_out = []
    for decoder in sorted({row["decoder"] for row in decoded}):
        subset = [row for row in decoded if row["decoder"] == decoder]
        rows_out.append(
            {
                "decoder": decoder,
                "candidate_count": len(subset),
                "hard_feasible_non_noop_count": sum(
                    1 for row in subset if row["hard_feasible_non_noop"]
                ),
                "overlap_after_decode_count": sum(
                    1 for row in subset if row["overlap_after_decode"]
                ),
                "quality_gate_pass_count": sum(1 for row in subset if row["quality_gate_pass"]),
                "official_like_improving_count": sum(
                    1 for row in subset if row["official_like_improving"]
                ),
            }
        )
    return {"schema": "step7ml_i_decoder_ablation_v1", "rows": rows_out}


def _failure_report(decoded: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "schema": "step7ml_i_failure_attribution_v1",
        "reason_counts": dict(Counter(str(row["failure_attribution"]) for row in decoded)),
        "by_decoder": {
            decoder: dict(
                Counter(
                    str(row["failure_attribution"])
                    for row in decoded
                    if row["decoder"] == decoder
                )
            )
            for decoder in sorted({row["decoder"] for row in decoded})
        },
    }


def decide(summary: dict[str, Any]) -> str:
    if summary["hard_feasible_non_noop_count"] > STEP7P_BASELINES["hard_feasible_non_noop"]:
        if summary["quality_gate_pass_count"] > STEP7P_BASELINES["quality_gate_pass"]:
            return "promote_geometry_aware_decoder"
        return "use_decoder_as_repack_initialization"
    if summary["no_slot_available_count"] > 0:
        return "refine_slot_generation"
    if summary["overlap_after_decode_count"] >= STEP7P_BASELINES["overlap_after_repack"]:
        return "build_sequence_pair_micro_decoder"
    return "refine_shelf_or_order_decoder"


def _decision_md(decision: str, summary: dict[str, Any]) -> str:
    return f"""# Step7ML-I Geometry-Aware Decoding / Slot Legalization Layer

Decision: `{decision}`

## Key metrics

- step7p_payload_count: {summary['step7p_payload_count']}
- decoded_candidate_count: {summary['decoded_candidate_count']}
- candidate_count_by_decoder: {summary['candidate_count_by_decoder']}
- non_original_non_noop_count: {summary['non_original_non_noop_count']}
- hard_feasible_non_noop_count: {summary['hard_feasible_non_noop_count']} vs Step7P baseline 11
- overlap_after_decode_count: {summary['overlap_after_decode_count']} vs Step7P baseline 53
  and Step7ML-H baseline 71
- overlap_reduction_vs_step7p: {summary['overlap_reduction_vs_step7p']}
- overlap_reduction_vs_step7ml_h: {summary['overlap_reduction_vs_step7ml_h']}
- official_like_improving_count: {summary['official_like_improving_count']} vs Step7P baseline 2
- quality_gate_pass_count: {summary['quality_gate_pass_count']} vs Step7P baseline 2
- no_slot_available_count: {summary['no_slot_available_count']}
- fixed_preplaced_blocked_count: {summary['fixed_preplaced_blocked_count']}
- bbox_containment_failure_count: {summary['bbox_containment_failure_count']}
- route_count_by_class: {summary['route_count_by_class']}

## Interpretation

Step7ML-I replaces independent x/y regression as the final placement with
geometry-aware deterministic slot/shelf decoders. Outputs are screened for
non-overlap, bbox containment, fixed/preplaced preservation, and no-op before
Step7P metric/gate provenance is joined. If hard-feasible candidates increase but
quality-gate wins do not, the decoder is useful as initialization only and needs
ranker/metric-aware selection next.
"""
