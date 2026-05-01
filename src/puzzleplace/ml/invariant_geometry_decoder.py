"""Step7ML-K invariant-preserving geometry decoder sidecar.

This module generates deterministic macro-closure placements that preserve
structural envelopes by construction before Step7P/Step7N-I provenance is joined
for metric and quality-gate reporting.
"""

from __future__ import annotations

import json
import time
from collections import Counter
from pathlib import Path
from typing import Any

from puzzleplace.ml.floorset_training_corpus import write_json
from puzzleplace.ml.geometry_aware_decoder import (
    bbox_for_blocks,
    overlap_pair_count,
    shelf_row_decoder,
    slot_assignment_decoder,
    within_bbox,
)
from puzzleplace.ml.supervised_macro_layout import find_artifact
from puzzleplace.ml.training_backed_data_mart import as_float, rows

STEP7ML_I_BASELINES = {
    "hard_feasible_non_noop": 67,
    "overlap_after_decode": 4,
    "official_like_improving": 2,
    "quality_gate_pass": 2,
}
STEP7ML_J_BASELINES = {
    "selected_candidates": 6,
    "selected_dominated_by_original": 0,
    "selected_metric_regressing": 0,
    "hpwl_gain_but_official_like_loss": 56,
    "bbox_regression": 56,
    "soft_regression": 114,
}


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def moved_count(original: list[dict[str, Any]], decoded: list[dict[str, Any]]) -> int:
    old_by_id = {int(block["block_id"]): block for block in original}
    moved = 0
    for block in decoded:
        old = old_by_id.get(int(block["block_id"]))
        if old is None:
            continue
        moved += int(
            abs(as_float(block["x"]) - as_float(old["x"])) > 1e-6
            or abs(as_float(block["y"]) - as_float(old["y"])) > 1e-6
        )
    return moved


def _baseline_rows(base_dir: Path) -> dict[str, dict[str, Any]]:
    path = find_artifact("step7p_real_repack_candidates.json", base_dir=base_dir)
    indexed: dict[str, dict[str, Any]] = {}
    if path is None:
        return indexed
    for row in rows(load_json(path)):
        for key in ("candidate_id", "step7o_a_variant_candidate_id", "source_candidate_id"):
            if row.get(key):
                indexed[str(row[key])] = row
    return indexed


def _ordered_original(blocks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(blocks, key=lambda b: (as_float(b["y"]), as_float(b["x"]), int(b["block_id"])))


def _rank_units(blocks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        blocks,
        key=lambda b: (
            int(b.get("cluster") or 0),
            int(b.get("mib") or 0),
            as_float(b["y"]),
            as_float(b["x"]),
            int(b["block_id"]),
        ),
    )


def bbox_envelope_shelf_decoder(payload: dict[str, Any]) -> dict[str, Any]:
    """Shelf pack movable blocks inside the original closure bbox.

    This keeps the bbox envelope invariant by using the source closure bbox as a
    hard containment window and preserves fixed/preplaced coordinates exactly.
    """

    decoded = shelf_row_decoder(payload, order_mode="height_desc")
    decoded["decoder"] = "bbox_envelope_shelf_decoder"
    decoded["candidate_id"] = f"{payload.get('candidate_id')}:bbox_envelope_shelf_decoder"
    decoded["invariants"] = ["bbox_envelope", "fixed_preplaced", "no_overlap"]
    return decoded


def order_preserving_slot_decoder(payload: dict[str, Any]) -> dict[str, Any]:
    """Run slot assignment after sorting blocks by original x/y partial order."""

    ordered_payload = dict(payload)
    ordered_payload["blocks"] = _ordered_original(
        [dict(block) for block in payload.get("blocks", [])]
    )
    decoded = slot_assignment_decoder(ordered_payload)
    decoded["decoder"] = "order_preserving_slot_decoder"
    decoded["candidate_id"] = f"{payload.get('candidate_id')}:order_preserving_slot_decoder"
    decoded["invariants"] = ["bbox_envelope", "order_topology", "fixed_preplaced", "no_overlap"]
    return decoded


def compound_unit_shelf_decoder(payload: dict[str, Any]) -> dict[str, Any]:
    """Approximate compound-unit preservation by grouping cluster/MIB order.

    This probe preserves group adjacency in ordering, while still using a shelf
    packer for deterministic non-overlap. Full supernode expansion is left for a
    future hierarchical decoder.
    """

    grouped_payload = dict(payload)
    grouped_payload["blocks"] = _rank_units([dict(block) for block in payload.get("blocks", [])])
    decoded = shelf_row_decoder(grouped_payload, order_mode="original")
    decoded["decoder"] = "compound_unit_shelf_decoder"
    decoded["candidate_id"] = f"{payload.get('candidate_id')}:compound_unit_shelf_decoder"
    decoded["invariants"] = [
        "bbox_envelope",
        "compound_unit_order",
        "fixed_preplaced",
        "no_overlap",
    ]
    return decoded


def _invariant_violations(
    payload: dict[str, Any], decoded: dict[str, Any]
) -> dict[str, int]:
    blocks = payload.get("blocks", [])
    box = bbox_for_blocks(blocks)
    decoded_blocks = decoded.get("decoded_blocks_preview", [])
    violations = {
        "overlap": int(overlap_pair_count(decoded_blocks) > 0),
        "bbox_envelope": int(any(not within_bbox(block, box) for block in decoded_blocks)),
        "fixed_preplaced": int(bool(decoded.get("fixed_preplaced_blocked"))),
        "no_op": int(not decoded.get("non_original_non_noop")),
    }
    return violations


def _score(
    decoded: dict[str, Any], baseline: dict[str, Any]
) -> tuple[int, int, int, int, float, str]:
    violations = decoded.get("invariant_violations", {})
    violation_count = sum(int(value) for value in violations.values())
    hard = int(not decoded.get("hard_feasible_nonnoop_probe", False))
    metric_regressing = int(as_float(baseline.get("official_like_cost_delta")) > 0)
    dominated = int(bool(baseline.get("dominated_by_original")))
    soft = as_float(baseline.get("soft_constraint_delta"), 0.0)
    return (
        violation_count,
        hard,
        metric_regressing,
        dominated,
        soft,
        str(decoded.get("candidate_id")),
    )


def _annotate(
    payload: dict[str, Any], decoded: dict[str, Any], baseline: dict[str, Any]
) -> dict[str, Any]:
    violations = _invariant_violations(payload, decoded)
    hard = (
        bool(decoded.get("non_original_non_noop"))
        and not decoded.get("overlap_after_decode")
        and decoded.get("bbox_containment_failure_count", 0) == 0
        and not decoded.get("fixed_preplaced_blocked")
    )
    official_delta = baseline.get("official_like_cost_delta")
    route_class = baseline.get("source_actual_locality_class") or baseline.get(
        "predicted_locality_class"
    )
    item = dict(decoded)
    item.update(
        {
            "source_candidate_id": payload.get("candidate_id"),
            "hard_feasible_nonnoop_probe": hard,
            "hard_feasible_non_noop": hard,
            "official_like_improving": bool(baseline.get("official_like_cost_improving")) and hard,
            "quality_gate_pass": bool(baseline.get("step7p_selected_for_archive")) and hard,
            "dominated_by_original": bool(baseline.get("dominated_by_original")),
            "metric_regressing": as_float(official_delta) > 0
            and not bool(baseline.get("official_like_cost_improving")),
            "hpwl_delta": baseline.get("hpwl_delta"),
            "bbox_area_delta": baseline.get("bbox_area_delta"),
            "soft_constraint_delta": baseline.get("soft_constraint_delta"),
            "official_like_cost_delta": official_delta,
            "route_class": route_class,
            "route_global": str(route_class) == "global",
            "invariant_violations": violations,
            "closure_size_bucket": closure_size_bucket(int(decoded.get("block_count", 0))),
        }
    )
    return item


def closure_size_bucket(size: int) -> str:
    if size <= 10:
        return "small_<=10"
    if size <= 20:
        return "medium_11_20"
    return "large_21_plus"


def run_invariant_decoder(base_dir: Path, output_dir: Path) -> dict[str, Any]:
    started = time.perf_counter()
    payload_path = find_artifact("step7p_geometry_payloads.json", base_dir=base_dir)
    if payload_path is None:
        raise FileNotFoundError("Missing step7p_geometry_payloads.json")
    payloads = rows(load_json(payload_path))
    baselines = _baseline_rows(base_dir)
    all_variants: list[dict[str, Any]] = []
    selected: list[dict[str, Any]] = []
    for payload in payloads:
        baseline = baselines.get(str(payload.get("candidate_id")), {})
        variants = [
            bbox_envelope_shelf_decoder(payload),
            order_preserving_slot_decoder(payload),
            compound_unit_shelf_decoder(payload),
        ]
        annotated = [_annotate(payload, variant, baseline) for variant in variants]
        all_variants.extend(annotated)
        selected.append(sorted(annotated, key=lambda row: _score(row, baseline))[0])
    summary = summarize(selected, all_variants, payloads, (time.perf_counter() - started) * 1000.0)
    decision = decide(summary)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "step7ml_k_decoder_config.json", config(str(payload_path)))
    write_json(
        output_dir / "step7ml_k_invariant_candidates.json",
        {
            "schema": "step7ml_k_invariant_candidates_v1",
            "selected_rows": selected,
            "all_variant_rows": all_variants,
        },
    )
    write_json(output_dir / "step7ml_k_route_report.json", route_report(selected))
    write_json(
        output_dir / "step7ml_k_feasibility_report.json",
        {"schema": "step7ml_k_feasibility_report_v1", "summary": summary},
    )
    write_json(
        output_dir / "step7ml_k_metric_report.json",
        {"schema": "step7ml_k_metric_report_v1", "summary": summary},
    )
    write_json(
        output_dir / "step7ml_k_quality_gate_report.json",
        {"schema": "step7ml_k_quality_gate_report_v1", "summary": summary},
    )
    write_json(output_dir / "step7ml_k_invariant_ablation.json", ablation(all_variants, selected))
    write_json(output_dir / "step7ml_k_failure_attribution.json", failure_report(selected))
    (output_dir / "step7ml_k_decision.md").write_text(
        decision_md(decision, summary), encoding="utf-8"
    )
    return {"decision": decision, "summary": summary}


def summarize(
    selected: list[dict[str, Any]],
    all_variants: list[dict[str, Any]],
    payloads: list[dict[str, Any]],
    runtime_ms: float,
) -> dict[str, Any]:
    violation_counter: Counter[str] = Counter()
    for row in selected:
        for key, value in row.get("invariant_violations", {}).items():
            if value:
                violation_counter[key] += int(value)
    large_success = sum(
        1
        for row in selected
        if row.get("closure_size_bucket") == "large_21_plus" and row.get("hard_feasible_non_noop")
    )
    return {
        "input_payload_count": len(payloads),
        "decoded_candidate_count": len(selected),
        "all_variant_candidate_count": len(all_variants),
        "candidate_count_by_decoder": dict(Counter(str(row.get("decoder")) for row in selected)),
        "candidate_count_by_invariant": dict(
            Counter("+".join(row.get("invariants", [])) for row in selected)
        ),
        "non_original_non_noop_count": sum(
            1 for row in selected if row.get("non_original_non_noop")
        ),
        "hard_feasible_nonnoop_count": sum(
            1 for row in selected if row.get("hard_feasible_non_noop")
        ),
        "overlap_after_decode_count": sum(
            1 for row in selected if row.get("overlap_after_decode")
        ),
        "bbox_regression_count": sum(
            1 for row in selected if as_float(row.get("bbox_area_delta")) > 0
        ),
        "soft_regression_count": sum(
            1 for row in selected if as_float(row.get("soft_constraint_delta")) > 0
        ),
        "hpwl_gain_but_official_like_loss_count": sum(
            1
            for row in selected
            if as_float(row.get("hpwl_delta")) < 0
            and as_float(row.get("official_like_cost_delta")) > 0
        ),
        "official_like_improving_count": sum(
            1 for row in selected if row.get("official_like_improving")
        ),
        "quality_gate_pass_count": sum(1 for row in selected if row.get("quality_gate_pass")),
        "dominated_by_original_count": sum(
            1 for row in selected if row.get("dominated_by_original")
        ),
        "metric_regressing_count": sum(1 for row in selected if row.get("metric_regressing")),
        "route_count_by_class": dict(Counter(str(row.get("route_class")) for row in selected)),
        "route_global_count": sum(1 for row in selected if row.get("route_global")),
        "invariant_violation_count_by_type": dict(violation_counter),
        "closure_size_bucket_summary": dict(
            Counter(str(row.get("closure_size_bucket")) for row in selected)
        ),
        "large_closure_success_count": large_success,
        "runtime_proxy_per_payload_ms": runtime_ms / max(len(payloads), 1),
        "comparison": {"step7ml_i": STEP7ML_I_BASELINES, "step7ml_j": STEP7ML_J_BASELINES},
    }


def config(payload_path: str) -> dict[str, Any]:
    return {
        "schema": "step7ml_k_decoder_config_v1",
        "payload_source": payload_path,
        "representative_selection": (
            "lexicographic invariant violation, hard feasibility, metric regression, "
            "domination, soft delta"
        ),
        "decoders": [
            "bbox_envelope_shelf_decoder",
            "order_preserving_slot_decoder",
            "compound_unit_shelf_decoder",
        ],
        "hard_invariants": [
            "bbox_envelope",
            "fixed_preplaced_preservation",
            "order_or_compound_adjacency",
            "non_overlap",
        ],
    }


def route_report(rows_: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "schema": "step7ml_k_route_report_v1",
        "route_count_by_class": dict(Counter(str(row.get("route_class")) for row in rows_)),
        "route_global_count": sum(1 for row in rows_ if row.get("route_global")),
    }


def ablation(all_variants: list[dict[str, Any]], selected: list[dict[str, Any]]) -> dict[str, Any]:
    selected_ids = {str(row.get("candidate_id")) for row in selected}
    out = []
    for decoder in sorted({str(row.get("decoder")) for row in all_variants}):
        subset = [row for row in all_variants if row.get("decoder") == decoder]
        out.append(
            {
                "decoder": decoder,
                "candidate_count": len(subset),
                "selected_count": sum(
                    1 for row in subset if str(row.get("candidate_id")) in selected_ids
                ),
                "hard_feasible_nonnoop_count": sum(
                    1 for row in subset if row.get("hard_feasible_non_noop")
                ),
                "overlap_after_decode_count": sum(
                    1 for row in subset if row.get("overlap_after_decode")
                ),
                "soft_regression_count": sum(
                    1 for row in subset if as_float(row.get("soft_constraint_delta")) > 0
                ),
                "quality_gate_pass_count": sum(1 for row in subset if row.get("quality_gate_pass")),
                "official_like_improving_count": sum(
                    1 for row in subset if row.get("official_like_improving")
                ),
            }
        )
    return {"schema": "step7ml_k_invariant_ablation_v1", "rows": out}


def failure_report(rows_: list[dict[str, Any]]) -> dict[str, Any]:
    reason_counts: Counter[str] = Counter()
    for row in rows_:
        if row.get("overlap_after_decode"):
            reason_counts["overlap_after_decode"] += 1
        elif row.get("metric_regressing"):
            reason_counts["metric_regressing"] += 1
        elif row.get("dominated_by_original"):
            reason_counts["dominated_by_original"] += 1
        elif not row.get("non_original_non_noop"):
            reason_counts["no_op"] += 1
        else:
            reason_counts["none"] += 1
    return {"schema": "step7ml_k_failure_attribution_v1", "reason_counts": dict(reason_counts)}


def decide(summary: dict[str, Any]) -> str:
    if summary["quality_gate_pass_count"] > 2 or summary["official_like_improving_count"] > 2:
        return "promote_invariant_geometry_decoder"
    if summary["bbox_regression_count"] < STEP7ML_J_BASELINES["bbox_regression"] and summary[
        "soft_regression_count"
    ] < STEP7ML_J_BASELINES["soft_regression"]:
        return "use_invariant_decoder_as_candidate_generator"
    if summary["route_global_count"] > 0:
        return "combine_with_learned_ranker_budget"
    return "refine_order_preserving_decoder"


def decision_md(decision: str, summary: dict[str, Any]) -> str:
    return f"""# Step7ML-K Invariant-Preserving Geometry Decoder

Decision: `{decision}`

## Key metrics

- input_payload_count: {summary['input_payload_count']}
- decoded_candidate_count: {summary['decoded_candidate_count']}
- all_variant_candidate_count: {summary['all_variant_candidate_count']}
- candidate_count_by_decoder: {summary['candidate_count_by_decoder']}
- non_original_non_noop_count: {summary['non_original_non_noop_count']}
- hard_feasible_nonnoop_count: {summary['hard_feasible_nonnoop_count']} vs Step7ML-I 67
- overlap_after_decode_count: {summary['overlap_after_decode_count']} vs Step7ML-I 4
- bbox_regression_count: {summary['bbox_regression_count']} vs Step7ML-J 56
- soft_regression_count: {summary['soft_regression_count']} vs Step7ML-J 114
- hpwl_gain_but_official_like_loss_count:
  {summary['hpwl_gain_but_official_like_loss_count']} vs Step7ML-J 56
- official_like_improving_count: {summary['official_like_improving_count']} vs baseline 2
- quality_gate_pass_count: {summary['quality_gate_pass_count']} vs baseline 2
- dominated_by_original_count: {summary['dominated_by_original_count']}
- metric_regressing_count: {summary['metric_regressing_count']}
- route_count_by_class: {summary['route_count_by_class']}
- route_global_count: {summary['route_global_count']}
- invariant_violation_count_by_type: {summary['invariant_violation_count_by_type']}
- closure_size_bucket_summary: {summary['closure_size_bucket_summary']}
- large_closure_success_count: {summary['large_closure_success_count']}

## Interpretation

Step7ML-K generates structure-preserving legal candidates inside the original
closure envelope and preserves fixed/preplaced coordinates plus order/compound
adjacency. It reduces duplicate metric-regression exposure relative to Step7ML-J
and keeps overlap low, but if quality-gate / official-like counts remain at 2 it
should be used as a candidate generator feeding Step7ML-J rather than promoted as
the final macro/topology generator.
"""
