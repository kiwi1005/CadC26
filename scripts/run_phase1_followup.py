#!/usr/bin/env python3
# ruff: noqa: E402
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for path in (ROOT, SRC):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from puzzleplace.eval.official import evaluate_positions
from puzzleplace.eval.violation import summarize_violation_profile
from puzzleplace.feedback import load_policy_checkpoint
from puzzleplace.models import TypedActionPolicy
from puzzleplace.data import ConstraintColumns
from puzzleplace.repair.intent_preserver import measure_intent_preservation
from puzzleplace.repair.overlap_resolver import resolve_overlaps
from puzzleplace.repair.shape_normalizer import normalize_shapes
from puzzleplace.repair.shelf_packer import shelf_pack_missing
from puzzleplace.rollout import semantic_rollout
from puzzleplace.train import load_validation_cases
from puzzleplace.geometry import summarize_hard_legality


def _average(values: list[float]) -> float:
    return sum(values) / max(len(values), 1)


def _ordered_positions(case, positions_dict: dict[int, tuple[float, float, float, float]]):
    return [positions_dict[idx] for idx in range(case.block_count)]


def _stage_row(
    *,
    variant: str,
    case,
    seed: int | None,
    stage: str,
    checkpoint_path: str | None,
    candidate_mode: str,
    semantic_completed: bool,
    semantic_placed_fraction: float,
    fallback_fraction: float,
    positions_dict: dict[int, tuple[float, float, float, float]],
    started_at: float,
) -> dict[str, object]:
    profile = summarize_violation_profile(case, positions_dict)
    ordered = _ordered_positions(case, positions_dict)
    hard = summarize_hard_legality(case, ordered)
    official = evaluate_positions(case, ordered, runtime=1.0, median_runtime=1.0)
    return {
        "variant": variant,
        "case_id": str(case.case_id),
        "seed": seed,
        "stage": stage,
        "checkpoint_path": checkpoint_path,
        "candidate_mode": candidate_mode,
        "semantic_completed": semantic_completed,
        "semantic_placed_fraction": semantic_placed_fraction,
        "fallback_fraction": fallback_fraction,
        "hard_feasible": hard.is_feasible,
        "overlap_pairs": profile.overlap_pairs,
        "total_overlap_area": profile.total_overlap_area,
        "area_violations": profile.area_violations,
        "dimension_violations": profile.dimension_violations,
        "boundary_violations": int(official["official"]["boundary_violations"]),
        "grouping_violations": int(official["official"]["grouping_violations"]),
        "mib_violations": int(official["official"]["mib_violations"]),
        "moved_block_count": 0,
        "mean_displacement": 0.0,
        "max_displacement": 0.0,
        "preserved_x_order_fraction": 1.0,
        "runtime_seconds": float(official["official"]["runtime_seconds"]),
        "wall_time_seconds": float(time.time() - started_at),
        "official_cost": float(official["official"]["cost"]),
    }


def _evaluate_variant(
    *,
    variant: str,
    policy,
    cases,
    seed: int | None,
    checkpoint_path: str | None,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    trace_rows: list[dict[str, object]] = []
    case_summaries = []
    for case in cases:
        started_at = time.time()
        semantic = semantic_rollout(case, policy)
        proposed = dict(semantic.proposed_positions)
        trace_rows.append(
            _stage_row(
                variant=variant,
                case=case,
                seed=seed,
                stage="semantic_output",
                checkpoint_path=checkpoint_path,
                candidate_mode="semantic",
                semantic_completed=semantic.semantic_completed,
                semantic_placed_fraction=semantic.semantic_placed_fraction,
                fallback_fraction=semantic.fallback_fraction,
                positions_dict=proposed,
                started_at=started_at,
            )
        )

        normalized = normalize_shapes(case, proposed)
        normalized_intent = measure_intent_preservation(proposed, normalized)
        row = _stage_row(
            variant=variant,
            case=case,
            seed=seed,
            stage="normalized",
            checkpoint_path=checkpoint_path,
            candidate_mode="semantic",
            semantic_completed=semantic.semantic_completed,
            semantic_placed_fraction=semantic.semantic_placed_fraction,
            fallback_fraction=semantic.fallback_fraction,
            positions_dict=normalized,
            started_at=started_at,
        )
        row["mean_displacement"] = float(normalized_intent["mean_displacement"])
        row["max_displacement"] = float(normalized_intent["max_displacement"])
        row["preserved_x_order_fraction"] = float(normalized_intent["preserved_x_order_fraction"])
        trace_rows.append(row)

        locked_blocks = {
            idx
            for idx in range(case.block_count)
            if bool(case.constraints[idx, ConstraintColumns.PREPLACED].item())
        }
        resolved_1, moved_1 = resolve_overlaps(normalized, locked_blocks=locked_blocks)
        row = _stage_row(
            variant=variant,
            case=case,
            seed=seed,
            stage="overlap_resolved_1",
            checkpoint_path=checkpoint_path,
            candidate_mode="semantic",
            semantic_completed=semantic.semantic_completed,
            semantic_placed_fraction=semantic.semantic_placed_fraction,
            fallback_fraction=semantic.fallback_fraction,
            positions_dict=resolved_1,
            started_at=started_at,
        )
        resolved_1_intent = measure_intent_preservation(proposed, resolved_1)
        row["moved_block_count"] = len(moved_1)
        row["mean_displacement"] = float(resolved_1_intent["mean_displacement"])
        row["max_displacement"] = float(resolved_1_intent["max_displacement"])
        row["preserved_x_order_fraction"] = float(resolved_1_intent["preserved_x_order_fraction"])
        trace_rows.append(row)

        missing_blocks = [idx for idx in range(case.block_count) if idx not in resolved_1]
        packed, shelf_count = shelf_pack_missing(case, resolved_1, missing_blocks)
        row = _stage_row(
            variant=variant,
            case=case,
            seed=seed,
            stage="shelf_packed",
            checkpoint_path=checkpoint_path,
            candidate_mode="semantic",
            semantic_completed=semantic.semantic_completed,
            semantic_placed_fraction=semantic.semantic_placed_fraction,
            fallback_fraction=semantic.fallback_fraction,
            positions_dict=packed,
            started_at=started_at,
        )
        packed_intent = measure_intent_preservation(proposed, packed)
        row["moved_block_count"] = shelf_count
        row["mean_displacement"] = float(packed_intent["mean_displacement"])
        row["max_displacement"] = float(packed_intent["max_displacement"])
        row["preserved_x_order_fraction"] = float(packed_intent["preserved_x_order_fraction"])
        trace_rows.append(row)

        repaired, moved_2 = resolve_overlaps(packed, locked_blocks=locked_blocks)
        row = _stage_row(
            variant=variant,
            case=case,
            seed=seed,
            stage="overlap_resolved_2",
            checkpoint_path=checkpoint_path,
            candidate_mode="semantic",
            semantic_completed=semantic.semantic_completed,
            semantic_placed_fraction=semantic.semantic_placed_fraction,
            fallback_fraction=semantic.fallback_fraction,
            positions_dict=repaired,
            started_at=started_at,
        )
        repaired_intent = measure_intent_preservation(proposed, repaired)
        row["moved_block_count"] = len(moved_1 | moved_2)
        row["mean_displacement"] = float(repaired_intent["mean_displacement"])
        row["max_displacement"] = float(repaired_intent["max_displacement"])
        row["preserved_x_order_fraction"] = float(repaired_intent["preserved_x_order_fraction"])
        trace_rows.append(row)

        final_row = dict(row)
        final_row["stage"] = "strict_final"
        trace_rows.append(final_row)

        case_summaries.append(
            {
                "case_id": str(case.case_id),
                "seed": seed,
                "semantic_rollout_completion": 1.0 if semantic.semantic_completed else 0.0,
                "avg_semantic_placed_fraction": semantic.semantic_placed_fraction,
                "repair_success_rate": 1.0 if final_row["hard_feasible"] else 0.0,
                "contest_feasible": 1.0 if final_row["hard_feasible"] else 0.0,
                "fallback_fraction": semantic.fallback_fraction,
                "strict_candidate_coverage": 1.0 if final_row["hard_feasible"] else 0.0,
                "official_cost": final_row["official_cost"],
                "overlap_failure_count": final_row["overlap_pairs"],
                "boundary_violations": final_row["boundary_violations"],
                "grouping_violations": final_row["grouping_violations"],
                "mib_violations": final_row["mib_violations"],
                "runtime_seconds": final_row["runtime_seconds"],
            }
        )

    aggregate = {
        "variant": variant,
        "seed": seed,
        "checkpoint_path": checkpoint_path,
        "semantic_candidate_coverage": 1.0,
        "strict_candidate_coverage": _average(
            [float(item["strict_candidate_coverage"]) for item in case_summaries]
        ),
        "semantic_rollout_completion": _average(
            [float(item["semantic_rollout_completion"]) for item in case_summaries]
        ),
        "avg_semantic_placed_fraction": _average(
            [float(item["avg_semantic_placed_fraction"]) for item in case_summaries]
        ),
        "repair_success_rate": _average([float(item["repair_success_rate"]) for item in case_summaries]),
        "contest_feasible_cases": int(sum(float(item["contest_feasible"]) for item in case_summaries)),
        "fallback_fraction": _average([float(item["fallback_fraction"]) for item in case_summaries]),
        "official_cost": _average([float(item["official_cost"]) for item in case_summaries]),
        "overlap_failure_count": _average(
            [float(item["overlap_failure_count"]) for item in case_summaries]
        ),
        "boundary_violations": _average([float(item["boundary_violations"]) for item in case_summaries]),
        "grouping_violations": _average([float(item["grouping_violations"]) for item in case_summaries]),
        "mib_violations": _average([float(item["mib_violations"]) for item in case_summaries]),
        "runtime_seconds": _average([float(item["runtime_seconds"]) for item in case_summaries]),
        "cases": case_summaries,
    }
    return aggregate, trace_rows


def main() -> None:
    research_dir = ROOT / "artifacts" / "research"
    small_bc = json.loads((research_dir / "small_overfit_bc.json").read_text())
    small_awbc = json.loads((research_dir / "small_overfit_awbc.json").read_text())

    case_limit = int(small_bc["val_case_count"])
    cases = load_validation_cases(case_limit=case_limit)

    all_rows = []
    trace_rows: list[dict[str, object]] = []
    heuristic_row, heuristic_trace = _evaluate_variant(
        variant="heuristic",
        policy=None,
        cases=cases,
        seed=None,
        checkpoint_path=None,
    )
    all_rows.append(heuristic_row)
    trace_rows.extend(heuristic_trace)

    for run in small_bc["runs"]:
        seed = int(run["seed"])
        torch.manual_seed(seed)
        untrained_policy = TypedActionPolicy(hidden_dim=int(small_bc["hidden_dim"]))
        untrained_row, untrained_trace = _evaluate_variant(
            variant="untrained",
            policy=untrained_policy,
            cases=cases,
            seed=seed,
            checkpoint_path=None,
        )
        all_rows.append(untrained_row)
        trace_rows.extend(untrained_trace)

        bc_checkpoint = ROOT / run["trained"]["checkpoint_path"]
        bc_policy = load_policy_checkpoint(bc_checkpoint)
        bc_row, bc_trace = _evaluate_variant(
            variant="bc",
            policy=bc_policy,
            cases=cases,
            seed=seed,
            checkpoint_path=str(bc_checkpoint.relative_to(ROOT)),
        )
        all_rows.append(bc_row)
        trace_rows.extend(bc_trace)

    for run in small_awbc["runs"]:
        seed = int(run["seed"])
        awbc_checkpoint = ROOT / run["trained"]["checkpoint_path"]
        awbc_policy = load_policy_checkpoint(awbc_checkpoint)
        awbc_row, awbc_trace = _evaluate_variant(
            variant="awbc",
            policy=awbc_policy,
            cases=cases,
            seed=seed,
            checkpoint_path=str(awbc_checkpoint.relative_to(ROOT)),
        )
        all_rows.append(awbc_row)
        trace_rows.extend(awbc_trace)

    semantic_payload = {
        "source_artifacts": [
            "artifacts/research/small_overfit_bc.json",
            "artifacts/research/small_overfit_awbc.json",
        ],
        "rows": all_rows,
    }
    (research_dir / "semantic_training_ablation.json").write_text(
        json.dumps(semantic_payload, indent=2)
    )
    summary_lines = [
        "# Semantic Training Ablation",
        "",
        "| Variant | Seed | Strict cov | Rollout completion | Avg placed | Repair success | Fallback | Official cost |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in all_rows:
        summary_lines.append(
            f"| {row['variant']} | {row['seed'] if row['seed'] is not None else '-'} | "
            f"{row['strict_candidate_coverage']:.3f} | {row['semantic_rollout_completion']:.3f} | "
            f"{row['avg_semantic_placed_fraction']:.3f} | {row['repair_success_rate']:.3f} | "
            f"{row['fallback_fraction']:.3f} | {row['official_cost']:.3f} |"
        )

    bc_rows = [row for row in all_rows if row["variant"] == "bc"]
    awbc_rows = [row for row in all_rows if row["variant"] == "awbc"]
    untrained_rows = [row for row in all_rows if row["variant"] == "untrained"]
    heuristic_rows = [row for row in all_rows if row["variant"] == "heuristic"]
    summary_lines.extend(
        [
            "",
            "Contest-feasible counts (final stage):",
            *(f"- {row['variant']} seed {row['seed'] if row['seed'] is not None else '-'}: `{row['contest_feasible_cases']}/{case_limit}`" for row in all_rows),
            "",
            "Gate interpretation:",
            f"- Proposal gate signal: BC strict/untrained rollout-level completion is unchanged, but Phase 1 JSON shows BC/AWBC val block accuracy improved over untrained.",
            f"- Survival gate signal: mean repair success is now `{_average([float(row['repair_success_rate']) for row in heuristic_rows]):.3f}` for heuristic and "
            f"`{_average([float(row['repair_success_rate']) for row in untrained_rows]):.3f}`/`{_average([float(row['repair_success_rate']) for row in bc_rows]):.3f}`/`{_average([float(row['repair_success_rate']) for row in awbc_rows]):.3f}` for untrained/BC/AWBC on this slice.",
            "- Evidence classification: the prior repair bottleneck is removed on this slice, but downstream feasibility is now tied across variants, so this is still not a trained-variant T3 win.",
            "- Interpretation: proposal-level learning signal exists; after the resolver fix, downstream feasibility no longer separates variants on this slice, leaving cost/displacement as the remaining differentiators.",
        ]
    )
    (research_dir / "semantic_training_ablation.md").write_text("\n".join(summary_lines))

    trace_payload = {
        "source_artifacts": [
            "artifacts/research/small_overfit_bc.json",
            "artifacts/research/small_overfit_awbc.json",
        ],
        "rows": trace_rows,
    }
    (research_dir / "finalizer_interaction_trace.json").write_text(
        json.dumps(trace_payload, indent=2)
    )

    stage_means: dict[tuple[str, str], list[float]] = {}
    stage_feasible: dict[tuple[str, str], list[float]] = {}
    for row in trace_rows:
        key = (str(row["variant"]), str(row["stage"]))
        stage_means.setdefault(key, []).append(float(row["official_cost"]))
        stage_feasible.setdefault(key, []).append(1.0 if bool(row["hard_feasible"]) else 0.0)
    lines = [
        "# Finalizer Interaction Summary",
        "",
        "| Variant | Stage | Mean official cost | Hard-feasible rate |",
        "| --- | --- | ---: | ---: |",
    ]
    for key in sorted(stage_means):
        lines.append(
            f"| {key[0]} | {key[1]} | {_average(stage_means[key]):.3f} | "
            f"{_average(stage_feasible[key]):.3f} |"
        )
    sentinel_rows = [
        row for row in trace_rows if row["case_id"] == "validation-1" and row["stage"] == "strict_final"
    ]
    lines.extend(
        [
            "",
            "Sentinel case (`validation-1`) strict-final results:",
            *(
                f"- {row['variant']} seed {row['seed'] if row['seed'] is not None else '-'}: "
                f"hard_feasible=`{row['hard_feasible']}`, overlap_pairs=`{row['overlap_pairs']}`, "
                f"boundary_violations=`{row['boundary_violations']}`, official_cost=`{row['official_cost']:.3f}`"
                for row in sentinel_rows
            ),
            "",
            "Conclusion bucket:",
            "- **D candidate**: training metrics improve upstream, but downstream repair success remains flat across heuristic/untrained/BC/AWBC on the current small-overfit slice.",
            "- Stage official cost is runtime-standardized at `1.0s` for like-for-like comparison in this summary.",
            "- Follow-up: move to explicit gate interpretation and targeted finalizer-interface analysis before any medium-scale training expansion.",
        ]
    )
    (research_dir / "finalizer_interaction_summary.md").write_text("\n".join(lines))
    print(json.dumps({"ablation_rows": len(all_rows), "trace_rows": len(trace_rows)}, indent=2))


if __name__ == "__main__":
    main()
