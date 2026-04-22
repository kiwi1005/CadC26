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
from puzzleplace.feedback import load_policy_checkpoint
from puzzleplace.models import TypedActionPolicy
from puzzleplace.repair import finalize_layout
from puzzleplace.rollout import semantic_rollout
from puzzleplace.train import load_validation_cases


def _average(values: list[float]) -> float:
    return sum(values) / max(len(values), 1)


def _evaluate_variant(variant: str, policy, cases, seed: int | None) -> dict[str, object]:
    per_case = []
    for case in cases:
        started = time.time()
        semantic = semantic_rollout(case, policy)
        repair = finalize_layout(case, semantic.proposed_positions)
        official = evaluate_positions(case, repair.positions, runtime=max(time.time() - started, 1e-6))
        per_case.append(
            {
                "case_id": str(case.case_id),
                "seed": seed,
                "semantic_completed": semantic.semantic_completed,
                "semantic_placed_fraction": semantic.semantic_placed_fraction,
                "fallback_fraction": semantic.fallback_fraction,
                "repair_hard_feasible_after": repair.report.hard_feasible_after,
                "repair_overlap_pairs_after": repair.report.overlap_pairs_after,
                "mean_displacement": repair.report.mean_displacement,
                "max_displacement": repair.report.max_displacement,
                "preserved_x_order_fraction": repair.report.preserved_x_order_fraction,
                "official_cost": float(official["official"]["cost"]),
                "hpwl_gap": float(official["official"]["hpwl_gap"]),
                "area_gap": float(official["official"]["area_gap"]),
                "total_soft_violations": int(official["official"]["total_soft_violations"]),
                "violations_relative": float(official["official"]["violations_relative"]),
                "boundary_violations": int(official["official"]["boundary_violations"]),
                "grouping_violations": int(official["official"]["grouping_violations"]),
                "mib_violations": int(official["official"]["mib_violations"]),
            }
        )
    return {
        "variant": variant,
        "seed": seed,
        "case_count": len(cases),
        "semantic_rollout_completion": _average(
            [1.0 if row["semantic_completed"] else 0.0 for row in per_case]
        ),
        "avg_semantic_placed_fraction": _average(
            [float(row["semantic_placed_fraction"]) for row in per_case]
        ),
        "repair_success_rate": _average(
            [1.0 if row["repair_hard_feasible_after"] else 0.0 for row in per_case]
        ),
        "contest_feasible_cases": int(
            sum(1 for row in per_case if bool(row["repair_hard_feasible_after"]))
        ),
        "avg_fallback_fraction": _average([float(row["fallback_fraction"]) for row in per_case]),
        "avg_repair_overlap_pairs_after": _average(
            [float(row["repair_overlap_pairs_after"]) for row in per_case]
        ),
        "avg_mean_displacement": _average([float(row["mean_displacement"]) for row in per_case]),
        "avg_max_displacement": _average([float(row["max_displacement"]) for row in per_case]),
        "avg_preserved_x_order_fraction": _average(
            [float(row["preserved_x_order_fraction"]) for row in per_case]
        ),
        "avg_official_cost": _average([float(row["official_cost"]) for row in per_case]),
        "avg_hpwl_gap": _average([float(row["hpwl_gap"]) for row in per_case]),
        "avg_area_gap": _average([float(row["area_gap"]) for row in per_case]),
        "avg_total_soft_violations": _average([float(row["total_soft_violations"]) for row in per_case]),
        "avg_violations_relative": _average([float(row["violations_relative"]) for row in per_case]),
        "avg_boundary_violations": _average([float(row["boundary_violations"]) for row in per_case]),
        "avg_grouping_violations": _average([float(row["grouping_violations"]) for row in per_case]),
        "avg_mib_violations": _average([float(row["mib_violations"]) for row in per_case]),
        "cases": per_case,
    }


def main() -> None:
    research_dir = ROOT / "artifacts" / "research"
    case_count = 20
    cases = load_validation_cases(case_limit=case_count)
    rows = []

    rows.append(_evaluate_variant("heuristic", None, cases, None))

    for seed in (0, 1):
        torch.manual_seed(seed)
        rows.append(_evaluate_variant("untrained", TypedActionPolicy(hidden_dim=32), cases, seed))
        rows.append(
            _evaluate_variant(
                "bc",
                load_policy_checkpoint(ROOT / "artifacts" / "models" / f"small_overfit_bc_seed{seed}.pt"),
                cases,
                seed,
            )
        )
        rows.append(
            _evaluate_variant(
                "awbc",
                load_policy_checkpoint(ROOT / "artifacts" / "models" / f"small_overfit_awbc_seed{seed}.pt"),
                cases,
                seed,
            )
        )

    payload = {
        "evaluation_slice": f"validation-0-{case_count - 1}",
        "source_checkpoints": [
            "artifacts/models/small_overfit_bc_seed0.pt",
            "artifacts/models/small_overfit_bc_seed1.pt",
            "artifacts/models/small_overfit_awbc_seed0.pt",
            "artifacts/models/small_overfit_awbc_seed1.pt",
        ],
        "rows": rows,
    }
    (research_dir / "generalization_followup_smallcheckpoints.json").write_text(
        json.dumps(payload, indent=2)
    )

    lines = [
        "# Generalization Follow-up (small-overfit checkpoints)",
        "",
        f"- evaluation slice: `validation-0-{case_count - 1}`",
        "",
        "| Variant | Seed | Feasible cases | Mean official cost | Mean HPWL gap | Mean area gap | Mean soft violations | Mean displacement |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row['variant']} | {row['seed'] if row['seed'] is not None else '-'} | "
            f"{row['contest_feasible_cases']}/{row['case_count']} | "
            f"{row['avg_official_cost']:.3f} | {row['avg_hpwl_gap']:.3f} | {row['avg_area_gap']:.3f} | "
            f"{row['avg_total_soft_violations']:.3f} | {row['avg_mean_displacement']:.3f} |"
        )

    trained_rows = [row for row in rows if row["variant"] in {"bc", "awbc"}]
    untrained_rows = [row for row in rows if row["variant"] == "untrained"]
    best_trained_cost = min(float(row["avg_official_cost"]) for row in trained_rows)
    best_untrained_cost = min(float(row["avg_official_cost"]) for row in untrained_rows)
    lines.extend(
        [
            "",
            f"- best untrained mean official cost: `{best_untrained_cost:.3f}`",
            f"- best trained mean official cost: `{best_trained_cost:.3f}`",
            "- Use this artifact to judge whether the resolver fix generalizes into a downstream cost/displacement win beyond the small-overfit slice.",
            "- This version also records HPWL/area/soft-violation terms so cost decomposition does not need to guess from displacement alone.",
        ]
    )
    (research_dir / "generalization_followup_smallcheckpoints.md").write_text("\n".join(lines))
    print(json.dumps({"rows": len(rows), "case_count": case_count}, indent=2))


if __name__ == "__main__":
    main()
