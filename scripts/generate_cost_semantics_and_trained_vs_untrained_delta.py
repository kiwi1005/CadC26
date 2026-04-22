#!/usr/bin/env python3
from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
RESEARCH_DIR = ROOT / "artifacts" / "research"
SOURCE_ARTIFACT = RESEARCH_DIR / "generalization_followup_smallcheckpoints.json"
EVALUATOR_PATH = ROOT / "external" / "FloorSet" / "iccad2026contest" / "iccad2026_evaluate.py"
OUTPUT_MD = RESEARCH_DIR / "cost_semantics_and_trained_vs_untrained_delta.md"
OUTPUT_JSON = RESEARCH_DIR / "cost_semantics_and_trained_vs_untrained_delta.json"

COMMANDS_USED = [
    "sed -n '300,380p' external/FloorSet/iccad2026contest/iccad2026_evaluate.py",
    "sed -n '380,540p' external/FloorSet/iccad2026contest/iccad2026_evaluate.py",
    "sed -n '1,260p' scripts/run_generalization_followup.py",
    ".venv/bin/python scripts/generate_cost_semantics_and_trained_vs_untrained_delta.py",
]


def _average(values: list[float]) -> float:
    return sum(values) / max(len(values), 1)


def _pearson(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) != len(ys) or len(xs) < 2:
        return None
    mean_x = _average(xs)
    mean_y = _average(ys)
    num = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    den_x = sum((x - mean_x) ** 2 for x in xs)
    den_y = sum((y - mean_y) ** 2 for y in ys)
    den = math.sqrt(den_x * den_y)
    if den == 0:
        return None
    return num / den


def _read_constants() -> dict[str, float]:
    text = EVALUATOR_PATH.read_text()
    values: dict[str, float] = {}
    for name in ("ALPHA", "BETA", "GAMMA", "M_PENALTY"):
        match = re.search(rf"^{name}\s*=\s*([0-9.]+)", text, flags=re.MULTILINE)
        if not match:
            raise ValueError(f"missing evaluator constant: {name}")
        values[name] = float(match.group(1))
    return values


def _quality_factor(alpha: float, hpwl_gap: float, area_gap: float) -> float:
    return 1.0 + alpha * (max(0.0, hpwl_gap) + max(0.0, area_gap))


def _violation_factor(beta: float, violations_relative: float) -> float:
    return math.exp(beta * violations_relative)


def _runtime_adjustment(gamma: float, runtime_factor: float) -> float:
    return max(0.7, math.pow(max(0.01, runtime_factor), gamma))


def _official_cost(
    alpha: float,
    beta: float,
    gamma: float,
    hpwl_gap: float,
    area_gap: float,
    violations_relative: float,
    runtime_factor: float,
    is_feasible: bool,
    m_penalty: float,
) -> float:
    if not is_feasible:
        return m_penalty
    return (
        _quality_factor(alpha, hpwl_gap, area_gap)
        * _violation_factor(beta, violations_relative)
        * _runtime_adjustment(gamma, runtime_factor)
    )


def _infer_runtime_terms(
    alpha: float,
    beta: float,
    gamma: float,
    case: dict[str, Any],
) -> dict[str, float | None]:
    quality = _quality_factor(alpha, float(case["hpwl_gap"]), float(case["area_gap"]))
    violation = _violation_factor(beta, float(case["violations_relative"]))
    runtime_adjustment = float(case["official_cost"]) / max(quality * violation, 1e-12)
    runtime_floor = 0.7
    if runtime_adjustment <= runtime_floor + 1e-12:
        return {
            "quality_factor": quality,
            "violation_factor": violation,
            "runtime_adjustment": runtime_adjustment,
            "runtime_factor": None,
            "runtime_factor_upper_bound": math.pow(runtime_floor, 1.0 / gamma),
        }
    return {
        "quality_factor": quality,
        "violation_factor": violation,
        "runtime_adjustment": runtime_adjustment,
        "runtime_factor": math.pow(runtime_adjustment, 1.0 / gamma),
        "runtime_factor_upper_bound": None,
    }


def _format_runtime(md_value: float | None, upper_bound: float | None) -> str:
    if md_value is not None:
        return f"{md_value:.3f}"
    return f"<= {upper_bound:.3f}"


def main() -> None:
    payload = json.loads(SOURCE_ARTIFACT.read_text())
    rows = payload["rows"]
    constants = _read_constants()
    alpha = constants["ALPHA"]
    beta = constants["BETA"]
    gamma = constants["GAMMA"]
    m_penalty = constants["M_PENALTY"]

    best_untrained = min(
        (row for row in rows if row["variant"] == "untrained"),
        key=lambda row: float(row["avg_official_cost"]),
    )
    best_trained = min(
        (row for row in rows if row["variant"] in {"bc", "awbc"}),
        key=lambda row: float(row["avg_official_cost"]),
    )

    paired_cases: list[dict[str, Any]] = []
    trained_wins = 0
    trained_losses = 0
    for untrained_case, trained_case in zip(best_untrained["cases"], best_trained["cases"]):
        untrained_runtime = _infer_runtime_terms(alpha, beta, gamma, untrained_case)
        trained_runtime = _infer_runtime_terms(alpha, beta, gamma, trained_case)
        delta_cost = float(trained_case["official_cost"]) - float(untrained_case["official_cost"])
        if delta_cost < 0:
            trained_wins += 1
        elif delta_cost > 0:
            trained_losses += 1
        paired_cases.append(
            {
                "case_id": str(untrained_case["case_id"]),
                "untrained": {
                    "variant": best_untrained["variant"],
                    "seed": best_untrained["seed"],
                    "official_cost": float(untrained_case["official_cost"]),
                    "hpwl_gap": float(untrained_case["hpwl_gap"]),
                    "area_gap": float(untrained_case["area_gap"]),
                    "violations_relative": float(untrained_case["violations_relative"]),
                    "quality_factor": float(untrained_runtime["quality_factor"]),
                    "violation_factor": float(untrained_runtime["violation_factor"]),
                    "runtime_adjustment": float(untrained_runtime["runtime_adjustment"]),
                    "runtime_factor": untrained_runtime["runtime_factor"],
                    "runtime_factor_upper_bound": untrained_runtime["runtime_factor_upper_bound"],
                    "mean_displacement": float(untrained_case["mean_displacement"]),
                },
                "trained": {
                    "variant": best_trained["variant"],
                    "seed": best_trained["seed"],
                    "official_cost": float(trained_case["official_cost"]),
                    "hpwl_gap": float(trained_case["hpwl_gap"]),
                    "area_gap": float(trained_case["area_gap"]),
                    "violations_relative": float(trained_case["violations_relative"]),
                    "quality_factor": float(trained_runtime["quality_factor"]),
                    "violation_factor": float(trained_runtime["violation_factor"]),
                    "runtime_adjustment": float(trained_runtime["runtime_adjustment"]),
                    "runtime_factor": trained_runtime["runtime_factor"],
                    "runtime_factor_upper_bound": trained_runtime["runtime_factor_upper_bound"],
                    "mean_displacement": float(trained_case["mean_displacement"]),
                },
                "delta": {
                    "official_cost": delta_cost,
                    "hpwl_gap": float(trained_case["hpwl_gap"]) - float(untrained_case["hpwl_gap"]),
                    "area_gap": float(trained_case["area_gap"]) - float(untrained_case["area_gap"]),
                    "violations_relative": float(trained_case["violations_relative"]) - float(untrained_case["violations_relative"]),
                    "runtime_factor": (
                        None
                        if untrained_runtime["runtime_factor"] is None
                        or trained_runtime["runtime_factor"] is None
                        else float(trained_runtime["runtime_factor"]) - float(untrained_runtime["runtime_factor"])
                    ),
                    "mean_displacement": float(trained_case["mean_displacement"]) - float(untrained_case["mean_displacement"]),
                },
            }
        )

    feasible_costs_above_penalty = sum(
        1
        for pair in paired_cases
        for item in (pair["untrained"], pair["trained"])
        if float(item["official_cost"]) > m_penalty
    )

    def counterfactual_cost(
        hpwl_gap: float,
        area_gap: float,
        violations_relative: float,
        runtime_factor: float,
    ) -> float:
        return _official_cost(
            alpha,
            beta,
            gamma,
            hpwl_gap,
            area_gap,
            violations_relative,
            runtime_factor,
            True,
            m_penalty,
        )

    trained_mean_cost = float(best_trained["avg_official_cost"])
    untrained_mean_cost = float(best_untrained["avg_official_cost"])
    pair_gap = trained_mean_cost - untrained_mean_cost

    trained_quality_swapped_costs = []
    trained_quality_and_viol_swapped_costs = []
    trained_all_swapped_costs = []
    single_term_gains = {"hpwl": [], "area": [], "violations_relative": [], "runtime": []}

    for pair in paired_cases:
        u = pair["untrained"]
        t = pair["trained"]
        if t["runtime_factor"] is None or u["runtime_factor"] is None:
            raise ValueError("expected exact runtime_factor for this artifact")
        trained_quality_swapped_costs.append(
            counterfactual_cost(
                float(u["hpwl_gap"]),
                float(u["area_gap"]),
                float(t["violations_relative"]),
                float(t["runtime_factor"]),
            )
        )
        trained_quality_and_viol_swapped_costs.append(
            counterfactual_cost(
                float(u["hpwl_gap"]),
                float(u["area_gap"]),
                float(u["violations_relative"]),
                float(t["runtime_factor"]),
            )
        )
        trained_all_swapped_costs.append(
            counterfactual_cost(
                float(u["hpwl_gap"]),
                float(u["area_gap"]),
                float(u["violations_relative"]),
                float(u["runtime_factor"]),
            )
        )
        base = counterfactual_cost(
            float(t["hpwl_gap"]),
            float(t["area_gap"]),
            float(t["violations_relative"]),
            float(t["runtime_factor"]),
        )
        single_term_gains["hpwl"].append(
            base
            - counterfactual_cost(
                float(u["hpwl_gap"]),
                float(t["area_gap"]),
                float(t["violations_relative"]),
                float(t["runtime_factor"]),
            )
        )
        single_term_gains["area"].append(
            base
            - counterfactual_cost(
                float(t["hpwl_gap"]),
                float(u["area_gap"]),
                float(t["violations_relative"]),
                float(t["runtime_factor"]),
            )
        )
        single_term_gains["violations_relative"].append(
            base
            - counterfactual_cost(
                float(t["hpwl_gap"]),
                float(t["area_gap"]),
                float(u["violations_relative"]),
                float(t["runtime_factor"]),
            )
        )
        single_term_gains["runtime"].append(
            base
            - counterfactual_cost(
                float(t["hpwl_gap"]),
                float(t["area_gap"]),
                float(t["violations_relative"]),
                float(u["runtime_factor"]),
            )
        )

    mean_quality_swapped_cost = _average(trained_quality_swapped_costs)
    mean_quality_and_viol_swapped_cost = _average(trained_quality_and_viol_swapped_costs)
    mean_all_swapped_cost = _average(trained_all_swapped_costs)

    top_5_trained_losses = sorted(
        (pair for pair in paired_cases if float(pair["delta"]["official_cost"]) > 0),
        key=lambda pair: float(pair["delta"]["official_cost"]),
        reverse=True,
    )[:5]

    mean_deltas = {
        "official_cost": _average([float(pair["delta"]["official_cost"]) for pair in paired_cases]),
        "hpwl_gap": _average([float(pair["delta"]["hpwl_gap"]) for pair in paired_cases]),
        "area_gap": _average([float(pair["delta"]["area_gap"]) for pair in paired_cases]),
        "violations_relative": _average(
            [float(pair["delta"]["violations_relative"]) for pair in paired_cases]
        ),
        "runtime_factor": _average(
            [float(pair["delta"]["runtime_factor"]) for pair in paired_cases if pair["delta"]["runtime_factor"] is not None]
        ),
        "mean_displacement": _average([float(pair["delta"]["mean_displacement"]) for pair in paired_cases]),
    }

    all_points = [
        case
        for row in rows
        for case in row["cases"]
    ]
    pair_points = [point for pair in paired_cases for point in (pair["untrained"], pair["trained"])]
    delta_points = [pair["delta"] for pair in paired_cases]

    all_corr = {
        "displacement_vs_official_cost": _pearson(
            [float(case["mean_displacement"]) for case in all_points],
            [float(case["official_cost"]) for case in all_points],
        ),
        "displacement_vs_hpwl_gap": _pearson(
            [float(case["mean_displacement"]) for case in all_points],
            [float(case["hpwl_gap"]) for case in all_points],
        ),
        "displacement_vs_area_gap": _pearson(
            [float(case["mean_displacement"]) for case in all_points],
            [float(case["area_gap"]) for case in all_points],
        ),
    }
    pair_corr = {
        "displacement_vs_official_cost": _pearson(
            [float(case["mean_displacement"]) for case in pair_points],
            [float(case["official_cost"]) for case in pair_points],
        ),
        "displacement_vs_hpwl_gap": _pearson(
            [float(case["mean_displacement"]) for case in pair_points],
            [float(case["hpwl_gap"]) for case in pair_points],
        ),
        "displacement_vs_area_gap": _pearson(
            [float(case["mean_displacement"]) for case in pair_points],
            [float(case["area_gap"]) for case in pair_points],
        ),
    }
    delta_corr = {
        "delta_displacement_vs_delta_official_cost": _pearson(
            [float(case["mean_displacement"]) for case in delta_points],
            [float(case["official_cost"]) for case in delta_points],
        ),
        "delta_displacement_vs_delta_hpwl_gap": _pearson(
            [float(case["mean_displacement"]) for case in delta_points],
            [float(case["hpwl_gap"]) for case in delta_points],
        ),
        "delta_displacement_vs_delta_area_gap": _pearson(
            [float(case["mean_displacement"]) for case in delta_points],
            [float(case["area_gap"]) for case in delta_points],
        ),
    }

    lower_displacement_cases = [
        pair for pair in paired_cases if float(pair["delta"]["mean_displacement"]) < 0
    ]
    lower_displacement_and_lower_cost = [
        pair
        for pair in lower_displacement_cases
        if float(pair["delta"]["official_cost"]) < 0
    ]
    lower_displacement_but_higher_cost = [
        pair
        for pair in lower_displacement_cases
        if float(pair["delta"]["official_cost"]) > 0
    ]

    quality_gap_contribution = trained_mean_cost - mean_quality_swapped_cost
    soft_gap_contribution = mean_quality_swapped_cost - mean_quality_and_viol_swapped_cost
    runtime_gap_contribution = mean_quality_and_viol_swapped_cost - mean_all_swapped_cost

    runtime_mean_gap = _average(
        [float(pair["trained"]["runtime_adjustment"]) - float(pair["untrained"]["runtime_adjustment"]) for pair in paired_cases]
    )

    next_recommendation = (
        "Run a top-5 loss drift audit (`validation-14/18/17/15/11`) that logs "
        "`semantic proposal -> repaired layout -> strict-final` HPWL and bbox-area deltas under "
        "runtime-standardized evaluation (`runtime=1.0`). That will separate proposal-quality loss "
        "from repair-induced quality loss; runtime is not the main issue here."
    )

    result = {
        "source_artifact": str(SOURCE_ARTIFACT.relative_to(ROOT)),
        "output_files": [
            str(OUTPUT_MD.relative_to(ROOT)),
            str(OUTPUT_JSON.relative_to(ROOT)),
        ],
        "official_evaluator": {
            "source_file": str(EVALUATOR_PATH.relative_to(ROOT)),
            "formula": "cost = (1 + alpha * (max(0, hpwl_gap) + max(0, area_gap))) * exp(beta * violations_relative) * max(0.7, max(0.01, runtime_factor) ** gamma); infeasible => M_penalty",
            "constants": {
                "alpha": alpha,
                "beta": beta,
                "gamma": gamma,
                "m_penalty": m_penalty,
            },
            "feasibility_explanation": {
                "hard_constraints_only": [
                    "no overlaps",
                    "soft-block areas within 1% tolerance",
                    "fixed-shape and preplaced dimensions/locations must match",
                ],
                "soft_constraints_scored_but_not_feasibility_blocking": [
                    "boundary",
                    "grouping",
                    "mib",
                ],
            },
            "feasible_cost_can_exceed_m_penalty": True,
            "feasible_costs_above_m_penalty_count_for_best_pair": feasible_costs_above_penalty,
            "best_untrained": {
                "variant": best_untrained["variant"],
                "seed": best_untrained["seed"],
                "mean_official_cost": float(best_untrained["avg_official_cost"]),
                "mean_hpwl_gap": float(best_untrained["avg_hpwl_gap"]),
                "mean_area_gap": float(best_untrained["avg_area_gap"]),
                "mean_violations_relative": float(best_untrained["avg_violations_relative"]),
                "mean_runtime_factor": _average(
                    [float(pair["untrained"]["runtime_factor"]) for pair in paired_cases if pair["untrained"]["runtime_factor"] is not None]
                ),
                "mean_runtime_adjustment": _average(
                    [float(pair["untrained"]["runtime_adjustment"]) for pair in paired_cases]
                ),
            },
            "best_trained": {
                "variant": best_trained["variant"],
                "seed": best_trained["seed"],
                "mean_official_cost": float(best_trained["avg_official_cost"]),
                "mean_hpwl_gap": float(best_trained["avg_hpwl_gap"]),
                "mean_area_gap": float(best_trained["avg_area_gap"]),
                "mean_violations_relative": float(best_trained["avg_violations_relative"]),
                "mean_runtime_factor": _average(
                    [float(pair["trained"]["runtime_factor"]) for pair in paired_cases if pair["trained"]["runtime_factor"] is not None]
                ),
                "mean_runtime_adjustment": _average(
                    [float(pair["trained"]["runtime_adjustment"]) for pair in paired_cases]
                ),
            },
        },
        "paired_comparison": {
            "best_untrained_ref": {"variant": best_untrained["variant"], "seed": best_untrained["seed"]},
            "best_trained_ref": {"variant": best_trained["variant"], "seed": best_trained["seed"]},
            "case_count": len(paired_cases),
            "trained_better_case_count": trained_wins,
            "trained_worse_case_count": trained_losses,
            "trained_win_case_ids": [
                pair["case_id"] for pair in paired_cases if float(pair["delta"]["official_cost"]) < 0
            ],
            "mean_deltas_trained_minus_untrained": mean_deltas,
            "driver_attribution": {
                "trained_mean_cost": trained_mean_cost,
                "untrained_mean_cost": untrained_mean_cost,
                "mean_cost_if_trained_kept_own_soft_runtime_but_used_untrained_hpwl_and_area": mean_quality_swapped_cost,
                "mean_cost_if_trained_also_used_untrained_soft_violations": mean_quality_and_viol_swapped_cost,
                "mean_cost_if_trained_used_all_untrained_terms": mean_all_swapped_cost,
                "gap_breakdown": {
                    "hpwl_and_area_quality_terms": {
                        "absolute": quality_gap_contribution,
                        "fraction_of_total_gap": quality_gap_contribution / pair_gap,
                    },
                    "soft_violations": {
                        "absolute": soft_gap_contribution,
                        "fraction_of_total_gap": soft_gap_contribution / pair_gap,
                    },
                    "runtime": {
                        "absolute": runtime_gap_contribution,
                        "fraction_of_total_gap": runtime_gap_contribution / pair_gap,
                    },
                },
                "single_term_mean_gain_if_swapped_to_untrained": {
                    key: _average(values) for key, values in single_term_gains.items()
                },
                "runtime_adjustment_mean_gap_trained_minus_untrained": runtime_mean_gap,
            },
            "top_5_trained_losses": top_5_trained_losses,
            "per_case": paired_cases,
        },
        "displacement_analysis": {
            "pearson_correlation": {
                "all_variant_case_points_140": all_corr,
                "best_pair_points_40": pair_corr,
                "best_pair_deltas_20_trained_minus_untrained": delta_corr,
            },
            "lower_displacement_check_on_best_pair": {
                "trained_lower_displacement_case_count": len(lower_displacement_cases),
                "trained_lower_displacement_and_lower_cost_count": len(lower_displacement_and_lower_cost),
                "trained_lower_displacement_but_higher_cost_count": len(lower_displacement_but_higher_cost),
                "case_ids_with_lower_displacement_but_higher_cost": [
                    pair["case_id"] for pair in lower_displacement_but_higher_cost
                ],
            },
            "judgement": (
                "repair displacement is a weak-to-misleading proxy here: across all 140 artifact points it is negatively correlated "
                "with official cost/HPWL/area, and in the best trained-vs-untrained pair the trained checkpoint lowers displacement "
                "on 7 cases but still loses cost on 6 of those 7."
            ),
        },
        "commands_used": COMMANDS_USED,
        "next_diagnostic_recommendation": next_recommendation,
    }

    lines = [
        "# Cost Semantics and Trained vs Untrained Delta",
        "",
        f"- source artifact: `{result['source_artifact']}`",
        f"- best untrained: `{best_untrained['variant']} seed {best_untrained['seed']}` mean cost `{best_untrained['avg_official_cost']:.3f}`",
        f"- best trained: `{best_trained['variant']} seed {best_trained['seed']}` mean cost `{best_trained['avg_official_cost']:.3f}`",
        "",
        "## 1. Why 20/20 feasible but mean official cost is still 19.784 / 23.651?",
        f"- Official evaluator formula: `cost = (1 + {alpha:.1f}*(HPWLgap + Areagap_bbox)) * exp({beta:.1f}*Violationsrelative) * max(0.7, RuntimeFactor^{gamma:.1f})`; infeasible only short-circuits to `M = {m_penalty:.1f}`.",
        "- Feasibility is only a hard-constraint check: no overlap, soft-block area within 1%, fixed/preplaced dimensions or locations exact. Boundary/grouping/MIB are soft and still accumulate cost on feasible layouts.",
        f"- Yes, feasible cost can exceed `M=10`: all `{feasible_costs_above_penalty}` best-pair case-runs are feasible and still have `cost > 10`.",
        f"- Mean runtime adjustment is almost unchanged (`{result['official_evaluator']['best_untrained']['mean_runtime_adjustment']:.3f}` vs `{result['official_evaluator']['best_trained']['mean_runtime_adjustment']:.3f}`), so the high costs come from quality + soft-violation terms, not from feasibility failure.",
        "",
        "| Case | U cost | U hpwl | U area | U viol | U rt | T cost | T hpwl | T area | T viol | T rt | Δcost |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]

    for pair in paired_cases:
        u = pair["untrained"]
        t = pair["trained"]
        lines.append(
            f"| {pair['case_id']} | "
            f"{u['official_cost']:.3f} | {u['hpwl_gap']:.3f} | {u['area_gap']:.3f} | {u['violations_relative']:.3f} | {_format_runtime(u['runtime_factor'], u['runtime_factor_upper_bound'])} | "
            f"{t['official_cost']:.3f} | {t['hpwl_gap']:.3f} | {t['area_gap']:.3f} | {t['violations_relative']:.3f} | {_format_runtime(t['runtime_factor'], t['runtime_factor_upper_bound'])} | "
            f"{pair['delta']['official_cost']:.3f} |"
        )

    lines.extend(
        [
            "",
            "## 2. Why does the trained checkpoint lose to untrained?",
            f"- Pairwise result on the exact quoted pair (`{best_trained['variant']} seed {best_trained['seed']}` vs `{best_untrained['variant']} seed {best_untrained['seed']}`): trained wins only `{trained_wins}/20` cases and loses `{trained_losses}/20`.",
            f"- Mean deltas (trained minus untrained): `Δcost={mean_deltas['official_cost']:.3f}`, `ΔHPWLgap={mean_deltas['hpwl_gap']:.3f}`, `ΔAreagap_bbox={mean_deltas['area_gap']:.3f}`, `ΔViolationsrelative={mean_deltas['violations_relative']:.3f}`, `ΔRuntimeFactor={mean_deltas['runtime_factor']:.3f}`.",
            f"- Driver attribution: swapping only trained `HPWLgap + Areagap_bbox` to untrained values drops mean cost from `{trained_mean_cost:.3f}` to `{mean_quality_swapped_cost:.3f}`; adding untrained soft-violation levels drops it to `{mean_quality_and_viol_swapped_cost:.3f}`; matching runtime closes only the last `{runtime_gap_contribution:.3f}`.",
            f"- Main loss driver: `HPWL + area` account for `{quality_gap_contribution / pair_gap:.1%}` of the mean gap, soft violations `{soft_gap_contribution / pair_gap:.1%}`, runtime `{runtime_gap_contribution / pair_gap:.1%}`. Between the two quality terms, area is slightly worse than HPWL (`{_average(single_term_gains['area']):.3f}` vs `{_average(single_term_gains['hpwl']):.3f}` mean single-term gain if swapped).",
            "",
            "| Top 5 trained losses | Δcost | Δhpwl | Δarea | Δviol | Δruntime | Δdisp |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )

    for pair in top_5_trained_losses:
        delta_runtime = pair["delta"]["runtime_factor"]
        delta_runtime_text = "n/a" if delta_runtime is None else f"{delta_runtime:.3f}"
        lines.append(
            f"| {pair['case_id']} | {pair['delta']['official_cost']:.3f} | {pair['delta']['hpwl_gap']:.3f} | "
            f"{pair['delta']['area_gap']:.3f} | {pair['delta']['violations_relative']:.3f} | "
            f"{delta_runtime_text} | {pair['delta']['mean_displacement']:.3f} |"
        )

    lines.extend(
        [
            "",
            "## 3. Is lower repair displacement actually useful?",
            f"- Pearson on all `140` artifact points: `disp vs cost = {all_corr['displacement_vs_official_cost']:.3f}`, `disp vs HPWLgap = {all_corr['displacement_vs_hpwl_gap']:.3f}`, `disp vs Areagap_bbox = {all_corr['displacement_vs_area_gap']:.3f}`.",
            f"- On the best-pair deltas only (`20` cases): `Δdisp vs Δcost = {delta_corr['delta_displacement_vs_delta_official_cost']:.3f}`, `Δdisp vs ΔHPWLgap = {delta_corr['delta_displacement_vs_delta_hpwl_gap']:.3f}`, `Δdisp vs ΔAreagap_bbox = {delta_corr['delta_displacement_vs_delta_area_gap']:.3f}`.",
            f"- Best-pair sanity check: trained lowers displacement on `{len(lower_displacement_cases)}/20` cases, but still raises cost on `{len(lower_displacement_but_higher_cost)}` of those `{len(lower_displacement_cases)}`.",
            "- Judgement: lower repair displacement is a poor proxy for official score on this slice. It does not reliably predict better HPWL, better bbox area, or lower official cost.",
            "",
            "## Commands Used",
        ]
    )
    lines.extend([f"- `{cmd}`" for cmd in COMMANDS_USED])
    lines.extend(
        [
            "",
            "## Next Diagnostic Recommendation",
            f"- {next_recommendation}",
        ]
    )

    OUTPUT_JSON.write_text(json.dumps(result, indent=2))
    OUTPUT_MD.write_text("\n".join(lines))
    print(
        json.dumps(
            {
                "markdown": str(OUTPUT_MD.relative_to(ROOT)),
                "json": str(OUTPUT_JSON.relative_to(ROOT)),
                "paired_cases": len(paired_cases),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
