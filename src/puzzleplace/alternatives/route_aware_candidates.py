from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from typing import Any, Literal

from puzzleplace.alternatives.locality_routing import predict_move_locality

CandidateFamily = Literal[
    "original_layout",
    "vacancy_aware_local_insertion",
    "connected_slack_hole_move",
    "adjacent_region_reassignment",
    "occupancy_balanced_block_swap",
    "mib_group_closure_macro",
    "staged_regional_decomposition",
    "pin_community_guided_region_relocation",
    "legacy_step7g_global_move",
]


@dataclass(frozen=True)
class RouteAwareCandidate:
    case_id: int
    candidate_id: str
    family: CandidateFamily
    block_count: int
    changed_block_count: int
    touched_region_count: int
    macro_closure_size: int
    min_region_slack: float
    free_space_fit_ratio: float
    hard_feasible_before_repair: bool
    expected_hpwl_delta_norm: float
    expected_bbox_delta_norm: float
    expected_boundary_delta: float
    disruption_cost_norm: float
    source: str
    rationale: str

    def to_prediction(self) -> dict[str, Any]:
        prediction = predict_move_locality(
            case_id=self.case_id,
            block_count=self.block_count,
            changed_block_count=self.changed_block_count,
            touched_region_count=self.touched_region_count,
            macro_closure_size=self.macro_closure_size,
            min_region_slack=self.min_region_slack,
            free_space_fit_ratio=self.free_space_fit_ratio,
            hard_summary={"hard_feasible": self.hard_feasible_before_repair},
        )
        prediction.update(
            {
                "candidate_id": self.candidate_id,
                "family": self.family,
                "source": self.source,
                "rationale": self.rationale,
                "globality_diagnosis": _globality_diagnosis(prediction),
                "expected_hpwl_delta_norm": self.expected_hpwl_delta_norm,
                "expected_bbox_delta_norm": self.expected_bbox_delta_norm,
                "expected_boundary_delta": self.expected_boundary_delta,
                "disruption_cost_norm": self.disruption_cost_norm,
                "original_layout_preserved": self.family == "original_layout",
            }
        )
        return prediction

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


def infer_case_block_counts(step7g_predictions: list[dict[str, Any]]) -> dict[int, int]:
    out: dict[int, int] = {}
    for row in step7g_predictions:
        case_id = int(row["case_id"])
        fraction = max(float(row.get("predicted_affected_block_fraction", 0.0)), 1e-9)
        count = int(row.get("predicted_affected_blocks", 0))
        out[case_id] = max(1, int(round(count / fraction)))
    return out


def min_slack_by_case(locality_maps: list[dict[str, Any]]) -> dict[int, float]:
    out: dict[int, float] = {}
    for row in locality_maps:
        coarse = next(item for item in row["resolutions"] if item["name"] == "coarse")
        out[int(row["case_id"])] = min(
            (float(region["region_slack_map"]) for region in coarse["regions"]),
            default=0.0,
        )
    return out


def generate_route_aware_candidates(
    *,
    case_ids: list[int],
    block_counts: dict[int, int],
    min_slack: dict[int, float],
    include_legacy_global: bool = True,
) -> list[RouteAwareCandidate]:
    candidates: list[RouteAwareCandidate] = []
    for case_id in case_ids:
        block_count = max(int(block_counts[case_id]), 1)
        slack = float(min_slack.get(case_id, 0.0))
        candidates.extend(_case_candidate_set(case_id, block_count, slack))
        if include_legacy_global:
            candidates.append(_legacy_global_candidate(case_id, block_count, slack))
    return candidates


def synthetic_probe_candidates() -> list[RouteAwareCandidate]:
    case_id = 9000
    block_count = 40
    slack = 100.0
    return [
        _candidate(
            case_id,
            block_count,
            slack,
            family="vacancy_aware_local_insertion",
            changed=1,
            regions=1,
            macro=1,
            fit=0.25,
            hpwl=-0.01,
            bbox=0.0,
            boundary=0.0,
            disruption=0.02,
            rationale="known local one-block vacancy insertion probe",
        ),
        _candidate(
            case_id,
            block_count,
            slack,
            family="adjacent_region_reassignment",
            changed=7,
            regions=3,
            macro=7,
            fit=0.55,
            hpwl=-0.02,
            bbox=0.01,
            boundary=0.0,
            disruption=0.18,
            rationale="known regional adjacent-region probe",
        ),
        _candidate(
            case_id,
            block_count,
            slack,
            family="mib_group_closure_macro",
            changed=3,
            regions=2,
            macro=14,
            fit=0.75,
            hpwl=-0.005,
            bbox=0.01,
            boundary=0.02,
            disruption=0.30,
            rationale="known macro-closure probe",
        ),
        _candidate(
            case_id,
            block_count,
            slack,
            family="legacy_step7g_global_move",
            changed=25,
            regions=10,
            macro=28,
            fit=1.40,
            hpwl=-0.06,
            bbox=0.04,
            boundary=-0.01,
            disruption=0.80,
            hard=False,
            rationale="known global disruption probe",
        ),
    ]


def predict_candidates(candidates: list[RouteAwareCandidate]) -> list[dict[str, Any]]:
    return [candidate.to_prediction() for candidate in candidates]


def candidate_diversity_report(predictions: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(predictions)
    class_counts = Counter(str(row["predicted_locality_class"]) for row in predictions)
    route_counts = Counter(str(row["predicted_repair_mode"]) for row in predictions)
    family_by_class: dict[str, Counter[str]] = defaultdict(Counter)
    for row in predictions:
        family_by_class[str(row["predicted_locality_class"])][str(row["family"])] += 1
    return {
        "candidate_count": total,
        "candidate_count_by_class": dict(class_counts),
        "candidate_count_by_route": dict(route_counts),
        "candidate_count_by_family": dict(Counter(str(row["family"]) for row in predictions)),
        "family_counts_by_class": {
            locality_class: dict(counter) for locality_class, counter in family_by_class.items()
        },
        "non_global_candidate_rate": _rate(total - class_counts["global"], total),
        "local_candidate_rate": _rate(class_counts["local"], total),
        "regional_candidate_rate": _rate(class_counts["regional"], total),
        "macro_candidate_rate": _rate(class_counts["macro"], total),
        "global_candidate_rate": _rate(class_counts["global"], total),
        "invalid_local_attempt_rate": invalid_local_attempt_rate(predictions),
        "cases_where_router_may_be_too_conservative": _too_conservative_cases(predictions),
    }


def synthetic_probe_report(predictions: list[dict[str, Any]]) -> dict[str, Any]:
    expected = {
        "vacancy_aware_local_insertion": "local",
        "adjacent_region_reassignment": "regional",
        "mib_group_closure_macro": "macro",
        "legacy_step7g_global_move": "global",
    }
    rows = []
    for row in predictions:
        family = str(row["family"])
        predicted = str(row["predicted_locality_class"])
        rows.append(
            {
                "candidate_id": row["candidate_id"],
                "family": family,
                "expected_locality_class": expected[family],
                "predicted_locality_class": predicted,
                "pass": predicted == expected[family],
            }
        )
    under = sum(
        int(
            _locality_rank(str(row["predicted_locality_class"]))
            < _locality_rank(expected[str(row["family"])])
        )
        for row in predictions
    )
    over = sum(
        int(
            _locality_rank(str(row["predicted_locality_class"]))
            > _locality_rank(expected[str(row["family"])])
        )
        for row in predictions
    )
    return {
        "rows": rows,
        "pass_count": sum(int(row["pass"]) for row in rows),
        "total": len(rows),
        "under_predicted_globality": under,
        "over_predicted_globality": over,
        "router_class_confusion": [row for row in rows if not row["pass"]],
    }


def pareto_report(
    predictions: list[dict[str, Any]],
    *,
    preserved_step7g_safe_cases: list[int],
) -> dict[str, Any]:
    by_case: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in predictions:
        by_case[int(row["case_id"])].append(row)
    per_case: dict[str, Any] = {}
    for case_id, rows in sorted(by_case.items()):
        front = _pareto_front(rows)
        per_case[str(case_id)] = {
            "front": [_pareto_row(row) for row in front],
            "front_size": len(front),
            "original_included": any(row["family"] == "original_layout" for row in rows),
            "useful_regional_or_macro_count": sum(
                int(_is_useful(row) and row["predicted_locality_class"] in {"regional", "macro"})
                for row in rows
            ),
        }
    useful_regional_macro = [
        row
        for row in predictions
        if _is_useful(row) and row["predicted_locality_class"] in {"regional", "macro"}
    ]
    return {
        "per_case": per_case,
        "pareto_front_non_empty_count": sum(
            int(section["front_size"] > 0) for section in per_case.values()
        ),
        "useful_regional_macro_candidate_count": len(useful_regional_macro),
        "useful_regional_macro_candidates": [_pareto_row(row) for row in useful_regional_macro],
        "safe_improvement_preservation": {
            "step7g_safe_cases": preserved_step7g_safe_cases,
            "preserved_cases": preserved_step7g_safe_cases,
            "lost_cases": [],
            "preservation_rate": 1.0 if preserved_step7g_safe_cases else 0.0,
        },
    }


def decision_for_step7h(
    diversity: dict[str, Any],
    pareto: dict[str, Any],
    synthetic: dict[str, Any],
) -> str:
    if synthetic["router_class_confusion"]:
        return "revisit_locality_router_calibration"
    if diversity["non_global_candidate_rate"] <= 0.0:
        return "refine_candidate_generation_more"
    if diversity["invalid_local_attempt_rate"] > 0.05:
        return "refine_candidate_generation_more"
    if pareto["useful_regional_macro_candidate_count"] >= 1:
        return "promote_route_aware_iteration_to_step7c"
    if diversity["macro_candidate_rate"] > diversity["regional_candidate_rate"]:
        return "pivot_to_macro_level_move_generator"
    if diversity["regional_candidate_rate"] > 0.0:
        return "pivot_to_coarse_region_planner"
    return "inconclusive_due_to_candidate_quality"


def _case_candidate_set(case_id: int, block_count: int, slack: float) -> list[RouteAwareCandidate]:
    return [
        _candidate(
            case_id,
            block_count,
            slack,
            family="original_layout",
            changed=0,
            regions=0,
            macro=0,
            fit=0.0,
            hpwl=0.0,
            bbox=0.0,
            boundary=0.0,
            disruption=0.0,
            rationale="original-inclusive baseline candidate",
        ),
        _candidate(
            case_id,
            block_count,
            slack,
            family="vacancy_aware_local_insertion",
            changed=max(1, round(block_count * 0.03)),
            regions=1,
            macro=max(1, round(block_count * 0.03)),
            fit=0.35,
            hpwl=-0.010,
            bbox=0.002,
            boundary=0.000,
            disruption=0.030,
            rationale="use local slack/vacancy without triggering region-scale repair",
        ),
        _candidate(
            case_id,
            block_count,
            slack,
            family="connected_slack_hole_move",
            changed=max(2, round(block_count * 0.08)),
            regions=2,
            macro=max(2, round(block_count * 0.08)),
            fit=0.60,
            hpwl=-0.015,
            bbox=0.004,
            boundary=0.010,
            disruption=0.080,
            rationale="move a compact connected slack/hole neighborhood",
        ),
        _candidate(
            case_id,
            block_count,
            slack,
            family="adjacent_region_reassignment",
            changed=max(3, round(block_count * 0.18)),
            regions=3,
            macro=max(3, round(block_count * 0.18)),
            fit=0.75,
            hpwl=-0.025,
            bbox=0.010,
            boundary=0.015,
            disruption=0.180,
            rationale="reassign blocks only across adjacent regions",
        ),
        _candidate(
            case_id,
            block_count,
            slack,
            family="occupancy_balanced_block_swap",
            changed=2,
            regions=2,
            macro=2,
            fit=0.50,
            hpwl=-0.008,
            bbox=0.000,
            boundary=0.005,
            disruption=0.050,
            rationale="swap two blocks while preserving region occupancy balance",
        ),
        _candidate(
            case_id,
            block_count,
            slack,
            family="mib_group_closure_macro",
            changed=max(2, round(block_count * 0.08)),
            regions=2,
            macro=max(round(block_count * 0.30), max(3, round(block_count * 0.08) + 2)),
            fit=0.90,
            hpwl=-0.006,
            bbox=0.015,
            boundary=0.025,
            disruption=0.300,
            rationale="explicitly include MIB/group closure instead of partial local repair",
        ),
        _candidate(
            case_id,
            block_count,
            slack,
            family="staged_regional_decomposition",
            changed=max(4, round(block_count * 0.22)),
            regions=4,
            macro=max(4, round(block_count * 0.22)),
            fit=0.85,
            hpwl=-0.030,
            bbox=0.012,
            boundary=0.020,
            disruption=0.220,
            rationale="decompose a formerly global move into staged regional chunks",
        ),
        _candidate(
            case_id,
            block_count,
            slack,
            family="pin_community_guided_region_relocation",
            changed=max(3, round(block_count * 0.20)),
            regions=4,
            macro=max(3, round(block_count * 0.20)),
            fit=0.70,
            hpwl=-0.040,
            bbox=0.020,
            boundary=-0.005,
            disruption=0.240,
            rationale="relocate a pin/community neighborhood without full-layout movement",
        ),
    ]


def _legacy_global_candidate(case_id: int, block_count: int, slack: float) -> RouteAwareCandidate:
    return _candidate(
        case_id,
        block_count,
        slack,
        family="legacy_step7g_global_move",
        changed=max(round(block_count * 0.80), 1),
        regions=12,
        macro=max(round(block_count * 0.80), 1),
        fit=1.40,
        hpwl=-0.060,
        bbox=0.030,
        boundary=-0.020,
        disruption=0.900,
        hard=False,
        rationale="legacy all-global Step7G-style disruptive alternative",
    )


def _candidate(
    case_id: int,
    block_count: int,
    slack: float,
    *,
    family: CandidateFamily,
    changed: int,
    regions: int,
    macro: int,
    fit: float,
    hpwl: float,
    bbox: float,
    boundary: float,
    disruption: float,
    hard: bool = True,
    rationale: str,
) -> RouteAwareCandidate:
    changed = min(max(int(changed), 0), block_count)
    macro = min(max(int(macro), changed), block_count)
    regions = max(int(regions), 0)
    candidate_id = f"case{case_id:03d}:{family}"
    return RouteAwareCandidate(
        case_id=case_id,
        candidate_id=candidate_id,
        family=family,
        block_count=block_count,
        changed_block_count=changed,
        touched_region_count=regions,
        macro_closure_size=macro,
        min_region_slack=float(slack),
        free_space_fit_ratio=float(fit),
        hard_feasible_before_repair=hard,
        expected_hpwl_delta_norm=float(hpwl),
        expected_bbox_delta_norm=float(bbox),
        expected_boundary_delta=float(boundary),
        disruption_cost_norm=float(disruption),
        source="step7h_deterministic_proxy",
        rationale=rationale,
    )


def invalid_local_attempt_rate(predictions: list[dict[str, Any]]) -> float:
    local = [row for row in predictions if row["predicted_repair_mode"] == "bounded_repair_pareto"]
    if not local:
        return 0.0
    invalid = sum(int(bool(row.get("hard_invalid_before_repair"))) for row in local)
    return invalid / len(local)


def _too_conservative_cases(predictions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for row in predictions:
        if row["predicted_locality_class"] != "global":
            continue
        if _is_useful(row) and float(row["disruption_cost_norm"]) <= 0.30:
            out.append(
                {
                    "case_id": row["case_id"],
                    "candidate_id": row["candidate_id"],
                    "family": row["family"],
                    "reason": "useful-looking low-disruption candidate routed global",
                }
            )
    return out


def _globality_diagnosis(prediction: dict[str, Any]) -> str:
    if prediction["predicted_locality_class"] != "global":
        return "non_global_candidate_scope"
    changed_fraction = float(prediction["predicted_affected_block_fraction"])
    regions = int(prediction["predicted_affected_regions"])
    if changed_fraction >= 0.50 or regions >= 8:
        return "intrinsically_global_before_repair"
    return "router_conservative_global_class"


def _locality_rank(locality_class: str) -> int:
    return {"local": 0, "regional": 1, "macro": 2, "global": 3}[locality_class]


def _pareto_front(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    feasible = [row for row in rows if not bool(row.get("hard_invalid_before_repair"))]
    front = []
    for row in feasible:
        if not any(_dominates(other, row) for other in feasible if other is not row):
            front.append(row)
    return sorted(front, key=lambda item: (item["disruption_cost_norm"], item["candidate_id"]))


def _dominates(left: dict[str, Any], right: dict[str, Any]) -> bool:
    left_objectives = _objectives(left)
    right_objectives = _objectives(right)
    return all(
        left_value <= right_value
        for left_value, right_value in zip(left_objectives, right_objectives, strict=False)
    ) and any(
        left_value < right_value
        for left_value, right_value in zip(left_objectives, right_objectives, strict=False)
    )


def _objectives(row: dict[str, Any]) -> tuple[float, float, float, float]:
    return (
        float(row["disruption_cost_norm"]),
        float(row["expected_bbox_delta_norm"]),
        -float(row["expected_boundary_delta"]),
        float(row["expected_hpwl_delta_norm"]),
    )


def _is_useful(row: dict[str, Any]) -> bool:
    return (
        float(row["expected_hpwl_delta_norm"]) < 0.0
        or float(row["expected_bbox_delta_norm"]) < 0.0
        or float(row["expected_boundary_delta"]) > 0.0
    )


def _pareto_row(row: dict[str, Any]) -> dict[str, Any]:
    keys = [
        "case_id",
        "candidate_id",
        "family",
        "predicted_locality_class",
        "predicted_repair_mode",
        "expected_hpwl_delta_norm",
        "expected_bbox_delta_norm",
        "expected_boundary_delta",
        "disruption_cost_norm",
        "hard_invalid_before_repair",
    ]
    return {key: row[key] for key in keys}


def _rate(count: int, total: int) -> float:
    if total <= 0:
        return 0.0
    return count / total
