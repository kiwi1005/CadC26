#!/usr/bin/env python3
"""Run the deterministic Boundary-First Obligation-Driven v1 prototype.

This is an experiment-side search.  It consumes fixed P8 incumbents, never
changes the contest solve path, and lets the official evaluator plus the local
hard verifier admit every candidate.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace
from typing import Any, Iterable

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from experiment_p10_case70 import (  # noqa: E402
    _baseline_row,
    _measure,
    _official_case,
    _placement_sha256,
    _render_png,
)
from hcfp.boundary_skeleton import boundary_skeleton_candidates  # noqa: E402
from hcfp.btree import (  # noqa: E402
    btree_dimension_variants,
    contact_aware_vertical_orders,
    decode_connectivity_btree_beam,
    local_tree_variants,
    subtree_move_variants,
)
from hcfp.btree_forest import btree_forest_candidates  # noqa: E402
from hcfp.constraints.boundary_slots import construct_boundary_slots  # noqa: E402
from hcfp.contact_patch import dense_contact_patch_candidates  # noqa: E402
from hcfp.contact_policy import (  # noqa: E402
    ContactPolicy,
    load_contact_policy,
    rank_contact_candidates,
)
from hcfp.contact_synthesis import synthesize_contact_obligations  # noqa: E402
from hcfp.mib_patch import mib_anchor_patch_candidates  # noqa: E402
from hcfp.verify import (  # noqa: E402
    bbox,
    bbox_area,
    boundary_missing,
    grouping_violation,
    verify_feasible,
)


DEFAULT_CASES = (70, 89, 90, 94, 97)
DEFAULT_INCUMBENTS = (
    "artifacts/experiments/p10_case70/p8_replay.json",
    "artifacts/benchmarks/hcfp5090-p8-btree-beam-lambda10-cases70-89-90-94-97.json",
)


@dataclass(frozen=True)
class Config:
    beam_width: int
    max_rounds: int
    top_experts: int
    proposals_per_operator: int
    exact_decode_cap: int
    runtime_ceiling: float
    patch_sizes: tuple[int, ...]
    contact_only: bool = False
    group_first: bool = False


@dataclass(frozen=True)
class Context:
    case_id: int
    evaluator_module: Any
    case: Any
    raw_case: dict[str, Any]
    raw_object: Any
    metric_args: tuple[Any, ...]
    visual_case: dict[str, Any]
    b2b_edges: list[Any]
    contact_policy: ContactPolicy | None
    contact_policy_metadata: dict[str, Any] | None
    group_first: bool = False


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--incumbent",
        action="append",
        default=None,
        help="ordered P8 replay JSON; first artifact containing a case wins",
    )
    parser.add_argument("--lane", default="learned")
    parser.add_argument("--cases", default=",".join(map(str, DEFAULT_CASES)))
    parser.add_argument("--data-path", default="artifacts/floorset-v10")
    parser.add_argument("--output-dir", default="artifacts/experiments/bfod_v1")
    parser.add_argument("--runtime-ceiling", type=float, default=30.0)
    parser.add_argument("--beam-width", type=int, default=4)
    parser.add_argument("--max-rounds", type=int, default=6)
    parser.add_argument("--top-experts", type=int, default=2)
    parser.add_argument("--proposals-per-operator", type=int, default=4)
    parser.add_argument("--exact-decode-cap", type=int, default=96)
    parser.add_argument("--patch-sizes", default="4,8,12,16")
    parser.add_argument(
        "--contact-policy",
        help="experiment-only learned contact-patch ranker checkpoint",
    )
    parser.add_argument(
        "--audit",
        action="store_true",
        help="record every scored candidate per round into case{id}/audit.json",
    )
    parser.add_argument(
        "--contact-only",
        action="store_true",
        help="disable S1/S2/S4 and the sparse route; run S0 -> S3 -> S5 only",
    )
    parser.add_argument(
        "--group-first",
        action="store_true",
        help=(
            "generator v2: rank disconnected-group obligations first and drop "
            "joint/MIB/HPWL obligation rows (removes wasted expert attempts)"
        ),
    )
    args = parser.parse_args(argv)
    cases = _parse_cases(args.cases)
    patch_sizes = _parse_positive_ints(args.patch_sizes, "--patch-sizes")
    config = Config(
        beam_width=_positive(args.beam_width, "--beam-width"),
        max_rounds=_positive(args.max_rounds, "--max-rounds"),
        top_experts=_positive(args.top_experts, "--top-experts"),
        proposals_per_operator=_positive(
            args.proposals_per_operator, "--proposals-per-operator"
        ),
        exact_decode_cap=_positive(args.exact_decode_cap, "--exact-decode-cap"),
        runtime_ceiling=_positive_float(args.runtime_ceiling, "--runtime-ceiling"),
        patch_sizes=patch_sizes,
        contact_only=bool(args.contact_only) or bool(args.group_first),
        group_first=bool(args.group_first),
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    contact_policy = None
    contact_policy_metadata = None
    if args.contact_policy:
        contact_policy, contact_policy_metadata = load_contact_policy(args.contact_policy)
    incumbents = _load_incumbents(args.incumbent or list(DEFAULT_INCUMBENTS))
    started = time.perf_counter()
    reports = []
    for case_id in cases:
        row, source = _incumbent_row(incumbents, case_id, args.lane)
        reports.append(
            _run_case(
                case_id,
                row,
                source,
                Path(args.data_path),
                output_dir,
                config,
                contact_policy,
                contact_policy_metadata,
                audit=bool(args.audit),
            )
        )
    summary = {
        "method": "Boundary-First Obligation-Driven Cooperative Floorplanning v1",
        "scope": "deterministic sidecar experiment; production solver unchanged",
        "config": _jsonable(config),
        "cases": reports,
        "runtime_seconds": time.perf_counter() - started,
        "provenance": _provenance(args, incumbents),
        "contact_policy": contact_policy_metadata,
    }
    _dump(output_dir / "summary.json", summary)
    _write_report(output_dir / "report.md", summary)
    print(output_dir / "summary.json")
    return 0


def _run_case(
    case_id: int,
    incumbent: dict[str, Any],
    incumbent_source: dict[str, Any],
    data_path: Path,
    output_dir: Path,
    config: Config,
    contact_policy: ContactPolicy | None,
    contact_policy_metadata: dict[str, Any] | None,
    *,
    audit: bool = False,
) -> dict[str, Any]:
    started = time.perf_counter()
    (
        evaluator_module,
        case,
        raw_case,
        metric_args,
        visual_case,
        b2b_edges,
        _official_baseline,
    ) = _official_case(data_path, case_id)
    raw_case = {
        **raw_case,
        "cluster_group_ids": case.cluster_group_ids,
        "mib_group_ids": case.mib_group_ids,
    }
    context = Context(
        case_id=case_id,
        evaluator_module=evaluator_module,
        case=case,
        raw_case=raw_case,
        raw_object=SimpleNamespace(**raw_case),
        metric_args=metric_args,
        visual_case=visual_case,
        b2b_edges=b2b_edges,
        contact_policy=contact_policy,
        contact_policy_metadata=contact_policy_metadata,
        group_first=config.group_first,
    )
    baseline_positions = torch.as_tensor(
        incumbent["positions"], dtype=torch.float64, device="cpu"
    )
    baseline_metrics = _measure(
        evaluator_module, raw_case, metric_args, baseline_positions
    )
    if not baseline_metrics["hard_feasible"]:
        raise RuntimeError(f"P8 incumbent for case{case_id} is not hard feasible")
    route = _route(raw_case, baseline_positions, baseline_metrics)
    if config.contact_only:
        route = {**route, "name": "dense_common_loop"}
    if config.group_first:
        route = {**route, "group_first": True}
    skeleton = _perimeter_skeleton(case, raw_case, baseline_positions)
    base_state = _state(baseline_positions, baseline_metrics, history=())

    audit_log: list[dict[str, Any]] = [] if audit else None  # type: ignore[assignment]
    if config.contact_only:
        mib_state, mib_report = base_state, {
            "decision": "SKIP",
            "reason": "contact-only mode",
            "before": _metrics_brief(base_state["metrics"]),
            "after": _metrics_brief(base_state["metrics"]),
            "rounds": [],
        }
    else:
        mib_state, mib_report = _mib_stage(context, base_state, config, audit_log)
    contact_state, contact_report = _bootstrap_contact(
        context, mib_state, route, config, audit_log
    )
    if config.contact_only:
        tree_state, tree_report = contact_state, {
            "decision": "SKIP",
            "reason": "contact-only mode",
            "expert": "tree",
            "candidate_count": 0,
        }
    else:
        tree_state, tree_report = _topology_stage(
            context, contact_state, route, config, audit_log
        )
    remaining_runtime = max(0.0, config.runtime_ceiling - (time.perf_counter() - started))
    loop_state, loop_report = _common_loop(
        context,
        tree_state,
        route,
        config,
        runtime_ceiling=remaining_runtime,
        audit=audit_log,
    )
    winner = min(
        (base_state, mib_state, contact_state, tree_state, loop_state), key=_state_key
    )
    case_dir = output_dir / f"case{case_id}"
    case_dir.mkdir(parents=True, exist_ok=True)
    if audit_log:
        _dump(case_dir / "audit.json", {"case_id": case_id, "stages": audit_log})
    _render_png(
        case_dir / "baseline.png",
        base_state["positions"],
        visual_case,
        b2b_edges,
        base_state["metrics"],
        title=f"BFOD-v1 case{case_id} P8 baseline",
    )
    for stage, stage_state in (
        ("s2_mib", mib_state),
        ("s3_contact", contact_state),
        ("s4_topology", tree_state),
        ("s5_common_loop", loop_state),
    ):
        _render_png(
            case_dir / f"{stage}.png",
            stage_state["positions"],
            visual_case,
            b2b_edges,
            stage_state["metrics"],
            title=f"BFOD-v1 case{case_id} {stage}",
            patch=stage_state["history"][-1].get("members", ())
            if stage_state["history"]
            else (),
        )
    _render_png(
        case_dir / "winner.png",
        winner["positions"],
        visual_case,
        b2b_edges,
        winner["metrics"],
        title=f"BFOD-v1 case{case_id} winner",
        patch=winner["history"][-1].get("members", ()) if winner["history"] else (),
    )
    result = {
        "test_id": case_id,
        "route": route,
        "incumbent": incumbent_source,
        "contact_policy": contact_policy_metadata,
        "baseline": _state_record(base_state, include_positions=True),
        "s0_hard_geometry": _hard_geometry(raw_case),
        "s1_perimeter_skeleton": skeleton,
        "s2_mib_shape_construction": mib_report,
        "s3_contact_topology": contact_report,
        "s4_topology_refinement": tree_report,
        "s5_common_loop": loop_report,
        "winner": _state_record(winner, include_positions=True),
        "runtime_seconds": time.perf_counter() - started,
    }
    _dump(case_dir / "result.json", result)
    return {
        "test_id": case_id,
        "route": route["name"],
        "baseline": _state_record(base_state),
        "winner": _state_record(winner),
        "decision": "KEEP" if _state_key(winner) < _state_key(base_state) else "REJECT",
        "runtime_seconds": result["runtime_seconds"],
        "artifact": str(case_dir / "result.json"),
    }


def _mib_stage(
    context: Context,
    state: dict[str, Any],
    config: Config,
    audit: list[dict[str, Any]] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if state["metrics"]["mib_violations"] == 0:
        return state, {
            "decision": "SKIP",
            "reason": "no residual MIB debt",
            "before": _metrics_brief(state["metrics"]),
            "after": _metrics_brief(state["metrics"]),
            "rounds": [],
        }
    current = state
    rounds = []
    audit_rounds = [] if audit is not None else None
    for step in range(1, max(1, current["metrics"]["mib_violations"]) + 1):
        candidates = mib_anchor_patch_candidates(
            context.raw_case,
            current["positions"],
            verify_case=context.raw_case,
            patch_sizes=config.patch_sizes,
            max_candidates=config.proposals_per_operator,
        )
        records = _score_candidates(context, "mib", candidates, _mib_details)
        accepted = _best_admitted(
            current,
            records,
            require_soft="mib",
        )
        rounds.append(
            {
                "step": step,
                "candidate_count": len(records),
                "accepted": None if accepted is None else _candidate_summary(accepted),
            }
        )
        if audit_rounds is not None:
            audit_rounds.append(
                _audit_round(
                    context,
                    current,
                    ("mib",),
                    records,
                    require_soft="mib",
                    round_label=step,
                )
            )
        if accepted is None:
            break
        current = _state(
            accepted["positions"],
            accepted["metrics"],
            history=(*current["history"], _history(accepted)),
        )
        if current["metrics"]["mib_violations"] == 0:
            break
    if audit is not None and audit_rounds is not None:
        audit.append({"stage": "s2_mib", "rounds": audit_rounds})
    return current, {
        "decision": "KEEP" if _state_key(current) < _state_key(state) else "REJECT",
        "before": _metrics_brief(state["metrics"]),
        "after": _metrics_brief(current["metrics"]),
        "rounds": rounds,
    }


def _bootstrap_contact(
    context: Context,
    state: dict[str, Any],
    route: dict[str, Any],
    config: Config,
    audit: list[dict[str, Any]] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    obligations = _residual_obligations(context, state["positions"])
    experts = [
        expert
        for expert in _choose_experts(obligations, route, config.top_experts)
        if expert in {"joint", "contact", "boundary"}
    ]
    if not experts:
        return state, {"decision": "SKIP", "reason": "no residual boundary/group obligation"}
    candidates = []
    attempted = []
    produced_counts: list[int] = []
    for expert in _fallback_experts(experts[:1]):
        attempted.append(expert)
        produced = _operator_candidates(context, state, expert, config)
        candidates.extend(produced)
        produced_counts.append(len(produced))
        if produced:
            break
    records = _score_raw_candidates(context, candidates)
    if audit is not None:
        runs = []
        cursor = 0
        for expert, count in zip(attempted, produced_counts):
            runs.append({"expert": expert, "records": records[cursor : cursor + count]})
            cursor += count
        audit.append(
            {
                "stage": "s3_contact_bootstrap",
                "rounds": [
                    _audit_round(
                        context,
                        state,
                        attempted,
                        records,
                        require_soft="boundary_or_group",
                        round_label=0,
                        runs=runs,
                    )
                ],
            }
        )
    accepted = _best_admitted(state, records, require_soft="boundary_or_group")
    if accepted is None:
        return state, {
            "decision": "REJECT",
            "experts": attempted,
            "obligation": obligations[0] if obligations else None,
            "candidate_count": len(records),
        }
    result = _state(
        accepted["positions"],
        accepted["metrics"],
        history=(*state["history"], _history(accepted)),
    )
    return result, {
        "decision": "KEEP",
        "experts": attempted,
        "obligation": obligations[0] if obligations else None,
        "candidate_count": len(records),
        "accepted": _candidate_summary(accepted),
    }


def _topology_stage(
    context: Context,
    state: dict[str, Any],
    route: dict[str, Any],
    config: Config,
    audit: list[dict[str, Any]] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    expert = "region" if route["name"] == "sparse_region" else "tree"
    records = _score_raw_candidates(
        context, _operator_candidates(context, state, expert, config)
    )
    if audit is not None:
        audit.append(
            {
                "stage": "s4_topology",
                "rounds": [
                    _audit_round(
                        context,
                        state,
                        (expert,),
                        records,
                        require_soft=None,
                        round_label=0,
                    )
                ],
            }
        )
    accepted = _best_admitted(state, records, require_soft=None)
    if accepted is None:
        return state, {"decision": "REJECT", "expert": expert, "candidate_count": len(records)}
    result = _state(
        accepted["positions"],
        accepted["metrics"],
        history=(*state["history"], _history(accepted)),
    )
    return result, {
        "decision": "KEEP",
        "expert": expert,
        "candidate_count": len(records),
        "accepted": _candidate_summary(accepted),
    }


def _common_loop(
    context: Context,
    initial: dict[str, Any],
    route: dict[str, Any],
    config: Config,
    *,
    runtime_ceiling: float,
    audit: list[dict[str, Any]] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    started = time.perf_counter()
    if runtime_ceiling <= 0.0:
        return initial, {
            "decision": "SKIP_RUNTIME",
            "beam_width": config.beam_width,
            "maximum_rounds": config.max_rounds,
            "exact_decode_cap": config.exact_decode_cap,
            "exact_decodes": 0,
            "runtime_seconds": 0.0,
            "runtime_budget_seconds": runtime_ceiling,
            "rounds": [],
            "best": _state_record(initial),
        }
    # A local exact repack is non-interruptible; preserve a small end reserve.
    in_flight_reserve = min(4.0, 0.2 * runtime_ceiling)
    search_budget = max(0.0, runtime_ceiling - in_flight_reserve)
    beam = [initial]
    best = initial
    decoded = 0
    stagnant = 0
    rounds = []
    audit_rounds = [] if audit is not None else None
    seen = {_placement_sha256(initial["positions"])}

    for round_index in range(1, config.max_rounds + 1):
        proposals = []
        round_detail = {"round": round_index, "states": []}
        audit_round_states = [] if audit is not None else None
        for state in beam:
            if decoded >= config.exact_decode_cap or time.perf_counter() - started >= search_budget:
                break
            obligations = _residual_obligations(context, state["positions"])
            experts = _choose_experts(obligations, route, config.top_experts)
            state_detail = {
                "obligations": obligations[: config.top_experts],
                "experts": experts,
                "attempted_experts": [],
                "candidate_count": 0,
                "accepted": 0,
            }
            audit_runs = [] if audit is not None else None
            for expert in experts:
                remaining = config.exact_decode_cap - decoded
                if remaining <= 0:
                    break
                records = []
                for attempted in _fallback_experts((expert,)):
                    state_detail["attempted_experts"].append(attempted)
                    raw_candidates = _operator_candidates(
                        context,
                        state,
                        attempted,
                        config,
                        limit=min(config.proposals_per_operator, remaining),
                    )
                    # Never score the same placement twice in one search: a
                    # fallback chain can re-enter an already-tried operator.
                    raw_candidates = [
                        item
                        for item in raw_candidates
                        if _placement_sha256(item["positions"]) not in seen
                    ]
                    records = _score_raw_candidates(context, raw_candidates)
                    if records:
                        break
                decoded += len(records)
                state_detail["candidate_count"] += len(records)
                if audit_runs is not None:
                    audit_runs.append({"expert": attempted, "records": records})
                for record in records:
                    digest = record["placement_sha256"]
                    if digest in seen:
                        if audit is not None:
                            record["_audit_duplicate"] = True
                        continue
                    if not _admitted(state["metrics"], record["metrics"], None):
                        continue
                    seen.add(digest)
                    proposals.append(
                        _state(
                            record["positions"],
                            record["metrics"],
                            history=(*state["history"], _history(record)),
                        )
                    )
                    state_detail["accepted"] += 1
                if time.perf_counter() - started >= search_budget:
                    break
            round_detail["states"].append(state_detail)
            if audit is not None:
                audit_round_states.append(
                    _audit_round(
                        context,
                        state,
                        experts,
                        [record for run in audit_runs for record in run["records"]],
                        require_soft=None,
                        round_label=round_index,
                        runs=audit_runs,
                    )
                )
        proposals.sort(key=_state_key)
        beam = proposals[: config.beam_width]
        round_detail["beam"] = [_state_record(item) for item in beam]
        rounds.append(round_detail)
        if audit_round_states is not None:
            accepted = None
            for state_index, entry in enumerate(audit_round_states):
                selected = entry["selected"]
                if selected is None:
                    continue
                key = (
                    selected["uncapped_cost"],
                    selected["metrics"]["total_soft_violations"],
                    selected["bbox_area"],
                    selected["hpwl_total"],
                    selected["candidate_index"],
                )
                if accepted is None or key < accepted[0]:
                    accepted = (key, state_index, selected)
            audit_rounds.append(
                {
                    "round": round_index,
                    "states": audit_round_states,
                    "accepted": None
                    if accepted is None
                    else {
                        "state_index": accepted[1],
                        "selected": accepted[2],
                        "state": audit_round_states[accepted[1]]["state"],
                    },
                }
            )
        if not beam:
            break
        if _state_key(beam[0]) < _state_key(best):
            best = beam[0]
            stagnant = 0
        else:
            stagnant += 1
        if stagnant >= 2 or decoded >= config.exact_decode_cap:
            break
    if audit is not None and audit_rounds is not None:
        audit.append({"stage": "s5_common_loop", "rounds": audit_rounds})
    return best, {
        "decision": "KEEP" if _state_key(best) < _state_key(initial) else "REJECT",
        "beam_width": config.beam_width,
        "maximum_rounds": config.max_rounds,
        "top_experts_per_state": config.top_experts,
        "proposals_per_operator": config.proposals_per_operator,
        "exact_decode_cap": config.exact_decode_cap,
        "exact_decodes": decoded,
        "runtime_seconds": time.perf_counter() - started,
        "runtime_budget_seconds": runtime_ceiling,
        "in_flight_reserve_seconds": in_flight_reserve,
        "rounds": rounds,
        "best": _state_record(best),
    }


def _operator_candidates(
    context: Context,
    state: dict[str, Any],
    expert: str,
    config: Config,
    *,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    budget = config.proposals_per_operator if limit is None else limit
    if budget <= 0:
        return []
    if (config.contact_only or config.group_first) and expert != "contact":
        return []
    if expert == "mib":
        return [
            {
                "family": "mib",
                "positions": candidate.placement,
                "details": _mib_details(candidate, context),
            }
            for candidate in mib_anchor_patch_candidates(
                context.raw_case,
                state["positions"],
                verify_case=context.raw_case,
                patch_sizes=config.patch_sizes,
                max_candidates=budget,
            )
        ]
    if expert == "boundary":
        candidates = boundary_skeleton_candidates(
            context.case,
            state["positions"],
            verify_case=context.raw_case,
            patch_sizes=config.patch_sizes,
            max_candidates=max(16, budget),
        )
        ordered = sorted(
            candidates,
            key=lambda candidate: (
                _side_priority(candidate.required_sides),
                candidate.missing_after,
                candidate.block,
            ),
        )[:budget]
        return [
            {
                "family": "boundary",
                "positions": candidate.placement,
                "details": {
                    "block": candidate.block,
                    "required_sides": candidate.required_sides,
                    "members": candidate.members,
                    "missing_before": candidate.missing_before,
                    "missing_after": candidate.missing_after,
                },
            }
            for candidate in ordered
        ]
    if expert == "contact":
        return _contact_candidates(context, state["positions"], config, budget)
    if expert == "joint":
        return _joint_candidates(context, state["positions"], config, budget)
    if expert == "tree":
        return _tree_candidates(context, state, config, budget)
    if expert == "region":
        return [
            {
                "family": "region",
                "positions": candidate,
                "details": {"strategy": "free-rectangle forest relocation"},
            }
            for candidate in btree_forest_candidates(
                context.raw_object,
                state["positions"],
                max_candidates=budget,
            )
        ]
    raise ValueError(f"unknown expert {expert}")


def _contact_candidates(
    context: Context, positions: Any, config: Config, budget: int
) -> list[dict[str, Any]]:
    candidates = dense_contact_patch_candidates(
        context.case,
        positions,
        verify_case=context.raw_case,
        patch_sizes=config.patch_sizes,
        max_candidates=max(16, budget),
    )
    priority = {3: 0, 4: 1, 1: 2, 2: 3}
    policy_scores: dict[tuple[int, int, int, str, tuple[int, ...]], float] = {}
    if context.contact_policy is not None:
        ranked = rank_contact_candidates(
            context.contact_policy,
            context.case,
            context.raw_case,
            positions,
            candidates,
        )
        ordered = []
        for candidate, score in ranked[:budget]:
            key = (
                candidate.group_index,
                candidate.bridge_member,
                candidate.anchor_member,
                candidate.side,
                candidate.members,
            )
            policy_scores[key] = score
            ordered.append(candidate)
    else:
        ordered = sorted(
            candidates,
            key=lambda candidate: (
                priority.get(int(context.case.cluster_group_ids[candidate.group_index]), 4),
                candidate.grouping_after,
                candidate.bridge_member,
                candidate.anchor_member,
            ),
        )[:budget]
    return [
        {
            "family": "contact",
            "positions": candidate.placement,
            "details": {
                **_contact_details(candidate, context),
                **(
                    {
                        "contact_policy_score": policy_scores[
                            (
                                candidate.group_index,
                                candidate.bridge_member,
                                candidate.anchor_member,
                                candidate.side,
                                candidate.members,
                            )
                        ]
                    }
                    if policy_scores
                    else {}
                ),
            },
        }
        for candidate in ordered
    ]


def _joint_candidates(
    context: Context, positions: Any, config: Config, budget: int
) -> list[dict[str, Any]]:
    before_boundary = int(torch.count_nonzero(boundary_missing(context.raw_case, positions)))
    before_group = grouping_violation(context.raw_case, positions)
    output: list[dict[str, Any]] = []
    seen: set[str] = set()
    contacts = _contact_candidates(context, positions, config, min(2, max(1, budget)))
    for contact in contacts:
        candidate = torch.as_tensor(contact["positions"], dtype=torch.float64)
        after_boundary = int(
            torch.count_nonzero(boundary_missing(context.raw_case, candidate))
        )
        after_group = grouping_violation(context.raw_case, candidate)
        if after_boundary < before_boundary and after_group < before_group:
            _append_unique(output, seen, {**contact, "family": "joint", "details": {"mode": "contact", **contact["details"]}}, budget)
        if len(output) >= budget:
            return output
        boundary = boundary_skeleton_candidates(
            context.case,
            candidate,
            verify_case=context.raw_case,
            patch_sizes=config.patch_sizes,
            max_candidates=1,
        )
        for repair in boundary:
            after_group = grouping_violation(context.raw_case, repair.placement)
            after_boundary = int(
                torch.count_nonzero(boundary_missing(context.raw_case, repair.placement))
            )
            if after_boundary < before_boundary and after_group < before_group:
                _append_unique(
                    output,
                    seen,
                    {
                        "family": "joint",
                        "positions": repair.placement,
                        "details": {
                            "mode": "contact_then_boundary",
                            "contact": contact["details"],
                            "boundary": {
                                "block": repair.block,
                                "required_sides": repair.required_sides,
                                "members": repair.members,
                            },
                            "members": tuple(
                                sorted(
                                    set(contact["details"]["members"])
                                    | set(repair.members)
                                )
                            ),
                        },
                    },
                    budget,
                )
            if len(output) >= budget:
                return output
    return output


def _tree_candidates(
    context: Context,
    state: dict[str, Any],
    config: Config,
    budget: int,
) -> list[dict[str, Any]]:
    boxes = torch.as_tensor(state["positions"], dtype=torch.float64, device="cpu")
    centers = boxes[:, :2] + 0.5 * boxes[:, 2:4]
    weights = torch.as_tensor(context.raw_case["b2b_weight"], dtype=torch.float64)
    degree = weights.sum(dim=1)
    for _pin, block, weight in torch.as_tensor(
        context.raw_case["p2b_connectivity"], dtype=torch.float64
    ).tolist():
        if block >= 0:
            degree[int(block)] += float(weight)
    distance = torch.abs(centers[:, None] - centers[None, :]).sum(dim=2)
    scale = max(float(distance[distance > 0].median()), 1.0e-9)
    base = -distance / scale + torch.log1p(weights)
    edge_logits = torch.stack((base, base), dim=2)
    edge_logits[:, :, 0] += 0.15 * (centers[:, 0, None] >= centers[None, :, 0])
    edge_logits[:, :, 1] += 0.15 * (centers[:, 1, None] >= centers[None, :, 1])
    root_logits = degree / max(float(degree.max()), 1.0) + 0.05 * context.case.boundary_bits.sum(dim=1)
    trees = list(
        decode_connectivity_btree_beam(
            root_logits,
            edge_logits,
            b2b_weight=context.case.b2b_weight,
            group_membership=context.case.group_membership,
            boundary_bits=context.case.boundary_bits,
            beam_width=config.beam_width,
            connectivity_weight=0.3,
        )
    )
    trees = _tree_move_beam(trees, context, config.beam_width)
    dimensions = _role_aware_dimensions(context, boxes, state["history"], degree)
    vertical = _preferred_order(
        boxes,
        context.case.boundary_bits,
        context.case.group_membership,
        axis=1,
    )
    horizontal = _preferred_order(
        boxes,
        context.case.boundary_bits,
        context.case.group_membership,
        axis=0,
    )
    left, bottom, _, _ = bbox(boxes)
    target = torch.as_tensor(context.raw_case["target"], dtype=torch.float64)
    preplaced = torch.as_tensor(context.raw_case["preplaced_mask"], dtype=torch.bool)
    candidates: list[dict[str, Any]] = []
    seen: set[str] = set()
    for tree_index, (variant, tree) in enumerate(trees):
        for axis, placement in (
            (
                "x_compacted",
                tree.pack_x_compacted(
                    dimensions, vertical, preplaced, target, origin=(left, bottom)
                ),
            ),
            (
                "y_compacted",
                tree.pack_y_compacted(
                    dimensions, horizontal, preplaced, target, origin=(left, bottom)
                ),
            ),
        ):
            digest = _placement_sha256(placement)
            if digest in seen or not verify_feasible(context.raw_case, placement):
                continue
            seen.add(digest)
            candidates.append(
                {
                    "family": "tree",
                    "positions": placement,
                    "details": {
                        "tree_variant": variant,
                        "tree_index": tree_index,
                        "axis": axis,
                        "shape_variant": "role_aware",
                    },
                }
            )
            if len(candidates) >= budget:
                return candidates
    return candidates


def _tree_move_beam(trees: list[Any], context: Context, width: int) -> list[tuple[str, Any]]:
    if not trees:
        return []
    root = trees[0]
    local = local_tree_variants(
        root,
        context.case.boundary_bits,
        context.case.group_membership,
        limit=max(32, root.block_count),
    )
    sibling = next((item for item in local if item[0].startswith("sibling_flip")), None)
    reinsert = next((item for item in local if item[0].startswith("group_reinsert")), None)
    transpose = next(iter(subtree_move_variants(root, limit=1)), None)
    pool: list[tuple[str, Any]] = [("connectivity_beam", root)]
    for item in (sibling, reinsert, transpose):
        if item is not None:
            pool.append(item)
    pool.extend(("connectivity_beam", tree) for tree in trees[1:])
    unique: list[tuple[str, Any]] = []
    seen: set[tuple[tuple[int, ...], tuple[int, ...]]] = set()
    for name, tree in pool:
        key = (tree.left, tree.right)
        if key not in seen:
            seen.add(key)
            unique.append((name, tree))
        if len(unique) >= width:
            break
    return unique


def _role_aware_dimensions(
    context: Context,
    boxes: torch.Tensor,
    history: tuple[dict[str, Any], ...],
    degree: torch.Tensor,
) -> torch.Tensor:
    dimensions = btree_dimension_variants(
        boxes[:, 2:4],
        fixed_mask=context.raw_case["fixed_mask"],
        preplaced_mask=context.raw_case["preplaced_mask"],
        mib_membership=context.raw_case["mib_membership"],
        weighted_degree=degree,
        areas=torch.as_tensor(context.raw_case["area"], dtype=torch.float64),
    )["net_aware"]
    if not history:
        return dimensions
    details = history[-1].get("details", {})
    bridge = details.get("bridge_member")
    side = details.get("side")
    if bridge is None or side not in {"left", "right", "top", "bottom"}:
        return dimensions
    index = int(bridge)
    hard = torch.as_tensor(context.raw_case["fixed_mask"], dtype=torch.bool) | torch.as_tensor(
        context.raw_case["preplaced_mask"], dtype=torch.bool
    )
    mib = torch.as_tensor(context.raw_case["mib_membership"], dtype=torch.bool)
    if bool(hard[index]) or bool(mib.any(dim=0)[index]):
        return dimensions
    area = float(torch.as_tensor(context.raw_case["area"])[index])
    ratio = 0.5 if side in {"left", "right"} else 2.0
    width = math.sqrt(area * ratio)
    dimensions[index] = dimensions.new_tensor((width, area / width))
    return dimensions


def _preferred_order(
    boxes: torch.Tensor, bits: torch.Tensor, groups: torch.Tensor, *, axis: int
) -> torch.Tensor:
    base = torch.tensor(
        sorted(range(len(boxes)), key=lambda index: (float(boxes[index, axis]), index)),
        dtype=torch.long,
    )
    variants = dict(contact_aware_vertical_orders(base, bits, groups))
    return variants.get("boundary_group", variants["base"])


def _residual_obligations(context: Context, positions: Any) -> list[dict[str, Any]]:
    boxes = torch.as_tensor(positions, dtype=torch.float64, device="cpu")
    missing = boundary_missing(context.raw_case, boxes)
    bounds = bbox(boxes)
    degree = torch.as_tensor(context.raw_case["b2b_weight"], dtype=torch.float64).sum(dim=1)
    rows: list[dict[str, Any]] = []
    for block in torch.nonzero(missing != 0, as_tuple=False).reshape(-1).tolist():
        sides = _missing_sides(int(missing[block]))
        distance = _boundary_distance(boxes[block], bounds, sides)
        rows.append(
            _obligation(
                "boundary",
                f"B{block}:{'+'.join(sides)}",
                benefit=len(sides),
                difficulty=1.0 + distance + 0.05 * float(degree[block]),
                details={"block": block, "sides": sides, "distance": distance},
            )
        )
    synthesis = synthesize_contact_obligations(context.case, boxes, tolerance=1.0e-6)
    for index, obligation in enumerate(synthesis.obligations):
        group_id = int(context.case.cluster_group_ids[obligation.group_index])
        details = {
            "group_id": group_id,
            "group_index": int(obligation.group_index),
            "bridge_member": int(obligation.bridge_member),
            "anchor_member": int(obligation.anchor_member),
            "side": obligation.side,
            "members": tuple(
                sorted(set(obligation.component_a) | set(obligation.component_b))
            ),
        }
        rows.append(
            _obligation(
                "group",
                f"G{group_id}:{index}",
                benefit=1.0,
                difficulty=1.0
                + float(obligation.move_distance)
                + 0.05 * float(obligation.net_incident),
                details=details,
            )
        )
        joint_sides = int(missing[obligation.bridge_member] | missing[obligation.anchor_member])
        if joint_sides and not context.group_first:
            rows.append(
                _obligation(
                    "joint",
                    f"J{group_id}:{index}",
                    benefit=1.0 + len(_missing_sides(joint_sides)),
                    difficulty=1.0
                    + float(obligation.move_distance)
                    + 0.05 * float(obligation.net_incident),
                    details={**details, "boundary_sides": _missing_sides(joint_sides)},
                )
            )
    for group_index, membership in enumerate(context.case.mib_membership):
        if context.group_first:
            break
        members = torch.nonzero(membership, as_tuple=False).reshape(-1).tolist()
        shapes = {
            tuple(round(float(value), 4) for value in boxes[member, 2:4].tolist())
            for member in members
        }
        if len(shapes) > 1:
            rows.append(
                _obligation(
                    "mib",
                    f"M{int(context.case.mib_group_ids[group_index])}",
                    benefit=float(len(shapes) - 1),
                    difficulty=1.0 + 0.25 * len(members),
                    details={"group_index": group_index, "members": tuple(members)},
                )
            )
    high = _high_weight_edge(boxes, context.raw_case["b2b_weight"])
    if high is not None and not context.group_first:
        first, second, weighted_distance = high
        rows.append(
            _obligation(
                "hpwl",
                f"N{first}-{second}",
                benefit=0.25,
                difficulty=1.0 + weighted_distance,
                details={"first": first, "second": second, "weighted_distance": weighted_distance},
            )
        )
    rank = {"joint": 0, "mib": 1, "group": 2, "boundary": 3, "hpwl": 4}
    return sorted(rows, key=lambda row: (-row["priority"], rank[row["kind"]], row["id"]))


def _obligation(kind: str, name: str, *, benefit: float, difficulty: float, details: dict[str, Any]) -> dict[str, Any]:
    return {
        "kind": kind,
        "id": name,
        "benefit": benefit,
        "difficulty": difficulty,
        "priority": benefit / max(difficulty, 1.0e-9),
        "details": details,
    }


def _choose_experts(
    obligations: list[dict[str, Any]], route: dict[str, Any], limit: int
) -> list[str]:
    if route["name"] == "sparse_region":
        return ["region"]
    if route.get("group_first"):
        # Generator v2: only the contact operator is allowed; boundary/MIB/tree
        # expert attempts are removed entirely (they produced no winners).
        return ["contact"]
    mapping = {
        "joint": "joint",
        "mib": "mib",
        "group": "contact",
        "boundary": "boundary",
        "hpwl": "tree",
    }
    experts: list[str] = []
    for obligation in obligations:
        expert = mapping[obligation["kind"]]
        if expert not in experts:
            experts.append(expert)
        if len(experts) >= limit:
            return experts
    if "tree" not in experts:
        experts.append("tree")
    return experts[:limit]


def _fallback_experts(experts: Iterable[str]) -> list[str]:
    """Try the selected operator first, then its closest bounded fallback."""

    fallback = {
        "joint": ("joint", "contact", "boundary"),
        "contact": ("contact", "boundary", "tree"),
        "boundary": ("boundary", "contact", "tree"),
        "mib": ("mib",),
        "tree": ("tree",),
        "region": ("region", "tree"),
    }
    output: list[str] = []
    for expert in experts:
        for candidate in fallback[expert]:
            if candidate not in output:
                output.append(candidate)
    return output


def _route(raw_case: dict[str, Any], positions: Any, metrics: dict[str, Any]) -> dict[str, Any]:
    boxes = torch.as_tensor(positions, dtype=torch.float64)
    utilization = float((boxes[:, 2] * boxes[:, 3]).sum()) / max(bbox_area(boxes), 1.0e-12)
    soft = int(metrics["total_soft_violations"])
    if utilization >= 0.90 and soft:
        name = "dense_common_loop"
    elif utilization < 0.50:
        name = "sparse_region"
    else:
        name = "normal_btree"
    return {"name": name, "utilization": utilization, "soft_debt": soft}


def _perimeter_skeleton(case: Any, raw_case: dict[str, Any], positions: Any) -> dict[str, Any]:
    boxes = torch.as_tensor(positions, dtype=torch.float64)
    slots = construct_boundary_slots(case.boundary_bits, boxes)
    bits = torch.as_tensor(case.boundary_bits, dtype=torch.bool)
    sides = ("left", "right", "top", "bottom")
    degree = torch.as_tensor(raw_case["b2b_weight"], dtype=torch.float64).sum(dim=1)
    centers = boxes[:, :2] + 0.5 * boxes[:, 2:4]
    membership = {side: [] for side in sides}
    corners: dict[str, list[int]] = {"top_left": [], "top_right": [], "bottom_left": [], "bottom_right": []}
    for block in range(len(boxes)):
        active = [side for index, side in enumerate(sides) if bool(bits[block, index])]
        for side in active:
            membership[side].append(block)
        active_set = set(active)
        for name, needed in (
            ("top_left", {"top", "left"}),
            ("top_right", {"top", "right"}),
            ("bottom_left", {"bottom", "left"}),
            ("bottom_right", {"bottom", "right"}),
        ):
            if needed <= active_set:
                corners[name].append(block)
    preplaced = torch.as_tensor(raw_case["preplaced_mask"], dtype=torch.bool)
    order_candidates = {}
    for side, members in membership.items():
        axis = 1 if side in {"left", "right"} else 0
        current = sorted(members, key=lambda block: (float(centers[block, axis]), block))
        order_candidates[side] = {
            "current": current,
            "net_aware": sorted(members, key=lambda block: (-float(degree[block]), block)),
            "group_aware": sorted(
                members,
                key=lambda block: (
                    _first_group(case.group_membership, block),
                    float(centers[block, axis]),
                    block,
                ),
            ),
            "hybrid": sorted(
                members,
                key=lambda block: (
                    _first_group(case.group_membership, block),
                    -float(degree[block]),
                    float(centers[block, axis]),
                    block,
                ),
            ),
        }
    return {
        "corner_first_members": corners,
        "side_membership": membership,
        "preplaced_boundary_members": {
            side: [block for block in members if bool(preplaced[block])]
            for side, members in membership.items()
        },
        "side_order_candidates": order_candidates,
        "slot_equalities": [_jsonable(item) for item in slots.equalities],
        "slot_orders": [_jsonable(item) for item in slots.orders],
    }


def _hard_geometry(raw_case: dict[str, Any]) -> dict[str, Any]:
    fixed = torch.as_tensor(raw_case["fixed_mask"], dtype=torch.bool)
    preplaced = torch.as_tensor(raw_case["preplaced_mask"], dtype=torch.bool)
    return {
        "preplaced_freeze": torch.nonzero(preplaced, as_tuple=False).reshape(-1).tolist(),
        "fixed_shape_freeze": torch.nonzero(fixed, as_tuple=False).reshape(-1).tolist(),
        "position_movable": torch.nonzero(~preplaced, as_tuple=False)
        .reshape(-1)
        .tolist(),
        "shape_movable": torch.nonzero(~(fixed | preplaced), as_tuple=False)
        .reshape(-1)
        .tolist(),
    }


def _score_candidates(
    context: Context,
    family: str,
    candidates: Iterable[Any],
    details_fn: Any,
) -> list[dict[str, Any]]:
    return _score_raw_candidates(
        context,
        [
            {"family": family, "positions": candidate.placement, "details": details_fn(candidate, context)}
            for candidate in candidates
        ],
    )


def _score_raw_candidates(context: Context, candidates: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for index, candidate in enumerate(candidates):
        positions = torch.as_tensor(candidate["positions"], dtype=torch.float64)
        output.append(
            {
                "family": candidate["family"],
                "candidate_index": index,
                "details": candidate["details"],
                "positions": positions.tolist(),
                "metrics": _measure(
                    context.evaluator_module,
                    context.raw_case,
                    context.metric_args,
                    positions,
                ),
                "placement_sha256": _placement_sha256(positions),
            }
        )
    return output


def _best_admitted(
    state: dict[str, Any], records: Iterable[dict[str, Any]], require_soft: str | None
) -> dict[str, Any] | None:
    candidates = [
        record
        for record in records
        if _admitted(state["metrics"], record["metrics"], require_soft)
    ]
    return min(candidates, key=_record_key) if candidates else None


def _admitted(
    before: dict[str, Any], after: dict[str, Any], require_soft: str | None
) -> bool:
    if not after["hard_feasible"]:
        return False
    if any(
        after[name] > before[name]
        for name in ("boundary_violations", "grouping_violations", "mib_violations")
    ):
        return False
    if require_soft == "mib" and not after["mib_violations"] < before["mib_violations"]:
        return False
    if require_soft == "boundary_or_group" and not (
        after["boundary_violations"] < before["boundary_violations"]
        or after["grouping_violations"] < before["grouping_violations"]
    ):
        return False
    return after["uncapped_cost"] < before["uncapped_cost"] - 1.0e-10


def _state(positions: Any, metrics: dict[str, Any], *, history: Iterable[dict[str, Any]]) -> dict[str, Any]:
    return {
        "positions": torch.as_tensor(positions, dtype=torch.float64).tolist(),
        "metrics": metrics,
        "history": tuple(history),
    }


def _state_key(state: dict[str, Any]) -> tuple[Any, ...]:
    metrics = state["metrics"]
    return (
        metrics["uncapped_cost"],
        metrics["total_soft_violations"],
        metrics["bbox_area"],
        metrics["hpwl_total"],
        _placement_sha256(state["positions"]),
    )


def _record_key(record: dict[str, Any]) -> tuple[Any, ...]:
    return _state_key({"positions": record["positions"], "metrics": record["metrics"]})


def _state_record(state: dict[str, Any], *, include_positions: bool = False) -> dict[str, Any]:
    record = {
        "metrics": _metrics_brief(state["metrics"]),
        "placement_sha256": _placement_sha256(state["positions"]),
        "history": list(state["history"]),
    }
    if include_positions:
        record["positions"] = state["positions"]
    return record


def _metrics_brief(metrics: dict[str, Any]) -> dict[str, Any]:
    return {
        key: metrics[key]
        for key in (
            "cost",
            "uncapped_cost",
            "boundary_violations",
            "grouping_violations",
            "mib_violations",
            "total_soft_violations",
            "hpwl_gap",
            "area_gap",
            "hard_feasible",
        )
    }


def _candidate_summary(candidate: dict[str, Any]) -> dict[str, Any]:
    return {
        "family": candidate["family"],
        "candidate_index": candidate["candidate_index"],
        "details": candidate["details"],
        "metrics": _metrics_brief(candidate["metrics"]),
        "placement_sha256": candidate["placement_sha256"],
    }


def _history(candidate: dict[str, Any]) -> dict[str, Any]:
    return {
        "family": candidate["family"],
        "details": candidate["details"],
        "metrics": _metrics_brief(candidate["metrics"]),
    }


def _audit_state_detail(state: dict[str, Any]) -> dict[str, Any]:
    boxes = torch.as_tensor(state["positions"], dtype=torch.float64)
    left, bottom, right, top = bbox(boxes)
    return {
        "metrics": _metrics_brief(state["metrics"]),
        "bbox": [float(left), float(bottom), float(right), float(top)],
        "bbox_area": float(state["metrics"]["bbox_area"]),
        "hpwl_total": float(state["metrics"]["hpwl_total"]),
        "uncapped_cost": float(state["metrics"]["uncapped_cost"]),
    }


def _audit_reject_reason(
    before: dict[str, Any], record: dict[str, Any], require_soft: str | None
) -> str | None:
    """Mirror _admitted and label why a record would not be admitted."""

    after = record["metrics"]
    if not after["hard_feasible"]:
        return "hard_infeasible"
    for name in ("boundary_violations", "grouping_violations", "mib_violations"):
        if after[name] > before[name]:
            return f"{name}_regression"
    if require_soft == "mib" and not after["mib_violations"] < before["mib_violations"]:
        return "mib_not_reduced"
    if require_soft == "boundary_or_group" and not (
        after["boundary_violations"] < before["boundary_violations"]
        or after["grouping_violations"] < before["grouping_violations"]
    ):
        return "soft_not_reduced"
    if not after["uncapped_cost"] < before["uncapped_cost"] - 1.0e-10:
        return "cost_not_lower"
    return None


def _audit_components(
    context: Context, positions: Any
) -> dict[tuple[int, int, int], dict[str, Any]]:
    """Deterministic re-derivation of the synthesis table the generator used."""

    synthesis = synthesize_contact_obligations(context.case, positions)
    output: dict[tuple[int, int, int], dict[str, Any]] = {}
    for obligation in synthesis.obligations + synthesis.candidate_edges:
        key = (
            int(obligation.group_index),
            int(obligation.bridge_member),
            int(obligation.anchor_member),
        )
        if key not in output:
            output[key] = {
                "component_a": list(obligation.component_a),
                "component_b": list(obligation.component_b),
                "component_a_size": len(obligation.component_a),
                "component_b_size": len(obligation.component_b),
                "moving_component": list(obligation.moving_component),
                "moving_component_size": len(obligation.moving_component),
                "member_a": int(obligation.member_a),
                "member_b": int(obligation.member_b),
                "side": obligation.side,
                "move_distance": float(obligation.move_distance),
                "bbox_expansion": float(obligation.bbox_expansion),
                "net_incident": float(obligation.net_incident),
            }
    return output


def _audit_record(
    context: Context,
    components: dict[tuple[int, int, int], dict[str, Any]],
    state: dict[str, Any],
    record: dict[str, Any],
    require_soft: str | None,
) -> dict[str, Any]:
    details = record["details"]
    base = (
        details.get("contact")
        if isinstance(details.get("contact"), dict)
        else details
    )
    key = (
        base.get("group_index"),
        base.get("bridge_member"),
        base.get("anchor_member"),
    )
    enriched = details
    if (
        key[0] is not None
        and key[1] is not None
        and key[2] is not None
        and key in components
    ):
        enriched = {**details, "obligation": components[key]}
    boxes = torch.as_tensor(record["positions"], dtype=torch.float64)
    left, bottom, right, top = bbox(boxes)
    return {
        "family": record["family"],
        "candidate_index": record["candidate_index"],
        "details": enriched,
        "metrics": _metrics_brief(record["metrics"]),
        "bbox": [float(left), float(bottom), float(right), float(top)],
        "bbox_area": float(record["metrics"]["bbox_area"]),
        "hpwl_total": float(record["metrics"]["hpwl_total"]),
        "uncapped_cost": float(record["metrics"]["uncapped_cost"]),
        "reject_reason": _audit_reject_reason(state["metrics"], record, require_soft),
        "duplicate": bool(record.get("_audit_duplicate", False)),
    }


def _audit_best(entries: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not entries:
        return None
    best = min(
        entries,
        key=lambda entry: (
            entry["uncapped_cost"],
            entry["metrics"]["total_soft_violations"],
            entry["bbox_area"],
            entry["hpwl_total"],
            entry["candidate_index"],
        ),
    )
    return {
        "family": best["family"],
        "candidate_index": best["candidate_index"],
        "details": best["details"],
        "uncapped_cost": best["uncapped_cost"],
        "bbox_area": best["bbox_area"],
        "hpwl_total": best["hpwl_total"],
        "reject_reason": best["reject_reason"],
        "duplicate": best["duplicate"],
        "metrics": _metrics_brief(best["metrics"]),
    }


def _audit_round(
    context: Context,
    state: dict[str, Any],
    experts: Iterable[str],
    records: list[dict[str, Any]],
    *,
    require_soft: str | None,
    round_label: int,
    runs: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Per-state round audit: every scored candidate + oracle vs selected."""

    obligations = _residual_obligations(context, state["positions"])
    components = _audit_components(context, state["positions"])
    entries = [
        _audit_record(context, components, state, record, require_soft)
        for record in records
    ]
    if runs is None:
        runs = [{"expert": "all", "records": entries}]
    else:
        cursor = 0
        rebuilt = []
        for run in runs:
            count = len(run["records"])
            rebuilt.append(
                {"expert": run["expert"], "records": entries[cursor : cursor + count]}
            )
            cursor += count
        runs = rebuilt
    feasible = [entry for entry in entries if entry["metrics"]["hard_feasible"]]
    admitted = [entry for entry in entries if entry["reject_reason"] is None]
    oracle = _audit_best(feasible)
    selected = _audit_best(admitted)
    if selected is None:
        classification = "generation_failure"
    elif oracle is not None and oracle["duplicate"]:
        classification = "duplicate_gap"
    elif oracle is not None and oracle["uncapped_cost"] < selected["uncapped_cost"] - 1.0e-12:
        classification = "ranking_failure"
    else:
        classification = "success"
    return {
        "state": _audit_state_detail(state),
        "obligations": obligations[:2],
        "experts": list(experts),
        "runs": runs,
        "oracle": oracle,
        "selected": selected,
        "classification": classification,
    }


def _mib_details(candidate: Any, context: Context) -> dict[str, Any]:
    return {
        "group_index": candidate.group_index,
        "group_id": int(context.case.mib_group_ids[candidate.group_index]),
        "anchor_member": candidate.anchor_member,
        "target_member": candidate.target_member,
        "members": candidate.members,
        "target_shape": candidate.target_shape,
        "mib_before": candidate.mib_before,
        "mib_after": candidate.mib_after,
    }


def _contact_details(candidate: Any, context: Context) -> dict[str, Any]:
    return {
        "group_index": candidate.group_index,
        "group_id": int(context.case.cluster_group_ids[candidate.group_index]),
        "bridge_member": candidate.bridge_member,
        "anchor_member": candidate.anchor_member,
        "members": candidate.members,
        "side": candidate.side,
        "grouping_before": candidate.grouping_before,
        "grouping_after": candidate.grouping_after,
    }


def _append_unique(
    output: list[dict[str, Any]], seen: set[str], candidate: dict[str, Any], limit: int
) -> None:
    digest = _placement_sha256(candidate["positions"])
    if digest not in seen and len(output) < limit:
        seen.add(digest)
        output.append(candidate)


def _side_priority(sides: Iterable[str]) -> int:
    order = {"right": 0, "top": 1, "left": 2, "bottom": 3}
    return min((order[side] for side in sides), default=4)


def _missing_sides(mask: int) -> tuple[str, ...]:
    return tuple(
        side
        for side, bit in (("left", 1), ("right", 2), ("top", 4), ("bottom", 8))
        if mask & bit
    )


def _boundary_distance(
    box: torch.Tensor, bounds: tuple[float, float, float, float], sides: Iterable[str]
) -> float:
    left, bottom, right, top = bounds
    distances = {
        "left": float(box[0]) - left,
        "right": right - float(box[0] + box[2]),
        "top": top - float(box[1] + box[3]),
        "bottom": float(box[1]) - bottom,
    }
    return sum(max(0.0, distances[side]) for side in sides)


def _high_weight_edge(boxes: torch.Tensor, weights: Any) -> tuple[int, int, float] | None:
    matrix = torch.as_tensor(weights, dtype=torch.float64)
    centers = boxes[:, :2] + 0.5 * boxes[:, 2:4]
    best: tuple[int, int, float] | None = None
    for first in range(len(boxes)):
        for second in range(first + 1, len(boxes)):
            weighted = float(matrix[first, second]) * float(
                torch.abs(centers[first] - centers[second]).sum()
            )
            if best is None or weighted > best[2]:
                best = (first, second, weighted)
    return best


def _first_group(groups: torch.Tensor, block: int) -> int:
    member = torch.nonzero(groups[:, block], as_tuple=False).reshape(-1)
    return int(member[0]) if member.numel() else int(groups.shape[0]) + block


def _load_incumbents(paths: Iterable[str]) -> list[dict[str, Any]]:
    output = []
    for raw in paths:
        path = Path(raw)
        payload = json.loads(path.read_text(encoding="utf-8"))
        output.append(
            {
                "path": str(path),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "payload": payload,
            }
        )
    return output


def _incumbent_row(
    incumbents: Iterable[dict[str, Any]], case_id: int, lane: str
) -> tuple[dict[str, Any], dict[str, Any]]:
    for source in incumbents:
        payload = source["payload"]
        rows = payload.get("lanes", {}).get(lane, ())
        for row in rows:
            if int(row.get("test_id", -1)) == case_id:
                return row, {key: source[key] for key in ("path", "sha256")}
        try:
            return _baseline_row(payload, case_id), {
                key: source[key] for key in ("path", "sha256")
            }
        except ValueError:
            continue
    raise ValueError(f"no ordered incumbent artifact contains case {case_id}")


def _parse_cases(value: str) -> tuple[int, ...]:
    result = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not result or any(item < 0 for item in result):
        raise ValueError("--cases must contain non-negative ids")
    return result


def _parse_positive_ints(value: str, name: str) -> tuple[int, ...]:
    result = tuple(sorted({int(item.strip()) for item in value.split(",") if item.strip()}))
    if not result or result[0] <= 0:
        raise ValueError(f"{name} must contain positive integers")
    return result


def _positive(value: int, name: str) -> int:
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


def _positive_float(value: float, name: str) -> float:
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return value


def _provenance(args: argparse.Namespace, incumbents: list[dict[str, Any]]) -> dict[str, Any]:
    status = subprocess.run(
        ["git", "status", "--short"],
        cwd=ROOT,
        check=True,
        text=True,
        capture_output=True,
    ).stdout
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        text=True,
        capture_output=True,
    ).stdout.strip()
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "command": " ".join(sys.argv),
        "git_commit": commit,
        "git_clean": not status.strip(),
        "git_status_sha256": hashlib.sha256(status.encode()).hexdigest(),
        "source_sha256": {
            str(path.relative_to(ROOT)): _file_sha256(path)
            for path in (
                ROOT / "scripts/experiment_bfod_v1.py",
                ROOT / "scripts/experiment_p10_case70.py",
                ROOT / "src/hcfp/contact_policy.py",
                ROOT / "src/hcfp/mib_patch.py",
            )
        },
        "incumbents": [{key: item[key] for key in ("path", "sha256")} for item in incumbents],
        "experiment_only": True,
    }


def _write_report(path: Path, summary: dict[str, Any]) -> None:
    rows = [
        "| Case | Route | Baseline | Winner | B/G/M | Feasible | Decision | Runtime |",
        "|---:|---|---:|---:|---|:---:|---|---:|",
    ]
    for item in summary["cases"]:
        baseline = item["baseline"]["metrics"]
        winner = item["winner"]["metrics"]
        rows.append(
            f"| {item['test_id']} | {item['route']} | {baseline['cost']:.6f} | "
            f"{winner['cost']:.6f} | {winner['boundary_violations']}/"
            f"{winner['grouping_violations']}/{winner['mib_violations']} | "
            f"{'yes' if winner['hard_feasible'] else 'no'} | {item['decision']} | "
            f"{item['runtime_seconds']:.3f}s |"
        )
    text = "\n".join(
        (
            "# Boundary-First Obligation-Driven Cooperative Floorplanning v1",
            "",
            "Deterministic prototype: hard masks and perimeter membership first, then bounded MIB/contact/tree operators under the exact scorer. No production solver path changed.",
            "",
            *rows,
            "",
            "The common loop locks only non-regression of B/G/M counts plus hard feasibility; it does not lock a particular witness or contact edge.",
        )
    )
    path.write_text(text + "\n", encoding="utf-8")


def _jsonable(value: Any) -> Any:
    if hasattr(value, "__dataclass_fields__"):
        return {name: _jsonable(getattr(value, name)) for name in value.__dataclass_fields__}
    if isinstance(value, torch.Tensor):
        return value.tolist()
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    return value


def _dump(path: Path, value: Any) -> None:
    path.write_text(json.dumps(_jsonable(value), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
