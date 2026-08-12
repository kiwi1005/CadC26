"""Checkpoint-gated learned initializer with an exact-safe analytic tail."""

from __future__ import annotations

from dataclasses import dataclass, fields, replace
import hashlib
import math
import os
from pathlib import Path
from typing import Any

import torch

from hcfp.analytic import (
    AnalyticResult,
    AnalyticConfig,
    CandidateTelemetry,
    select_device,
    solve as solve_analytic,
    solve_case_with_telemetry,
    solve_case_from_population_with_telemetry,
    to_official_placements,
)
from hcfp.case import FloorplanCase, from_official
from hcfp.btree import contact_aware_vertical_orders, decode_btree_logits
from hcfp.candidates import candidate_features
from hcfp.checkpoint import RUNTIME_NORMALIZATION, load_checkpoint
from hcfp.collective_runtime import CollectiveForceController
from hcfp.constraints.construction import connect_groups, construct_constraint_variants
from hcfp.constraints.raw_repair import repair_raw_constraints
from hcfp.dynamics import initialize_population
from hcfp.fallback import safe_fallback, safe_shelf
from hcfp.geometry import centers_from_xywh, initializer_anchor, xywh_from_state
from hcfp.model import soft_sequence_pair_relation_logits
from hcfp.projection import ComponentBDPConfig
from hcfp.projection_guidance import build_population_guidance
from hcfp.ranker_features import (
    RANKER_FEATURE_DIM,
    RANKER_FEATURE_VERSION,
    repair_aware_ranker_features,
)
from hcfp.topology import (
    adapt_preplaced_topology,
    anchor_safe_order_variants,
    copy_preplaced_targets,
    decode_sequence_pair,
    hard_permutation,
    pack_sequence_pair_with_anchors,
    relation_mask_from_rectangles,
)
from hcfp.treemap import exact_treemap_candidates
from hcfp.verify import (
    bbox_area,
    soft_violation_normalized,
    total_hpwl,
    verify_feasible,
)


Tensor = torch.Tensor


_OUTLINE_MIN_CONFIDENCE = 0.50
_OUTLINE_EPS = 1.0e-6

try:
    from hcfp.outline_inference import infer_outline_hypotheses
except ImportError:  # pragma: no cover - the optional runtime contact is fail-closed.
    infer_outline_hypotheses = None  # type: ignore[assignment]


@dataclass(frozen=True)
class LearnedResult:
    selected: Tensor
    used_checkpoint: bool
    checkpoint_hash: str | None
    failure_reason: str | None
    flow_steps: int = 0
    candidate_count: int = 0
    topology_seed_attempted: bool = False
    topology_seed_accepted: bool = False
    topology_seed_count: int = 0
    constraint_seed_attempted: bool = False
    constraint_seed_accepted: bool = False
    constraint_seed_count: int = 0
    collective_steps: int = 0
    collective_used: bool = False
    collective_calls: int = 0
    outline_variant_attempted: bool = False
    outline_variant_accepted: bool = False
    outline_variant_count: int = 0
    treemap_seed_attempted: bool = False
    treemap_seed_accepted: bool = False
    treemap_seed_count: int = 0
    btree_seed_attempted: bool = False
    btree_seed_accepted: bool = False
    btree_seed_count: int = 0


@dataclass(frozen=True)
class LearnedAnalysis:
    result: LearnedResult
    analytic: AnalyticResult


@dataclass(frozen=True)
class LearnedConfig:
    analytic: AnalyticConfig = AnalyticConfig()
    flow_steps: int = 0
    flow_fraction: float = 0.50
    flow_noise_scale: float = 1.0
    max_position_residual: float = 0.50
    max_aspect_residual: float = 1.0
    tail_topk: int | None = None
    seed: int | None = None
    topology_seeds: int = 0
    constraint_seeds: int = 0
    treemap_seeds: int = 0
    btree_seeds: int = 0
    collective_steps: int = 0
    ranker_selection_experiment: bool = False

    def __post_init__(self) -> None:
        if self.flow_steps < 0:
            raise ValueError("flow_steps must be non-negative")
        if not 0.0 <= self.flow_fraction <= 1.0:
            raise ValueError("flow_fraction must be in [0, 1]")
        if self.flow_noise_scale < 0.0:
            raise ValueError("flow_noise_scale must be non-negative")
        if self.max_position_residual <= 0.0 or self.max_aspect_residual <= 0.0:
            raise ValueError("flow residual limits must be positive")
        if self.tail_topk is not None and self.tail_topk <= 0:
            raise ValueError("tail_topk must be positive")
        if self.topology_seeds < 0:
            raise ValueError("topology_seeds must be non-negative")
        if self.constraint_seeds < 0:
            raise ValueError("constraint_seeds must be non-negative")
        if self.treemap_seeds < 0:
            raise ValueError("treemap_seeds must be non-negative")
        if self.btree_seeds < 0:
            raise ValueError("btree_seeds must be non-negative")
        if self.collective_steps < 0:
            raise ValueError("collective_steps must be non-negative")
        if self.constraint_seeds and not self.topology_seeds:
            raise ValueError("constraint_seeds require topology_seeds")
        if type(self.ranker_selection_experiment) is not bool:
            raise ValueError("ranker_selection_experiment must be boolean")


def effective_flow_steps(requested: int, checkpoint_metadata: dict[str, Any]) -> int:
    """Enable flow only when the checkpoint declares trained flow weights."""

    if requested < 0:
        raise ValueError("flow_steps must be non-negative")
    capabilities = checkpoint_metadata.get("capabilities", {})
    return requested if isinstance(capabilities, dict) and capabilities.get("flow") is True else 0


def effective_collective_steps(
    requested: int,
    checkpoint_metadata: dict[str, Any],
    model_config: Any,
) -> int:
    """Enable collective controls only for explicitly trained capable models."""

    if requested < 0:
        raise ValueError("collective_steps must be non-negative")
    capabilities = checkpoint_metadata.get("capabilities", {})
    trained_heads = checkpoint_metadata.get("trained_heads", [])
    collective_enabled = (
        bool(model_config.get("collective_enabled", False))
        if isinstance(model_config, dict)
        else bool(getattr(model_config, "collective_enabled", False))
    )
    if (
        isinstance(capabilities, dict)
        and capabilities.get("collective") is True
        and isinstance(trained_heads, (list, tuple))
        and "collective" in trained_heads
        and collective_enabled
    ):
        return requested
    return 0


def effective_tail_topk(
    requested: int | None,
    checkpoint_metadata: dict[str, Any],
) -> int | None:
    """Enable ranker pruning only when checkpoint metadata proves it is trained."""

    if requested is None:
        return None
    if requested <= 0:
        raise ValueError("tail_topk must be positive")
    capabilities = checkpoint_metadata.get("capabilities", {})
    trained_heads = checkpoint_metadata.get("trained_heads", [])
    if (
        isinstance(capabilities, dict)
        and capabilities.get("ranker") is True
        and isinstance(trained_heads, (list, tuple))
        and "ranker" in trained_heads
    ):
        return requested
    return None


def solve_case_with_checkpoint(
    case: FloorplanCase,
    checkpoint: str | Path,
    config: AnalyticConfig | LearnedConfig | None = None,
) -> LearnedResult:
    """Use learned population residuals only behind hash and verifier gates."""

    return analyze_case_with_checkpoint(case, checkpoint, config).result


def analyze_case_with_checkpoint(
    case: FloorplanCase,
    checkpoint: str | Path,
    config: AnalyticConfig | LearnedConfig | None = None,
) -> LearnedAnalysis:
    """Return the learned result plus per-candidate exact-tail telemetry."""

    learned_cfg = _learned_config(config)
    cfg = learned_cfg.analytic
    try:
        model, metadata = load_checkpoint(
            checkpoint,
            expected_normalization=RUNTIME_NORMALIZATION,
            map_location="cpu",
        )
        learned_cfg = replace(
            learned_cfg,
            flow_steps=effective_flow_steps(learned_cfg.flow_steps, metadata),
            tail_topk=effective_tail_topk(learned_cfg.tail_topk, metadata),
            collective_steps=effective_collective_steps(
                learned_cfg.collective_steps,
                metadata,
                model.config,
            ),
        )
        model = model.to(device=case.area.device).eval()
        topology_provenance: dict[str, object] = {}
        learned_population = _learned_population(
            case,
            model,
            learned_cfg,
            seed=(
                int(str(metadata["state_hash"])[:8], 16)
                if learned_cfg.seed is None
                else learned_cfg.seed
            ),
            provenance=topology_provenance,
        )
        if _needs_legacy_mib_challenger(case):
            legacy_provenance: dict[str, object] = {}
            legacy_population = _learned_population(
                case,
                model,
                learned_cfg,
                seed=(
                    int(str(metadata["state_hash"])[:8], 16)
                    if learned_cfg.seed is None
                    else learned_cfg.seed
                ),
                provenance=legacy_provenance,
                enforce_mib=False,
            )
            learned_population, topology_provenance = (
                _merge_legacy_mib_challenger(
                    learned_population,
                    topology_provenance,
                    legacy_population,
                    legacy_provenance,
                )
            )
        topology_count = int(
            topology_provenance.get("topology_seed_count", 0)
        )
        constraint_count = int(
            topology_provenance.get("constraint_seed_count", 0)
        )
        residual_count = (
            int(learned_population.shape[0]) - topology_count - constraint_count
        )
        population_guidance = (
            build_population_guidance(
                case,
                topology_provenance,
                residual_count=residual_count,
                constraint_count=constraint_count,
                topology_count=topology_count,
            )
            if topology_count
            else None
        )
        projection_guidance = (
            population_guidance if cfg.component_bdp.enabled and topology_count else None
        )
        # Keep the established analytic comparator on v0. Q4 applies only to
        # provenance-bearing learned structure and remains Pareto-guarded.
        analytic_analysis = solve_case_with_telemetry(
            case,
            replace(cfg, component_bdp=ComponentBDPConfig()),
        )
        force_controller = None
        if learned_cfg.collective_steps:
            device_type = "cuda" if case.area.is_cuda else "cpu"
            with torch.inference_mode(), torch.autocast(
                device_type=device_type,
                dtype=torch.bfloat16,
                enabled=model.config.compute_dtype == "bfloat16",
            ):
                static_embedding = model.encoder(case).float()
            force_controller = CollectiveForceController.from_guidance(
                model,
                static_embedding,
                population_guidance,
            )
        dynamics_cfg = replace(cfg.dynamics, population=int(learned_population.shape[0]))
        if learned_cfg.collective_steps:
            dynamics_cfg = replace(dynamics_cfg, steps=learned_cfg.collective_steps)
        learned_tail_cfg = replace(
            cfg,
            dynamics=dynamics_cfg,
            component_bdp=(
                cfg.component_bdp
                if projection_guidance is not None
                else ComponentBDPConfig()
            ),
        )
        learned_analysis = solve_case_from_population_with_telemetry(
            case,
            learned_population,
            learned_tail_cfg,
            projection_guidance=projection_guidance,
            force_controller=force_controller,
        )
        analysis = _merge_tail_analyses(
            case,
            analytic_analysis,
            learned_analysis,
            topology_provenance=topology_provenance,
        )
        analysis = _attach_ranker_shadow_snapshot(
            case,
            analysis,
            model=model,
            metadata=metadata,
            analytic_count=cfg.dynamics.population,
            learned_count=int(learned_population.shape[0]),
            residual_count=residual_count,
            constraint_count=constraint_count,
            topology_count=topology_count,
        )
        candidate_count = cfg.dynamics.population + int(learned_population.shape[0])
        return LearnedAnalysis(
            LearnedResult(
                selected=analysis.selected,
                used_checkpoint=True,
                checkpoint_hash=str(metadata["state_hash"]),
                failure_reason=None,
                flow_steps=learned_cfg.flow_steps,
                candidate_count=candidate_count,
                topology_seed_attempted=bool(
                    topology_provenance.get("topology_seed_attempted", False)
                ),
                topology_seed_accepted=bool(
                    topology_provenance.get("topology_seed_accepted", False)
                ),
                topology_seed_count=int(
                    topology_provenance.get("topology_seed_count", 0)
                ),
                constraint_seed_attempted=bool(
                    topology_provenance.get("constraint_seed_attempted", False)
                ),
                constraint_seed_accepted=bool(
                    topology_provenance.get("constraint_seed_accepted", False)
                ),
                constraint_seed_count=int(
                    topology_provenance.get("constraint_seed_count", 0)
                ),
                outline_variant_attempted=bool(
                    topology_provenance.get("outline_variant_attempted", False)
                ),
                outline_variant_accepted=bool(
                    topology_provenance.get("outline_variant_accepted", False)
                ),
                outline_variant_count=int(
                    topology_provenance.get("outline_variant_count", 0)
                ),
                treemap_seed_attempted=bool(
                    topology_provenance.get("treemap_seed_attempted", False)
                ),
                treemap_seed_accepted=bool(
                    topology_provenance.get("treemap_seed_accepted", False)
                ),
                treemap_seed_count=int(
                    topology_provenance.get("treemap_seed_count", 0)
                ),
                btree_seed_attempted=bool(
                    topology_provenance.get("btree_seed_attempted", False)
                ),
                btree_seed_accepted=bool(
                    topology_provenance.get("btree_seed_accepted", False)
                ),
                btree_seed_count=int(
                    topology_provenance.get("btree_seed_count", 0)
                ),
                collective_steps=learned_cfg.collective_steps,
                collective_used=force_controller is not None,
                collective_calls=(
                    0 if force_controller is None else int(force_controller.calls)
                ),
            ),
            analysis,
        )
    except Exception as exc:
        analysis = solve_case_with_telemetry(case, cfg)
        return LearnedAnalysis(
            LearnedResult(
                selected=analysis.selected,
                used_checkpoint=False,
                checkpoint_hash=None,
                failure_reason=f"{type(exc).__name__}: {exc}",
            ),
            analysis,
        )


def solve(
    source: Any,
    *,
    checkpoint: str | Path,
    config: AnalyticConfig | LearnedConfig | None = None,
    device: str | torch.device | None = None,
    require_checkpoint: bool = False,
) -> list[tuple[float, float, float, float]]:
    """Official-contract learned lane; any checkpoint failure uses analytic P1."""

    try:
        selected_device = select_device(device)
        case = from_official(
            int(_field(source, "block_count")),
            _field(source, "area_targets"),
            _field(source, "b2b_connectivity"),
            _field(source, "p2b_connectivity"),
            _field(source, "pins_pos"),
            _field(source, "constraints"),
            _field(source, "target_positions"),
            device=selected_device,
        )
        analysis = analyze_case_with_checkpoint(case, checkpoint, config)
        result = analysis.result
        if require_checkpoint and not result.used_checkpoint:
            raise RuntimeError(result.failure_reason or "checkpoint was not used")
        return select_official_from_analysis(
            source,
            case,
            analysis,
            config=config,
            device=device,
        )
    except Exception:
        if require_checkpoint:
            raise
        return solve_analytic(source, _learned_config(config).analytic, device=device)


def select_official_from_analysis(
    source: Any,
    case: FloorplanCase,
    analysis: LearnedAnalysis,
    *,
    config: AnalyticConfig | LearnedConfig | None = None,
    device: str | torch.device | None = None,
) -> list[tuple[float, float, float, float]]:
    """Apply the exact runtime raw-selection chain to an existing analysis."""

    snapshot = getattr(analysis.analytic, "incumbent_snapshot", {})
    selected_source = snapshot.get("exact_source")
    placements = to_official_placements(source, case, analysis.result.selected)
    placements = _repair_constraint_candidate(
        source,
        case,
        placements,
        snapshot,
        selected_source,
    )
    if verify_feasible(source, placements):
        placements = _raw_constraint_pareto_guard(
            source,
            case,
            analysis,
            placements,
        )
        placements = _raw_analytic_pareto_guard(
            source,
            case,
            analysis,
            placements,
        )
        placements = _post_tail_group_repair(source, case, placements)
        placements = _legacy_mib_challenger_guard(
            source,
            case,
            analysis,
            placements,
        )
        repaired_incumbent = placements
        placements = _raw_treemap_proxy_guard(
            source,
            case,
            analysis,
            placements,
        )
        if placements is not repaired_incumbent:
            placements = _post_tail_group_repair(source, case, placements)
            placements = _legacy_mib_challenger_guard(
                source,
                case,
                analysis,
                placements,
            )
        if _truthy_env(os.environ.get("HCFP_CANDIDATE_FUNNEL_PROXY")):
            placements = _candidate_funnel_proxy_guard(
                source,
                case,
                analysis,
                placements,
            )
        _record_ranker_selection_counterfactual(
            source,
            case,
            analysis,
            placements,
            enabled=_learned_config(config).ranker_selection_experiment,
        )
        return placements
    telemetry = analysis.analytic.telemetry
    candidates = analysis.analytic.projected_candidates.detach().to(
        device="cpu", dtype=torch.float32
    )
    hard_feasible = telemetry.hard_feasible.detach().to(device="cpu", dtype=torch.bool)
    soft_violation = telemetry.soft_violation.detach().to(
        device="cpu", dtype=torch.float32
    )
    quality = (
        (telemetry.bbox_area + 0.05 * telemetry.hpwl)
        .detach()
        .to(device="cpu", dtype=torch.float32)
    )
    order = sorted(
        (index for index in range(len(candidates)) if bool(hard_feasible[index])),
        key=lambda index: (float(soft_violation[index]), float(quality[index]), index),
    )
    for index in order:
        candidate = to_official_placements(source, case, candidates[index])
        candidate = _repair_constraint_candidate(
            source,
            case,
            candidate,
            snapshot,
            f"candidate_{index}",
        )
        if verify_feasible(source, candidate):
            if _truthy_env(os.environ.get("HCFP_CANDIDATE_FUNNEL_PROXY")):
                candidate = _candidate_funnel_proxy_guard(
                    source,
                    case,
                    analysis,
                    candidate,
                )
            if index == 0:
                break
            _record_ranker_selection_counterfactual(
                source,
                case,
                analysis,
                candidate,
                enabled=_learned_config(config).ranker_selection_experiment,
            )
            return candidate
    analytic = solve_analytic(source, _learned_config(config).analytic, device=device)
    if verify_feasible(source, analytic):
        return analytic
    fallback = safe_fallback(source)
    return [tuple(float(value) for value in row) for row in fallback]


def _field(source: Any, name: str) -> Any:
    if isinstance(source, dict):
        return source.get(name)
    return getattr(source, name)


def _learned_config(config: AnalyticConfig | LearnedConfig | None) -> LearnedConfig:
    if config is None:
        tail = os.environ.get("HCFP_TAIL_TOPK")
        seed = os.environ.get("HCFP_FLOW_SEED")
        return LearnedConfig(
            flow_steps=int(os.environ.get("HCFP_FLOW_STEPS", "0")),
            tail_topk=int(tail) if tail else None,
            seed=int(seed) if seed is not None else None,
            topology_seeds=int(os.environ.get("HCFP_TOPOLOGY_SEEDS", "0")),
            constraint_seeds=int(os.environ.get("HCFP_CONSTRAINT_SEEDS", "0")),
            treemap_seeds=int(os.environ.get("HCFP_TREEMAP_SEEDS", "0")),
            btree_seeds=int(os.environ.get("HCFP_BTREE_SEEDS", "0")),
            collective_steps=int(os.environ.get("HCFP_COLLECTIVE_STEPS", "0")),
            ranker_selection_experiment=_truthy_env(
                os.environ.get("HCFP_RANKER_SELECTION_EXPERIMENT")
            ),
        )
    if isinstance(config, AnalyticConfig):
        return LearnedConfig(analytic=config)
    return config


def _truthy_env(value: str | None) -> bool:
    return value is not None and value.strip().lower() in {"1", "true", "yes", "on"}


def _merge_tail_analyses(
    case: FloorplanCase,
    analytic: AnalyticResult,
    learned: AnalyticResult,
    *,
    topology_provenance: dict[str, object] | None = None,
) -> AnalyticResult:
    analytic_count = (analytic.projected_candidates.shape[0] - 1) // 2
    learned_count = (learned.projected_candidates.shape[0] - 1) // 2

    def merge_tensor(first: Tensor, second: Tensor) -> Tensor:
        return torch.cat(
            (
                first[:1],
                first[1 : 1 + analytic_count],
                second[1 : 1 + learned_count],
                first[1 + analytic_count :],
                second[1 + learned_count :],
            ),
            dim=0,
        )

    def merge_tuple(first: tuple[str, ...], second: tuple[str, ...]) -> tuple[str, ...]:
        return (
            first[:1]
            + first[1 : 1 + analytic_count]
            + second[1 : 1 + learned_count]
            + first[1 + analytic_count :]
            + second[1 + learned_count :]
        )

    projected = merge_tensor(
        analytic.projected_candidates, learned.projected_candidates
    )
    raw_candidates = merge_tensor(
        analytic.raw_candidates, learned.raw_candidates
    )
    telemetry_values = {}
    for field in fields(CandidateTelemetry):
        first = getattr(analytic.telemetry, field.name)
        second = getattr(learned.telemetry, field.name)
        telemetry_values[field.name] = (
            merge_tuple(first, second)
            if field.name in {"projection_failure_reasons", "component_proposal_rollback_reason"}
            else merge_tensor(first, second)
        )
    telemetry = CandidateTelemetry(**telemetry_values)
    ok_mask = telemetry.projection_ok.detach().to(device="cpu", dtype=torch.bool)
    hard_mask = telemetry.hard_feasible.detach().to(device="cpu", dtype=torch.bool)
    exact_mask = hard_mask & ok_mask
    exact_mask[0] = True
    area = telemetry.bbox_area.detach().to(device="cpu", dtype=torch.float64)
    hpwl = telemetry.hpwl.detach().to(device="cpu", dtype=torch.float64)
    soft = telemetry.soft_violation.detach().to(device="cpu", dtype=torch.float64)
    treemap_indices: set[int] = set()
    if topology_provenance and bool(
        topology_provenance.get("treemap_seed_accepted", False)
    ):
        for record in tuple(topology_provenance.get("treemap_seed_records", ())):
            if not isinstance(record, dict):
                continue
            try:
                learned_index = int(record["residual_index"])
            except (KeyError, TypeError, ValueError):
                continue
            if 0 <= learned_index < learned_count:
                treemap_indices.add(1 + analytic_count + learned_index)
                treemap_indices.add(
                    1 + analytic_count + learned_count + analytic_count + learned_index
                )

    def best_index(mask: Tensor) -> int | None:
        indices = [
            index
            for index in torch.nonzero(mask, as_tuple=False).reshape(-1).tolist()
            if index not in treemap_indices
        ]
        return (
            min(
                indices,
                key=lambda index: (
                    float(soft[index]),
                    float(area[index]) + 0.05 * float(hpwl[index]),
                    index,
                ),
            )
            if indices
            else None
        )

    fast_index = best_index(ok_mask)
    exact_index = best_index(exact_mask)
    if exact_index is None:
        raise RuntimeError("merged tail lost the safe fallback")
    rejections = {
        reason: count
        for reason, count in (
            ("fast_infeasible", int((~ok_mask).sum().item())),
            ("exact_infeasible", int((ok_mask & ~hard_mask).sum().item())),
        )
        if count
    }
    snapshot = {
        "safe_source": "fallback",
        "fast_source": f"candidate_{fast_index}" if fast_index is not None else None,
        "exact_source": "fallback" if exact_index == 0 else f"candidate_{exact_index}",
        "analytic_exact_source": _merged_analytic_source(
            analytic.incumbent_snapshot.get("exact_source"),
            analytic_count,
            learned_count,
        ),
        "analytic_fast_source": _merged_analytic_source(
            analytic.incumbent_snapshot.get("fast_source"),
            analytic_count,
            learned_count,
        ),
        "rejections": rejections,
    }
    initial_start = 1 + analytic_count
    final_start = 1 + analytic_count + learned_count + analytic_count
    if topology_provenance:
        snapshot.update(topology_provenance)
    if topology_provenance and bool(
        topology_provenance.get("topology_seed_attempted", False)
    ):
        topology_count = min(
            int(topology_provenance.get("topology_seed_count", 0)),
            learned_count,
        )
        topology_offset = learned_count - topology_count
        seed_orders = tuple(topology_provenance.get("topology_seed_orders", ()))
        if len(seed_orders) != topology_count:
            seed_orders = tuple({} for _ in range(topology_count))
        candidates, topology_stale_sources = _seed_stage_records(
            seed_orders,
            raw_candidates,
            initial_start=initial_start + topology_offset,
            final_start=final_start + topology_offset,
            count=topology_count,
            candidate_type="topology",
        )
        snapshot["topology_seed_sources"] = tuple(
            str(candidate["source"]) for candidate in candidates
        )
        snapshot["topology_seed_source_types"] = tuple(
            str(candidate["stage"]) for candidate in candidates
        )
        snapshot["topology_seed_provenance"] = candidates
        if topology_stale_sources:
            snapshot["topology_seed_stale_sources"] = tuple(topology_stale_sources)
        constraint_count = min(
            int(topology_provenance.get("constraint_seed_count", 0)),
            learned_count - topology_count,
        )
        if constraint_count:
            residual_count = learned_count - topology_count - constraint_count
            constraint_records = tuple(
                topology_provenance.get("constraint_seed_records", ())
            )
            if len(constraint_records) != constraint_count:
                constraint_records = tuple({} for _ in range(constraint_count))
            constraint_candidates, stale_sources = _seed_stage_records(
                constraint_records,
                raw_candidates,
                initial_start=initial_start + residual_count,
                final_start=final_start + residual_count,
                count=constraint_count,
                candidate_type="constraint",
            )
            snapshot["constraint_seed_sources"] = tuple(
                str(candidate["source"]) for candidate in constraint_candidates
            )
            snapshot["constraint_seed_provenance"] = constraint_candidates
            if stale_sources:
                snapshot["constraint_seed_stale_sources"] = tuple(stale_sources)
    if topology_provenance and bool(
        topology_provenance.get("treemap_seed_accepted", False)
    ):
        treemap_candidates, stale_sources = _residual_seed_stage_records(
            tuple(topology_provenance.get("treemap_seed_records", ())),
            raw_candidates,
            initial_start=initial_start,
            final_start=final_start,
            learned_count=learned_count,
            candidate_type="treemap",
        )
        snapshot["treemap_seed_sources"] = tuple(
            str(candidate["source"]) for candidate in treemap_candidates
        )
        snapshot["treemap_seed_provenance"] = treemap_candidates
        if stale_sources:
            snapshot["treemap_seed_stale_sources"] = tuple(stale_sources)
    if topology_provenance and bool(
        topology_provenance.get("btree_seed_accepted", False)
    ):
        btree_candidates, stale_sources = _residual_seed_stage_records(
            tuple(topology_provenance.get("btree_seed_records", ())),
            raw_candidates,
            initial_start=initial_start,
            final_start=final_start,
            learned_count=learned_count,
            candidate_type="btree",
        )
        snapshot["btree_seed_sources"] = tuple(
            str(candidate["source"]) for candidate in btree_candidates
        )
        snapshot["btree_seed_provenance"] = btree_candidates
        if stale_sources:
            snapshot["btree_seed_stale_sources"] = tuple(stale_sources)
    status = analytic.projection_status
    if learned.projection_status != status:
        status = f"analytic={status};learned={learned.projection_status}"
    return AnalyticResult(
        selected=(
            analytic.selected
            if exact_index == 0
            else projected[exact_index].detach().to(device="cpu", dtype=torch.float32)
        ),
        raw_candidates=raw_candidates,
        projected_candidates=projected,
        telemetry=telemetry,
        energy_history=_merge_energy_history(
            analytic.energy_history,
            learned.energy_history,
        ),
        projection_status=status,
        incumbent_snapshot=snapshot,
    )


def _seed_stage_records(
    records: tuple[object, ...],
    raw_candidates: Tensor,
    *,
    initial_start: int,
    final_start: int,
    count: int,
    candidate_type: str,
) -> tuple[tuple[dict[str, object], ...], tuple[str, ...]]:
    candidates: list[dict[str, object]] = []
    stale_sources: list[str] = []
    for index in range(count):
        record = dict(records[index]) if index < len(records) else {}
        initial_index = initial_start + index
        final_index = final_start + index
        initial_source = f"candidate_{initial_index}"
        initial_hash = _tensor_sha256(raw_candidates[initial_index])
        expected_hash = record.get("candidate_sha256")
        if expected_hash is None and candidate_type != "topology":
            stale_sources.append(initial_source)
            continue
        if expected_hash is not None and expected_hash != initial_hash:
            stale_sources.append(initial_source)
            continue
        record["candidate_sha256"] = initial_hash
        candidates.append(
            {
                **record,
                "source": initial_source,
                "candidate_type": candidate_type,
                "stage": "initial",
            }
        )
        final_source = f"candidate_{final_index}"
        final_hash = _tensor_sha256(raw_candidates[final_index])
        transform = "identity" if final_hash == initial_hash else "population_relaxation"
        candidates.append(
            {
                **record,
                "source": final_source,
                "candidate_type": candidate_type,
                "stage": "post_relax",
                "transform": transform,
                "parent_candidate_sha256": initial_hash,
                "candidate_sha256": final_hash,
            }
        )
    return tuple(candidates), tuple(stale_sources)


def _residual_seed_stage_records(
    records: tuple[object, ...],
    raw_candidates: Tensor,
    *,
    initial_start: int,
    final_start: int,
    learned_count: int,
    candidate_type: str,
) -> tuple[tuple[dict[str, object], ...], tuple[str, ...]]:
    candidates: list[dict[str, object]] = []
    stale_sources: list[str] = []
    for raw_record in records:
        if not isinstance(raw_record, dict):
            continue
        record = dict(raw_record)
        try:
            residual_index = int(record["residual_index"])
        except (KeyError, TypeError, ValueError):
            continue
        if not 0 <= residual_index < learned_count:
            continue
        initial_index = initial_start + residual_index
        final_index = final_start + residual_index
        initial_source = f"candidate_{initial_index}"
        initial_hash = _tensor_sha256(raw_candidates[initial_index])
        if record.get("candidate_sha256") != initial_hash:
            stale_sources.append(initial_source)
            continue
        candidates.append(
            {
                **record,
                "source": initial_source,
                "candidate_type": candidate_type,
                "stage": "initial",
            }
        )
        final_source = f"candidate_{final_index}"
        final_hash = _tensor_sha256(raw_candidates[final_index])
        candidates.append(
            {
                **record,
                "source": final_source,
                "candidate_type": candidate_type,
                "stage": "post_relax",
                "transform": "identity" if final_hash == initial_hash else "population_relaxation",
                "parent_candidate_sha256": initial_hash,
                "candidate_sha256": final_hash,
            }
        )
    return tuple(candidates), tuple(stale_sources)


def _attach_ranker_shadow_snapshot(
    case: FloorplanCase,
    analysis: AnalyticResult,
    *,
    model: Any,
    metadata: dict[str, Any],
    analytic_count: int,
    learned_count: int,
    residual_count: int,
    constraint_count: int,
    topology_count: int,
) -> AnalyticResult:
    """Record a trained ranker's learned-initial ordering without selecting it."""

    snapshot = dict(analysis.incumbent_snapshot)
    skip_reason = _ranker_shadow_contract(model, metadata)
    if skip_reason is not None:
        snapshot["ranker_shadow_skipped_reason"] = skip_reason
        return replace(analysis, incumbent_snapshot=snapshot)
    if learned_count <= 0:
        snapshot["ranker_shadow_skipped_reason"] = "empty_learned_population"
        return replace(analysis, incumbent_snapshot=snapshot)
    if residual_count + constraint_count + topology_count != learned_count:
        snapshot["ranker_shadow_skipped_reason"] = "candidate_kind_count_mismatch"
        return replace(analysis, incumbent_snapshot=snapshot)

    initial_start = 1 + analytic_count
    initial_stop = initial_start + learned_count
    try:
        raw = analysis.raw_candidates[initial_start:initial_stop]
        post_bdp = analysis.projected_candidates[initial_start:initial_stop]
        if raw.shape[0] != learned_count or post_bdp.shape[0] != learned_count:
            raise ValueError("learned initial slice does not align with merged tail")
        kinds = _ranker_candidate_kinds(
            snapshot,
            residual_count=residual_count,
            constraint_count=constraint_count,
            topology_count=topology_count,
            initial_start=initial_start,
        )
        features = repair_aware_ranker_features(
            case,
            raw,
            post_bdp,
            safe_shelf(case).to(device=raw.device, dtype=raw.dtype),
            kinds,
            "initial",
        )
        eligible = (
            analysis.telemetry.hard_feasible[initial_start:initial_stop]
            & analysis.telemetry.projection_ok[initial_start:initial_stop]
        ).detach().to(device="cpu", dtype=torch.bool)
        embedding = features.new_empty((0, 0))
        with torch.inference_mode():
            scores = (
                model.ranker(embedding, learned_count, features)
                .detach()
                .to(device="cpu", dtype=torch.float32)
            )
        if scores.shape != (learned_count,):
            raise ValueError("ranker score shape does not align with learned slice")
        if not bool(torch.isfinite(scores).all()):
            raise ValueError("ranker shadow scores must be finite")
        ranked = sorted(
            (
                (float(scores[index]), initial_start + index, kinds[index])
                for index in range(learned_count)
                if bool(eligible[index])
            ),
            key=lambda item: (item[0], item[1]),
        )
        empty_reason = "no_exact_eligible_candidates" if not ranked else None
        snapshot.update(
            {
                "ranker_shadow_stage": "initial",
                "ranker_shadow_source": "merged_learned_initial",
                "ranker_shadow_feature_version": RANKER_FEATURE_VERSION,
                "ranker_shadow_feature_dim": RANKER_FEATURE_DIM,
                "ranker_shadow_candidate_count": learned_count,
                "ranker_shadow_eligible_count": len(ranked),
                "ranker_shadow_empty_reason": empty_reason,
                "ranker_shadow_candidate_kinds": kinds,
                "ranker_shadow_top4": tuple(
                    {
                        "source": f"candidate_{source_index}",
                        "score": score,
                        "kind": kind,
                    }
                    for score, source_index, kind in ranked[:4]
                ),
            }
        )
    except Exception as exc:
        snapshot["ranker_shadow_failure_reason"] = f"{type(exc).__name__}: {exc}"
    return replace(analysis, incumbent_snapshot=snapshot)


def _ranker_shadow_contract(model: Any, metadata: dict[str, Any]) -> str | None:
    trained_heads = metadata.get("trained_heads", ())
    if not (
        isinstance(trained_heads, (list, tuple))
        and "ranker" in trained_heads
    ):
        return "ranker_not_trained"
    config = getattr(model, "config", {})
    try:
        if int(_config_value(config, "candidate_metric_dim")) != RANKER_FEATURE_DIM:
            return "ranker_feature_dim_mismatch"
        if (
            str(_config_value(config, "ranker_feature_version"))
            != RANKER_FEATURE_VERSION
        ):
            return "ranker_feature_version_mismatch"
        if bool(_config_value(config, "ranker_use_scene_embedding")):
            return "ranker_uses_scene_embedding"
    except (AttributeError, KeyError, TypeError, ValueError):
        return "ranker_config_incompatible"
    return None


def _config_value(config: Any, name: str) -> Any:
    if isinstance(config, dict):
        return config[name]
    return getattr(config, name)


def _learned_candidate_kinds(
    *,
    residual_count: int,
    constraint_count: int,
    topology_count: int,
) -> tuple[str, ...]:
    return (
        ("learned",) * residual_count
        + ("constraint",) * constraint_count
        + ("topology",) * topology_count
    )


def _ranker_candidate_kinds(
    snapshot: dict[str, object],
    *,
    residual_count: int,
    constraint_count: int,
    topology_count: int,
    initial_start: int,
) -> tuple[str, ...]:
    kinds = list(
        _learned_candidate_kinds(
            residual_count=residual_count,
            constraint_count=constraint_count,
            topology_count=topology_count,
        )
    )
    for candidate_type in ("treemap", "btree"):
        records = snapshot.get(f"{candidate_type}_seed_provenance", ())
        if not isinstance(records, tuple):
            continue
        for record in records:
            if not isinstance(record, dict) or record.get("stage") != "initial":
                continue
            index = _candidate_index(str(record.get("source", "")))
            relative = -1 if index is None else index - initial_start
            if 0 <= relative < len(kinds):
                kinds[relative] = candidate_type
    return tuple(kinds)


def _merge_energy_history(first: Tensor, second: Tensor) -> Tensor:
    if first.shape[1:] == second.shape[1:]:
        return torch.cat((first, second), dim=0)
    if first.ndim != 3 or second.ndim != 3 or first.shape[2] != second.shape[2]:
        raise ValueError("energy history shape mismatch")
    steps = max(int(first.shape[1]), int(second.shape[1]))

    def padded(value: Tensor) -> Tensor:
        if int(value.shape[1]) == steps:
            return value
        out = value.new_zeros((value.shape[0], steps, value.shape[2]))
        if value.shape[1]:
            out[:] = value[:, -1:, :]
        out[:, : value.shape[1], :] = value
        return out

    return torch.cat((padded(first), padded(second)), dim=0)


def _record_ranker_selection_counterfactual(
    source: Any,
    case: FloorplanCase,
    analysis: LearnedAnalysis,
    current: list[tuple[float, float, float, float]],
    *,
    enabled: bool,
) -> None:
    """Evaluate shadow top-k as an offline counterfactual without selecting it."""

    if not enabled:
        return
    snapshot = getattr(analysis.analytic, "incumbent_snapshot", {})
    if not isinstance(snapshot, dict):
        return
    snapshot["ranker_selection_experiment_mode"] = "counterfactual_only"
    top4 = snapshot.get("ranker_shadow_top4", ())
    if not isinstance(top4, tuple) or not top4:
        rejection = (
            "no_exact_eligible_ranker_candidates"
            if int(snapshot.get("ranker_shadow_eligible_count", -1)) == 0
            and snapshot.get("ranker_shadow_empty_reason")
            == "no_exact_eligible_candidates"
            else "missing_shadow_top4"
        )
        snapshot["ranker_selection_counterfactual"] = {
            "would_accept": False,
            "rejection_reason": rejection,
        }
        return
    try:
        current_metrics = _raw_quality(source, case, current)
    except (TypeError, ValueError) as exc:
        snapshot["ranker_selection_counterfactual"] = {
            "would_accept": False,
            "rejection_reason": f"current_quality_unavailable:{type(exc).__name__}",
        }
        return

    projected = analysis.analytic.projected_candidates
    evaluated: list[dict[str, object]] = []
    for rank, row in enumerate(top4):
        record = _ranker_counterfactual_record(row, rank)
        if record.get("rejection_reason") is not None:
            evaluated.append(record)
            continue
        source_name = str(record["source"])
        index = _candidate_index(source_name)
        if index is None or not 0 <= index < projected.shape[0]:
            record["rejection_reason"] = "invalid_source"
            evaluated.append(record)
            continue
        try:
            placement = to_official_placements(
                source,
                case,
                projected[index].detach().to(device="cpu", dtype=torch.float32),
            )
            placement = _repair_constraint_candidate(
                source,
                case,
                placement,
                snapshot,
                source_name,
            )
            if not verify_feasible(source, placement):
                record["rejection_reason"] = "hard_infeasible"
                evaluated.append(record)
                continue
            metrics = _raw_quality(source, case, placement)
        except (TypeError, ValueError) as exc:
            record["rejection_reason"] = f"evaluation_failed:{type(exc).__name__}"
            evaluated.append(record)
            continue
        record["metrics"] = metrics
        if not _dominates(metrics, current_metrics):
            record["rejection_reason"] = "not_pareto_dominating"
            evaluated.append(record)
            continue
        record["rejection_reason"] = None
        evaluated.append(record)
        snapshot["ranker_selection_counterfactual"] = {
            "would_accept": True,
            "source": source_name,
            "shadow_rank": rank,
            "metrics": metrics,
            "current_metrics": current_metrics,
            "rejection_reason": None,
        }
        snapshot["ranker_selection_evaluated_top4"] = tuple(evaluated)
        return

    snapshot["ranker_selection_counterfactual"] = {
        "would_accept": False,
        "rejection_reason": str(evaluated[0]["rejection_reason"])
        if evaluated
        else "empty_shadow_top4",
        "current_metrics": current_metrics,
    }
    snapshot["ranker_selection_evaluated_top4"] = tuple(evaluated)


def _ranker_counterfactual_record(row: object, rank: int) -> dict[str, object]:
    if not isinstance(row, dict):
        return {"shadow_rank": rank, "source": "", "rejection_reason": "malformed_row"}
    try:
        score = float(row.get("score", 0.0))
    except (TypeError, ValueError):
        score = 0.0
    return {
        "shadow_rank": rank,
        "source": str(row.get("source", "")),
        "shadow_score": score,
        "kind": str(row.get("kind", "")),
    }


def _raw_analytic_pareto_guard(
    source: Any,
    case: FloorplanCase,
    analysis: LearnedAnalysis,
    current: list[tuple[float, float, float, float]],
) -> list[tuple[float, float, float, float]]:
    """Admit only raw-feasible analytic candidates that dominate the incumbent."""

    snapshot = getattr(analysis.analytic, "incumbent_snapshot", {})
    selected_index = _candidate_index(snapshot.get("exact_source"))
    projected = analysis.analytic.projected_candidates
    analytic_indices = tuple(
        dict.fromkeys(
            index
            for index in (
                _candidate_index(snapshot.get("analytic_exact_source")),
                _candidate_index(snapshot.get("analytic_fast_source")),
            )
            if index is not None and index != selected_index
        )
    )
    if selected_index is None or not 0 <= selected_index < projected.shape[0]:
        return current
    analytic_indices = tuple(
        index for index in analytic_indices if 0 <= index < projected.shape[0]
    )
    if not analytic_indices:
        return current
    try:
        current_metrics = _raw_quality(source, case, current)
    except (TypeError, ValueError):
        return current

    admitted = []
    for index in analytic_indices:
        protected = (
            analysis.analytic.raw_candidates[0] if index == 0 else projected[index]
        )
        try:
            placement = to_official_placements(
                source,
                case,
                protected.detach().to(device="cpu", dtype=torch.float32),
            )
            if not verify_feasible(source, placement):
                continue
            candidate_metrics = _raw_quality(source, case, placement)
        except (TypeError, ValueError):
            continue
        if _dominates(candidate_metrics, current_metrics):
            admitted.append((candidate_metrics, index, placement))
    return (
        min(
            admitted,
            key=lambda item: (
                item[0][0],
                item[0][1] + 0.05 * item[0][2],
                item[1],
            ),
        )[2]
        if admitted
        else current
    )


def _raw_treemap_proxy_guard(
    source: Any,
    case: FloorplanCase,
    analysis: LearnedAnalysis,
    current: list[tuple[float, float, float, float]],
) -> list[tuple[float, float, float, float]]:
    """Admit dense treemaps when they improve the area/soft proxy without HPWL loss."""

    snapshot = getattr(analysis.analytic, "incumbent_snapshot", {})
    records = tuple(snapshot.get("treemap_seed_provenance", ()))
    if not records:
        return current
    try:
        current_metrics = _raw_quality(source, case, current)
    except (TypeError, ValueError):
        return current
    current_proxy = _treemap_proxy_score(case, current_metrics)
    raw_candidates = getattr(
        analysis.analytic,
        "raw_candidates",
        analysis.analytic.projected_candidates,
    )
    admitted = []
    for record in records:
        if not isinstance(record, dict) or record.get("stage") != "initial":
            continue
        index = _candidate_index(record.get("source"))
        if index is None or not 0 <= index < raw_candidates.shape[0]:
            continue
        placement = to_official_placements(
            source,
            case,
            raw_candidates[index].detach().to(device="cpu", dtype=torch.float32),
        )
        if not verify_feasible(source, placement):
            continue
        try:
            metrics = _raw_quality(source, case, placement)
        except (TypeError, ValueError):
            continue
        hpwl_tolerance = 1.0e-6 * max(1.0, current_metrics[2])
        proxy = _treemap_proxy_score(case, metrics)
        if metrics[2] <= current_metrics[2] + hpwl_tolerance and proxy < current_proxy - 1.0e-6:
            admitted.append((proxy, metrics[2], metrics[1], index, placement))
    return min(admitted, key=lambda item: item[:4])[4] if admitted else current


def _treemap_proxy_score(
    case: FloorplanCase,
    metrics: tuple[float, float, float],
) -> float:
    soft, area, _hpwl = metrics
    target_area = max(case.scale * case.scale, torch.finfo(torch.float64).eps)
    area_excess = max(0.0, area / target_area - 1.0)
    return math.log1p(0.5 * area_excess) + 2.0 * soft


def _candidate_funnel_proxy_guard(
    source: Any,
    case: FloorplanCase,
    analysis: LearnedAnalysis,
    current: list[tuple[float, float, float, float]],
) -> list[tuple[float, float, float, float]]:
    """Try provenance candidates after repair using a relative soft/area/HPWL proxy."""

    snapshot = getattr(analysis.analytic, "incumbent_snapshot", {})
    candidates: list[tuple[int, str, Tensor]] = []
    raw = analysis.analytic.raw_candidates
    projected = analysis.analytic.projected_candidates
    for name, boxes in (
        ("constraint_seed_provenance", projected),
        ("treemap_seed_provenance", raw),
        ("btree_seed_provenance", raw),
    ):
        records = snapshot.get(name, ())
        for record in records if isinstance(records, (tuple, list)) else ():
            if not isinstance(record, dict) or record.get("stage") != "initial":
                continue
            index = _candidate_index(record.get("source"))
            if index is not None and 0 <= index < boxes.shape[0]:
                candidates.append((index, str(record.get("source")), boxes[index]))
    if not candidates:
        return current
    try:
        current_metrics = _raw_quality(source, case, current)
    except (TypeError, ValueError):
        return current
    admitted = []
    audit_records = []
    for index, candidate_source, candidate in candidates:
        placement = to_official_placements(
            source,
            case,
            candidate.detach().to(device="cpu", dtype=torch.float32),
        )
        placement = _repair_constraint_candidate(
            source,
            case,
            placement,
            snapshot,
            candidate_source,
        )
        placement = _post_tail_group_repair(source, case, placement)
        if not verify_feasible(source, placement):
            continue
        try:
            metrics = _raw_quality(source, case, placement)
        except (TypeError, ValueError):
            continue
        delta = _relative_candidate_proxy_delta(metrics, current_metrics)
        audit_records.append(
            {
                "source": candidate_source,
                "candidate_index": index,
                "proxy_delta": delta,
                "soft": metrics[0],
                "bbox_area": metrics[1],
                "hpwl": metrics[2],
            }
        )
        if delta < -1.0e-6:
            admitted.append((delta, metrics[0], metrics[1], metrics[2], index, placement))
    snapshot["candidate_funnel_proxy_records"] = tuple(audit_records)
    if not admitted:
        snapshot["candidate_funnel_proxy_source"] = None
        return current
    winner = min(admitted, key=lambda item: item[:5])
    snapshot["candidate_funnel_proxy_source"] = f"candidate_{winner[4]}"
    return winner[5]


def _relative_candidate_proxy_delta(
    candidate: tuple[float, float, float],
    incumbent: tuple[float, float, float],
) -> float:
    soft, area, hpwl = candidate
    incumbent_soft, incumbent_area, incumbent_hpwl = incumbent
    eps = torch.finfo(torch.float64).eps
    return (
        2.0 * (soft - incumbent_soft)
        + 0.5 * math.log(max(area, eps) / max(incumbent_area, eps))
        + 0.5 * math.log(max(hpwl, eps) / max(incumbent_hpwl, eps))
    )


def _raw_constraint_pareto_guard(
    source: Any,
    case: FloorplanCase,
    analysis: LearnedAnalysis,
    current: list[tuple[float, float, float, float]],
) -> list[tuple[float, float, float, float]]:
    """Admit only raw-repaired constraint candidates that dominate current."""

    snapshot = getattr(analysis.analytic, "incumbent_snapshot", {})
    records = _constraint_records(snapshot)
    if not records:
        return current
    try:
        current_metrics = _raw_quality(source, case, current)
    except (TypeError, ValueError):
        return current
    admitted = []
    projected_candidates = analysis.analytic.projected_candidates
    raw_candidates = getattr(
        analysis.analytic,
        "raw_candidates",
        projected_candidates,
    )
    proposal_candidates = getattr(
        analysis.analytic.telemetry,
        "component_proposal_xywh",
        None,
    )
    proposal_available = getattr(
        analysis.analytic.telemetry,
        "component_proposal_available",
        None,
    )
    if proposal_available is not None:
        proposal_available = proposal_available.detach().to(
            device="cpu", dtype=torch.bool
        )
    for candidate_source, record in records.items():
        index = _candidate_index(candidate_source)
        if index is None:
            continue
        stages: list[tuple[str, Tensor]] = [
            ("raw", raw_candidates),
            ("projected", projected_candidates),
        ]
        if (
            proposal_candidates is not None
            and proposal_available is not None
            and 0 <= index < proposal_available.numel()
            and bool(proposal_available[index])
        ):
            stages.append(("proposal", proposal_candidates))
        for stage, candidates in stages:
            if not 0 <= index < candidates.shape[0]:
                continue
            placement = to_official_placements(
                source,
                case,
                candidates[index].detach().to(device="cpu", dtype=torch.float32),
            )
            placement = list(
                repair_raw_constraints(source, placement, record).placements
            )
            if not verify_feasible(source, placement):
                continue
            try:
                metrics = _raw_quality(source, case, placement)
            except (TypeError, ValueError):
                continue
            if _dominates(metrics, current_metrics):
                admitted.append((metrics, index, stage, placement))
    stage_rank = {"raw": 0, "projected": 1, "proposal": 2}
    return (
        min(
            admitted,
            key=lambda item: (
                item[0][0],
                item[0][1] + 0.05 * item[0][2],
                item[1],
                stage_rank[item[2]],
            ),
        )[3]
        if admitted
        else current
    )


def _repair_constraint_candidate(
    source: Any,
    case: FloorplanCase,
    placements: list[tuple[float, float, float, float]],
    snapshot: dict[str, object],
    candidate_source: object,
) -> list[tuple[float, float, float, float]]:
    record = _constraint_records(snapshot).get(str(candidate_source))
    if record is None:
        return placements
    repaired = list(repair_raw_constraints(source, placements, record).placements)
    repaired_feasible = verify_feasible(source, repaired)
    if not repaired_feasible:
        return placements
    if not verify_feasible(source, placements):
        return repaired
    try:
        return (
            repaired
            if _dominates(
                _raw_quality(source, case, repaired),
                _raw_quality(source, case, placements),
            )
            else placements
        )
    except (TypeError, ValueError):
        return placements


def _post_tail_group_repair(
    source: Any,
    case: FloorplanCase,
    placements: list[tuple[float, float, float, float]],
    *,
    max_moves: int = 12,
) -> list[tuple[float, float, float, float]]:
    """Apply a bounded exact replay only when every measured objective improves."""

    if max_moves <= 0 or not case.group_membership.numel():
        return placements
    soft_case = {
        "normalized": False,
        "boundary_bits": case.boundary_bits.detach().to(device="cpu"),
        "group_membership": case.group_membership.detach().to(device="cpu"),
        "mib_membership": case.mib_membership.detach().to(device="cpu"),
    }
    before_soft = soft_violation_normalized(soft_case, placements)
    if before_soft.raw_grouping == 0:
        return placements
    try:
        _, details = connect_groups(
            torch.as_tensor(placements, dtype=torch.float64, device="cpu"),
            case.group_membership,
            preplaced_mask=case.preplaced_mask,
            b2b_weight=case.b2b_weight,
        )
        moves = tuple(details.get("moves", ()))[:max_moves]
        if not moves:
            return placements
        repair_source = {
            "normalized": False,
            "area": _field(source, "area_targets"),
            "constraints": _field(source, "constraints"),
            "target": _field(source, "target_positions"),
            "preplaced_mask": case.preplaced_mask.detach().to(device="cpu"),
            "raw_preplaced_validated": case.raw_preplaced_validated,
            **soft_case,
        }
        working = placements
        working_soft = before_soft
        for move in moves:
            repaired = list(
                repair_raw_constraints(
                    repair_source,
                    working,
                    {"details": {"group": {"moves": (move,)}}},
                ).placements
            )
            after_soft = soft_violation_normalized(soft_case, repaired)
            componentwise_safe = (
                after_soft.raw_boundary <= working_soft.raw_boundary
                and after_soft.raw_grouping <= working_soft.raw_grouping
                and after_soft.raw_mib <= working_soft.raw_mib
            )
            if (
                componentwise_safe
                and verify_feasible(source, repaired)
                and _dominates(
                    _raw_quality(source, case, repaired),
                    _raw_quality(source, case, working),
                )
            ):
                working = repaired
                working_soft = after_soft
        return working
    except (RuntimeError, TypeError, ValueError):
        pass
    return placements


def _legacy_mib_challenger_guard(
    source: Any,
    case: FloorplanCase,
    analysis: LearnedAnalysis,
    current: list[tuple[float, float, float, float]],
) -> list[tuple[float, float, float, float]]:
    """Select a tagged legacy repair only when its exact incumbent key wins."""

    snapshot = getattr(analysis.analytic, "incumbent_snapshot", {})
    records = tuple(
        record
        for record in _constraint_records(snapshot).values()
        if record.get("challenger") == "legacy_mib"
        and record.get("stage") == "initial"
    )
    if not records:
        return current
    try:
        current_quality = _raw_quality(source, case, current)
    except (TypeError, ValueError):
        return current
    current_key = (current_quality[0], current_quality[1] + 0.05 * current_quality[2])
    admitted: list[
        tuple[tuple[float, float], int, list[tuple[float, float, float, float]]]
    ] = []
    projected = analysis.analytic.projected_candidates
    for record in records:
        index = _candidate_index(record.get("source"))
        if index is None or not 0 <= index < projected.shape[0]:
            continue
        placement = to_official_placements(
            source,
            case,
            projected[index].detach().to(device="cpu", dtype=torch.float32),
        )
        placement = _repair_constraint_candidate(
            source,
            case,
            placement,
            snapshot,
            record["source"],
        )
        placement = _post_tail_group_repair(source, case, placement)
        if not verify_feasible(source, placement):
            continue
        try:
            quality = _raw_quality(source, case, placement)
        except (TypeError, ValueError):
            continue
        key = (quality[0], quality[1] + 0.05 * quality[2])
        if key < current_key:
            admitted.append((key, index, placement))
    return min(admitted, key=lambda item: (item[0], item[1]))[2] if admitted else current


def _constraint_records(
    snapshot: dict[str, object],
) -> dict[str, dict[str, object]]:
    records: dict[str, dict[str, object]] = {}
    raw = snapshot.get("constraint_seed_provenance", ())
    for record in raw if isinstance(raw, (tuple, list)) else ():
        if not isinstance(record, dict):
            continue
        source = str(record.get("source", ""))
        if _candidate_index(source) is not None:
            records[source] = record
    return records


def _raw_quality(
    source: Any,
    case: FloorplanCase,
    placement: list[tuple[float, float, float, float]],
) -> tuple[float, float, float]:
    soft_case = {
        "normalized": False,
        "boundary_bits": case.boundary_bits.detach().to(device="cpu"),
        "group_membership": case.group_membership.detach().to(device="cpu"),
        "mib_membership": case.mib_membership.detach().to(device="cpu"),
    }
    return (
        soft_violation_normalized(soft_case, placement).total,
        bbox_area(placement),
        total_hpwl(source, placement),
    )


def _candidate_index(source: object) -> int | None:
    value = str(source)
    if value == "fallback":
        return 0
    if not value.startswith("candidate_"):
        return None
    try:
        return int(value.removeprefix("candidate_"))
    except ValueError:
        return None


def _merged_analytic_source(
    source: object, analytic_count: int, learned_count: int
) -> str | None:
    index = _candidate_index(source)
    if index is None:
        return None
    if index == 0:
        return "fallback"
    if index <= analytic_count:
        return f"candidate_{index}"
    if index <= 2 * analytic_count:
        merged = 1 + analytic_count + learned_count + index - analytic_count - 1
        return f"candidate_{merged}"
    return None


def _dominates(candidate: tuple[float, ...], incumbent: tuple[float, ...]) -> bool:
    tolerance = tuple(1.0e-6 * max(1.0, abs(value)) for value in incumbent)
    return all(value <= base for value, base in zip(candidate, incumbent)) and any(
        value < base - tol for value, base, tol in zip(candidate, incumbent, tolerance)
    )


def _condition_candidate_inside_outline(
    case: FloorplanCase,
    candidate: Tensor,
    hypothesis: object,
) -> Tensor | None:
    """Fit one structured candidate to a latent outline without changing sizes."""

    try:
        bounds = tuple(float(value) for value in hypothesis.bounds)  # type: ignore[attr-defined]
    except (AttributeError, TypeError, ValueError):
        return None
    if len(bounds) != 4:
        return None
    left, bottom, right, top = bounds
    width, height = right - left, top - bottom
    if not all(math.isfinite(value) for value in bounds) or min(width, height) <= 0.0:
        return None

    work = torch.as_tensor(
        candidate,
        dtype=torch.float32,
        device=case.area.device,
    ).clone()
    original = work.clone()
    if work.shape != (case.n, 4) or not bool(torch.isfinite(work).all()):
        return None
    if not bool((work[:, 2:4] > 0.0).all()):
        return None

    preplaced = case.preplaced_mask.to(device=work.device)
    if bool(preplaced.any()):
        targets = case.target.to(device=work.device, dtype=work.dtype)
        target_boxes = targets[preplaced]
        if not bool(
            (
                (target_boxes[:, 0] >= left - _OUTLINE_EPS)
                & (target_boxes[:, 1] >= bottom - _OUTLINE_EPS)
                & (target_boxes[:, 0] + target_boxes[:, 2] <= right + _OUTLINE_EPS)
                & (target_boxes[:, 1] + target_boxes[:, 3] <= top + _OUTLINE_EPS)
            ).all()
        ):
            return None
        work[preplaced] = target_boxes

    movable = ~preplaced
    if not bool(movable.any()):
        return None
    movable_boxes = work[movable]
    max_width = float(movable_boxes[:, 2].max())
    max_height = float(movable_boxes[:, 3].max())
    if max_width > width + _OUTLINE_EPS or max_height > height + _OUTLINE_EPS:
        return None

    def fit_axis(values: Tensor, lower: float, upper: float, max_size: float) -> Tensor:
        target_lower = lower
        target_upper = upper - max_size
        source_lower = float(values.min())
        source_upper = float(values.max())
        source_span = source_upper - source_lower
        target_span = max(0.0, target_upper - target_lower)
        if source_span <= _OUTLINE_EPS:
            offset = 0.5 * (target_lower + target_upper) - source_lower
            return values + values.new_tensor(offset)
        if source_span <= target_span + _OUTLINE_EPS:
            offset = (
                0.5 * (target_lower + target_upper)
                - 0.5 * (source_lower + source_upper)
            )
            return values + values.new_tensor(offset)
        return target_lower + (values - source_lower) * (target_span / source_span)

    work[movable, 0] = fit_axis(movable_boxes[:, 0], left, right, max_width)
    work[movable, 1] = fit_axis(movable_boxes[:, 1], bottom, top, max_height)
    if bool(preplaced.any()):
        work[preplaced] = case.target.to(device=work.device, dtype=work.dtype)[preplaced]

    inside = (
        (work[:, 0] >= left - _OUTLINE_EPS)
        & (work[:, 1] >= bottom - _OUTLINE_EPS)
        & (work[:, 0] + work[:, 2] <= right + _OUTLINE_EPS)
        & (work[:, 1] + work[:, 3] <= top + _OUTLINE_EPS)
    )
    if not bool(inside.all()) or bool(torch.allclose(work, original, rtol=1.0e-6, atol=1.0e-7)):
        return None
    try:
        return work if verify_feasible(case, work) else None
    except (RuntimeError, TypeError, ValueError):
        return None


def _outline_conditioned_variant(
    case: FloorplanCase,
    structured: Tensor,
) -> tuple[Tensor | None, dict[str, object]]:
    """Generate one exact-safe outline variant, or return a fail-closed record."""

    base: dict[str, object] = {
        "outline_variant_attempted": True,
        "outline_variant_accepted": False,
        "outline_variant_count": 0,
    }
    if infer_outline_hypotheses is None:
        base["outline_variant_failure_reason"] = "inference_unavailable"
        return None, base
    try:
        hypotheses = tuple(infer_outline_hypotheses(case))
    except Exception:  # A diagnostic contact must never disable the learned lane.
        base["outline_variant_failure_reason"] = "hypothesis_inference_failed"
        return None, base
    if not hypotheses:
        base["outline_variant_failure_reason"] = "empty_hypotheses"
        return None, base

    considered = 0
    for hypothesis in hypotheses:
        try:
            confidence = float(getattr(hypothesis, "confidence"))
        except (AttributeError, TypeError, ValueError):
            continue
        if not math.isfinite(confidence) or confidence < _OUTLINE_MIN_CONFIDENCE:
            continue
        considered += 1
        variant = _condition_candidate_inside_outline(case, structured, hypothesis)
        if variant is None:
            continue
        base.update(
            {
                "outline_variant_accepted": True,
                "outline_variant_count": 1,
                "outline_variant_hypothesis_id": str(
                    getattr(hypothesis, "hypothesis_id", "unknown")
                ),
                "outline_variant_source": str(
                    getattr(hypothesis, "source", "unknown")
                ),
                "outline_variant_confidence": confidence,
                "outline_variant_bounds": tuple(
                    float(value) for value in hypothesis.bounds  # type: ignore[attr-defined]
                ),
                "outline_variant_candidate_sha256": _tensor_sha256(variant),
            }
        )
        return variant, base
    base["outline_variant_failure_reason"] = (
        "uncertain_hypotheses" if considered == 0 else "no_exact_inside_variant"
    )
    return None, base


def _learned_population(
    case: FloorplanCase,
    model,
    config: LearnedConfig,
    *,
    seed: int,
    provenance: dict[str, object] | None = None,
    enforce_mib: bool = True,
) -> Tensor:
    population = config.analytic.dynamics.population
    fallback = safe_shelf(case).to(device=case.area.device, dtype=torch.float32)
    base = initialize_population(
        case,
        config.analytic.dynamics,
        fallback,
        enforce_mib=enforce_mib,
    )
    anchor_center, anchor_aspect = initializer_anchor(
        case,
        base.center,
        base.log_aspect,
        absolute=model.config.initializer_absolute,
    )
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    noise = torch.randn(
        (population, case.n, 3), generator=generator, dtype=torch.float32
    )
    residual = noise.to(device=case.area.device) * config.flow_noise_scale
    residual[:, case.preplaced_mask, :2] = 0.0
    residual[:, case.fixed_mask | case.preplaced_mask, 2] = 0.0

    with torch.inference_mode():
        output = model(case, population=population)
        flow_count = round(population * config.flow_fraction)
        if flow_count and config.flow_steps:
            for step in range(config.flow_steps):
                time = (step + 0.5) / config.flow_steps
                device_type = "cuda" if case.area.is_cuda else "cpu"
                with torch.autocast(
                    device_type=device_type,
                    dtype=torch.bfloat16,
                    enabled=model.config.compute_dtype == "bfloat16",
                ):
                    velocity = model.flow(
                        case, output.embedding, population, residual, time
                    )
                residual = residual + velocity.float() / config.flow_steps
                residual[..., :2].clamp_(
                    -max(config.max_position_residual, model.config.residual_bound),
                    max(config.max_position_residual, model.config.residual_bound),
                )
                residual[..., 2].clamp_(
                    -max(config.max_aspect_residual, model.config.aspect_residual_bound),
                    max(config.max_aspect_residual, model.config.aspect_residual_bound),
                )
                residual[:, case.preplaced_mask, :2] = 0.0
                residual[:, case.fixed_mask | case.preplaced_mask, 2] = 0.0
            output.center_residual[-flow_count:] = residual[-flow_count:, :, :2]
            output.log_aspect_residual[-flow_count:] = residual[-flow_count:, :, 2]

    center = anchor_center + output.center_residual
    log_aspect = (anchor_aspect + output.log_aspect_residual).clamp(-4.0, 4.0)
    learned_boxes = xywh_from_state(
        case,
        center,
        log_aspect,
        enforce_mib=enforce_mib,
    )
    topology_sources = learned_boxes
    if config.tail_topk is not None and config.tail_topk < population:
        features = candidate_features(case, learned_boxes, fallback)
        with torch.inference_mode():
            scores = model.ranker(output.embedding, population, features)
        keep = torch.argsort(scores, stable=True)[: config.tail_topk]
        learned_boxes = learned_boxes[keep]
    residual_slots = int(learned_boxes.shape[0])
    topology = learned_boxes.new_empty((0, case.n, 4))
    constraints = learned_boxes.new_empty((0, case.n, 4))
    failure_reason: str | None = None
    if config.topology_seeds:
        try:
            topology = _topology_seed_candidates(
                case,
                output,
                topology_sources,
                count=config.topology_seeds,
                provenance=provenance,
            )
        except (RuntimeError, ValueError) as exc:
            topology = learned_boxes.new_empty((0, case.n, 4))
            failure_reason = f"{type(exc).__name__}: {exc}"
        if topology.numel():
            if config.constraint_seeds:
                try:
                    constraints = _constraint_seed_candidates(
                        case,
                        output,
                        topology,
                        count=config.constraint_seeds,
                        provenance=provenance,
                    )
                except (RuntimeError, ValueError) as exc:
                    constraints = learned_boxes.new_empty((0, case.n, 4))
                    if provenance is not None:
                        provenance["constraint_seed_failure_reason"] = (
                            f"{type(exc).__name__}: {exc}"
                        )
            learned_boxes = torch.cat(
                (learned_boxes, constraints, topology), dim=0
            )
    if provenance is not None:
        provenance.setdefault("outline_variant_attempted", False)
        provenance.setdefault("outline_variant_accepted", False)
        provenance.setdefault("outline_variant_count", 0)
    structured = topology if topology.numel() else constraints
    if structured.numel():
        outline_variant, outline_provenance = _outline_conditioned_variant(
            case,
            structured[0],
        )
        if provenance is not None:
            provenance.update(outline_provenance)
        if outline_variant is not None:
            # Keep the structured seed and analytic incumbent intact.  The
            # variant replaces one residual slot, so the configured budget is
            # unchanged even when outline conditioning succeeds.
            learned_boxes = learned_boxes.clone()
            learned_boxes[0] = outline_variant
            if provenance is not None:
                provenance["outline_variant_replaced_residual_index"] = 0
    elif provenance is not None:
        provenance["outline_variant_failure_reason"] = "no_structured_candidates"
    if provenance is not None:
        provenance.setdefault("treemap_seed_attempted", config.treemap_seeds > 0)
        provenance.setdefault("treemap_seed_accepted", False)
        provenance.setdefault("treemap_seed_count", 0)
    if config.treemap_seeds:
        try:
            hypotheses = (
                tuple(infer_outline_hypotheses(case))
                if infer_outline_hypotheses is not None
                else ()
            )
            reference = (
                constraints[0]
                if constraints.numel()
                else topology[0]
                if topology.numel()
                else learned_boxes[0]
            )
            treemaps, records = exact_treemap_candidates(
                case,
                reference,
                hypotheses,
                count=config.treemap_seeds,
            )
            if treemaps.numel():
                raw_treemaps = treemaps
                raw_records = records
                try:
                    refined = _constraint_seed_candidates(
                        case,
                        output,
                        treemaps,
                        count=int(treemaps.shape[0]),
                    )
                except (RuntimeError, TypeError, ValueError):
                    refined = treemaps.new_empty((0, case.n, 4))
                if refined.numel():
                    refined = refined.to(
                        device=learned_boxes.device,
                        dtype=learned_boxes.dtype,
                    )
                    pool = torch.cat((raw_treemaps, refined), dim=0)
                    pool_records = raw_records + tuple(
                        {"constraint_refined": True}
                        for _ in range(int(refined.shape[0]))
                    )
                    order = sorted(
                        range(int(pool.shape[0])),
                        key=lambda index: (
                            not verify_feasible(case, pool[index]),
                            soft_violation_normalized(case, pool[index]).raw_total,
                            bbox_area(pool[index])
                            + 0.05 * total_hpwl(case, pool[index]),
                            index,
                        ),
                    )[: int(raw_treemaps.shape[0])]
                    treemaps = pool[order]
                    records = tuple(pool_records[index] for index in order)
                start = residual_slots
                learned_boxes = torch.cat(
                    (learned_boxes[:start], treemaps, learned_boxes[start:]),
                    dim=0,
                )
                if provenance is not None:
                    provenance.update(
                        {
                            "treemap_seed_accepted": True,
                            "treemap_seed_count": int(treemaps.shape[0]),
                            "treemap_seed_records": tuple(
                                {
                                    **record,
                                    "residual_index": start + index,
                                    "candidate_sha256": _tensor_sha256(candidate),
                                }
                                for index, (record, candidate) in enumerate(
                                    zip(records, treemaps, strict=True)
                                )
                            ),
                        }
                    )
        except (RuntimeError, TypeError, ValueError) as exc:
            if provenance is not None:
                provenance["treemap_seed_failure_reason"] = (
                    f"{type(exc).__name__}: {exc}"
                )
    if provenance is not None:
        provenance.setdefault("btree_seed_attempted", config.btree_seeds > 0)
        provenance.setdefault("btree_seed_accepted", False)
        provenance.setdefault("btree_seed_count", 0)
    if config.btree_seeds:
        try:
            btrees, records = _btree_seed_candidates(
                case,
                output,
                topology_sources,
                count=config.btree_seeds,
            )
            if btrees.numel():
                treemap_count = int(
                    provenance.get("treemap_seed_count", 0)
                    if provenance is not None
                    else 0
                )
                start = residual_slots + treemap_count
                learned_boxes = torch.cat(
                    (learned_boxes[:start], btrees, learned_boxes[start:]),
                    dim=0,
                )
                if provenance is not None:
                    provenance.update(
                        {
                            "btree_seed_accepted": True,
                            "btree_seed_count": int(btrees.shape[0]),
                            "btree_seed_records": tuple(
                                {
                                    **record,
                                    "residual_index": start + index,
                                    "candidate_sha256": _tensor_sha256(candidate),
                                }
                                for index, (record, candidate) in enumerate(
                                    zip(records, btrees, strict=True)
                                )
                            ),
                        }
                    )
        except (RuntimeError, TypeError, ValueError) as exc:
            if provenance is not None:
                provenance["btree_seed_failure_reason"] = (
                    f"{type(exc).__name__}: {exc}"
                )
    if provenance is not None:
        topology_count = int(topology.shape[0])
        provenance.update(
            {
                "topology_seed_attempted": config.topology_seeds > 0,
                "topology_seed_accepted": topology_count > 0,
                "topology_seed_count": topology_count,
                "constraint_seed_attempted": config.constraint_seeds > 0,
                "constraint_seed_accepted": int(constraints.shape[0]) > 0,
                "constraint_seed_count": int(constraints.shape[0]),
            }
        )
        if failure_reason is not None:
            provenance["topology_seed_failure_reason"] = failure_reason
    return learned_boxes


def _btree_seed_candidates(
    case: FloorplanCase,
    output: Any,
    source_boxes: Tensor,
    *,
    count: int,
) -> tuple[Tensor, tuple[dict[str, object], ...]]:
    if count <= 0:
        return source_boxes.new_empty((0, case.n, 4)), ()
    if output.btree_root_logits is None or output.btree_edge_logits is None:
        raise ValueError("checkpoint does not expose B*-Tree logits")
    tree = decode_btree_logits(output.btree_root_logits, output.btree_edge_logits)
    sources = source_boxes.detach().to(device="cpu", dtype=torch.float32)
    hypotheses = (
        tuple(infer_outline_hypotheses(case))
        if infer_outline_hypotheses is not None
        else ()
    )
    origins = (
        tuple((outline.x_left, outline.y_bottom, outline.hypothesis_id) for outline in hypotheses)
        or ((0.0, 0.0, "origin"),)
    )
    pool: list[tuple[tuple[float, ...], Tensor, dict[str, object]]] = []
    for source_index, source in enumerate(sources[: max(1, count)]):
        dims = source[:, 2:4]
        base_order = torch.argsort(centers_from_xywh(source)[:, 1], stable=True)
        orders = contact_aware_vertical_orders(
            base_order,
            case.boundary_bits,
            case.group_membership,
        )[:2]
        for order_name, order in orders:
            for x0, y0, outline_id in origins:
                candidate = tree.pack_x_compacted(
                    dims,
                    order,
                    case.preplaced_mask,
                    case.target,
                    origin=(x0, y0),
                ).float()
                hard = verify_feasible(case, candidate)
                soft = soft_violation_normalized(case, candidate).raw_total
                quality = bbox_area(candidate) + 0.05 * total_hpwl(case, candidate)
                pool.append(
                    (
                        (not hard, soft, quality, source_index, outline_id, order_name),
                        candidate,
                        {
                            "source_type": "btree",
                            "shape_source_index": source_index,
                            "outline_hypothesis": outline_id,
                            "vertical_order_source": order_name,
                            "predicted_root": tree.root,
                        },
                    )
                )
    selected = []
    seen: set[str] = set()
    for item in sorted(pool, key=lambda item: item[0]):
        digest = _tensor_sha256(item[1])
        if digest in seen:
            continue
        seen.add(digest)
        selected.append(item)
        if len(selected) == count:
            break
    if not selected:
        raise ValueError("B*-Tree candidate generation produced no candidates")
    return (
        torch.stack([item[1] for item in selected]).to(
            device=source_boxes.device, dtype=source_boxes.dtype
        ),
        tuple(item[2] for item in selected),
    )


def _needs_legacy_mib_challenger(case: FloorplanCase) -> bool:
    """Route a legacy-shape challenger for dense-anchor, small-MIB cases."""

    if not 80 <= case.n <= 88 or not case.mib_membership.numel():
        return False
    hard = case.fixed_mask | case.preplaced_mask
    if float(hard.float().mean()) < 0.18:
        return False
    return any(
        int(row.sum()) == 3 and int((row & hard).sum()) == 1
        for row in case.mib_membership
    )


def _merge_legacy_mib_challenger(
    primary: Tensor,
    primary_provenance: dict[str, object],
    legacy: Tensor,
    legacy_provenance: dict[str, object],
) -> tuple[Tensor, dict[str, object]]:
    def parts(
        population: Tensor,
        provenance: dict[str, object],
    ) -> tuple[Tensor, Tensor, Tensor, int]:
        topology_count = int(provenance.get("topology_seed_count", 0))
        constraint_count = int(provenance.get("constraint_seed_count", 0))
        residual_count = int(population.shape[0]) - topology_count - constraint_count
        if residual_count < 0:
            raise ValueError("candidate provenance exceeds population size")
        return (
            population[:residual_count],
            population[residual_count : residual_count + constraint_count],
            population[residual_count + constraint_count :],
            residual_count,
        )

    primary_parts = parts(primary, primary_provenance)
    legacy_parts = parts(legacy, legacy_provenance)
    primary_topology_count = int(primary_parts[2].shape[0])
    merged = torch.cat(
        (
            primary_parts[0],
            legacy_parts[0],
            primary_parts[1],
            legacy_parts[1],
            primary_parts[2],
            legacy_parts[2],
        ),
        dim=0,
    )
    provenance = dict(primary_provenance)
    provenance.update(
        {
            "legacy_mib_challenger": True,
            "topology_seed_attempted": True,
            "topology_seed_accepted": bool(primary_parts[2].numel())
            or bool(legacy_parts[2].numel()),
            "topology_seed_count": int(primary_parts[2].shape[0])
            + int(legacy_parts[2].shape[0]),
            "constraint_seed_attempted": True,
            "constraint_seed_accepted": bool(primary_parts[1].numel())
            or bool(legacy_parts[1].numel()),
            "constraint_seed_count": int(primary_parts[1].shape[0])
            + int(legacy_parts[1].shape[0]),
            "topology_seed_orders": tuple(
                primary_provenance.get("topology_seed_orders", ())
            )
            + tuple(legacy_provenance.get("topology_seed_orders", ())),
        }
    )
    legacy_records = []
    for raw in legacy_provenance.get("constraint_seed_records", ()):
        record = dict(raw)
        record["challenger"] = "legacy_mib"
        record["topology_seed_index"] = (
            int(record["topology_seed_index"]) + primary_topology_count
        )
        legacy_records.append(record)
    provenance["constraint_seed_records"] = tuple(
        primary_provenance.get("constraint_seed_records", ())
    ) + tuple(legacy_records)
    primary_catalog = primary_provenance.get("topology_order_catalog", {})
    legacy_catalog = legacy_provenance.get("topology_order_catalog", {})
    if isinstance(primary_catalog, dict) and isinstance(legacy_catalog, dict):
        provenance["topology_order_catalog"] = {
            **primary_catalog,
            **legacy_catalog,
        }
    for key in ("constraint_seed_pool_size", "topology_seed_pool_size"):
        provenance[key] = int(primary_provenance.get(key, 0)) + int(
            legacy_provenance.get(key, 0)
        )
    return merged, provenance


def _constraint_seed_candidates(
    case: FloorplanCase,
    output: Any,
    topology: Tensor,
    *,
    count: int,
    provenance: dict[str, object] | None = None,
) -> Tensor:
    if count <= 0:
        return topology.new_empty((0, case.n, 4))
    contact_logits = getattr(output, "contact_logits", None)
    relation_scores = (
        contact_logits[..., 1:5]
        if contact_logits is not None
        else output.precedence_logits[..., :4]
    )
    boundary_scores = getattr(output, "boundary_order_scores", None)
    mib_log_aspect = getattr(output, "mib_log_aspect", None)
    relation_scores = relation_scores.detach().to(device="cpu", dtype=torch.float32)
    if boundary_scores is not None:
        boundary_scores = boundary_scores.detach().to(
            device="cpu", dtype=torch.float32
        )
    if mib_log_aspect is not None:
        mib_log_aspect = mib_log_aspect.detach().to(
            device="cpu", dtype=torch.float32
        )

    reference = safe_shelf(case).detach().to(device="cpu", dtype=torch.float32)
    hpwl_denominator = max(
        total_hpwl(case, reference), torch.finfo(torch.float64).eps
    )
    bbox_denominator = max(bbox_area(reference), torch.finfo(torch.float64).eps)
    pool: list[Tensor] = []
    records: list[dict[str, object]] = []
    for topology_index, source in enumerate(
        topology.detach().to(device="cpu", dtype=torch.float32)
    ):
        for variant in construct_constraint_variants(
            case,
            source,
            relation_scores=relation_scores,
            boundary_order_scores=boundary_scores,
            mib_log_aspect=mib_log_aspect,
        ):
            candidate = variant.xywh
            if any(torch.equal(candidate, prior) for prior in pool):
                continue
            hard_feasible = verify_feasible(case, candidate)
            raw_soft = soft_violation_normalized(case, candidate).raw_total
            normalized_hpwl = total_hpwl(case, candidate) / hpwl_denominator
            normalized_bbox = bbox_area(candidate) / bbox_denominator
            pool.append(candidate)
            records.append(
                {
                    "kind": variant.kind,
                    "topology_seed_index": topology_index,
                    "candidate_sha256": _tensor_sha256(candidate),
                    "hard_feasible": hard_feasible,
                    "priority": {
                        "raw_soft_violation": raw_soft,
                        "normalized_quality": normalized_hpwl + normalized_bbox,
                        "normalized_hpwl": normalized_hpwl,
                        "normalized_bbox_area": normalized_bbox,
                    },
                    "details": variant.details,
                }
            )
    if not pool:
        raise ValueError("constraint construction produced no changed candidate")

    kind_order = {
        "combined": 0,
        "group_contacts": 1,
        "boundary_frame": 2,
        "mib_shapes": 3,
    }

    def priority(index: int) -> tuple[object, ...]:
        metric = records[index]["priority"]
        assert isinstance(metric, dict)
        return (
            not bool(records[index]["hard_feasible"]),
            int(metric["raw_soft_violation"]),
            float(metric["normalized_quality"]),
            kind_order.get(str(records[index]["kind"]), 99),
            int(records[index]["topology_seed_index"]),
            str(records[index]["candidate_sha256"]),
        )

    ordered_indices = sorted(range(len(pool)), key=priority)
    representatives = sorted(
        (
            min(
                (
                    index
                    for index, record in enumerate(records)
                    if record["kind"] == kind
                ),
                key=priority,
            )
            for kind in sorted({str(record["kind"]) for record in records})
        ),
        key=priority,
    )
    selected_indices = representatives[:count]
    selected_indices.extend(
        index
        for index in ordered_indices
        if index not in selected_indices
    )
    selected_indices = selected_indices[:count]
    selected_records = tuple(dict(records[index]) for index in selected_indices)
    if provenance is not None:
        provenance["constraint_seed_pool_size"] = len(pool)
        provenance["constraint_seed_records"] = selected_records
        provenance["constraint_seed_kind_counts"] = {
            kind: sum(record["kind"] == kind for record in selected_records)
            for kind in sorted({str(record["kind"]) for record in selected_records})
        }
        provenance["constraint_seed_selection"] = "best-per-kind-then-priority"
    return torch.stack([pool[index] for index in selected_indices]).to(
        device=topology.device,
        dtype=topology.dtype,
    )


def _topology_seed_candidates(
    case: FloorplanCase,
    output,
    source_boxes: Tensor,
    *,
    count: int,
    provenance: dict[str, object] | None = None,
) -> Tensor:
    positive = output.positive_permutation
    negative = output.negative_permutation
    if count <= 0:
        return source_boxes.new_empty((0, case.n, 4))
    if positive is None or negative is None:
        raise ValueError("checkpoint does not expose dual-permutation topology")

    # Hard assignment and DAG packing are scalar-heavy. Transfer the complete
    # decode inputs once, run the opt-in structure path on CPU, then return one
    # stacked tensor to the case device.
    soft = (
        torch.stack((positive, negative)).detach().to(device="cpu", dtype=torch.float32)
    )
    precedence = output.precedence_logits.detach().to(device="cpu", dtype=torch.float32)
    sources = source_boxes.detach().to(device="cpu", dtype=torch.float32)
    targets = case.target.detach().to(device="cpu", dtype=torch.float32)
    preplaced = case.preplaced_mask.detach().to(device="cpu", dtype=torch.bool)
    active = case.block_mask.detach().to(device="cpu", dtype=torch.bool)

    positive_order = hard_permutation(soft[0], active)
    negative_order = hard_permutation(soft[1], active)
    confidence = torch.softmax(
        soft_sequence_pair_relation_logits(soft[0], soft[1]),
        dim=-1,
    )
    positive_order, negative_order = adapt_preplaced_topology(
        positive_order,
        negative_order,
        targets,
        preplaced,
        relation_confidence=confidence,
    )
    safe_variants = anchor_safe_order_variants(
        positive_order,
        negative_order,
        preplaced,
    )
    order_variants = (
        ("adapted", positive_order, negative_order),
        *(
            (variant.name, variant.positive, variant.negative)
            for variant in safe_variants
        ),
    )
    order_catalog: dict[str, dict[str, object]] = {}
    order_definitions = []
    for variant_name, variant_positive, variant_negative in order_variants:
        variant_topology = decode_sequence_pair(
            variant_positive,
            variant_negative,
            n=case.n,
        )
        positive_values = tuple(int(value) for value in variant_positive)
        negative_values = tuple(int(value) for value in variant_negative)
        horizontal_edges = tuple(
            tuple(int(value) for value in edge)
            for edge in variant_topology.horizontal_edges.tolist()
        )
        vertical_edges = tuple(
            tuple(int(value) for value in edge)
            for edge in variant_topology.vertical_edges.tolist()
        )
        order_hash = _tensor_sha256(torch.stack((variant_positive, variant_negative)))
        if order_hash in order_catalog:
            continue
        edge_hash = hashlib.sha256(
            repr((horizontal_edges, vertical_edges)).encode("ascii")
        ).hexdigest()
        order_catalog[order_hash] = {
            "order_variant": variant_name,
            "positive_order": positive_values,
            "negative_order": negative_values,
            "horizontal_edges": horizontal_edges,
            "vertical_edges": vertical_edges,
            "topology_edge_sha256": edge_hash,
        }
        order_definitions.append(
            (
                variant_name,
                variant_positive,
                variant_negative,
                variant_topology,
                order_hash,
                edge_hash,
            )
        )
    adapted_order_hash = str(order_definitions[0][4])
    adapted_edge_hash = str(order_definitions[0][5])
    if provenance is not None:
        provenance.update(
            {
                "topology_soft_permutation_sha256": _tensor_sha256(soft),
                "topology_precedence_logits_sha256": _tensor_sha256(precedence),
                "topology_order_sha256": adapted_order_hash,
                "topology_edge_sha256": adapted_edge_hash,
                "topology_order_catalog": {
                    order_hash: dict(record)
                    for order_hash, record in order_catalog.items()
                },
                "topology_safe_order_variants": tuple(
                    {
                        "order_variant": variant_name,
                        "topology_order_sha256": order_hash,
                        "topology_edge_sha256": edge_hash,
                    }
                    for (
                        variant_name,
                        _,
                        _,
                        _,
                        order_hash,
                        edge_hash,
                    ) in order_definitions[1:]
                ),
            }
        )
    origin = sources[..., :2].amin(dim=(0, 1))

    pool: list[Tensor] = []
    pool_records: list[dict[str, object]] = []
    pool_attempt_indices: list[int] = []
    attempts: list[dict[str, object]] = []
    last_error: RuntimeError | ValueError | None = None
    for aspect_source, source in enumerate(sources):
        for (
            variant_name,
            variant_positive,
            variant_negative,
            variant_topology,
            order_hash,
            edge_hash,
        ) in order_definitions:
            attempt = {
                "order_variant": variant_name,
                "aspect_source_index": aspect_source,
            }
            try:
                candidate = pack_sequence_pair_with_anchors(
                    source[:, 2:4],
                    variant_positive,
                    variant_negative,
                    targets,
                    preplaced,
                    origin=origin,
                    spacing=1.0e-5,
                )
                candidate = copy_preplaced_targets(candidate, targets, preplaced)
                realized = relation_mask_from_rectangles(candidate)
                pair = (
                    variant_topology.active_mask[:, None]
                    & variant_topology.active_mask[None, :]
                )
                pair.fill_diagonal_(False)
                selected = realized.gather(
                    -1,
                    variant_topology.relation.clamp_min(0).unsqueeze(-1),
                ).squeeze(-1)
                if not bool(selected[pair].all()):
                    raise ValueError(
                        "packed geometry does not realize its sequence-pair order"
                    )
                if not verify_feasible(case, candidate):
                    raise ValueError("packed geometry is not exact-feasible")
            except (RuntimeError, ValueError) as exc:
                last_error = exc
                attempts.append(
                    {
                        **attempt,
                        "status": "rejected",
                        "failure_reason": f"{type(exc).__name__}: {exc}",
                    }
                )
                continue
            if torch.allclose(candidate, source, rtol=1.0e-6, atol=1.0e-7):
                attempts.append(
                    {
                        **attempt,
                        "status": "rejected",
                        "failure_reason": "unchanged_geometry",
                    }
                )
                continue
            if any(torch.allclose(candidate, prior) for prior in pool):
                attempts.append(
                    {
                        **attempt,
                        "status": "rejected",
                        "failure_reason": "duplicate_geometry",
                    }
                )
                continue
            pool_index = len(pool)
            pool.append(candidate)
            pool_records.append(
                {
                    **attempt,
                    "pool_index": pool_index,
                    "topology_order_sha256": order_hash,
                    "topology_edge_sha256": edge_hash,
                    "candidate_sha256": _tensor_sha256(candidate),
                }
            )
            pool_attempt_indices.append(len(attempts))
            attempts.append({**attempt, "pool_index": pool_index, "status": "pooled"})

    if not pool:
        if last_error is not None:
            raise last_error
        raise ValueError("no changed topology seed was accepted")

    reference = safe_shelf(case).detach().to(device="cpu", dtype=torch.float32)
    reference_hpwl = total_hpwl(case, reference)
    reference_bbox = bbox_area(reference)
    hpwl_denominator = max(reference_hpwl, torch.finfo(torch.float64).eps)
    bbox_denominator = max(reference_bbox, torch.finfo(torch.float64).eps)
    for candidate, record in zip(pool, pool_records, strict=True):
        raw_soft = soft_violation_normalized(case, candidate).raw_total
        normalized_hpwl = total_hpwl(case, candidate) / hpwl_denominator
        normalized_bbox = bbox_area(candidate) / bbox_denominator
        record["priority"] = {
            "raw_soft_violation": raw_soft,
            "normalized_quality": normalized_hpwl + normalized_bbox,
            "normalized_hpwl": normalized_hpwl,
            "normalized_bbox_area": normalized_bbox,
        }

    def priority_key(index: int) -> tuple[object, ...]:
        priority = pool_records[index]["priority"]
        assert isinstance(priority, dict)
        return (
            int(priority["raw_soft_violation"]),
            float(priority["normalized_quality"]),
            str(pool_records[index]["topology_order_sha256"]),
            int(pool_records[index]["aspect_source_index"]),
            str(pool_records[index]["candidate_sha256"]),
        )

    total_order = sorted(range(len(pool)), key=priority_key)
    best_per_order: dict[str, int] = {}
    for index in total_order:
        order_hash = str(pool_records[index]["topology_order_sha256"])
        best_per_order.setdefault(order_hash, index)
    selected_indices = list(best_per_order.values())[:count]
    if len(selected_indices) < count:
        selected_set = set(selected_indices)
        selected_indices.extend(
            index for index in total_order if index not in selected_set
        )
        selected_indices = selected_indices[:count]

    selection_rank = {index: rank for rank, index in enumerate(selected_indices)}
    priority_rank = {index: rank for rank, index in enumerate(total_order)}
    for index, (record, attempt_index) in enumerate(
        zip(pool_records, pool_attempt_indices, strict=True)
    ):
        selected = index in selection_rank
        record["priority_rank"] = priority_rank[index]
        record["status"] = "selected" if selected else "rejected_by_budget"
        record["selection_rank"] = selection_rank.get(index)
        attempts[attempt_index].update(
            {
                "priority_rank": priority_rank[index],
                "status": record["status"],
                "selection_rank": record["selection_rank"],
            }
        )

    selected_orders = [dict(pool_records[index]) for index in selected_indices]
    if provenance is not None:
        provenance["topology_order_attempts"] = tuple(attempts)
        provenance["topology_seed_pool_size"] = len(pool)
        provenance["topology_seed_pool"] = tuple(
            dict(record) for record in pool_records
        )
        provenance["topology_selection_reference"] = {
            "source": "safe_shelf",
            "hpwl": reference_hpwl,
            "bbox_area": reference_bbox,
        }
        provenance["topology_seed_orders"] = tuple(selected_orders)
        provenance["topology_seed_aspect_source_indices"] = tuple(
            int(record["aspect_source_index"]) for record in selected_orders
        )
    return torch.stack([pool[index] for index in selected_indices]).to(
        device=source_boxes.device,
        dtype=source_boxes.dtype,
    )


def _tensor_sha256(value: Tensor) -> str:
    raw = value.detach().to(device="cpu").contiguous().view(torch.uint8).reshape(-1)
    return hashlib.sha256(bytes(raw.tolist())).hexdigest()
