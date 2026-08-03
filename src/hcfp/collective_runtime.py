"""Opt-in runtime adapter for geometry-aware collective controls."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from hcfp.case import FloorplanCase
from hcfp.collective import dynamic_pair_features
from hcfp.dynamics import ForceControl, PopulationState
from hcfp.geometry import exact_shape_projection
from hcfp.model import HCFPModel
from hcfp.projection_guidance import ProjectionGuidance


Tensor = torch.Tensor
_BDP_TO_COLLECTIVE = (0, 1, 3, 2)


def relation_from_bdp(direction: Tensor) -> Tensor:
    """Map BDP ``L/R/BELOW/ABOVE`` ids to ``L/R/ABOVE/BELOW``."""

    value = torch.as_tensor(direction)
    if torch.is_floating_point(value) or value.dtype == torch.bool:
        raise ValueError("BDP direction must use an integer dtype")
    valid = (value == -1) | ((0 <= value) & (value < 4))
    if not bool(valid.all()):
        raise ValueError("BDP direction must contain -1 or ids 0..3")
    mapping = torch.tensor(_BDP_TO_COLLECTIVE, dtype=torch.long, device=value.device)
    return torch.where(value >= 0, mapping[value.clamp_min(0).long()], -1)


@dataclass
class CollectiveForceController:
    """Recompute pair geometry on every analytic relaxation step."""

    model: HCFPModel
    embedding: Tensor
    topology_relation: Tensor | None = None
    active_latch: Tensor | None = None
    calls: int = 0

    @classmethod
    def from_guidance(
        cls,
        model: HCFPModel,
        embedding: Tensor,
        guidance: ProjectionGuidance | None,
    ) -> CollectiveForceController:
        if guidance is None:
            return cls(model, embedding)
        return cls(
            model,
            embedding,
            relation_from_bdp(guidance.preferred_direction),
            relation_from_bdp(guidance.contact_direction),
        )

    def __post_init__(self) -> None:
        if not self.model.config.collective_enabled or not hasattr(
            self.model, "collective"
        ):
            raise ValueError("model does not provide collective dynamics")
        if self.embedding.ndim != 2:
            raise ValueError("embedding must have shape [N,H]")

    def __call__(
        self,
        case: FloorplanCase,
        state: PopulationState,
        step_fraction: float,
    ) -> ForceControl:
        population = state.center.shape[0]
        topology = self._relation_for_population(
            self.topology_relation,
            population,
            case,
            "topology_relation",
        )
        latch = self._relation_for_population(
            self.active_latch,
            population,
            case,
            "active_latch",
        )
        dimensions = exact_shape_projection(case, state.log_aspect)
        pairs = dynamic_pair_features(
            case,
            state.center,
            dimensions,
            topology_relation=topology,
            active_latch=latch,
        )
        geometry = torch.cat(
            (state.log_aspect.unsqueeze(-1), dimensions),
            dim=-1,
        )
        device_type = "cuda" if case.area.is_cuda else "cpu"
        with torch.inference_mode(), torch.autocast(
            device_type=device_type,
            dtype=torch.bfloat16,
            enabled=self.model.config.compute_dtype == "bfloat16",
        ):
            output = self.model.collective(
                case,
                self.embedding,
                geometry,
                pairs.features,
                pairs.pair_mask,
                step_fraction,
            )
        self.calls += 1
        return ForceControl(output.force_gates.float(), output.velocity.float())

    @staticmethod
    def _relation_for_population(
        relation: Tensor | None,
        population: int,
        case: FloorplanCase,
        name: str,
    ) -> Tensor | None:
        if relation is None:
            return None
        value = relation.to(device=case.area.device)
        if value.shape != (population, case.n, case.n):
            raise ValueError(f"{name} must match the active [K,N,N] population")
        return value
