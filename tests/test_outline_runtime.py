from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import torch

from hcfp.case import from_official
from hcfp.dynamics import DynamicsConfig
import hcfp.learned as learned
from hcfp.learned import (
    LearnedConfig,
    _condition_candidate_inside_outline,
    _learned_population,
)
from hcfp.model import HCFPModel, ModelConfig
from hcfp.verify import verify_feasible


def _case():
    return from_official(
        2,
        [4.0, 4.0],
        [],
        [],
        [],
        [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0]],
    )


def _hypothesis(*, confidence: float = 1.0):
    return SimpleNamespace(
        hypothesis_id="outline-test",
        source="test",
        confidence=confidence,
        bounds=(-0.5, -0.5, 1.0, 0.5),
    )


def test_inside_outline_variant_preserves_area_and_exact_geometry() -> None:
    case = _case()
    side = float(torch.sqrt(case.area[0]))
    source = torch.tensor(
        [[2.0, 0.0, side, side], [3.0, 0.0, side, side]],
        dtype=torch.float32,
    )

    variant = _condition_candidate_inside_outline(case, source, _hypothesis())

    assert variant is not None
    assert verify_feasible(case, variant)
    assert torch.allclose(variant[:, 2:].prod(dim=1), case.area, atol=1.0e-6)
    assert float(variant[:, 0].min()) >= -0.5 - 1.0e-6
    assert float((variant[:, 0] + variant[:, 2]).max()) <= 1.0 + 1.0e-6
    assert not torch.allclose(variant, source)


def test_runtime_contact_replaces_one_residual_slot_without_growing_budget(
    monkeypatch,
) -> None:
    case = _case()
    model = HCFPModel(
        ModelConfig(hidden_dim=16, encoder_layers=1, topology_enabled=True)
    )
    side = float(torch.sqrt(case.area[0]))
    structured = torch.tensor(
        [[[2.0, 0.0, side, side], [3.0, 0.0, side, side]]],
        dtype=torch.float32,
    )

    def fake_topology(_case, _output, source_boxes, *, count, provenance=None):
        del _case, _output, count, provenance
        return structured.to(device=source_boxes.device)

    monkeypatch.setattr(learned, "_topology_seed_candidates", fake_topology)
    monkeypatch.setattr(
        learned,
        "infer_outline_hypotheses",
        lambda _case: (_hypothesis(),),
    )
    config = LearnedConfig(
        analytic=replace(
            LearnedConfig().analytic,
            dynamics=DynamicsConfig(population=2, steps=0),
        ),
        topology_seeds=1,
        seed=7,
    )
    provenance: dict[str, object] = {}

    population = _learned_population(
        case,
        model,
        config,
        seed=7,
        provenance=provenance,
    )

    assert population.shape[0] == 3
    assert provenance["outline_variant_attempted"] is True
    assert provenance["outline_variant_accepted"] is True
    assert provenance["outline_variant_count"] == 1
    assert provenance["outline_variant_replaced_residual_index"] == 0
    assert torch.equal(population[-1], structured[0])
    assert not torch.allclose(population[0], structured[0])
    assert verify_feasible(case, population[0])


def test_empty_or_uncertain_outline_beam_falls_back_cleanly(monkeypatch) -> None:
    case = _case()
    side0 = float(torch.sqrt(case.area[0]))
    side1 = float(torch.sqrt(case.area[1]))
    source = torch.tensor(
        [[2.0, 0.0, side0, side0], [3.0, 0.0, side1, side1]],
        dtype=torch.float32,
    )
    monkeypatch.setattr(learned, "infer_outline_hypotheses", lambda _case: ())
    variant, empty = learned._outline_conditioned_variant(case, source)
    assert variant is None
    assert empty["outline_variant_failure_reason"] == "empty_hypotheses"

    monkeypatch.setattr(
        learned,
        "infer_outline_hypotheses",
        lambda _case: (_hypothesis(confidence=0.01),),
    )
    variant, uncertain = learned._outline_conditioned_variant(case, source)
    assert variant is None
    assert uncertain["outline_variant_failure_reason"] == "uncertain_hypotheses"
