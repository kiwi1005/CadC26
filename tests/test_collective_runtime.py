from __future__ import annotations

import torch
from torch import nn

from hcfp.case import from_official
from hcfp.collective import PAIR_FEATURES
from hcfp.collective_runtime import CollectiveForceController, relation_from_bdp
from hcfp.dynamics import DynamicsConfig, initialize_population, relax
from hcfp.geometry import normalize_xywh
from hcfp.model import CollectiveStepOutput, HCFPModel, ModelConfig
from hcfp.projection_guidance import ProjectionGuidance


def _case():
    return from_official(
        3,
        [4.0, 4.0, 4.0],
        [[0, 1, 3.0], [1, 2, 1.0]],
        [],
        [],
        [[0, 1, 0, 1, 1], [0, 0, 0, 1, 0], [1, 0, 0, 0, 2]],
        [
            [0.0, 0.0, 2.0, 2.0],
            [-1.0, -1.0, -1.0, -1.0],
            [-1.0, -1.0, 2.0, 2.0],
        ],
    )


def _initial(case):
    return normalize_xywh(
        case,
        torch.tensor(
            [
                [0.0, 0.0, 2.0, 2.0],
                [0.5, 0.0, 2.0, 2.0],
                [1.0, 0.0, 2.0, 2.0],
            ]
        ),
    )


def _model(case) -> tuple[HCFPModel, torch.Tensor]:
    model = HCFPModel(
        ModelConfig(
            hidden_dim=12,
            encoder_layers=1,
            collective_enabled=True,
            collective_message_dim=10,
            collective_passes=2,
        )
    ).eval()
    return model, model.encoder(case)


def test_bdp_relation_mapping_is_explicit() -> None:
    direction = torch.tensor([-1, 0, 1, 2, 3], dtype=torch.long)
    assert torch.equal(
        relation_from_bdp(direction),
        torch.tensor([-1, 0, 1, 3, 2], dtype=torch.long),
    )


def test_neutral_collective_controller_matches_default_relaxation() -> None:
    torch.manual_seed(31)
    case = _case()
    model, embedding = _model(case)
    cfg = DynamicsConfig(population=2, steps=2)
    baseline = relax(case, cfg, initial_xywh=_initial(case))
    controller = CollectiveForceController(model, embedding)
    controlled = relax(
        case,
        cfg,
        initial_xywh=_initial(case),
        force_controller=controller,
    )

    assert controller.calls == cfg.steps
    assert torch.equal(controlled.boxes, baseline.boxes)
    assert torch.equal(controlled.state.energy_history, baseline.state.energy_history)
    assert torch.equal(controlled.boxes[:, 0], case.target[0].expand(2, -1))
    assert torch.equal(controlled.boxes[:, 2, 2:4], case.target[2, 2:4].expand(2, -1))


def test_controller_recomputes_pair_geometry_and_maps_guidance() -> None:
    case = _case()
    model, embedding = _model(case)

    class Recorder(nn.Module):
        def __init__(self):
            super().__init__()
            self.features: list[torch.Tensor] = []

        def forward(
            self,
            _case,
            _embedding,
            geometry,
            pair_features,
            _pair_mask,
            _step_fraction,
        ):
            self.features.append(pair_features.clone())
            return CollectiveStepOutput(
                torch.zeros_like(geometry),
                torch.ones((*geometry.shape[:2], 7)),
            )

    recorder = Recorder()
    model.collective = recorder
    direction = torch.full((2, case.n, case.n), -1, dtype=torch.long)
    direction[:, 0, 1] = 3  # BDP ABOVE -> collective ABOVE (channel 2).
    direction[:, 1, 0] = 2  # Inverse BDP BELOW relation.
    confidence = (direction >= 0).to(dtype=torch.float32)
    guidance = ProjectionGuidance(
        preferred_direction=direction,
        preferred_confidence=confidence,
        contact_direction=direction,
        contact_confidence=confidence,
        boundary_axis_lock=torch.zeros((2, case.n, 2), dtype=torch.bool),
    )
    controller = CollectiveForceController.from_guidance(model, embedding, guidance)
    state = initialize_population(
        case,
        DynamicsConfig(population=2, steps=0),
        _initial(case),
    )
    controller(case, state, 0.0)
    state.center[:, 1, 0] += 1.0
    controller(case, state, 0.5)

    dx = PAIR_FEATURES.index("dx")
    topology_above = PAIR_FEATURES.index("topology_above")
    latch_above = PAIR_FEATURES.index("latch_above")
    assert recorder.features[1][0, 0, 1, dx] == recorder.features[0][0, 0, 1, dx] + 1.0
    assert recorder.features[0][0, 0, 1, topology_above] == 1.0
    assert recorder.features[0][0, 0, 1, latch_above] == 1.0
