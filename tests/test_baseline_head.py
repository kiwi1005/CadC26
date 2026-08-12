from __future__ import annotations

import pytest
import torch

from hcfp.data import DataSample, extract_labels
from hcfp.model import HCFPModel, ModelConfig
from hcfp.profile import synthetic_case
from hcfp.training import supervised_loss


def _sample() -> DataSample:
    case = synthetic_case(32, device="cpu")
    rectangles = torch.cat(
        (
            torch.zeros((case.n, 2)),
            torch.sqrt(case.area).unsqueeze(1).repeat(1, 2),
        ),
        dim=1,
    )
    return DataSample(
        "baseline", case, extract_labels(case, rectangles, normalized=True)
    )


def test_baseline_head_is_opt_in_and_emits_normalized_log_scalars() -> None:
    sample = _sample()
    disabled = HCFPModel(ModelConfig(hidden_dim=12, encoder_layers=1))
    assert disabled(sample.case, population=1).baseline_log_area is None
    assert not any(name.startswith("baseline.") for name in disabled.state_dict())

    model = HCFPModel(
        ModelConfig(hidden_dim=12, encoder_layers=1, baseline_enabled=True)
    )
    output = model(sample.case, population=2)
    assert output.baseline_log_area is not None
    assert output.baseline_log_hpwl is not None
    assert output.baseline_log_area.ndim == 0
    assert output.baseline_log_hpwl.ndim == 0
    assert torch.isfinite(output.baseline_log_area)
    assert torch.isfinite(output.baseline_log_hpwl)


def test_legacy_state_loads_with_only_optional_baseline_keys_missing() -> None:
    legacy = HCFPModel(ModelConfig(hidden_dim=12, encoder_layers=1))
    upgraded = HCFPModel(
        ModelConfig(hidden_dim=12, encoder_layers=1, baseline_enabled=True)
    )
    incompatible = upgraded.load_state_dict(legacy.state_dict(), strict=False)
    assert incompatible.unexpected_keys == []
    assert incompatible.missing_keys
    assert all(name.startswith("baseline.") for name in incompatible.missing_keys)


def test_baseline_stage_supervises_both_predictions() -> None:
    sample = _sample()
    model = HCFPModel(
        ModelConfig(hidden_dim=12, encoder_layers=1, baseline_enabled=True)
    )
    report = supervised_loss(model, sample, population=2, stage="baseline")
    assert torch.isfinite(report.total)
    assert torch.equal(report.total, report.baseline)
    assert float(report.baseline.detach()) > 0.0
    report.total.backward()
    gradients = [
        parameter.grad
        for name, parameter in model.named_parameters()
        if name.startswith("baseline.")
    ]
    assert gradients
    assert all(
        gradient is not None and torch.isfinite(gradient).all()
        for gradient in gradients
    )


def test_baseline_stage_requires_opt_in() -> None:
    with pytest.raises(ValueError, match="baseline_enabled"):
        supervised_loss(
            HCFPModel(ModelConfig(hidden_dim=12, encoder_layers=1)),
            _sample(),
            population=1,
            stage="baseline",
        )
