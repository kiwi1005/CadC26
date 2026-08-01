from __future__ import annotations

from hcfp.training_profile import TrainingProfileConfig, run_training_profile


def test_training_profile_reports_one_forward_and_finite_timing() -> None:
    report = run_training_profile(
        TrainingProfileConfig(
            block_count=32,
            population=2,
            hidden_dim=16,
            encoder_layers=1,
            warmups=0,
            steps=2,
            device="cpu",
        )
    )

    assert report["forward_calls_per_step"] == 1
    assert report["timing_seconds"]["p50"] > 0.0
    assert report["timing_seconds"]["steps_per_second"] > 0.0
    assert report["last_loss"]["total"] > 0.0
