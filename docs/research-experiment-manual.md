# Research Experiment Manual

## Goal
This repository is currently tuned for Milestone 1 proof rather than final contest score.
The current objective is to demonstrate:

1. pseudo trajectories are replayable
2. candidate coverage is measurable
3. BC can learn single-step typed actions
4. rollout can place at least some blocks
5. the official contest interface is wired end-to-end

## Current Pipeline
1. Bootstrap and dataset smoke:
   - `python scripts/download_smoke.py`
2. Candidate coverage:
   - `PYTHONPATH=src python scripts/check_candidate_coverage.py`
3. Behavior cloning overfit:
   - `PYTHONPATH=src python scripts/train_bc_small.py`
4. Rollout smoke:
   - `PYTHONPATH=src python scripts/rollout_validate.py`
5. Aggregated reporting / ablation:
   - `PYTHONPATH=src python scripts/run_agent10_ablation.py`
6. Advantage-weighted BC:
   - `PYTHONPATH=src python scripts/train_awbc_small.py`
7. Contest optimizer validation/eval:
   - `PYTHONPATH=src python scripts/evaluate_contest_optimizer.py`
8. Full smoke/regression matrix:
   - `PYTHONPATH=src python scripts/run_smoke_regression_matrix.py`

## Artifact Map
- `artifacts/reports/agent6_candidate_coverage_validation0.json`
- `artifacts/reports/agent8_bc_overfit_validation0_4.json`
- `artifacts/reports/agent9_rollout_validation0_4.json`
- `artifacts/reports/agent10_ablation_validation0_4.json`
- `artifacts/reports/agent11_awbc_validation0_4.json`
- `artifacts/reports/agent12_contest_validation0_4.json`
- `artifacts/reports/agent13_regression_matrix.json`
- `artifacts/models/agent11_awbc_policy.pt`

## Recommended Experiment Order
1. Run the candidate coverage report first.
2. Run plain BC and compare it against AWBC.
3. Run rollout validation after each policy update.
4. Only then evaluate the contest optimizer wrapper.

## How To Compare BC vs AWBC
- Compare `primitive_accuracy` and `block_accuracy`.
- Compare rollout mean placed blocks on the same validation slice.
- Treat improved block selection accuracy as more important than reduced loss alone.

## Current Bottlenecks
- Strict coverage remains low even when semantic coverage is high.
- One validation case still falls back after repair/finalization.
- AWBC improves weighting, but it does not remove fallback reliance by itself.

## Contest Integration Notes
- `contest_optimizer.py` is the official entrypoint file for validator/evaluator runs.
- If `artifacts/models/agent11_awbc_policy.pt` exists, the optimizer loads it.
- If the checkpoint is absent, the optimizer falls back to a deterministic heuristic policy.
- The solver now routes `semantic rollout -> repair/finalizer -> strict check -> last-resort fallback`.

## Validation Discipline
- Keep using `PYTHONPATH=src` unless the package is installed editable.
- Run the scripts with the currently activated interpreter/environment; the regression matrix now uses `sys.executable`.
- Re-run validator/evaluator after any optimizer or rollout change.
- Use `agent13_regression_matrix.json` as the compact readiness snapshot.

## Next Research Targets
1. Improve inference-time candidate families rather than relying on teacher hints.
2. Use stronger step/state value signals than simple legality-aware negatives.
3. Save and compare more than one checkpoint family, not just the latest AWBC run.
4. Replace fallback packing in the contest optimizer with a constructive legal completion strategy.
