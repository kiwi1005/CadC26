# Step4 Final Synthesis Note (Draft)

> Status: waiting for `artifacts/research/best_of_k_oracle_reranking.{md,json}` before finalizing.

## Locked facts already validated
- Best untrained mean official cost: `19.784`
- Best trained mean official cost: `23.651`
- Top-5 trained loss cases: `validation-14/18/17/15/11`
- Top-5 drift audit is present and consistent with `AGENT_step4.md` plus tracked reports.

## Top-5 audit synthesis
- Trained checkpoints lose mainly on HPWL + bbox quality terms.
- Lower repair displacement is not a reliable proxy for better official cost.
- Current tracked evidence still points to proposal/objective mismatch, with stagewise persistence instrumentation still needed.

## Best-of-K oracle synthesis
- Pending upstream artifact landing.
- Validator is prepared to check these expected aggregate signals once the artifact appears:
  - `oracle_best` mean cost ≈ `18.355`
  - `hpwl_bbox_proxy_best` mean cost ≈ `19.316`
  - `displacement_best` mean cost ≈ `37.174`

## Exact next code files to change
1. `src/puzzleplace/rollout/semantic.py`
2. `src/puzzleplace/optimizer/contest.py`
3. `src/puzzleplace/repair/finalizer.py`
4. `scripts/run_generalization_followup.py`
5. `scripts/generate_cost_semantics_and_trained_vs_untrained_delta.py`
6. `scripts/run_best_of_k_oracle_reranking.py`

## Final recommendation
- Pending Best-of-K validation.
