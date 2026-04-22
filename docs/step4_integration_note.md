# Step4 Integration Note

## Validation status
- `top5_loss_drift_audit.md` is present in the leader workspace and is consistent with `AGENT_step4.md`, `generalization_followup_smallcheckpoints.md`, and `cost_semantics_and_trained_vs_untrained_delta.md` on the locked facts: best untrained mean cost `19.784`, best trained mean cost `23.651`, and the top-5 loss cases `validation-14/18/17/15/11`.
- `best_of_k_oracle_reranking.md` is **not present yet** in the leader workspace, so the oracle/reranking lane cannot be fully integrated or cross-validated in this pass.

## What the current evidence says
- The top-5 audit is directionally aligned with Step4: the trained failure signal is still dominated by HPWL + bbox quality terms, and lower repair displacement is not rescuing official cost.
- The present evidence is still incomplete for a full proposal-versus-repair verdict because stagewise snapshots are not persisted; the current audit explicitly flags that gap instead of over-claiming.

## Immediate next code files to change
1. `scripts/run_best_of_k_oracle_reranking.py` — complete the missing oracle/reranking artifact lane.
2. `src/puzzleplace/repair/finalizer.py` — serialize stagewise post-repair snapshots and moved-block statistics needed by the drift audit.
3. `src/puzzleplace/rollout/semantic.py` — persist semantic action traces / placement-order metadata for proposal-stage diagnosis.
4. `scripts/run_generalization_followup.py` — emit the extra stagewise metrics into research artifacts so future drift audits are data-backed instead of reconstruction-heavy.
5. `scripts/generate_cost_semantics_and_trained_vs_untrained_delta.py` — extend the report once stagewise HPWL/bbox terms and oracle comparisons exist.

## Integration recommendation
- Do **not** start another AWBC training iteration until the missing best-of-K result lands and stagewise instrumentation is added.
- Once the oracle artifact exists, re-run the integration validator and decide between a reranker/scorer path versus candidate/finalizer redesign.
