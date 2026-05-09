# Step7P-CCR: Causal Closure Repack

Step7P-CCR is a sidecar research lane created after Step7O Phase2 and the
Oracle stagnation diagnosis.  It is the current Step7 handoff point as of
2026-05-08.

## Decision boundary

Step7P is **not**:

- Step7O Phase3 masked replay;
- Step7N reservoir reopening;
- Step7M micro-axis or paired micro-shift widening;
- a contest runtime/finalizer integration;
- a GNN/RL training lane.

Step7P is a gated attempt to create a new candidate universe through causal move
intents and affected-closure repacking. A failed gate must stop later phases; do
not keep producing replay/ablation numbers from stale or invalid states.

## Current status on 2026-05-08

Step7P is **blocked at Phase3 bounded replay**, after the Phase3 request-source
coverage issue was repaired.

Current artifacts:

- request summary: `artifacts/research/step7p_phase3_causal_request_summary.json`
- replay summary: `artifacts/research/step7p_phase3_causal_replay_summary.json`
- replay blocker diagnosis: `artifacts/research/step7p_phase3_replay_blocker_diagnosis.json`
- Phase4 blocked summary: `artifacts/research/step7p_phase4_causal_ablation_summary.json`

Request gate result:

- `decision=promote_to_bounded_causal_replay`
- `request_count=120`
- `unique_request_signature_count=120`
- `represented_case_count=8`
- case counts: `19=16`, `24=16`, `25=16`, `51=16`, `76=16`, `79=9`, `91=16`, `99=15`
- `largest_case_share=0.13333333333333333`
- `case024_case025_request_share=0.26666666666666666`
- `non_micro_intent_share=1.0`
- `forbidden_request_term_count=0`
- `phase3_replay_gate_open=true`
- `gnn_rl_gate_open=false`

Replay gate result:

- `decision=stop_phase3_replay_gate`
- `request_count=120`
- `hard_feasible_nonnoop_count=79`
- `overlap_after_splice_count=41`
- `represented_hard_feasible_case_count=8`
- `actual_all_vector_nonregressing_count=18`
- `soft_regression_rate=0.525`
- `bbox_regression_rate=0.15833333333333333`
- `strict_meaningful_winner_count=0`
- `non_case024_non_case025_strict_winner_count=0`
- `unique_strict_winner_signature_count=0`
- `phase4_gate_open=false`
- `gnn_rl_gate_open=false`

Interpretation: request-source coverage is no longer the blocker. The current
causal request/operator source still reproduces the old quality problem: many
replay rows are not all-vector-nonregressing, overlap-after-splice remains high,
and there are zero strict meaningful winners. A source scan confirms this is not
fixable by selector/request ordering alone: among non-forbidden exact hard
source rows there are 107 eligible rows, 30 all-vector-nonregressing rows, and
0 strict meaningful source rows. Phase4, selector PRD, GNN, and RL remain closed.

Phase4 artifact hygiene:

- `artifacts/research/step7p_phase4_causal_ablation_summary.json`
  - `decision=blocked_by_phase3_replay_gate`
  - `family_count=0`
  - `passed_family_count=0`
  - `gnn_rl_gate_open=false`

## Phase0 stagnation lock

Artifacts:

- `artifacts/research/step7p_phase0_stagnation_lock.json`
- `artifacts/research/step7p_phase0_stagnation_lock.md`
- `artifacts/research/step7p_phase0_candidate_universe_audit.jsonl`

The lock records:

- `decision=start_causal_closure_repack`
- Step7O Phase3 closed;
- Step7ML-I/K winner baseline remains 2;
- Step7N strict archive candidate count remains 0;
- Step7M meaningful winner count remains 0;
- GNN/RL closed.

## Phase1 causal subproblem atlas

Artifacts:

- `artifacts/research/step7p_phase1_causal_subproblem_atlas.jsonl`
- `artifacts/research/step7p_phase1_causal_subproblem_summary.json`
- `artifacts/research/step7p_phase1_causal_subproblem_summary.md`
- `artifacts/research/step7p_phase1_failures_by_cause.json`

Gate result:

- `decision=promote_to_synthetic_causal_repacker`
- `phase2_gate_open=true`
- `subproblem_count=325`
- `represented_case_count=8`
- `largest_case_share=0.32`
- `nonzero_intent_family_count=6`
- `unknown_failure_bucket_share=0.28615384615384615`
- `forbidden_validation_label_term_count=0`

The atlas attributes existing failures and candidates to causal mechanisms such
as soft regression, bbox regression, HPWL gain but official-like loss,
blocker-chain / wrong-slot / overlap failures, closure translation, and
boundary/MIB risks. Phase1 does not generate new placement requests.

## Phase2 synthetic deterministic causal repacker

Artifacts:

- `artifacts/research/step7p_phase2_synthetic_repacker_report.json`
- `artifacts/research/step7p_phase2_synthetic_repacker_report.md`
- `artifacts/research/step7p_phase2_operator_contract.json`

Gate result:

- `decision=promote_to_causal_request_deck`
- `phase3_gate_open=true`
- `fixture_count=5`
- all fixtures legal, non-noop, overlap-free, area-preserving;
- fixed/preplaced blocks unchanged;
- MIB equal-shape guard passed;
- boundary guard passed;
- HPWL-only soft-regression candidate rejected;
- at least 3 fixtures emit 3+ Pareto alternatives.

Phase2 proves the synthetic operator contract only. It does not prove validation
replay quality.

## Phase3 request repair and replay

Implemented files:

- `src/puzzleplace/experiments/step7p_causal_request_replay.py`
- `scripts/step7p_generate_causal_repack_requests.py`
- `scripts/step7p_replay_causal_repack_requests.py`
- `tests/test_step7p_causal_request_contract.py`
- `tests/test_step7p_causal_replay_contract.py`

The request deck repair adds a narrow fail-closed fallback for exact,
non-forbidden Step7M Phase4 closure rows when a representative case has no
non-unknown source after forbidden micro/soft-budgeted filters. This restored
case `51` without allowing `micro_axis_corridor`, `soft_repair_budgeted`, or
HPWL-only request policies.

The replay gate failed, so the current line stops here.

## Phase4 family ablation

Implemented files exist for the ablation contract, but Phase4 is blocked until
Phase3 replay passes:

- `scripts/step7p_run_causal_family_ablation.py`
- `tests/test_step7p_causal_ablation.py`

The ablation runner now checks the sibling replay summary and writes a blocked
summary when `phase4_gate_open=false`, preventing accidental stale family
ablation claims.


## Three-branch operator redesign result

Artifact:
`artifacts/research/step7p_operator_branch_summary.json`.

Three method branches were compared against the Phase3 replay baseline:

1. `branch_a_overlap_zero_hard_only` — keeps hard-feasible exact rows only.
   It reduces overlap to 0 but worsens soft/bbox regression, so it is not a
   balanced method.
2. `branch_b_vector_guarded_narrow` — keeps all-vector-nonregressing rows only.
   It removes overlap/soft/bbox regression but covers only 30 requests and 3
   cases, so it is too narrow.
3. `branch_c_balanced_failure_budget` — mixes 60 hard-feasible exact rows with
   bounded non-hard rows to preserve 96 requests and 8 cases while reducing all
   three target failures. This is the best partial operator-source method.

Best branch metrics:

- `best_branch_name=branch_c_balanced_failure_budget`
- `request_count=96`
- `represented_case_count=8`
- `hard_feasible_nonnoop_count=60`
- `largest_case_share=0.25`
- `overlap_after_splice_count=36` vs baseline 41
- `soft_regression_rate=0.2916666666666667` vs baseline 0.525
- `bbox_regression_rate=0.0` vs baseline 0.15833333333333333
- `strict_meaningful_winner_count=0`
- `phase4_gate_open=false`

Decision: `partial_operator_source_found_phase4_still_closed`. Branch C is
effective for the stated near-term objective of reducing overlap/soft/bbox, but
it is not enough for Phase4 because it still has zero strict meaningful winners.


## RL/GNN readiness gate

Artifact: `artifacts/research/step7p_gnn_rl_readiness_summary.json`.

Current decision:

- `decision=open_constrained_operator_learning_phase`
- `gnn_rl_gate_open=true`
- `allowed_phase=step7q_constrained_operator_learning`

This does **not** promote Phase4, contest runtime, finalizer changes, or a direct
coordinate-generation RL solver. It opens only the constrained Step7Q operator-
learning phase because hand-designed branches reached a partial plateau and the
corpus has enough labeled failure/improvement signal for a GNN/RL-style operator
policy. See `docs/step7q_constrained_operator_learning.md`.

## Next safe task

Start Step7Q constrained operator learning; do not relax replay gates:

1. Use `branch_c_balanced_failure_budget` as the partial operator-source baseline.
2. Train or smoke-test a GNN/RL-style operator policy only inside Step7Q.
3. Use learned scores/actions to produce a new request/operator-source deck.
4. Rerun request, replay, and branch comparison gates.
5. Continue to Phase4 only if replay reports `phase4_gate_open=true`.

## Verification evidence

Latest combined check from 2026-05-08:

```bash
PYTHONPATH=src .venv/bin/python -m pytest \
  tests/test_step7p_stagnation_lock.py \
  tests/test_step7p_causal_subproblem_atlas.py \
  tests/test_step7p_causal_repacker_synthetic.py \
  tests/test_step7p_causal_request_contract.py \
  tests/test_step7p_causal_replay_contract.py \
  tests/test_step7p_causal_ablation.py -q
```

Expected after the current repair:

```text
7 passed
```
