# Step7Q: Constrained GNN/RL Operator Learning

Step7Q is the first RL/GNN-allowed phase after Step7P. It is opened only for
**causal operator learning**, not for contest runtime integration or a direct
coordinate-generation solver.

## Gate artifact

- `artifacts/research/step7p_gnn_rl_readiness_summary.json`

Current decision on 2026-05-08:

- `decision=open_constrained_operator_learning_phase`
- `gnn_rl_gate_open=true`
- `allowed_phase=step7q_constrained_operator_learning`
- `best_branch_name=branch_c_balanced_failure_budget`

The gate opened because all readiness criteria passed:

- Phase1 atlas is broad: 325 rows, 8 cases, largest case share 0.32.
- Failure and intent labels are diverse.
- Three operator-source branches were tried.
- Branch C is a partial effective method: overlap 41 -> 36, soft regression
  0.525 -> 0.2916666666666667, bbox regression 0.15833333333333333 -> 0.0.
- Manual operators still do not open Phase4: strict meaningful winners remain 0.
- There is enough labeled corpus for learning: 107 exact hard non-forbidden rows
  and 30 all-vector-nonregressing source rows.

## Allowed scope

Step7Q may train or evaluate a GNN/RL-style sidecar policy/scorer for:

- overlap-after-splice risk;
- soft-regression risk;
- bbox-regression risk;
- all-vector-nonregressing policy selection;
- strict-winner candidate prior.

The first recommended model is a small graph/ranking policy that learns to choose
or synthesize causal operator sources from Branch C-style rows, then replays the
chosen actions through the existing Phase3 vector gates.

## Forbidden scope

Step7Q still must not do the following:

- contest runtime integration;
- finalizer semantic changes;
- real Phase4 ablation before `phase4_gate_open=true`;
- RL direct coordinate solver;
- GNN result claims without vector replay gate evidence.


## Step7Q-A data mart result

Artifacts:

- `artifacts/research/step7q_operator_examples.jsonl`
- `artifacts/research/step7q_operator_label_summary.json`
- `artifacts/research/step7q_operator_feature_summary.json`
- `artifacts/research/step7q_operator_data_mart_summary.json`
- `artifacts/research/step7q_operator_data_mart_summary.md`

Current decision on 2026-05-08:

- `decision=promote_to_constrained_risk_ranking`
- `example_count=325`
- `represented_case_count=8`
- `largest_case_share=0.32`
- `feature_label_leakage_count=0`
- `direct_coordinate_field_count=0`
- `eligible_exact_hard_nonforbidden_count=107`
- `all_vector_nonregressing_positive_count=123`
- `strict_meaningful_positive_count=0`
- `strict_supervision_enabled=false`
- `allowed_next_phase=step7q_constrained_risk_ranking`

Interpretation: Step7Q-A succeeded. The next step is a constrained risk/ranking
smoke or model, not strict-winner supervision and not direct coordinate RL.

Implementation note: Step7Q-B fixed the Step7Q-A forbidden-mask search so the
Phase3 request `forbidden` policy declaration is not counted as actual forbidden
operator usage. After rebuild, `eligible_for_selection_count=120` and
`eligible_for_training_count=253`; feature/label leakage remains 0.

## Step7Q-B constrained policy smoke result

Artifacts:

- `scripts/step7q_score_operator_policy_smoke.py`
- `artifacts/research/step7q_operator_policy_scores.jsonl`
- `artifacts/research/step7q_selected_source_deck.jsonl`
- `artifacts/research/step7q_operator_policy_summary.json`
- `artifacts/research/step7q_operator_policy_summary.md`

Current decision on 2026-05-08:

- `decision=risk_ranking_smoke_pass_strict_gate_closed`
- `selected_request_count=96`
- `represented_case_count=8`
- `largest_case_share=0.25`
- `forbidden_request_term_count=0`
- `score_feature_label_leakage_count=0`
- `overlap_after_splice_count=23` vs Branch C 36
- `soft_regression_rate=0.052083333333333336` vs Branch C
  0.2916666666666667
- `bbox_regression_rate=0.0`
- `actual_all_vector_nonregressing_count=66`
- `strict_meaningful_winner_count=0`
- `risk_profile_pass=true`
- `strict_gate_pass=false`
- `phase4_gate_open=false`
- `allowed_next_phase=step7q_constrained_operator_parameter_expansion`

Interpretation: the non-leaky feature/mask risk scorer can select a balanced
96-row operator-source deck that materially improves Branch C's risk profile,
but it still finds zero strict meaningful winners. The project is no longer
stuck on risk filtering; it is stuck on generating/parameterizing operators that
can create strict winners under the same risk guard.

## Promotion gate out of Step7Q

A Step7Q model/policy may reopen Phase3 replay only if it produces a new request
or operator-source deck satisfying:

- at least 96 requests;
- at least 8 represented cases;
- largest case share <= 0.25;
- forbidden request term count = 0;
- overlap-after-splice count < 36;
- soft regression rate < 0.2916666666666667;
- bbox regression rate = 0.0;
- strict meaningful winner count >= 3.

Only after that may Phase4 be considered.

## Step7Q-C constrained operator-parameter expansion result

Artifacts:

- `scripts/step7q_expand_operator_parameters.py`
- `artifacts/research/step7q_parameter_expansion_candidates.jsonl`
- `artifacts/research/step7q_parameter_expansion_deck.jsonl`
- `artifacts/research/step7q_parameter_expansion_summary.json`
- `artifacts/research/step7q_parameter_expansion_summary.md`

Current decision on 2026-05-08:

- `decision=parameter_expansion_deck_ready_for_fresh_replay`
- `parent_source_deck_count=96`
- `candidate_count=550`
- `selected_expansion_count=96`
- `represented_case_count=8`
- `largest_case_share=0.25`
- `unique_parent_source_count=69`
- `unique_action_signature_count=8`
- `forbidden_action_term_count=0`
- `direct_coordinate_field_count=0`
- `policy_risk_profile_pass=true`
- `representation_pass=true`
- `fresh_replay_required=true`
- `strict_winner_evidence_count=0`
- `phase4_gate_open=false`
- `allowed_next_phase=step7q_fresh_metric_replay_executor`

Interpretation: Step7Q-C converts the low-risk Step7Q-B source deck into finite
operator-action variants without direct coordinates or forbidden action terms.
It does not claim strict winners because the variants have not been executed by
a geometry/fresh-metric replay engine. The next bottleneck is implementing that
fresh replay executor for the expansion deck, not opening Phase4.

## Step7Q-D fresh-metric replay result

Artifacts:

- `scripts/step7q_replay_parameter_expansion.py`
- `artifacts/research/step7q_fresh_metric_replay_rows.jsonl`
- `artifacts/research/step7q_fresh_metric_replay_summary.json`
- `artifacts/research/step7q_fresh_metric_replay_summary.md`
- `artifacts/research/step7q_fresh_metric_failures_by_case.json`

Current decision on 2026-05-08:

- `decision=fresh_replay_executed_strict_gate_closed`
- `request_count=96`
- `fresh_metric_available_count=9`
- `fresh_hard_feasible_nonnoop_count=9`
- `overlap_after_splice_count=87`
- `soft_regression_rate=0.020833333333333332`
- `bbox_regression_rate=0.0`
- `actual_all_vector_nonregressing_count=7`
- `strict_meaningful_winner_count=0`
- `represented_case_count=8`
- `unique_replayed_signature_count=3`
- `risk_replay_gate_open=false`
- `phase4_gate_open=false`
- `allowed_next_phase=null`

Interpretation: Step7Q-D proves the next real bottleneck is not ranking or
finite-action selection. The low-risk deck collapses at geometry realization:
87/96 finite actions overlap after splice, leaving only 9 fresh-metric rows and
zero strict winners. The next safe direction is an obstacle-aware target/slot
executor for Step7Q actions, with local vacancy search before metric replay.

## Step7Q-E obstacle-aware slot replay result

Artifacts:

- `scripts/step7q_replay_parameter_expansion.py --slot-aware`
- `artifacts/research/step7q_slot_aware_replay_rows.jsonl`
- `artifacts/research/step7q_slot_aware_replay_summary.json`
- `artifacts/research/step7q_slot_aware_replay_summary.md`
- `artifacts/research/step7q_slot_aware_failures_by_case.json`

Current decision on 2026-05-08:

- `decision=fresh_replay_executed_strict_gate_closed`
- `request_count=96`
- `fresh_metric_available_count=96`
- `fresh_hard_feasible_nonnoop_count=96`
- `overlap_after_splice_count=0`
- `slot_adjusted_count=87`
- `soft_regression_rate=0.9270833333333334`
- `bbox_regression_rate=0.90625`
- `hpwl_regression_rate=0.90625`
- `actual_all_vector_nonregressing_count=7`
- `strict_meaningful_winner_count=0`
- `risk_replay_gate_open=false`
- `phase4_gate_open=false`

Interpretation: obstacle-aware slotting solves the geometric overlap bottleneck
from Step7Q-D, but the feasible slots are far from the intended objective
movement and cause severe HPWL/bbox/soft regressions. The next bottleneck is not
legality; it is objective-aware slot scoring/selection under the same
non-overlap guard.

## Step7Q-F objective-aware slot replay result

Artifacts:

- `scripts/step7q_replay_parameter_expansion.py --objective-aware-slot`
- `artifacts/research/step7q_objective_slot_replay_rows.jsonl`
- `artifacts/research/step7q_objective_slot_replay_summary.json`
- `artifacts/research/step7q_objective_slot_replay_summary.md`
- `artifacts/research/step7q_objective_slot_failures_by_case.json`

Current decision on 2026-05-08:

- `decision=fresh_replay_executed_strict_gate_closed`
- `request_count=96`
- `fresh_metric_available_count=96`
- `fresh_hard_feasible_nonnoop_count=96`
- `overlap_after_splice_count=0`
- `slot_adjusted_count=87`
- `objective_aware_slot_replay=true`
- `soft_regression_rate=0.71875`
- `bbox_regression_rate=0.6979166666666666`
- `hpwl_regression_rate=0.6979166666666666`
- `actual_all_vector_nonregressing_count=27`
- `strict_meaningful_winner_count=0`
- `risk_replay_gate_open=false`
- `phase4_gate_open=false`

Interpretation: objective-aware slot selection improves Step7Q-E materially
(all-vector nonregressing rows 7 -> 27; soft/bbox/hpwl regression rates drop),
but remains far from the risk replay gate and still has zero strict meaningful
winners. The next bottleneck is candidate-set quality: the slot generator must
produce local objective-preserving vacancies, not just score a poor candidate
pool better.
