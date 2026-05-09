# Memory: ICCAD 2026 FloorSet-Lite Step7 Research

Last updated: 2026-05-08 Asia/Taipei

## Purpose

This file is a session memory / handoff for future Codex sessions working on the ICCAD 2026 FloorSet-Lite floorplanning solver research.

The project is research-oriented, not product implementation. Negative results are valuable. Do not force positive conclusions. Preserve artifacts, decisions, and failure attributions so later agents understand what was tried.

## User Preferences And Hard Principles

- Do not solve by case-specific rules, especially not rules tuned to case024/case025/case076.
- Do not tune scalar penalty soup or magic bbox/soft/HPWL thresholds.
- Prefer architecture-level fixes over parameter tuning.
- Prefer training/data-driven methods using FloorSet training data when possible.
- Models should learn priors, targets, actions, heatmaps, orderings, or rankings.
- Models must not bypass hard feasibility, full-case replay, or Pareto/quality gates.
- Fresh full-case official-like replay is required before counting a candidate as a real win.
- Provenance wins, projected wins, proxy wins, and closure-local wins are not fresh wins.
- Keep all experimental work sidecar-only until explicitly promoted.
- No runtime solver integration and no finalizer semantic changes unless explicitly requested.
- Original/current anchors must remain in Pareto comparisons.
- Failed, no-op, infeasible, dominated, metric-regressing, global, and timeout candidates must be reported, not silently dropped.
- The user has enough compute, roughly 48 CPU cores. Prefer parallel branches and broad sweeps when useful, but keep artifacts inspectable.

## Dataset Clarification

The official FloorSet repository provides:

- FloorSet-Lite training: 1,000,000 samples.
- FloorSet-Lite validation: 100 visible cases.
- Hidden test: 100 cases.
- Cases contain 21 to 120 blocks.

The number 120 is the maximum block count, not the number of samples.

Step7DATA / Step7ML-G verified local FloorSet-Lite training access:

- Training samples loaded: 10,000.
- `fp_sol` contract validated: 10,000 / 10,000.
- Extracted macro closures: 44,884.
- Region heatmap examples: 10,000.
- Candidate quality examples from Step7 artifacts: 146.

Important distinction:

- FloorSet training labels teach general layout / macro closure / heatmap priors.
- Step7 artifact labels teach sidecar candidate quality, e.g. Step7N-I gate pass, dominated, metric-regressing, official-like improving.
- Do not mix these as one ambiguous target.

## Current Architecture

The current Step7 research architecture is:

```text
diagnose
-> generate alternatives
-> route by locality
-> legalize / repack
-> full-case fresh metric replay
-> constrained Pareto / quality gate
-> decision
```

Current main foundation:

- Step7G: spatial locality routing.
- Step7ML-P: full-case obstacle-aware macro repacker.
- Step7ML-J / Step7N-I: constrained Pareto / quality gate selector.
- Step7DATA / Step7ML-G: real FloorSet training data access and data mart.
- Step7P-CCR: causal move-intent and affected-closure repack sidecar, currently
  blocked at Phase3 bounded replay quality.

Current blocker:

```text
Full-case legality is possible, but non-case candidates tend to be HPWL/cost regressing.
The active Step7P blocker is causal operator quality and target-direction quality, not request coverage, selector quality, or GNN/RL capacity.
```

## Major Experimental History

### Step7G: Spatial Locality Routing

Result:

- Classified candidate moves as local / regional / macro / global.
- Prevented global moves from being sent to bounded-local repair.

Interpretation:

- Routing works as safety layer.
- It does not generate new useful candidates by itself.

### Step7H / Step7C-thin / Step7C-real-A..F

Result:

- Local slack-fit / HPWL-directed edits are safe.
- Local sequential iteration quickly starves.
- Step7C-local-iter0 selected only 2 edits across 8 cases.

Interpretation:

- Local lane is useful for detailed refinement.
- It is not enough as the main architecture for large/XL or macro/topology moves.

### Step7I / Step7I-R / Step7I-S

Result:

- Coarse region flow produced regional/macro structure.
- Raw regional edits were infeasible.
- Regional slot legalizer could make some candidates feasible but not improving.

Interpretation:

- Region/topology signal exists.
- Macro/MIB/group closure and internal repacking are the next barrier.

### Step7N-G / Step7N-I

Step7N-G integrated retrieval-guided targets, macro-aware slot matching, macro internal repack, and archive reporting.

Step7N-I added quality filtering:

- Preserved 3 / 3 official-like winners.
- Preserved 5 / 5 non-anchor Pareto rows.
- Removed dominated-by-original rows from archive selection.

Interpretation:

- Quality gate / archive filtering is useful.
- It cannot create new winners.
- Generator quality remains the bottleneck.

### Step7ML-G / H / I / J / K

Step7ML-G:

- Built training-backed macro data mart from 10k FloorSet training samples.

Step7ML-H:

- Supervised x/y macro closure layout model improved validation MAE slightly.
- Failed transfer: 71 / 71 Step7P-style candidates overlapped.

Interpretation:

- Independent x/y regression is the wrong output form.

Step7ML-I:

- Geometry-aware slot/shelf decoding fixed overlap:
  - Hard-feasible non-noop: 67 vs Step7P baseline 11.
  - Overlap after decode: 4 vs Step7ML-H 71.
- But official-like improving stayed at 2.

Step7ML-J:

- Metric-aware selector compressed 67 feasible candidates to 6 clean candidates.
- Selected dominated/regressing rows dropped to 0.
- Winner count stayed 2.

Step7ML-K:

- Invariant-preserving decoder reduced bbox/soft regressions:
  - Bbox regression: 56 -> 28.
  - Soft regression: 114 -> 57.
  - HPWL gain but official-like loss: 56 -> 28.
- Winner count still stayed 2.

Interpretation:

- Architecture constraints help.
- Candidate pool still lacks enough metric-improving diversity.

### Step7L / M / N / O / P-CCR

Step7L built a learning-guided target/heatmap sidecar and Step7M tested
objective-aligned corridors, but meaningful winners stayed at 0. Step7N archive
lineage showed no strict non-micro archive source. Step7O calibrated a
training-demand prior, but Phase2 concluded `decision=keep_prior_report_only`,
`phase3_gate_open=false`, `gnn_rl_gate_open=false`, and winner concentration
failed.

Oracle reviewed the Step7O stagnation evidence and matched the local diagnosis:
the missing component is a causal move/repack operator, not another selector,
prior calibration, or GNN/RL launch.

Step7P-CCR status on 2026-05-08:

- Phase0 stagnation lock passed: existing candidate universe has no new winner
  signal; Step7O Phase3, Step7N reservoir reopen, Step7M micro widening, and
  GNN/RL remain forbidden.
- Phase1 causal atlas passed: 325 subproblems, 8 represented cases, 6 nonzero
  intent families, largest case share 0.32.
- Phase2 synthetic causal repacker passed: 5 fixtures, no-overlap/area/fixed/MIB/
  boundary guards passed, HPWL-only soft-regression candidate rejected.
- Phase3 request deck coverage was repaired with exact non-forbidden closure
  fallback rows for case `51`: 120 unique non-micro requests, 8 represented
  cases, forbidden term count 0, `phase3_replay_gate_open=true`.
- Phase3 bounded replay failed: 79 hard-feasible non-noop rows, 41 overlap-after-
  splice failures, 18 all-vector-nonregressing rows, soft regression rate 0.525,
  bbox regression rate 0.15833333333333333, and 0 strict meaningful winners.
- Phase4 ablation is intentionally blocked by the replay gate until a redesigned
  causal operator passes Phase3 replay.
- A three-branch operator-source experiment found one partial improvement:
  `branch_c_balanced_failure_budget` keeps 96 requests and 8 cases while reducing
  overlap 41 -> 36, soft regression 0.525 -> 0.2916666666666667, and bbox
  regression 0.15833333333333333 -> 0.0. It still has 0 strict meaningful
  winners and `phase4_gate_open=false`.

Next safe work: Step7Q constrained GNN/RL operator learning is open via
`artifacts/research/step7p_gnn_rl_readiness_summary.json` with
`gnn_rl_gate_open=true`. Use `branch_c_balanced_failure_budget` as the partial
baseline and learn overlap/soft/bbox/strict-winner operator policy sidecar-only.
Rerun request/replay/branch gates and only then run Phase4; contest runtime,
finalizer changes, and direct coordinate RL remain forbidden.

### Step7ML-M / N / O / P

Step7ML-M tried representation-based macro closure floorplanning:

- Sequence-pair micro decoder: diversity, no wins.
- B-tree/O-tree micro search: some diversity, no standalone promotion.
- CP-SAT / NoOverlap fallback: strongest signal, recovered overlap/no-op failures.
- WFC-style retrieval constraints: domain pruning, not standalone.

Step7ML-N:

- Integrated representation proposals + fallback exact local legalizer.
- Found that closure-local candidates were not enough for fresh full-case metrics.

Step7ML-O:

- Built full-case reconstruction and fresh replay.
- Reconstructed 37 candidates with fresh metrics.
- 28 / 37 overlapped outside closure after splice.
- 9 / 37 hard-feasible non-noop all regressed HPWL/cost.

Interpretation:

- Full-case replay works.
- Closure-local legal does not imply full-case legal.

Step7ML-P:

- Added full-case-aware obstacle repacker.
- Key improvement:
  - Full-case overlap after splice: 28 -> 0.
  - Fresh official-like improving: 0 -> 3.
  - Fresh quality gate pass: 0 -> 3.
  - Dominated by original: 37 -> 6.
  - Metric-regressing: 37 -> 6.
- Winners currently concentrated in case025.

Interpretation:

- Full-case obstacle-aware generation is the correct legality architecture.
- Remaining blocker is target direction / HPWL alignment / cross-case generalization.

### Step7ML-Q / QD / T / U / Step7R

Step7ML-Q:

- Broader proposal diversity preserved Step7ML-P baseline but did not improve.
- Winners remained case025 only.

Step7ML-QD:

- Most proposal windows timed out or had no solution.
- Solved proposals were case025 only.

Step7ML-T/U:

- Target prior allowed non-case025 full-case feasible rows, especially case024.
- But all non-case025 solved rows were HPWL/cost regressing.

Step7R:

- Tested staged placement ideas inspired by global placement -> legalization -> detailed placement.
- Branches:
  - relaxed target field,
  - displacement-preserving legalizer,
  - adaptive epsilon Pareto scheduler,
  - multi-objective representation search.
- Results:
  - Legalizer and representation diversity were not the main blocker.
  - Non-case candidates can be legal but are HPWL/cost regressing.

Interpretation:

```text
The current bottleneck is not overlap, replay, or selector.
It is target direction / objective alignment.
```

## Paper-Inspired Direction

Relevant ideas already discussed:

- ChiPFormer: offline decision transformer learns transferable placement policy from fixed offline data.
- MaskPlace: visual representation + action masking for valid placement actions.
- DREAMPlace / RePlAce / ePlace: analytical placement with wirelength + density relaxation before legalization.
- Graph placement / AlphaChip-style methods: graph representation and transferable placement policy, but do not blindly copy claims.
- WireMask-BBO: wire-mask-guided macro placement / fine-tuning existing placements.
- FloorSet: 1M training samples with realistic constraints, suitable for ML-driven floorplanning research.

Core lesson:

```text
Use training/data-driven models to learn where/what to try.
Use architecture-constrained decoders/legalizers to enforce legality.
Use full-case fresh replay and Pareto gates to validate.
```

## Current Recommended Direction

The next phase should not be case-specific. Case025 is only a positive exemplar, not an algorithm target.

Recommended architecture:

```text
FloorSet training data
-> topology / wire-demand / target heatmap prior
-> masked action or target policy
-> full-case obstacle-aware repacker
-> fresh full-case metrics
-> Step7N-I / Step7ML-J constrained Pareto gate
```

Recommended branches:

1. Supervised wire/demand heatmap target model.
2. Masked action target policy.
3. WireMask / net-demand map prior.
4. Offline trajectory / decision-transformer data builder.
5. Training-augmented candidate/target ranker.

These branches should use FloorSet training data, not just 8 Step7 focus cases.

## Judgment Chain For Future Agents

Use this decision chain:

```text
1. Does the method use case-id-blind, general features?
   If not, reject or rewrite.

2. Does it use FloorSet training data or produce training-compatible labels?
   If not, justify why.

3. Does it propose targets/actions/priors rather than bypassing legality?
   If not, require hard checks.

4. Does full-case obstacle-aware replay produce fresh metrics?
   If not, do not count wins.

5. Does it improve official-like metrics or quality gate passes cross-case?
   If not, report as negative or infrastructure evidence.

6. Does it avoid scalar penalty soup and magic thresholds?
   If not, redesign as architecture / Pareto / mask / learned prior.
```

## Merge / Artifact Guidance

- Do not whole-branch merge experimental worktrees.
- Cherry-pick only sidecar files with clear value.
- Keep downloaded FloorSet data and model checkpoints uncommitted.
- Generated artifacts may be ignored by git; document exact paths and regeneration commands.
- Commit docs/modules/scripts/tests for promoted sidecar infrastructure.

## Suggested Next Prompt Topic

The next useful controller task is likely:

```text
Step7S-Learning:
Parallel training-guided macro target experiments
```

Goal:

```text
Use FloorSet training data to learn topology-demand / wire-demand / masked target priors,
then feed targets through Step7ML-P full-case obstacle-aware repacker and Step7N-I/J quality gate.
```

Expected output:

- Fresh full-case official-like metrics.
- Cross-case validation.
- Explicit negative-result reporting.
- No case-specific rules.



## 2026-05-08 Step7Q-A data mart

Step7Q-A non-leaky operator-learning data mart passed: `artifacts/research/step7q_operator_data_mart_summary.json` reports `decision=promote_to_constrained_risk_ranking`, `example_count=325`, `feature_label_leakage_count=0`, `strict_meaningful_positive_count=0`, and `allowed_next_phase=step7q_constrained_risk_ranking`. Next work is constrained risk/ranking, not direct coordinate RL or Phase4.

## 2026-05-08 Step7Q-B constrained risk/ranking smoke

Step7Q-B completed in `/home/hwchen/PROJ/CadC26`. Files added: `src/puzzleplace/ml/step7q_operator_policy_smoke.py`, `scripts/step7q_score_operator_policy_smoke.py`, `tests/test_step7q_operator_policy_smoke.py`. Step7Q-A mask bug fixed in `src/puzzleplace/ml/step7q_operator_learning.py`: request `forbidden` declarations are safety metadata and should not count as selected forbidden operator terms. Rebuilt artifacts: `step7q_operator_examples.jsonl`, label/feature/data-mart summaries, plus `step7q_operator_policy_scores.jsonl`, `step7q_selected_source_deck.jsonl`, and `step7q_operator_policy_summary.json/md`. Current summary: `decision=risk_ranking_smoke_pass_strict_gate_closed`, `selected_request_count=96`, `represented_case_count=8`, `largest_case_share=0.25`, `forbidden_request_term_count=0`, `score_feature_label_leakage_count=0`, `overlap_after_splice_count=23`, `soft_regression_rate=0.052083333333333336`, `bbox_regression_rate=0.0`, `actual_all_vector_nonregressing_count=66`, `strict_meaningful_winner_count=0`, `phase4_gate_open=false`. Next work: constrained operator parameter expansion under the risk scorer; do not open Phase4 or touch runtime/finalizer until strict winners appear and replay gate passes.

## 2026-05-08 Step7Q-C operator-parameter expansion

Step7Q-C completed in `/home/hwchen/PROJ/CadC26`. Added `src/puzzleplace/ml/step7q_operator_parameter_expansion.py`, `scripts/step7q_expand_operator_parameters.py`, and `tests/test_step7q_operator_parameter_expansion.py`. Artifacts: `artifacts/research/step7q_parameter_expansion_candidates.jsonl`, `step7q_parameter_expansion_deck.jsonl`, `step7q_parameter_expansion_summary.json/md`. Summary: `decision=parameter_expansion_deck_ready_for_fresh_replay`, `candidate_count=550`, `selected_expansion_count=96`, `represented_case_count=8`, `largest_case_share=0.25`, `unique_parent_source_count=69`, `unique_action_signature_count=8`, `forbidden_action_term_count=0`, `direct_coordinate_field_count=0`, `fresh_replay_required=true`, `strict_winner_evidence_count=0`, `phase4_gate_open=false`, `allowed_next_phase=step7q_fresh_metric_replay_executor`. Next work: implement fresh metric replay/executor for the expansion deck; do not open Phase4 or touch contest runtime/finalizer/direct-coordinate RL until strict winner evidence exists.

## 2026-05-08 Step7Q-D fresh metric replay

Step7Q-D completed in `/home/hwchen/PROJ/CadC26`. Added `src/puzzleplace/ml/step7q_fresh_metric_replay.py`, `scripts/step7q_replay_parameter_expansion.py`, and `tests/test_step7q_fresh_metric_replay.py`. Artifacts: `artifacts/research/step7q_fresh_metric_replay_rows.jsonl`, `step7q_fresh_metric_replay_summary.json/md`, `step7q_fresh_metric_failures_by_case.json`. Summary: `decision=fresh_replay_executed_strict_gate_closed`, `request_count=96`, `fresh_metric_available_count=9`, `fresh_hard_feasible_nonnoop_count=9`, `overlap_after_splice_count=87`, `soft_regression_rate=0.020833333333333332`, `bbox_regression_rate=0.0`, `actual_all_vector_nonregressing_count=7`, `strict_meaningful_winner_count=0`, `risk_replay_gate_open=false`, `phase4_gate_open=false`. Next bottleneck is geometry realization: implement obstacle-aware local vacancy/slot search for finite Step7Q actions before more learning. Keep runtime/finalizer/Phase4 frozen.

## 2026-05-08 Step7Q-E slot-aware replay

Step7Q-E completed in `/home/hwchen/PROJ/CadC26`. `src/puzzleplace/ml/step7q_fresh_metric_replay.py` and `scripts/step7q_replay_parameter_expansion.py` now support `--slot-aware`, using local non-overlap slot search inside sidecar replay only. Artifacts: `artifacts/research/step7q_slot_aware_replay_rows.jsonl`, `step7q_slot_aware_replay_summary.json/md`, `step7q_slot_aware_failures_by_case.json`. Summary: `decision=fresh_replay_executed_strict_gate_closed`, `request_count=96`, `fresh_metric_available_count=96`, `fresh_hard_feasible_nonnoop_count=96`, `overlap_after_splice_count=0`, `slot_adjusted_count=87`, `soft_regression_rate=0.9270833333333334`, `bbox_regression_rate=0.90625`, `hpwl_regression_rate=0.90625`, `actual_all_vector_nonregressing_count=7`, `strict_meaningful_winner_count=0`, `risk_replay_gate_open=false`, `phase4_gate_open=false`. Next bottleneck: objective-aware slot scoring/selection under non-overlap guard; keep runtime/finalizer/Phase4 frozen.

## 2026-05-08 Step7Q-F objective-aware slot replay

Step7Q-F completed in `/home/hwchen/PROJ/CadC26`. `src/puzzleplace/ml/step7q_fresh_metric_replay.py` and `scripts/step7q_replay_parameter_expansion.py` now support `--objective-aware-slot`, selecting from feasible non-overlap slots using fresh objective deltas. Artifacts: `artifacts/research/step7q_objective_slot_replay_rows.jsonl`, `step7q_objective_slot_replay_summary.json/md`, `step7q_objective_slot_failures_by_case.json`. Summary: `decision=fresh_replay_executed_strict_gate_closed`, `request_count=96`, `fresh_metric_available_count=96`, `fresh_hard_feasible_nonnoop_count=96`, `overlap_after_splice_count=0`, `slot_adjusted_count=87`, `soft_regression_rate=0.71875`, `bbox_regression_rate=0.6979166666666666`, `hpwl_regression_rate=0.6979166666666666`, `actual_all_vector_nonregressing_count=27`, `strict_meaningful_winner_count=0`, `risk_replay_gate_open=false`, `phase4_gate_open=false`. Next bottleneck: generate better local objective-preserving vacancy candidates; keep runtime/finalizer/Phase4 frozen.
