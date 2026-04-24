# Step6D Experiment Plan — Relation-Aware Action-Q Generalization

## Requirements Summary

Current Lane 13 result shows that per-case normalization plus `RelationAwareGraphStateEncoder` is wired, but 5-case and 10-case LOCO still fail:

| lane | cases | pools | mean rank | regret | top1 | gate |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Lane 13 relation-aware + per-case | 5 | 120 | 5.6833 | 0.9444 | 0.1667 | fail |
| Lane 13 relation-aware + per-case | 10 | 240 | 6.2542 | 1.1657 | 0.1042 | fail |

Operating constraints:

- Do not tune fusion weight.
- Keep Wave 2 / contest runtime locked.
- Keep `quality_cost_runtime1`, `HPWLgap`, `Areagap_bbox`, and `Violationsrelative` as the primary evidence.
- Validate every promoted variant on both 5-case and 10-case LOCO.
- Treat zero-top1 held-out cases as hard evidence of non-transfer.

## Core Hypothesis

The current blocker is not score fusion or raw feature scale. It is that the ranker has not learned a transferable relation-aware action-value function. The next experiment should move from hand-crafted feature-row ranking toward a state/action representation that directly compares candidate actions inside the same state.

## Decision Drivers

1. **Transfer over local fit**: a variant is only useful if 10-case LOCO improves, not if it only overfits 5-case.
2. **Action-value target quality**: labels must express robust preference / regret, not brittle exact top-1 index only.
3. **Representation invariance**: state/action encoding must normalize case scale and encode block-block, pin-block, target-block, and partial-layout context.
4. **Fast failure**: run micro-overfit and 5-case smoke before spending on 10-case full LOCO.

## Viable Options

### Option A — Target-only reformulation on existing features

Use the existing `relational_state_pool_no_raw_logits` feature mode and relation-aware policy, but change the training target:

- `soft_quality` listwise target;
- pairwise quality-margin loss;
- top-k acceptable set rather than hard oracle CE.

Pros:
- Fastest experiment.
- Uses existing script surface.
- Isolates whether hard top1 CE is the main issue.

Cons:
- Still relies on hand-crafted feature rows.
- Does not solve missing state/action representation if that is the dominant blocker.

Use as a cheap baseline, not as the final architecture.

### Option B — True relation-aware action-Q ranker

Add a ranker that consumes encoded state plus candidate actions:

- `RelationAwareGraphStateEncoder(case, placements, state_step)` produces block embeddings and graph embedding.
- Candidate action encoder uses:
  - source block embedding;
  - target block embedding when present;
  - primitive embedding;
  - normalized candidate geometry;
  - relation deltas to pins / connected blocks / placed neighbors;
  - partial-layout summary.
- Candidate-set self-attention compares all candidates in the same pool.
- Output is listwise score per candidate.

Pros:
- Directly addresses the identified representation problem.
- Removes dependence on raw policy-logit calibration.
- Can still be audited without touching contest runtime.

Cons:
- More implementation work.
- Needs careful micro-overfit tests before LOCO.

This is the favored next experiment.

### Option C — Rollout-return labels

Replace immediate `_quality_after_action` labels with short/full continuation labels:

- apply candidate action;
- continue with fixed policy/heuristic for N steps or to completion;
- evaluate final `quality_cost_runtime1`;
- use the final return as action-Q target.

Pros:
- Targets the long-horizon consequence that immediate labels may miss.
- Directly diagnoses whether area/violation regret is delayed.

Cons:
- More expensive.
- Label noise depends on continuation policy.

Use after Option B micro-overfits, or as a parallel diagnostic on worst held-out cases.

## Decision

Execute a staged B-first plan with A as a cheap target ablation and C as a diagnostic extension:

1. Run target-only ablations to establish whether hard CE is the main blocker.
2. Implement a true relation-aware action-Q ranker.
3. Validate 5-case and 10-case LOCO with the same command shape.
4. If still failing, add rollout-return labels for worst held-out cases before any architecture widening.

## Experiment Matrix

### Stage 0 — Baseline freeze and diagnostic cleanup

Purpose: keep comparisons honest and avoid chasing noisy shift columns.

Tasks:

- Record Lane 12 and Lane 13 baselines in one summary table.
- Add diagnostic handling so feature-shift reports label near-zero-variance features instead of over-interpreting standardized deltas from constant per-case features.
- Preserve current artifacts:
  - `artifacts/research/step6c_relation_aware_per_case_loco5.{json,md}`
  - `artifacts/research/step6c_relation_aware_per_case_loco10.{json,md}`

Acceptance:

- Summary contains best prior 5-case baseline: rank `5.3417`, regret `0.6915`, top1 `0.1667`.
- Summary contains best prior 10-case baseline: rank `5.9375`, regret `1.3659`, top1 `0.1167`.
- No source/runtime behavior changes.

### Stage 1 — Target ablation smoke

Purpose: determine whether the main issue is hard top1 CE target brittleness.

Commands:

```bash
.venv/bin/python scripts/run_step6c_hierarchical_action_q_audit.py \
  --case-ids 0 1 2 3 4 \
  --leave-one-case-out \
  --policy-seeds 0 1 2 \
  --feature-mode relational_state_pool_no_raw_logits \
  --feature-normalization per_case \
  --encoder-kind relation_aware \
  --ranker-kind scalar \
  --target-kind soft_quality \
  --quality-temperature 0.5 \
  --policy-epochs 120 \
  --ranker-epochs 200 \
  --workers 48 \
  --output artifacts/research/step6d_target_soft_quality_loco5.json
```

Then repeat on cases `0..9` only if the 5-case result improves over Lane 13 in at least 2 of 3 metrics: mean rank, regret, top1.

Acceptance:

- If soft target does not improve 5-case, hard CE is not the primary blocker.
- If soft target improves 5-case but not 10-case, record as target helps local fit but not transfer.
- Do not tune temperatures beyond one bounded comparison (`0.5` and optionally `1.0`); this must not become a new weight sweep.

### Stage 2 — Implement relation-aware action-Q ranker

Purpose: replace feature-row scalar ranking with a transferable state/action ranker.

Suggested files:

- Add model code in `src/puzzleplace/models/hierarchical.py` or a new `src/puzzleplace/models/action_q.py`:
  - `CandidateRelationalActionQRanker`
  - `ActionCandidateEncoder`
- Export from `src/puzzleplace/models/__init__.py`.
- Extend `scripts/run_step6c_hierarchical_action_q_audit.py`:
  - `--ranker-kind relational_action_q`
  - reuse existing pool collection and quality labels;
  - avoid contest runtime path changes.
- Add tests in `tests/test_hierarchical_policy.py` or `tests/test_action_q_ranker.py`.

Minimum model shape:

```text
state_encoder(case, placements, step) -> block_embeddings, graph_embedding
candidate_encoder(action, block_embeddings, graph_embedding, normalized_geometry, relation_features)
candidate_set_attention(candidate_embeddings)
score_head(candidate_context) -> score per candidate
```

Training targets:

- default: soft quality listwise distribution;
- also report hard oracle top1 CE metrics for comparability;
- pairwise margin can be added only after the basic listwise path works.

Micro acceptance before LOCO:

- same-pool overfit: oracle top1 selected fraction >= 0.90 on a tiny synthetic/tiny real pool set;
- same-case micro-overfit: mean rank <= 1.5 and top1 >= 0.75 on train pools;
- tests pass.

### Stage 3 — Dual LOCO validation

Run both 5-case and 10-case LOCO for the same promoted variant.

5-case command shape:

```bash
.venv/bin/python scripts/run_step6c_hierarchical_action_q_audit.py \
  --case-ids 0 1 2 3 4 \
  --leave-one-case-out \
  --policy-seeds 0 1 2 \
  --feature-mode relational_state_pool_no_raw_logits \
  --feature-normalization per_case \
  --encoder-kind relation_aware \
  --ranker-kind relational_action_q \
  --target-kind soft_quality \
  --policy-epochs 120 \
  --ranker-epochs 200 \
  --workers 48 \
  --output artifacts/research/step6d_relational_action_q_loco5.json
```

10-case command shape:

```bash
.venv/bin/python scripts/run_step6c_hierarchical_action_q_audit.py \
  --case-ids 0 1 2 3 4 5 6 7 8 9 \
  --leave-one-case-out \
  --policy-seeds 0 1 2 \
  --feature-mode relational_state_pool_no_raw_logits \
  --feature-normalization per_case \
  --encoder-kind relation_aware \
  --ranker-kind relational_action_q \
  --target-kind soft_quality \
  --policy-epochs 120 \
  --ranker-epochs 200 \
  --workers 48 \
  --output artifacts/research/step6d_relational_action_q_loco10.json
```

Promotion criteria:

- 5-case must beat Lane 13 on at least 2 of 3 metrics: rank `< 5.6833`, regret `< 0.9444`, top1 `> 0.1667`.
- 10-case must beat Lane 13 on at least 2 of 3 metrics: rank `< 6.2542`, regret `< 1.1657`, top1 `> 0.1042`.
- Strong pass requires meeting the existing LOCO gate: mean rank `< 4.0`, top1 `> 0.30`, no held-out case with top1 `0.0`.
- If 5-case improves but 10-case regresses, do not claim success; classify as small-slice overfit.

### Stage 4 — Worst-case rollout-return diagnostic

Trigger only if Stage 3 still has zero-top1 held-out cases.

Target cases based on current evidence:

- 5-case: `validation-2`.
- 10-case: `validation-0`, `validation-3`, `validation-5`; also inspect high-regret `validation-7`.

Tasks:

- Add a bounded `--label-kind rollout_return` or separate diagnostic script.
- For each candidate action in a pool:
  - apply candidate;
  - continue with fixed semantic/hierarchical rollout for a bounded horizon;
  - evaluate final `quality_cost_runtime1`;
  - compare immediate-label oracle vs rollout-return oracle.

Acceptance:

- Report whether immediate labels disagree with rollout-return labels on the worst held-out cases.
- If disagreement is high, prioritize label design before architecture widening.
- If disagreement is low, prioritize representation/candidate generation instead.

## Outputs

Required artifacts:

- `artifacts/research/step6d_experiment_summary.md`
- `artifacts/research/step6d_experiment_summary.json`
- Per-run `{json,md}` outputs for each LOCO run.

Each summary must include:

- 5-case and 10-case metrics side by side;
- per-held-out-case rank/top1/regret;
- objective component regret (`HPWLgap`, `Areagap_bbox`, `Violationsrelative`);
- zero-top1 case list;
- conclusion: target issue, representation issue, label-horizon issue, or mixed.

## Risks and Mitigations

| Risk | Mitigation |
| --- | --- |
| Another implicit weight sweep | Limit target-temperature comparisons to one optional extra value; no fusion-weight work. |
| 5-case improvement hides 10-case failure | Always run dual LOCO before claiming progress. |
| Per-case normalization makes feature-shift diagnostics misleading | Mark near-zero-variance features and do not treat inflated standardized deltas as root cause. |
| New model overfits tiny pools only | Require same-case micro-overfit and LOCO before promotion. |
| Rollout-return labels become too expensive | Restrict first pass to worst held-out cases and bounded horizon. |

## Verification Steps

1. Unit tests:

```bash
.venv/bin/python -m pytest tests/test_hierarchical_policy.py tests/test_policy_shapes.py -q
```

2. Full regression:

```bash
.venv/bin/python -m pytest -q
```

3. Static compile:

```bash
.venv/bin/python -m compileall -q src scripts tests
```

4. Experiment evidence:

```bash
.venv/bin/python scripts/run_step6c_hierarchical_action_q_audit.py ...loco5...
.venv/bin/python scripts/run_step6c_hierarchical_action_q_audit.py ...loco10...
```

## Stop Conditions

Stop and report rather than continuing if:

- same-case micro-overfit fails;
- target-only ablation and relational action-Q both fail to beat Lane 13 on 5-case;
- 10-case shows more zero-top1 cases than Lane 13;
- rollout-return labels reveal that immediate labels are systematically misleading.

## Expected Interpretation Matrix

| Result | Interpretation | Next move |
| --- | --- | --- |
| Soft target improves both 5/10 | target brittleness was material | keep target, then implement stronger model only if gate still fails |
| Relational action-Q improves 5 but not 10 | architecture helps local fit, transfer still weak | inspect case taxonomy / split shift / candidate diversity |
| Relational action-Q improves regret but not top1 | model finds safer actions but not oracle exact actions | use top-k / regret-weighted gate, inspect oracle equivalence |
| Rollout-return oracle differs from immediate oracle | label horizon is wrong | redesign target around rollout return |
| No variant improves over Lane 13 | representation/candidate generation still insufficient | revisit candidate-pool diversity and case-conditioned invariants |
