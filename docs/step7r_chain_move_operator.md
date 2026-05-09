---
status: closed
opened: 2026-05-08
closed: 2026-05-08
predecessor: docs/step7q_constrained_operator_learning.md (Step7Q-F)
close_decision: artifacts/research/step7r_close_decision.json
oracle_prompt: docs/oracle_step7r_close_prompt.md
---

# Step7R: Chain-Move Operator (Lin-Kernighan-Style k-Block Ripple)

Step7R is the next research lane after Step7Q-F stalled at
`fresh_replay_executed_strict_gate_closed` with `strict_meaningful_winner_count=0`.

## Why Step7R exists

Step7Q-A..F evolved through six bottlenecks and ended on a fundamental ceiling:

```
Step7Q-D  → 87/96 overlap-after-splice  (single-block move into target slot)
Step7Q-E  → overlap fixed by slot search; soft 0.927, bbox 0.906  (slot far from target)
Step7Q-F  → objective-aware slot scoring; soft 0.719, bbox 0.698, strict=0
```

The diagnosis that closes Step7Q-F:

> **Single-block move cannot find an objective-preserving free vacancy near
> the target.** Either the target is occupied (overlap) or the legal vacancy
> is far from target (metric regression).

Chain-move sidesteps the vacancy problem entirely: instead of moving block A
into a free slot, it displaces the block currently at A's target, which
displaces another, until the chain closes at an existing vacancy or returns
to the original anchor (rotation). This is the classical Lin-Kernighan move
adapted to 2D macro placement.

## Decision boundary

Step7R **is**:

- a sidecar causal operator family that produces multi-block coordinated moves;
- replayed through the same Phase3 vector gates as Step7P/Step7Q;
- evaluated under the same risk profile (overlap, soft, bbox, hpwl, strict).

Step7R **is not**:

- a contest runtime / finalizer integration;
- a direct-coordinate RL solver;
- an unbounded global re-placement (k is bounded);
- a Step7Q-F slot scorer replacement (Step7R reuses Step7Q-F objective scoring
  for the closing slot of each chain).

## Forbidden scope

- `k > k_max` chains (k_max starts at 3, only widened with phase-gate evidence);
- chains that touch fixed/preplaced blocks;
- chains that violate the MIB equal-shape guard or boundary guard;
- splice rows that drop original-inclusive Pareto reporting;
- claiming strict winners without fresh-metric replay evidence.

## Phase plan and gates

### Phase 0 — Stagnation lock

Snapshot the Step7Q-F result so later phases cannot drift the baseline.

Artifacts:

- `artifacts/research/step7r_phase0_stagnation_lock.json`
- `artifacts/research/step7r_phase0_stagnation_lock.md`

Required fields: `predecessor_step=step7q_f`,
`predecessor_strict_winner_count=0`,
`predecessor_soft_regression_rate=0.71875`,
`predecessor_bbox_regression_rate=0.6979166666666666`,
`predecessor_overlap_after_splice_count=0`,
`open_step=step7r_chain_move_operator`,
`gnn_rl_gate_open=true` (inherited from Step7P).

### Phase 1 — k=2 swap operator

The simplest chain (k=2) is a pure swap: block A and block B exchange centers,
no vacancy required. Phase 1 implements the swap as a deterministic operator
with the same contract as Step7P Phase2:

- legal, non-noop, overlap-free under the MIB/boundary guards;
- area-preserving (swap centers; if shapes differ, only swap when both blocks
  fit each other's footprint without overlap with neighbors);
- fixed/preplaced blocks unchanged;
- emits at least 3 swap candidates per fixture (different B partners).

Implementation:

- `src/puzzleplace/alternatives/chain_move.py` — `propose_swap_candidates(...)`,
  shared swap-validation helpers.
- `scripts/step7r_generate_swap_candidates.py` — emits a swap source deck for
  the 8 representative cases mirroring the Step7Q-B selected deck shape (96
  rows, ≤25% per case).
- `tests/test_step7r_swap_contract.py` — fixture set parallel to Step7P Phase2.

Gate to advance to Phase 2:

- `phase1_swap_legal_count == phase1_swap_request_count`;
- `phase1_strict_winner_count >= 1` OR
  `phase1_actual_all_vector_nonregressing_count > step7q_f_baseline (27)`;
- if both fail, do not extend k. Pivot or close the lane.

### Phase 2 — k=3 ripple chain

A ripple is A → slot(B) → slot(C) → vacancy_or_rotate_back. k=3 is the smallest
chain that strictly subsumes swap. Phase 2 must:

- enumerate chain heads from Step7Q-F's selected deck (96 rows);
- bound chain expansion at k_max=3 with branch budget per head;
- close each chain by either an existing legal vacancy (Step7Q-F objective slot)
  or a rotation back to the original head (closed loop);
- reuse the swap-validation helpers from Phase 1 for each pair-wise displacement;
- reject chains touching fixed/preplaced blocks or violating MIB/boundary.

Implementation:

- extend `chain_move.py` with `propose_ripple_chain_candidates(...)`;
- `scripts/step7r_generate_ripple_candidates.py`;
- `tests/test_step7r_ripple_contract.py`.

Gate to advance to Phase 3 replay:

- `phase2_chain_legal_count >= 0.7 * phase2_chain_request_count`;
- coverage at least 8 cases, largest case share ≤ 0.25;
- forbidden-action-term count = 0.

### Phase 3 — Fresh-metric replay

Reuse the Step7Q-D fresh-metric replay engine (`scripts/step7q_replay_parameter_expansion.py`)
adapted to consume chain-move candidate rows. No new replay engine.

Promotion gate (same shape as Step7Q exit gate):

- `selected_request_count >= 96`
- `represented_case_count >= 8`
- `largest_case_share <= 0.25`
- `forbidden_request_term_count = 0`
- `overlap_after_splice_count < 36`
- `soft_regression_rate < 0.2916666666666667`
- `bbox_regression_rate = 0.0`
- `strict_meaningful_winner_count >= 3`

If the gate passes, Phase4 family ablation reopens. If it fails after k=2 and
k=3, Step7R is closed and the next Oracle diagnosis prompt is requested before
opening Step7S.

## Verification commands

```bash
PYTHONPATH=src .venv/bin/python -m pytest \
  tests/test_step7r_swap_contract.py \
  tests/test_step7r_ripple_contract.py -q

PYTHONPATH=src .venv/bin/python scripts/step7r_generate_swap_candidates.py \
  --output artifacts/research/step7r_swap_source_deck.jsonl

PYTHONPATH=src .venv/bin/python scripts/step7q_replay_parameter_expansion.py \
  --input artifacts/research/step7r_ripple_source_deck.jsonl \
  --objective-aware-slot \
  --summary-out artifacts/research/step7r_phase3_replay_summary.json
```

## Orchestration note

Step7R is being executed under a main-agent / Codex-sub-agent split:

- main agent (Claude in Claude Code): plan, verify, integrate, gate decisions;
- Codex sub-agent (`codex exec`): implements modules, scripts, and tests;
- main agent never relaxes a phase gate to pass a Codex result.

## Phase 1.5 result and pivot to Step7R-C (2026-05-08)

Phase 1.5 swap replay closed the chain-move family early:

| metric | step7r k=2 swap | step7q_f baseline |
|--------|-----------------|-------------------|
| hard_feasible_nonnoop | 96/96 | 96/96 |
| overlap_after_splice | 0 | 0 |
| bbox_regression_rate | 0.0 | 0.6979 |
| soft_regression_rate | 0.677 | 0.7188 |
| hpwl_regression_rate | 0.969 | 0.6979 |
| actual_all_vector_nonregressing | 3 | 27 |
| strict_meaningful_winner | 0 | 0 |

Decision: `phase2_gate_open=false`. Do not extend to k=3. Reason: swap forces
paired displacement — even when block A wants block B's slot, block B almost
never wants block A's slot, so 96.9% of swaps degrade HPWL. A k=3 ripple
inherits the same forced-pairing failure mode for the first two blocks and
only adds one slot-search degree of freedom that Step7Q-F already exploited.

Artifact: `artifacts/research/step7r_swap_replay_summary.json`.

### Step7R-C: HPWL gradient nudge on AVNR rows

Pivot rationale: Step7Q-F has 27 actual-all-vector-nonregressing rows. These
candidates are tied — non-regressing on every vector but not strictly winning
on any. The Step7R-C operator perturbs each AVNR candidate by a small step
along the per-block HPWL pin-centroid gradient, then snaps to the nearest
non-overlapping slot using the existing Step7Q-F objective-aware slot finder.
This sidesteps the chain-move forced-pairing problem entirely.

Phase plan:

- **Step7R-C Phase 0** — gradient operator contract:
  - `src/puzzleplace/alternatives/hpwl_gradient_nudge.py`
  - inputs: a Step7Q-F replay row + the original case;
  - per-block gradient: vector from block center to weighted pin centroid of
    every net the block participates in;
  - step length: a finite ladder `{0.25, 0.5, 1.0}` × min(block_w, block_h);
  - emits at most `len(ladder)` candidate variants per AVNR row;
  - reuses Step7Q-F objective-aware slot finder for legalization;
  - pure function, picklable for multiprocessing.

- **Step7R-C Phase 1** — fresh-metric replay:
  - `src/puzzleplace/ml/step7r_gradient_replay.py` reusing
    `concurrent.futures.ProcessPoolExecutor(max_workers=min(48, n_rows))`;
  - input: the 27 AVNR rows from `artifacts/research/step7q_objective_slot_replay_rows.jsonl`;
  - replay each variant through the Step7Q legality+quality evaluator;
  - decision gate (Step7R close gate):
    - `strict_meaningful_winner_count >= 1` — promote to Step7R Phase 3 final replay;
    - else close Step7R and request Oracle diagnosis.

- **Step7R-C Phase 2** — final replay (only if Phase 1 gate opens):
  - run on all 96 source rows, not just the 27 AVNR survivors;
  - apply the Step7P/Step7Q promotion gate verbatim.

Forbidden in Step7R-C:

- modifying contest_optimizer.py or finalizer paths;
- relaxing the Step7Q-F numbers in the Step7R Phase 0 lock;
- claiming `phase4_gate_open=true` without final-replay evidence.

## Step7R-C result and lane close (2026-05-08)

Step7R-C executed in 4.8 s on 48 workers over 27 AVNR input rows × 3 step
factors = 81 variants:

| metric | step7r_c | step7q_f baseline |
|--------|---------|-------------------|
| variant_count | 81 | - |
| fresh_hard_feasible_nonnoop | 81 | 96 |
| hpwl_strict_improvement | 72 (89%) | - |
| actual_all_vector_nonregressing | 24 | 27 |
| strict_meaningful_winner | 0 | 0 |
| represented_case_count | 2 | 8 |

Decision: `gradient_replay_strict_gate_closed`. The HPWL gradient direction is
correct (89% of variants strictly improve HPWL) but every gradient step either
collapses back to the Step7Q-F slot or trips bbox/soft regression.

### AVNR source-duplication finding

Deduping the 27 Step7Q-F AVNR rows on `(case_id, block_id, target_box)` gives
**2 unique candidates**:

- case 51 / block 14 / target (117.35, 138.0, 21.0, 7.0) — appears 24×;
  official_like_cost_delta = -1.256e-8.
- case 24 / block 32 / target (95.5, 56.0, 10.0, 15.0) — appears 3×;
  official_like_cost_delta = -1.477e-8.

Both unique candidates sit at ~1/8 of `MEANINGFUL_COST_EPS = 1e-7`. They are
real micro-improvements that do not regress any vector but cannot pass the
strict-winner cost cutoff. The Step7Q-F selector + slot finder collapses
distinct source rows to the same geometric outcome — reported AVNR breadth is
inflated.

### Lane close

Step7R is closed. Both chain-move and HPWL gradient operators failed under the
unchanged Step7Q-F gates. Close artifacts:

- `artifacts/research/step7r_close_decision.json`
- `artifacts/research/step7r_close_decision.md`

Next phase: `request_oracle_stagnation_diagnosis_after_step7r` (see
`docs/oracle_step7r_close_prompt.md`).
