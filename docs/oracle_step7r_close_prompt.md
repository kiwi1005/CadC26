# Oracle Stagnation Diagnosis Prompt — Step7R Close (2026-05-08)

You are an Oracle for the FloorSet-Lite / ICCAD 2026 floorplanning research
project (CadC26). Step7R has just closed with 0 strict meaningful winners
across two operator families. We need a **mathematically-grounded** next
direction, not another empirical operator family.

## Required output

1. A formal mathematical model of the post-Step7Q-F state of the 8
   representative cases — what optimization problem are we actually solving,
   and what does the geometry of its feasible / Pareto frontier imply?
2. A derivation (not a guess) of why the current operator search is stuck.
3. A *derived* next-method proposal that follows from the model, with the
   gate it should be evaluated under and the failure modes the model predicts.
4. An explicit list of the assumptions your model makes, and which ones the
   user should verify before implementation.

Use the language of constrained optimization, Pareto theory, KKT / first-order
conditions, multi-objective scalarization, or graph-combinatorial structure as
appropriate. Avoid "try X" recommendations without an explanatory model behind
them.

## Background — only what is needed

- Project: FloorSet-Lite floorplan placement. 8 representative cases. Each case
  is a set of axis-aligned rectangular macros placed inside a frame, with
  constraints (fixed/preplaced blocks, MIB equal-shape groups, boundary
  attachments) and a cost vector
  `(official_like_cost, hpwl, bbox_area, soft_constraint_violation)`.
- Strict meaningful winner =
  `hard_feasible AND official_like_cost_delta < -MEANINGFUL_COST_EPS (1e-7)
   AND hpwl_delta <= EPS AND bbox_area_delta <= EPS AND soft_constraint_delta <= EPS`.
- AVNR (all-vector-nonregressing) = `hard_feasible AND every-vector-delta <= EPS`.
- Phase4 gate requires ≥3 strict winners (current count: 0 over the entire
  Step7P/Q/R sidecar history).

## Evidence summary (short)

Step7P → Step7Q-F (six sub-phases): every phase ended with strict_count = 0.

Step7Q-F (objective-aware single-block slot finder, latest local-search lane):

- 96 source rows, 96 hard_feasible_nonnoop, 0 overlap_after_splice
- soft 0.7188, bbox 0.6979, hpwl 0.6979 regression
- AVNR 27, strict 0, 8 cases represented

Step7R (closed 2026-05-08):

- Phase 1.5 k=2 swap over 96 rows: bbox 0, soft 0.677, hpwl 0.969 regression,
  AVNR 3, strict 0. Conclusion: forced pairing degrades HPWL.
- Step7R-C HPWL gradient nudge over 27 AVNR rows × 3 step factors = 81 variants:
  72 (89%) strictly improve HPWL, AVNR 24, strict 0. Conclusion: gradient
  direction is correct, but slot finder collapses back to the Step7Q-F basin.

## The duplication finding (key fact for the Oracle)

After dedup on `(case_id, block_id, target_box)`, Step7Q-F's 27 AVNR rows
collapse to **2 unique candidates**:

| case | block | official_cost_delta | hpwl_delta | bbox_delta | soft_delta | occurrences |
|-----:|------:|--------------------:|-----------:|-----------:|-----------:|------------:|
|   51 |    14 |          -1.256e-8 |   -3.20e-5 |        0.0 |        0.0 |          24 |
|   24 |    32 |          -1.477e-8 |   -4.95e-4 |        0.0 |        0.0 |           3 |

Both are at ~1/8 of `MEANINGFUL_COST_EPS = 1e-7`. They are hard-feasible
non-regressing improvements that fail the strict cost cutoff by roughly one
order of magnitude.

## Hypotheses we have NOT yet rejected

- H1: The 8 representative cases are at or near a local Pareto-optimal corner
  of the cost vector. Any single-block move tilts at least one component
  upward. Strict winners require multi-block coordinated moves whose
  feasibility region is small under the current constraints.
- H2: `MEANINGFUL_COST_EPS = 1e-7` was set conservatively for numerical noise
  protection on much larger cases (XL). On the representative cases it may be
  larger than the actual achievable per-move cost improvement, so the gate is
  unreachable by construction.
- H3: The Step7Q-F slot finder is a *contraction* operator on the layout
  space — distinct candidate moves converge to the same fixed point. The
  effective search space is therefore much smaller than the candidate count
  suggests.
- H4: The cost function decomposes into terms whose pairwise gradients are
  nearly orthogonal in the current local neighborhood; improving HPWL pushes
  bbox or soft up by an amount that cancels the gain on the scalarized cost.

We need the Oracle to model these formally and tell us which ones are
load-bearing.

## What we want the Oracle to derive

Please produce, in order:

### A. Formal model of the local optimization landscape

Define the variables and constraints for one representative case as a
multi-objective constrained optimization problem
`min f(x) = (f_hpwl, f_bbox, f_soft, f_official) s.t. g_overlap(x) <= 0,
g_mib(x) = 0, g_boundary(x) <= 0, fixed/preplaced equality constraints`, where
`x` is the vector of block centers (and possibly orientations). State
explicitly:

- the dimension of `x` and its constraint manifold;
- whether the feasible set is convex / connected / discretized by overlap;
- whether `f_official` is a known scalarization of the other three or has
  independent terms.

### B. Derive the local Pareto / KKT structure

Given the empirical fact that single-block moves cannot strictly improve all
four objectives simultaneously without slot-finder collapse, derive:

- whether the current Step7Q-F outcome is a (weak) Pareto-stationary point
  under the chosen scalarization (using KKT or vector-valued first-order
  conditions over the inequality constraints);
- what the smallest-norm feasible perturbation looks like that *strictly*
  improves the scalarized cost — is it single-block, k-block coordinated, or
  unbounded under the current constraint surface?
- whether the slot-finder convergence basin is consistent with the gradient
  of `f_official` projected onto the active constraint surface.

### C. Derived method proposal

Output one method that follows from the model. For example: if the analysis
shows the current state is Pareto-stationary on a face of the active overlap
constraints, the derived method might be a constrained second-order step or a
constraint-relaxation cone search, not a heuristic.

For the proposed method, give:

1. its mathematical statement;
2. the smallest sub-problem it should be tested on first (which case, which
   blocks);
3. the expected sign / magnitude of the cost improvement, with derivation;
4. failure modes the model predicts (e.g. "this method cannot escape if
   condition X holds, in which case fall back to Y").

### D. Threshold question

Separately answer: given the derived per-move achievable improvement on the 8
representative cases, is `MEANINGFUL_COST_EPS = 1e-7` numerically
self-consistent, or is it on the wrong side of the natural noise floor for
this scale? If it is wrong, what threshold *follows from the model* (e.g. a
multiple of cost-function gradient norm × machine epsilon × case scale)?

If your answer is "the threshold is right and the cases are genuinely Pareto-
optimal at this resolution," say so and give the implication for the project.

### E. Verifiable assumptions

List each modeling assumption you made and label it:

- `verifiable_immediately` — the user can run a script in the existing repo to
  check (state the script and the artifact path);
- `verifiable_with_small_extension` — a short new sidecar will verify (state
  what to write);
- `unverifiable_locally` — must be accepted as a working assumption.

## Constraints on the Oracle answer

- Do not propose modifications to `contest_optimizer.py`, `external/FloorSet/`,
  or finalizer paths.
- Do not justify lowering `MEANINGFUL_COST_EPS` empirically without the
  derivation in section D.
- Do not propose unbounded-scope GNN/RL training. Step7Q already has the
  GNN/RL gate open under the constrained-operator-learning scope; any learned
  component must be a sidecar policy guided by your derived model.
- Do not propose trying every combination of operator families. The user wants
  one derived next direction, not a menu.
- Do not assert that strict winners are impossible without showing the
  derivation in section B that establishes Pareto stationarity.

## Pointer to the on-disk evidence

If you want to cross-check claims, the following artifacts exist:

- `artifacts/research/step7r_close_decision.json`
- `artifacts/research/step7r_phase0_stagnation_lock.json`
- `artifacts/research/step7r_swap_replay_summary.json`
- `artifacts/research/step7r_c_gradient_replay_summary.json`
- `artifacts/research/step7q_objective_slot_replay_summary.json`
- `artifacts/research/step7q_objective_slot_replay_rows.jsonl` (the 96 rows)

The user will run any verification script you specify in section E.
