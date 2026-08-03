# HCFP-5090 QoR-first implementation plan

Date: 2026-08-03  
Branch: `feat/hcfp5090-qor-first`  
Base: `origin/main` at `2ddc494`  
Status: Q0--Q2 verified; Q3 HOLD; Q4 exact-safe/runtime-blocked; Q5 checkpoint PASS/system HOLD

## Execution checkpoint

- Q0 exact cap attribution is committed and does not change runtime selection.
- Q1 structure now drives opt-in, cycle-free candidate geometry and passes the
  disjoint large-case internal held-out gate.
- The Q1 gate uses 16 samples from 16 FloorSet-Lite source files, spans
  107--120 blocks, reconstructs and excludes the exact 1,000 samples consumed
  from the configured 2,048-sample source stream, and records `+24.3934`
  weighted post-BDP topology-oracle gain with 512/512 hard-feasible
  topology-stage records. The training report is bound to checkpoint state hash
  `48ba552c518f...` and any provenance mismatch fails closed.
- Q2 contact, boundary, and MIB construction now controls runtime geometry.
  The disjoint official-raw audit records 16/16 constraint-oracle wins over
  topology, `+0.5753` score-weighted post-BDP `J` gain, and 10/16 cap crosses.
- Q2 is not default-promoted: group connectivity is 77.3% rather than 95% and
  the one area-compatible MIB candidate is legalizable but not oracle quality.
- Q4 now retains uncommitted component geometry as a separate exact-guarded
  alternative. A clean large16 benchmark raises exact constraint coverage to
  364/512 versus the legacy control's 338/512 and improves weighted oracle `J`
  to `2.276194`.
- The proposal materially improves held-out case 14 from `J=3.052426` and 45
  soft violations to `J=2.415978` and 27 soft violations without weakening the
  exact verifier or Pareto guard.
- Schema-v7 timing separates solver work from offline evaluator work. The clean
  `846147d` benchmark preserves every placement hash while vectorized group
  collision checks reduce solver p50/p95 from 22.913/27.623 seconds to
  10.070/14.387 seconds. Analytic p95 is still only 0.0441 seconds, so Q4 fails
  runtime promotion by a wide margin. Q3 may proceed only as a default-off lane.
- Q3 geometry-aware collective dynamics is implemented, checkpointed, and
  audited on the same disjoint large16 split. Its two-step rollout destroys the
  exact-feasible structure of every transformed topology/constraint row; only
  the retained initial copy preserves the oracle. Q3 therefore remains
  default-off and supplies paired hard negatives to Q5 rather than runtime QoR.
- Q5 replay schema v3 now binds initial/post-relax geometry, post-BDP geometry,
  exact-repair outcomes, candidate features, row identity, source lineage, and
  checkpoint/config hashes. Repair-aware feature v4, fail-closed checkpoint
  upgrades, listwise training, exact stage gates, output-neutral runtime shadow,
  and selected-versus-oracle visualization are implemented.
- The 256-list initial-stage pilot passes all checkpoint gates on seeds 5104 and
  5105: 12/16 exact top-1, 15/16 top-4 oracle recall, and zero false promotion.
  Seed 5102 records 11/16, 14/16, and one false promotion. All three post-relax
  gates fail. Q5 therefore remains shadow-only; 5,000-record replay, broad
  three-seed validation, and A100 profiling remain pending.

## Decision

The next phase improves candidate quality before revisiting activation or model
scale. The learned system currently predicts structure, outline, force gates,
and residual geometry, but only residual geometry materially affects runtime
candidates. The required causal chain is:

```text
structure prediction
  -> hard cycle-free topology decode
  -> actual candidate geometry
  -> constraint construction
  -> objective-preserving projection
  -> post-repair ranking
  -> Pareto-safe selection
```

Sequence-Pair is a cycle-free topology prior. It does not replace collective
dynamics, bounded disjunctive projection (BDP), exact verification, or final
selection.

## Observed architecture gaps

1. `precedence_logits`, `outline`, and `force_gates` are trained but do not
   control active runtime geometry. `_learned_population()` consumes center,
   aspect, and flow residuals only.
2. The scene encoder has membership-presence bits but no group/MIB identity or
   same-group/same-MIB pair information.
3. The flow head is a per-node residual MLP. It does not observe current pair
   gaps, overlap depth, contact state, boundary slack, or conflict components.
4. Learned candidates remain bounded perturbations of the safe shelf and lack
   topology diversity.
5. Active typed forces have no precedence/SP-anchor channel. Grouping uses
   centroid attraction rather than explicit connected contacts.
6. BDP beam variants choose the same rank of direction independently for all
   conflict pairs. They are not component-level combinations and do not fully
   rebuild conflicts after movement.
7. Unique-class precedence labels discard diagonal pairs with multiple valid
   relations instead of preserving their set-valued supervision.

These are treated as architecture defects, not reasons to enlarge the model or
run more random samples.

## Runtime invariants

- Exact official raw-coordinate hard verification remains authoritative.
- Fixed and preplaced target geometry is copied exactly.
- Safe analytic incumbent and Pareto guard remain available at all times.
- Official validation data is never used for training or replay fitting.
- Geometry predicates and force accumulation use FP32 or higher precision.
- No new dependency is introduced without a measured need.
- Learned features are promoted only after held-out QoR, feasibility, and
  runtime gates pass.

## Q0: exact cap attribution

For a hard-feasible candidate, define

```text
quality_factor = 1 + 0.5 * (max(0, hpwl_gap) + max(0, area_gap))
runtime_term   = max(0.7, max(0.01, runtime_factor) ** 0.3)
uncapped_cost  = quality_factor * exp(2 * violations_relative) * runtime_term
J              = log(uncapped_cost)
cap_margin     = log(10) - J
```

Soft attribution is exact:

```text
boundary_contribution = 2 * boundary_violations / max_possible_violations
grouping_contribution = 2 * grouping_violations / max_possible_violations
mib_contribution      = 2 * mib_violations / max_possible_violations
```

The minimum number of soft violations that could cross the cap while holding
quality and runtime fixed is:

```text
required_soft_fixes_to_uncap =
    ceil(max(0, J - log(10)) * max_possible_violations / 2)
```

Required telemetry:

- uncapped and log-uncapped cost;
- cap margin;
- quality factor, HPWL gap, area gap, and runtime term;
- boundary/grouping/MIB contributions;
- required soft fixes and required quality improvement;
- raw, projected, and post-repair cap margin;
- repair displacement and candidate source;
- blocker class: hard, soft, quality, mixed, or projection dominated.

Q0 must not change runtime selection. Its report is an analysis input for all
later promotion decisions.

### Q0 acceptance

- Components reconstruct `J` within floating-point tolerance.
- Cost matches the pinned official v10 formula before the official cap.
- Reports validate finite inputs and reject inconsistent soft counts.
- One deterministic command classifies all available validation cases.

## Q1: cycle-free structured topology seeds

### Set-valued relation labels

Each ordered pair receives a mask over `{LEFT, RIGHT, ABOVE, BELOW}`. Diagonal
pairs may permit more than one relation. Training minimizes partial-label NLL:

```text
-log(sum(probability[r] for r in allowed_relations[i,j]))
```

and an antisymmetry term:

```text
p(i,j,LEFT)  ~= p(j,i,RIGHT)
p(i,j,ABOVE) ~= p(j,i,BELOW)
```

### Dual permutations

The structure model emits two soft permutations. Sinkhorn normalization is used
during training; runtime converts each matrix to one deterministic total order.
The two orders define every horizontal/vertical relation and therefore produce
acyclic H/V constraint graphs by construction.

The first contest-safe implementation uses only PyTorch. A dependency-backed
Hungarian solver is deferred until an oracle gap demonstrates that deterministic
hardening is the limiting error.

### Seed construction

```text
4 topology hypotheses x 2 deterministic perturbations x 2 aspect hypotheses
  -> up to 16 structured seeds
```

Each seed is packed by DAG longest paths. Preplaced anchors are immutable. A
predicted order that conflicts with anchors is repaired only by reordering
low-confidence movable blocks; otherwise that seed is rejected.

### Q1 acceptance

- H/V cycle count is exactly zero for every hard-decoded seed.
- Seed ID and input hash reproduce byte-identical relations and geometry.
- Preplaced anchors remain exact or the seed is rejected.
- Provenance links permutation logits, hard orders, H/V edges, and final boxes.
- Internal held-out topology oracle@16 beats the shelf oracle, including the
  score-weighted 106--120-block subset.

## Q2: constraint construction

### Group contacts

Gold rectangles produce side-specific contact edges when edges coincide within
tolerance and the orthogonal projections overlap. For each cluster, a weighted
maximum spanning tree selects the minimum contact set needed for connectivity.

Runtime predicts edge probability, side, orthogonal overlap, and latch
confidence, then performs an exact deterministic tree decode inside each
cluster.

### Hysteretic latching

A contact becomes active only after satisfying the on-threshold for multiple
steps and is released only at a wider off-threshold or when exact conflict
attribution marks it infeasible. Latched components move as semi-rigid
supernodes while retaining bounded internal adjustment.

### Boundary slots

Four virtual boundary nodes represent left, right, top, and bottom. Blocks on
the same side receive a deterministic 1-D order so boundary contact does not
create side-local overlap. Corner blocks attach to two virtual nodes.

### MIB construction

Compatible MIB members share a group shape variable. Fixed/preplaced members
anchor that variable. When area tolerances cannot support a common shape, hard
area remains authoritative and the structurally unavoidable soft violation is
reported rather than hidden.

### Q2 acceptance

- Compatible MIB groups have zero MIB violation.
- Group connected ratio is at least 95% on internal held-out data.
- Boundary violation falls materially on eligible cases.
- Post-BDP repair displacement falls by at least 20%.

### Q2 measured result

The implementation and candidate-density gate pass, but default promotion is
held until Q4 closes the remaining projection gaps:

| Gate | Result | Decision |
|---|---:|---|
| constraint oracle versus topology | 16/16 wins; weighted `J` gain `+0.5753` | pass |
| hard-feasible constraint candidates | 305/512 raw; 330/512 post-BDP | pass |
| boundary violations | 492 -> 393; boundary-frame representatives 78/78 feasible with zero boundary violations | pass |
| edge-normalized group connectivity | 0.7% -> 77.3% | hold; target is 95% |
| compatible MIB construction | zero MIB violation in the sole compatible case, but candidate overlaps | semantic pass; promotion hold |
| post-BDP displacement | weighted mean `1.2216` raw-coordinate units versus `0` for topology | hold |
| final runtime safety | 16/16 hard feasible; 16/16 better than analytic oracle | pass |

The authoritative result record is
[`hcfp5090_q2_constraint_results_2026-08-03.md`](hcfp5090_q2_constraint_results_2026-08-03.md).

## Q3: geometry-aware collective dynamics

Each rollout step rebuilds pair features from current geometry:

```text
net weight, dx, dy, four edge gaps, overlap depth,
same group, same MIB, topology relation, latch state
```

Dynamic messages are aggregated per block and combined with RMS-normalized
analytic typed-force channels. Existing learned force gates become active in
this combination. Axis semantics remain explicit; required invariances are
block permutation, translation, and augmentation consistency rather than full
rotation equivariance.

Initial budget:

```text
static hidden: 192
message hidden: 128
message-passing layers: 3
rollout steps: 4--8
population: 16 structured + 16 perturbed
```

This stage proceeds only after structured topology raises the positive-candidate
density. Model size and rollout depth are not first-line remedies.

### Q3 measured result

The implementation satisfies the wiring and training requirements but fails the
promotion gate. On the clean large16 audit, the merged population retains one
initial and one collective-transformed copy of each learned row. Exact-feasible
topology rows fall from 512 in the no-collective control to 256, and exact
constraint rows fall from 364 to 182: the initial copies survive and all
transformed copies fail. The selected weighted `J` remains protected only by
the unchanged alternatives. Measured p95 is 11.253 seconds versus 0.0456
seconds for the same-run analytic comparator, and repeated CUDA replay changes
the post-relax geometry hashes. The controller remains a training-only source
of post-repair hard negatives. Full evidence is in
[`hcfp5090_q3_collective_results_2026-08-03.md`](hcfp5090_q3_collective_results_2026-08-03.md).

## Q4: topology-aware component BDP

Direction cost combines movement, topology confidence, active-contact latch
cost, and boundary disruption. Complete projected branches are then compared
with construction and HPWL/bbox Pareto tiers. Conflict connected components
are solved independently with a bounded beam:

1. fix high-confidence directions;
2. branch only on uncertain pairs;
3. reject H/V cycle creation;
4. project fixed-direction linear constraints;
5. rebuild all conflicts after movement;
6. reset low-confidence directions when a component signature repeats;
7. select objective-preserving projected geometry with HPWL/bbox Pareto tiers.

Full differentiable QP/linear-constraint layers are deferred. The existing
projector is retained until the objective-aware beam proves a QoR gain.

### Q4 acceptance

- Projection hard-feasibility success does not fall.
- Repair displacement falls another 20%.
- HPWL and bbox show no systematic regression.
- Learned p95 is at most 1.20 times analytic p95.

Q4 currently passes exact safety and the clean feasibility/QoR benchmark, but
not runtime promotion. Retaining the changed, uncommitted component proposal
behind raw constraint replay, exact verification, and Pareto dominance raises
constraint coverage from the component primary's 322/512 and legacy control's
338/512 to 364/512. Weighted constraint-oracle `J` improves to `2.276194`.

The component lane therefore remains opt-in. Schema-v7 clean paired timing
shows the next blocker directly: learned p95 is 14.387 seconds versus analytic
p95 0.0441 seconds, a 326.1x ratio. Already-feasible component rows are skipped,
and the former 3.48-million scalar rectangle checks are vectorized. The next
implementation must keep Q3 default-off, preserve exact hashes, and use Q5
ranking to reduce expensive tail work before promotion. Explicit
pre-projection HPWL perturbation remains deferred.

## Q5: post-repair DAgger and listwise ranker

Replay schema v3 stores paired `initial` and `post_relax` candidate lists,
post-BDP geometry, and post-repair outcomes plus topology/constraint lineage,
exact feasibility, cap attribution, repair displacement, source, case,
checkpoint/config hashes, and population seed. Geometry-derived features are
recomputed and checked when writing and reading; mismatched lineage fails
closed. Mid-flow snapshots and decision-level teacher actions remain deferred
until the paired-state pilot demonstrates useful signal.

First replay target: 5,000 training-source records.

```text
40% post-repair hard negatives
25% near-cap candidates
20% difficult 106--120-block cases
15% diverse successful positives
```

The first ranker increment preserves the existing scalar-cost output for
checkpoint compatibility but replaces capped-score regression with post-repair
list order. Its 26-D feature v4 is derived only from raw/post-BDP geometry,
source kind, stage, and pre-tail boundary/group/MIB proxies; post-repair targets
cannot leak into the input. ListMLE is the primary loss, feasibility ordering
is an auxiliary term, and standardized uncapped `J` is a small pointwise
regularizer. Training-only z-score statistics are stored in the checkpoint.
Legacy schema-v2 replay remains readable and uses the old pointwise objective.
Multi-task output heads for hard feasibility, soft contributions, repair
displacement, and runtime remain deferred.

The evaluator reports initial and post-relax stages separately, rejects sample
overlap and checkpoint-lineage mismatch, uses stable row IDs only for prediction
tie-breaking, and measures top-1, top-4 recall, false promotion, weighted rank
regret, and nonnegative score regret. The runtime shadow now reproduces the
exact merged learned initial slice used by replay and records top-4 provenance,
but deliberately cannot change selection or the exact source.

Promotion targets:

- exact top-1 at least 12/16;
- top-4 oracle recall at least 15/16;
- false promotion zero;
- weighted rank regret at least 50% below the current ranker;
- full validation Pareto regressions zero.

### Q5 measured checkpoint

The initial-stage pilot trains on 256 sample-ID-disjoint lists and evaluates on
a separate 16-sample replay. Seeds 5104 and 5105 each achieve 12/16 top-1,
15/16 top-4, zero false promotion, and weighted rank regret 0.4375. Seed 5102
misses at 11/16, 14/16, and one false promotion. Post-relax results are 6/10,
3/6, and 6/9 for top-1/top-4 across the three seeds, so that stage remains
ineligible.

The checkpoint gate is therefore a 2/3-seed pass for the initial list only.
System promotion remains on hold until a larger replay and validation-like
split prove the signal, active-selection experiments preserve the Pareto
invariant, and A100 runtime gates pass. The authoritative record is
[`hcfp5090_q5_ranker_results_2026-08-04.md`](hcfp5090_q5_ranker_results_2026-08-04.md).

## Activation policy

Activation remains shadow-only. It is reconsidered only after structured
candidates increase the learned-improvement positive density. The future risk
target is false skip probability conditioned on a real learned improvement,
not ordinary binary accuracy.

## Work packages and dependency order

| ID | Deliverable | Depends on | Runtime effect | Stop gate |
|---|---|---|---|---|
| Q0 | exact cap attribution and report | current scorer parity | none | exact parity |
| Q1a | set-valued labels and losses | Q0 telemetry schema | training only | label tests |
| Q1b | dual permutation and hard SP decode | Q1a | opt-in | zero cycles |
| Q1c | preplaced adaptation and longest-path seeds | Q1b | opt-in | oracle@16 gain |
| Q2a | contact labels and MST decoder | Q1c | opt-in | connected ratio |
| Q2b | latch/boundary/MIB construction | Q2a | opt-in | soft QoR gain |
| Q3 | dynamic messages and live force gates | promoted Q4 gate | deferred | oracle gain |
| Q4 | component beam, reset, superiorization | Q1/Q2 | opt-in checkpoint | displacement/runtime |
| Q5 | replay v3 and listwise ranker | Q0--Q4 | training/shadow selection | rank gates |
| Q6 | A100 profile and submission freeze | all promoted gates | packaging | final smoke |

Every opt-in stage retains the analytic incumbent and exact Pareto guard. If a
stage misses its stop gate, diagnose that stage rather than stacking the next
one on top.

## Schedule

| Date | Milestone |
|---|---|
| Aug 3--4 | Q0 attribution and 100-case dashboard |
| Aug 5--8 | Q1 structured topology and runtime provenance |
| Aug 9--11 | Q2 contact, boundary, and MIB construction |
| Aug 12--14 | Q4 component-aware BDP; Q3 only after topology gain |
| Aug 15--17 | Q5 5k replay and ranker v2 |
| Aug 18--19 | three-seed validation and A100 cold profile |
| Aug 20 | freeze algorithms; packaging fixes only |
| Aug 21 | submission blockers only |

## Explicit non-goals

- no larger activation classifier;
- no flow-step or random-candidate increase as a substitute for topology;
- no hidden-dimension expansion without a measured bottleneck;
- no capped-cost ranker target;
- no centroid-only grouping promotion;
- no independent pair relation decode without global cycle control;
- no weakened exact verifier or removed Pareto guard;
- no official-validation replay training;
- no contest-critical differentiable QP before the simpler projector is shown
  to be the limiting factor.

## Evidence and references

The implementation is grounded in the user-supplied primary references for
Sequence-Pair, obstacle-aware Sequence-Pair adaptation, Gumbel-Sinkhorn,
FloatForm, graph simulators, EGNN, constraint-based simulators, OptNet,
Per-RMAP, DAgger, ListMLE, selective prediction, diffusion placement, and the
official ICCAD contest schedule. Source URLs are preserved below for audit:

1. https://cir.nii.ac.jp/crid/1363670318478789760
2. https://doi.org/10.1145/267665.267675
3. https://arxiv.org/abs/1802.08665
4. https://www.nature.com/articles/s41467-026-74527-6
5. https://proceedings.mlr.press/v119/sanchez-gonzalez20a.html
6. https://proceedings.mlr.press/v139/satorras21a
7. https://proceedings.mlr.press/v162/rubanova22a
8. https://proceedings.mlr.press/v70/amos17a.html
9. https://pubmed.ncbi.nlm.nih.gov/39735901/
10. https://proceedings.mlr.press/v15/ross11a.html
11. https://doi.org/10.1145/1390156.1390306
12. https://proceedings.mlr.press/v130/gangrade21a.html
13. https://proceedings.neurips.cc/paper_files/paper/2024/hash/fe224a60b878e79d5b3d79d7f113f76b-Abstract-Conference.html
14. https://proceedings.mlr.press/v267/lee25y.html
15. https://www.iccad-contest.org/index.html
