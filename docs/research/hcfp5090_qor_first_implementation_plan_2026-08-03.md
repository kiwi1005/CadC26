# HCFP-5090 QoR-first implementation plan

Date: 2026-08-03  
Branch: `feat/hcfp5090-qor-first`  
Base: `origin/main` at `2ddc494`  
Status: approved for staged implementation

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

## Q4: topology-aware component BDP

Direction cost combines movement, estimated HPWL delta, topology confidence,
latch breakage, and boundary disruption. Conflict connected components are
solved independently with a bounded beam:

1. fix high-confidence directions;
2. branch only on uncertain pairs;
3. reject H/V cycle creation;
4. project fixed-direction linear constraints;
5. rebuild all conflicts after movement;
6. reset low-confidence directions when a component signature repeats;
7. interleave small HPWL/bbox superiorization steps.

Full differentiable QP/linear-constraint layers are deferred. The existing
projector is retained until the objective-aware beam proves a QoR gain.

### Q4 acceptance

- Projection hard-feasibility success does not fall.
- Repair displacement falls another 20%.
- HPWL and bbox show no systematic regression.
- Learned p95 is at most 1.20 times analytic p95.

## Q5: post-repair DAgger and listwise ranker

Replay v2 stores raw, mid-flow, pre-BDP, post-BDP, and post-repair states plus
topology/contact/boundary/MIB decisions, exact feasibility, cap attribution,
repair displacement, source, case/checkpoint hashes, and population seed.

First replay target: 5,000 training-source records.

```text
40% post-repair hard negatives
25% near-cap candidates
20% difficult 106--120-block cases
15% diverse successful positives
```

Ranker v2 predicts hard feasibility, post-repair `J`, each soft contribution,
repair displacement, and runtime. A listwise term orders candidates inside each
case; feasibility and regression targets remain explicit auxiliary losses.

Promotion targets:

- exact top-1 at least 12/16;
- top-4 oracle recall at least 15/16;
- false promotion zero;
- weighted rank regret at least 50% below the current ranker;
- full validation Pareto regressions zero.

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
| Q3 | dynamic messages and live force gates | Q1/Q2 | opt-in | oracle gain |
| Q4 | component beam, reset, superiorization | Q1/Q2 | opt-in | displacement/runtime |
| Q5 | replay v2 and listwise ranker | Q0--Q4 | training/selection | rank gates |
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

