# HCFP-5090 Q2 constraint-construction results

Date: 2026-08-03
Branch: `feat/hcfp5090-qor-first`
Base: `origin/main` at `2ddc494`
Status: **PARTIAL PASS — QoR/candidate-density gate passed; default promotion held for Q4**

## Decision

Q2 proves the central architecture hypothesis: learned structure improves QoR
once it directly constructs boundary placement and group contacts. The
constraint oracle beats the topology oracle on all 16 disjoint large held-out
cases under the pinned official evaluator.

Q2 is not enabled as an unconditional default. It does not yet reach the 95%
group-connectivity gate, its compatible MIB candidate still requires a better
overlap projector, and it has not demonstrated the required displacement
reduction. Q4 component-aware BDP is the next stage; the exact Pareto guard and
analytic incumbent remain mandatory.

## Implemented causal path

```text
same-group / same-MIB scene identity
  -> learned contact side, boundary order, and MIB group shape
  -> deterministic contact / boundary / MIB variants
  -> raw FP64 contact and boundary replay
  -> existing exact-safe tail
  -> raw Pareto guard
```

Implemented modules include:

- `src/hcfp/constraints/contact_tree.py`: side-specific gold contacts and a
  deterministic weighted maximum spanning tree;
- `src/hcfp/constraints/latching.py`: pure hysteretic latch state and exact
  side/orthogonal-overlap predicates;
- `src/hcfp/constraints/boundary_slots.py`: virtual boundary equalities and
  deterministic per-side ordering;
- `src/hcfp/constraints/mib_shapes.py`: hard-by-construction shared shape only
  when the official 1% area gate permits it;
- `src/hcfp/constraints/construction.py`: bounded group, boundary, MIB, and
  combined candidate variants;
- `src/hcfp/constraints/raw_repair.py`: exact raw-coordinate replay that rejects
  every hard-feasibility regression;
- optional Q2 model heads, training losses, runtime provenance, and a
  fail-closed official-raw audit.

## Training evidence

Checkpoint:

```text
artifacts/checkpoints/hcfp5090-q2-constraints-s1000-seed5090.pt
file SHA-256: 5c013e14b7b172f40a8be2434a0f185645bdd3d2a7d26125504b72da6c29a4be
state hash: 5f6f23da42410e92d48b2b665c3bd0ec75577a334f26e0a6e28e39369c9da477
```

Training sidecar:

```text
artifacts/checkpoints/hcfp5090-q2-constraints-s1000-seed5090.pt.training.json
SHA-256: e61c4f93197ba6f813939e44f095fe836ca87aeb90363a963d2a4d46353a4187
```

The Q2 checkpoint warm-started the verified Q1 checkpoint and consumed 1,000
unique score-aware FloorSet-Lite training samples from the configured 2,048
sample source stream. Training used BF16 on CUDA, 1,000 structure steps, EMA
`0.999`, seed `5090`, and no official validation data. Constraint loss fell
from `1.89919` to `0.91283`; total structure loss fell from `2.39654` to
`1.17590`.

## Authoritative official-raw audit

Artifact:

```text
artifacts/benchmarks/hcfp5090-q2-s1000-seed5090-topology16-constraint16-official-raw-heldout-large16.json
SHA-256: e30a836806590b73b84d1bb52f5f339027b1dd2a7e81be330cf93f495094f813
```

Contract:

- 16 FloorSet-Lite training-source held-out samples from 16 source files;
- 107--120 blocks, including the high-weight 116--120 bucket;
- exact exclusion of the 1,000 samples consumed by Q2 training;
- held-out seed `5091`, training seed `5090`;
- 16 topology and 16 constraint seeds per case, represented at initial and
  post-relax stages;
- exact raw coordinates evaluated by official-v10 evaluator commit
  `aadddcc2238695eb21e6542b8a6cd9e9fe6b80fa`;
- evaluator SHA-256
  `64db37865b42baf11add62bdbf035690dca086cd4be7b5b4e58db756f20d8498`.

The held-out sample-list SHA-256 is
`e79491726efed65ebb6c55f7e63564e4896cc7d345bca18da725660de7d8fab9`.

## QoR result

| Metric | Analytic | Topology | Constraint |
|---|---:|---:|---:|
| raw oracle available cases | 16 | 16 | 16 |
| raw weighted mean `J` | 4.0968 | 2.8574 | **2.2800** |
| post-BDP oracle available cases | 16 | 16 | 16 |
| post-BDP weighted mean `J` | 3.4647 | 2.8574 | **2.2821** |
| raw hard-feasible candidates | 46/256 | 512/512 | 305/512 |
| post-BDP hard-feasible candidates | 112/256 | 512/512 | 330/512 |

Constraint versus topology:

- raw: 16 wins, 0 ties, 0 losses; weighted `J` gain `+0.57735`;
- post-BDP: 16 wins, 0 ties, 0 losses; weighted `J` gain `+0.57527`;
- 10/16 constraint oracles have positive cap margin, versus 0/16 topology and
  0/16 analytic oracles;
- final runtime output is hard feasible on 16/16 and beats the analytic
  post-BDP oracle on 16/16, with weighted `J` gain `+0.84844`;
- current runtime selection crosses the cap on 6/16, leaving an observable
  oracle-to-selection gap for Q5 ranker work.

Final raw geometry hashes resolve every runtime output to exactly one post-BDP
candidate: 9 topology candidates and 7 constraint candidates (4 group-contact
and 3 boundary-frame). All 6 cap crossings come from the constraint lane. This
closes the earlier final-candidate provenance gap without changing selection.

## Constraint attribution

Post-BDP oracle totals:

| Soft term | Topology | Constraint | Change |
|---|---:|---:|---:|
| boundary violations | 492 | 393 | -99 (-20.1%) |
| grouping violations | 407 | 93 | -314 (-77.1%) |
| MIB violations | 66 | 66 | 0 |
| total soft violations | 965 | 552 | -413 (-42.8%) |

The Q2 oracle trades some HPWL/bbox quality for a much larger soft-constraint
reduction; the net result remains positive in official log-uncapped objective.

Boundary construction is already exact-safe: all 78 raw boundary-frame
representatives are hard feasible and have zero boundary violations. Group
construction is useful but incomplete: edge-normalized connectivity rises from
`1 - 407/410 = 0.7%` to `1 - 93/410 = 77.3%`, below the 95% gate.

Fifteen of the 16 held-out MIB groups have no common shape compatible with every
member's 1% hard-area interval, so their MIB penalty is structurally
unavoidable without violating a hard gate. The sole compatible group is
constructed with zero MIB violation, but its candidate has three raw overlaps
and six after the current BDP. This is projection evidence, not a reason to
weaken hard area semantics.

## Projection and runtime safety

Constraint candidates have weighted mean raw-to-post-BDP movement `1.2216` in
official coordinate units; topology candidates need zero movement because they
are already hard feasible. Therefore the Q2 displacement-reduction gate is not
proven. The current BDP sometimes damages a useful constructed candidate, which
directly motivates dynamic conflict rebuilding and component-aware direction
selection in Q4.

The runtime path remains fail-closed:

- constructed variants are additive and opt-in;
- every raw repair is accepted only after exact hard verification;
- the analytic incumbent remains present;
- final constraint promotion is Pareto guarded;
- final official output is hard feasible on all 16 audit cases.

## Problems found and resolved

1. CUDA training failed because same-group adjacency used integer matrix
   multiplication. Membership multiplication now uses floating tensors and is
   thresholded, with a CUDA regression test.
2. Zero-gap group contacts were perturbed by center/aspect reconstruction before
   the tail. A fully supplied population is now preserved as the initial
   candidate set; post-relax geometry remains a separate stage.
3. FP32-to-official conversion could turn intended contact into a tiny overlap.
   Raw repair now uses FP64 and one-axis `nextafter` snaps, never orthogonal
   recentering.
4. Repairing contact edges greedily could reject a tree whose intermediate
   states were infeasible even though the completed tree was feasible. The
   whole contact tree is replayed atomically first, followed by a safe per-edge
   salvage fallback.
5. Constraint candidates could disappear behind one aggregate source label.
   Candidate index, kind, stage, exact geometry hash, and raw repair counters
   now survive through the audit and final-selection provenance.

## Promotion decision and next tasks

Q2 remains opt-in. Q4 starts with these bounded tasks:

1. rebuild overlap conflict components after every projection sweep;
2. score separation directions with movement, HPWL delta, learned topology,
   latch breakage, and boundary disruption;
3. branch only low-margin relations and reject H/V cycles;
4. reset repeated component signatures;
5. insert bounded HPWL/bbox superiorization between feasibility sweeps;
6. re-run this exact split and require group connectivity at least 95%, no loss
   of hard-feasible coverage, lower movement, and no weighted `J` regression.

Q3 geometry-aware collective dynamics remains deferred until this projection
path proves it can preserve Q2's candidate advantage.
