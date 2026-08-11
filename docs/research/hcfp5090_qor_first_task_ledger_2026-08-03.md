# HCFP-5090 QoR-first task ledger

Canonical design:
[`hcfp5090_qor_first_implementation_plan_2026-08-03.md`](hcfp5090_qor_first_implementation_plan_2026-08-03.md)

Status values: `[ ]` pending, `[~]` active, `[x]` verified, `[blocked]` gate failed.
A later stage does not start when its required promotion gate is blocked.

## Q0 — exact cap attribution

- [x] Implement exact log-domain uncapped cost and cap margin.
- [x] Attribute boundary, grouping, MIB, quality, and runtime terms.
- [x] Calculate counterfactual soft/quality fixes and blocker class.
- [x] Add deterministic JSON/Markdown reporting CLI.
- [x] Match pinned official evaluator on unit fixtures.
- [x] Run 100-case attribution and preserve its report artifact.
- [x] Commit Q0 with no runtime-selection diff (`54c763d`).

## Q1 — cycle-free structured topology

- [x] Extract set-valued relation masks from gold rectangles.
- [x] Add partial-label NLL and relation antisymmetry loss.
- [x] Implement dependency-free Sinkhorn and deterministic hard permutation.
- [x] Decode dual permutations into cycle-free H/V constraints.
- [x] Add preplaced compatibility/adaptation and exact anchored longest-path packing.
- [x] Add group/MIB identity through separate same-membership message channels.
- [x] Connect topology output to learned runtime candidates behind an opt-in flag.
- [x] Record topology provenance through raw and post-BDP candidates.
- [x] Add deterministic anchor-safe order variants for movable-mediated anchor paths.
- [x] Rechoose low-confidence ambiguous anchor relations with bounded cycle-free repair.
- [x] Compact topology provenance into one reconstructable order catalog.
- [x] Add a fail-closed training-only held-out audit with exact sample exclusion.
- [x] Record and hash the actual consumed training stream in a schema-v2 sidecar.
- [x] Reconstruct the consumed stream and reject checkpoint/config/split mismatch.
- [x] Run shelf-vs-topology oracle@16 on internal held-out cases.
- [x] Stop and diagnose if the 106--120 weighted oracle does not improve.

Current gate: **PASS**. The promotion run reconstructs and excludes the exact
1,000 samples consumed from a source stream configured with limit 2,048,
evaluates 16 different FloorSet-Lite source files spanning 107--120 blocks,
and produces all 16 requested topology seeds for every case.
All 512 raw/post-BDP topology-stage records are hard feasible. The post-BDP
topology oracle beats analytic on 16/16 cases with `+24.3934` score-weighted
uncapped-objective gain. Q2 is unblocked after the Q1 integration commit.

## Q2 — constraint construction

- [x] Extract side-specific cluster contact labels.
- [x] Decode a deterministic maximum contact spanning tree per group.
- [x] Implement hysteretic contact latch state.
- [x] Add boundary virtual nodes and per-side slot order.
- [x] Construct compatible MIB shapes from one group variable.
- [x] Connect learned contact, boundary-order, and MIB heads to runtime variants.
- [x] Replay zero-gap contacts in raw FP64 coordinates without hard regressions.
- [x] Preserve constructed populations through the dynamics/tail boundary.
- [x] Run a disjoint 16-case official-raw audit on 106--120-block cases.
- [~] Raise edge-normalized group connectivity from 77.3% to the 95% gate.
- [~] Carry compatible zero-MIB construction through overlap legalization.
- [~] Prove at least 20% lower post-BDP repair displacement.

Current gate: **PARTIAL PASS — useful candidate density and QoR pass; default
promotion remains on hold**. On the pinned official evaluator, the constraint
oracle beats topology on 16/16 cases. Its score-weighted post-BDP `J` is
`2.2821` versus `2.8574` for topology, a `+0.5753` gain, and 10/16 constraint
oracles cross below the official cap. Grouping violations fall from 407 to 93
and boundary violations from 492 to 393. Final Pareto-safe runtime output is
hard feasible on 16/16 and beats the analytic post-BDP oracle on 16/16.

The full Q2 gate is not yet met: edge-normalized group connectivity is 77.3%,
the single area-compatible MIB case constructs zero MIB violations but still
needs overlap legalization, and constraint candidates require nonzero movement
where the already-feasible topology seeds require none. These are the direct
inputs to Q4 component-aware projection. Q4 has now preserved and extended the
Q2 QoR gain under exact replay, so Q3 is unblocked as a default-off lane.

## Q3 — collective dynamics

- [x] Build current-geometry pair features each rollout step.
- [x] Add three message-passing updates with explicit x/y semantics.
- [x] Feed RMS-normalized typed forces into coordinate updates.
- [x] Connect learned force gates to active typed-force channels.
- [x] Train the collective-only checkpoint and preserve its parent lineage.
- [x] Run a clean disjoint large16 audit against the Q2/Q4 control.
- [blocked] Preserve initial feasibility through two collective rollout steps.
- [blocked] Reproduce same-seed CUDA post-relax geometry bitwise.
- [blocked] Meet the learned/analytic p95 runtime gate.

Current gate: **HOLD — implementation/training complete, runtime promotion
blocked**. The clean large16 population contains initial and post-relax copies.
The control has 512 exact-feasible topology and 364 constraint rows; collective
two-step has exactly 256 and 182. The initial half survives and every transformed
copy fails. Same-run p95 remains about 247x analytic, and repeated CUDA replay
changes post-relax geometry hashes. Q3 stays default-off and its failed
post-relax rows become Q5 hard negatives. Full evidence is in
[`hcfp5090_q3_collective_results_2026-08-03.md`](hcfp5090_q3_collective_results_2026-08-03.md).

## Q4 — component-aware BDP

- [x] Build and refresh conflict connected components every outer sweep.
- [x] Score directions with movement, topology, latch, and boundary terms.
- [x] Rank complete projected branches with HPWL/bbox and construction Pareto tiers.
- [x] Branch independently on uncertain component directions and reject cycles.
- [x] Add repeated-signature reset within the same construction-safe stratum.
- [x] Add exact FP64 commit, active-contact latching, and preserve-original routing.
- [x] Retain changed, uncommitted component proposals behind raw replay, exact
  verification, and the existing Pareto guard.
- [x] Add exact raw/projected/proposal portfolio telemetry and schema-v7 runtime
  separation.
- [x] Compare feasibility, displacement, HPWL, bbox, and p95 runtime from one clean commit.
- [x] Embed solver commit and clean-tree proof in component/analytic artifacts.
- [x] Rerun a pure analytic runtime comparator on the identical case list.
- [x] Skip component work for already exact-feasible guided rows and profile the
  remaining solver-core hot path.
- [x] Vectorize rigid-component collision checks without changing candidate or
  provenance hashes.

Current gate: **EXACT-SAFE QoR CHECKPOINT; runtime promotion blocked**. The clean
large16 retained-proposal benchmark reaches 364/512 exact hard-feasible
constraint candidates, above the legacy control's 338/512 and component
primary's 322/512. Weighted constraint-oracle `J` improves to `2.276194`, and
case 14 improves from `J=3.052426`/45 soft violations to
`J=2.415978`/27 soft violations. Original geometry remains recoverable and the
exact verifier plus Pareto guard remain authoritative.

Schema-v7 clean timing exposes the remaining blocker: vectorized construction
reduces p50/p95 from 22.913/27.623 seconds to 10.070/14.387 seconds with zero
placement-hash differences, but analytic p95 is 0.0441 seconds and the
learned/analytic p95 ratio remains 326.1x. Offline candidate audit is reported
separately and does not pollute solver runtime. Full evidence and limitations
are in
[`hcfp5090_q4_component_bdp_results_2026-08-03.md`](hcfp5090_q4_component_bdp_results_2026-08-03.md).

Explicit pre-projection HPWL perturbation remains deferred. Q4 uses
objective-aware selection among already projected branches, but this does not
substitute for the pending promotion rerun.

## Q5 — DAgger replay and ranker v2

- [x] Define replay schema v3 with fail-closed candidate and checkpoint lineage.
- [x] Record paired initial/post-relax, post-BDP, and exact post-repair states.
- [x] Recompute candidate features from stored geometry on write and read.
- [x] Record feasibility tiers, uncapped `J`, cap margin, soft violations,
  center repair displacement, source kind, row ID, and geometry hashes.
- [x] Derive a versioned 26-D pre-tail feature contract with no post-repair
  target leakage and exact CPU/CUDA grouping-proxy parity.
- [x] Store training-only feature normalization and reject incompatible
  checkpoint continuation/runtime ingestion.
- [~] Add mid-flow snapshots and decision-level teacher actions after pilot proof.
- [x] Build 5,000-record training-only replay with deterministic overlapping
  draw quotas and checksum-valid shards.
- [x] Add post-repair ListMLE with bounded cap-cross weight and legacy fallback.
- [~] Add post-repair multi-task heads after listwise signal is held-out verified.
- [x] Implement top-1, top-4 recall, false-promotion, stage, and weighted-regret reporting.
- [x] Build two sample-ID-disjoint 128-list training replays and one disjoint
  paired 16-sample dev replay.
- [x] Run the 256-list, three-seed initial-stage pilot and evaluate exact gates.
- [x] Add an output-neutral runtime shadow aligned to the replay initial slice.
- [x] Add row-identity-checked selected/oracle visualization across raw,
  post-BDP, and post-repair geometry.
- [x] Run 100-case three-seed shadow validation through the preserved dominance
  safety invariant.

Current gate: **BROAD PROMOTION HOLD**. The historical 16-case pilot passes on
two checkpoints, but the disjoint 512-case split records only 49.2% best top-1,
72.1% best top-4, and one to four false promotions per checkpoint. The
5,000-record v5 training improves top-1 by about five percentage points over v4
but does not approach the 75% / 93.75% gates.

The current post-tail runtime shadow masks exact-ineligible rows and remains
output-neutral. The dormant pre-tail pruning path cannot use that information,
so it remains disabled. A more selective confusable-hard-negative sampler was
also tested and rejected after regressing to 45.9% / 70.5% on the broad split.
Full evidence is in
[`hcfp5090_q6_release_results_2026-08-08.md`](hcfp5090_q6_release_results_2026-08-08.md).

## Q6 — release verification

- [x] Run 100 cases with three deterministic seeds.
- [x] Audit hard feasibility, cap crossings, weighted QoR, ranker coverage, and
  counterfactual selection.
- [ ] Profile A100 cold load, p50/p95 runtime, and peak memory.
- [ ] Verify missing/corrupt checkpoint fallback.
- [ ] Freeze algorithm semantics on Aug 20.
- [ ] Run official wrapper and package dry runs from both supported CWDs.

Current gate: **QOR PASS; RELEASE HOLD**. All three seeds produce 100/100 hard
feasible outputs, 32 improvements, 68 ties, zero regressions, and weighted cost
`9.724486415` versus `9.999999` analytic. The learned p95 remains about
`6.45--6.59 s` versus `0.508 s` analytic, A100 evidence is absent, and active
ranker pruning is unproven.

## Q7 — QoR-first constraint preservation

- [x] Trace MIB loss through shape construction, projection, raw repair, and
  final selection.
- [x] Broadcast compatible shared MIB shapes in the common learned geometry
  projection while preserving hard targets and one-percent area tolerance.
- [x] Add a bounded exact post-tail group repair with componentwise soft and
  bbox/HPWL Pareto guards.
- [x] Uncap cases 96 and 98 without collective rollout or active ranker pruning.
- [x] Eliminate aggregate official-visible MIB violations (`243 -> 0`).
- [x] Cross at least 8/12 original near-cap cases (`11/12`).
- [x] Preserve 100/100 hard feasibility and zero regressions against analytic.
- [~] Preserve every previously retained Q6 learned improvement.
- [x] Add a bounded case-signature portfolio challenger for the six Q6-to-Q7
  regressions, especially case 63, without case-ID memorization.
- [ ] Rerun three deterministic seeds after the portfolio selection fix.

Current gate: **QOR CHECKPOINT PASS; DEFAULT PROMOTION HOLD**. Q7 improves
weighted capped cost from Q6 `9.724486` to `8.989727`, raises exact uncapped
cases from 32 to 70, and improves the 106--120 weighted cost from `9.814157` to
`9.190193`. Runtime increases only about 0.05 seconds at p50 and p95 relative
to Q6, but remains far above analytic. Direct Q6-to-Q7 comparison still has six
per-case regressions because MIB-preserving geometry replaced the old candidate
family. Full evidence is in
[`hcfp5090_q7_mib_group_repair_results_2026-08-11.md`](hcfp5090_q7_mib_group_repair_results_2026-08-11.md).

The first bounded portfolio slice now restores case 63 from the cap to
`7.427046`, improves Q7 weighted cost further to `8.979482`, and changes no
other Q7 case. Five small Q6-to-Q7 quality regressions remain, while the
portfolio stays at 100/100 hard feasible and zero regressions against Q7.

## Required verification per code-bearing commit

```bash
PYTHONPATH=src python -m pytest <targeted tests> -q
PYTHONPATH=src python -m pytest -q
PYTHONPATH=src python -m compileall -q src/hcfp submission
python -m ruff check src/hcfp scripts tests
git diff --check
```

For GPU-bearing changes, add the existing device-parity smoke. For evaluator
changes, include pinned official parity. Documentation-only commits require
Markdown link/path inspection and `git diff --check`.
