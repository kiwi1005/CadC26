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
inputs to Q4 component-aware projection. Q4 is therefore unblocked as the
remediation stage; Q3 remains deferred until Q4 preserves the Q2 QoR gain.

## Q3 — collective dynamics

- [ ] Build current-geometry pair features each rollout step.
- [ ] Add three message-passing updates with explicit x/y semantics.
- [ ] Feed RMS-normalized typed forces into coordinate updates.
- [ ] Connect learned force gates to active typed-force channels.
- [ ] Run rollout stability and CPU/CUDA differential tests.

Q3 remains deferred. It starts only after a clean-commit Q4 rerun preserves
projection success and proves the learned-versus-analytic runtime gate.

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
- [~] Compare feasibility, displacement, HPWL, bbox, and p95 runtime from one clean commit.
- [~] Embed solver commit and clean-tree proof in component/analytic artifacts.
- [~] Rerun a pure analytic runtime comparator on the identical case list.
- [~] Skip component work for already exact-feasible guided rows and profile the
  remaining solver-core hot path.

Current gate: **EXACT-SAFE QoR CHECKPOINT; runtime promotion blocked**. A dirty
large16 retained-proposal diagnostic reaches 364/512 exact hard-feasible
constraint candidates, above the legacy control's 338/512 and component
primary's 322/512. Weighted constraint-oracle `J` improves to `2.276194`, and
case 14 improves from `J=3.052426`/45 soft violations to
`J=2.415978`/27 soft violations. Original geometry remains recoverable and the
exact verifier plus Pareto guard remain authoritative.

Schema-v7 timing exposes the remaining blocker: a paired one-case smoke is
22.765 seconds learned versus 0.243 seconds analytic (93.6x), with 22.004
seconds in solver core and 0.761 seconds in runtime-final selection. Offline
candidate audit is reported separately and does not pollute solver runtime.
The full large16 comparator still needs a clean commit, but this gap already
prevents promotion. Full evidence and limitations are in
[`hcfp5090_q4_component_bdp_results_2026-08-03.md`](hcfp5090_q4_component_bdp_results_2026-08-03.md).

Explicit pre-projection HPWL perturbation remains deferred. Q4 uses
objective-aware selection among already projected branches, but this does not
substitute for the pending promotion rerun.

## Q5 — DAgger replay and ranker v2

- [ ] Define replay-v2 schema and provenance hashes.
- [ ] Record raw/mid-flow/pre-BDP/post-BDP/post-repair states.
- [ ] Build 5,000-record training-only replay with required composition.
- [ ] Add post-repair multi-task targets and listwise loss.
- [ ] Validate top-1, top-4 recall, false promotions, and weighted regret.
- [ ] Run full validation through the preserved dominance safety invariant.

## Q6 — release verification

- [ ] Run 100 cases with at least three deterministic seeds.
- [ ] Audit hard feasibility, cap crossings, weighted `J`, oracle@K, and regret.
- [ ] Profile A100 cold load, p50/p95 runtime, and peak memory.
- [ ] Verify missing/corrupt checkpoint fallback.
- [ ] Freeze algorithm semantics on Aug 20.
- [ ] Run official wrapper and package dry runs from both supported CWDs.

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
