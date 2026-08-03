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

- [ ] Extract side-specific cluster contact labels.
- [ ] Decode a deterministic maximum contact spanning tree per group.
- [ ] Implement hysteretic contact latch state.
- [ ] Add boundary virtual nodes and per-side slot order.
- [ ] Construct compatible MIB shapes from one group variable.
- [ ] Measure group connectivity, soft violations, and repair displacement.

## Q3 — collective dynamics

- [ ] Build current-geometry pair features each rollout step.
- [ ] Add three message-passing updates with explicit x/y semantics.
- [ ] Feed RMS-normalized typed forces into coordinate updates.
- [ ] Connect learned force gates to active typed-force channels.
- [ ] Run rollout stability and CPU/CUDA differential tests.

Q3 starts only when Q1/Q2 increase useful candidate density.

## Q4 — component-aware BDP

- [ ] Build and refresh conflict connected components every outer sweep.
- [ ] Score directions with movement, HPWL, topology, latch, and boundary terms.
- [ ] Branch only uncertain directions and reject cycles.
- [ ] Add repeated-signature reset and bounded HPWL superiorization.
- [ ] Compare feasibility, displacement, HPWL, bbox, and p95 runtime.

## Q5 — DAgger replay and ranker v2

- [ ] Define replay-v2 schema and provenance hashes.
- [ ] Record raw/mid-flow/pre-BDP/post-BDP/post-repair states.
- [ ] Build 5,000-record training-only replay with required composition.
- [ ] Add post-repair multi-task targets and listwise loss.
- [ ] Validate top-1, top-4 recall, false promotions, and weighted regret.
- [ ] Run full validation through unchanged Pareto guard.

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
