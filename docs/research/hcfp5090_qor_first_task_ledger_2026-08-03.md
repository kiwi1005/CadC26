# HCFP-5090 QoR-first task ledger

Canonical design:
[`hcfp5090_qor_first_implementation_plan_2026-08-03.md`](hcfp5090_qor_first_implementation_plan_2026-08-03.md)

Status values: `[ ]` pending, `[~]` active, `[x]` verified, `[blocked]` gate failed.
A later stage does not start when its required promotion gate is blocked.

## Q0 — exact cap attribution

- [~] Implement exact log-domain uncapped cost and cap margin.
- [~] Attribute boundary, grouping, MIB, quality, and runtime terms.
- [~] Calculate counterfactual soft/quality fixes and blocker class.
- [~] Add deterministic JSON/Markdown reporting CLI.
- [ ] Match pinned official evaluator on unit fixtures.
- [ ] Run 100-case attribution and preserve its report artifact.
- [ ] Commit Q0 with no runtime-selection diff.

## Q1 — cycle-free structured topology

- [~] Extract set-valued relation masks from gold rectangles.
- [~] Add partial-label NLL and relation antisymmetry loss.
- [~] Implement dependency-free Sinkhorn and deterministic hard permutation.
- [~] Decode dual permutations into cycle-free H/V constraints.
- [~] Add preplaced compatibility/adaptation and longest-path packing.
- [ ] Add group/MIB identity and same-group/same-MIB pair features.
- [ ] Connect topology output to learned runtime candidates behind an opt-in flag.
- [ ] Record topology provenance through post-BDP candidates.
- [ ] Run shelf-vs-topology oracle@16 on internal held-out cases.
- [ ] Stop and diagnose if the 106--120 weighted oracle does not improve.

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

