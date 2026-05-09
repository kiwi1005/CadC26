# CadC26 research handoff — condensed source of truth

Updated: 2026-05-09 Asia/Taipei. This file replaces the old per-step diary docs
for Step4, Step6, Step7, and Oracle prompts. Closed experiment details were
condensed here; old step docs/prompts were intentionally deleted to keep the
workspace small. For machine evidence, prefer `artifacts/research/*.json` and
`*.jsonl`; generated Markdown/log artifacts are disposable summaries.

## Current state in one minute

- Active architecture: **Step7 DGLPR** — Diagnose -> Generate Alternatives ->
  Legalize -> Pareto Select -> Refine / Iterate.
- Step6 is frozen historical evidence. Do not reopen Step6 sidecars unless the
  user explicitly asks.
- Step7P/Q/R single-block, chain, slot, and gradient local operators produced
  **0 strict meaningful winners** under `MEANINGFUL_COST_EPS=1e-7`.
- Step7S closed the smooth active-face local-search family with
  `local_kkt_stationarity_certified_with_hpwl_hinge_cap` on all 8 representative
  cases: 6 direct KKT-stationary, 8 stationary under HPWL hinge cap, 0 strict
  winners.
- Step7T is the current positive branch: active-soft boundary repair found
  **3 strict winners on 3/8 representative cases** from 400 candidates, with
  `phase4_gate_open=true` and visual sanity / exact replay passing.
- Step7U is a certificate branch, not a generator: bounded disjunctive blocker
  surrogate for case 24/block 32 returned `blocker_obstruction_certificate`,
  0 hard-feasible candidates out of 33, and no strict winner.

Current next action:

```text
Review/integrate the Step7T active-soft exact winners for Phase4 comparison.
Do not lower MEANINGFUL_COST_EPS. Do not restart P/Q/R-style local operators.
Keep Step7U as blocker evidence unless explicitly widening to larger MILP/CP-SAT.
```

## Stable operating rules

- `external/FloorSet/` is upstream truth and local-only. Do not edit it in place.
- Use `PYTHONPATH=src .venv/bin/python ...` for local commands.
- Keep contest/runtime/finalizer paths frozen unless the user explicitly widens
  scope and fresh verification passes.
- Preserve original/current layout anchors in Pareto comparisons.
- Report per-case evidence; do not hide dominated, infeasible, no-op, or
  metric-regressing rows in generated JSON/JSONL.
- HPWL improvement alone is not success if bbox or soft violations regress.
- For future sidecars: keep temporary notes only while active; once closed,
  merge the useful conclusion into this handoff and delete the temporary doc.

## Step4 condensed facts

- Locked comparison: best-untrained mean official cost **19.784**;
  best-trained mean official cost **23.651**; top-5 validation cases
  **14, 18, 17, 15, 11**.
- Failure driver was HPWL + bbox degradation. Displacement proxy was not a
  reliable primary selector.
- Candidate pools contained better-than-baseline solutions, so the promising
  direction was objective-aware reranking, not more coordinate regression.
- V0 reranker artifact: 20-case validation slice, 16 candidates.
- Oracle-style best mean official cost **18.001**; proxy best
  `hpwl_bbox_soft_repair_proxy` mean **18.955**, regret **0.953**.
- Two-stage pairwise LOCO reranker over proxy top-M reached mean **18.578**,
  regret **0.577**, oracle hits **8/20**. Useful, but not a final integration
  gate.

## Step6 condensed facts

Step6G-P was checkpointed before the Step7 pivot. Treat it as evidence only.

- Step6C/6E: immediate labels were horizon-limited on hard cases; rollout-return
  labels were unstable; typed/action-conditioned structure looked more promising
  than scalar hand-crafted deltas, but transfer did not pass gate-level evidence.
- Step6I: virtual-frame boundary commitment passed zero-protrusion and completion
  checks on cases 0..4; sidecar only.
- Step6J: reframed boundary from hard frame-edge commitment to predicted compact
  hull ownership to avoid edge over-commitment.
- Step6K: diagnostic attribution only: bbox edge owners, role overlap, compaction
  baseline; no runtime changes.
- Step6L: global simple compaction was unsafe; guarded selective hull/shape
  probes can help by failure class.
- Step6M: 40-case diagnostic+holdout, 2080 moves; no selected hard-infeasible or
  protrusion rows; boundary gains modest and case-dependent.
- Step6N: added metric-pathology semantics and flagged 12 suspicious simple
  compaction cases; recommended normalized HPWL and spatial imbalance guards.
- Step6O: guard replay reduced suspicious accepts **12 -> 2** and HPWL regression
  mean **0.892 -> 0.471**, but boundary/soft/bbox tradeoffs weakened.
- Step6P: original-inclusive Pareto selector remained a sidecar; do not use it as
  active architecture while Step7 is open.

## Step7 early/mid condensed facts

### A-G/H/I/C real-edit path

- Step7A/B: aspect pathology and shape-policy smoke diagnostics; no final
  architecture change.
- Step7D/E/F: representative coverage, region/topology, and bounded-repair radius
  probes; legality and route classification improved, but no broad winner source.
- Step7G/H: spatial locality and route-aware diversification were useful safety
  layers and were promoted into the real-edit loop. Global candidates must not be
  sent to bounded-local repair.
- Step7C-thin -> real-A..F -> local-iter0 converted route-aware proxies into
  deterministic real rectangle edits. The productive pattern was local/one-block
  real edits with dominance-aware preselection, but strict winners stayed narrow.
- Step7C local-iter0 found **2/8 strict winners**, too narrow to be the final
  path; it motivated stronger preselection and later non-micro operators.
- Step7I/I-R showed coarse regional assignment signal exists, but regional moves
  fail without macro closure / legalization.

### ML/data branch

- Step7DATA and Step7ML-G..K established training-corpus loading, macro-layout
  data mart, geometry-aware decoding, invariant-preserving decoding, and decoder
  candidate ranking.
- These remain candidate-generator/filter sidecars only; no direct solver/runtime
  integration decision was made.

### L/M/N/O closeout

- Step7L learning-guided heatmap targets: 296 phase2 requests; 256 hard-feasible
  non-noop; 0 official-like improving and 0 quality pass; HPWL/bbox/soft all
  regressed on the 256 feasible non-noops. Generator expansion stopped.
- Step7M objective-aligned corridors: 632 opportunity rows, 72 phase1 requests,
  72 hard-feasible non-noop phase2 rows, 50 all-vector nonregressing, 0
  meaningful winners; phase4 multiblock also 0 winners. Keep vector gate as
  diagnostic, not generator.
- Step7N archive-lineage reservoir: 7507 normalized rows, 824 exact comparable,
  0 strict archive candidates and 0 strict meaningful non-micro winners.
- Step7O training-demand prior: 96 atlas rows, 223 calibration rows; preserved
  known winners but concentration failed and no new cross-case source emerged.

Decision from L/M/N/O: stop reranking/micro-widening and seek a causal move or
active-soft repair source.

## Step7 terminal/current condensed facts

- Step7P causal closure repack: Phase0/1/2 plumbing worked; Phase3 had 120 unique
  requests over 8 cases but replay stopped at `stop_phase3_replay_gate`, 0 strict
  meaningful winners. Branch C (`balanced_failure_budget`) was best partial
  method but still had no strict winners.
- Step7Q constrained operator learning: non-leaky data mart and risk ranking
  worked as filters. Step7Q-F objective-aware slot replay reached 96 hard-feasible
  non-noop rows and 27 all-vector nonregressing rows but still 0 strict winners.
- Step7R chain/swap/gradient expansion: 0 strict winners. The 27 AVNR Step7Q-F
  rows collapsed to 2 unique projected candidates; best official decrease was
  about `1.477e-8`, roughly 6.77x short of the `1e-7` strict threshold.
- Step7S critical cone: closed the smooth active-face hypothesis. If future data
  or threshold changes, rerun the certificate before reopening this family.
- Step7T active-soft cone: branch `exp/step7t-active-soft-cone`, worktree
  `/home/hwchen/PROJ/floorplan-step7t-active-soft`. Results:
  - `candidate_count=400`
  - `strict_winner_count=3`
  - `strict_winner_case_count=3/8`
  - `phase4_gate_open=true`
  - exact visual sanity replay passed for cases 24, 51, 76 with stored/exact delta
    max error `0.0`.
- Step7U blocker MILP surrogate: branch `exp/step7u-disjunctive-blocker-milp`,
  worktree `/home/hwchen/PROJ/floorplan-step7u-blocker-milp`. Results:
  - `decision=blocker_obstruction_certificate`
  - case 24/block 32 closure `[3, 32, 39, 44]`
  - 9 true flips rejected as unbounded; required moves roughly 28-46 > `max_move=8`
  - 24 bounded local repack surrogates all hard-infeasible
  - 0 strict winners.

## Retention policy after 2026-05-09 cleanup

Kept docs:

- `README.md` — external onboarding and current status.
- `AGENT.md` — AI/operator routing and red lines.
- `HELP.md` — short live workspace note.
- `docs/cadc26_research_handoff.md` — this condensed research history.
- `docs/05_semantic_rollout.md`, `docs/06_repair_finalizer.md`,
  `docs/07_strict_evaluation.md`, `docs/08_known_failure_modes.md`,
  `docs/research-experiment-manual.md` — evergreen conceptual/manual docs.

Deleted/merged docs:

- Step4 docs: objective reranking plan and Step4 synthesis/integration notes.
- Step6 docs: Step6C/E/F/I/J/K/L/M/N/O/P and legacy inventory notes.
- Step7 diary docs: Step7A/B/C/D/E/F/G/H/I/L/M/N/O/P/Q/R/S, Step7ML, Step7DATA,
  old midphase/terminal digests, and cleanup inventory.
- Oracle prompt docs: Step7 RL/GNN, L/M/O/Q/R prompts.
- Generated artifact Markdown/log files under `artifacts/research/` are safe to
  delete after their key facts are captured here; retain JSON/JSONL evidence when
  exact replay data may still be needed.
