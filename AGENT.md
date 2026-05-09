# AGENT.md — CadC26 / FloorSet Puzzle Agent Guide

This file is the project-local operating guide for AI/coding agents working in
`/home/hwchen/PROJ/CadC26`. It is intentionally project-specific: keep global
Codex/OMX behavior in the root/session `AGENTS.md`, and keep this file focused
on CadC26 architecture, workflow, verification, and red lines.

> Codex note: OpenAI Codex natively discovers `AGENTS.md` files, with closer
> directory files overriding broader guidance. This repository historically uses
> `AGENT.md` as the project memory/operator guide; if automatic Codex discovery
> is required, either keep a root `AGENTS.md` in sync or configure Codex fallback
> filenames. Official reference: https://developers.openai.com/codex/guides/agents-md

## Project identity

CadC26 is a local FloorSet-Lite / ICCAD 2026 floorplanning research workspace.
It contains a runnable Python package, contest smoke entrypoints, tests, and a
sequence of sidecar research steps around floorplan diagnosis, candidate
generation, legalization, Pareto/ranker selection, and supervised macro-layout
experiments.

- Package root: `src/puzzleplace/`
- Scripts: `scripts/`
- Tests: `tests/`
- Research plans/results: `docs/step*.md`
- Generated evidence: `artifacts/research/`
- Research protocol / handoff notes: `GUIDE.md`, `MEMORY.md`
- Contest entrypoint: `contest_optimizer.py` -> `src/puzzleplace/optimizer/contest.py`
- Official FloorSet checkout: `external/FloorSet/` (local-only, untracked)

## Current architectural frame

The active research architecture is Step7, not the older Step6 training-first or
coordinate-regression framing.

```text
DGLPR = Diagnose -> Generate Alternatives -> Legalize -> Pareto Select -> Refine / Iterate
```

Primary Step7 module lanes:

- Diagnose: `src/puzzleplace/diagnostics/`
- Generate alternatives: `src/puzzleplace/alternatives/`
- Legalize / repair: `src/puzzleplace/legalization/`, `src/puzzleplace/repair/`
- Pareto / ranking / search: `src/puzzleplace/search/`, `src/puzzleplace/scoring/`
- ML sidecars: `src/puzzleplace/ml/`
- Representative suites / replays: `src/puzzleplace/experiments/`

Treat Step6 sidecars as frozen historical evidence unless the user explicitly
asks to return to Step6.

## Current evidence chain and likely next bottleneck

When local docs disagree, prefer the freshest `artifacts/research/*_decision.md`
and matching `docs/step*.md` over older README/HELP summaries.

Current high-level chain:

1. Step7G established spatial locality routing as a safety layer:
   global candidates must not be sent to bounded-local repair.
2. Step7H and Step7C-thin proved route-aware candidate descriptors can become
   deterministic real rectangle edits while preserving original-inclusive Pareto
   reporting.
3. Step7C-real-A..F moved to official-aligned real placement edits. The best
   local-lane result is Step7C-real-F:
   `promote_to_step7c_local_lane_iteration`, with dominance-aware preselection
   reducing dominated probes and preserving Step7C-real-E winners.
4. Step7I / Step7I-R showed coarse regional assignment signal exists, but raw
   regional translations and small slot matching do not solve useful legality;
   macro/group closure dominates the failure mode.
5. Step7N-G/H/I shifted the blocker from legality to quality: many hard-feasible
   non-noop macro/topology rows exist, but most are dominated or metric-regressing.
   Step7N-I promotes a constrained Pareto quality filter back into the Step7N-G
   sidecar archive path.
6. Step7ML-G..K built the training-backed macro-layout data mart and geometry
   decoders. Step7ML-I/K improve hard feasibility and reduce overlap, but the
   quality-gate / official-like winner count remains narrow; use these decoders
   as candidate generators feeding constrained selectors, not as final solvers.
7. Step7L/M/N/O tested heatmap targets, objective corridors, archive lineage,
   and training-demand prior calibration. They improved diagnosis but did not
   create a broad new winner source: Step7O Phase2 is report-only, Step7O Phase3
   and GNN/RL remain closed, and the Oracle diagnosis says the missing piece is
   a causal move/repack operator rather than another ranker or micro-perturbation.
8. Step7P-CCR was the active sidecar lane through 2026-05-08. Phase0 locked the
   stagnation evidence, Phase1 produced a causal atlas (`subproblem_count=325`,
   8 represented cases), Phase2 proved a synthetic causal closure repacker,
   Phase3 covered all 8 cases with 120 unique requests. The bounded replay gate
   reported zero strict meaningful winners. A three-branch redesign found
   `branch_c_balanced_failure_budget` as the best partial method.
9. Step7Q-A..F evolved through risk ranking (Step7Q-B), parameter expansion
   (Step7Q-C), fresh-metric replay (Step7Q-D), obstacle-aware slot replay
   (Step7Q-E), and objective-aware slot replay (Step7Q-F). Step7Q-F is the
   single-block local-search reference point: 96 source rows, 27 AVNR, 0 strict.
10. Step7R (chain-move operator, 2026-05-08, closed) tested k=2 swap and HPWL
    gradient nudge. Both produced 0 strict winners. The lane also exposed that
    Step7Q-F's 27 AVNR rows collapse to **2 unique** `(case, block, target_box)`
    candidates after slot-finder projection. See
    `artifacts/research/step7r_close_decision.json`.
11. **Step7S (critical-cone certificate, 2026-05-09, closed) is the project's
    terminal mathematical certificate** for the single-block local-search
    family. The Oracle-derived primal/dual program (CCQP) returned
    `local_kkt_stationarity_certified_with_hpwl_hinge_cap` across all 8
    representative cases:
    - 6 cases: `kkt_stationary` (analytic g_cost = 0; both HPWL and bbox hinges
      are inactive, so no first-order descent direction exists);
    - 2 cases (19, 25): `kkt_stationary_under_hpwl_hinge_cap` (LP rho looks
      promising but the analytic hinge cap `0.5·exp(2v)·max(0,h0)` is < 1e-7
      by ~6-8x, so no realizable strict descent exists).
    The certificate also proved Oracle's hinge formula
    `c = (1 + 0.5·max(0,h) + 0.5·max(0,a))·exp(2v)` to bit-level (max abs err
    `0.0` across 192 cost rows). Single-block, k=2 swap, HPWL gradient nudge,
    and coordinated multi-block CCQP descent are mathematically exhausted on
    the 8 representative cases under the unchanged `MEANINGFUL_COST_EPS = 1e-7`
    threshold.

Current bottleneck to state explicitly in future reports:

```text
single-block local-search family is closed by mathematical certificate;
strict winners now require either discrete soft-violation repair (active-soft
cone search), bounded disjunctive blocker MILP across separation cells, or
project-level reformulation
```

Forbidden as a "next research lane" without explicit user direction:

- Re-opening any single-block local operator family (Step7P/Q/R-style) on the
  same 8 representative cases. The Step7S certificate forbids spending compute
  on it. If a future change to FloorSet, the case set, or `MEANINGFUL_COST_EPS`
  invalidates the certificate, re-run `scripts/step7s_run_all_cases_certificate.py`
  first to update the artifact before opening anything new.
- Lowering `MEANINGFUL_COST_EPS` below `1e-7` to "rescue" cases 19/24/25/51.
  The Oracle threshold analysis explicitly identified these as HPWL hinge
  cleanups, not meaningful official progress.

Recommended next research directions (in increasing scope):

```text
A. Active-soft constrained descent — case 24 unique violated soft component first;
   expected Δc ≈ -7.4e-2 if the local repair is feasible, else Farkas certificate.
B. Disjunctive blocker MILP — bounded separation-flip search over the smallest
   blocker component for case 24 / block 32 closure.
C. Reformulate at a higher level — e.g. global SA over macro subsets, or a
   CP-SAT/MILP fixed-shape repacker on the closure-aware sub-instance.
```

Default when the user does not specify a lane: do NOT auto-pick. Ask which of
A/B/C, or wait for an Oracle re-diagnosis.

## Default task routing

Before editing, read the relevant local governing artifacts:

- Always: `AGENT.md`, `README.md`, `GUIDE.md`, `MEMORY.md`, and the freshest
  relevant `artifacts/research/*_decision.md`.
- Step7G/H/C/I/N/ML/R work: matching `docs/step7*.md`, script, module, test,
  and any handoff notes in `MEMORY.md`.
- Step6 work: read the Step6 docs and keep Step6 sidecars frozen unless the user
  explicitly opens that branch.
- Contest/runtime work: inspect `contest_optimizer.py`,
  `src/puzzleplace/optimizer/contest.py`, and current official smoke/eval paths
  before changing runtime behavior.

Use sidecar-first execution for research ideas:

1. Write or update the `docs/step*.md` plan/result note.
2. Implement the smallest module/script/test slice.
3. Emit machine-readable artifacts under `artifacts/research/`.
4. Report exact commands, artifact paths, metrics, and the next bottleneck.
5. Only widen scope after evidence justifies it.

## Non-negotiable constraints

- Do not edit `external/FloorSet/` in place. Treat it as upstream truth and
  local data source only.
- Do not vendor FloorSet datasets or generated heavy artifacts into git.
- Do not promote sidecar behavior into `contest_optimizer.py` or runtime solver
  paths without explicit scope widening and fresh verification.
- Do not change finalizer semantics as a side effect of a sidecar experiment.
- Do not send global candidates to bounded-local repair.
- Preserve original/current layout anchors in Pareto comparisons.
- Keep rejected, dominated, infeasible, no-op, global, and metric-regressing rows
  visible in JSON/report artifacts; do not silently drop evidence.
- Avoid scalar penalty soup. Prefer constrained objective vectors, Pareto fronts,
  explicit filters, and per-reason attribution.
- Do not report only averages for large/XL or representative cases; include
  per-case and per-profile breakdowns.
- Do not treat HPWL improvement alone as success when bbox area or soft
  violations regress.
- Do not introduce new dependencies unless explicitly requested.

## Environment and commands

Use the repository environment explicitly:

```bash
PYTHONPATH=src .venv/bin/python <script-or-module>
```

This checkout's `.venv` may point at the shared `cadc-baseline` environment;
do not create a second environment unless the user asks.

### Hardware and parallelism (default ON)

Workstation: **48 CPU cores, 128 GB RAM**. Parallelize embarrassingly-parallel
work by default — do not silently run single-process for per-row / per-case
loops. The user has explicitly asked that this stay on without being reminded.

Apply to:

- Per-case / per-row research replays (Step7P/Q/R fresh-metric loops, ablation
  family loops, candidate enumeration). Use `multiprocessing.Pool`,
  `concurrent.futures.ProcessPoolExecutor`, or `joblib.Parallel(n_jobs=-1)`.
- pytest: prefer `pytest -n auto` (pytest-xdist) for suites that take more than
  a few seconds. Do not enable for tests with shared filesystem state.
- ruff / mypy / pytest verification: launch in parallel Bash calls in the same
  message rather than chaining sequentially.
- Codex sub-agent prompts: explicitly tell Codex to parallelize per-row replay
  with a worker count of `min(48, n_rows)` or `n_jobs=-1`.

Caveats: keep single-process when the operation has cross-row state, ordering
matters, or the underlying library is not process-safe. GPU/RL training is
unaffected by this rule (single-GPU semantics still apply).

Common verification commands:

```bash
PYTHONPATH=src .venv/bin/python -m pytest tests/test_<changed_area>.py
PYTHONPATH=src .venv/bin/python -m mypy <typed modules/scripts>
PYTHONPATH=src .venv/bin/python -m ruff check <touched python files>
```

For broad checks after larger edits:

```bash
PYTHONPATH=src .venv/bin/python -m pytest
PYTHONPATH=src .venv/bin/python -m ruff check src scripts tests
```

For Step7 scripts, prefer targeted smoke runs first, then tests:

```bash
PYTHONPATH=src .venv/bin/python scripts/<step7_script>.py
PYTHONPATH=src .venv/bin/python -m pytest tests/<matching_test>.py
```

## Documentation and artifact expectations

Every new Step7 sidecar should normally produce:

- `docs/step7*.md` with scope, method, artifacts, run result, interpretation,
  decision, and next direction.
- `scripts/step7*.py` as the deterministic runner.
- `src/puzzleplace/...` module code in the correct lane.
- `tests/test_step7*.py` for the new behavior and artifact contracts.
- `artifacts/research/step7*_decision.md` plus JSON reports sufficient to audit
  counts, routes, feasibility, metrics, dominance/filtering, and failures.

Reports should answer:

- What changed?
- Which cases were covered?
- Which candidates were generated, retained, rejected, or report-only?
- Which route classes appeared?
- Did hard feasibility improve?
- Did official-like cost improve?
- Were HPWL gains cancelled by bbox/soft penalties?
- Is the current blocker legality, targeting, ranking, quality, runtime, or data?

## Current module map

Use existing lanes before adding new abstractions:

- `src/puzzleplace/alternatives/locality_routing.py` — Step7G route classifier.
- `src/puzzleplace/alternatives/route_aware_candidates.py` — route-aware
  descriptor generation.
- `src/puzzleplace/alternatives/route_aware_layout_edits.py` — deterministic
  layout-edit candidates.
- `src/puzzleplace/alternatives/real_placement_edits.py` — official-aligned real
  placement edit helpers.
- `src/puzzleplace/alternatives/metric_directed_edits.py` — metric-directed edit
  targeting.
- `src/puzzleplace/alternatives/window_repack_edits.py` — window repack probes.
- `src/puzzleplace/alternatives/pareto_slack_fit_edits.py` — Pareto-constrained
  slack-fit candidates.
- `src/puzzleplace/alternatives/dominance_slack_preselection.py` — dominance-aware
  preselection.
- `src/puzzleplace/alternatives/coarse_region_flow.py` — Step7I coarse region
  assignment probe.
- `src/puzzleplace/alternatives/regional_legalizer.py` — route-specific regional
  legalizer probe.
- `src/puzzleplace/ml/` — training-backed macro layout, topology/heatmap, and
  training-prior sidecars.
- `src/puzzleplace/repack/` — Step7P causal subproblem and synthetic causal
  closure repacker utilities.
- `src/puzzleplace/search/` — iteration/ranking/integration helpers.
- `src/puzzleplace/eval/` — metrics, official-like scoring, reports, violations.

## Working style for agents

- Start from real local artifacts and logs, not memory alone.
- Keep edits small, reviewable, and reversible.
- Prefer extending existing sidecar patterns over inventing a new framework.
- If the user says to read a plan/doc first, do that before code changes.
- If the user asks for progress/status, give exact branch, files, commands,
  metrics, artifacts, blockers, and next bottleneck.
- If the user asks for visuals, produce directly inspectable PNG artifacts by
  default when possible.
- If a task uncovers open questions but the user asked not to interrupt, record
  them in `HELP.md`.
- Before claiming done, run the smallest verification that proves the changed
  behavior and report the evidence.

## Commit guidance

If asked to commit, follow the repository/session Lore commit protocol: the
intent line explains why, and trailers capture constraints, rejected alternatives,
confidence, scope risk, directives, tested commands, and not-tested gaps.


## 2026-05-08 Step7Q-A data mart

Step7Q-A non-leaky operator-learning data mart passed: `artifacts/research/step7q_operator_data_mart_summary.json` reports `decision=promote_to_constrained_risk_ranking`, `example_count=325`, `feature_label_leakage_count=0`, `strict_meaningful_positive_count=0`, and `allowed_next_phase=step7q_constrained_risk_ranking`. Next work is constrained risk/ranking, not direct coordinate RL or Phase4.

## 2026-05-08 Step7Q-B constrained risk/ranking smoke

Step7Q-B completed a non-leaky feature/mask risk scorer and source-deck selector. Important fix: do not count the Phase3 request `forbidden` policy declaration as selected forbidden operator usage; only actual source/action text should drive `forbidden_request_term`. Current artifact `artifacts/research/step7q_operator_policy_summary.json` reports `decision=risk_ranking_smoke_pass_strict_gate_closed`, 96 selected rows, 8 represented cases, largest share 0.25, forbidden count 0, overlap 23 (< Branch C 36), soft regression 0.052083333333333336 (< Branch C 0.2916666666666667), bbox 0.0, but strict meaningful winners still 0. Keep Phase4 closed. Next safe lane: constrained operator parameter expansion guarded by this risk scorer; still no contest-runtime/finalizer/direct-coordinate RL changes.

## 2026-05-08 Step7Q-C operator-parameter expansion

Step7Q-C is complete as a sidecar expansion deck. New files: `src/puzzleplace/ml/step7q_operator_parameter_expansion.py`, `scripts/step7q_expand_operator_parameters.py`, `tests/test_step7q_operator_parameter_expansion.py`. Summary `artifacts/research/step7q_parameter_expansion_summary.json`: `decision=parameter_expansion_deck_ready_for_fresh_replay`, 550 candidates, 96 selected variants, 8 cases, largest share 0.25, 69 unique parents, 8 unique action signatures, forbidden action terms 0, direct coordinate fields 0. Do not treat this as a win claim: `fresh_replay_required=true`, `strict_winner_evidence_count=0`, `phase4_gate_open=false`. Next safe lane is `step7q_fresh_metric_replay_executor` for the expansion deck; still no contest runtime/finalizer/direct-coordinate RL changes.

## 2026-05-08 Step7Q-D fresh metric replay

Step7Q-D added `src/puzzleplace/ml/step7q_fresh_metric_replay.py`, `scripts/step7q_replay_parameter_expansion.py`, and `tests/test_step7q_fresh_metric_replay.py`. It executed `artifacts/research/step7q_parameter_expansion_deck.jsonl` through a sidecar finite-action single-block bridge using validation labels only inside replay/evaluation. Summary `artifacts/research/step7q_fresh_metric_replay_summary.json`: `decision=fresh_replay_executed_strict_gate_closed`, 96 requests, 9 fresh metrics, 9 hard feasible non-noop, 87 overlap-after-splice, soft regression rate 0.020833333333333332, bbox regression rate 0.0, 7 all-vector nonregressing, strict winners 0, `phase4_gate_open=false`. Next safe lane: obstacle-aware target/slot executor for finite Step7Q actions before more RL/GNN; do not open Phase4 or touch contest runtime/finalizer.

## 2026-05-08 Step7Q-E slot-aware replay

Step7Q-E extended `src/puzzleplace/ml/step7q_fresh_metric_replay.py` and `scripts/step7q_replay_parameter_expansion.py` with `--slot-aware`, using local non-overlap slot search only inside the replay/evaluation boundary. Summary `artifacts/research/step7q_slot_aware_replay_summary.json`: 96/96 fresh metric and hard feasible non-noop, 0 overlap-after-splice, 87 slot-adjusted actions, but soft regression 0.9270833333333334, bbox regression 0.90625, hpwl regression 0.90625, all-vector nonregressing 7, strict winners 0, `phase4_gate_open=false`. Next safe lane: objective-aware slot scoring/selection under non-overlap guard. Do not open Phase4 or touch contest runtime/finalizer.

## 2026-05-08 Step7Q-F objective-aware slot replay

Step7Q-F extended `src/puzzleplace/ml/step7q_fresh_metric_replay.py` and `scripts/step7q_replay_parameter_expansion.py` with `--objective-aware-slot`. Summary `artifacts/research/step7q_objective_slot_replay_summary.json`: 96 requests, 96 fresh/hard feasible, overlap 0, slot-adjusted 87, objective-aware true, soft regression 0.71875, bbox regression 0.6979166666666666, hpwl regression 0.6979166666666666, all-vector nonregressing 27, strict winners 0, `risk_replay_gate_open=false`, `phase4_gate_open=false`. This improves Step7Q-E but shows the slot candidate pool is still poor. Next safe lane: generate better local objective-preserving vacancy candidates before scoring. Runtime/finalizer/Phase4 remain frozen.
