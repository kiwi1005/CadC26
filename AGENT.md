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
- Condensed research handoff: `docs/cadc26_research_handoff.md`
- Generated machine evidence: `artifacts/research/`
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

When local docs disagree, prefer `docs/cadc26_research_handoff.md` plus the
freshest matching `artifacts/research/*.json` / `*.jsonl` evidence over older
README/HELP summaries. Generated Markdown/log artifacts are disposable summaries.

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
11. **Step7S (critical-cone certificate, 2026-05-09, closed) is the smooth
    active-face mathematical certificate** for the single-block / chain / HPWL
    gradient / coordinated-CCQP local-search family. It returned
    `local_kkt_stationarity_certified_with_hpwl_hinge_cap` across all 8
    representative cases: 6 direct `kkt_stationary`, 2
    `kkt_stationary_under_hpwl_hinge_cap`, 0 strict winners. This closes the
    P/Q/R-style local operator family under the unchanged
    `MEANINGFUL_COST_EPS = 1e-7`, but it explicitly leaves discrete soft repair
    and cross-cell MILP-style moves as separate hypotheses.
12. **Step7T (active-soft cone, branch `exp/step7t-active-soft-cone`,
    2026-05-09) is the current positive branch.** In worktree
    `/home/hwchen/PROJ/floorplan-step7t-active-soft`, the sidecar found
    `strict_winner_count=3` on `3/8` representative cases from 400 bounded
    boundary-snap / seed-compensation candidates, with `phase4_gate_open=true`
    and `MEANINGFUL_COST_EPS` unchanged. Lead-side visual sanity replay passed
    for cases 24, 51, and 76 (`exact_strict_winner_count=3`, max stored/exact
    delta error `0.0`).
13. **Step7U (bounded disjunctive blocker MILP surrogate, branch
    `exp/step7u-disjunctive-blocker-milp`, 2026-05-09) is a certificate branch.**
    For case 24 / block 32 it returned
    `decision=blocker_obstruction_certificate`, `strict_meaningful_winner_count=0`,
    `hard_feasible_count=0/33`. The minimal seed component `[3,32,39,44]` needs
    vertical separation flips of roughly 28-46 units, beyond the bounded
    `max_move=8` probe. Keep it as evidence unless explicitly widening to a
    larger MILP / CP-SAT repacker.

Current bottleneck to state explicitly in future reports:

```text
Step7S closes smooth local-search faces; Step7T proves discrete active-soft
boundary repair is the current viable path (3 strict winners / 8 reps); Step7U
shows bounded one-blocker separation flips do not solve case24/block32.
```

Forbidden as a "next research lane" without explicit user direction:

- Re-opening any single-block Step7P/Q/R-style local operator family on the same
  8 representative cases. The Step7S certificate closes it. If FloorSet, the
  case set, or `MEANINGFUL_COST_EPS` changes, re-run
  `scripts/step7s_run_all_cases_certificate.py` first.
- Lowering `MEANINGFUL_COST_EPS` below `1e-7` to rescue HPWL hinge cleanups.
- Treating Step7U bounded one-blocker results as a generator without widening the
  formal search scope; its current role is obstruction certificate.

Recommended next research direction:

```text
Prioritize Step7T exact/visual-reviewed active-soft winners for Phase4 review
and integration comparison. Preserve Step7U as blocker evidence; only widen it
if explicitly moving to larger MILP/CP-SAT separation-cell search.
```

Condensed handoff doc:

- `docs/cadc26_research_handoff.md` — Step4/6/7 condensed history, current
  Step7T/Step7U direction, and cleanup retention policy. Old per-step diary docs
  and Oracle prompt docs were merged here and deleted on 2026-05-09.

## Default task routing

Before editing, read the relevant local governing artifacts:

- Always: `AGENT.md`, `README.md`, `GUIDE.md`, `MEMORY.md`,
  `docs/cadc26_research_handoff.md`, and the freshest relevant
  `artifacts/research/*.json` / `*.jsonl` evidence.
- Step7G/H/C/I/N/ML/R/T/U work: start from `docs/cadc26_research_handoff.md`,
  the matching script/module/test, and the freshest machine evidence under
  `artifacts/research/`.
- Step6 work: use the condensed Step6 section in
  `docs/cadc26_research_handoff.md`; Step6 sidecars stay frozen unless the user
  explicitly opens that branch.
- Contest/runtime work: inspect `contest_optimizer.py`,
  `src/puzzleplace/optimizer/contest.py`, and current official smoke/eval paths
  before changing runtime behavior.

Use sidecar-first execution for research ideas:

1. If needed, keep a temporary active plan/result note while the sidecar is open.
2. Implement the smallest module/script/test slice.
3. Emit machine-readable artifacts under `artifacts/research/`.
4. Report exact commands, artifact paths, metrics, and the next bottleneck.
5. When the sidecar closes, merge useful conclusions into
   `docs/cadc26_research_handoff.md` and delete the temporary diary doc.
6. Only widen scope after evidence justifies it.

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

- `scripts/step7*.py` as the deterministic runner.
- `src/puzzleplace/...` module code in the correct lane.
- `tests/test_step7*.py` for the new behavior and artifact contracts.
- JSON/JSONL reports under `artifacts/research/` sufficient to audit counts,
  routes, feasibility, metrics, dominance/filtering, and failures.
- A temporary Markdown note only if it helps active work; close it by merging the
  useful facts into `docs/cadc26_research_handoff.md` and deleting the diary note.

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
