# Oracle request: CadC26 Step7 RL/GNN system redesign plan

You are GPT-5.5 Pro acting as a senior EDA floorplanning + ML/RL/GNN research architect. I need an evidence-grounded, implementable experimental plan for the CadC26 / FloorSet-Lite research repo. Please read the attached project docs, decision artifacts, and representative source files before answering. Your output will be handed to a coding agent to execute in this repository.

## Project briefing

Repository: `/home/hwchen/PROJ/CadC26`.

This project targets ICCAD 2026 FloorSet-Lite floorplanning. The local Python package is under `src/puzzleplace/`, runnable scripts are under `scripts/`, tests under `tests/`, research notes under `docs/`, and machine-readable evidence under `artifacts/research/`. The official FloorSet checkout and datasets live under `external/FloorSet/` and must not be edited or vendored. Development uses:

```bash
PYTHONPATH=src .venv/bin/python <script-or-module>
PYTHONPATH=src .venv/bin/python -m pytest <tests>
PYTHONPATH=src .venv/bin/python -m mypy <typed modules/scripts>
PYTHONPATH=src .venv/bin/python -m ruff check <touched files>
```

The current architecture is Step7, summarized as:

```text
diagnose
-> generate alternatives
-> route by locality
-> legalize / repack
-> full-case fresh metric replay
-> constrained Pareto / quality gate
-> decision
```

Core project rules:

- Keep all new experiments sidecar-only until explicitly promoted.
- Do not integrate into contest runtime or change finalizer semantics unless the plan explicitly reaches that gate and fresh verification passes.
- Preserve original/current layout anchors in Pareto comparisons.
- Report failed, no-op, infeasible, dominated, metric-regressing, global, and timeout candidates; do not silently drop evidence.
- Avoid case-specific rules, especially case024/case025/case076 tuning.
- Avoid scalar penalty soup and magic HPWL/bbox/soft thresholds; prefer objective vectors, constrained Pareto gates, masks, learned priors, and explicit failure attribution.
- Fresh full-case official-like replay is required before counting a candidate as a real win. Provenance/projected/proxy/closure-local wins are not fresh wins.
- Training/data-driven methods are preferred when they learn general priors, targets, actions, heatmaps, orderings, or rankings from FloorSet training data.

## Dataset facts already verified locally

FloorSet-Lite provides:

- 1,000,000 training samples.
- 100 visible validation cases.
- 100 hidden test cases.
- Cases contain 21 to 120 blocks.
- Training labels include `fp_sol` final placement geometry; Step7DATA / Step7ML-G verified 10,000 training samples, 10,000/10,000 `fp_sol` contract validation, 44,884 extracted macro closures, 10,000 region heatmap examples, and 146 Step7 candidate-quality examples.

Important label distinction:

- FloorSet training labels teach general layout / macro closure / heatmap priors.
- Step7 sidecar artifacts teach candidate quality labels such as gate pass, dominated, metric-regressing, official-like improving.
- Do not collapse these into one ambiguous target.

## Current evidence and bottleneck

Please use the attached decision files as source evidence, but here is the short status:

1. Step7G spatial locality routing works as a safety layer: it classifies local / regional / macro / global moves and prevents global moves from being sent to bounded-local repair. It does not generate useful candidates by itself.
2. Step7H / Step7C-thin / Step7C-real-A..F show that local slack-fit and HPWL-directed edits are safe and can improve a few rows, but local iteration quickly starves. Step7C-local-iter0 selected only 2 edits across 8 cases.
3. Step7I / Step7I-R show coarse region/topology signal exists, but raw regional edits and simple slot legalizers are not enough; macro/MIB/group closure and internal repacking are major failure modes.
4. Step7N-G / Step7N-I show quality filtering is useful: constrained Pareto filters preserve winners and non-anchor Pareto rows while dropping many dominated/regressing rows. But selectors do not create new winners.
5. Step7ML-G..K built training data mart and geometry-aware decoders. Geometry-aware deterministic decoders reduce overlap and improve feasibility, but quality-gate / official-like winner counts stay narrow.
6. Later worktrees, especially Step7ML-P, showed full-case obstacle-aware repacking is the right legality architecture:
   - full-case overlap after splice: 28 -> 0
   - fresh official-like improving: 3
   - fresh quality gate pass: 3
   - dominated by original: 37 -> 6
   - metric-regressing: 37 -> 6
   But winners are concentrated in case025.
7. Step7R representation/staged-placement sidecars added diversity but did not beat the Step7ML-P winner/gate baseline; legal non-case candidates still regressed HPWL/cost.
8. Step7S soft-boundary/headroom work found a real but narrow signal. Step7S-W produced 25 fresh improving rows but only 1 unique improving signature and no case025-outside winners, so it preserves known boundary signal rather than solving generalization.

Current blocker:

```text
Not overlap, replay, or selector quality.
The bottleneck is target direction / topology-demand / objective alignment.
Full-case legality is increasingly possible, but non-case candidates tend to be HPWL/cost regressing.
```

## What I want from you

Please redesign the next feasible experimental program around RL and/or GNN, informed by relevant floorplanning and placement literature. Do not just suggest more threshold tuning. I need a detailed plan that a coding agent can execute in this repo.

Please consider and reference, at least conceptually, relevant ideas from:

- analytical placement / global placement -> legalization -> detailed placement, e.g. ePlace, RePlAce, DREAMPlace;
- graph or hypergraph neural placement / transferable macro placement, e.g. Circuit Training / AlphaChip-style RL, ChiPFormer-style offline decision transformer, MaskPlace-style visual/action masks, GraphPlace-style GNN placement if relevant;
- wirelength/density/target-field methods, WireMask-style net-demand maps, and offline RL / behavior cloning from fixed floorplanning data;
- hierarchical RL or GNN action policies for selecting target windows, blocks, macro closures, and legal move families.

If you are uncertain about exact paper names/details, say so rather than inventing specifics. The plan should be literature-grounded but still constrained by the attached repo reality.

## Required output format

Return a structured answer in Traditional Chinese with technical English terms preserved where useful.

### 1. Root-cause diagnosis

Explain why the current Step7/Step7ML/Step7S path is stuck. Explicitly separate:

- legality / overlap;
- target direction / topology-demand;
- candidate generation;
- selector / quality gate;
- training-data use;
- official-like metric alignment.

### 2. Architecture options

Propose 2-4 candidate system architectures, for example:

- supervised GNN target/heatmap prior + full-case obstacle-aware repacker;
- masked action policy with behavior cloning from FloorSet `fp_sol` / pseudo-trajectories;
- offline RL / decision transformer over sidecar replay trajectories;
- hierarchical policy: global target field -> closure/window action -> deterministic legalizer -> Pareto gate.

For each option include: expected benefit, risk, required data, implementation cost, and why it addresses target-direction/generalization.

### 3. Recommended architecture

Pick one primary architecture to implement first. Define the full system:

- input representation: block features, net/hypergraph features, pin/external terminal features, constraints, current placement, occupancy/free-space maps, route/locality maps;
- model components: GNN / hypergraph encoder, target heatmap head, action/mask head, candidate/ranker head, uncertainty or confidence if useful;
- action space: what the model predicts (region, block/closure, target window, move family, order, or target field), and what stays deterministic;
- legalizer/repacker interface: how predictions feed Step7ML-P-style full-case obstacle-aware repacker or equivalent;
- training targets: from FloorSet training `fp_sol`, Step7 sidecar candidate labels, pseudo-trajectories, or generated teacher labels;
- losses/objectives: supervised losses, pairwise/ranking losses, offline RL reward definitions, mask/legality losses, calibration losses;
- inference loop: how to generate K candidates, route, legalize, replay, and select;
- anti-leak boundaries: what not to use at inference and how to avoid target-position leakage.

### 4. Detailed phased experiment plan

Give a practical sequence with small gates. Each phase must include:

- goal;
- files/modules/scripts/tests likely to add or edit in this repo;
- data artifacts to write under `artifacts/research/`;
- smoke commands using `PYTHONPATH=src .venv/bin/python`;
- metrics to report;
- pass/fail decision and next branch.

Please include at least:

- Phase 0: data/interface audit and artifact normalization;
- Phase 1: supervised target / heatmap or topology-demand prior baseline;
- Phase 2: full-case candidate generation through existing repacker and gate;
- Phase 3: GNN or masked action policy training;
- Phase 4: offline RL / decision-transformer or policy improvement only if justified;
- Phase 5: integration and ablation against Step7ML-P / Step7S-W baselines.

### 5. First executable slice

The most important part: propose the first 1-3 executable tasks that I should implement immediately. They should be narrow, testable, and not require long GPU training. Prefer CPU-friendly or small-sample validation first. Provide exact expected files, tests, commands, and artifacts.

### 6. Evaluation design

Define a robust evaluation protocol that prevents self-deception:

- visible validation split vs training split;
- cross-case generalization, not just case025;
- fresh full-case official-like replay;
- unique-signature accounting;
- HPWL/bbox/soft tradeoff accounting;
- runtime accounting;
- negative-result reporting;
- comparison against Step7ML-P and Step7S-W.

### 7. Risks and kill criteria

List likely failure modes and explicit kill criteria. Examples: case-specific overfit, proxy-target mismatch, legalizer mismatch, target heatmap learns `fp_sol` but does not produce useful moves, offline RL reward hacking, selector leakage.

### 8. Implementation notes for the coding agent

Be concrete about how to fit into the current repo layout. Name likely modules under `src/puzzleplace/`, scripts under `scripts/`, docs under `docs/`, tests under `tests/`, and artifacts under `artifacts/research/`. Mention which existing code should be reused and which worktrees/artifacts should only be referenced/cherry-picked.

## Attachments to inspect

The attached files include:

- `AGENT.md`, `README.md`, `GUIDE.md`, `MEMORY.md`, `pyproject.toml` for project state and constraints.
- Mainline decision artifacts through Step7ML-K.
- Worktree decision artifacts for Step7ML-P, Step7R-D, and Step7S-T/U/V/W.
- Representative data/eval/model/search modules to understand current code layout and integration points.

Please answer as an actionable research engineering plan, not a generic survey.
