# Oracle Review Request: CadC26 Step7M-OAC Completion and Next Plan

You are reviewing `/home/hwchen/PROJ/CadC26`, a Python sidecar research repo for ICCAD 2026 FloorSet-Lite floorplanning experiments. Please act as an external research/architecture reviewer. The user asked for a concrete next plan after a sequence of deterministic Step7M experiments.

## Project constraints

- Use zh-TW or concise technical English; final plan should be operational.
- This is a sidecar research lane. Do **not** propose contest runtime/finalizer edits unless the evidence justifies a later integration gate.
- Official FloorSet checkout under `external/FloorSet/` is truth/evaluator source only; do not mutate it.
- Local commands use `PYTHONPATH=src .venv/bin/python`.
- Existing verification stack: `ruff`, `mypy`, focused `pytest`, artifact summaries under `artifacts/research/`.
- Avoid scalar penalty soup. Prior successful diagnostics use explicit vector gates for HPWL / bbox / soft constraints / overlap.
- GNN/RL should remain closed unless the evidence supports a broad, non-micro, cross-case corpus.

## Background: why Step7M was created

Earlier Step7L learning/topology heatmap work completed with a negative result:

- It produced hard-feasible non-noop candidates after sidecar replay.
- But fresh replay showed `0` improving candidates and universal objective regressions: HPWL, bbox, and soft all regressed for the fresh candidates.
- Diagnosis: target direction/objective alignment is the bottleneck, not legality or replay plumbing.

A previous Oracle review recommended `Step7M-OAC: Objective-Aligned Corridors`:

```text
old Step7L: topology/terminal heatmap -> target -> legalize/replay
new Step7M: objective corridor gate -> safe target mask -> heatmap only as tie-break/report -> replay/gate
```

## What I implemented after that advice

I implemented Step7M as a sidecar-only lane and verified each phase. Files/artifacts are attached.

### Phase 0: opportunity atlas

Purpose: inspect movable blocks, terminal/wire proxy, bbox hull role, soft roles, free-slot proxy, and per-case distribution.

Key result:

- `row_count`: 632
- `movable_row_count`: 550
- `case_count`: 8
- decision: `promote_to_objective_corridor_requests`
- forbidden validation-label terms: 0

### Phase 1: objective-corridor requests

Purpose: emit only candidate target windows accepted by vector gates before replay.

Key result:

- `request_count`: 72
- `unique_request_signature_count`: 72
- `represented_case_count`: 8
- `wire_safe`: 49
- `soft_repair_budgeted`: 23
- `case025_request_share`: 0.069
- `soft_budgeted_request_share`: 0.319
- predicted `wire_safe` regressions: HPWL 0, bbox 0, soft 0
- accepted requests were effectively `micro_axis_corridor`; heatmap support count was 0.

### Phase 2: exact objective-guarded replay

Purpose: replay the 72 exact target windows with official-like metrics, compare proxy vs actual signs.

Important: I added a meaningful improvement threshold (`1e-7`) because many apparent improvements were numerical micro-deltas around `1e-8`.

Key result:

- `fresh_hard_feasible_nonnoop_count`: 72 / 72
- `actual_metric_regression_rate`: 0.3056
- `actual_all_vector_nonregressing_count`: 50 / 72
- `actual_bbox_regression_count`: 0
- `actual_hpwl_regression_count`: 12
- `actual_soft_regression_count`: 10
- micro official-like improving count: 23, but `meaningful_official_like_improving_count`: 0
- `fresh_quality_gate_pass_count`: 0
- proxy/actual sign precision: component 0.9537, all-component 0.8611
- decision: `promote_to_step7m_corridor_ablation`
- `gnn_rl_gate_open`: false

Interpretation: Step7M fixed the Step7L universal regression symptom, but only at micro-axis local perturbation scale and produced no meaningful official-like winner.

### Phase 3: corridor ablation

Purpose: determine which gate/source family caused remaining regressions.

Key result:

- all Phase2 regression rate: 0.306
- `hpwl_bbox_soft` strict subset regression rate: 0.167
- `wire_safe_gate` regression rate: 0.204
- `soft_budgeted_gate` regression rate: 0.522
- `meaningful_official_like_improving_total`: 0
- `heatmap_supported_request_count`: 0
- decision: `tighten_to_hpwl_bbox_soft_then_multiblock_v1`
- `gnn_rl_gate_open`: false

Interpretation: soft-budgeted HPWL budget is the bad branch; bbox was not causing regressions. A stricter HPWL/BBox/Soft gate improves regression rate but still has no meaningful winner.

### Phase 4: deterministic paired/block-shift corridors

Purpose: minimal deterministic widening: combine strict proxy-safe micro-axis moves into paired x/y block shifts; exclude soft-budgeted by default.

Key result:

- generated `40` multiblock requests from `49` strict candidates
- represented cases: 2 (`19`, `51`)
- all 40 replayed hard-feasible non-noop
- meaningful improvements: 0
- regression rate: 0.500
- HPWL regression: 0
- bbox regression: 0
- soft regression: 20
- actual all-vector nonregressing: 20
- decision: `complete_step7m_deterministic_multiblock_and_defer_gnn_rl`
- `gnn_rl_gate_open`: false

Interpretation: deterministic micro-axis widening made soft behavior worse and did not create meaningful winners. I stopped the deterministic widening lane.

## Current conclusion from local evidence

Step7M succeeded as a falsification/diagnostic lane:

1. Objective-vector gates matter: regression rate improved from Step7L's 100% to 30.6% in Phase2 and 16.7% in the strict subset.
2. However, the accepted source collapsed to tiny micro-axis perturbations and did not create meaningful official-like improvements.
3. Pairing those micro moves did not help; it increased soft regressions.
4. GNN/RL remains unjustified because there is no broad, non-micro, cross-case winner corpus.

## Attached context files

Please inspect the attached docs, summary artifacts, and Step7M implementation files. The JSONL row files are intentionally not attached unless needed; summaries contain the important evidence.

## Questions for you

Please provide a concrete post-Step7M plan.

1. Do you agree with the local stop decision for deterministic micro-axis / multiblock widening? If not, what evidence says to continue?
2. What is the next most plausible non-micro opportunity source? Examples might include:
   - archive/candidate lineage mining from earlier Step7S/Step7ML results,
   - soft-boundary opportunity source discovery,
   - region/window-level compaction with real free-space reservoirs,
   - pair/cluster moves derived from net cut or slack, not micro perturbation,
   - selector/ranker over pre-existing candidates rather than coordinate generation,
   - or a different architecture.
3. Should we start a new lane (`Step7N` or another name)? Define objectives, phases, artifacts, exact gates, and kill criteria.
4. Should GNN/RL remain closed? If you think it should reopen, define the minimum corpus metrics and the exact model target (ranker, policy, GNN encoder, RL environment) and why it avoids Step7L/M failures.
5. Give a detailed experimental plan that is executable in this repo: files/modules/scripts/tests/artifacts, smoke commands, and expected decision thresholds.
6. Identify which existing Step7M code should be preserved, deleted, or only used as diagnostic support.
7. Warn about likely traps, especially validation-label leakage, case025 overfitting, micro-delta false positives, soft-regression hiding behind HPWL gates, and scalar objective mixing.

Desired output format:

- Verdict: continue/stop Step7M and why.
- Recommended next lane name and hypothesis.
- Phase-by-phase plan with artifacts and pass/kill gates.
- Whether/when GNN/RL can reopen.
- Minimal implementation plan for the next coding session.
- Risks and falsification tests.
