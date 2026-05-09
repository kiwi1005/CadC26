# Step7N-ALR Plan: Archive-Lineage + Reservoir Region Opportunities

## Current diagnosis
Step7M reduced Step7L's universal regression but still produced no meaningful official-like winners. The surviving deterministic source collapsed to micro-axis perturbations and paired/block shifts. Earlier Step7N-G/H/I artifacts show sparse official-like winners and useful filters, but they are not normalized as a corpus. Step7N-ALR therefore starts with lineage mining, not another generator.

Current prior: existing Step7N-G/I decision artifacts indicate winners are case024-centered and sparse. Phase0 is expected to be diagnostic unless it finds additional exact-comparable, non-micro, cross-case strict winners.

## System architecture

```text
prior Step7 artifacts
  -> Phase0 lineage normalizer
  -> strict/non-micro/archive signal decision
  -> Phase1 reservoir atlas
  -> Phase2 non-micro request deck
  -> Phase3 exact replay
  -> Phase4 ablation/archive selector
  -> later-only model-readiness review
```

### Phase0 lineage schema
Each normalized row should include:
- `source_artifact`, `source_branch`, `source_phase`, `source_candidate_id`
- `case_id`, `candidate_id`
- `moved_block_count`, `moved_block_ids`, `locality_class`, `move_taxonomy`
- `hard_feasible`, `non_original_non_noop`, `fresh_metric_available`
- `hpwl_delta`, `bbox_area_delta`, `soft_constraint_delta`, `official_like_cost_delta`
- `all_vector_nonregressing`
- `meaningful_improving` using `MEANINGFUL_COST_EPS = 1e-7`
- `strict_archive_candidate`
- `metric_confidence`: `exact_replay_comparable`, `exact_replay_partial`, `proxy_or_sweep_only`, `summary_only`, or `unknown`
- `dominance_status`, `failure_attribution`, `quality_filter_reason`
- `forbidden_label_term_count`, `validation_label_policy`

Phase0 must be manifest-driven using `.omx/plans/step7n-phase0-source-manifest.txt`. It must not discover extra artifacts by broad glob. Self-outputs (`step7n_phase0_*`), completion reports, summary-only files, request decks without actual replay vectors, and proxy/sweep-only rows cannot create strict winners.

The Phase0 summary must explicitly report `case024_share`, `case025_share`, `largest_case_id`, and `largest_case_share` so the plan can distinguish the current case024-centered prior from the historical case025 concentration risk.

### Move taxonomy
Use deterministic taxonomy, not model inference:
- `micro_axis`: moved distance below configured threshold or Step7M micro-axis source.
- `single_block_region`: one block moved into a different non-micro region/window.
- `multi_block_shift`: paired/block-shift or corridor multiblock move.
- `macro_slot_repack`: Step7N slot/repack lineage.
- `window_repack`: local/regional window repack lineage.
- `global_route`: broad/global move; report-only unless exact vector-safe.
- `anchor/noop`: excluded from winner counts.

Non-micro classification must be scale-aware and reported with threshold sensitivity at `1e-4`, `1e-3`, and `max(1e-2, 0.5% case diagonal)`.

### Reservoir families
Only after Phase0 passes:
- `empty_rect`: exact/grid-assisted maximal empty rectangles.
- `boundary_slot`: legal windows near chip/boundary contact.
- `cluster_window`: local windows around known lineage winners and feasible non-winners.
- `bbox_hull`: windows that can shrink or preserve affected bbox hull.
- `net_slack`: whitespace near low-risk net-cut/slack proxy regions.

### Future model architecture, gated
If the GNN/RL gate eventually opens, the first model should be a selector/ranker:
- Heterogeneous graph:
  - block nodes with geometry, soft-role, current position, candidate displacement;
  - net/terminal nodes with incidence and HPWL contribution;
  - reservoir/window nodes with fit and clearance features;
  - candidate/action nodes with lineage provenance and proxy vector features.
- Edges:
  - block-net incidence;
  - block-reservoir fit;
  - geometric adjacency / overlap-risk;
  - candidate-source lineage;
  - candidate-objective proxy links.
- Model:
  - GNN or graph transformer encoder;
  - pairwise/listwise ranking head over existing candidate requests;
  - optional contextual-bandit policy after offline precision is proven.
- Forbidden first models:
  - direct coordinate generator;
  - RL environment trained on mostly regressing rows;
  - scalar reward without component-wise safety reports.

## Literature-backed design choices
- Placement RL can use graph representations and transfer, but it depends on reliable reward/evaluation and substantial training experience; therefore Step7N collects replay truth first. See Mirhoseini et al. Nature graph placement and the related arXiv placement RL papers.
- ML for graph combinatorial optimization often learns to select greedy actions over embedded graph state; this supports candidate ranking over generated legal actions rather than direct coordinate synthesis.
- Attention-based routing/constructive heuristics reinforce the same pattern: generate a feasible action set, then learn/rank selections.
- DREAMPlace-style differentiable placement motivates keeping exact objective/replay evaluation separate from speculative candidate generation.

References:
- `https://www.nature.com/articles/s41586-021-03544-w`
- `https://arxiv.org/abs/2004.10746`
- `https://arxiv.org/abs/2003.08445`
- `https://arxiv.org/abs/1704.01665`
- `https://arxiv.org/abs/1803.08475`
- `https://research.nvidia.com/publication/2019-06_dreamplace-deep-learning-toolkit-enabled-gpu-acceleration-modern-vlsi-placement`

## Execution order
1. Implement Phase0 only.
2. Run Phase0 smoke + tests + static checks.
3. Read `step7n_phase0_lineage_summary.json`.
4. If Phase0 says `stop_no_archive_signal`, stop and report.
5. If Phase0 says `diagnostic_only_due_concentration`, build at most a report-only atlas and do not generate requests.
6. If Phase0 says `promote_to_reservoir_atlas`, implement Phase1 atlas.
7. Continue gates strictly; no phase auto-opens itself.

Fail-closed rule: every downstream script must read the previous phase summary and refuse to run unless the prior phase explicitly promoted. Diagnostic/report-only mode must be named in output summaries and must not emit request decks.

## Expected bottlenecks
- Existing winners may be concentrated in case024 or another single case.
- Some older artifacts may not expose complete objective-vector fields.
- "Official-like" deltas may not be comparable across all older phases; the lineage miner must record metric confidence.
- Reservoir detection can inflate candidate count without objective safety; Phase2 must keep strict vector gates.

## Final success condition
Step7N-ALR succeeds only if it either:
- proves a cross-case non-micro reservoir opportunity source and emits strict replay winners; or
- terminates with a clear archive-grounded reason that prevents another low-signal widening loop.
