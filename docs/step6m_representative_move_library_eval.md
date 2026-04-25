# Step6M Representative Benchmark and Legal Move Library Evaluation

## Scope

Step6M remains a research/safe sidecar. It does not integrate any move into
contest runtime. The goal is to collect move-effect data and decide which move
families deserve a later runtime plan.

## Suite

The default runner builds a two-layer validation suite:

- diagnostic: validation cases `0..19`
- holdout: validation cases `20..39`

The local validation slice currently covers:

```text
diagnostic: small bucket, block counts 21..40
holdout: medium bucket, block counts 41..60
```

Large/xl buckets are tracked in the schema but are not present in this local
40-case validation prefix.

## Move library

Implemented legal move families:

- `simple_compaction`
- `edge_aware_compaction`
- `soft_aspect_flip`
- `soft_shape_stretch`
- `mib_master_aspect_flip`
- `mib_master_edge_slot_shape`
- `group_template_rotate`
- `group_template_mirror`
- `group_boundary_touch_template`
- `local_region_repack`
- `boundary_edge_reassign`
- `cluster_split_or_two_lobe_repack`

Moves are generated only from attribution targets:

- unsatisfied boundary blocks
- final edge owners / hull stealers
- MIB/grouping role-conflict targets
- local region blocks from same MIB/group and top connected neighbors

Each move is evaluated as an alternative layout. The original runtime layout is
not overwritten.

## Command

```bash
PYTHONPATH=src .venv/bin/python scripts/run_step6m_move_library_eval.py \
  --case-limit 40 \
  --diagnostic-count 20 \
  --holdout-count 20 \
  --top-k-targets 5 \
  --top-m-moves 12 \
  --mode safe \
  --output artifacts/research/step6m_report.json
```

## Artifacts

- `artifacts/research/step6m_case_suite.json`
- `artifacts/research/step6m_case_profiles.json`
- `artifacts/research/step6m_move_library_eval.json`
- `artifacts/research/step6m_selected_alternatives.json`
- `artifacts/research/step6m_move_costs.json`
- `artifacts/research/step6m_profile_summary.md`
- `artifacts/research/step6m_report.json`

## Current result

Safe-mode run:

```text
cases = 40
moves_evaluated = 2080
hard_infeasible_selected = 0
frame_protrusion_selected = 0
diagnostic_mean_boundary_delta = +0.0619
holdout_mean_boundary_delta = +0.0738
```

Selected move counts:

```text
simple_compaction = 16
original = 15
boundary_edge_reassign = 7
group_boundary_touch_template = 2
```

Decision:

```text
promote_case_selective_compaction_research_next
```

Interpretation:

- `simple_compaction` is not globally safe, but it wins often enough when
  selected by the safe gate.
- `boundary_edge_reassign` is cheap and useful for several boundary failures.
- `edge_aware_compaction` frequently improves boundary metrics but is rejected
  in safe mode due bbox/HPWL side effects; it needs a tighter variant before
  runtime consideration.
- MIB/group macro moves produced data but are not yet stable enough to promote.
