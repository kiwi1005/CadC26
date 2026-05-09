# Step7M-OAC: Objective-Aligned Corridors

Step7M-OAC starts after Step7L completed with a negative but useful result:
heatmap/topology requests were often hard-feasible after sidecar replay, but all
fresh candidates regressed official-like objective components. Step7M therefore
puts vector objective corridors before target generation.

## Contract

- Sidecar-only; no contest runtime/finalizer changes.
- Validation geometry is used only as sidecar anchor/proxy/replay state, not as
  model-training labels.
- Do not combine HPWL/bbox/soft into one scalar penalty. Use explicit vector
  gates and report every rejection class.
- Step7L heatmaps may seed or tie-break candidate centers, but cannot bypass
  HPWL/bbox/soft gates.
- Micro-axis probes are allowed only to detect local piecewise-linear HPWL/bbox
  plateaus; they still need explicit vector-gate acceptance.

## Phase 0: opportunity atlas

Artifacts:

- `artifacts/research/step7m_phase0_opportunity_atlas.jsonl`
- `artifacts/research/step7m_phase0_opportunity_summary.json`
- `artifacts/research/step7m_phase0_opportunity_summary.md`

Smoke:

```bash
PYTHONPATH=src .venv/bin/python scripts/step7m_build_opportunity_atlas.py \
  --validation-cases 19,24,25,51,76,79,91,99 \
  --grid 16
```

## Phase 1: objective-corridor requests

Artifacts:

- `artifacts/research/step7m_phase1_corridor_requests.jsonl`
- `artifacts/research/step7m_phase1_corridor_request_summary.json`
- `artifacts/research/step7m_phase1_corridor_request_summary.md`

Smoke:

```bash
PYTHONPATH=src .venv/bin/python scripts/step7m_generate_corridor_requests.py \
  --validation-cases 19,24,25,51,76,79,91,99 \
  --grid 16 \
  --candidate-cells-per-block 24 \
  --max-blocks-per-case 16 \
  --windows-per-block 8 \
  --families wire_safe,bbox_shrink_wire_safe,soft_repair_budgeted
```

Promotion to replay requires `unique_request_signature_count >= 64`,
`represented_case_count >= 6`, `case025_request_share <= 0.35`,
`soft_budgeted_request_share <= 0.50`, no forbidden validation-label terms,
and zero predicted HPWL/bbox/soft regressions in the `wire_safe` family.


## Phase 2: objective-guarded replay

Artifacts:

- `artifacts/research/step7m_phase2_replay_rows.jsonl`
- `artifacts/research/step7m_phase2_replay_summary.json`
- `artifacts/research/step7m_phase2_replay_summary.md`
- `artifacts/research/step7m_phase2_failures_by_case.json`

Smoke:

```bash
PYTHONPATH=src .venv/bin/python scripts/step7m_replay_corridor_requests.py \
  --requests artifacts/research/step7m_phase1_corridor_requests.jsonl \
  --replay-rows-out artifacts/research/step7m_phase2_replay_rows.jsonl \
  --out artifacts/research/step7m_phase2_replay_summary.json \
  --failures-out artifacts/research/step7m_phase2_failures_by_case.json
```

Promotion to ablation requires fresh hard-feasible non-noop candidates and an
actual metric regression rate below Step7L's 100% regression baseline. GNN/RL
remains closed unless a much broader replay corpus produces cross-case winners.


## Phase 3: corridor ablation

Artifacts:

- `artifacts/research/step7m_phase3_ablation_rows.jsonl`
- `artifacts/research/step7m_phase3_ablation_summary.json`
- `artifacts/research/step7m_phase3_ablation_summary.md`

Smoke:

```bash
PYTHONPATH=src .venv/bin/python scripts/step7m_run_corridor_ablation.py \
  --replay-rows artifacts/research/step7m_phase2_replay_rows.jsonl \
  --rows-out artifacts/research/step7m_phase3_ablation_rows.jsonl \
  --out artifacts/research/step7m_phase3_ablation_summary.json
```

Phase3 compares explicit gate subsets and source tags over the replayed exact
targets. It is an analysis gate, not a GNN/RL training step.


## Phase 4: deterministic paired/block-shift corridors

Artifacts:

- `artifacts/research/step7m_phase4_multiblock_requests.jsonl`
- `artifacts/research/step7m_phase4_multiblock_request_summary.json`
- `artifacts/research/step7m_phase4_multiblock_replay_rows.jsonl`
- `artifacts/research/step7m_phase4_multiblock_replay_summary.json`
- `artifacts/research/step7m_phase4_multiblock_failures_by_case.json`

Smoke:

```bash
PYTHONPATH=src .venv/bin/python scripts/step7m_generate_multiblock_corridors.py \
  --phase1-requests artifacts/research/step7m_phase1_corridor_requests.jsonl \
  --opportunity-atlas artifacts/research/step7m_phase0_opportunity_atlas.jsonl \
  --max-pairs-per-case 24
PYTHONPATH=src .venv/bin/python scripts/step7m_replay_multiblock_corridors.py \
  --requests artifacts/research/step7m_phase4_multiblock_requests.jsonl
```

Phase4 keeps GNN/RL closed and tests only strict proxy-safe paired micro-axis
block shifts. The default excludes `soft_repair_budgeted` because Phase3 showed
that branch caused the highest regression rate.
