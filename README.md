# FloorSet Puzzle Research Workspace

This repository is a local FloorSet-Lite floorplanning research workspace for
ICCAD 2026.  It contains runnable Python package code, tests, contest smoke
entrypoints, and Step6/Step7 sidecar experiments.

The upstream FloorSet checkout lives under `external/FloorSet/`, stays
untracked, and should not be modified in place.

## Current project direction

The active architecture is the Step7 **DGLPR** loop:

```text
Diagnose -> Generate Alternatives -> Legalize -> Pareto Select -> Refine / Iterate
```

Step6 sidecars are preserved for traceability, but new research work should use
the Step7 directories:

- `src/puzzleplace/diagnostics/`
- `src/puzzleplace/alternatives/`
- `src/puzzleplace/legalization/`
- `src/puzzleplace/experiments/`
- `scripts/step7*.py`
- `docs/step7*.md`

As of 2026-05-08, the freshest active lane is **Step7P-CCR causal closure
repack**, opened after Step7L/M/N/O and Oracle agreed that the project was
stagnating because it kept ranking or perturbing a weak candidate universe rather
than creating a causal move/repack operator. Step7P Phase0-2 passed, and the
Phase3 request-source coverage repair now covers all 8 representative cases.
However, bounded replay fails the Phase3 replay gate with 0 strict meaningful
winners. A three-branch redesign found `branch_c_balanced_failure_budget` as a
partial improvement that reduces overlap, soft regression, and bbox regression,
but Phase4 remains closed. A constrained Step7Q RL/GNN operator-learning phase
is now open via `artifacts/research/step7p_gnn_rl_readiness_summary.json`; do
not run Phase4 as a real ablation or integrate runtime/finalizer changes until a
learned operator deck opens `phase4_gate_open=true`.

## Environment

### 1. Clone the official FloorSet repo locally only

```bash
git clone https://github.com/IntelLabs/FloorSet.git external/FloorSet
```

`external/FloorSet/` is ignored by git on purpose so upstream source and downloaded datasets do not get vendored into this repository.

### 2. Create and activate a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
```

### 3. Install contest + local development dependencies

```bash
pip install -r external/FloorSet/iccad2026contest/requirements.txt
pip install -e .[dev]
```

If this checkout already has `.venv` configured, prefer using it instead of
creating a second environment:

```bash
source .venv/bin/activate
python -m pip install -e .[dev]
```

## Smoke download / loader check

Run the bootstrap smoke script:

```bash
python scripts/download_smoke.py
```

What it does:
- imports the official `get_validation_dataloader()` and `get_training_dataloader()` helpers
- auto-approves the upstream dataset download prompt for non-interactive runs
- loads the first validation batch and prints field names, tensor shapes, and inferred block count
- loads a training dataloader with `batch_size=1, num_samples=10` and prints the first batch summary

### Data locations

The script targets the official checkout root at `external/FloorSet/`.
With the current upstream loader implementation, downloaded data is expected under:

- validation: `external/FloorSet/LiteTensorDataTest/`
- training: `external/FloorSet/floorset_lite/`

The contest README still describes `LiteTensorData/` as the training location, but the current upstream training loader checks `floorset_lite/`; downstream code should treat the loader implementation as the source of truth.

## Downstream environment contract

Consumers may assume the following bootstrap contract:

- official imports come from `external/FloorSet/iccad2026contest/`
- the repo root script entrypoint is `python scripts/download_smoke.py`
- the smoke script inserts both `external/FloorSet/` and `external/FloorSet/iccad2026contest/` onto `sys.path`
- validation batches follow the official structure:
  - inputs: `(area_target, b2b_connectivity, p2b_connectivity, pins_pos, placement_constraints)`
  - labels: `(polygons, metrics)`
- training batches follow the official structure:
  - `(area_target, b2b_connectivity, p2b_connectivity, pins_pos, placement_constraints, tree_sol, fp_sol, metrics)`
- all upstream datasets and artifacts remain local-only and untracked

## Current milestone scripts

Bootstrap / contest smoke:

- `python scripts/download_smoke.py`
- `python scripts/evaluate_contest_optimizer.py`
- `python scripts/run_smoke_regression_matrix.py`

Legacy training / rollout baseline:

- `python scripts/report_candidate_coverage.py`
- `python scripts/train_bc_small.py`
- `python scripts/rollout_validate.py`
- `python scripts/run_agent10_ablation.py`
- `python scripts/train_awbc_small.py`

Step6 frozen sidecars:

- `python scripts/run_step6g_puzzle_policy_audit.py`
- `python scripts/run_step6p_pareto_selector.py`
- `python scripts/visualize_step6p_pareto_results.py`

Step7 active sidecars:

- `python scripts/step7a_run_aspect_pathology.py`
- `python scripts/step7b_run_shape_policy_study.py`
- `python scripts/step7d_run_representative_replay.py`
- `python scripts/step7e_run_region_topology_diagnosis.py`
- `python scripts/step7f_run_bounded_repair_study.py`
- `python scripts/step7g_run_spatial_locality_routing.py`
- `python scripts/step7h_run_route_aware_candidate_diversification.py`
- `python scripts/step7c_thin_run_route_aware_layout_edit_loop.py`
- `python scripts/step7l_replay_heatmap_candidates.py`
- `python scripts/step7m_replay_corridor_requests.py`
- `python scripts/step7n_mine_archive_lineage.py`
- `python scripts/step7o_calibrate_training_demand_prior.py`
- `python scripts/step7p_lock_stagnation_state.py`
- `python scripts/step7p_build_causal_subproblem_atlas.py`
- `python scripts/step7p_smoke_causal_repacker.py`
- `python scripts/step7p_generate_causal_repack_requests.py`

Current Step7P stop point: request generation now reports
`phase3_replay_gate_open=true`, but bounded replay reports
`decision=stop_phase3_replay_gate` and `phase4_gate_open=false`. The ablation
runner writes a blocked Phase4 summary from this replay state; do not treat it as
a real family ablation.

## Contest entrypoint

- The repository-level submission file is `contest_optimizer.py`.
- It delegates to `src/puzzleplace/optimizer/contest.py`.
- If `artifacts/models/agent11_awbc_policy.pt` exists, the contest optimizer loads it.
- Otherwise it falls back to a deterministic heuristic policy so the validator/evaluator path remains runnable.

## Verification shortcuts

Use the repository virtual environment and `PYTHONPATH=src` for local checks:

```bash
PYTHONPATH=src .venv/bin/python -m pytest tests/test_step7g_spatial_locality.py
PYTHONPATH=src .venv/bin/python -m mypy src/puzzleplace/diagnostics/spatial_locality.py src/puzzleplace/alternatives/locality_routing.py
PYTHONPATH=src .venv/bin/python -m ruff check src/puzzleplace/diagnostics/spatial_locality.py src/puzzleplace/alternatives/locality_routing.py scripts/step7g_run_spatial_locality_routing.py tests/test_step7g_spatial_locality.py
```

Step7P gate repair checks:

```bash
PYTHONPATH=src .venv/bin/python scripts/step7p_generate_causal_repack_requests.py   --subproblem-atlas artifacts/research/step7p_phase1_causal_subproblem_atlas.jsonl   --operator-contract artifacts/research/step7p_phase2_operator_contract.json   --out artifacts/research/step7p_phase3_causal_requests.jsonl   --summary-out artifacts/research/step7p_phase3_causal_request_summary.json   --markdown-out artifacts/research/step7p_phase3_causal_request_summary.md
PYTHONPATH=src .venv/bin/python -m pytest tests/test_step7p_causal_request_contract.py -q
```


## 2026-05-08 Step7Q-A data mart

Step7Q-A non-leaky operator-learning data mart passed: `artifacts/research/step7q_operator_data_mart_summary.json` reports `decision=promote_to_constrained_risk_ranking`, `example_count=325`, `feature_label_leakage_count=0`, `strict_meaningful_positive_count=0`, and `allowed_next_phase=step7q_constrained_risk_ranking`. Next work is constrained risk/ranking, not direct coordinate RL or Phase4.

## 2026-05-08 Step7Q-B constrained risk/ranking smoke

Step7Q-B completed a feature/mask-only policy smoke. It first corrected the Step7Q-A forbidden-mask search so the Phase3 request `forbidden` policy declaration is not treated as actual forbidden operator usage; rebuilt data mart now keeps `feature_label_leakage_count=0`, `eligible_for_selection_count=120`, and `eligible_for_training_count=253`. The selected deck in `artifacts/research/step7q_operator_policy_summary.json` reports `decision=risk_ranking_smoke_pass_strict_gate_closed`, `selected_request_count=96`, `represented_case_count=8`, `largest_case_share=0.25`, `forbidden_request_term_count=0`, `overlap_after_splice_count=23`, `soft_regression_rate=0.052083333333333336`, `bbox_regression_rate=0.0`, and `strict_meaningful_winner_count=0`. Next work is constrained operator parameter expansion under this risk filter; Phase4 remains closed.

## 2026-05-08 Step7Q-C operator-parameter expansion

Step7Q-C produced a bounded finite-action expansion deck from the Step7Q-B low-risk sources. `artifacts/research/step7q_parameter_expansion_summary.json` reports `decision=parameter_expansion_deck_ready_for_fresh_replay`, `candidate_count=550`, `selected_expansion_count=96`, `represented_case_count=8`, `largest_case_share=0.25`, `unique_parent_source_count=69`, `unique_action_signature_count=8`, `forbidden_action_term_count=0`, and `direct_coordinate_field_count=0`. This is only replay-ready: `fresh_replay_required=true`, `strict_winner_evidence_count=0`, and `phase4_gate_open=false`. Next work is a fresh metric replay/executor for `artifacts/research/step7q_parameter_expansion_deck.jsonl`.

## 2026-05-08 Step7Q-D fresh metric replay

Step7Q-D executed the finite-action expansion deck through a sidecar fresh-metric bridge. `artifacts/research/step7q_fresh_metric_replay_summary.json` reports `decision=fresh_replay_executed_strict_gate_closed`, `request_count=96`, `fresh_metric_available_count=9`, `fresh_hard_feasible_nonnoop_count=9`, `overlap_after_splice_count=87`, `soft_regression_rate=0.020833333333333332`, `bbox_regression_rate=0.0`, `actual_all_vector_nonregressing_count=7`, and `strict_meaningful_winner_count=0`. Phase4 remains closed. The current bottleneck is geometry realization/obstacle-aware slotting, not the risk scorer.

## 2026-05-08 Step7Q-E slot-aware replay

Step7Q-E enabled `--slot-aware` fresh replay for the Step7Q parameter expansion deck. `artifacts/research/step7q_slot_aware_replay_summary.json` reports `fresh_metric_available_count=96`, `fresh_hard_feasible_nonnoop_count=96`, and `overlap_after_splice_count=0`, so the overlap realization bottleneck is fixed. However, the slot choices regress objectives badly: `soft_regression_rate=0.9270833333333334`, `bbox_regression_rate=0.90625`, `hpwl_regression_rate=0.90625`, `actual_all_vector_nonregressing_count=7`, `strict_meaningful_winner_count=0`, and `phase4_gate_open=false`. Next work is objective-aware slot scoring, not another legality-only slot search.

## 2026-05-08 Step7Q-F objective-aware slot replay

Step7Q-F added `--objective-aware-slot`, scoring feasible non-overlap slots with fresh objective deltas before selection. `artifacts/research/step7q_objective_slot_replay_summary.json` reports `fresh_metric_available_count=96`, `fresh_hard_feasible_nonnoop_count=96`, `overlap_after_splice_count=0`, `slot_adjusted_count=87`, `actual_all_vector_nonregressing_count=27`, and `strict_meaningful_winner_count=0`. It improves Step7Q-E (`all_vector` 7 -> 27, soft regression 0.9270833333333334 -> 0.71875, bbox/hpwl 0.90625 -> 0.6979166666666666) but `risk_replay_gate_open=false` and `phase4_gate_open=false`. Next work is improving the slot candidate pool itself.
