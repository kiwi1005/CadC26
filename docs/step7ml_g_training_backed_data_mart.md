# Step7ML-G Training-Backed Macro Layout Data Mart

Step7ML-G is a sidecar-only data mart expansion. It uses the official
FloorSet-Lite training loader validated by Step7DATA and keeps training layout
labels separate from Step7 sidecar candidate-quality labels.

## Label families

- `layout_prior_examples`: FloorSet training `fp_sol [w,h,x,y]` closure layouts
  for supervised macro/internal-placement priors.
- `region_heatmap_examples`: coarse 4x4 block-region distributions and masks
  derived from FloorSet training placements.
- `candidate_quality_examples`: Step7N-I candidate rows with gate, archive,
  feasibility, route, dominance, and metric-regression labels.

The data mart does not collapse these into one ambiguous target: FloorSet
training data teaches general layout priors; Step7 artifacts teach sidecar
candidate quality.

## Inputs

- Official FloorSet root with `floorset_lite/worker_*` training data.
- `artifacts/research/step7n_i_annotated_candidates.json`
- `artifacts/research/step7n_i_quality_filtered_candidates.json`
- `artifacts/research/step7n_i_archive_replay.json`

## Outputs

- `step7ml_g_data_inventory.json`
- `step7ml_g_layout_prior_examples.json`
- `step7ml_g_region_heatmap_examples.json`
- `step7ml_g_candidate_quality_examples.json`
- `step7ml_g_schema_report.json`
- `step7ml_g_split_report.json`
- `step7ml_g_missing_field_report.json`
- `step7ml_g_decision.md`

## Run

```bash
PYTHONPATH=src /home/hwchen/PROJ/CadC26/.venv/bin/python \
  scripts/step7ml_g_build_training_backed_data_mart.py \
  --training-samples 10000 --batch-size 64
```

Use `--training-samples 100000` for the next scale-up after the 10k gate passes.
Downloaded FloorSet data remains local under `external/FloorSet/floorset_lite/`
and must not be committed.
