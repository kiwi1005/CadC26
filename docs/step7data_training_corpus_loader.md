# Step7DATA FloorSet-Lite Training Corpus Download and Loader Validation

Step7DATA is a sidecar-only data access gate for Step7ML.  It exists because
Step7ML-F proved the learning branches were using sparse Step7 artifact labels
and local validation cases, not the official FloorSet-Lite training corpus.

## Scope

- Use the official FloorSet `get_training_dataloader` / `FloorplanDatasetLite`
  path.
- Treat `external/FloorSet` source code as a read-only upstream dependency.
- Data download/extraction may write the official dataset payload under the
  FloorSet root (`floorset_lite/worker_*`), which is ignored/local-only.
- Do not train or promote any generator until this loader gate passes.

## Artifacts

- `step7data_inventory.json`: local FloorSet data availability.
- `step7data_loader_smoke.json`: first-batch shape contract and loader result.
- `step7data_training_samples.json`: bounded JSON summaries of sampled training
  layouts.
- `step7data_macro_labels.json`: MIB/cluster closure labels extracted from
  `fp_sol` `[w, h, x, y]` training labels.
- `step7data_decision.md`: decision memo.

## Decisions

- `training_loader_validated`: official training loader produced samples and
  macro labels.
- `download_required`: no local `floorset_lite/worker_*` training corpus and no
  download was requested.
- `download_attempt_failed`: auto-download path was attempted but failed.
- `loader_valid_but_macro_labels_sparse`: training loader works, but sampled
  macro labels are too sparse for macro-learning.
- `inconclusive_due_to_loader_error`: loader failed for a non-download reason.

## Commands

Dry inventory / no download:

```bash
PYTHONPATH=src .venv/bin/python scripts/step7data_validate_training_loader.py
```

Official auto-download and 1k sample smoke:

```bash
PYTHONPATH=src .venv/bin/python scripts/step7data_validate_training_loader.py \
  --auto-download --num-samples 1000 --max-examples 1000
```
