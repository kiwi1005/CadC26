# FloorSet Puzzle Bootstrap

This repository currently contains the **bootstrap scaffold** for a local-only FloorSet puzzle workflow.
The upstream FloorSet checkout lives under `external/FloorSet/`, stays untracked, and should not be modified in place.

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
