# 06 — Repair Finalizer

## Purpose

Repair is not fallback packing.
It is the intent-preserving step that turns a provisional semantic / relaxed layout into a hard-feasible layout with minimal displacement.

## Required previous steps

- Semantic or relaxed rollout must already produce a provisional layout.
- Constraint metadata must be available for each case.
- The system must be able to measure overlap, area, dimension, and intent preservation before and after repair.

## Pipeline role

```text
semantic / relaxed rollout -> repair / finalizer -> strict hard-feasible layout -> official evaluation
```

The repair layer should:

- keep policy intent visible;
- reduce overlap and other infeasibilities;
- report what moved and why;
- only use shelf-style fallback when local repair cannot finish.

## Key data / contracts

### Repair input

- `FloorSetCase`
- provisional positions
- semantic action history
- constraint metadata
- role labels

### Repair output

- repaired positions
- `RepairReport`

### RepairReport fields

```text
hard_feasible_before
hard_feasible_after
overlap_pairs_before
overlap_pairs_after
total_overlap_area_before
total_overlap_area_after
area_violations_before
area_violations_after
dimension_violations_before
dimension_violations_after
moved_block_count
mean_displacement
max_displacement
preserved_x_order_fraction
shelf_fallback_count
```

## Smoke commands

These are the pivot smoke targets from `AGENT_step2.md`; they describe the intended repair/finalizer check once the stage script is wired in.

### Pivot smoke target

```bash
python scripts/repair_validate.py \
  --case-ids 0 1 2 3 4 \
  --output artifacts/reports/sprint2_pivot_repair_validate.json
```

## Expected outputs

- Overlap and area violations decrease after repair.
- At least one case becomes hard-feasible after repair.
- The report explains how much the layout moved.

## Key metrics to inspect

- `overlap_pairs_before` / `overlap_pairs_after`
- `total_overlap_area_before` / `total_overlap_area_after`
- `area_violations_before` / `area_violations_after`
- `dimension_violations_before` / `dimension_violations_after`
- `mean_displacement`
- `max_displacement`
- `preserved_x_order_fraction`
- `shelf_fallback_count`

## Common failure modes

- Repair reduces neither overlap count nor overlap area.
- Shelf fallback moves almost everything and destroys intent.
- Preplaced or fixed blocks are moved during normalization.
- Intent preservation collapses because the finalizer behaves like silent packing.
- The layout is still infeasible after repair because shape normalization failed.

## How to debug

1. Compare the before / after overlap pairs first.
2. Check whether anchors are locked before the overlap resolver runs.
3. Inspect whether local shifts create new large overlaps.
4. Review the blocks sent to shelf fallback.
5. Confirm that intent-preservation counters are not all zero.

## Next step

If repair yields a hard-feasible layout, use it in `docs/07_strict_evaluation.md`.
