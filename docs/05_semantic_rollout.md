# 05 — Semantic Rollout

## Purpose

Semantic rollout teaches the policy to express placement intent before the layout is fully legal.
It exists so early training can maximize recall, expose violation signals, and avoid stopping after the first overlap.

## Required previous steps

- Candidate generation must support `candidate_mode = semantic | relaxed | strict`.
- Action schema must already define the primitive / argument format.
- The policy should be able to score candidate actions, even when the layout is still provisional.

## Pipeline role

```text
candidate generation -> semantic rollout -> provisional layout -> violation profile -> repair/finalizer
```

Semantic rollout is the bridge between action selection and strict repair:

- it must try to place every block;
- it may allow overlaps and provisional bounding boxes;
- it must not terminate just because the state is infeasible;
- it must report the violation profile instead of hiding it.

## Key data / contracts

### Rollout contract

```text
rollout(case, model=None, mode="semantic" | "relaxed" | "strict")
```

### BoardState fields

```text
positions
proposed_positions
shape_assigned
semantic_placed
physically_placed
frozen
step
history
violation_profile
```

### Semantic invariants

- `semantic_placed=True` means the block has a proposed intent, not a hard-feasible committed layout.
- `frozen=True` means the block itself should not move, but it can still be an attach target.
- Type-level legality stays hard: invalid block ids, invalid target ids, or invalid primitive arguments are not allowed.
- Overlap is allowed during early semantic learning, but progress must still be recorded.

## Smoke commands

These are the pivot smoke targets from `AGENT_step2.md`; they are documented here as the intended rollout checks, not as verified current-baseline commands.

### Pivot smoke target

```bash
python scripts/rollout_validate.py \
  --split validation \
  --case-ids 0 1 2 3 4 \
  --rollout-mode semantic \
  --output artifacts/reports/sprint2_pivot_semantic_rollout.json
```

```bash
python scripts/rollout_validate.py \
  --split validation \
  --case-ids 0 1 2 3 4 \
  --rollout-mode relaxed \
  --output artifacts/reports/sprint2_pivot_relaxed_rollout.json
```

## Expected outputs

- A provisional layout for every case.
- A violation profile that names overlap, area, boundary, grouping, and MIB issues.
- Per-step progress showing how many blocks became semantically placed.

## Key metrics to inspect

- `semantic_rollout_completion_rate`
- `avg_semantic_placed_fraction`
- `overlap_pair_count`
- `total_overlap_area`
- `area_error_total`
- `boundary_distance_total`
- `group_fragmentation`
- `mib_shape_inconsistency`

## Common failure modes

- Rollout stops after 1–2 blocks.
- `SET_SHAPE` is never followed by an actual placement action.
- `FREEZE` is selected too early and blocks progress.
- A candidate exists but is masked out before the policy can score it.
- `executor` applies an action but `semantic_placed` is not updated.
- No-progress fallback loops on the same block.

## How to debug

1. Check whether `semantic_placed` increases every step.
2. Inspect `stopped_reason` and the last primitive chosen.
3. Compare candidate generation in `semantic` vs `strict` mode.
4. Verify that the violation profile is changing, not staying empty.
5. Confirm that frozen blocks are still available as attach targets.

## Next step

If semantic rollout completes, hand the provisional layout to `docs/06_repair_finalizer.md`.
