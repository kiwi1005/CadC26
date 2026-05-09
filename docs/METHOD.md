# METHOD.md — Multi-Stage Active-Soft Boundary Repair

## Problem formulation

The boundary repair problem is a constrained optimization:

```
min  cost(x)      cost = HPWL + α·bbox_area + β·soft_viol
s.t. h_i(x) = 0     (hard overlap / dimension constraints)
     x_j ∈ edge_j   (boundary constraints for j ∈ B)
```

The current single-stage snap (greedy per-block projection onto boundary edges)
makes a fatal assumption: moving one block does not significantly change the
bounding box or HPWL. This fails on large cases (e.g., case 79, 100 blocks)
where:

- Boundary-constrained blocks often define the bbox envelope edges.
- Snapping a block outward expands the bbox, and the bbox_area term in the
  official-like cost function penalizes expansion quadratically.
- Moving a highly-connected block causes cascading HPWL changes across many nets.

### Data evidence (case 79, 200 candidates)

| Metric | Value |
|---|---|
| Feasible candidates | 6 / 200 (3%) |
| Infeasibility: dimension violation | 137 / 194 |
| Largest snap (block 76) | `|dx| = 316` units |
| Consistent cost regression | all feasible candidates regress HPWL or cost |

In contrast, winning cases (19, 24, 25, 51, 76) have 47–108 feasible candidates
out of 200 and manageable snap distances.

## Mathematical analysis

### 1. HPWL sensitivity

For a net connecting blocks {i₁, ..., iₖ} with positions (x_i, y_i) and
dimensions (w_i, h_i) (using block centers c_i = (x_i + w_i/2, y_i + h_i/2)):

```
HPWL_net = (max c_x - min c_x) + (max c_y - min c_y)
```

Gradient (subgradient due to max/min):
```
∂HPWL/∂c_x[k] = +1  if block k is the unique rightmost
                -1  if block k is the unique leftmost
                 0  otherwise (with linear interpolation at ties)
```

Total HPWL sensitivity:
```
σ_x[k] = Σ_{net ∋ k} ∂HPWL_net/∂c_x[k]
σ_y[k] = Σ_{net ∋ k} ∂HPWL_net/∂c_y[k]
```

A boundary snap (dx, dy) on block k changes HPWL by approximately:
```
ΔHPWL ≈ σ_x[k]·dx + σ_y[k]·dy
```

When σ_x[k]·dx > 0 (moving with the gradient), HPWL increases. This is the
"HPWL regression" blocker observed on cases 19, 25, 79 in the POC.

### 2. Bounding-box expansion

The bbox of the layout: `B = (x_min, y_min, x_max, y_max)`. The bbox_area term
in official-like cost is proportional to `(x_max - x_min)·(y_max - y_min)`.

If block k defines the right edge (`x_k + w_k == x_max`), snapping it right by
m expands the bbox: `x_max ← x_max + m`. The cost increase is:
```
Δcost_bbox ≈ α·m·(y_max - y_min)
```

For large cases (100 blocks), bbox height can be 500–1000 units, so α·m·H is
large even for modest m. This is the likely source of the -63 cost delta on
case 79 (snaps to blocks that define bbox edges).

### 3. Linearized overlap constraint

Two blocks i, j overlap if:
```
|x_i - x_j| < (w_i + w_j)/2  AND  |y_i - y_j| < (h_i + h_j)/2
```

After snap (dx, dy) on block k, a previously non-overlapping pair (k, j) may
overlap. The minimum displacement to resolve:
```
δx_req = max(0, (w_k + w_j)/2 - |x_k + dx - x_j|)
δy_req = max(0, (h_k + h_j)/2 - |y_k + dy - y_j|)
```

This explains the 137 dimension violations on case 79: large snaps displace
blocks into neighbors.

## Multi-stage architecture

### Stage 1: HPWL-sensitive, bbox-preserving partial snap

**Goal**: Fix easy boundary violations without causing collateral damage.

For each boundary-constrained block j with violated edge e, margin m:

```
1. Compute σ[j] = HPWL sensitivity at current position.
2. Compute bbox_impact = 1 if block j defines edge e of the bbox, else 0.
3. If bbox_impact == 1:
     α_max = ε_bbox / m   (limit bbox expansion to ε_bbox)
   Else:
     α_max = 1.0
4. Optimal fraction:
     α* = argmin_{α ∈ [0, α_max]}  σ[j]·d(α) + γ·(1-α)·m
   where d(α) is the displacement vector at fraction α.
   This is a 1D convex problem solved by evaluating at candidate fractions.
5. If α* > 0 and snap is feasible and HPWL does not regress beyond τ:
     Accept snap at fraction α*.
     Update positions, bbox, sensitivities.
```

Output: partially-repaired positions, set of unresolved boundary violations.

Threshold `ε_bbox` ~ 0 (strict bbox preservation) or small positive value.
Threshold `τ` = MEANINGFUL_COST_EPS for HPWL regression tolerance.

### Stage 2: Joint feasibility-preserving boundary moves

**Goal**: Fix boundary violations that Stage 1 could not (because the required
snap would overlap a neighbor).

For each unresolved boundary block j, edge e, margin m:

```
1. Compute required snap d_j = (dx_j, dy_j) toward edge e.
2. Find all blocks O = {i : i would overlap with j at position j + d_j}.
3. For each obstructing block i ∈ O:
     Compute minimal push d_i that resolves the overlap:
       d_i = resolve_overlap(j + d_j, i)
     Create joint candidate: move j by d_j, move i by d_i.
     Evaluate feasibility.
4. If no single-obstruction resolution works:
     Try pair of boundary blocks simultaneously:
       For each other unresolved boundary block k:
         Try snapping both j and k together (reduces bbox expansion per block).
```

Output: joint-snapped positions.

### Stage 3: HPWL-gradient compensation

**Goal**: Recover HPWL lost in Stages 1–2 by adjusting non-boundary blocks.

```
1. Identify hot nets: nets where HPWL increased beyond threshold.
2. For each hot net n, for each non-boundary block k in n:
     Compute -∇HPWL_n(c_k) (the direction that reduces HPWL for this net).
     Try a small step (η·∇) and check legality + overall cost delta.
     Accept if HPWL improves without regressing other metrics.
3. Limit total compensation displacement per block to avoid oscillation.
```

Output: final positions.

### Stage fusion: shared FusionState

```python
FusionState {
    positions: list[Box]                    # current positions (updated incrementally)
    bbox: (x_min, y_min, x_max, y_max)      # current bbox (recomputed after each stage)
    before_eval: dict                        # cached evaluation at original positions
    hpwl_sensitivity: dict[int, (σ_x, σ_y)] # HPWL gradient per block (updated per stage)
    net_graph: dict[int, list[int]]          # net_id -> [block_ids] (static)
    block_nets: dict[int, list[int]]         # block_id -> [net_ids] (static)
    bbox_edge_owners: dict[str, int]         # which block defines each bbox edge
    boundary_margins: dict[int, dict[str, float]]  # per-block, per-edge margin
    stage_log: dict[str, Any]                # per-stage diagnostic log
    block_displacement: dict[int, float]     # cumulative displacement budget
}
```

The fusion state lets Stage 2 know which violations Stage 1 left unresolved,
and Stage 3 know exactly which nets were damaged by Stage 2's joint moves.

## Expected impact

| Stage | Case 79 expected effect |
|---|---|
| Stage 1 (bbox-preserving) | Eliminates the -63 cost regression; keeps bbox unchanged |
| Stage 1 (HPWL-sensitive) | Only snaps blocks with favorable/neutral HPWL gradient |
| Stage 2 (joint feasibility) | Resolves dimension violations by pushing obstructing blocks |
| Stage 3 (compensation) | Recovers HPWL on nets affected by successful snaps |

The three-stage pipeline targets the root cause: boundary repairs should never
expand the bounding box, should account for local HPWL gradient, and should
compensate neighboring blocks when one block's move causes collateral damage.

## References

- Step7T POC: `docs/step7t_active_soft_cone.md`
- Step7V live adapter: `artifacts/research/step7v_live_active_soft_parallel_summary.md`
- Current postprocessor: `src/puzzleplace/repair/active_soft_postprocess.py`
- Candidate generator: `src/puzzleplace/experiments/step7t_active_soft_cone.py`
