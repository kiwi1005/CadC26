# HCFP-5090 P3.3 Contact-Aware B*-Tree Search Results

## Hypothesis

Small deterministic topology changes can improve a usable B*-Tree candidate without returning to free coordinate optimization.

## Experiment A: contact-aware vertical order

For each fixed tree/shape/outline candidate, three y-order variants were tested:

- bottom/top boundary band ordering;
- grouping members clustered in the order;
- combined boundary and group ordering.

All candidates are decoded by `pack_x_compacted`, then passed through the existing exact tail. The original tree/order remains in the pool.

On the same 16 large samples used by P3.2:

| Metric | Topology-only | Contact-aware |
| --- | ---: | ---: |
| Weighted cost | 6.569232 | 6.411334 |
| Contact-order wins | - | 10/16 |
| Hard-feasible oracle | 16/16 | 16/16 |

The boundary-band order generated most wins. Global group clustering was sometimes useful but often paid too much HPWL/bbox cost, so it should remain a bounded candidate rather than a mandatory transform.

## Experiment B: bounded local tree moves

Only two always-valid moves were added:

- swap the two children of a parent;
- detach a group leaf and reinsert it into an empty branch of another group member.

At most eight neighbors were generated per base tree. On eight large samples:

| Metric | Topology-only | Local tree oracle |
| --- | ---: | ---: |
| Weighted cost | 6.374376 | 6.266845 |
| Local-tree wins | - | 5/8 |
| Hard-feasible oracle | 8/8 | 8/8 |

The observed winners were sibling flips. The group-leaf move is retained as an experimental candidate but has not yet demonstrated a unique win.

## Decision

**KEEP** both bounded candidate mechanisms. Do not cross-product all variants in the final runtime: contact-order and local-tree candidates must be routed/pruned after a runtime topology predictor exists. The next required experiment is a neural/runtime B*-Tree constructor trained from `tree_sol`; gold topology is not available during official inference.

Artifacts:

- `artifacts/benchmarks/hcfp5090-p33-contact-order-large16.json`
- `artifacts/benchmarks/hcfp5090-p33-local-tree-large8.json`
