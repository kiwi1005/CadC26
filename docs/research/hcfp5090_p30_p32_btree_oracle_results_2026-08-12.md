# HCFP-5090 P3.0-P3.2 B*-Tree Oracle Results

## Hypothesis

The raw FloorSet `tree_sol` contains a stable B*-Tree topology, and that topology can add QoR beyond P2 when paired only with runtime-available shapes, anchors, and vertical ordering.

## P3.0 schema audit

`scripts/audit_tree_sol_schema.py` reads raw `payload[4]` without changing the production loader.

On 512 large samples:

| Metric | Result |
| --- | ---: |
| Valid rooted trees | 512/512 |
| Connected and acyclic | 512/512 |
| Inferred row schema | `[parent, child, side]` |
| Side-0 child is right of parent | 100% |
| Side-1 child shares parent x | 100% |
| Standard contour decode overlap-free | 100% |

The standard DFS contour is not the final FloorSet compaction rule. With gold sizes its median bbox-area relative error is 77.9%, while exact-position agreement is only 31.6%. The tree provides the horizontal structure; final y placement is substantially compacted.

## Minimal usable decoder

`src/hcfp/btree.py` implements:

- validated B*-Tree parsing;
- standard contour packing for schema checks;
- anchor-aware contour initialization for exact preplaced obstacles;
- x-structure decoding from the B*-Tree;
- y compaction using a runtime-available vertical order;
- a small numerical gutter to avoid float-touch overlap false positives.

The usable oracle uses no gold coordinates or soft-block dimensions. Its candidates use:

- the gold tree only as the topology oracle;
- square/current learned shape hypotheses;
- current MIB shape construction;
- the P1 outline beam;
- current learned-candidate y orders;
- exact feasibility and official offline metrics.

## Results

Single-case smoke:

```text
current cost:       9.423926
B*-Tree oracle:     6.816200
hard feasible:      yes
overlap pairs:      0
unique win:         yes
```

Sixteen large samples:

| Metric | Current portfolio | Usable B*-Tree oracle |
| --- | ---: | ---: |
| Weighted cost | 8.064641 | 6.569232 |
| Below cap | 16/16 | 15/16 |
| Hard-feasible oracle | - | 16/16 |
| Unique wins | - | 14/16 |

The experiment passes the planned 5% unique-win gate by a wide margin. The one capped B*-Tree candidate and one mild regression show that the family must coexist with the incumbent rather than replace it.

## Decision

**KEEP / PROMOTE TO P3.3.** The topology oracle has clear usable headroom. Proceed with a runtime topology constructor and bounded contact-aware tree search. Do not replace the P2 portfolio, and do not judge the next model by serialized tree-edge accuracy alone; candidate-level QoR remains the gate.

Known experiment limitation: the historical checkpoint's stored training-stream hash no longer matches the reconstructed local stream, so this first 16-case run used explicit experiment-mode training-exclusion bypass with a different held-out seed. Treat it as topology-headroom evidence, not final clean generalization evidence.

Artifacts:

- `artifacts/benchmarks/hcfp5090-p30-tree-sol-schema-large512.json`
- `artifacts/benchmarks/hcfp5090-p32-btree-xcompact-debug1.json`
- `artifacts/benchmarks/hcfp5090-p32-btree-xcompact-large16.json`
