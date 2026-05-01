# Step7ML-I Geometry-Aware Decoding / Slot Legalization Layer

Step7ML-I is a sidecar probe that replaces independent normalized x/y regression
as the final macro closure placement with deterministic geometry-aware decoders.
It consumes Step7P exact geometry payloads and enforces non-overlap, bbox
containment, fixed/preplaced preservation, and no-op detection before joining
Step7P/Step7G/Step7N-I provenance for comparison.

## Decoders

- `slot_assignment_decoder`: generate legal top-left slots inside the closure
  bbox and greedily assign movable blocks to non-overlapping slots with separate
  displacement and centroid-pull cost components.
- `shelf_row_decoder`: pack movable blocks into deterministic rows/shelves while
  preserving fixed/preplaced blocks.

## Scope

This does not integrate into the runtime solver or change finalizer semantics.
Official-like metric labels are not recomputed by this probe; they are joined
from Step7P rows only after deterministic geometry screening.
