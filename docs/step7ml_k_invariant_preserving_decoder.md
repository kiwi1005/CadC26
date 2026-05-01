# Step7ML-K Invariant-Preserving Geometry Decoder

Step7ML-K is a sidecar macro/topology decoder that generates only
structure-preserving placements before Step7N-I/Step7ML-J style selection. It
keeps placement inside the original closure bbox envelope, preserves fixed and
preplaced coordinates, and probes order/compound-unit shelf/slot structure.

Implemented probes:

- `bbox_envelope_shelf_decoder`
- `order_preserving_slot_decoder`
- `compound_unit_shelf_decoder`

This does not integrate into the runtime solver, does not change finalizer
semantics, and does not introduce scalar bbox/soft penalty thresholds.
