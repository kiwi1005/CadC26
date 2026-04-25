# Step6K Boundary Failure Attribution Plan

## Scope

Step6K follows the updated `AGENT.md` directive: do attribution before adding edge segment assignment.  Step6J already fixed the Step6I virtual-frame over-commit regression by preferring predicted compact hull ownership, but cases 3/4 still have low final-bbox boundary satisfaction.

Runtime remains frozen.  Step6K is a sidecar diagnostic pass over Step6J winning layouts.

## Implementation Plan

1. **Final bbox edge owner audit**
   - For each final post-repair layout, identify left/right/bottom/top edge owners.
   - Record role flags, boundary-required ownership, regular/non-boundary stealing, external ratio, and leave-one-out bbox expansion contribution.
   - Artifact: `artifacts/research/step6k_final_bbox_edge_owner_audit.json`.

2. **Boundary failure classification**
   - Classify each unsatisfied boundary block using Step6J pre-repair placement, post-repair final bbox, predicted hull satisfaction, candidate coverage, final edge owner rows, role flags, and same-edge conflict checks.
   - Primary classes include predicted-hull mismatch, edge stealing, segment conflict, role conflicts, shape mismatch, candidate missing/not-selected, and postprocess change.
   - Artifact: `artifacts/research/step6k_boundary_failure_classification.json`.

3. **Boundary role overlap audit**
   - Count boundary-only, boundary+grouping, boundary+MIB, terminal-heavy, fixed/preplaced, and multi-role blocks, with special unsatisfied counts.
   - Artifact: `artifacts/research/step6k_boundary_role_overlap_audit.json`.

4. **Compaction baseline**
   - Run deterministic left/bottom compaction within the virtual frame on Step6J post-repair placements.
   - Preserve fixed/preplaced blocks and no-overlap legality.
   - Compare boundary satisfaction, bbox area, internal/external HPWL, and hard feasibility.
   - Artifact: `artifacts/research/step6k_compaction_baseline.json`.

5. **Decision tree**
   - Choose the next minimal fix from AGENT.md only after the audits: hull stealing prevention, edge-aware compaction, group-aware candidates, MIB-compatible slots, edge segment assignment, or a narrower candidate-selection audit.

## Gates

- Reuse Step6J quality guardrails unchanged because Step6K should not modify layouts.
- Emit all four diagnostic artifacts.
- Explicitly identify case 3/4 failure types.
- Keep runtime/contest solver untouched.
