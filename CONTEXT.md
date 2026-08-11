# HCFP-5090 Floorplanning

This context describes the shared language used to evaluate and improve HCFP-5090 floorplans. It separates official scoring states from internal benchmark labels so that case diagnosis is unambiguous.

## Cases and placements

**Case**:
One official floorplanning instance, including blocks, connectivity, pins, fixed or preplaced geometry, and soft constraints.
_Avoid_: Sample, layout, testcase when referring to the complete official instance

**Placement**:
A complete assignment of one axis-aligned rectangle to every block in a case.
_Avoid_: Candidate when the placement has already been selected as final output

**Candidate**:
A proposed placement that has not yet become the retained output for its case.
_Avoid_: Solution, result

**Incumbent**:
The currently retained hard-feasible placement against which another placement must be compared.
_Avoid_: Baseline when the retained placement may already include an improvement

**Analytic incumbent**:
The deterministic non-learned placement retained before learned candidates are considered.
_Avoid_: Ground truth, oracle

**Exact-eligible candidate**:
A candidate that passes exact hard verification and is eligible to challenge the incumbent.
_Avoid_: Feasible-looking candidate, projected candidate

**Candidate coverage failure**:
A case for which the learned lane produces no exact-eligible candidate in the evaluated candidate slice.
_Avoid_: Ranker failure

## Scoring states

**Hard feasible**:
A placement with no illegal overlap, no disallowed area error, and exact fixed-shape and preplaced geometry.
_Avoid_: Legal when only soft constraints are satisfied

**Soft violation**:
One unsatisfied boundary, grouping-connectivity, or MIB-shape obligation. Soft violations affect cost but do not make a placement hard infeasible.
_Avoid_: Constraint failure without naming whether the constraint is hard or soft

**Quality gap**:
The sum of the positive HPWL and bounding-area gaps used by the official quality factor.
_Avoid_: QoR when referring only to this factor

**Exact cap margin**:
`log(10) - log(uncapped_cost)` for a hard-feasible placement. A positive margin is below the official cap; a negative margin is still above it.
_Avoid_: Score margin, benchmark class

**Exact uncapped**:
A hard-feasible placement whose exact cap margin is positive.
_Avoid_: Competitive, because the internal competitive label uses a different threshold

**Benchmark-competitive**:
The internal benchmark label for a placement whose capped cost is below `9.99`. It is intentionally stricter than exact uncapped status.
_Avoid_: Uncapped

**Cap-saturated improvement**:
A placement whose geometry or uncapped factors improve while its reported capped cost remains tied at the cap.
_Avoid_: No improvement

**Cap blocker class**:
A counterfactual classification describing whether removing all quality contribution, all soft contribution, either one, or neither one could cross the cap. It does not claim that the other contribution is absent.
_Avoid_: Dominant constraint, quality-only

## Repair and selection

**Repair displacement**:
The movement between a candidate before exact repair and the corresponding post-repair placement.
_Avoid_: Optimization distance

**Constraint preservation failure**:
A soft relation constructed in an earlier stage that is lost during projection, legalization, or final repair.
_Avoid_: Constraint-generation failure when the relation was initially constructed

**Pareto guard**:
The safety rule that retains the incumbent unless a verified challenger satisfies the configured non-regression relation.
_Avoid_: Ranker

## Contest specialization

**Case signature**:
A set of observable properties used to route an unseen case, such as block count, constraint density, connectivity, anchor span, or incumbent geometry. It excludes a validation case identifier or a stored solution fingerprint.
_Avoid_: Case ID, testcase number

**Contest-aware specialization**:
Deliberate tuning to the official scoring rule and the FloorSet case distribution while preserving behavior on unseen cases through case-signature routing.
_Avoid_: Overfitting when the intended behavior is distribution-aware and transferable

**Solver portfolio**:
A bounded collection of placement and repair strategies from which one or more members are selected for a case by its signature.
_Avoid_: Random restarts when portfolio members have distinct structural roles

**Visible-case memorization**:
Returning case-specific parameters or geometry by recognizing a validation identifier or exact input fingerprint.
_Avoid_: Contest-aware specialization
