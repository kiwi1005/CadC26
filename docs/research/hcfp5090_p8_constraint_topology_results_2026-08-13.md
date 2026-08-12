# HCFP-5090 P8 constraint/topology experiments

## Outcome

P8 tested four bounded operators against the frozen P7 portfolio. The guarded
portfolio lowers the pinned full100 weighted local cost from `6.554801` to
`6.414040` with `100/100` hard feasibility, `48` improvements, `52` ties and
zero per-case regressions. Runtime increases from `4.098 s` to `4.862 s` at
p50 and from `12.274 s` to `13.120 s` at p95.

The result supports local exact re-slicing for dense constraint debt. It does
not support promoting unconstrained connectivity beams or the current sparse
region assignment.

## Experiment ledger

| Ticket | Hypothesis | Experiment result | Decision |
| --- | --- | --- | --- |
| #9 dense contact patch | A fixed-frame local re-slice can create missing group contact without requiring whitespace | cases `70,89,90,94,97`: weighted `7.733907 -> 7.550080`, hard `5/5`, 3 wins, 2 ties; 90/94/97 improve while 70/89 remain capped | KEEP guarded |
| #10 boundary skeleton | Reassigning current bbox witnesses can remove boundary misses without moving the frame | exact candidates reduce case 89 boundary misses by one, but the five-case selected score is unchanged beyond #9 | MODIFY; opt-in |
| #11 obstacle-aware region assignment | Maximal free rectangles plus at most one low-cut split can repair sparse anchor fragmentation | cases `85,92,93,96,98,99` unchanged; p50 rises to about `20.1 s`; case 93 candidates are tight but carry much higher soft/topology debt | REJECT current lane |
| #12 connectivity-aware B*-Tree beam | Runtime net/group/boundary rewards can recover part of the gold-tree topology headroom | unguarded full100 reaches `6.316556` but has 9 regressions; additive base preservation plus componentwise exact guard removes all regressions | KEEP shadow/guarded |

## Implemented operators

### Dense contact patch

`contact_patch.py` extracts a bounded 4/8/12/16-block patch around a broken
group obligation, preserves the patch frame, and enumerates exact slicing
orders that force one side contact. Candidates must pass the exact verifier and
reduce grouping violations before reaching the relative proxy guard.

### Boundary skeleton

`boundary_skeleton.py` reassigns one or two bbox witnesses to blocks that
actually owe the corresponding side. It preserves protected anchors and uses
existing free-rectangle geometry when a direct swap is blocked.

### Obstacle-aware region assignment

`region_assignment.py` subtracts preplaced obstacles from latent-outline
hypotheses, assigns movable islands to free rectangles, and allows one
low-connectivity cut only when a rigid unit does not fit. Fixed-shape blocks may
move but keep exact dimensions; preplaced blocks remain immobile. The mechanism
works in the synthetic split test but failed the real sparse-case QoR gate.

### Connectivity-aware B*-Tree beam

The hard decoder scores valid tree expansions with learned edge logits plus
runtime-visible b2b, same-group and boundary-side rewards. The P7 greedy tree
and all of its x/y seeds remain unchanged; beam trees are additional
challengers. Generic selection cannot auto-promote them. A dedicated exact
guard admits a beam candidate only when soft violation, bbox area and HPWL are
all no worse and at least one improves.

## Full100 evidence

| Portfolio | Weighted cost | Hard feasible | Capped | Wins / ties / regressions | p50 | p95 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| P7 frozen | 6.554801 | 100/100 | 2 | baseline | 4.098 s | 12.274 s |
| P8 unguarded | 6.316556 | 100/100 | 1 | 64 / 27 / 9 | 4.824 s | 12.911 s |
| P8 guarded | **6.414040** | **100/100** | **2** | **48 / 52 / 0** | 4.862 s | 13.120 s |

For the score-heavy 106-120 bucket, guarded weighted cost is `6.530033`.
The unguarded beam reaches `6.410328` in that bucket but is not promoted because
its regressions include cases 85 (`n=106`) and 95 (`n=116`).

Artifacts:

- `artifacts/benchmarks/hcfp5090-p7-dual-island-full100.json`
- `artifacts/benchmarks/hcfp5090-p8-integrated-full100.json`
- `artifacts/benchmarks/hcfp5090-p8-guarded-dense5.json`
- `artifacts/benchmarks/hcfp5090-p8-guarded-full100.json`
- `artifacts/benchmarks/hcfp5090-p8-region-assignment-cases85-92-93-96-98-99.json`

## Decision

**KEEP** the dense contact patch and the additive/guarded connectivity decode.
Keep boundary witness reassignment opt-in until it produces selected QoR wins.
Disable region assignment by default.

The next experiment should calibrate a bounded official-like trade-off guard
for connectivity beams using training-held-out cases. The unguarded result
shows headroom, but the current hidden-baseline-free selector cannot safely
exchange a large bbox increase for lower soft violation and HPWL.
