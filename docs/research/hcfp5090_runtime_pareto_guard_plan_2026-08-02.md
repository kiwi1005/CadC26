# HCFP-5090 runtime Pareto guard phase

Date: 2026-08-02

## Outcome

Prevent the learned sidecar from returning a placement that is strictly worse
than the protected standalone analytic incumbent.
The guard must be baseline-free, deterministic, and preserve every learned
candidate that trades one official-quality dimension for another.

## Evidence and root cause

The official validation replay found material regressions on cases 88 and 97.
The candidate attribution report shows that the existing population already
contains a better `analytic_initial` candidate in both cases:

| Case | Current source | Analytic source | Soft | Area gap | HPWL gap |
| ---: | --- | --- | ---: | ---: | ---: |
| 88 | `analytic_initial[8]` | `analytic_initial[5]` | 0.904762 -> 0.888889 | 2.974807 -> 2.492084 | 3.524091 -> 2.972465 |
| 97 | `fallback[0]` | `analytic_initial[7]` | 0.933333 -> 0.900000 | 13.742664 -> 1.986263 | 12.251835 -> 2.831914 |

Both replacements strictly dominate the incumbent in all three monotone
runtime-independent official-quality dimensions. No official baseline is
needed to prove that they cannot be worse.

The first mixed-tail diagnosis found normalized FP32 verifier false negatives:
edge touching a preplaced block acquires a 1--2 ULP overlap during normalized
geometry arithmetic. Replaying the original hard target in raw coordinates
produces official-feasible geometry. Loosening the normalized hard tolerance
would weaken every analytic hard gate, so this phase keeps it unchanged and
performs final admission checks through the existing raw verifier.

The first fast-candidate guard experiment reduced case 88 from `25.953674` to `22.082727` and
case 97 from `90.516082` to `20.623784`. Case 97 then beat its standalone
analytic result, but case 88 still trailed the standalone analytic objective
`21.335809`. The mixed 12-candidate relax/projection batch therefore does not
preserve the exact FP32 path of the standalone 8-candidate analytic lane. That
mixed-batch-only experiment was rejected in favor of protecting the real
standalone exact incumbent while retaining its fast candidate as a secondary
raw-replay opportunity.

## Minimal runtime change

Add one helper in `hcfp.learned` and call it only after the currently selected
placement has passed the raw verifier:

1. Recover the selected candidate plus the standalone analytic exact and fast
   incumbent indices from the merged snapshot.
2. Skip candidates that already identify the selected placement.
3. Convert at most those two analytic candidates to raw coordinates, replay
   exact hard targets,
   recompute all three dimensions for both sides, and require strict Pareto
   dominance plus the existing exact raw feasibility check.
4. Return the protected incumbent only when it is raw-feasible, no worse in all
   three dimensions, and materially better in at least one; otherwise keep the
   learned result.

This deliberately does not add a second analytic solve, an official evaluator
dependency, a baseline lookup, or a new model/configuration surface.

To preserve the real analytic incumbent without duplicating tail work, execute
the analytic and ranker-pruned learned populations as separate tail batches and
merge their raw candidates, projected candidates, telemetry, and incumbent
selection back into the existing source layout:

```text
fallback
analytic initial
learned initial
analytic relaxed
learned relaxed
```

The analytic batch uses the unchanged standalone solver path. The learned batch
contains only the sidecar candidates. Their combined relax/projection candidate
count is the same apart from one duplicated fallback, while the analytic
geometry is now exactly reusable as the protected baseline.

## Verification gates

- Unit: a raw-feasible analytic dominator replaces the current candidate.
- Unit: trade-off, equal, raw-infeasible, and learned-only candidates do not
  replace the current result.
- Unit: malformed/unknown incumbent source fails closed to the current result.
- Unit: merged analytic initial/relaxed tensors exactly equal a standalone
  analytic telemetry run with the same configuration.
- Material replay: cases 88 and 97 must become non-regressing with flow and
  execution seed `0`.
- Preservation replay: the eight previously improved cases must not regress to
  analytic merely because the guard exists.
- Runtime: report p50/p95 on the material-case set; the guard converts at most
  the protected analytic exact and fast candidates, not the full population.
- Full gate: run official validation 100 only after the material replay passes;
  retain 100/100 hard feasibility and no new `cost=10` case.

## Stop conditions

- Do not change `IncumbentManager`, normalized overlap tolerance, strict
  projection, submission entrypoint, or fallback semantics in this phase.
- Do not promote the learned lane if weighted uncapped objective or the
  106--120 subset still regresses after the guard.
