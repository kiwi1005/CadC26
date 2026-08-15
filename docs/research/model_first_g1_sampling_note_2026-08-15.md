# G1 K-sample topology note

日期：2026-08-15  
Parent：P12 Model-First Structured E2E Floorplanning  
Issue：#25 / #26

## Observed code fact

The existing `HCFPModel` produces:

- one case-conditioned positive soft assignment matrix;
- one case-conditioned negative soft assignment matrix;
- K population-conditioned center/aspect residuals.

`DualPermutationHead` is not population-conditioned. Therefore copying the same positive/negative hard permutation across K aspect samples would not constitute K diverse global Floorplan Programs.

## G1 decision

Do not add a second topology model before measuring the existing learned distribution. Generate K hard topology samples through deterministic seeded perturb-and-MAP:

```text
positive / negative soft assignments
-> log probabilities
-> sample-index Gumbel perturbation at frozen temperature
-> greedy hard one-to-one assignment
-> positive / negative permutations
```

Rules:

1. K=1 uses the unperturbed deterministic hard assignment.
2. K=4 uses fixed sample seeds and a frozen temperature.
3. Positive and negative assignments are sampled independently under the same sample ID.
4. Topology sample `i` is paired with aspect population member `i`.
5. Fixed/preplaced shapes override learned aspect exactly.
6. The compiler decodes each sampled program once and does not enumerate handcrafted variants.
7. Record unique topology count, duplicate rate, assignment likelihood, compile outcome and exact QoR.

This is sampling from the model's learned soft program distribution, not algorithmic topology search.

## Gate

- If K=4 produces meaningful topology diversity and improves exact QoR, KEEP the sampler for G1.
- If the soft assignments are too peaked and diversity collapses, report that as the isolated model-distribution limitation.
- Only after that evidence may a minimal sample-conditioned topology head be considered.
- Do not misattribute aspect-only diversity as global topology diversity.
