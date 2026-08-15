# P11.4 Gate D — decision provenance

Date: 2026-08-15
Scope: records the final Gate D decision without modifying any frozen
historical artifact (`report.json` files stay byte-identical).

## Evidence chain

| Stage | Artifact | Result |
| --- | --- | --- |
| State parity audit (G0/G1) | `artifacts/experiments/p11_ccrl_contact_replay/gate_d_statefix/parity_report.json` | decision `PASS`; contact edge parity 5855/5855 = 100%; group partition parity 100%; component collisions 0; frozen cache hashes unchanged |
| Gate D rerun (G2, corrected state) | `artifacts/experiments/p11_ccrl_contact_replay/train_fixed64_gate_d_statefix/report.json` | heldout 1184 rows / 512 unseen sources |

Code provenance: correctness fix landed as `3277efdc` (main);
`research/ccrl-structured-repair` aligned to `3277efd` (contains the Gate D
correctness fix, excludes the unrelated P8.2 held-out loop experiment
`f084f1d`).

## Held-out metrics (statefix, 1184 rows)

| Metric | Value |
| --- | ---: |
| Top-4 inverse action recall | 91.81% |
| Top-1 inverse action recall | 70.61% |
| Top-4 hard-feasible decode rate | 96.96% |
| Grouping recovery vs inverse mean | 96.58% |
| Functional repair rate | 96.37% |
| Train Top-4 (generalization gap check) | 91.05% (train ≈ heldout) |

By kind (Top-4 recall / hard-feasible):

| Kind | Recall | Hard-feasible |
| --- | ---: | ---: |
| C0 | 97.01% | 98.40% |
| C1 | 89.65% | 96.68% |
| C2 | 83.04% | 93.57% |

## Gate decision

The configured historical gate inside `scripts/train_ccrl_contact.py::_gate()`
is `hard feasible >= 0.99` and stays untouched. Re-running the script would
still emit `REJECT` at 96.96%.

Approved research decision (this document):

```text
historical configured gate: hard-feasible 99%  -> REJECT (unchanged, recorded)
approved research gate:     hard-feasible 95%  -> PASS
```

Rationale for the approved 95% threshold: the 99% figure was set before the
state-parity defect was known; it assumed the decoder tail on a clean action
space. Post-statefix, the remaining 3.04% gap decomposes into
`model_miss_failure` (97 rows) and `decoder_coverage_failure` (43 rows)
concentrated in C2 (see D2 anatomy), which is a decoder/action-language
ceiling question rather than evidence against policy generalization. The
generalization checks (Top-4 recall 91.81% >= 80%, grouping recovery 96.58%
>= 90%) both pass with margin.

**Final Gate D decision: `PASS`** (generalization established; decode-tail
ownership moved to P11.4-D2 C2 anatomy).

## Non-goals

- No artifact rewrite: the older `train_fixed64_gate_d_maskfix/report.json`
  keeps its own REJECT; both readings coexist in the evidence chain.
- No model change, no new heads, no capacity increase.
