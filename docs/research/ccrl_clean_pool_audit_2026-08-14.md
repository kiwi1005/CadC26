# CCRL clean-placement pool audit

Date: 2026-08-14  
Source: `/home/hwchen/PROJ/CadC26/artifacts/floorset-v10`  
Samples: 10000

## Result

| Metric | Value |
| --- | ---: |
| Exact hard-feasible `fp_sol` | 10000 / 10000 |
| Contact-clean train sources | 7864 |
| Contact-clean held-out sources | 2136 |
| Structural Contact C0 / C1 / C2 | 9873 / 10000 / 10000 |
| Connected groups | 34268 / 35008 |
| Uniform MIB groups | 76 / 10000 |
| Valid `tree_sol` | 10000 / 10000 |
| Split overlap | 0 |

## Gates

- hard verifier parity: `True`
- source split disjoint: `True`
- 2K train / 512 held-out Contact volume: `True`
- actual C0/C1 corruption yield: `DEFERRED_TO_ISSUE_18`

## Decision

`KEEP` for the P11.1 data foundation. Actual corruption success remains owned by issue #18.
