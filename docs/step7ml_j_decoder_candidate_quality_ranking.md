# Step7ML-J Decoder Candidate Quality Ranking

Step7ML-J is a sidecar selector over Step7ML-I geometry-aware decoded macro
candidates. It converts decoded rows into a common objective-vector schema and
uses constrained Pareto selection with original/current anchors instead of scalar
penalty thresholds.

The selector keeps global/no-op/infeasible/dominated/metric-regressing rows in
reports. A row is selected when it is hard-feasible, non-global,
non-dominated, and metric-safe, or when it preserves a known Step7P/Step7N-I
winner/gate row.

This remains sidecar-only and does not modify the runtime solver or finalizer.
