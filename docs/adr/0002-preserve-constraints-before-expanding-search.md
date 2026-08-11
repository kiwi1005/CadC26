# Preserve constructed constraints before expanding search

HCFP-5090 repairs constraint preservation through projection and exact repair before increasing candidate count or reactivating collective rollout. Q6 reduces boundary and grouping violations but increases MIB violations from 55 to 243, so expanding search first would multiply candidates that can lose the same constructed relation in the tail.
