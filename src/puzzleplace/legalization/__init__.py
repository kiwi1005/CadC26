"""Step7 sidecar legalization experiments.

This package is intentionally isolated from the contest runtime path.
"""

from .bounded_repair import BoundedRepairResult, RepairMode, bounded_repair

__all__ = ["BoundedRepairResult", "RepairMode", "bounded_repair"]
