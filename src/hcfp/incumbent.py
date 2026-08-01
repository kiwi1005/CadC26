"""Incumbent tiers that never lose the exact-safe HCFP placement."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from hcfp.case import FloorplanCase
from hcfp.geometry import bbox_area_tensor, centers_from_xywh, hpwl_tensor
from hcfp.verify import soft_violation_normalized, verify_feasible


Tensor = torch.Tensor


@dataclass(frozen=True)
class Incumbent:
    source: str
    tier: str
    xywh: Tensor
    key: tuple[float, float]


class IncumbentManager:
    """Track safe, fast-feasible, and exact-feasible candidates."""

    def __init__(self, case: FloorplanCase, safe: Tensor, *, source: str = "fallback") -> None:
        self.case = case.to(device="cpu", dtype=torch.float32)
        candidate = self._cpu_candidate(safe)
        if not verify_feasible(self.case, candidate):
            raise ValueError("safe incumbent must be exact-feasible")
        incumbent = Incumbent(source, "exact", candidate, self._key(candidate))
        self.safe = incumbent
        self.best_fast: Incumbent | None = None
        self.best_exact = incumbent
        self.rejections: dict[str, int] = {}

    def consider(self, candidate: Tensor, *, source: str, fast_feasible: bool = True) -> Incumbent:
        try:
            work = self._cpu_candidate(candidate)
        except (TypeError, ValueError):
            self._reject("invalid")
            return self.best_exact
        if not fast_feasible:
            self._reject("fast_infeasible")
            return self.best_exact

        key = self._key(work)
        fast = Incumbent(source, "fast", work, key)
        if self.best_fast is None or key < self.best_fast.key:
            self.best_fast = fast
        if not verify_feasible(self.case, work):
            self._reject("exact_infeasible")
            return self.best_exact
        exact = Incumbent(source, "exact", work, key)
        if key < self.best_exact.key:
            self.best_exact = exact
        return self.best_exact

    def snapshot(self) -> dict[str, object]:
        return {
            "safe_source": self.safe.source,
            "fast_source": self.best_fast.source if self.best_fast else None,
            "exact_source": self.best_exact.source,
            "rejections": dict(sorted(self.rejections.items())),
        }

    def _cpu_candidate(self, candidate: Tensor) -> Tensor:
        work = torch.as_tensor(candidate, dtype=torch.float32, device="cpu")
        if work.shape != (self.case.n, 4):
            raise ValueError(f"candidate must have shape [{self.case.n},4]")
        if not bool(torch.isfinite(work).all()) or not bool((work[:, 2:4] > 0.0).all()):
            raise ValueError("candidate must contain finite positive rectangles")
        return work.detach().clone()

    def _key(self, candidate: Tensor) -> tuple[float, float]:
        soft = soft_violation_normalized(self.case, candidate).total
        quality = float(bbox_area_tensor(candidate)) + 0.05 * float(
            hpwl_tensor(self.case, centers_from_xywh(candidate))
        )
        return soft, quality

    def _reject(self, reason: str) -> None:
        self.rejections[reason] = self.rejections.get(reason, 0) + 1
