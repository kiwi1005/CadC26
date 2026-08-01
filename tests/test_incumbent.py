from __future__ import annotations

import torch

from hcfp.case import from_official
from hcfp.fallback import safe_shelf
from hcfp.incumbent import IncumbentManager


def _case():
    return from_official(
        3,
        [1.0, 1.0, 1.0],
        [[0, 1, 1.0], [1, 2, 1.0]],
        [],
        [],
        [[0, 0, 0, 0, 0]] * 3,
        [[-1.0] * 4] * 3,
    )


def test_exact_candidate_promotes_without_losing_safe_incumbent() -> None:
    case = _case()
    safe = safe_shelf(case)
    manager = IncumbentManager(case, safe)
    compact = safe.clone()
    compact[:, 0] = torch.tensor([0.0, compact[0, 2], compact[0, 2] + compact[1, 2]])
    compact[:, 1] = 0.0

    manager.consider(compact, source="compact")

    assert manager.safe.source == "fallback"
    assert manager.best_exact.source == "compact"
    assert manager.snapshot()["exact_source"] == "compact"


def test_fast_and_exact_failures_never_replace_exact_incumbent() -> None:
    case = _case()
    safe = safe_shelf(case)
    manager = IncumbentManager(case, safe)
    overlap = torch.tensor([[0.0, 0.0, 1.0, 1.0]] * 3)

    manager.consider(overlap, source="fast-reject", fast_feasible=False)
    manager.consider(overlap, source="exact-reject")
    manager.consider(torch.full((3, 4), float("nan")), source="invalid")

    assert manager.best_exact.source == "fallback"
    assert manager.snapshot()["rejections"] == {
        "exact_infeasible": 1,
        "fast_infeasible": 1,
        "invalid": 1,
    }
