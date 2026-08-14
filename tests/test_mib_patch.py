from __future__ import annotations

from types import SimpleNamespace

import torch

from hcfp.mib_patch import mib_anchor_patch_candidates
from hcfp.verify import mib_violation, verify_feasible


def test_mib_anchor_patch_repacks_one_member_to_fixed_anchor_shape() -> None:
    boxes = torch.tensor(
        [
            [0.0, 0.0, 1.0, 1.0],
            [1.0, 0.0, 0.5, 2.0],
            [1.5, 0.0, 0.5, 2.0],
        ],
        dtype=torch.float64,
    )
    case = SimpleNamespace(
        n=3,
        normalized=False,
        area=torch.ones(3),
        target=boxes.clone(),
        fixed_mask=torch.tensor([True, False, False]),
        preplaced_mask=torch.zeros(3, dtype=torch.bool),
        mib_membership=torch.tensor([[True, True, False]]),
        boundary_bits=torch.zeros((3, 4), dtype=torch.bool),
    )

    candidates = mib_anchor_patch_candidates(
        case,
        boxes,
        patch_sizes=(3,),
        max_candidates=4,
    )

    assert candidates
    result = candidates[0].placement
    assert verify_feasible(case, result)
    assert mib_violation(case, result) == 0
    assert torch.equal(result[0], boxes[0])
    assert torch.allclose(
        result[:, 2:].prod(dim=1), torch.ones(3, dtype=result.dtype)
    )
