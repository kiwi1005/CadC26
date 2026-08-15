from __future__ import annotations

import torch
import pytest

from hcfp.geometry import normalize_xywh
from hcfp.repair.corruption.contact import generate_contact_corruptions
from hcfp.repair.decoders.contact import decode_contact_action
from hcfp.repair.losses import contact_action_loss
from hcfp.repair.model import (
    ContactRepairModel,
    RepairModelConfig,
    contact_action_masks,
    topk_contact_actions,
)
from hcfp.repair.schema import ExpertKind, RepairObligation
from hcfp.repair.state import build_repair_state
from test_repair_dataset import _clean_source


def _c1_fixture():
    sample, source = _clean_source()
    clean = source["fp_sol_xywh"].double()
    corruption = generate_contact_corruptions(
        sample.case, clean, verify_case=source, kinds=("C1",)
    )[0]
    group_index = int(corruption.inverse_action.obligation_id.rsplit(":", 1)[1])
    members = tuple(
        torch.nonzero(sample.case.group_membership[group_index], as_tuple=False)
        .reshape(-1)
        .tolist()
    )
    clean_state = build_repair_state(
        sample.case,
        normalize_xywh(sample.case, clean),
        exact_contact_placement=clean,
        corruption_kind="clean",
    )
    corrupt_state = build_repair_state(
        sample.case,
        normalize_xywh(sample.case, corruption.placement),
        exact_contact_placement=corruption.placement,
        corruption_kind=corruption.kind.lower(),
        corruption_level=1,
    )
    obligation = RepairObligation(
        ExpertKind.CONTACT,
        corruption.inverse_action.obligation_id,
        members,
        debt=corruption.debt_after,
    )
    return sample, source, corruption, clean_state, corrupt_state, obligation


def test_dynamic_encoder_sees_state_and_mobility_masks() -> None:
    sample, _source, corruption, clean, corrupt, obligation = _c1_fixture()
    torch.manual_seed(5090)
    model = ContactRepairModel(RepairModelConfig())

    clean_output = model(clean, obligation)
    corrupt_output = model(corrupt, obligation)
    masks = contact_action_masks(corrupt, obligation)

    assert torch.isfinite(clean_output.embedding).all()
    assert not torch.allclose(clean_output.embedding, corrupt_output.embedding)
    assert not masks.target[int(sample.case.preplaced_mask.nonzero()[0])]
    fixed = int(sample.case.fixed_mask.nonzero()[0])
    assert masks.target[fixed]
    assert not bool(masks.anchor[int(sample.case.preplaced_mask.nonzero()[0])].any())
    target = corruption.inverse_action.target_ids[0]
    anchor = corruption.inverse_action.anchor_ids[0]
    assert bool(masks.patch_budget[target, anchor].all())
    assert torch.all(corrupt_output.side_logits[~masks.side] < -1.0e20)


def test_contact_debug_model_overfits_one_state_and_decodes() -> None:
    sample, source, corruption, _clean, state, obligation = _c1_fixture()
    torch.manual_seed(5090)
    model = ContactRepairModel(RepairModelConfig())
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.0e-2)

    for _ in range(320):
        optimizer.zero_grad(set_to_none=True)
        output = model(state, obligation)
        report = contact_action_loss(output, corruption.inverse_action)
        report.total.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        output = model(state, obligation)
        report = contact_action_loss(output, corruption.inverse_action)
        predicted = topk_contact_actions(output, obligation, k=1)[0]

    assert torch.exp(-report.total) > 0.95
    decoded = decode_contact_action(
        sample.case,
        corruption.placement,
        predicted,
        verify_case=source,
    )
    assert decoded.succeeded
    assert decoded.debt_after is not None
    assert decoded.debt_after < corruption.debt_after


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_contact_debug_model_has_finite_cuda_gradients() -> None:
    _sample, _source, corruption, _clean, state, obligation = _c1_fixture()
    torch.manual_seed(5090)
    model = ContactRepairModel(RepairModelConfig()).cuda()

    report = contact_action_loss(model(state, obligation), corruption.inverse_action)
    report.total.backward()

    assert torch.isfinite(report.total)
    assert all(
        parameter.grad is None or torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    )
