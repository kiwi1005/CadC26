from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from puzzleplace.actions.schema import ActionPrimitive, TypedAction
from puzzleplace.data.schema import FloorSetCase
from puzzleplace.models import (
    CandidateComponentRanker,
    CandidateLateFusionRanker,
    CandidateQualityRanker,
    CandidateRelationalActionQRanker,
    CandidateSetPairwiseRanker,
    HierarchicalSetPolicy,
    RelationAwareGraphStateEncoder,
)
from puzzleplace.roles import label_case_roles
from scripts.run_step6c_hierarchical_action_q_audit import FEATURE_NAMES, _aggregate


def _make_case() -> FloorSetCase:
    return FloorSetCase(
        case_id="hier-1",
        block_count=3,
        area_targets=torch.tensor([6.0, 6.0, 4.0]),
        b2b_edges=torch.tensor([[0.0, 1.0, 3.0], [1.0, 2.0, 1.0]]),
        p2b_edges=torch.empty((0, 3)),
        pins_pos=torch.empty((0, 2)),
        constraints=torch.zeros((3, 5)),
        target_positions=torch.tensor(
            [[0.0, 0.0, 2.0, 3.0], [2.0, 0.0, 3.0, 2.0], [0.0, 3.0, 2.0, 2.0]]
        ),
        metrics=None,
    )


def test_hierarchical_policy_output_shapes() -> None:
    case = _make_case()
    policy = HierarchicalSetPolicy(hidden_dim=32)
    output = policy(
        case,
        role_evidence=label_case_roles(case),
        placements={0: (0.0, 0.0, 2.0, 3.0)},
        state_step=1,
    )

    assert output.block_logits.shape == (case.block_count,)
    assert output.primitive_logits_by_block.shape == (
        case.block_count,
        len(ActionPrimitive),
    )
    assert output.target_logits.shape == (case.block_count, case.block_count)
    assert output.geometry.shape == (case.block_count, 4)


def test_relation_aware_encoder_output_shapes() -> None:
    case = _make_case()
    encoder = RelationAwareGraphStateEncoder(hidden_dim=32)
    encoded = encoder(
        case,
        role_evidence=label_case_roles(case),
        placements={0: (0.0, 0.0, 2.0, 3.0)},
        state_step=1,
    )

    assert encoded.block_embeddings.shape == (case.block_count, 32)
    assert encoded.graph_embedding.shape == (32,)
    assert encoded.block_mask.shape == (case.block_count,)


def test_hierarchical_policy_supports_relation_aware_encoder() -> None:
    case = _make_case()
    policy = HierarchicalSetPolicy(hidden_dim=32, encoder_kind="relation_aware")
    output = policy(
        case,
        role_evidence=label_case_roles(case),
        placements={0: (0.0, 0.0, 2.0, 3.0)},
        state_step=1,
    )

    assert output.block_logits.shape == (case.block_count,)
    assert output.primitive_logits_by_block.shape == (
        case.block_count,
        len(ActionPrimitive),
    )


def test_hierarchical_policy_can_optimize_block_and_primitive_set() -> None:
    case = _make_case()
    roles = label_case_roles(case)
    action = TypedAction(
        ActionPrimitive.PLACE_ABSOLUTE,
        block_index=1,
        x=2.0,
        y=0.0,
        w=3.0,
        h=2.0,
    )
    target_block = torch.tensor([action.block_index])
    primitive_ids = torch.tensor(
        [
            list(ActionPrimitive).index(ActionPrimitive.PLACE_ABSOLUTE),
            list(ActionPrimitive).index(ActionPrimitive.PLACE_RELATIVE),
        ],
        dtype=torch.long,
    )
    policy = HierarchicalSetPolicy(hidden_dim=32)
    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-2)

    first_loss = None
    last_loss = None
    for step in range(20):
        optimizer.zero_grad(set_to_none=True)
        output = policy(case, role_evidence=roles, placements={}, state_step=0)
        block_loss = torch.nn.functional.cross_entropy(
            output.block_logits.unsqueeze(0), target_block
        )
        primitive_log_probs = torch.nn.functional.log_softmax(
            output.primitive_logits_by_block[action.block_index], dim=0
        )
        primitive_loss = -torch.logsumexp(primitive_log_probs[primitive_ids], dim=0)
        loss = block_loss + primitive_loss
        loss.backward()
        optimizer.step()
        if step == 0:
            first_loss = float(loss.item())
        last_loss = float(loss.item())

    assert first_loss is not None and last_loss is not None
    assert last_loss < first_loss


def test_candidate_quality_ranker_can_fit_tiny_preference() -> None:
    features = torch.tensor(
        [
            [0.0, 1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=torch.float32,
    )
    target = torch.tensor([1])
    ranker = CandidateQualityRanker(feature_dim=3, hidden_dim=8)
    optimizer = torch.optim.Adam(ranker.parameters(), lr=1e-2)
    first_loss = None
    last_loss = None
    for step in range(80):
        optimizer.zero_grad(set_to_none=True)
        logits = ranker(features).unsqueeze(0)
        loss = torch.nn.functional.cross_entropy(logits, target)
        loss.backward()
        optimizer.step()
        if step == 0:
            first_loss = float(loss.item())
        last_loss = float(loss.item())

    assert first_loss is not None and last_loss is not None
    assert last_loss < first_loss
    assert int(ranker(features).argmax().item()) == 1


def test_candidate_component_ranker_outputs_overall_and_components() -> None:
    ranker = CandidateComponentRanker(feature_dim=4, hidden_dim=8, component_count=3)
    overall, components = ranker(torch.zeros((5, 4), dtype=torch.float32))

    assert overall.shape == (5,)
    assert components.shape == (5, 3)


def test_candidate_set_pairwise_ranker_scores_candidate_pool() -> None:
    ranker = CandidateSetPairwiseRanker(feature_dim=4, hidden_dim=8, num_heads=2)
    features = torch.zeros((5, 4), dtype=torch.float32)

    pair_logits = ranker.pair_logits(features)
    scalar_scores = ranker.scalar_scores(features)
    scores = ranker.score_candidates(features)
    hybrid_scores = ranker.hybrid_scores(features, pairwise_weight=0.5)

    assert pair_logits.shape == (5, 5)
    assert scalar_scores.shape == (5,)
    assert scores.shape == (5,)
    assert hybrid_scores.shape == (5,)


def test_candidate_late_fusion_ranker_scores_candidate_pool() -> None:
    ranker = CandidateLateFusionRanker(feature_dim=4, hidden_dim=8, num_heads=2)
    features = torch.zeros((5, 4), dtype=torch.float32)

    scalar_scores = ranker.scalar_scores(features)
    pairwise_scores = ranker.pairwise_scores(features)
    hybrid_scores = ranker.hybrid_scores(features, pairwise_weight=0.5)

    assert scalar_scores.shape == (5,)
    assert pairwise_scores.shape == (5,)
    assert hybrid_scores.shape == (5,)


def test_candidate_relational_action_q_ranker_scores_candidate_pool() -> None:
    ranker = CandidateRelationalActionQRanker(feature_dim=4, hidden_dim=8, num_heads=2)
    features = torch.zeros((5, 4), dtype=torch.float32)

    scores = ranker.score_candidates(features)
    pair_logits = ranker.pair_logits(features)
    component_logits = ranker.component_logits(features)

    assert scores.shape == (5,)
    assert pair_logits.shape == (5, 5)
    assert component_logits.shape == (5, 3)


def test_candidate_relational_action_q_ranker_can_fit_tiny_preference() -> None:
    features = torch.tensor(
        [
            [0.0, 1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=torch.float32,
    )
    target = torch.tensor([1])
    ranker = CandidateRelationalActionQRanker(feature_dim=3, hidden_dim=8, num_heads=2)
    optimizer = torch.optim.Adam(ranker.parameters(), lr=1e-2)
    first_loss = None
    last_loss = None
    for step in range(80):
        optimizer.zero_grad(set_to_none=True)
        scores = ranker.score_candidates(features).unsqueeze(0)
        loss = torch.nn.functional.cross_entropy(scores, target)
        loss.backward()
        optimizer.step()
        if step == 0:
            first_loss = float(loss.item())
        last_loss = float(loss.item())

    assert first_loss is not None and last_loss is not None
    assert last_loss < first_loss
    assert int(ranker.score_candidates(features).argmax().item()) == 1


def test_action_q_audit_aggregate_supports_heldout_split() -> None:
    feature_dim = len(FEATURE_NAMES)
    action_a = {"action_key": "a", "quality_cost_runtime1": 1.0}
    action_b = {"action_key": "b", "quality_cost_runtime1": 2.0}
    collections = [
        {
            "case_id": "validation-0",
            "case_index": 0,
            "policy_seed": 0,
            "training_summary": {},
            "pools": [
                {
                    "case_id": "validation-0",
                    "case_index": 0,
                    "policy_seed": 0,
                    "step": 0,
                    "feature_rows": [[1.0] * feature_dim, [0.0] * feature_dim],
                    "quality_values": [1.0, 2.0],
                    "oracle_index": 0,
                    "candidate_rows": [action_a, action_b],
                }
            ],
        },
        {
            "case_id": "validation-1",
            "case_index": 1,
            "policy_seed": 0,
            "training_summary": {},
            "pools": [
                {
                    "case_id": "validation-1",
                    "case_index": 1,
                    "policy_seed": 0,
                    "step": 0,
                    "feature_rows": [[0.5] * feature_dim, [0.25] * feature_dim],
                    "quality_values": [2.0, 1.0],
                    "oracle_index": 1,
                    "candidate_rows": [action_b, action_a],
                }
            ],
        },
    ]
    args = argparse.Namespace(
        train_case_ids=[0],
        eval_case_ids=[1],
        ranker_epochs=2,
        ranker_lr=1e-2,
        feature_mode="legacy",
        feature_normalization="per_case",
        encoder_kind="relation_aware",
        ranker_kind="relational_action_q",
        component_loss_weight=0.5,
        component_score_weight=0.25,
        target_kind="oracle_ce",
        quality_temperature=0.5,
        pairwise_loss_weight_kind="quality_delta",
        pairwise_listwise_loss_weight=0.0,
        hybrid_scalar_loss_weight=1.0,
        hybrid_pairwise_score_weight=0.5,
        relational_pairwise_loss_weight=0.5,
        seed=0,
    )

    payload = _aggregate(collections, workers=2, args=args)

    assert payload["evaluation_mode"] == "heldout_split"
    assert payload["case_count"] == 2
    assert payload["collection_count"] == 2
    assert payload["policy_seeds"] == [0]
    assert payload["train_case_ids"] == [0]
    assert payload["eval_case_ids"] == [1]
    assert payload["train_pool_count"] == 1
    assert payload["eval_pool_count"] == 1
    assert payload["heldout_gate"] is not None
    assert payload["evaluation"]["pool_count"] == 1
    assert payload["train_evaluation"]["pool_count"] == 1


def test_action_q_audit_aggregate_supports_leave_one_case_out() -> None:
    feature_dim = len(FEATURE_NAMES)
    action_a = {
        "action_key": "a",
        "quality_cost_runtime1": 1.0,
        "HPWLgap": 1.0,
        "Areagap_bbox": 0.0,
        "Violationsrelative": 0.0,
    }
    action_b = {
        "action_key": "b",
        "quality_cost_runtime1": 2.0,
        "HPWLgap": 2.0,
        "Areagap_bbox": 0.0,
        "Violationsrelative": 0.0,
    }
    collections = []
    for case_index in range(3):
        collections.append(
            {
                "case_id": f"validation-{case_index}",
                "case_index": case_index,
                "policy_seed": 0,
                "training_summary": {},
                "pools": [
                    {
                        "case_id": f"validation-{case_index}",
                        "case_index": case_index,
                        "policy_seed": 0,
                        "step": 0,
                        "feature_rows": [
                            [float(case_index + 1)] * feature_dim,
                            [0.0] * feature_dim,
                        ],
                        "quality_values": [1.0, 2.0],
                        "oracle_index": 0,
                        "candidate_rows": [action_a, action_b],
                    }
                ],
            }
        )
    args = argparse.Namespace(
        train_case_ids=None,
        eval_case_ids=None,
        leave_one_case_out=True,
        ranker_epochs=1,
        ranker_lr=1e-2,
        feature_mode="legacy",
        feature_normalization="per_case",
        encoder_kind="relation_aware",
        ranker_kind="scalar",
        component_loss_weight=0.5,
        component_score_weight=0.25,
        target_kind="oracle_ce",
        quality_temperature=0.5,
        pairwise_loss_weight_kind="quality_delta",
        pairwise_listwise_loss_weight=0.0,
        hybrid_scalar_loss_weight=1.0,
        hybrid_pairwise_score_weight=0.5,
        relational_pairwise_loss_weight=0.5,
        seed=0,
    )

    payload = _aggregate(collections, workers=3, args=args)

    assert payload["evaluation_mode"] == "leave_one_case_out"
    assert payload["case_count"] == 3
    assert payload["collection_count"] == 3
    assert payload["policy_seeds"] == [0]
    assert payload["evaluation"]["split_count"] == 3
    assert payload["leave_one_case_out_gate"] is not None
    assert len(payload["split_results"]) == 3
    assert all(result["feature_shift"]["top_features"] for result in payload["split_results"])


def test_action_q_audit_per_case_normalization_removes_case_scale() -> None:
    from scripts.run_step6c_hierarchical_action_q_audit import _normalize_pools_for_ranker

    pools = [
        {
            "case_index": 0,
            "feature_rows": [[1.0, 10.0], [3.0, 30.0]],
        },
        {
            "case_index": 1,
            "feature_rows": [[100.0, 1000.0], [300.0, 3000.0]],
        },
    ]

    normalized = _normalize_pools_for_ranker(pools, mode="per_case")

    for pool in normalized:
        values = torch.tensor(pool["feature_rows"], dtype=torch.float32)
        assert torch.allclose(values.mean(dim=0), torch.zeros(2), atol=1e-5)
