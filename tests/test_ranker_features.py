from __future__ import annotations

import torch

from hcfp.case import from_official
from hcfp.ranker_features import (
    KIND_FEATURE_OFFSET,
    PROJECTION_OK_FEATURE_INDEX,
    PROXY_FEATURE_OFFSET,
    RANKER_FEATURE_DIM,
    RANKER_FEATURE_NAMES,
    RANKER_FEATURE_VERSION,
    STAGE_FEATURE_INDEX,
    repair_aware_ranker_features,
)


def test_repair_aware_ranker_features_shape_finite_and_deterministic() -> None:
    case = _case()
    raw, post = _candidates()
    anchor = _anchor()
    kinds = ("learned", "constraint")
    first = repair_aware_ranker_features(case, raw, post, anchor, kinds, "initial")
    second = repair_aware_ranker_features(case, raw, post, anchor, kinds, "initial")

    assert RANKER_FEATURE_VERSION == "repair_aware_ranker_features_v5_family_identity"
    assert len(RANKER_FEATURE_NAMES) == RANKER_FEATURE_DIM == 28
    assert first.shape == (2, RANKER_FEATURE_DIM)
    assert torch.isfinite(first).all()
    torch.testing.assert_close(first, second, rtol=0.0, atol=0.0)


def test_repair_aware_ranker_features_encode_source_kind_one_hot() -> None:
    case = _case()
    raw, post = _three_candidates()
    features = repair_aware_ranker_features(
        case,
        raw,
        post,
        _anchor(),
        ("learned", "constraint", "topology"),
        "post_relax",
    )

    expected = torch.eye(3)
    torch.testing.assert_close(features[:, KIND_FEATURE_OFFSET : KIND_FEATURE_OFFSET + 3], expected)
    torch.testing.assert_close(
        features[:, STAGE_FEATURE_INDEX],
        torch.ones(3),
    )
    torch.testing.assert_close(
        features[:, PROJECTION_OK_FEATURE_INDEX],
        torch.tensor([1.0, 0.0, 0.0]),
    )


def test_repair_aware_ranker_features_keep_structured_families_distinct() -> None:
    case = _case()
    raw, post = _candidates()
    features = repair_aware_ranker_features(
        case,
        raw,
        post,
        _anchor(),
        ("treemap", "btree"),
        "initial",
    )

    torch.testing.assert_close(
        features[:, KIND_FEATURE_OFFSET : KIND_FEATURE_OFFSET + 5],
        torch.eye(5)[3:],
    )


def test_repair_aware_ranker_proxy_features_respond_to_repairs() -> None:
    case = _case()
    raw, post = _candidates()
    post = post.clone()
    post[1, 3, 0] = -1.0
    features = repair_aware_ranker_features(
        case,
        raw,
        post,
        _anchor(),
        ("learned", "constraint"),
        "initial",
    )
    boundary = features[:, PROXY_FEATURE_OFFSET]
    group = features[:, PROXY_FEATURE_OFFSET + 1]
    mib = features[:, PROXY_FEATURE_OFFSET + 2]

    assert boundary[1] > boundary[0]
    assert group[1] > group[0]
    assert mib[1] > mib[0]


def test_repair_aware_ranker_features_are_candidate_permutation_equivariant() -> None:
    case = _case()
    raw, post = _three_candidates()
    kinds = ("learned", "constraint", "topology")
    permutation = torch.tensor([2, 0, 1])

    features = repair_aware_ranker_features(
        case, raw, post, _anchor(), kinds, "initial"
    )
    permuted = repair_aware_ranker_features(
        case,
        raw[permutation],
        post[permutation],
        _anchor(),
        tuple(kinds[index] for index in permutation.tolist()),
        "initial",
    )

    torch.testing.assert_close(permuted, features[permutation], rtol=0.0, atol=0.0)


def test_group_proxy_cpu_and_cuda_paths_match() -> None:
    if not torch.cuda.is_available():
        return
    case = _case()
    raw, post = _three_candidates()
    kinds = ("learned", "constraint", "topology")

    cpu = repair_aware_ranker_features(case, raw, post, _anchor(), kinds, "initial")
    cuda = repair_aware_ranker_features(
        case.to(device="cuda"),
        raw.cuda(),
        post.cuda(),
        _anchor().cuda(),
        kinds,
        "initial",
    ).cpu()

    torch.testing.assert_close(
        cuda[:, PROXY_FEATURE_OFFSET + 1],
        cpu[:, PROXY_FEATURE_OFFSET + 1],
        rtol=0.0,
        atol=0.0,
    )


def _case():
    return from_official(
        4,
        torch.ones(4),
        [],
        [],
        [],
        torch.tensor(
            [
                [0, 0, 1, 1, 1],
                [0, 0, 1, 1, 0],
                [0, 0, 0, 0, 2],
                [0, 0, 0, 0, 0],
            ],
            dtype=torch.long,
        ),
    )


def _anchor() -> torch.Tensor:
    return torch.tensor(
        [
            [0.0, 0.0, 1.0, 1.0],
            [1.0, 0.0, 1.0, 1.0],
            [3.0, 0.0, 1.0, 1.0],
            [5.0, 0.0, 1.0, 1.0],
        ],
        dtype=torch.float32,
    )


def _candidates() -> tuple[torch.Tensor, torch.Tensor]:
    good = _anchor()
    bad = torch.tensor(
        [
            [0.5, 0.0, 1.0, 2.0],
            [3.0, 0.0, 2.0, 1.0],
            [2.5, 0.0, 1.0, 1.0],
            [5.0, 0.0, 1.0, 1.0],
        ],
        dtype=torch.float32,
    )
    raw = torch.stack((good, good))
    post = torch.stack((good, bad))
    return raw, post


def _three_candidates() -> tuple[torch.Tensor, torch.Tensor]:
    raw, post = _candidates()
    third = post[1].clone()
    third[:, 1] += 0.25
    return torch.cat((raw, raw[:1]), dim=0), torch.cat((post, third.unsqueeze(0)), dim=0)
