from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch

from hcfp.checkpoint import (
    RUNTIME_NORMALIZATION,
    _payload_hash,
    load_checkpoint,
    save_checkpoint,
)
from hcfp.data import DataSample, extract_labels, write_shard
from hcfp.dynamics import DynamicsConfig
from hcfp.fallback import safe_shelf
from hcfp.learned import (
    LearnedConfig,
    _learned_population,
    _tensor_sha256,
    _topology_seed_candidates,
    analyze_case_with_checkpoint,
)
from hcfp.model import HCFPModel, ModelConfig
from hcfp.topology import decode_sequence_pair, relation_mask_from_rectangles
from hcfp.profile import synthetic_case
from hcfp.training import supervised_loss


def _case():
    from hcfp.case import from_official

    return from_official(
        4,
        [4.0, 9.0, 16.0, 25.0],
        [[0, 1, 2.0], [1, 2, 3.0]],
        [],
        [],
        [
            [0, 1, 0, 1, 0],
            [1, 0, 0, 1, 0],
            [0, 0, 7, 0, 0],
            [0, 0, 7, 0, 0],
        ],
        [
            [0.0, 0.0, 2.0, 2.0],
            [4.0, 0.0, 3.0, 3.0],
            [-1.0, -1.0, -1.0, -1.0],
            [-1.0, -1.0, -1.0, -1.0],
        ],
    )


def _sample() -> DataSample:
    case = synthetic_case(32, device="cpu")
    return DataSample(
        "train-topology",
        case,
        extract_labels(case, safe_shelf(case), normalized=True),
    )


def _catalog_topology(provenance: dict[str, object], record: dict[str, object]):
    catalog = provenance["topology_order_catalog"]
    assert isinstance(catalog, dict)
    order_hash = str(record["topology_order_sha256"])
    entry = catalog[order_hash]
    assert isinstance(entry, dict)
    assert record["topology_edge_sha256"] == entry["topology_edge_sha256"]

    orders = torch.tensor(
        (entry["positive_order"], entry["negative_order"]),
        dtype=torch.long,
    )
    order_bytes = orders.contiguous().view(torch.uint8).reshape(-1)
    assert order_hash == hashlib.sha256(bytes(order_bytes.tolist())).hexdigest()
    topology = decode_sequence_pair(orders[0], orders[1])
    horizontal_edges = tuple(
        tuple(int(value) for value in edge)
        for edge in topology.horizontal_edges.tolist()
    )
    vertical_edges = tuple(
        tuple(int(value) for value in edge) for edge in topology.vertical_edges.tolist()
    )
    assert entry["horizontal_edges"] == horizontal_edges
    assert entry["vertical_edges"] == vertical_edges
    assert (
        entry["topology_edge_sha256"]
        == hashlib.sha256(
            repr((horizontal_edges, vertical_edges)).encode("ascii")
        ).hexdigest()
    )
    return entry, topology


def test_default_model_keeps_legacy_modules_and_checkpoint_payloads_load(
    tmp_path: Path,
) -> None:
    model = HCFPModel(ModelConfig(hidden_dim=16, encoder_layers=1))
    assert not hasattr(model, "topology")
    assert not any(name.startswith("topology.") for name in model.state_dict())

    checkpoint = tmp_path / "legacy.pt"
    save_checkpoint(model, checkpoint, RUNTIME_NORMALIZATION)
    payload = torch.load(checkpoint, weights_only=True)
    payload["config"].pop("topology_enabled")
    payload["state_hash"] = _payload_hash(payload)
    torch.save(payload, checkpoint)

    loaded, _ = load_checkpoint(
        checkpoint,
        expected_config=model.config,
        expected_normalization=RUNTIME_NORMALIZATION,
    )
    assert loaded.config.topology_enabled is False
    assert loaded.state_dict().keys() == model.state_dict().keys()


def test_topology_model_exposes_soft_permutations_and_membership_messages() -> None:
    torch.manual_seed(3)
    case = _case()
    model = HCFPModel(
        ModelConfig(hidden_dim=16, encoder_layers=1, topology_enabled=True)
    )
    output = model(case, population=2)

    assert output.positive_permutation is not None
    assert output.negative_permutation is not None
    for permutation in (output.positive_permutation, output.negative_permutation):
        assert permutation.shape == (case.n, case.n)
        assert torch.allclose(permutation.sum(dim=0), torch.ones(case.n), atol=2.0e-4)
        assert torch.allclose(permutation.sum(dim=1), torch.ones(case.n), atol=2.0e-4)

    shared = torch.tensor([[True, True, False, False]])
    split = torch.tensor([[True, False, False, False], [False, True, False, False]])
    common = {
        "b2b_weight": torch.zeros_like(case.b2b_weight),
        "mib_membership": torch.empty((0, case.n), dtype=torch.bool),
    }
    shared_case = replace(case, group_membership=shared, **common)
    split_case = replace(case, group_membership=split, **common)
    shared_embedding = model.encoder(shared_case)
    split_embedding = model.encoder(split_case)

    assert not torch.allclose(shared_embedding[:2], split_embedding[:2])


def test_structure_loss_uses_set_labels_and_trains_dual_permutations() -> None:
    torch.manual_seed(5)
    sample = _sample()
    model = HCFPModel(
        ModelConfig(hidden_dim=16, encoder_layers=1, topology_enabled=True)
    )
    original = supervised_loss(model, sample, population=2, stage="structure")
    ignored_legacy_labels = replace(
        sample.labels,
        pairwise_precedence=torch.zeros_like(sample.labels.pairwise_precedence),
        precedence_tie_mask=torch.ones_like(sample.labels.precedence_tie_mask),
    )
    changed = supervised_loss(
        model,
        DataSample(sample.sample_id, sample.case, ignored_legacy_labels),
        population=2,
        stage="structure",
    )

    assert torch.equal(original.structure, changed.structure)
    original.total.backward()
    gradients = [parameter.grad for parameter in model.topology.parameters()]
    assert all(
        gradient is not None and torch.isfinite(gradient).all()
        for gradient in gradients
    )
    assert (
        sum(
            float(gradient.abs().sum())
            for gradient in gradients
            if gradient is not None
        )
        > 0.0
    )


def test_opt_in_topology_seed_changes_geometry_and_records_provenance(
    tmp_path: Path,
) -> None:
    torch.manual_seed(7)
    case = _case()
    checkpoint = tmp_path / "topology.pt"
    save_checkpoint(
        HCFPModel(ModelConfig(hidden_dim=16, encoder_layers=1, topology_enabled=True)),
        checkpoint,
        RUNTIME_NORMALIZATION,
    )
    config = LearnedConfig(
        analytic=replace(
            LearnedConfig().analytic,
            dynamics=DynamicsConfig(population=2, steps=0),
            projection_iterations=4,
            direction_beam=1,
        ),
        flow_steps=0,
        topology_seeds=1,
        seed=11,
    )

    analysis = analyze_case_with_checkpoint(case, checkpoint, config)

    assert analysis.result.topology_seed_attempted is True
    assert analysis.result.topology_seed_accepted is True
    assert analysis.result.topology_seed_count == 1
    snapshot = analysis.analytic.incumbent_snapshot
    assert snapshot["topology_seed_attempted"] is True
    assert snapshot["topology_seed_accepted"] is True
    assert snapshot["topology_seed_count"] == 1
    sources = snapshot["topology_seed_sources"]
    assert len(sources) == 2
    assert len(snapshot["topology_soft_permutation_sha256"]) == 64
    assert len(snapshot["topology_precedence_logits_sha256"]) == 64
    catalog = snapshot["topology_order_catalog"]
    adapted = catalog[snapshot["topology_order_sha256"]]
    assert snapshot["topology_edge_sha256"] == adapted["topology_edge_sha256"]
    repeated_order_fields = {
        "positive_order",
        "negative_order",
        "horizontal_edges",
        "vertical_edges",
    }
    records = snapshot["topology_seed_provenance"]
    record = records[0]
    assert record["candidate_type"] == "topology"
    assert record["stage"] == "initial"
    assert record["source"] == sources[0]
    assert record["order_variant"]
    assert 0 <= record["aspect_source_index"] < config.analytic.dynamics.population
    assert record["status"] == "selected"
    assert record["selection_rank"] == 0
    assert snapshot["topology_seed_pool_size"] == len(snapshot["topology_seed_pool"])
    assert snapshot["topology_seed_pool"][record["pool_index"]] == {
        key: value
        for key, value in record.items()
        if key not in {"source", "candidate_type", "stage"}
    }
    assert snapshot["topology_selection_reference"]["source"] == "safe_shelf"
    for candidate_record in records:
        assert repeated_order_fields.isdisjoint(candidate_record)
        entry, topology = _catalog_topology(snapshot, candidate_record)
        assert entry["order_variant"] == candidate_record["order_variant"]
        index = int(str(candidate_record["source"]).removeprefix("candidate_"))
        topology_seed = analysis.analytic.raw_candidates[index]
        assert torch.equal(
            topology_seed[case.preplaced_mask], case.target[case.preplaced_mask]
        )
        realized = relation_mask_from_rectangles(topology_seed)
        first, second = torch.where(~torch.eye(case.n, dtype=torch.bool))
        assert realized[first, second, topology.relation[first, second]].all()
    initial = int(str(sources[0]).removeprefix("candidate_"))
    topology_seed = analysis.analytic.raw_candidates[initial]
    assert not any(
        torch.allclose(topology_seed, candidate)
        for candidate in analysis.analytic.raw_candidates[3:initial]
    )


def test_opt_in_constraint_seeds_follow_topology_and_keep_source_provenance(
    tmp_path: Path,
) -> None:
    torch.manual_seed(19)
    case = _case()
    checkpoint = tmp_path / "topology-constraints.pt"
    save_checkpoint(
        HCFPModel(
            ModelConfig(hidden_dim=16, encoder_layers=1, topology_enabled=True)
        ),
        checkpoint,
        RUNTIME_NORMALIZATION,
    )
    config = LearnedConfig(
        analytic=replace(
            LearnedConfig().analytic,
            dynamics=DynamicsConfig(population=2, steps=0),
            projection_iterations=4,
            direction_beam=1,
        ),
        flow_steps=0,
        topology_seeds=2,
        constraint_seeds=2,
        seed=23,
    )

    analysis = analyze_case_with_checkpoint(case, checkpoint, config)

    assert analysis.result.used_checkpoint
    assert analysis.result.topology_seed_count == 2
    assert analysis.result.constraint_seed_attempted
    assert analysis.result.constraint_seed_accepted
    assert analysis.result.constraint_seed_count == 2
    snapshot = analysis.analytic.incumbent_snapshot
    assert len(snapshot["topology_seed_sources"]) == 4
    assert len(snapshot["constraint_seed_sources"]) == 4
    assert len(snapshot["constraint_seed_provenance"]) == 4
    initial_constraint = int(
        str(snapshot["constraint_seed_sources"][0]).removeprefix("candidate_")
    )
    initial_topology = int(
        str(snapshot["topology_seed_sources"][0]).removeprefix("candidate_")
    )
    assert initial_constraint < initial_topology
    assert snapshot["constraint_seed_kind_counts"]


def test_post_relax_constraint_provenance_tracks_derived_geometry(
    tmp_path: Path,
) -> None:
    torch.manual_seed(29)
    case = _case()
    checkpoint = tmp_path / "topology-constraints-relaxed.pt"
    save_checkpoint(
        HCFPModel(
            ModelConfig(hidden_dim=16, encoder_layers=1, topology_enabled=True)
        ),
        checkpoint,
        RUNTIME_NORMALIZATION,
    )
    config = LearnedConfig(
        analytic=replace(
            LearnedConfig().analytic,
            dynamics=DynamicsConfig(population=2, steps=1),
            projection_iterations=4,
            direction_beam=1,
        ),
        flow_steps=0,
        topology_seeds=1,
        constraint_seeds=1,
        seed=31,
    )

    analysis = analyze_case_with_checkpoint(case, checkpoint, config)
    snapshot = analysis.analytic.incumbent_snapshot

    sources = tuple(snapshot["constraint_seed_sources"])
    records = tuple(snapshot["constraint_seed_provenance"])
    assert len(sources) == 2
    assert len(records) == 2
    assert "constraint_seed_stale_sources" not in snapshot
    assert tuple(record["source"] for record in records) == sources
    initial, post_relax = records
    assert initial["stage"] == "initial"
    assert post_relax["stage"] == "post_relax"
    assert post_relax["transform"] == "population_relaxation"
    assert post_relax["parent_candidate_sha256"] == initial["candidate_sha256"]
    for record in records:
        index = int(str(record["source"]).removeprefix("candidate_"))
        assert record["candidate_sha256"] == _tensor_sha256(
            analysis.analytic.raw_candidates[index]
        )


def test_constraint_seeds_require_topology_seeds() -> None:
    with pytest.raises(ValueError, match="require topology"):
        LearnedConfig(constraint_seeds=1)


def test_topology_seed_failure_returns_unmodified_learned_population(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import hcfp.learned as learned

    torch.manual_seed(13)
    case = _case()
    model = HCFPModel(
        ModelConfig(hidden_dim=16, encoder_layers=1, topology_enabled=True)
    )
    base_config = LearnedConfig(
        analytic=replace(
            LearnedConfig().analytic,
            dynamics=DynamicsConfig(population=2, steps=0),
        ),
        flow_steps=0,
        seed=17,
    )
    baseline = _learned_population(case, model, base_config, seed=17)
    monkeypatch.setattr(
        learned,
        "pack_sequence_pair_with_anchors",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(ValueError("decode failed")),
    )
    provenance: dict[str, object] = {}

    attempted = _learned_population(
        case,
        model,
        replace(base_config, topology_seeds=1),
        seed=17,
        provenance=provenance,
    )

    assert torch.equal(attempted, baseline)
    assert provenance["topology_seed_attempted"] is True
    assert provenance["topology_seed_accepted"] is False
    assert provenance["topology_seed_count"] == 0
    assert "decode failed" in str(provenance["topology_seed_failure_reason"])


def test_topology_seed_uses_actual_safe_order_and_aspect_provenance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import hcfp.learned as learned
    from hcfp.case import from_official

    positive_order = torch.tensor((0, 1, 2, 3, 4, 5))
    negative_order = torch.tensor((0, 1, 2, 5, 4, 3))
    constraints = torch.zeros((6, 5), dtype=torch.long)
    constraints[[0, 2, 3, 5], 1] = 1
    targets = torch.full((6, 4), -1.0)
    targets[0] = torch.tensor((0.0, 0.0, 1.0, 1.0))
    targets[2] = torch.tensor((1.0, 0.0, 1.0, 1.0))
    targets[3] = torch.tensor((2.0, 1.0, 1.0, 1.0))
    targets[5] = torch.tensor((2.0, 0.0, 1.0, 1.0))
    case = from_official(6, torch.ones(6), [], [], [], constraints, targets)

    def assignment(order: torch.Tensor) -> torch.Tensor:
        matrix = torch.zeros((6, 6))
        matrix[order, torch.arange(6)] = 1.0
        return matrix

    output = SimpleNamespace(
        positive_permutation=assignment(positive_order),
        negative_permutation=assignment(negative_order),
        precedence_logits=torch.zeros((6, 6, 5)),
    )
    shelf = safe_shelf(case)
    movable = ~case.preplaced_mask
    aspects = []
    for factor in (0.75, 0.90, 1.00, 1.10, 1.25, 1.50):
        candidate = shelf.clone()
        candidate[movable, 2] *= factor
        candidate[movable, 3] /= factor
        aspects.append(candidate)
    source = torch.stack(aspects)
    late_width = float(source[-1, 1, 2])

    def raw_soft_priority(_case, candidate):
        return SimpleNamespace(
            raw_total=0 if float(candidate[1, 2]) == late_width else 1
        )

    monkeypatch.setattr(learned, "soft_violation_normalized", raw_soft_priority)
    first_provenance: dict[str, object] = {}
    second_provenance: dict[str, object] = {}

    first = _topology_seed_candidates(
        case,
        output,
        source,
        count=2,
        provenance=first_provenance,
    )
    second = _topology_seed_candidates(
        case,
        output,
        source,
        count=2,
        provenance=second_provenance,
    )

    assert torch.equal(first, second)
    first_bytes = first.contiguous().view(torch.uint8).reshape(-1)
    second_bytes = second.contiguous().view(torch.uint8).reshape(-1)
    assert (
        hashlib.sha256(bytes(first_bytes.tolist())).digest()
        == hashlib.sha256(bytes(second_bytes.tolist())).digest()
    )
    assert first_provenance == second_provenance
    attempts = first_provenance["topology_order_attempts"]
    assert attempts[0]["order_variant"] == "adapted"
    assert attempts[0]["status"] == "rejected"
    pool = first_provenance["topology_seed_pool"]
    assert first_provenance["topology_seed_pool_size"] == len(pool)
    assert len(pool) > 2
    assert sum(record["status"] == "selected" for record in pool) == 2
    assert any(record["status"] == "rejected_by_budget" for record in pool)
    assert {
        record["selection_rank"] for record in pool if record["status"] == "selected"
    } == {
        0,
        1,
    }
    assert all(
        record["selection_rank"] is None
        for record in pool
        if record["status"] == "rejected_by_budget"
    )
    assert all("priority" in record and "priority_rank" in record for record in pool)
    records = first_provenance["topology_seed_orders"]
    assert len(records) == 2
    assert all(record["aspect_source_index"] == 5 for record in records)
    assert all(record["priority"]["raw_soft_violation"] == 0 for record in records)
    assert any(
        record["aspect_source_index"] < 5
        and record["priority"]["raw_soft_violation"] == 1
        for record in pool
    )
    assert len({record["topology_order_sha256"] for record in records}) == 2
    assert len({record["topology_edge_sha256"] for record in records}) == 2
    repeated_order_fields = {
        "positive_order",
        "negative_order",
        "horizontal_edges",
        "vertical_edges",
    }
    assert all(repeated_order_fields.isdisjoint(record) for record in pool)
    assert all(repeated_order_fields.isdisjoint(record) for record in records)
    catalog = first_provenance["topology_order_catalog"]
    assert set(record["topology_order_sha256"] for record in pool) <= set(catalog)
    assert len(catalog) == len(
        {
            hashlib.sha256(
                bytes(
                    torch.tensor(
                        (entry["positive_order"], entry["negative_order"]),
                        dtype=torch.long,
                    )
                    .contiguous()
                    .view(torch.uint8)
                    .reshape(-1)
                    .tolist()
                )
            ).hexdigest()
            for entry in catalog.values()
        }
    )
    assert len(catalog) < len(pool)
    assert (
        len(json.dumps(first_provenance, sort_keys=True, separators=(",", ":")))
        < 25_000
    )
    for candidate, record in zip(first, records, strict=True):
        assert record == pool[record["pool_index"]]
        assert record["order_variant"] != "adapted"
        entry, topology = _catalog_topology(first_provenance, record)
        assert entry["order_variant"] == record["order_variant"]
        realized = relation_mask_from_rectangles(candidate)
        pair = ~torch.eye(case.n, dtype=torch.bool)
        assert realized.gather(-1, topology.relation.clamp_min(0).unsqueeze(-1))[
            pair
        ].all()
        assert torch.equal(
            candidate[case.preplaced_mask], case.target[case.preplaced_mask]
        )


def test_training_cli_reports_explicit_topology_flag(tmp_path: Path) -> None:
    shard = tmp_path / "train.tar"
    checkpoint = tmp_path / "model.pt"
    write_shard(
        [_sample()],
        shard,
        provenance={
            "source": "FloorSet-train",
            "source_version": "fixture-v1",
            "split": "train",
            "denylist_sha256": "fixture-denylist",
        },
    )

    subprocess.run(
        [
            sys.executable,
            "scripts/train_hcfp.py",
            str(shard),
            "-o",
            str(checkpoint),
            "--steps",
            "1",
            "--population",
            "2",
            "--hidden-dim",
            "16",
            "--encoder-layers",
            "1",
            "--device",
            "cpu",
            "--amp",
            "off",
            "--topology",
        ],
        cwd=Path(__file__).resolve().parents[1],
        check=True,
        text=True,
        capture_output=True,
    )

    report = json.loads(Path(f"{checkpoint}.training.json").read_text(encoding="utf-8"))
    assert report["schema_version"] == 3
    assert report["command"][0] == "scripts/train_hcfp.py"
    assert report["seed"] == 0
    loaded, _ = load_checkpoint(
        checkpoint, expected_normalization=RUNTIME_NORMALIZATION
    )
    assert report["topology_enabled"] is True
    assert report["model_config"]["topology_enabled"] is True
    assert report["direct_floorset_lite_stream"] is None
    assert loaded.config.topology_enabled is True
