from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from hcfp.case import from_official
from hcfp.checkpoint import RUNTIME_NORMALIZATION, save_checkpoint
from hcfp.data import DataSample, extract_labels
from hcfp.fallback import safe_shelf
from hcfp.model import HCFPModel, ModelConfig


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/audit_hcfp_topology_heldout.py"
SPEC = importlib.util.spec_from_file_location("topology_heldout_audit", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit)
TRAIN_SCRIPT = ROOT / "scripts/train_hcfp.py"
TRAIN_SPEC = importlib.util.spec_from_file_location(
    "train_hcfp_audit_test", TRAIN_SCRIPT
)
assert TRAIN_SPEC is not None and TRAIN_SPEC.loader is not None
train_cli = importlib.util.module_from_spec(TRAIN_SPEC)
TRAIN_SPEC.loader.exec_module(train_cli)


def _sample(sample_id: str, block_count: int = 2) -> DataSample:
    rectangles = torch.tensor(
        [[float(index), 0.0, 1.0, 1.0] for index in range(block_count)]
    )
    case = from_official(
        block_count,
        torch.ones(block_count),
        [],
        [],
        [],
        torch.zeros((block_count, 5), dtype=torch.long),
    )
    return DataSample(
        sample_id,
        case,
        extract_labels(
            case,
            rectangles,
            baseline_area=float(block_count),
            baseline_hpwl=0.0,
        ),
    )


def _training_report_payload(
    root: Path,
    checkpoint: Path,
    sample_ids: list[str],
    *,
    seed: int = 10,
    source_limit: int | None = 2,
    checkpoint_hash: str = "state-hash",
    model_config: dict[str, object] | None = None,
    checkpoint_metadata: dict[str, object] | None = None,
    parent_training_report: dict[str, object] | None = None,
    schema_version: int = 3,
) -> dict[str, object]:
    unique_ids = sorted(set(sample_ids))
    return {
        "schema_version": schema_version,
        "command": ["scripts/train_hcfp.py", "--floorset-lite-root", str(root)],
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_hash": checkpoint_hash,
        "checkpoint_metadata": checkpoint_metadata
        or {
            "capabilities": {"flow": False},
            "parent_state_hash": None,
            "trained_heads": [],
            "training_objective_version": "supervised_loss_v1",
        },
        "model_config": model_config or {"hidden_dim": 16},
        "steps": len(sample_ids),
        "direct_floorset_lite_stream": {
            "root": str(root.resolve()),
            "sampling": "score-aware",
            "seed": seed,
            "source_limit": source_limit,
            "max_layouts_per_file": None,
            "consumed_count": len(sample_ids),
            "ordered_sample_id_count": len(sample_ids),
            "ordered_sample_id_sha256": audit._sample_id_hash(sample_ids),
            "unique_sample_id_count": len(unique_ids),
            "unique_sample_id_sha256": audit._sample_id_hash(unique_ids),
            "checkpoint_hash": checkpoint_hash,
        },
        "parent_training_report": parent_training_report,
    }


def test_heldout_sampling_excludes_training_ids_and_filters_block_range(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    excluded = [_sample("train/a", 3), _sample("train/b", 4)]
    stream = [
        (_sample("train/a", 3), {}),
        (_sample("heldout/small", 1), {}),
        (_sample("heldout/keep-5", 5), {}),
        (_sample("heldout/large", 7), {}),
        (_sample("heldout/keep-4", 4), {}),
    ]

    def heldout_iterator(_root, **kwargs):
        assert kwargs == {
            "limit": None,
            "seed": 11,
            "score_aware": True,
            "max_layouts_per_file": 1,
        }
        return iter(stream)

    monkeypatch.setattr(audit, "iter_floorset_lite_with_source", heldout_iterator)

    excluded_ids = {sample.sample_id for sample in excluded}
    expected_hash = audit._sample_id_hash(sorted(excluded_ids))
    exclude_provenance = {
        "count": len(excluded_ids),
        "sample_id_sha256": expected_hash,
    }

    selected, provenance = audit._collect_heldout(
        "training-root",
        exclude_ids=excluded_ids,
        exclude_provenance=exclude_provenance,
        heldout_limit=2,
        heldout_seed=11,
        heldout_max_layouts_per_file=1,
        min_blocks=2,
        max_blocks=5,
        score_aware=True,
    )

    selected_ids = [sample.sample_id for sample, _source in selected]
    assert selected_ids == ["heldout/keep-5", "heldout/keep-4"]
    assert not set(selected_ids).intersection(sample.sample_id for sample in excluded)
    assert [sample.case.n for sample, _source in selected] == [5, 4]
    assert provenance["exclude_training"]["count"] == 2
    assert provenance["heldout"]["overlap_filtered_count"] == 1
    assert provenance["heldout"]["block_filtered_count"] == 2
    assert provenance["heldout"]["source_file_count"] == 2
    assert provenance["exclude_training"]["sample_id_sha256"] == expected_hash


def test_heldout_sampling_fails_closed_when_filtered_stream_is_short(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        audit,
        "iter_floorset_lite_with_source",
        lambda *_args, **_kwargs: iter(((_sample("train/a", 2), {}),)),
    )

    with pytest.raises(RuntimeError, match="0 disjoint in-range samples, expected 1"):
        audit._collect_heldout(
            "training-root",
            exclude_ids={"train/a"},
            exclude_provenance={"count": 1},
            heldout_limit=1,
            heldout_seed=11,
            heldout_max_layouts_per_file=1,
            min_blocks=2,
            max_blocks=5,
            score_aware=True,
        )


def test_consumed_stream_reconstruction_cycles_deterministically(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "training"
    samples = (_sample("train/a"), _sample("train/b"))
    calls: list[tuple[Path, dict[str, object]]] = []

    def sample_iterator(path, **kwargs):
        calls.append((Path(path), kwargs))
        return iter(samples)

    monkeypatch.setattr(audit, "iter_floorset_lite", sample_iterator)

    expected = ["train/a", "train/b", "train/a", "train/b", "train/a"]
    first = audit._reconstruct_consumed_sample_ids(
        root,
        source_limit=2,
        seed=10,
        score_aware=True,
        consumed_count=5,
    )
    second = audit._reconstruct_consumed_sample_ids(
        root,
        source_limit=2,
        seed=10,
        score_aware=True,
        consumed_count=5,
    )

    assert first == expected
    assert second == expected
    assert calls == [(root, {"limit": 2, "seed": 10, "score_aware": True})] * 6


def test_training_cli_reports_exact_consumed_direct_stream(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "training"
    checkpoint = tmp_path / "model.pt"
    samples = (_sample("train/a"), _sample("train/b"))
    pulled_ids: list[str] = []

    def sample_iterator(path, **kwargs):
        assert Path(path).resolve() == root.resolve()
        assert kwargs == {
            "limit": 2,
            "seed": 10,
            "score_aware": True,
            "min_blocks": 1,
            "max_blocks": 120,
        }
        return iter(samples)

    def fake_train_steps(_model, sample_factory, _optimizer, *, steps, **_kwargs):
        iterator = iter(sample_factory())
        history = []
        for _index in range(steps):
            try:
                sample = next(iterator)
            except StopIteration:
                iterator = iter(sample_factory())
                sample = next(iterator)
            pulled_ids.append(sample.sample_id)
            history.append({"total": 1.0})
        return history

    monkeypatch.setattr(train_cli, "iter_floorset_lite", sample_iterator)
    monkeypatch.setattr(train_cli, "train_steps", fake_train_steps)
    monkeypatch.setattr(train_cli, "save_checkpoint", lambda *_args, **_kwargs: "state-hash")
    argv = [
        "--floorset-lite-root",
        str(root),
        "--sample-limit",
        "2",
        "-o",
        str(checkpoint),
        "--steps",
        "5",
        "--population",
        "1",
        "--hidden-dim",
        "16",
        "--encoder-layers",
        "1",
        "--device",
        "cpu",
        "--amp",
        "off",
        "--ema-decay",
        "0",
        "--seed",
        "10",
    ]

    assert train_cli.main(argv) == 0
    first = Path(f"{checkpoint}.training.json").read_bytes()
    assert train_cli.main(argv) == 0
    second = Path(f"{checkpoint}.training.json").read_bytes()

    assert first == second
    assert pulled_ids == ["train/a", "train/b", "train/a", "train/b", "train/a"] * 2
    report = json.loads(first)
    contract = report["direct_floorset_lite_stream"]
    consumed_ids = ["train/a", "train/b", "train/a", "train/b", "train/a"]
    assert report["schema_version"] == 3
    assert report["command"] == ["scripts/train_hcfp.py", *argv]
    assert report["checkpoint"] == str(checkpoint.resolve())
    assert report["checkpoint_hash"] == "state-hash"
    assert report["model_config"]["hidden_dim"] == 16
    assert report["parent_training_report"] is None
    assert contract == {
        "checkpoint_hash": "state-hash",
        "consumed_count": 5,
        "max_layouts_per_file": None,
        "ordered_sample_id_count": 5,
        "ordered_sample_id_sha256": audit._sample_id_hash(consumed_ids),
        "root": str(root.resolve()),
        "sampling": "score-aware",
        "seed": 10,
        "source_limit": 2,
        "unique_sample_id_count": 2,
        "unique_sample_id_sha256": audit._sample_id_hash(["train/a", "train/b"]),
    }


def test_training_cli_records_parent_report_lineage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "training"
    parent = tmp_path / "q2.pt"
    parent_hash = save_checkpoint(
        HCFPModel(ModelConfig(hidden_dim=16, encoder_layers=1)),
        parent,
        RUNTIME_NORMALIZATION,
    )
    parent_report = Path(f"{parent}.training.json")
    parent_report.write_text(
        json.dumps(
            _training_report_payload(
                root,
                parent,
                ["parent/a"],
                source_limit=1,
                checkpoint_hash=parent_hash,
                model_config=load_checkpoint_config(parent),
            ),
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    samples = (_sample("active/a"),)

    monkeypatch.setattr(
        train_cli,
        "iter_floorset_lite",
        lambda *_args, **_kwargs: iter(samples),
    )
    monkeypatch.setattr(
        train_cli,
        "train_steps",
        lambda _model, sample_factory, _optimizer, *, steps, **_kwargs: [
            {"total": 1.0}
            for _sample in [next(iter(sample_factory()))]
            for _index in range(steps)
        ],
    )
    monkeypatch.setattr(train_cli, "save_checkpoint", lambda *_args, **_kwargs: "q3-hash")
    output = tmp_path / "q3.pt"

    assert train_cli.main(
        [
            "--floorset-lite-root",
            str(root),
            "--sample-limit",
            "1",
            "--output",
            str(output),
            "--steps",
            "1",
            "--population",
            "1",
            "--init-checkpoint",
            str(parent),
            "--device",
            "cpu",
            "--amp",
            "off",
            "--ema-decay",
            "0",
        ]
    ) == 0

    report = json.loads(Path(f"{output}.training.json").read_text(encoding="utf-8"))
    assert report["parent_training_report"] == {
        "checkpoint_hash": parent_hash,
        "path": str(parent_report.resolve()),
        "sha256": audit.file_sha256(parent_report),
    }


def load_checkpoint_config(path: Path) -> dict[str, object]:
    _model, metadata = train_cli.load_checkpoint(
        path,
        expected_normalization=RUNTIME_NORMALIZATION,
    )
    return metadata["config"]


def test_training_report_reconstructs_exact_unique_exclusion(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "training"
    checkpoint = tmp_path / "model.pt"
    training_report = Path(f"{checkpoint}.training.json")
    consumed_ids = ["train/a", "train/b", "train/a"]
    payload = _training_report_payload(root, checkpoint, consumed_ids)
    training_report.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    samples = (_sample("train/a"), _sample("train/b"))
    monkeypatch.setattr(
        audit,
        "iter_floorset_lite",
        lambda *_args, **_kwargs: iter(samples),
    )

    excluded, provenance, contract = audit._load_training_exclusion(
        training_report,
        root=root,
        checkpoint=checkpoint,
        checkpoint_hash="state-hash",
        checkpoint_config={"hidden_dim": 16},
        asserted_seed=10,
        asserted_limit=2,
        asserted_sampling="score-aware",
    )

    assert excluded == {"train/a", "train/b"}
    assert provenance["consumed_count"] == 3
    assert provenance["ordered_sample_id_sha256"] == audit._sample_id_hash(consumed_ids)
    assert provenance["unique_sample_id_sha256"] == audit._sample_id_hash(
        ["train/a", "train/b"]
    )
    assert provenance["training_report"] == str(training_report.resolve())
    assert contract == payload["direct_floorset_lite_stream"]


def test_training_report_exclusion_unions_parent_lineage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "training"
    parent = tmp_path / "q2.pt"
    parent_hash = save_checkpoint(
        HCFPModel(ModelConfig(hidden_dim=16, encoder_layers=1)),
        parent,
        RUNTIME_NORMALIZATION,
    )
    parent_model_config = load_checkpoint_config(parent)
    parent_report = Path(f"{parent}.training.json")
    parent_payload = _training_report_payload(
        root,
        parent,
        ["parent/a", "parent/b", "parent/a"],
        seed=10,
                source_limit=2,
                checkpoint_hash=parent_hash,
                model_config=parent_model_config,
                schema_version=2,
            )
    parent_report.write_text(
        json.dumps(parent_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    active = tmp_path / "q3.pt"
    active_payload = _training_report_payload(
        root,
        active,
        ["active/a", "parent/b"],
        seed=20,
        source_limit=2,
        checkpoint_hash="q3-hash",
        model_config={"hidden_dim": 16, "collective_enabled": True},
        checkpoint_metadata={
            "capabilities": {"collective": True, "flow": False},
            "parent_state_hash": parent_hash,
            "trained_heads": ["collective"],
            "training_objective_version": "collective_rollout_v1",
        },
        parent_training_report={
            "path": str(parent_report.resolve()),
            "sha256": audit.file_sha256(parent_report),
            "checkpoint_hash": parent_hash,
        },
    )
    active_report = Path(f"{active}.training.json")
    active_report.write_text(
        json.dumps(active_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    def sample_iterator(_root, **kwargs):
        if kwargs["seed"] == 10:
            return iter((_sample("parent/a"), _sample("parent/b")))
        if kwargs["seed"] == 20:
            return iter((_sample("active/a"), _sample("parent/b")))
        raise AssertionError(kwargs)

    monkeypatch.setattr(audit, "iter_floorset_lite", sample_iterator)

    excluded, provenance, contract = audit._load_training_exclusion(
        active_report,
        root=root,
        checkpoint=active,
        checkpoint_hash="q3-hash",
        checkpoint_config={"hidden_dim": 16, "collective_enabled": True},
        asserted_seed=20,
        asserted_limit=2,
        asserted_sampling="score-aware",
    )

    assert contract == active_payload["direct_floorset_lite_stream"]
    assert excluded == {"active/a", "parent/a", "parent/b"}
    assert provenance["consumed_count"] == 2
    assert provenance["count"] == 3
    assert provenance["active_unique_sample_id_count"] == 2
    assert provenance["ancestor_unique_sample_id_count"] == 2
    assert provenance["lineage_report_count"] == 2
    assert provenance["ancestor_reports"][0]["checkpoint_hash"] == parent_hash


def test_training_report_parent_lineage_tamper_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "training"
    parent = tmp_path / "q2.pt"
    parent_hash = save_checkpoint(
        HCFPModel(ModelConfig(hidden_dim=16, encoder_layers=1)),
        parent,
        RUNTIME_NORMALIZATION,
    )
    parent_report = Path(f"{parent}.training.json")
    parent_report.write_text(
        json.dumps(
            _training_report_payload(
                root,
                parent,
                ["parent/a"],
                source_limit=1,
                checkpoint_hash=parent_hash,
                model_config=load_checkpoint_config(parent),
                schema_version=2,
            ),
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    active = tmp_path / "q3.pt"
    active_report = Path(f"{active}.training.json")
    payload = _training_report_payload(
        root,
        active,
        ["active/a"],
        seed=20,
        source_limit=1,
        checkpoint_hash="q3-hash",
        model_config={"hidden_dim": 16, "collective_enabled": True},
        checkpoint_metadata={
            "capabilities": {"collective": True, "flow": False},
            "parent_state_hash": parent_hash,
            "trained_heads": ["collective"],
            "training_objective_version": "collective_rollout_v1",
        },
        parent_training_report={
            "path": str(parent_report.resolve()),
            "sha256": "tampered",
            "checkpoint_hash": parent_hash,
        },
    )
    active_report.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        audit,
        "iter_floorset_lite",
        lambda *_args, **_kwargs: iter((_sample("active/a"),)),
    )

    with pytest.raises(ValueError, match="parent training report sha256 mismatch"):
        audit._load_training_exclusion(
            active_report,
            root=root,
            checkpoint=active,
            checkpoint_hash="q3-hash",
            checkpoint_config={"hidden_dim": 16, "collective_enabled": True},
        )


@pytest.mark.parametrize(
    ("mismatch", "message"),
    (
        ("hash", "ordered sample ID hash mismatch"),
        ("seed", "manual exclude seed"),
        ("root", "root mismatch"),
        ("checkpoint", "checkpoint hash mismatch"),
        ("config", "model config mismatch"),
        ("sampling", "manual sampling mode"),
        ("count", "consumed count mismatch"),
    ),
)
def test_training_report_mismatch_fails_closed(
    mismatch: str,
    message: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "training"
    checkpoint = tmp_path / "model.pt"
    training_report = tmp_path / "explicit-training.json"
    payload = _training_report_payload(root, checkpoint, ["train/a", "train/b"])
    contract = payload["direct_floorset_lite_stream"]
    assert isinstance(contract, dict)
    kwargs: dict[str, object] = {}
    if mismatch == "hash":
        contract["ordered_sample_id_sha256"] = "wrong"
    elif mismatch == "seed":
        kwargs["asserted_seed"] = 11
    elif mismatch == "root":
        contract["root"] = str(tmp_path / "other-root")
    elif mismatch == "checkpoint":
        payload["checkpoint_hash"] = "wrong"
    elif mismatch == "config":
        payload["model_config"] = {"hidden_dim": 32}
    elif mismatch == "sampling":
        kwargs["asserted_sampling"] = "uniform"
    elif mismatch == "count":
        payload["steps"] = 3
    training_report.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        audit,
        "iter_floorset_lite",
        lambda *_args, **_kwargs: iter((_sample("train/a"), _sample("train/b"))),
    )

    with pytest.raises(ValueError, match=message):
        audit._load_training_exclusion(
            training_report,
            root=root,
            checkpoint=checkpoint,
            checkpoint_hash="state-hash",
            checkpoint_config={"hidden_dim": 16},
            **kwargs,
        )


def test_candidate_record_uses_training_baselines_and_uncapped_formula() -> None:
    side = 2.0**-0.5
    case = from_official(
        2,
        [1.0, 1.0],
        [[0, 1, 1.0]],
        [],
        [],
        torch.zeros((2, 5), dtype=torch.long),
    )
    candidate = torch.tensor(
        [
            [0.0, 0.0, side, side],
            [side, 0.0, side, side],
        ]
    )

    record = audit._candidate_record(
        case,
        candidate,
        1,
        "analytic_initial",
        frozenset(),
        1.0,
        0.5,
    )

    assert record["hard_feasible"] is True
    assert record["hpwl_total"] == pytest.approx(1.0)
    assert record["bbox_area"] == pytest.approx(2.0)
    assert record["hpwl_gap"] == pytest.approx(1.0)
    assert record["area_gap"] == pytest.approx(1.0)
    assert record["violations_relative"] == 0.0
    assert record["uncapped_objective"] == pytest.approx(2.0)
    assert record["official_capped_cost"] is None

    constraint = audit._candidate_record(
        case,
        candidate,
        1,
        "learned_initial",
        frozenset(),
        1.0,
        0.5,
        constraint_indices=frozenset({1}),
    )
    assert constraint["candidate_type"] == "constraint"


@pytest.mark.parametrize("produced", (0, 1))
def test_topology_audit_fails_when_requested_count_is_missing(produced: int) -> None:
    analysis = SimpleNamespace(
        result=SimpleNamespace(
            used_checkpoint=True,
            checkpoint_hash="state-hash",
            failure_reason=None,
            topology_seed_count=produced,
        ),
        analytic=SimpleNamespace(
            incumbent_snapshot={"topology_seed_failure_reason": "decode shortfall"}
        ),
    )

    with pytest.raises(
        RuntimeError,
        match=rf"requested 2 topology seeds, produced {produced}",
    ):
        audit._validate_topology_result("heldout/a", analysis, "state-hash", 2)


def test_constraint_audit_fails_when_requested_count_is_missing() -> None:
    analysis = SimpleNamespace(
        result=SimpleNamespace(
            used_checkpoint=True,
            checkpoint_hash="state-hash",
            failure_reason=None,
            topology_seed_count=2,
            constraint_seed_count=1,
        ),
        analytic=SimpleNamespace(
            incumbent_snapshot={
                "constraint_seed_failure_reason": "construction shortfall"
            }
        ),
    )

    with pytest.raises(
        RuntimeError,
        match=r"requested 2 constraint seeds, produced 1",
    ):
        audit._validate_topology_result(
            "heldout/a",
            analysis,
            "state-hash",
            2,
            2,
        )


def test_constraint_seeds_require_positive_topology_count() -> None:
    with pytest.raises(ValueError, match="requires --topology-seeds"):
        audit._validate_args(
            SimpleNamespace(
                constraint_seeds=1,
                topology_seeds=0,
                collective_steps=0,
            )
        )


def test_topology_audit_rejects_negative_collective_steps() -> None:
    with pytest.raises(audit.argparse.ArgumentTypeError):
        audit._non_negative_int("-1")


def test_audit_sample_classifies_constraint_snapshot_sources(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sample = _sample("heldout/a")
    fallback = safe_shelf(sample.case)
    candidates = fallback.unsqueeze(0).repeat(9, 1, 1)
    snapshot = {
        "exact_source": "fallback",
        "topology_seed_sources": ("candidate_4", "candidate_8"),
        "constraint_seed_sources": ("candidate_3", "candidate_7"),
        "constraint_seed_provenance": (
            {"source": "candidate_3", "kind": "combined"},
            {"source": "candidate_7", "kind": "combined"},
        ),
    }
    analysis = SimpleNamespace(
        result=SimpleNamespace(
            used_checkpoint=True,
            checkpoint_hash="state-hash",
            topology_seed_count=1,
            constraint_seed_count=1,
            candidate_count=4,
            selected=fallback,
        ),
        analytic=SimpleNamespace(
            raw_candidates=candidates,
            projected_candidates=candidates.clone(),
            incumbent_snapshot=snapshot,
        ),
    )
    monkeypatch.setattr(
        audit,
        "analyze_case_with_checkpoint",
        lambda *_args, **_kwargs: analysis,
    )

    result = audit._audit_sample(
        0,
        sample,
        Path("checkpoint.pt"),
        "state-hash",
        torch.device("cpu"),
        SimpleNamespace(),
        1,
        1,
        1,
    )

    assert result["candidate_layout"]["constraint_count"] == 1
    assert result["raw"]["candidates"][3]["candidate_type"] == "constraint"
    assert result["raw"]["candidates"][4]["candidate_type"] == "topology"
    assert result["post_bdp"]["candidates"][7]["candidate_type"] == "constraint"
    assert result["post_bdp"]["candidates"][8]["candidate_type"] == "topology"
    assert result["raw"]["oracles"]["constraint"]["candidate_index"] == 3
    assert result["constraint_provenance"]["constraint_seed_sources"] == (
        "candidate_3",
        "candidate_7",
    )


def test_constraint_oracle_reports_soft_counts_and_topology_gain() -> None:
    def candidate(
        index: int,
        candidate_type: str,
        objective: float,
        counts: tuple[int, int, int],
    ) -> dict[str, object]:
        boundary, grouping, mib = counts
        return {
            "candidate_index": index,
            "source": "learned_initial",
            "candidate_type": candidate_type,
            "hard_feasible": True,
            "hpwl_gap": 0.0,
            "area_gap": 0.0,
            "boundary_violations": boundary,
            "grouping_violations": grouping,
            "mib_violations": mib,
            "total_soft_violations": boundary + grouping + mib,
            "max_possible_violations": 10,
            "violations_relative": (boundary + grouping + mib) / 10.0,
            "official_capped_cost": None,
            "uncapped_objective": objective,
        }

    candidates = [
        candidate(1, "topology", 2.0, (2, 2, 2)),
        candidate(2, "constraint", 1.5, (1, 2, 3)),
    ]
    oracles = audit._oracles(candidates)
    case = {
        "test_id": 0,
        "block_count": 120,
        "raw": {"candidates": candidates, "oracles": oracles},
        "post_bdp": {"candidates": candidates, "oracles": oracles},
        "incumbent": candidates[1],
    }

    summary = audit._summary([case])

    assert oracles["constraint"] == {
        "area_gap": 0.0,
        "boundary_violations": 1,
        "candidate_index": 2,
        "candidate_type": "constraint",
        "grouping_violations": 2,
        "hard_feasible": True,
        "hpwl_gap": 0.0,
        "max_possible_violations": 10,
        "mib_violations": 3,
        "official_capped_cost": None,
        "source": "learned_initial",
        "total_soft_violations": 6,
        "uncapped_objective": 1.5,
        "violations_relative": 0.6,
    }
    assert summary["constraint_oracle"]["raw"]["available_cases"] == 1
    assert summary["constraint_oracle"]["raw"]["total_boundary_violations"] == 1
    assert summary["constraint_oracle"]["raw"]["total_grouping_violations"] == 2
    assert summary["constraint_oracle"]["raw"]["total_mib_violations"] == 3
    assert summary["topology_vs_constraint_gain"]["raw"] == {
        "comparable_cases": 1,
        "constraint_better_cases": 1,
        "mean_constraint_gain": 0.5,
        "tied_cases": 0,
        "topology_better_cases": 0,
        "weighted_mean_constraint_gain": 0.5,
    }
    assert summary["topology_vs_constraint_weighted_gain"]["post_bdp"] == 0.5


def test_main_report_is_byte_deterministic_and_records_disjoint_provenance(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    excluded = _sample("train/a")
    heldout = _sample("heldout/a")
    training_root = tmp_path / "training"
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"checkpoint")
    checkpoint_config = {"hidden_dim": 16, "collective_enabled": True}
    training_report = Path(f"{checkpoint.resolve()}.training.json")
    training_report.write_text(
        json.dumps(
            _training_report_payload(
                training_root,
                checkpoint,
                ["train/a"],
                source_limit=1,
                model_config=checkpoint_config,
            ),
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        audit,
        "iter_floorset_lite",
        lambda *_args, **_kwargs: iter((excluded,)),
    )
    monkeypatch.setattr(
        audit,
        "iter_floorset_lite_with_source",
        lambda *_args, **_kwargs: iter(((excluded, {}), (heldout, {}))),
    )
    monkeypatch.setattr(
        audit,
        "load_checkpoint",
        lambda *_args, **_kwargs: (
            object(),
            {
                "state_hash": "state-hash",
                "normalization": RUNTIME_NORMALIZATION,
                "config": checkpoint_config,
                "capabilities": {"collective": True, "flow": False},
                "trained_heads": ["collective"],
            },
        ),
    )

    def fake_analysis(case, _checkpoint, config):
        assert config.topology_seeds == 1
        assert config.constraint_seeds == 0
        assert config.collective_steps == 2
        assert config.analytic.dynamics.population == 1
        fallback = safe_shelf(case)
        candidates = fallback.unsqueeze(0).repeat(7, 1, 1)
        snapshot = {
            "exact_source": "fallback",
            "topology_seed_sources": ("candidate_3", "candidate_6"),
            "topology_seed_count": 1,
            "topology_seed_provenance": (
                {"source": "candidate_3", "stage": "initial"},
                {"source": "candidate_6", "stage": "post_relax"},
            ),
        }
        return SimpleNamespace(
            result=SimpleNamespace(
                used_checkpoint=True,
                checkpoint_hash="state-hash",
                failure_reason=None,
                topology_seed_count=1,
                candidate_count=3,
                selected=fallback,
            ),
            analytic=SimpleNamespace(
                raw_candidates=candidates,
                projected_candidates=candidates.clone(),
                incumbent_snapshot=snapshot,
            ),
        )

    monkeypatch.setattr(audit, "analyze_case_with_checkpoint", fake_analysis)
    output = tmp_path / "heldout.json"
    argv = [
        "--root",
        str(training_root),
        "--checkpoint",
        str(checkpoint),
        "--output",
        str(output),
        "--heldout-limit",
        "1",
        "--heldout-seed",
        "11",
        "--exclude-train-limit",
        "1",
        "--exclude-train-seed",
        "10",
        "--min-blocks",
        "2",
        "--max-blocks",
        "2",
        "--population",
        "1",
        "--topology-seeds",
        "1",
        "--device",
        "cpu",
        "--projection-steps",
        "1",
        "--collective-steps",
        "2",
    ]

    assert audit.main(argv) == 0
    first = output.read_bytes()
    assert audit.main(argv) == 0
    second = output.read_bytes()

    assert first == second
    report = json.loads(first)
    assert report["sampling"]["heldout"]["sample_ids"] == ["heldout/a"]
    assert (
        report["sampling"]["heldout"]["sample_id_sha256"]
        == hashlib.sha256(b"heldout/a").hexdigest()
    )
    assert report["sampling"]["heldout"]["exclude_overlap_count"] == 0
    assert report["sampling"]["exclude_training"]["count"] == 1
    assert report["sampling"]["exclude_training"]["consumed_count"] == 1
    assert report["sampling"]["exclude_training"]["training_report"] == str(
        training_report
    )
    assert report["sampling"]["training_report"] == {
        "path": str(training_report),
        "sha256": audit.file_sha256(training_report),
    }
    assert report["checkpoint"]["state_hash"] == "state-hash"
    assert report["checkpoint"]["capabilities"]["collective"] is True
    assert report["checkpoint"]["trained_heads"] == ["collective"]
    assert report["config"]["constraint_seeds"] == 0
    assert report["config"]["requested_collective_steps"] == 2
    assert report["config"]["collective_steps"] == 2
    assert report["evaluation"]["official_raw_replay"] is False
    assert report["cases"][0]["baseline"] == {"area": 2.0, "hpwl": 0.0}
    assert report["cases"][0]["candidate_layout"]["topology_count"] == 1
    assert report["cases"][0]["candidate_layout"]["constraint_count"] == 0
    assert report["cases"][0]["raw"]["oracles"]["constraint"] is None
    assert report["cases"][0]["raw"]["candidates"][3]["source"] == "learned_initial"
    assert report["cases"][0]["raw"]["candidates"][3]["candidate_type"] == "topology"
    assert (
        report["cases"][0]["post_bdp"]["candidates"][6]["source"] == "learned_relaxed"
    )
    assert (
        report["cases"][0]["post_bdp"]["candidates"][6]["candidate_type"] == "topology"
    )
    assert report["cases"][0]["topology_provenance"]["topology_seed_sources"] == [
        "candidate_3",
        "candidate_6",
    ]
    assert report["summary"]["hard_feasibility"]["post_bdp"]["rate"] == 1.0
    assert report["summary"]["topology_vs_analytic_weighted_gain"] == {
        "post_bdp": 0.0,
        "raw": 0.0,
    }
    assert report["summary"]["selected_vs_analytic"] == {
        "analytic_better_cases": 0,
        "comparable_cases": 1,
        "mean_selected_gain": 0.0,
        "selected_better_cases": 0,
        "tied_cases": 1,
        "weighted_mean_selected_gain": 0.0,
    }
    assert report["command"] == ["scripts/audit_hcfp_topology_heldout.py", *argv]
