#!/usr/bin/env python3
"""Render ranker selected-vs-oracle replay candidates from an eval report."""

from __future__ import annotations

import argparse
import hashlib
from html import escape
import json
from pathlib import Path
import re
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hcfp.data import file_sha256  # noqa: E402
from hcfp.replay import OFFICIAL_TARGET_KIND, ReplayRecord, iter_replay  # noqa: E402
from hcfp.visualize import render_html  # noqa: E402


_SAFE_NAME = re.compile(r"[^A-Za-z0-9_.-]+")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay", required=True, help="schema v3 replay JSONL")
    parser.add_argument("--evaluation", required=True, help="ranker evaluation JSON")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--split", help="evaluation split name; required when ambiguous")
    parser.add_argument("--checkpoint", help="evaluation checkpoint name; required when ambiguous")
    parser.add_argument("--stage", action="append", help="candidate stage filter; may be repeated")
    parser.add_argument("--sample-id", action="append", help="sample id filter; may be repeated")
    parser.add_argument("--limit", type=_positive_int)
    parser.add_argument("--force", action="store_true", help="overwrite files created by a previous run")
    args = parser.parse_args(argv)

    replay_path = Path(args.replay)
    evaluation_path = Path(args.evaluation)
    output_dir = Path(args.output_dir)
    records = _load_replay_records(replay_path)
    report = _load_evaluation(evaluation_path)
    split_name, checkpoint_name, cases = _select_eval_cases(
        report,
        split=args.split,
        checkpoint=args.checkpoint,
    )
    selected_stages = set(args.stage or [])
    selected_samples = set(args.sample_id or [])
    selected_cases = []
    for case in cases:
        stage = str(case.get("candidate_stage", "unknown"))
        sample_id = str(case.get("sample_id", ""))
        if selected_stages and stage not in selected_stages:
            continue
        if selected_samples and sample_id not in selected_samples:
            continue
        selected_cases.append(case)
    if args.limit is not None:
        selected_cases = selected_cases[: args.limit]
    if not selected_cases:
        raise ValueError("selection contains no ranker evaluation cases")

    output_dir.mkdir(parents=True, exist_ok=True)
    entries = []
    rendered_pages: list[tuple[Path, str]] = []
    seen_files: set[str] = set()
    for ordinal, case in enumerate(selected_cases):
        sample_id = _required_string(case, "sample_id")
        stage = _required_string(case, "candidate_stage")
        record = records.get((sample_id, stage))
        if record is None:
            raise ValueError(f"replay is missing sample-stage record {sample_id!r}/{stage!r}")
        selected_index = _candidate_index(case, "selected_index", record)
        oracle_index = _candidate_index(case, "oracle_index", record)
        _validate_eval_row_id(case, "selected_row_id", record, selected_index)
        _validate_eval_row_id(case, "oracle_row_id", record, oracle_index)
        filename = _case_filename(ordinal, sample_id, stage)
        if filename in seen_files:
            raise ValueError(f"duplicate visualization filename {filename!r}")
        seen_files.add(filename)
        path = output_dir / filename
        html = render_html(
            _visual_items(record, case, selected_index, oracle_index),
            title=f"HCFP ranker {sample_id} {stage}",
        )
        rendered_pages.append((path, html))
        entries.append(
            {
                "sample_id": sample_id,
                "candidate_stage": stage,
                "file": filename,
                "selected_index": selected_index,
                "oracle_index": oracle_index,
                "selected_row_id": _row_id(record, selected_index),
                "oracle_row_id": _row_id(record, oracle_index),
                "top1_exact_best": bool(case.get("top1_exact_best", selected_index == oracle_index)),
                "top4_oracle_recall": bool(case.get("top4_oracle_recall", False)),
                "rank_regret": int(case.get("rank_regret", 0)),
                "score_regret": float(case.get("score_regret", 0.0)),
            }
        )

    manifest = {
        "schema_version": 1,
        "inputs": {
            "replay": str(replay_path),
            "replay_sha256": file_sha256(replay_path),
            "evaluation": str(evaluation_path),
            "evaluation_sha256": file_sha256(evaluation_path),
            "split": split_name,
            "checkpoint": checkpoint_name,
        },
        "selection": {
            "stages": sorted(selected_stages),
            "sample_ids": sorted(selected_samples),
            "limit": args.limit,
        },
        "cases": entries,
    }
    manifest_path = output_dir / "manifest.json"
    index_path = output_dir / "index.html"
    _ensure_writable(
        [*(path for path, _html in rendered_pages), manifest_path, index_path],
        force=args.force,
    )
    for path, html in rendered_pages:
        _write_text(path, html, force=args.force)
    _write_text(
        manifest_path,
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        force=args.force,
    )
    _write_text(index_path, _index_html(manifest), force=args.force)
    print(output_dir / "index.html")
    print(output_dir / "manifest.json")
    return 0


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def _load_replay_records(path: Path) -> dict[tuple[str, str], ReplayRecord]:
    records: dict[tuple[str, str], ReplayRecord] = {}
    for record in iter_replay(path):
        if record.target_kind != OFFICIAL_TARGET_KIND:
            raise ValueError("ranker visualization requires official v10 replay targets")
        if record.candidate_stage is None:
            raise ValueError("ranker visualization requires schema v3 candidate_stage")
        if record.candidate_geometry is None or record.post_bdp_geometry is None or record.post_repair_geometry is None:
            raise ValueError("ranker visualization requires schema v3 candidate geometries")
        key = (record.sample.sample_id, record.candidate_stage)
        if key in records:
            raise ValueError(f"duplicate replay sample-stage record {key!r}")
        records[key] = record
    if not records:
        raise ValueError("replay is empty")
    return records


def _load_evaluation(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if int(payload.get("schema_version", -1)) != 2:
        raise ValueError("ranker evaluation schema_version must be 2")
    if payload.get("target_kind") != OFFICIAL_TARGET_KIND:
        raise ValueError("ranker evaluation target_kind mismatch")
    results = payload.get("results")
    if not isinstance(results, dict) or not results:
        raise ValueError("ranker evaluation contains no split results")
    return payload


def _select_eval_cases(
    report: dict[str, Any],
    *,
    split: str | None,
    checkpoint: str | None,
) -> tuple[str, str, list[dict[str, Any]]]:
    results = report["results"]
    split_name = split or _single_key(results, "split")
    split_payload = results.get(split_name)
    if not isinstance(split_payload, dict):
        raise ValueError(f"evaluation split {split_name!r} is missing")
    checkpoint_name = checkpoint or _single_key(split_payload, "checkpoint")
    checkpoint_payload = split_payload.get(checkpoint_name)
    if not isinstance(checkpoint_payload, dict):
        raise ValueError(f"evaluation checkpoint {checkpoint_name!r} is missing")
    cases = checkpoint_payload.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError("evaluation checkpoint contains no cases")
    if any(not isinstance(case, dict) for case in cases):
        raise ValueError("evaluation cases must be JSON objects")
    return split_name, checkpoint_name, cases


def _single_key(payload: dict[str, Any], noun: str) -> str:
    keys = sorted(payload)
    if len(keys) != 1:
        raise ValueError(f"ambiguous evaluation {noun}; pass --{noun}")
    return keys[0]


def _candidate_index(case: dict[str, Any], key: str, record: ReplayRecord) -> int:
    if key not in case:
        raise ValueError(f"evaluation case is missing {key}")
    value = case[key]
    if type(value) is not int:
        raise ValueError(f"evaluation {key} must be an integer")
    index = value
    population = int(record.candidate_geometry.shape[0])  # type: ignore[union-attr]
    if index < 0 or index >= population:
        raise ValueError(f"{key}={index} is outside replay population {population}")
    return index


def _validate_eval_row_id(
    case: dict[str, Any],
    key: str,
    record: ReplayRecord,
    index: int,
) -> None:
    if key not in case or not isinstance(case[key], str) or not case[key]:
        raise ValueError(f"evaluation case is missing {key}")
    actual = _row_id(record, index)
    if str(case[key]) != actual:
        raise ValueError(f"evaluation {key} does not match replay candidate row id")


def _visual_items(
    record: ReplayRecord,
    case: dict[str, Any],
    selected_index: int,
    oracle_index: int,
) -> list[dict[str, Any]]:
    items = []
    for label, index in (("ranker selected", selected_index), ("exact oracle", oracle_index)):
        for stage_name, geometry in (
            ("raw", record.candidate_geometry),
            ("post-BDP", record.post_bdp_geometry),
            ("post-repair", record.post_repair_geometry),
        ):
            assert geometry is not None
            items.append(
                {
                    "title": _candidate_title(record, case, label, stage_name, index),
                    "placements": geometry[index].tolist(),
                    "case": record.sample.case,
                    "telemetry": _candidate_telemetry(record, case, index),
                }
            )
    return items


def _candidate_title(
    record: ReplayRecord,
    case: dict[str, Any],
    label: str,
    stage_name: str,
    index: int,
) -> str:
    row_id = _row_id(record, index)
    target_rank = _tensor_int(record.target_rank, index)
    tier = _tensor_int(record.feasibility_tier, index)
    return (
        f"{label} candidate {index} {stage_name} "
        f"rank={target_rank} tier={tier} row={row_id} "
        f"pred={_predicted_cost(case, label):.6g}"
    )


def _predicted_cost(case: dict[str, Any], label: str) -> float:
    key = "prediction_selected_cost" if label == "ranker selected" else "prediction_oracle_cost"
    return float(case.get(key, 0.0))


def _candidate_telemetry(record: ReplayRecord, case: dict[str, Any], index: int) -> dict[str, float | int | bool]:
    return {
        "target_score": _tensor_float(record.target_score, index),
        "target_rank": _tensor_int(record.target_rank, index),
        "feasibility_tier": _tensor_int(record.feasibility_tier, index),
        "cap_margin": _tensor_float(record.post_repair_cap_margin, index),
        "repair_displacement": _tensor_float(record.repair_displacement, index),
        "boundary_violations": _tensor_int(record.boundary_violations, index),
        "grouping_violations": _tensor_int(record.grouping_violations, index),
        "mib_violations": _tensor_int(record.mib_violations, index),
        "hard_feasible": bool(_tensor_int(record.post_repair_hard_feasible, index)),
        "rank_regret": int(case.get("rank_regret", 0)),
        "score_regret": float(case.get("score_regret", 0.0)),
    }


def _tensor_float(value: Any, index: int) -> float:
    if value is None:
        return 0.0
    return float(value[index].item())


def _tensor_int(value: Any, index: int) -> int:
    if value is None:
        return 0
    return int(value[index].item())


def _row_id(record: ReplayRecord, index: int) -> str:
    if record.candidate_row_ids is None:
        return f"candidate-{index}"
    return record.candidate_row_ids[index]


def _required_string(case: dict[str, Any], key: str) -> str:
    value = case.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"evaluation case is missing {key}")
    return value


def _case_filename(ordinal: int, sample_id: str, stage: str) -> str:
    digest = hashlib.sha256(f"{sample_id}\n{stage}".encode()).hexdigest()[:12]
    name = _SAFE_NAME.sub("_", f"{stage}_{sample_id}")[:72].strip("._-") or "case"
    return f"{ordinal:04d}_{name}_{digest}.html"


def _write_text(path: Path, text: str, *, force: bool) -> None:
    if path.exists() and not force:
        raise FileExistsError(f"refusing to overwrite existing visualization file: {path}")
    path.write_text(text, encoding="utf-8")


def _ensure_writable(paths: list[Path], *, force: bool) -> None:
    if force:
        return
    existing = next((path for path in paths if path.exists()), None)
    if existing is not None:
        raise FileExistsError(
            f"refusing to overwrite existing visualization file: {existing}"
        )


def _index_html(manifest: dict[str, Any]) -> str:
    rows = []
    for case in manifest["cases"]:
        rows.append(
            "<tr>"
            f"<td><a href=\"{escape(case['file'])}\">{escape(case['sample_id'])}</a></td>"
            f"<td>{escape(case['candidate_stage'])}</td>"
            f"<td>{int(case['selected_index'])}</td>"
            f"<td>{int(case['oracle_index'])}</td>"
            f"<td>{escape(str(case['top1_exact_best']))}</td>"
            f"<td>{int(case['rank_regret'])}</td>"
            f"<td>{float(case['score_regret']):.6g}</td>"
            "</tr>"
        )
    title = "HCFP ranker selected-vs-oracle visualizations"
    return (
        "<!doctype html>\n"
        '<html lang="en">\n'
        "<head><meta charset=\"utf-8\">"
        f"<title>{title}</title>"
        "<style>body{font-family:sans-serif;margin:20px;color:#202124}"
        "table{border-collapse:collapse}td,th{border:1px solid #dadce0;padding:6px 8px}"
        "th{background:#f1f3f4;text-align:left}</style></head>\n"
        f"<body><h1>{title}</h1>"
        f"<p>split={escape(manifest['inputs']['split'])} checkpoint={escape(manifest['inputs']['checkpoint'])}</p>"
        "<table><thead><tr><th>sample</th><th>stage</th><th>selected</th>"
        "<th>oracle</th><th>top1</th><th>rank regret</th><th>score regret</th></tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table></body></html>\n"
    )


if __name__ == "__main__":
    raise SystemExit(main())
