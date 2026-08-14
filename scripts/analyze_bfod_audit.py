#!/usr/bin/env python3
"""Per-round trajectory audit for the BFOD common loop.

Consumes case{id}/audit.json (produced by experiment_bfod_v1.py --audit) plus
case{id}/result.json, and prints a compact per-round table plus the five
contact-pattern checks (group, bridge, patch size, orientation, HPWL sign).

Usage:
  python3 scripts/analyze_bfod_audit.py artifacts/experiments/bfod_v1_audit/case70
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _brief(metrics: dict[str, Any]) -> str:
    return (
        f"c={metrics['cost']:.4f} B/G/M="
        f"{metrics['boundary_violations']}/{metrics['grouping_violations']}/"
        f"{metrics['mib_violations']}"
    )


def _contact_summary(details: dict[str, Any]) -> str:
    base = (
        details.get("contact")
        if isinstance(details.get("contact"), dict)
        else details
    )
    obligation = base.get("obligation", {})
    moving = obligation.get("moving_component_size")
    extra = (
        f" comps={obligation.get('component_a_size')}+{obligation.get('component_b_size')}"
        f" moving={moving}"
        if obligation
        else ""
    )
    return (
        f"G{base.get('group_id')} bridge={base.get('bridge_member')}"
        f" anchor={base.get('anchor_member')} side={base.get('side')}"
        f" patch={len(base.get('members', ()))}" + extra
    )


def _orientation(side: str | None) -> str | None:
    if side is None:
        return None
    return "H" if side in {"left", "right"} else "V"


def chain_table(result: dict[str, Any], out: list[str]) -> None:
    baseline = result["baseline"]["metrics"]
    history = result["winner"]["history"]
    out.append("## Accepted chain (winner history)")
    out.append(
        "| step | family | detail | cost | B/G/M | ΔG | Δcost | Δhpwl_gap | Δarea_gap |"
    )
    out.append("|---|---|---|---:|---|---:|---:|---:|---:|")
    prev = baseline
    out.append(
        f"| base | — | — | {prev['cost']:.4f} | {prev['boundary_violations']}/"
        f"{prev['grouping_violations']}/{prev['mib_violations']} | — | — | — | — |"
    )
    for index, entry in enumerate(history, start=1):
        m = entry["metrics"]
        detail = entry.get("details", {})
        if entry["family"] == "contact" or "group_id" in detail:
            text = _contact_summary(detail)
        elif entry["family"] == "joint":
            text = "joint:" + _contact_summary(detail)
        else:
            text = json.dumps(detail, sort_keys=True)[:80]
        out.append(
            f"| {index} | {entry['family']} | {text} | {m['cost']:.4f} | "
            f"{m['boundary_violations']}/{m['grouping_violations']}/{m['mib_violations']} | "
            f"{m['grouping_violations'] - prev['grouping_violations']:+d} | "
            f"{m['uncapped_cost'] - prev['uncapped_cost']:+.4f} | "
            f"{m['hpwl_gap'] - prev['hpwl_gap']:+.6f} | "
            f"{m['area_gap'] - prev['area_gap']:+.6f} |"
        )
        prev = m
    out.append("")


def round_table(audit: dict[str, Any], out: list[str]) -> list[dict[str, Any]]:
    out.append("## Per-round oracle vs selected")
    out.append(
        "| stage | rnd | state cost (B/G/M) | obligations | experts | oracle cost "
        "| selected cost | Δcost | ΔG | ΔHPWL | Δbbox | gap | class | accepted detail |"
    )
    out.append("|---|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---|---|")
    accepted_contacts: list[dict[str, Any]] = []

    def emit(
        stage_name: str,
        round_label: int,
        parent: dict[str, Any],
        obligations: str,
        experts: str,
        oracle: dict[str, Any] | None,
        selected: dict[str, Any] | None,
        cls: str,
    ) -> None:
        if selected is None:
            out.append(
                f"| {stage_name} | {round_label} | {parent['uncapped_cost']:.4f} "
                f"({parent['metrics']['boundary_violations']}/"
                f"{parent['metrics']['grouping_violations']}/"
                f"{parent['metrics']['mib_violations']}) | {obligations} | {experts} | "
                f"{oracle['uncapped_cost'] if oracle else '—'} | — | — | — | — | — | — | "
                f"{cls} | — |"
            )
            return
        delta_cost = selected["uncapped_cost"] - parent["uncapped_cost"]
        delta_g = (
            selected["metrics"]["grouping_violations"]
            - parent["metrics"]["grouping_violations"]
        )
        delta_hpwl = selected["hpwl_total"] - parent["hpwl_total"]
        delta_bbox = selected["bbox_area"] - parent["bbox_area"]
        gap = (
            oracle["uncapped_cost"] - selected["uncapped_cost"]
            if oracle and selected
            else None
        )
        details = selected.get("details", {})
        text = (
            _contact_summary(details)
            if selected["family"] in {"contact", "joint"}
            else selected["family"]
        )
        out.append(
            f"| {stage_name} | {round_label} | {parent['uncapped_cost']:.4f} "
            f"({parent['metrics']['boundary_violations']}/"
            f"{parent['metrics']['grouping_violations']}/"
            f"{parent['metrics']['mib_violations']}) | {obligations} | {experts} | "
            f"{oracle['uncapped_cost']:.4f} | {selected['uncapped_cost']:.4f} | "
            f"{delta_cost:+.4f} | {delta_g:+d} | {delta_hpwl:+.2e} | "
            f"{delta_bbox:+.2e} | {gap:+.4f} | {cls} | {text} |"
        )
        if selected["family"] in {"contact", "joint"} and "group_id" in details:
            accepted_contacts.append(
                {
                    "stage": stage_name,
                    "round": round_label,
                    "family": selected["family"],
                    "details": details,
                    "delta_cost": delta_cost,
                    "delta_g": delta_g,
                    "delta_hpwl": delta_hpwl,
                    "delta_bbox": delta_bbox,
                }
            )

    for stage in audit["stages"]:
        for entry in stage["rounds"]:
            obligations = ",".join(
                f"{o['kind']}:{o['id']}" for o in entry.get("obligations", [])
            ) or "—"
            experts = ",".join(entry.get("experts", [])) or "—"
            if "states" in entry:  # common-loop multi-state round
                accepted = entry.get("accepted")
                if accepted is None:
                    continue
                selected = accepted["selected"]
                parent = accepted["state"]
                sub = entry["states"][accepted["state_index"]]
                obligations = ",".join(
                    f"{o['kind']}:{o['id']}" for o in sub.get("obligations", [])
                ) or "—"
                experts = ",".join(sub.get("experts", [])) or "—"
                oracles = [
                    item.get("oracle") for item in entry["states"] if item.get("oracle")
                ]
                oracle = min(oracles, key=lambda item: item["uncapped_cost"]) if oracles else None
                cls = sub["classification"]
                emit(stage["stage"], entry["round"], parent, obligations, experts, oracle, selected, cls)
            else:  # single-state stage round (mib / bootstrap / topology)
                emit(
                    stage["stage"],
                    entry.get("round", 0),
                    entry["state"],
                    obligations,
                    experts,
                    entry.get("oracle"),
                    entry.get("selected"),
                    entry.get("classification", "?"),
                )
    out.append("")
    return accepted_contacts


def checks(
    result: dict[str, Any],
    audit: dict[str, Any],
    audit_contacts: list[dict[str, Any]],
    out: list[str],
) -> None:
    out.append("## Five checks on accepted contact repairs (winner chain)")
    # Component table keyed by (group_id, bridge, anchor, side) from every
    # scored contact record in the audit; the generator is deterministic, so
    # the accepted chain's repairs are guaranteed present in some state.
    components: dict[tuple[Any, Any, Any, Any], dict[str, Any]] = {}
    for stage in audit["stages"]:
        for entry in stage["rounds"]:
            subs = entry["states"] if "states" in entry else [entry]
            for sub in subs:
                for run in sub.get("runs", []):
                    for record in run["records"]:
                        if record["family"] not in {"contact", "joint"}:
                            continue
                        details = record["details"]
                        base = (
                            details.get("contact")
                            if isinstance(details.get("contact"), dict)
                            else details
                        )
                        obligation = base.get("obligation", {})
                        if obligation:
                            components[
                                (
                                    base.get("group_id"),
                                    base.get("bridge_member"),
                                    base.get("anchor_member"),
                                    base.get("side"),
                                )
                            ] = obligation
    rows = []
    prev = result["baseline"]["metrics"]
    for entry in result["winner"]["history"]:
        details = entry.get("details", {})
        if entry["family"] not in {"contact", "joint"} or "group_id" not in details:
            prev = entry["metrics"]
            continue
        obligation = components.get(
            (
                details.get("group_id"),
                details.get("bridge_member"),
                details.get("anchor_member"),
                details.get("side"),
            ),
            {},
        )
        rows.append(
            (
                f"step{len(rows) + 1}",
                details.get("group_id"),
                details.get("bridge_member"),
                obligation.get("moving_component_size"),
                len(details.get("members", ())),
                _orientation(details.get("side")),
                entry["metrics"]["hpwl_gap"] - prev["hpwl_gap"],
                entry["metrics"]["uncapped_cost"] - prev["uncapped_cost"],
                entry["metrics"]["grouping_violations"] - prev["grouping_violations"],
                obligation.get("component_a_size"),
                obligation.get("component_b_size"),
            )
        )
        prev = entry["metrics"]
    if not rows:
        out.append("no accepted contact-family repair in winner chain")
        out.append("")
        return
    out.append("| step | G | bridge | moving | patch | orient | ΔHPWL | Δcost | ΔG | comps |")
    out.append("|---|---|---:|---:|---:|---|---:|---:|---:|---:|")
    for row in rows:
        out.append(
            f"| {row[0]} | G{row[1]} | {row[2]} | {row[3]} | {row[4]} | {row[5]} | "
            f"{row[6]:+.2e} | {row[7]:+.4f} | {row[8]:+d} | "
            f"{row[9]}+{row[10] if row[10] is not None else '?'} |"
        )
    out.append("")
    from collections import Counter

    out.append(f"- group distribution: {dict(sorted(Counter(r[1] for r in rows).items()))}")
    out.append(
        f"- moving component size: {dict(sorted(Counter(r[3] for r in rows).items()))} "
        f"(1 = singleton bridge)"
    )
    out.append(f"- patch size: {dict(sorted(Counter(r[4] for r in rows).items()))}")
    out.append(
        f"- orientation: {dict(sorted(Counter(r[5] for r in rows).items()))} "
        f"(H=left/right, V=top/bottom)"
    )
    out.append(
        f"- ΔHPWL sign: {sum(1 for r in rows if r[6] < 0)} down / "
        f"{sum(1 for r in rows if r[6] > 0)} up / "
        f"{sum(1 for r in rows if r[6] == 0)} flat"
    )
    out.append(
        f"- Δbbox_area: "
        f"{sum(1 for r in rows if abs(r[7]) < 1.0e-12 and r[8] < 0)} of {len(rows)} "
        f"repairs changed no bbox with a grouping gain"
    )
    out.append("")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("case_dir", help="path to case{id} directory with audit.json + result.json")
    parser.add_argument("--out", help="optional markdown output path")
    args = parser.parse_args(argv)
    case_dir = Path(args.case_dir)
    audit = json.loads((case_dir / "audit.json").read_text(encoding="utf-8"))
    result = json.loads((case_dir / "result.json").read_text(encoding="utf-8"))
    out: list[str] = []
    out.append(f"# BFOD trajectory audit — case {result['test_id']}")
    out.append("")
    out.append(f"baseline: {_brief(result['baseline']['metrics'])}")
    out.append(f"winner:   {_brief(result['winner']['metrics'])}")
    out.append("")
    chain_table(result, out)
    audit_contacts = round_table(audit, out)
    checks(result, audit, audit_contacts, out)
    text = "\n".join(out)
    print(text)
    if args.out:
        Path(args.out).write_text(text + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
