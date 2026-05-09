#!/usr/bin/env python3
"""Step7S Phase S2: run CCQP across all 8 representative cases.

For each case, picks the seed block from the Step7Q-F replay row with the
smallest official_like_cost_delta (most improving, including the two AVNR
candidates for cases 24 and 51). Builds the contact graph (applying the
AVNR target for cases that have one), runs CCQP, and aggregates the
primal/dual certificates.

Output: artifacts/research/step7s_critical_cone_summary.json
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import time
from pathlib import Path
from typing import Any

REPRESENTATIVE_CASE_IDS = [19, 24, 25, 51, 76, 79, 91, 99]
DEFAULT_RESEARCH = Path("artifacts/research")
STEP7Q_ROWS = DEFAULT_RESEARCH / "step7q_objective_slot_replay_rows.jsonl"


def pick_seed_per_case(rows_path: Path) -> dict[str, dict[str, Any]]:
    best: dict[str, dict[str, Any]] = {}
    with rows_path.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if not row.get("hard_feasible_nonnoop"):
                continue
            cid = str(row.get("case_id"))
            olc = float(row.get("official_like_cost_delta", 0.0))
            best.setdefault(cid, {"olc": float("inf")})
            if olc < best[cid]["olc"]:
                best[cid] = {
                    "case_id": cid,
                    "block_id": int(row["block_id"]),
                    "target_box": row.get("target_box"),
                    "olc": olc,
                    "is_avnr": bool(row.get("actual_all_vector_nonregressing")),
                }
    return best


def run_subprocess(cmd: list[str]) -> tuple[int, str, str]:
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    return proc.returncode, proc.stdout, proc.stderr


def run_case_certificate(
    case_id: int,
    seed: dict[str, Any],
    apply_target: bool,
    contact_path: Path,
    cert_path: Path,
    base_dir: Path,
    trust_region: float,
    fd_step: float,
    n_workers: int,
) -> dict[str, Any]:
    target_arg: list[str] = []
    if apply_target and seed.get("target_box"):
        target_arg = [
            "--apply-avnr-target",
            ",".join(str(v) for v in seed["target_box"]),
        ]
    base_dir_args = ["--base-dir", str(base_dir)]

    contact_cmd = [
        ".venv/bin/python", "scripts/step7s_contact_graph.py",
        *base_dir_args,
        "--case-id", str(case_id),
        "--seed-block", str(seed["block_id"]),
        *target_arg,
        "--eps-contact", "1e-3",
        "--out", str(contact_path),
    ]
    rc, stdout, stderr = run_subprocess(contact_cmd)
    if rc != 0:
        return {
            "case_id": case_id,
            "stage": "contact_graph",
            "stderr": stderr.strip(),
            "stdout": stdout.strip(),
            "result": "contact_graph_failed",
        }

    cert_cmd = [
        ".venv/bin/python", "scripts/step7s_critical_cone_certificate.py",
        *base_dir_args,
        "--case-id", str(case_id),
        "--seed-block", str(seed["block_id"]),
        *target_arg,
        "--contact-graph", str(contact_path),
        "--trust-region", str(trust_region),
        "--fd-step", str(fd_step),
        "--n-workers", str(n_workers),
        "--out", str(cert_path),
    ]
    rc, stdout, stderr = run_subprocess(cert_cmd)
    if rc != 0:
        return {
            "case_id": case_id,
            "stage": "ccqp",
            "stderr": stderr.strip(),
            "stdout": stdout.strip(),
            "result": "ccqp_failed",
        }
    cert = json.loads(cert_path.read_text(encoding="utf-8"))
    diag = cert.get("gradient_diagnostics", {})
    lp = cert.get("lp_result", {})
    ls = cert.get("line_search", {})
    best = ls.get("best") if isinstance(ls, dict) else None
    return {
        "case_id": case_id,
        "seed_block": seed["block_id"],
        "applied_avnr_target": cert.get("applied_avnr_target"),
        "is_avnr_seed": seed.get("is_avnr", False),
        "closure_size": cert.get("closure_size"),
        "active_pair_count": cert.get("active_pair_count"),
        "baseline_h": diag.get("baseline_h"),
        "baseline_a": diag.get("baseline_a"),
        "baseline_v": diag.get("baseline_v"),
        "hpwl_hinge_active": diag.get("hpwl_hinge_active"),
        "bbox_hinge_active": diag.get("bbox_hinge_active"),
        "g_cost_2_norm": diag.get("g_cost_2_norm"),
        "g_hpwl_2_norm": diag.get("g_hpwl_2_norm"),
        "g_bbox_2_norm": diag.get("g_bbox_2_norm"),
        "g_soft_used_2_norm": diag.get("g_soft_used_2_norm"),
        "g_soft_raw_fd_2_norm": diag.get("g_soft_raw_fd_2_norm"),
        "rho_pred": lp.get("rho_pred"),
        "linear_pred_cost_delta": lp.get("linear_pred_cost_delta"),
        "delta_inf_norm": lp.get("delta_inf_norm"),
        "result": cert.get("result"),
        "best_strict_alpha": best.get("alpha") if best else None,
        "best_strict_cost_delta": best.get("official_like_cost_delta") if best else None,
        "cert_path": str(cert_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-dir", type=Path, default=Path("."))
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_RESEARCH / "step7s_critical_cone_summary.json",
    )
    parser.add_argument("--trust-region", type=float, default=0.5)
    parser.add_argument("--fd-step", type=float, default=1e-4)
    parser.add_argument("--n-workers", type=int, default=48)
    parser.add_argument(
        "--apply-target-only-when-avnr",
        action="store_true",
        default=True,
        help="Apply Step7Q-F target only for AVNR seeds (cases 24, 51). "
             "For others use original baseline.",
    )
    args = parser.parse_args()

    seeds = pick_seed_per_case(STEP7Q_ROWS)
    started = time.perf_counter()
    per_case: list[dict[str, Any]] = []
    for case_id in REPRESENTATIVE_CASE_IDS:
        seed = seeds.get(str(case_id))
        if seed is None:
            per_case.append({"case_id": case_id, "result": "no_seed_in_step7q_rows"})
            continue
        apply_target = bool(seed.get("is_avnr"))
        contact_path = (
            DEFAULT_RESEARCH
            / f"step7s_case{case_id:03d}_block{seed['block_id']:03d}_contact_graph.json"
        )
        cert_path = (
            DEFAULT_RESEARCH
            / f"step7s_case{case_id:03d}_block{seed['block_id']:03d}_cone_certificate.json"
        )
        record = run_case_certificate(
            case_id,
            seed,
            apply_target=apply_target,
            contact_path=contact_path,
            cert_path=cert_path,
            base_dir=args.base_dir,
            trust_region=args.trust_region,
            fd_step=args.fd_step,
            n_workers=args.n_workers,
        )
        per_case.append(record)

    runtime = time.perf_counter() - started

    # Compute the HPWL hinge cap per case and re-classify "linearized_promising" cases.
    # Hinge cap = 0.5 * exp(2v) * max(0, h0). When < MEANINGFUL_COST_EPS, the LP rho
    # is fictitious and the case is effectively first-order stationary at threshold.
    MEANINGFUL_COST_EPS = 1e-7
    for record in per_case:
        h0 = record.get("baseline_h")
        v0 = record.get("baseline_v")
        if h0 is None or v0 is None:
            continue
        cap = 0.5 * math.exp(2.0 * float(v0)) * max(0.0, float(h0))
        record["hpwl_hinge_cap"] = cap
        record["hpwl_hinge_cap_below_strict"] = cap < MEANINGFUL_COST_EPS
        if (
            record.get("result") == "linearized_promising_no_strict_after_linesearch"
            and cap < MEANINGFUL_COST_EPS
        ):
            record["result_with_hinge_cap"] = "kkt_stationary_under_hpwl_hinge_cap"
        else:
            record["result_with_hinge_cap"] = record.get("result")

    kkt_count = sum(1 for r in per_case if r.get("result") == "kkt_stationary")
    strict_count = sum(1 for r in per_case if r.get("result") == "strict_winner")
    avnr_only = sum(1 for r in per_case if r.get("result") == "avnr_only_below_threshold")
    linearized_promising = sum(
        1
        for r in per_case
        if r.get("result") == "linearized_promising_no_strict_after_linesearch"
    )
    kkt_with_hinge_cap = sum(
        1
        for r in per_case
        if r.get("result_with_hinge_cap")
        in ("kkt_stationary", "kkt_stationary_under_hpwl_hinge_cap")
    )

    if strict_count >= 3:
        terminal = "strict_winner_set"
    elif kkt_count == len(REPRESENTATIVE_CASE_IDS):
        terminal = "local_kkt_stationarity_certified"
    elif kkt_with_hinge_cap == len(REPRESENTATIVE_CASE_IDS):
        terminal = "local_kkt_stationarity_certified_with_hpwl_hinge_cap"
    elif kkt_count + linearized_promising + avnr_only == len(REPRESENTATIVE_CASE_IDS):
        terminal = "no_strict_after_linesearch_no_universal_kkt"
    else:
        terminal = "mixed"

    summary = {
        "schema": "step7s_critical_cone_summary_v1",
        "terminal_result": terminal,
        "kkt_stationary_count": kkt_count,
        "kkt_with_hinge_cap_count": kkt_with_hinge_cap,
        "strict_winner_count": strict_count,
        "avnr_only_count": avnr_only,
        "linearized_promising_count": linearized_promising,
        "case_count": len(REPRESENTATIVE_CASE_IDS),
        "trust_region": args.trust_region,
        "fd_step": args.fd_step,
        "n_workers": args.n_workers,
        "runtime_seconds": runtime,
        "per_case": per_case,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "terminal_result": terminal,
                "kkt_stationary_count": kkt_count,
                "strict_winner_count": strict_count,
                "avnr_only_count": avnr_only,
                "linearized_promising_count": linearized_promising,
                "runtime_seconds": round(runtime, 2),
            }
        )
    )


if __name__ == "__main__":
    main()
