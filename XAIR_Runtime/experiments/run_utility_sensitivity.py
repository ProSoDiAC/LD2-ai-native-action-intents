#!/usr/bin/env python3
"""
Utility sensitivity: recompute U under alternate (λ, μ) for the policies already selected.

U is a mean per-trial reward on an exclusive outcome partition, so for B2–B4
U(λ, μ) = SAR − λ·Hazard − μ·(wrongful revokes / n) exactly (unconditional wrongful
rate, not the conditional WRR). B5 episodes can carry step costs and repeated
penalties; those are kept as a per-episode residual at the default weights.
Policies are re-weighted, not re-selected or re-trained (see
run_paper2_revision_analyses.py:b3_reselection for that).

Usage:
    python3 experiments/run_utility_sensitivity.py
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.paper2_common import LAMBDA_HAZARD, RESULTS_DIR  # noqa: E402

LAMBDAS = (1.0, 5.0, 10.0)
MUS = (0.5, 1.0, 2.0)


def u(sar: float, hazard: float, wrr: float, lam: float, mu: float) -> float:
    return sar - lam * hazard - mu * wrr


def trial_wrr(sar: float, hazard: float, reported_wrr: float, mean_reward: float | None) -> float:
    """Unconditional wrongful-revoke rate, inverted from the mean reward at the default (λ, μ=1)."""
    if mean_reward is None:
        return reported_wrr
    return sar - LAMBDA_HAZARD * hazard - mean_reward


def b5_from_trials(path: Path) -> list[dict]:
    """B5: exact U(λ, μ) per policy from episode trials, keeping step costs as a residual."""
    import json as _json
    from collections import defaultdict

    by = defaultdict(list)
    for line in path.read_text().splitlines():
        t = _json.loads(line)
        if t.get("setting") == "headline":
            by[t["policy"]].append(t)
    out = []
    for name, ts in sorted(by.items()):
        n = len(ts)
        s = sum(bool(t["successful_actuation"]) for t in ts) / n
        h = sum(bool(t["hazardous_publish"]) for t in ts) / n
        w = sum(bool(t["wrongful_revoke"]) for t in ts) / n
        resid = sum(float(t["reward"]) for t in ts) / n - (s - LAMBDA_HAZARD * h - w)
        grid = [{"lambda": lam, "mu": mu, "U": s - lam * h - mu * w + resid} for lam in LAMBDAS for mu in MUS]
        out.append({"name": name, "SAR": s, "hazard": h, "wrongful_revoke_rate": w, "residual": resid, "grid": grid})
    return out


def rank_suite(suite_out: list[dict]) -> dict:
    rankings = {}
    for lam in LAMBDAS:
        for mu in MUS:
            # Rank on the stored U(λ, μ), which for B5 includes step costs and repeated penalties.
            ordered = sorted(
                suite_out,
                key=lambda x: -next(g["U"] for g in x["grid"] if g["lambda"] == lam and g["mu"] == mu),
            )
            rankings[f"l{lam}_m{mu}"] = [x["name"] for x in ordered]
    baseline = rankings.get("l5.0_m1.0", [])
    kendallish = {}
    for key, order in rankings.items():
        if not baseline:
            continue
        pos_b = {n: i for i, n in enumerate(baseline)}
        agree = total = 0
        for i, a in enumerate(order):
            for b in order[i + 1 :]:
                total += 1
                if pos_b[a] < pos_b[b]:
                    agree += 1
        kendallish[key] = agree / total if total else 1.0
    return {"policies": suite_out, "rankings": rankings, "rank_agreement_vs_default": kendallish}


def load_rows(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    with path.open() as fh:
        return list(csv.DictReader(fh))


def main() -> int:
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    del args

    sources = {
        "b3": (RESULTS_DIR / "b3_headline_table.csv", "policy", None),
        "b4": (RESULTS_DIR / "b4_routing_table.csv", "router", "p50"),
        "b4_fixed_window": (RESULTS_DIR / "b4_routing_table_fixedw.csv", "router", "p50"),
    }
    out: dict = {"grid": {"lambda": list(LAMBDAS), "mu": list(MUS)}, "suites": {}}

    stats_path = RESULTS_DIR / "paper2_stats.json"
    if stats_path.is_file():
        stats = json.loads(stats_path.read_text())
        rates = stats.get("b2_headline_rates") or {}
        if rates:
            suite_out = []
            for name, r in rates.items():
                sar, hazard = float(r["SAR"]), float(r["hazard"])
                wrr = float(r.get("wrongful_revoke_rate_unconditional", r["WRR"]))
                grid = [
                    {"lambda": lam, "mu": mu, "U": u(sar, hazard, wrr, lam, mu)}
                    for lam in LAMBDAS
                    for mu in MUS
                ]
                suite_out.append({"name": name, "SAR": sar, "hazard": hazard, "wrongful_revoke_rate": wrr, "grid": grid})
            blob = rank_suite(suite_out)
            blob["source"] = "paper2_stats.b2_headline_rates"
            out["suites"]["b2"] = blob

    for suite, (path, name_key, setting) in sources.items():
        rows = load_rows(path)
        if setting:
            rows = [r for r in rows if r.get("setting") == setting]
        suite_out = []
        for r in rows:
            sar = float(r.get("SAR") or r.get("sar") or 0)
            hazard = float(r.get("hazard") or r.get("hazardous_publish_rate") or 0)
            reported_wrr = float(r.get("WRR") or r.get("wrr") or 0)
            mean_r = r.get("mean_reward") or r.get("utility")
            mean_r_f = float(mean_r) if mean_r not in (None, "") else None
            wrr = trial_wrr(sar, hazard, reported_wrr, mean_r_f)
            name = r[name_key]
            grid = [
                {"lambda": lam, "mu": mu, "U": u(sar, hazard, wrr, lam, mu)}
                for lam in LAMBDAS
                for mu in MUS
            ]
            suite_out.append({"name": name, "SAR": sar, "hazard": hazard, "wrongful_revoke_rate": wrr, "grid": grid})
        out["suites"][suite] = rank_suite(suite_out)

    b5_trials = RESULTS_DIR / "b5_eval_trials.jsonl"
    if b5_trials.is_file():
        out["suites"]["b5"] = rank_suite(b5_from_trials(b5_trials))

    path = RESULTS_DIR / "utility_sensitivity.json"
    path.write_text(json.dumps(out, indent=2))
    # CSV flat
    csv_path = RESULTS_DIR / "utility_sensitivity.csv"
    flat = []
    for suite, blob in out["suites"].items():
        for pol in blob["policies"]:
            for g in pol["grid"]:
                flat.append(
                    {
                        "suite": suite,
                        "name": pol["name"],
                        "lambda": g["lambda"],
                        "mu": g["mu"],
                        "U": g["U"],
                        "SAR": pol["SAR"],
                        "hazard": pol["hazard"],
                        "wrongful_revoke_rate": pol["wrongful_revoke_rate"],
                    }
                )
    with csv_path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(flat[0].keys()))
        w.writeheader()
        w.writerows(flat)
    print(json.dumps({"out": str(path), "rank_agreement_b5": out["suites"].get("b5", {}).get("rank_agreement_vs_default")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
