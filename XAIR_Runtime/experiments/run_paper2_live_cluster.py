#!/usr/bin/env python3
"""Category-cluster bootstrap of the live OPC UA contrasts (paper2_live_opcua_r*.json).

Frames of one VisA/MVTec category share visual structure, so the live contrasts are
re-estimated by resampling categories (10,000 draws) instead of single frames.
Output: paper2_live_cluster.json.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.paper2_common import RESULTS_DIR  # noqa: E402
from experiments.run_paper2_live_opcua import reward  # noqa: E402

N_BOOT = 10000


def u(r: dict, path: str) -> float:
    if path == "revoke_all":
        return reward("forgone" if r["valid_at_release"] and r["correct"] else "correct_revoke")
    return reward(r[path + "_kind"])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", default="paper2_live_opcua_r", help="input file prefix before the rate")
    ap.add_argument("--out", default="paper2_live_cluster.json")
    args = ap.parse_args()
    out = {}
    for rate in ("0.05", "0.2", "1.0"):
        rows = [r for r in json.loads((RESULTS_DIR / f"{args.prefix}{rate}.json").read_text())["rows"] if "error" not in r]
        cats = sorted({r["frame_id"].rsplit("_", 1)[0] for r in rows})
        ci = np.array([cats.index(r["frame_id"].rsplit("_", 1)[0]) for r in rows])
        n_c = np.bincount(ci, minlength=len(cats)).astype(float)
        draws = np.random.default_rng(0).integers(0, len(cats), size=(N_BOOT, len(cats)))
        res = {"n": len(rows), "n_categories": len(cats)}
        for a, b in (("xair", "freshness_only"), ("xair", "revoke_all")):
            d = np.array([u(r, a) - u(r, b) for r in rows])
            s = np.bincount(ci, weights=d, minlength=len(cats))
            boots = s[draws].sum(axis=1) / n_c[draws].sum(axis=1)
            res[f"{a}_minus_{b}"] = {"delta": float(d.mean()), "ci95": [float(np.quantile(boots, .025)), float(np.quantile(boots, .975))]}
        out[rate] = res
        print(rate, {k: v for k, v in res.items() if k.startswith("xair")})
    (RESULTS_DIR / args.out).write_text(json.dumps(out, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
