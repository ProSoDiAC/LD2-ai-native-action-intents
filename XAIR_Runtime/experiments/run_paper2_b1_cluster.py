#!/usr/bin/env python3
"""B1 grounding accuracy with category-cluster inference.

The 2,000 frames come from 27 VisA/MVTec categories and frames of one category share
visual structure, so frame-level Wilson/binomial/McNemar inference understates uncertainty.
For every model and both prompt versions (original blind, corrected blind_v2) this script
resamples the 27 categories with replacement (10,000 draws) and reports the accuracy, the
accuracy of every task-aware majority rule (one per tie-break) and the paired difference.
A sensitivity run re-scores the corrected prompts without UC2 (blind_v2_excl_uc2).
Output: paper2_b1_cluster.json.
"""
from __future__ import annotations

import json
import sys
from itertools import product
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.paper2_common import RESULTS_DIR  # noqa: E402
from experiments.run_paper2_v2_invariance import raw  # noqa: E402

MODELS = ["qwen2.5vl:7b", "qwen2.5vl:32b", "qwen2.5vl:3b", "gemma3:12b", "llama3.2-vision:11b"]
N_BOOT = 10000


def category(frame_id: str) -> str:
    return frame_id.rsplit("_", 1)[0]


def majority_rules(rows: list[dict]) -> list[dict[str, str]]:
    from collections import Counter

    ucs = sorted({r["use_case"] for r in rows})
    tops = {}
    for uc in ucs:
        c = Counter(r["gt_action"] for r in rows if r["use_case"] == uc)
        best = max(c.values())
        tops[uc] = sorted(a for a, v in c.items() if v == best)
    return [dict(zip(ucs, ch)) for ch in product(*(tops[uc] for uc in ucs))]


def analyse(rows: list[dict], rng: np.random.Generator) -> dict:
    cats = sorted({category(r["frame_id"]) for r in rows})
    cat_idx = {c: i for i, c in enumerate(cats)}
    ci = np.array([cat_idx[category(r["frame_id"])] for r in rows])
    ok = np.array([bool(r.get("action")) and r["action"] == r["gt_action"] for r in rows], dtype=float)
    rules = majority_rules(rows)
    base = [np.array([float(r["gt_action"] == m[r["use_case"]]) for r in rows]) for m in rules]
    # per-category sums so a draw is a weighted sum over categories
    k = len(cats)
    n_c = np.bincount(ci, minlength=k).astype(float)
    s_model = np.bincount(ci, weights=ok, minlength=k)
    s_base = [np.bincount(ci, weights=b, minlength=k) for b in base]
    draws = rng.integers(0, k, size=(N_BOOT, k))
    n_d = n_c[draws].sum(axis=1)
    acc = s_model[draws].sum(axis=1) / n_d
    out = {"n": len(rows), "n_categories": k, "accuracy": float(ok.mean()),
           "accuracy_cluster_ci95": [float(np.quantile(acc, .025)), float(np.quantile(acc, .975))],
           "vs_majority": []}
    for m, b, sb in zip(rules, base, s_base):
        d = acc - sb[draws].sum(axis=1) / n_d
        out["vs_majority"].append({
            "tie_break": {uc: a for uc, a in m.items()},
            "majority_accuracy": float(b.mean()), "diff": float(ok.mean() - b.mean()),
            "diff_cluster_ci95": [float(np.quantile(d, .025)), float(np.quantile(d, .975))],
            "share_draws_diff_positive": float((d > 0).mean())})
    diffs = [v["diff_cluster_ci95"] for v in out["vs_majority"]]
    out["diff_ci_envelope"] = [min(d[0] for d in diffs), max(d[1] for d in diffs)]
    return out


def main() -> int:
    orig, v2 = raw("phase_p", "blind"), raw("phase_p_v2", "blind_v2")
    res = {}
    for model in MODELS:
        res[model] = {}
        for name, src in (("blind", orig), ("blind_v2", v2)):
            rows = [r for (f, m), r in src.items() if m == model]
            if rows:
                res[model][name] = analyse(rows, np.random.default_rng(0))
        # Sensitivity: in UC2 the corrected task text says the cell is paused while the MES
        # field shows line.state = RUN, so the corrected prompts are re-scored without UC2.
        rows = [r for (f, m), r in v2.items() if m == model and r["use_case"] != "uc2_restart"]
        if rows:
            res[model]["blind_v2_excl_uc2"] = analyse(rows, np.random.default_rng(0))
    (RESULTS_DIR / "paper2_b1_cluster.json").write_text(json.dumps(res, indent=2) + "\n")
    for model, d in res.items():
        for name, a in d.items():
            print(f"{model:22s} {name:9s} n={a['n']} acc={a['accuracy']:.3f} cl95={a['accuracy_cluster_ci95'][0]:.3f}-{a['accuracy_cluster_ci95'][1]:.3f} "
                  f"diff-env=[{a['diff_ci_envelope'][0]:+.3f},{a['diff_ci_envelope'][1]:+.3f}] rules={len(a['vs_majority'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
