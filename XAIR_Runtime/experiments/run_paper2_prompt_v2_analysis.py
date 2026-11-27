#!/usr/bin/env python3
"""
Revision experiments on the primary 7B model (2026-09 review):

  * blind-v2   corrected task texts (every ground-truth action listed, neutral UC4),
               cache tag ``phase_p_v2``, compared with the original blind prompt;
  * repeat     the original blind prompt re-run on a stratified 400-frame subset,
               cache tag ``phase_p_rep``, compared with the Phase-P decisions.

Writes ``results/paper2_v2_analysis.json``.

Usage:
    python3 experiments/run_paper2_prompt_v2_analysis.py
"""

from __future__ import annotations

import json
import statistics
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.paper2_common import RESULTS_DIR, load_manifest, wilson_ci  # noqa: E402
from experiments.perception_cache import cache_path  # noqa: E402
from experiments.run_paper2_revision_analyses import (  # noqa: E402
    _binom_two_sided,
    _kappa,
    _mcnemar_exact,
    majority_mcnemar_by_tiebreak,
)

MODEL = "qwen2.5vl:7b"


def _load(tag: str, variant: str) -> dict[str, dict]:
    path = cache_path(tag)
    out: dict[str, dict] = {}
    if not path.is_file():
        return out
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if r.get("error") or r.get("model") != MODEL or r.get("prompt_variant") != variant:
            continue
        out[r["frame_id"]] = r
    return out


def _correct(r: dict) -> bool:
    return bool(r.get("action")) and r.get("action") == r.get("gt_action")


def summarize_variant(recs: dict[str, dict], episodes: dict[str, dict]) -> dict:
    rows = list(recs.values())
    n = len(rows)
    if n == 0:
        return {"n": 0}
    k = sum(_correct(r) for r in rows)
    maj = {}
    for uc in {r["use_case"] for r in rows}:
        # Ties (UC2/UC4 are 200/200) are broken by action name so the baseline is deterministic.
        counts = Counter(r["gt_action"] for r in rows if r["use_case"] == uc)
        maj[uc] = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]
    maj_ok = [r["gt_action"] == maj[r["use_case"]] for r in rows]
    b = sum(1 for r, m in zip(rows, maj_ok) if _correct(r) and not m)
    c = sum(1 for r, m in zip(rows, maj_ok) if m and not _correct(r))
    per_uc = {}
    for uc in sorted({r["use_case"] for r in rows}):
        s = [r for r in rows if r["use_case"] == uc]
        per_uc[uc] = {
            "n": len(s),
            "accuracy": sum(_correct(r) for r in s) / len(s),
            "distinct_actions": len({r.get("action") for r in s}),
            "action_counts": dict(Counter(r.get("action") for r in s).most_common()),
            "cohen_kappa": _kappa([(r.get("action"), r["gt_action"]) for r in s]),
        }
    by_sev = {}
    for sev in ("none", "minor", "major"):
        s = [r for r in rows if episodes.get(r["frame_id"], {}).get("severity") == sev]
        if s:
            by_sev[sev] = sum(_correct(r) for r in s) / len(s)
    tp = sum(1 for r in rows if r.get("defect_judgement") and r.get("defect_present"))
    fn = sum(1 for r in rows if not r.get("defect_judgement") and r.get("defect_present"))
    fp = sum(1 for r in rows if r.get("defect_judgement") and not r.get("defect_present"))
    tn = sum(1 for r in rows if not r.get("defect_judgement") and not r.get("defect_present"))
    tpr, fpr = tp / max(1, tp + fn), fp / max(1, fp + tn)
    lat = sorted(float(r["latency_ms"]) for r in rows)
    return {
        "n": n,
        "accuracy": k / n,
        "accuracy_wilson95": list(wilson_ci(k, n)),
        "p_vs_0.5_binomial": _binom_two_sided(k, n, 0.5),
        "majority_accuracy": sum(maj_ok) / n,
        "vs_majority_mcnemar": {"b_model_only": b, "c_majority_only": c, "p_exact": _mcnemar_exact(b, c)},
        "vs_majority_mcnemar_all_tiebreaks": majority_mcnemar_by_tiebreak(
            [(r["use_case"], r["gt_action"], bool(_correct(r))) for r in rows]),
        "by_severity": by_sev,
        "defect_judgement": {"TPR": tpr, "FPR": fpr, "balanced_accuracy": (tpr + 1 - fpr) / 2},
        "schema_valid_rate": sum(bool(r.get("schema_valid")) for r in rows) / n,
        "schema_valid_repaired_rate": sum(bool(r.get("schema_valid_repaired")) for r in rows) / n,
        "emits_bare_defect_absent": sum(
            1 for r in rows if any(str(e).strip() == "defect.absent" for e in (r.get("preconditions") or []))
        ) / n,
        "latency_median_ms": statistics.median(lat),
        "per_use_case": per_uc,
    }


def paired(a: dict[str, dict], b: dict[str, dict]) -> dict:
    shared = sorted(set(a) & set(b))
    if not shared:
        return {"n": 0}
    ca = [_correct(a[f]) for f in shared]
    cb = [_correct(b[f]) for f in shared]
    x = sum(1 for p, q in zip(ca, cb) if p and not q)
    y = sum(1 for p, q in zip(ca, cb) if q and not p)
    la = [float(a[f]["latency_ms"]) for f in shared]
    lb = [float(b[f]["latency_ms"]) for f in shared]
    return {
        "n": len(shared),
        "accuracy_a": sum(ca) / len(shared),
        "accuracy_b": sum(cb) / len(shared),
        "a_only_correct": x,
        "b_only_correct": y,
        "mcnemar_p_exact": _mcnemar_exact(x, y),
        "same_action": sum(a[f].get("action") == b[f].get("action") for f in shared) / len(shared),
        "same_correctness": sum(p == q for p, q in zip(ca, cb)) / len(shared),
        "latency_median_a_ms": statistics.median(la),
        "latency_median_b_ms": statistics.median(lb),
        "latency_crosses_4000ms": sum((p > 4000) != (q > 4000) for p, q in zip(la, lb)) / len(shared),
    }


def main() -> int:
    episodes = {e["frame_id"]: e for e in load_manifest()}
    base = _load("phase_p", "blind")
    v2 = _load("phase_p_v2", "blind_v2")
    rep = _load("phase_p_rep", "blind")
    out = {
        "model": MODEL,
        "blind_original": summarize_variant(base, episodes),
        "blind_v2": summarize_variant(v2, episodes),
        "v2_vs_original_same_frames": paired(v2, base),
        "repeat_vs_original_same_frames": paired(rep, base),
        "note": (
            "Revision runs on the shared L40 (2026-09-23) while another GPU job was active: "
            "latencies are not comparable with Phase P and are not replayed through B2."
        ),
    }
    path = RESULTS_DIR / "paper2_v2_analysis.json"
    path.write_text(json.dumps(out, indent=2))
    print(json.dumps({k: (v if not isinstance(v, dict) else {kk: vv for kk, vv in v.items()
                                                           if not isinstance(vv, dict)})
                      for k, v in out.items()}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
