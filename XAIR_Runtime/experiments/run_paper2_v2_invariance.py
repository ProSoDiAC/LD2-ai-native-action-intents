#!/usr/bin/env python3
"""Do the B1-B5 conclusions survive the corrected task texts (blind_v2) for every model?

The primary campaign used the original blind task texts, which omitted some ground-truth
actions and biased UC4. The ``phase_p_v2`` cache holds one blind_v2 decision per frame and
model. The corrected run shared the accelerator with other jobs, so its latencies are not
those of Phase P. Two derived caches separate the prompt from the latency:

  phase_p_v2_matched   blind_v2 decisions with the Phase-P evidence age of the same
                       (frame, model): only the decision changes (primary comparison)
  phase_p_v2_own       blind_v2 decisions with their own measured latencies

Both relabel the variant as ``blind`` so that the unchanged B3-B5 scripts replay them.
For each model the script compares B1 grounding, and for each replayed cache the B2
reference cell (w = 4 s, p_drift = 0.5, tau = 250 ms, capture anchoring, five seeds, test
frames): utility per path, XAIR minus freshness-only, XAIR minus revoke-all, and the
emitted-minus-harness precondition contrast. With ``--adaptive`` it also reruns B3, B4 and
B5 on the matched cache (output suffix ``_v2``) and collects their headline contrasts.

Output: paper2_v2_invariance.json.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.paper2_common import RESULTS_DIR, load_manifest, wilson_ci  # noqa: E402
from experiments.paper2_splits import load_split  # noqa: E402
from experiments.perception_cache import cache_path  # noqa: E402
from experiments.run_b2_validity_frontier import load_replay_records  # noqa: E402
from experiments.run_paper2_latency_factorial import revoke_all  # noqa: E402
from experiments.run_paper2_revision_analyses import (  # noqa: E402
    _binom_two_sided, _mcnemar_exact, b2_harness_echo, majority_mcnemar_by_tiebreak,
)
from experiments.run_paper2_stats import (  # noqa: E402
    bootstrap_utility_frames, generate_trials, paired_frame_contrast,
)

MODELS = ("qwen2.5vl:7b", "qwen2.5vl:32b", "qwen2.5vl:3b", "gemma3:12b", "llama3.2-vision:11b")
GATES = ("direct", "freshness_only", "xair")
N_BOOT = 1000
N_PAIRED = 2000
TIMING = ("capture_ts", "emitted_ts", "latency_ms")


def raw(tag: str, variant: str) -> dict[tuple[str, str], dict]:
    out = {}
    for line in cache_path(tag).read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if r.get("error") or r.get("prompt_variant") != variant:
            continue
        out[(r["frame_id"], r["model"])] = r  # the last write wins, as in the replay loader
    return out


def derived_caches(orig: dict, v2: dict) -> dict[str, int]:
    matched, own = [], []
    for key, r in sorted(v2.items()):
        own.append(dict(r, prompt_variant="blind", source_variant="blind_v2"))
        if key in orig:
            matched.append(dict(r, prompt_variant="blind", source_variant="blind_v2",
                                **{k: orig[key].get(k) for k in TIMING}))
    for tag, rows in (("phase_p_v2_matched", matched), ("phase_p_v2_own", own)):
        cache_path(tag).write_text("".join(json.dumps(r) + "\n" for r in rows))
    return {"phase_p_v2_matched": len(matched), "phase_p_v2_own": len(own)}


def correct(r: dict) -> bool:
    return bool(r.get("action")) and r.get("action") == r.get("gt_action")


def b1(orig: dict, v2: dict, model: str) -> dict:
    a = {f: r for (f, m), r in orig.items() if m == model}
    b = {f: r for (f, m), r in v2.items() if m == model}
    if not b:
        return {"n_v2": 0}

    def summary(rows: list[dict]) -> dict:
        k, n = sum(map(correct, rows)), len(rows)
        tp = sum(1 for r in rows if r.get("defect_judgement") and r.get("defect_present"))
        pos = sum(1 for r in rows if r.get("defect_present"))
        fp = sum(1 for r in rows if r.get("defect_judgement") and not r.get("defect_present"))
        return {
            "n": n, "accuracy": k / n, "wilson95": list(wilson_ci(k, n)), "p_vs_0.5": _binom_two_sided(k, n, 0.5),
            "vs_majority_all_tiebreaks": majority_mcnemar_by_tiebreak(
                [(r["use_case"], r["gt_action"], correct(r)) for r in rows]),
            "defect_balanced_accuracy": (tp / max(1, pos) + 1 - fp / max(1, n - pos)) / 2,
            "schema_valid_rate": sum(bool(r.get("schema_valid")) for r in rows) / n,
        }

    shared = sorted(set(a) & set(b))
    x = sum(1 for f in shared if correct(b[f]) and not correct(a[f]))
    y = sum(1 for f in shared if correct(a[f]) and not correct(b[f]))
    return {"blind": summary(list(a.values())), "blind_v2": summary(list(b.values())),
            "paired": {"n": len(shared), "v2_only_correct": x, "blind_only_correct": y,
                       "mcnemar_p": _mcnemar_exact(x, y),
                       "same_action": sum(a[f].get("action") == b[f].get("action") for f in shared) / len(shared)}}


def b2(tag: str, model: str, split: dict, episodes: dict) -> dict:
    recs = [r for r in load_replay_records(tag, models={model}, variants={"blind"}, use_repaired=False)
            if split.get(r.frame_id) == "test"]
    if not recs:
        return {"n_frames": 0}
    ts = {g: generate_trials(recs, gate=g, freshness_ms=4000) for g in GATES}
    ra = revoke_all(ts["xair"])
    out = {"n_frames": len(recs)}
    for g, t in (*ts.items(), ("revoke_all", ra)):
        out[g] = bootstrap_utility_frames(t, n_boot=N_BOOT, seed=0)["U"]
    for name, other in (("freshness_only", ts["freshness_only"]), ("revoke_all", ra)):
        c = paired_frame_contrast(ts["xair"], other, fields=("reward",), n_boot=N_PAIRED)["fields"]["U"]
        out[f"xair_minus_{name}"] = c
    echo = b2_harness_echo(recs, episodes, N_BOOT)["model_minus_echo"]
    out["emitted_minus_harness"] = echo["fields"]["U"] if "fields" in echo else echo
    return out


def adaptive(tag: str) -> dict:
    """Rerun B3-B5 on ``tag`` (outputs suffixed _v2) and the paper's headline contrasts."""
    from collections import defaultdict

    for script in ("run_b3_validity_budget.py", "run_b4_model_routing.py", "run_b5_agent_policy.py"):
        subprocess.run([sys.executable, str(ROOT / "experiments" / script), "--tag", tag, "--out-suffix", "_v2"],
                       cwd=ROOT, check=True)

    def trials(name: str, setting: str, key: str) -> dict[str, list[dict]]:
        by = defaultdict(list)
        for line in (RESULTS_DIR / f"{name}_v2.jsonl").read_text().splitlines():
            r = json.loads(line)
            if r.get("setting") == setting and not r.get("privileged"):
                by[str(r[key])].append(r)
        return by

    def contrast(a: list[dict], b: list[dict]) -> dict:
        return paired_frame_contrast(a, b, fields=("reward",), n_boot=N_PAIRED)["fields"]["U"]

    def utility(ts: list[dict]) -> float:
        return sum(t["reward"] for t in ts) / len(ts)

    b3, b4, b5 = (trials("b3_eval_trials", "headline", "policy"), trials("b4_eval_trials", "p50", "router"),
                  trials("b5_eval_trials", "headline", "policy"))
    static = {k: utility(v) for k, v in b4.items() if k.startswith("static:")}
    best = max(static, key=static.get)
    return {
        "b3": {"U": {k: utility(v) for k, v in b3.items()},
               **{f"{a}_minus_best_fixed": contrast(b3[a], b3["best_fixed:train_selected"])
                  for a in ("linucb:a1.0", "qlearn:e0.15", "oracle")}},
        "b4": {"U": {k: utility(v) for k, v in b4.items()}, "best_static": best,
               "best_static_minus_always_7b": contrast(b4[best], b4["static:qwen2.5vl:7b"])},
        "b5": {"U": {k: utility(v) for k, v in b5.items()},
               **{f"{a}_minus_single_shot": contrast(b5[a], b5["single_shot"])
                  for a in ("always_reobserve", "retry_stale", "always_escalate", "qlearn:e0.1", "linucb:a0.75")}},
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--adaptive", action="store_true", help="also rerun B3-B5 on the matched cache")
    args = ap.parse_args()
    episodes = {e["frame_id"]: e for e in load_manifest()}
    split = load_split()
    orig, v2 = raw("phase_p", "blind"), raw("phase_p_v2", "blind_v2")
    out: dict = {"derived_caches": derived_caches(orig, v2), "models": {}}
    for m in MODELS:
        if not any(k[1] == m for k in v2):
            print(f"[v2] no blind_v2 decisions for {m}", flush=True)
            continue
        out["models"][m] = {
            "b1": b1(orig, v2, m),
            "b2_blind": b2("phase_p", m, split, episodes),
            "b2_v2_matched": b2("phase_p_v2_matched", m, split, episodes),
            "b2_v2_own_latency": b2("phase_p_v2_own", m, split, episodes),
        }
        print(f"[v2] {m} done", flush=True)
    if args.adaptive:
        out["adaptive_v2_matched"] = adaptive("phase_p_v2_matched")
    path = RESULTS_DIR / "paper2_v2_invariance.json"
    path.write_text(json.dumps(out, indent=2, default=float) + "\n")
    print(f"[v2] wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
