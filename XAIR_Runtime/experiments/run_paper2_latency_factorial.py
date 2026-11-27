#!/usr/bin/env python3
"""Latency factorial: evidence age assigned independently of source, model and correctness.

In the preserved campaign latency is observed, and for the 7B model it is almost a
function of the image source (VisA decisions are slower). Here every preserved decision
is replayed at each of several fixed evidence ages, so age is crossed with source,
model and correctness by construction. Everything else is the B2 protocol: the same
decisions, gates, anchors, drift draws (common random numbers) and utility.

Outputs ``paper2_latency_factorial.json``:
  cells      rates and U per (producer, age, window, drift offset, anchor, gate, source)
  contrasts  paired frame-cluster XAIR minus freshness-only and XAIR minus revoke-all
             per (producer, age, source) at w = 4 s, tau = 250 ms, capture anchoring
  permuted   the same contrasts when the observed evidence ages are randomly permuted
             across decisions (20 permutations), which keeps the latency distribution
             but removes its association with source, model and correctness

Producers: the primary 7B model (blind) and, if its cache exists, the controlled upper-bound
detector of run_detector_analysis.py.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments import paper2_common  # noqa: E402
from experiments.paper2_common import RESULTS_DIR, load_manifest  # noqa: E402
from experiments.paper2_splits import load_split  # noqa: E402
from experiments.run_b2_validity_frontier import ReplayRecord, cache_path, load_replay_records  # noqa: E402
from experiments.run_paper2_stats import generate_trials, paired_frame_contrast  # noqa: E402

AGES_MS = (500, 1000, 2000, 3000, 4000, 6000, 8000)
WINDOWS_MS = (2000, 4000, 8000)
OFFSETS_MS = (250, 2000)
ANCHORS = ("capture", "emission")
GATES = ("direct", "freshness_only", "xair")
PRODUCERS = {
    "qwen7b_blind": ("phase_p", {"qwen2.5vl:7b"}, {"blind"}),
    "patchcore_detector": ("detector_patchcore", None, None),
}
N_PAIRED = 2000
N_PERMUTATIONS = 20


def with_age(records: list[ReplayRecord], age_ms: float) -> list[ReplayRecord]:
    out = []
    for r in records:
        c = object.__new__(ReplayRecord)
        for s in ReplayRecord.__slots__:
            setattr(c, s, getattr(r, s))
        c.latency_ms = float(age_ms)
        out.append(c)
    return out


def rates(trials: list[dict]) -> dict:
    n = len(trials)
    opp = sum(bool(t["successful_actuation"]) or bool(t["wrongful_revoke"]) for t in trials)
    return {
        "n": n,
        "U": sum(t["reward"] for t in trials) / n,
        "SAR": sum(bool(t["successful_actuation"]) for t in trials) / n,
        "SER": sum(bool(t["stale_publish"]) for t in trials) / n,
        "hazard": sum(bool(t["hazardous_publish"]) for t in trials) / n,
        "WRR": sum(bool(t["wrongful_revoke"]) for t in trials) / opp if opp else 0.0,
        "U_revoke_all": -paper2_common.MU_WRONGFUL_REVOKE * opp / n,
    }


def revoke_all(trials: list[dict]) -> list[dict]:
    out = []
    for t in trials:
        opp = bool(t["successful_actuation"]) or bool(t["wrongful_revoke"])
        out.append(dict(t, successful_actuation=0, hazardous_publish=0, wrongful_revoke=int(opp),
                        reward=-paper2_common.MU_WRONGFUL_REVOKE * opp))
    return out


def permuted(recs: list[ReplayRecord], source: dict, n_paired: int) -> dict:
    """XAIR minus freshness-only by source with evidence ages permuted across decisions."""
    import random

    ages = [r.latency_ms for r in recs]
    fo_all, xa_all = [], []
    for k in range(N_PERMUTATIONS):
        shuffled = ages[:]
        random.Random(1000 + k).shuffle(shuffled)
        aged = with_age(recs, 0.0)
        for r, a in zip(aged, shuffled):
            r.latency_ms = a
        for gate, sink in (("freshness_only", fo_all), ("xair", xa_all)):
            for t in generate_trials(aged, gate=gate, freshness_ms=4000, drift_offset_ms=250):
                sink.append(dict(t, seed=(k, t["seed"])))
    res = {"n_permutations": N_PERMUTATIONS}
    for src in ("all", "visa", "mvtec_ad"):
        sel = (lambda t: True) if src == "all" else (lambda t, s=src: source[t["frame_id"]] == s)
        f_s, x_s = [t for t in fo_all if sel(t)], [t for t in xa_all if sel(t)]
        res[src] = {
            "freshness_only_publish_rate": sum(bool(t["published"]) for t in f_s) / len(f_s),
            "xair_minus_freshness_only": paired_frame_contrast(x_s, f_s, fields=("reward",),
                                                               n_boot=n_paired)["fields"]["U"],
        }
    return res


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-paired", type=int, default=N_PAIRED)
    args = ap.parse_args()
    split = load_split()
    source = {e["frame_id"]: e["source_dataset"] for e in load_manifest()}
    out: dict = {"ages_ms": AGES_MS, "windows_ms": WINDOWS_MS, "offsets_ms": OFFSETS_MS,
                 "p_drift": 0.5, "cells": {}, "contrasts": {}}
    for name, (tag, models, variants) in PRODUCERS.items():
        if not cache_path(tag).is_file():
            print(f"[latency] skip {name}: no cache {tag}")
            continue
        recs = [r for r in load_replay_records(tag, models=models, variants=variants, use_repaired=False)
                if split.get(r.frame_id) == "test"]
        out["cells"][name], out["contrasts"][name] = {}, {}
        for age in AGES_MS:
            aged = with_age(recs, age)
            for w in WINDOWS_MS:
                for tau in OFFSETS_MS:
                    for anchor in ANCHORS:
                        for gate in GATES:
                            trials = generate_trials(aged, gate=gate, freshness_ms=w, drift_offset_ms=tau,
                                                     anchor=anchor)
                            key = f"a{age}_w{w}_tau{tau}_{anchor}_{gate}"
                            out["cells"][name][key] = {
                                "all": rates(trials),
                                **{src: rates([t for t in trials if source[t["frame_id"]] == src])
                                   for src in ("visa", "mvtec_ad")},
                            }
            fo = generate_trials(aged, gate="freshness_only", freshness_ms=4000, drift_offset_ms=250)
            xa = generate_trials(aged, gate="xair", freshness_ms=4000, drift_offset_ms=250)
            cell = {}
            for src in ("all", "visa", "mvtec_ad"):
                sel = (lambda t: True) if src == "all" else (lambda t, s=src: source[t["frame_id"]] == s)
                f_s, x_s = [t for t in fo if sel(t)], [t for t in xa if sel(t)]
                cell[src] = {
                    "xair_minus_freshness_only": paired_frame_contrast(x_s, f_s, fields=("reward",),
                                                                       n_boot=args.n_paired)["fields"]["U"],
                    "xair_minus_revoke_all": paired_frame_contrast(x_s, revoke_all(x_s), fields=("reward",),
                                                                   n_boot=args.n_paired)["fields"]["U"],
                }
            out["contrasts"][name][f"a{age}"] = cell
            print(f"[latency] {name} age {age} ms done", flush=True)
        out.setdefault("permuted", {})[name] = permuted(recs, source, args.n_paired)
    path = RESULTS_DIR / "paper2_latency_factorial.json"
    path.write_text(json.dumps(out, indent=2) + "\n")
    print(f"[latency] wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
