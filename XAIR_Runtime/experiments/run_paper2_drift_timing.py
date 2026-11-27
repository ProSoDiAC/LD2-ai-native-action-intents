#!/usr/bin/env python3
"""Drift timing that is stochastic or taken from a real production log, instead of a fixed offset.

The B2 reference cell draws drift with probability p_drift at a fixed offset tau below every
latency, so every drawn drift precedes submission. Here the instant at which context is
invalidated is a random variable T measured from evidence capture, and an intent is invalid at
submission iff T <= its evidence age:

  poisson_r   T ~ Exponential(rate r) for r in {0.05, 0.2, 1.0} changes per second
  job_shop    T = residual time to the next machine-state change of a real job-shop event log
              (4TU.ResearchData "Production Analysis with Process Mining Technology",
              DOI 10.4121/uuid:68726926-5ac5-4fab-b873-ee76ea412399), from decision instants
              drawn uniformly over the log on every machine
  job_shop_all  the same log with the changes of all machines superposed, i.e. a cell whose
              context changes whenever any machine of the shop changes state (a fast bound)

The log supplies timing only: its idle/setup/busy changes stand in for the instants at which
an inspection cell's context would change, as an order-of-magnitude rate. Unlike the reference
cell, drift is not thinned by p_drift: every realization has a change at T.

Everything else is the B2 reference protocol (w = 4 s, capture anchoring, common random numbers
across paths, five seeds). Producers: the primary 7B model, the controlled upper-bound
producer (detector) at its own latency, and the same producer at a uniform 1.5 s evidence age.
Output: paper2_drift_timing.json.
"""

from __future__ import annotations

import bisect
import csv
import datetime as dt
import hashlib
import io
import json
import math
import random
import re
import sys
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments import paper2_common  # noqa: E402
from experiments.paper2_common import RESULTS_DIR, trial_reward  # noqa: E402
from experiments.paper2_splits import load_split  # noqa: E402
from experiments.run_b2_validity_frontier import cache_path, load_replay_records, replay_one  # noqa: E402
from experiments.run_paper2_latency_factorial import with_age  # noqa: E402
from experiments.run_paper2_stats import SEEDS, bootstrap_utility_frames, paired_frame_contrast  # noqa: E402

LOG_URL = "https://data.4tu.nl/file/67d71073-7ece-4455-9840-9d48ee480160/68a909ed-ea0c-4e30-841a-4bce336e8e8c"
LOG_SHA256 = "b3a43f457393bc9dd0c616db189d01aa55e823f202cc8ce0cd537572840644cf"
LOG_CACHE = ROOT / "experiments" / "datasets" / "raw" / "production_log.zip"
RATES = (0.05, 0.2, 1.0)
GATES = ("direct", "freshness_only", "xair")
WINDOW_MS = 4000
N_BOOT = 1000
N_PAIRED = 2000


def production_log() -> Path:
    if not LOG_CACHE.exists():
        LOG_CACHE.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(LOG_URL, timeout=120) as r:
            LOG_CACHE.write_bytes(r.read())
    digest = hashlib.sha256(LOG_CACHE.read_bytes()).hexdigest()
    if digest != LOG_SHA256:
        raise SystemExit(f"{LOG_CACHE}: SHA-256 {digest} does not match the published archive")
    return LOG_CACHE


def state_changes() -> list[list[float]]:
    """Per machine, the sorted instants (seconds from the first event) at which its state changes."""
    def t(s: str) -> float:
        return dt.datetime.strptime(s, "%Y/%m/%d %H:%M:%S.%f").replace(tzinfo=dt.timezone.utc).timestamp()

    with zipfile.ZipFile(production_log()) as z, z.open("Production_Data.csv") as f:
        rows = list(csv.DictReader(io.TextIOWrapper(f, encoding="latin-1")))
    steps = [(re.sub(r"\W+", "_", r["Resource"].strip().lower()), t(r["Start Timestamp"]),
              t(r["Complete Timestamp"]), r["Report Type"]) for r in rows]
    t0 = min(a for _, a, _, _ in steps)
    points: dict[str, list[tuple[float, int, int]]] = {}
    for m, a, b, kind in steps:
        points.setdefault(m, []).extend([(a - t0, 1, int(kind == "S")), (b - t0, -1, -int(kind == "S"))])
    out = []
    for pts in points.values():
        active = setup = 0
        last, times = "IDLE", []
        for time, d_a, d_s in sorted(pts):
            active += d_a
            setup += d_s
            s = "IDLE" if active <= 0 else "SETUP" if setup > 0 else "BUSY"
            if s != last:
                times.append(time)
                last = s
        out.append(times)
    return out


def job_shop_sampler(changes: list[list[float]]):
    """Residual time to the next state change after a uniform decision instant on a random machine."""
    usable = [c for c in changes if len(c) > 1]

    def draw(rng: random.Random) -> float:
        c = usable[rng.randrange(len(usable))]
        now = rng.uniform(c[0], c[-1])
        return c[bisect.bisect_right(c, now)] - now if now < c[-1] else math.inf

    return draw


def job_shop_all_sampler(changes: list[list[float]]):
    """Residual time to the next change of any machine after a uniform instant over the log."""
    allc = sorted(t for c in changes for t in c)

    def draw(rng: random.Random) -> float:
        now = rng.uniform(allc[0], allc[-1])
        return allc[bisect.bisect_right(allc, now)] - now

    return draw


def trials(recs, gate: str, draw) -> list[dict]:
    out = []
    for seed in SEEDS:
        for rec in recs:
            rng = random.Random(f"{seed}:{rec.frame_id}")  # the same plant realization for every path
            tau_ms = draw(rng) * 1000.0
            scored = replay_one(rec, gate=gate, anchor="capture", freshness_ms=WINDOW_MS,
                                invalid_at_submit=tau_ms <= rec.latency_ms)
            scored["seed"] = seed
            scored["reward"] = trial_reward(scored)
            out.append(scored)
    return out


def revoke_all(ts: list[dict]) -> list[dict]:
    mu = paper2_common.MU_WRONGFUL_REVOKE
    return [dict(t, successful_actuation=0, hazardous_publish=0,
                 wrongful_revoke=int(bool(t["successful_actuation"]) or bool(t["wrongful_revoke"])),
                 reward=-mu * (bool(t["successful_actuation"]) or bool(t["wrongful_revoke"]))) for t in ts]


def cell(recs, draw) -> dict:
    ts = {g: trials(recs, g, draw) for g in GATES}
    ra = revoke_all(ts["xair"])
    res = {}
    for g, t in (*ts.items(), ("revoke_all", ra)):
        n = len(t)
        res[g] = {**bootstrap_utility_frames(t, n_boot=N_BOOT, seed=0),
                  "SAR": sum(bool(x["successful_actuation"]) for x in t) / n,
                  "hazard": sum(bool(x["hazardous_publish"]) for x in t) / n}
    res["invalid_at_submit_share"] = sum(1 for x in ts["xair"] if x.get("context_valid_at_eval") is False) / len(ts["xair"])
    res["xair_minus_freshness_only"] = paired_frame_contrast(ts["xair"], ts["freshness_only"], fields=("reward",),
                                                             n_boot=N_PAIRED)["fields"]["U"]
    res["xair_minus_revoke_all"] = paired_frame_contrast(ts["xair"], ra, fields=("reward",), n_boot=N_PAIRED)["fields"]["U"]
    return res


def main() -> int:
    split = load_split()
    producers = {"qwen7b_blind": ("phase_p", {"qwen2.5vl:7b"}, {"blind"})}
    if cache_path("detector_patchcore").is_file():
        producers["patchcore_detector"] = ("detector_patchcore", None, None)
        producers["patchcore_detector_age1500ms"] = ("detector_patchcore", None, None)
    changes = state_changes()
    js = job_shop_sampler(changes)
    ja = job_shop_all_sampler(changes)
    probe = random.Random(0)
    residual = sorted(js(probe) for _ in range(20000))
    residual_all = sorted(ja(probe) for _ in range(20000))
    out = {
        "window_ms": WINDOW_MS,
        "job_shop_log": {"machines": len(changes), "state_changes": sum(len(c) for c in changes),
                         "residual_s_quantiles": {q: residual[int(q * (len(residual) - 1))] for q in (0.01, 0.05, 0.5)},
                         "residual_all_machines_s_quantiles": {
                             q: residual_all[int(q * (len(residual_all) - 1))] for q in (0.01, 0.05, 0.5)}},
        "cells": {},
    }
    samplers = {f"poisson_{r:g}": (lambda rate: (lambda rng: rng.expovariate(rate)))(r) for r in RATES}
    samplers["job_shop"] = js
    samplers["job_shop_all"] = ja
    for name, (tag, models, variants) in producers.items():
        recs = [r for r in load_replay_records(tag, models=models, variants=variants, use_repaired=False)
                if split.get(r.frame_id) == "test"]
        if name.endswith("_age1500ms"):
            recs = with_age(recs, 1500.0)
        out["cells"][name] = {s: cell(recs, draw) for s, draw in samplers.items()}
        print(f"[drift-timing] {name} done", flush=True)
    path = RESULTS_DIR / "paper2_drift_timing.json"
    path.write_text(json.dumps(out, indent=2, default=float) + "\n")
    print(f"[drift-timing] wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
