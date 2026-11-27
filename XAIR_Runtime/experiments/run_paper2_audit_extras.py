#!/usr/bin/env python3
"""Checks added in the final audit (2026-10-02). Output: paper2_audit_extras.json.

  sensitivity_vs_schema  full XAIR minus schema+freshness in every sensitivity row of the B2 table
  break_even_conservative  lambda* when over-conservative publications cost mu instead of lambda
  decidable_grounding    grounding on frames whose action follows from defect presence alone
  recheck_components     queueing-delay replay with the recheck split into its age and its
                         predicate part, against the analytic share of drift inside the queue
  confidence_gate        publish only if the model's confidence reaches a threshold chosen on
                         training frames, with and without full XAIR
  qwen3_harness          break-even weight of Qwen3-VL 8B with the harness predicate
"""

from __future__ import annotations

import json
import math
import random
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.paper2_common import RESULTS_DIR, Measurement, classify, load_manifest, trial_reward  # noqa: E402
from experiments.paper2_splits import load_split  # noqa: E402
from experiments.run_b2_validity_frontier import gate_publishes, load_replay_records  # noqa: E402
from experiments.run_paper2_review_analyses import (  # noqa: E402
    MODELS,
    MU,
    SEEDS,
    W_MS,
    break_even,
    contrast,
    load_raw,
    rates,
    records_from,
    ref_trials,
    revoke_all,
)
from experiments.run_paper2_revision_analyses import _reward_conservative  # noqa: E402
from experiments.run_paper2_stats import generate_trials  # noqa: E402
from experiments.run_paper2_latency_factorial import with_age  # noqa: E402


def recs_for(split: dict, part: str, model: str = "qwen2.5vl:7b"):
    return [r for r in load_replay_records("phase_p", models={model}, variants={"blind"}, use_repaired=False)
            if split.get(r.frame_id) == part]


def sensitivity_vs_schema(split: dict, episodes: dict) -> dict:
    recs = recs_for(split, "test")
    out = {}
    cells = {
        "visa": dict(recs=[r for r in recs if episodes[r.frame_id]["source_dataset"] == "visa"]),
        "mvtec_ad": dict(recs=[r for r in recs if episodes[r.frame_id]["source_dataset"] == "mvtec_ad"]),
        "w2000": dict(freshness_ms=2000),
        "w8000": dict(freshness_ms=8000),
        "tau4000": dict(drift_offset_ms=4000),
    }
    for name, kw in cells.items():
        rr = kw.pop("recs", recs)
        w = kw.pop("freshness_ms", W_MS)
        x = generate_trials(rr, gate="xair", freshness_ms=w, **kw)
        s = generate_trials(rr, gate="schema_freshness", freshness_ms=w, **kw)
        f = generate_trials(rr, gate="freshness_only", freshness_ms=w, **kw)
        out[name] = {"xair_minus_schema_freshness": contrast(x, s), "xair_minus_freshness_only": contrast(x, f)}
    x = generate_trials(recs, gate="xair", freshness_ms=W_MS)
    s = generate_trials(recs, gate="schema_freshness", freshness_ms=W_MS)
    xc = [dict(t, reward=_reward_conservative(t)) for t in x]
    sc = [dict(t, reward=_reward_conservative(t)) for t in s]
    out["over_conservative_mu"] = {"xair_minus_schema_freshness": contrast(xc, sc)}
    return out


def break_even_conservative(split: dict) -> dict:
    """lambda* = ((1 + mu) SAR - mu H_cons) / H_perm, from per-frame means with a frame bootstrap."""
    out = {}
    for m in MODELS:
        ts = generate_trials(recs_for(split, "test", m), gate="xair", freshness_ms=W_MS)
        by: dict[str, list[float]] = {}
        for t in ts:
            d = by.setdefault(t["frame_id"], [0.0, 0.0, 0.0, 0])
            d[0] += bool(t["successful_actuation"])
            if t["hazardous_publish"]:
                cons = _reward_conservative(t) == -MU
                d[1] += cons
                d[2] += not cons
            d[3] += 1
        a = np.array([[v[0] / v[3], v[1] / v[3], v[2] / v[3]] for v in by.values()])

        def lam(x):
            s, hc, hp = x[..., 0], x[..., 1], x[..., 2]
            return np.where(hp > 0, ((1 + MU) * s - MU * hc) / np.maximum(hp, 1e-12), np.inf)

        point = float(lam(a.mean(axis=0)))
        idx = np.random.default_rng(0).integers(0, len(a), size=(2000, len(a)))
        boots = lam(a[idx].mean(axis=1))
        lo, hi = np.quantile(boots, [0.025, 0.975]) if np.isfinite(point) else (math.inf, math.inf)
        out[m] = {"lambda_star": point, "ci95": [float(lo), float(hi)]}
    return out


def decidable_grounding(episodes: dict) -> dict:
    out = {}
    raw = load_raw("phase_p", set(MODELS), {"blind"})
    for m in MODELS:
        rs = [r for r in raw if r["model"] == m and (not episodes[r["frame_id"]]["defect_present"]
                                                     or r["use_case"] in ("uc2_restart", "uc4_conflict"))]
        out[m] = {"n": len(rs), "accuracy": sum(r["action"] == r["gt_action"] for r in rs) / len(rs)}
    return out


def recheck_components(split: dict) -> dict:
    out = {}
    producers = {"qwen7b_blind": recs_for(split, "test"),
                 "detector_age1500ms": with_age([r for r in load_replay_records(
                     "detector_patchcore", models=None, variants=None, use_repaired=False)
                     if split.get(r.frame_id) == "test"], 1500.0)}
    for pname, recs in producers.items():
        out[pname] = {}
        for rate in (0.2, 1.0):
            for q in (250, 1000, 2000):
                arms = {a: [] for a in ("validate_only", "age_recheck_only", "predicate_recheck_only", "recheck")}
                for seed in SEEDS:
                    for rec in recs:
                        rng = random.Random(f"{seed}:{rec.frame_id}")
                        t_drift = rng.expovariate(rate) * 1000.0
                        t_v, t_g = rec.latency_ms, rec.latency_ms + q
                        valid = t_drift > t_g
                        for arm, t_age, t_pred in (("validate_only", t_v, t_v), ("age_recheck_only", t_g, t_v),
                                                   ("predicate_recheck_only", t_v, t_g), ("recheck", t_g, t_g)):
                            hold = rec.preconds_hold_drifted if t_drift <= t_pred else rec.preconds_hold_nominal
                            pub, _ = gate_publishes("xair", elapsed_ms=t_age, freshness_ms=W_MS,
                                                    preconds_hold=hold, schema_valid=rec.gate_schema_valid)
                            s = classify(gt_action=rec.gt_action, model_action=rec.model_action, published=pub,
                                         measurement=Measurement(valid, valid, 1, 1))
                            s.update(frame_id=rec.frame_id, seed=seed)
                            s["reward"] = trial_reward(s)
                            arms[arm].append(s)
                analytic = float(np.mean([math.exp(-rate * r.latency_ms / 1000) * (1 - math.exp(-rate * q / 1000))
                                          for r in recs]))
                out[pname][f"r{rate:g}_q{q}"] = {
                    "analytic_drift_in_queue_share": analytic,
                    "age_part": contrast(arms["age_recheck_only"], arms["validate_only"]),
                    "predicate_part": contrast(arms["predicate_recheck_only"], arms["validate_only"]),
                    "full": contrast(arms["recheck"], arms["validate_only"]),
                }
    return out


def confidence_gate(split: dict) -> dict:
    raw = load_raw("phase_p", {"qwen2.5vl:7b"}, {"blind"})
    conf = {r["frame_id"]: float(r.get("confidence") or 0.0) for r in raw}
    out = {"confidence_levels": sorted({round(c, 2) for c in conf.values()})}

    def gated(ts, t):
        res = []
        for x in ts:
            if x["published"] and conf.get(x["frame_id"], 0.0) < t:
                valid = x["context_valid_at_eval"]
                x = dict(x, published=False, successful_actuation=False, stale_publish=False, unsafe_publish=False,
                         hazardous_publish=False, wrongful_revoke=bool(valid and x["grounding_correct"]))
                x["reward"] = trial_reward(x)
            res.append(x)
        return res

    thresholds = sorted(set(out["confidence_levels"]) | {0.0, 1.01})
    for gate in ("schema_freshness", "xair"):
        train = generate_trials(recs_for(split, "train"), gate=gate, freshness_ms=W_MS)
        best = max(thresholds, key=lambda t: sum(x["reward"] for x in gated(train, t)))
        test = generate_trials(recs_for(split, "test"), gate=gate, freshness_ms=W_MS)
        g = gated(test, best)
        out[gate] = {"threshold": best, "test": rates(g),
                     "minus_ungated": contrast(g, test), "minus_revoke_all": contrast(g, revoke_all(test))}
    return out


def qwen3_harness(split: dict, episodes: dict) -> dict:
    raw = [r for r in load_raw("phase_p_qwen3", {"qwen3-vl:8b-instruct"}, {"blind"}) if split.get(r["frame_id"]) == "test"]

    def harness(ep, rec):
        rec["preconditions"] = ["line.state == 'RUN'"]

    ts = ref_trials(records_from(raw, episodes, split, rec_fn=harness), "xair")
    return {"own_ages": {"break_even": break_even(ts), "minus_revoke_all": contrast(ts, revoke_all(ts)), **rates(ts)}}


def traceability(split: dict) -> dict:
    """Values quoted in the manuscript that no other result file stores."""
    from experiments.run_b2_validity_frontier import replay_one
    from experiments.run_paper2_drift_timing import job_shop_all_sampler, job_shop_sampler, state_changes
    from experiments.run_paper2_stats import bootstrap_utility_frames

    det = with_age([r for r in load_replay_records("detector_patchcore", models=None, variants=None, use_repaired=False)
                    if split.get(r.frame_id) == "test"], 1500.0)
    out = {"detector_age1500ms_xair_U": bootstrap_utility_frames(ref_trials(det, "xair"), n_boot=1000, seed=0)}
    recs = recs_for(split, "test")
    changes = state_changes()
    for name, draw in (("job_shop", job_shop_sampler(changes)), ("job_shop_all", job_shop_all_sampler(changes))):
        ts = {}
        for g in ("schema_freshness", "xair"):
            rows = []
            for seed in SEEDS:
                for rec in recs:
                    rng = random.Random(f"{seed}:{rec.frame_id}")
                    tau = draw(rng) * 1000.0
                    t = replay_one(rec, gate=g, anchor="capture", freshness_ms=W_MS, invalid_at_submit=tau <= rec.latency_ms)
                    t["seed"] = seed
                    t["reward"] = trial_reward(t)
                    rows.append(t)
            ts[g] = rows
        out[f"{name}_xair_minus_schema_freshness"] = contrast(ts["xair"], ts["schema_freshness"])
    return out


def main() -> int:
    split = load_split()
    episodes = {e["frame_id"]: e for e in load_manifest()}
    out = {}
    for name, fn in (("sensitivity_vs_schema", lambda: sensitivity_vs_schema(split, episodes)),
                     ("break_even_conservative", lambda: break_even_conservative(split)),
                     ("decidable_grounding", lambda: decidable_grounding(episodes)),
                     ("recheck_components", lambda: recheck_components(split)),
                     ("confidence_gate", lambda: confidence_gate(split)),
                     ("qwen3_harness", lambda: qwen3_harness(split, episodes)),
                     ("traceability", lambda: traceability(split))):
        out[name] = fn()
        print(f"[audit] {name} done", flush=True)
    path = RESULTS_DIR / "paper2_audit_extras.json"
    path.write_text(json.dumps(out, indent=2, default=float) + "\n")
    print(f"[audit] wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
