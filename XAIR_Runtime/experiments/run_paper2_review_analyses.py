#!/usr/bin/env python3
"""Offline analyses added in the pre-submission review (2026-10-02). All replay the preserved
decisions; no inference is run. Output: paper2_review_analyses.json.

  schema_baseline   a "schema + freshness" path (malformed intents rejected, no preconditions)
                    at the B2 reference cell, for every model, so that the contribution of
                    preconditions is separated from that of schema filtering
  defect_signal     the evaluation context exposes defect.absent derived from ground truth; we
                    replace it by the controlled detector's estimate (a realistic upstream
                    inspection signal) or drop the emitted preconditions that reference it
  hidden_predicate  drift on a context path the prompt never shows (cell.interlock): emitted
                    preconditions cannot cover it; an integrator-declared predicate can
  recheck           a queueing delay q between validation (t_v) and gateway release (t_g), with
                    drift times T ~ Exp(r): validation only at t_v versus the recheck at t_g
  drift_curve       the XAIR contrasts over a finer grid of Poisson drift rates, plus the job-shop
                    log with decision instants restricted to periods in which machines operate
  break_even        lambda* at which full XAIR equals revoke-all: (1 + mu) SAR / Hazard
  grounding_extra   attainable-accuracy ceiling, VLM defect judgments mapped through the
                    use-case table, the "always nominal action" baseline, and the frames of
                    the resolution ablation
"""

from __future__ import annotations

import bisect
import json
import math
import random
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments import paper2_common  # noqa: E402
from experiments.paper2_common import RESULTS_DIR, Measurement, classify, load_manifest, trial_reward  # noqa: E402
from experiments.paper2_splits import load_split  # noqa: E402
from experiments.run_b2_validity_frontier import (  # noqa: E402
    ReplayRecord,
    cache_path,
    gate_publishes,
    load_replay_records,
    mcnemar_exact_p,
)
from experiments.run_paper2_drift_timing import production_log  # noqa: E402
from experiments.run_paper2_latency_factorial import with_age  # noqa: E402
from experiments.run_paper2_stats import (  # noqa: E402
    SEEDS,
    bootstrap_utility_frames,
    generate_trials,
    paired_frame_contrast,
)

W_MS = 4000
MU = paper2_common.MU_WRONGFUL_REVOKE
LAM = paper2_common.LAMBDA_HAZARD
N_PAIRED = 2000
MODELS = ("qwen2.5vl:7b", "qwen2.5vl:32b", "qwen2.5vl:3b", "gemma3:12b", "llama3.2-vision:11b")
UC_TABLE = {
    "uc1_triage": ("ACCEPT", "HOLD_FOR_OPERATOR", "REJECT_TO_BIN"),
    "uc2_restart": ("RESUME", "STOP", "STOP"),
    "uc3_speed": ("ACCEPT", "SLOW_DOWN", "STOP"),
    "uc4_conflict": ("RESUME", "STOP", "STOP"),
    "uc5_safety": ("ACCEPT", "HOLD_FOR_OPERATOR", "E_STOP"),
}
RATES_CURVE = (0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def contrast(a: list[dict], b: list[dict]) -> dict:
    c = paired_frame_contrast(a, b, fields=("reward",), n_boot=N_PAIRED)["fields"]["U"]
    return {"delta": c["mean_delta_a_minus_b"], "ci95": c["ci95"]}


def rates(ts: list[dict]) -> dict:
    n = len(ts)
    out = {k: sum(bool(t.get(f)) for t in ts) / n for k, f in (
        ("SAR", "successful_actuation"), ("hazard", "hazardous_publish"), ("stale", "stale_publish"),
        ("unsafe", "unsafe_publish"), ("wrongful", "wrongful_revoke"))}
    cv = [t for t in ts if t.get("context_valid_at_eval") and t.get("grounding_correct")]
    out["WRR"] = sum(bool(t.get("wrongful_revoke")) for t in cv) / len(cv) if cv else 0.0
    out["U"] = sum(t["reward"] for t in ts) / n
    return out


def summary(ts: list[dict]) -> dict:
    b = bootstrap_utility_frames(ts, n_boot=1000, seed=0)
    return {**rates(ts), "U_ci95": b["U_ci95"]}


def revoke_all(ts: list[dict]) -> list[dict]:
    out = []
    for t in ts:
        cv = bool(t["successful_actuation"]) or bool(t["wrongful_revoke"])
        out.append(dict(t, successful_actuation=0, hazardous_publish=0, stale_publish=0, unsafe_publish=0,
                        wrongful_revoke=int(cv), reward=-MU * cv))
    return out


def break_even(ts: list[dict], n_boot: int = 2000) -> dict:
    """lambda* with U_path(lambda*) = U_revoke_all: (1 + mu) SAR / Hazard, frame-cluster CI."""
    by: dict[str, list[float]] = {}
    for t in ts:
        d = by.setdefault(t["frame_id"], [0.0, 0.0, 0])
        d[0] += bool(t["successful_actuation"])
        d[1] += bool(t["hazardous_publish"])
        d[2] += 1
    arr = np.array([[v[0] / v[2], v[1] / v[2]] for v in by.values()])
    point = (1 + MU) * arr[:, 0].mean() / arr[:, 1].mean() if arr[:, 1].mean() > 0 else math.inf
    rng = np.random.default_rng(0)
    idx = rng.integers(0, len(arr), size=(n_boot, len(arr)))
    s, h = arr[idx, 0].mean(axis=1), arr[idx, 1].mean(axis=1)
    with np.errstate(divide="ignore"):
        boots = np.where(h > 0, (1 + MU) * s / np.maximum(h, 1e-12), np.inf)
    lo, hi = np.quantile(boots, [0.025, 0.975])
    return {"lambda_star": float(point), "ci95": [float(lo), float(hi)]}


def load_raw(tag: str, models: set[str] | None, variants: set[str] | None) -> list[dict]:
    latest: dict[tuple, dict] = {}
    for line in cache_path(tag).read_text().splitlines():
        if not line.strip():
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        if (models and rec.get("model") not in models) or (variants and rec.get("prompt_variant") not in variants):
            continue
        if rec.get("error"):
            continue
        latest[(rec["frame_id"], rec.get("model"), rec.get("prompt_variant"), rec.get("use_case"))] = rec
    return list(latest.values())


def records_from(raw: list[dict], episodes: dict, split: dict, *, ep_fn=None, rec_fn=None) -> list[ReplayRecord]:
    out = []
    for rec in raw:
        if split.get(rec["frame_id"]) != "test":
            continue
        ep = json.loads(json.dumps(episodes[rec["frame_id"]]))
        rec = json.loads(json.dumps(rec))
        if ep_fn:
            ep_fn(ep, rec)
        if rec_fn:
            rec_fn(ep, rec)
        out.append(ReplayRecord(rec, ep, use_repaired=False))
    out.sort(key=lambda r: (r.frame_id, r.model, r.prompt_variant))
    return out


def ref_trials(recs: list[ReplayRecord], gate: str) -> list[dict]:
    return generate_trials(recs, gate=gate, freshness_ms=W_MS)


def test_records(tag: str, models: set[str] | None, variants: set[str] | None, split: dict) -> list[ReplayRecord]:
    return [r for r in load_replay_records(tag, models=models, variants=variants, use_repaired=False)
            if split.get(r.frame_id) == "test"]


# ---------------------------------------------------------------------------
# analyses
# ---------------------------------------------------------------------------


def schema_baseline(split: dict) -> dict:
    out = {}
    for m in MODELS:
        recs = test_records("phase_p", {m}, {"blind"}, split)
        ts = {g: ref_trials(recs, g) for g in ("freshness_only", "schema_freshness", "xair")}
        ra = revoke_all(ts["xair"])
        out[m] = {
            "schema_freshness": summary(ts["schema_freshness"]),
            "xair": summary(ts["xair"]),
            "schema_minus_freshness_only": contrast(ts["schema_freshness"], ts["freshness_only"]),
            "xair_minus_schema_freshness": contrast(ts["xair"], ts["schema_freshness"]),
            "schema_freshness_minus_revoke_all": contrast(ts["schema_freshness"], ra),
        }
    return out


def defect_signal(split: dict, episodes: dict) -> dict:
    """Replace the ground-truth defect flag by the detector's estimate, or drop the emitted
    preconditions on it (malformed ones stay malformed: the recorded schema verdict is kept)."""
    det = {r["frame_id"]: bool(r.get("defect_judgement")) for r in load_raw("detector_patchcore", None, None)}

    def noisy(ep, rec):
        ep["context"]["defect"]["absent"] = not det[rec["frame_id"]]

    def drop(ep, rec):
        rec["preconditions"] = [p for p in rec.get("preconditions") or [] if not p.strip().startswith("defect.")]

    out = {}
    for m in MODELS:
        raw = load_raw("phase_p", {m}, {"blind"})
        res = {"n_test_emissions_with_defect_precondition": sum(
            1 for r in raw if split.get(r["frame_id"]) == "test"
            and any(p.strip().startswith("defect.") for p in r.get("preconditions") or []))}
        base = records_from(raw, episodes, split)
        fo = ref_trials(base, "freshness_only")
        xb = ref_trials(base, "xair")
        res["ground_truth_flag"] = {**summary(xb), "minus_freshness_only": contrast(xb, fo)}
        for name, kw in (("detector_flag", {"ep_fn": noisy}), ("defect_preconditions_dropped", {"rec_fn": drop})):
            xt = ref_trials(records_from(raw, episodes, split, **kw), "xair")
            res[name] = {**summary(xt), "minus_freshness_only": contrast(xt, fo),
                         "minus_ground_truth_flag": contrast(xt, xb)}
        out[m] = res
    return out


def hidden_predicate(split: dict, episodes: dict) -> dict:
    """Drift on cell.interlock, a path the prompt never names."""
    mandated = "cell.interlock == 'CLEAR'"

    def ep_fn(ep, rec):
        ep["context"]["cell"] = {"interlock": "CLEAR"}
        ep["drift_patch"] = {"cell": {"interlock": "ENGAGED"}}
        ep["hazard_predicate"] = mandated

    def add_mandated(ep, rec):
        ep_fn(ep, rec)
        rec["preconditions"] = list(rec.get("preconditions") or []) + [mandated]

    out = {}
    for name, tag, models, variants in (("qwen7b_blind", "phase_p", {"qwen2.5vl:7b"}, {"blind"}),
                                        ("detector", "detector_patchcore", None, None)):
        raw = load_raw(tag, models, variants)
        emitted = records_from(raw, episodes, split, ep_fn=ep_fn)
        withm = records_from(raw, episodes, split, rec_fn=add_mandated)
        if name == "detector":
            emitted, withm = with_age(emitted, 1500.0), with_age(withm, 1500.0)
        fo = ref_trials(emitted, "freshness_only")
        sf = ref_trials(emitted, "schema_freshness")
        xe = ref_trials(emitted, "xair")
        xm = ref_trials(withm, "xair")
        ra = revoke_all(xe)
        out[name] = {
            "emitted_paths_referencing_hidden": sum(
                1 for r in raw if split.get(r["frame_id"]) == "test"
                and any("cell." in p for p in r.get("preconditions") or [])),
            "freshness_only": summary(fo), "schema_freshness": summary(sf),
            "xair_emitted": summary(xe), "xair_mandated": summary(xm), "revoke_all": summary(ra),
            "xair_emitted_minus_schema_freshness": contrast(xe, sf),
            "xair_mandated_minus_xair_emitted": contrast(xm, xe),
            "xair_mandated_minus_revoke_all": contrast(xm, ra),
            "break_even_mandated": break_even(xm),
        }
    return out


def recheck(split: dict) -> dict:
    """Validation at t_v = t_d only, versus a recheck at t_g = t_v + q, under T ~ Exp(r)."""
    out = {}
    producers = {"qwen7b_blind": test_records("phase_p", {"qwen2.5vl:7b"}, {"blind"}, split),
                 "detector_age1500ms": with_age(test_records("detector_patchcore", None, None, split), 1500.0)}
    for pname, recs in producers.items():
        out[pname] = {}
        for r_rate in (0.2, 1.0):
            for q_ms in (250, 1000, 2000):
                arms: dict[str, list[dict]] = {"validate_only": [], "recheck": []}
                for seed in SEEDS:
                    for rec in recs:
                        rng = random.Random(f"{seed}:{rec.frame_id}")
                        t_drift = rng.expovariate(r_rate) * 1000.0
                        t_pub = rec.latency_ms + q_ms
                        valid_pub = t_drift > t_pub
                        for arm, t_check in (("validate_only", rec.latency_ms), ("recheck", t_pub)):
                            hold = rec.preconds_hold_drifted if t_drift <= t_check else rec.preconds_hold_nominal
                            published, _ = gate_publishes("xair", elapsed_ms=t_check, freshness_ms=W_MS,
                                                          preconds_hold=hold, schema_valid=rec.gate_schema_valid)
                            s = classify(gt_action=rec.gt_action, model_action=rec.model_action, published=published,
                                         measurement=Measurement(valid_pub, valid_pub, 1, 1))
                            s.update(frame_id=rec.frame_id, seed=seed)
                            s["reward"] = trial_reward(s)
                            arms[arm].append(s)
                drifted_between = sum(1 for t in arms["recheck"] if not t["context_valid_at_eval"])
                out[pname][f"r{r_rate:g}_q{q_ms}"] = {
                    "validate_only": rates(arms["validate_only"]), "recheck": rates(arms["recheck"]),
                    "recheck_minus_validate_only": contrast(arms["recheck"], arms["validate_only"]),
                    "invalid_at_release_share": drifted_between / len(arms["recheck"]),
                }
    return out


def machine_timelines() -> list[tuple[list[float], list[str]]]:
    """Per machine, change instants and the state entered at each (reconstructed as in drift timing)."""
    import csv
    import datetime as dt
    import io
    import re
    import zipfile

    def t(s: str) -> float:
        return dt.datetime.strptime(s, "%Y/%m/%d %H:%M:%S.%f").replace(tzinfo=dt.timezone.utc).timestamp()

    with zipfile.ZipFile(production_log()) as z, z.open("Production_Data.csv") as f:
        rows = list(csv.DictReader(io.TextIOWrapper(f, encoding="latin-1")))
    steps = [(re.sub(r"\W+", "_", r["Resource"].strip().lower()), t(r["Start Timestamp"]),
              t(r["Complete Timestamp"]), r["Report Type"]) for r in rows]
    t0 = min(a for _, a, _, _ in steps)
    points: dict[str, list] = {}
    for m, a, b, kind in steps:
        points.setdefault(m, []).extend([(a - t0, 1, int(kind == "S")), (b - t0, -1, -int(kind == "S"))])
    out = []
    for pts in points.values():
        active = setup = 0
        last, times, states = "IDLE", [], []
        for time, d_a, d_s in sorted(pts):
            active += d_a
            setup += d_s
            s = "IDLE" if active <= 0 else "SETUP" if setup > 0 else "BUSY"
            if s != last:
                times.append(time)
                states.append(s)
                last = s
        out.append((times, states))
    return out


def operating_samplers():
    """Decision instants drawn only while the machine (or, superposed, any machine) operates."""
    tl = [x for x in machine_timelines() if len(x[0]) > 1]
    per_machine = []
    for times, states in tl:
        iv = [(times[i], times[i + 1]) for i in range(len(times) - 1) if states[i] != "IDLE"]
        if iv:
            cum = np.cumsum([b - a for a, b in iv])
            per_machine.append((times, iv, cum))

    def one(rng: random.Random) -> float:
        times, iv, cum = per_machine[rng.randrange(len(per_machine))]
        u = rng.uniform(0, cum[-1])
        k = int(np.searchsorted(cum, u))
        a, b = iv[k]
        now = a + (u - (cum[k - 1] if k else 0.0))
        return times[bisect.bisect_right(times, now)] - now

    allc = sorted(t for times, _ in tl for t in times)
    lo, hi = allc[0], allc[-1]

    def active_any(now: float) -> bool:
        for times, states in tl:
            i = bisect.bisect_right(times, now) - 1
            if 0 <= i < len(states) - 1 and states[i] != "IDLE":
                return True
        return False

    def anym(rng: random.Random) -> float:
        while True:
            now = rng.uniform(lo, hi)
            if active_any(now):
                return allc[bisect.bisect_right(allc, now)] - now

    return one, anym


def drift_curve(split: dict) -> dict:
    from experiments.run_b2_validity_frontier import replay_one

    producers = {"qwen7b_blind": test_records("phase_p", {"qwen2.5vl:7b"}, {"blind"}, split),
                 "detector_age1500ms": with_age(test_records("detector_patchcore", None, None, split), 1500.0)}
    one, anym = operating_samplers()
    probe = random.Random(0)
    q1 = sorted(one(probe) for _ in range(20000))
    q2 = sorted(anym(probe) for _ in range(20000))
    samplers = {f"poisson_{r:g}": (lambda rate: (lambda rng: rng.expovariate(rate)))(r) for r in RATES_CURVE}
    samplers["job_shop_operating"] = one
    samplers["job_shop_all_operating"] = anym
    out = {"operating_residual_s_quantiles": {
        "one_machine": {q: q1[int(q * (len(q1) - 1))] for q in (0.01, 0.05, 0.5)},
        "all_machines": {q: q2[int(q * (len(q2) - 1))] for q in (0.01, 0.05, 0.5)}}, "cells": {}}
    for pname, recs in producers.items():
        out["cells"][pname] = {}
        for sname, draw in samplers.items():
            ts = {}
            for g in ("freshness_only", "schema_freshness", "xair"):
                rows = []
                for seed in SEEDS:
                    for rec in recs:
                        rng = random.Random(f"{seed}:{rec.frame_id}")
                        tau = draw(rng) * 1000.0
                        s = replay_one(rec, gate=g, anchor="capture", freshness_ms=W_MS,
                                       invalid_at_submit=tau <= rec.latency_ms)
                        s["seed"] = seed
                        s["reward"] = trial_reward(s)
                        rows.append(s)
                ts[g] = rows
            ra = revoke_all(ts["xair"])
            out["cells"][pname][sname] = {
                "invalid_share": sum(1 for t in ts["xair"] if t["context_valid_at_eval"] is False) / len(ts["xair"]),
                "xair": rates(ts["xair"]),
                "xair_minus_freshness_only": contrast(ts["xair"], ts["freshness_only"]),
                "xair_minus_schema_freshness": contrast(ts["xair"], ts["schema_freshness"]),
                "xair_minus_revoke_all": contrast(ts["xair"], ra),
                "break_even": break_even(ts["xair"]),
            }
            print(f"[review] drift curve {pname} {sname}", flush=True)
    return out


def break_even_reference(split: dict) -> dict:
    out = {}
    for m in MODELS:
        recs = test_records("phase_p", {m}, {"blind"}, split)
        out[m] = {"xair": break_even(ref_trials(recs, "xair")),
                  "freshness_only": break_even(ref_trials(recs, "freshness_only"))}
    det = test_records("detector_patchcore", None, None, split)
    out["detector_own_latency"] = {"xair": break_even(ref_trials(det, "xair"))}
    det15 = ref_trials(with_age(det, 1500.0), "xair")
    out["detector_age1500ms"] = {"xair": break_even(det15)}
    # Residual hazard of the detector under XAIR when drift precedes submission.
    haz = [t for t in det15 if t["hazardous_publish"]]
    out["detector_age1500ms_residual_hazard"] = {
        "hazard": len(haz) / len(det15),
        "unsafe_misgrounded": sum(bool(t["unsafe_publish"]) for t in haz) / len(det15),
        "stale": sum(bool(t["stale_publish"]) for t in haz) / len(det15),
        "stale_uc5_defective": sum(1 for t in haz if t["stale_publish"] and t["use_case"] == "uc5_safety"
                                   and t["defect_present"]) / len(det15),
    }
    return out


def grounding_extra(episodes: dict) -> dict:
    eps = list(episodes.values())
    n = len(eps)
    # Perfect defect presence, no severity information: nominal frames and UC2/UC4 defects are
    # decidable; UC1/UC3/UC5 defects need the (image-invisible) severity cut.
    decidable = sum(1 for e in eps if not e["defect_present"] or e["use_case"] in ("uc2_restart", "uc4_conflict"))
    need_sev = [e for e in eps if e["defect_present"] and e["use_case"] not in ("uc2_restart", "uc4_conflict")]
    best_const = max(sum(1 for e in need_sev if e["severity"] == s) for s in ("minor", "major"))
    out = {"ceiling_perfect_defect_random_severity": (decidable + 0.5 * len(need_sev)) / n,
           "ceiling_perfect_defect_best_constant_severity": (decidable + best_const) / n,
           "ceiling_perfect_defect_presence_only_share_decidable": decidable / n}

    def mapped(rec: dict) -> str:
        nom, minor, major = UC_TABLE[rec["use_case"]]
        if not rec.get("defect_judgement"):
            return nom
        return major if (rec.get("vlm_severity") or "").lower() == "major" else minor

    raw = load_raw("phase_p", set(MODELS), {"blind"})
    table = {}
    for m in MODELS:
        rs = [r for r in raw if r["model"] == m]
        k = sum(1 for r in rs if mapped(r) == r["gt_action"])
        table[m] = {"n": len(rs), "table_mapped_accuracy": k / len(rs),
                    "emitted_accuracy": sum(1 for r in rs if r["action"] == r["gt_action"]) / len(rs)}
    split = load_split()
    det = load_raw("detector_patchcore", None, None)
    dt_ = [r for r in det if split.get(r["frame_id"]) == "test"]
    table["detector_test"] = {"n": len(dt_), "emitted_accuracy": sum(r["action"] == r["gt_action"] for r in dt_) / len(dt_)}
    rs7 = [r for r in raw if r["model"] == "qwen2.5vl:7b"]
    table["qwen2.5vl:7b_test"] = {"table_mapped_accuracy": sum(mapped(r) == r["gt_action"] for r in rs7 if split.get(r["frame_id"]) == "test")
                                  / sum(1 for r in rs7 if split.get(r["frame_id"]) == "test")}
    out["table_mapped"] = table

    # "Always nominal action" baseline (one of the tie-breaks; the 7B model's own default).
    def nominal_correct(fid: str) -> bool:
        return not episodes[fid]["defect_present"]

    for tag, var in (("phase_p", "blind"), ("phase_p_v2", "blind_v2")):
        rs = load_raw(tag, {"qwen2.5vl:7b"}, {var})
        b = sum(1 for r in rs if r["action"] == r["gt_action"] and not nominal_correct(r["frame_id"]))
        c = sum(1 for r in rs if r["action"] != r["gt_action"] and nominal_correct(r["frame_id"]))
        out[f"always_nominal_vs_7b_{var}"] = {"n": len(rs), "model_only": b, "baseline_only": c,
                                             "mcnemar_exact_p": mcnemar_exact_p(b, c)}

    abl = json.loads((RESULTS_DIR / "b1_downscale_ablation.json").read_text())
    fids = {r["frame_id"] for r in abl["rows"]}
    main = {r["frame_id"]: r for r in rs7}
    out["downscale_subset"] = {"n": len(fids), "main_campaign_accuracy_on_subset":
                               sum(main[f]["action"] == main[f]["gt_action"] for f in fids if f in main) / len(fids),
                               "defective_share": sum(episodes[f]["defect_present"] for f in fids) / len(fids)}
    return out


def main() -> int:
    split = load_split()
    episodes = {e["frame_id"]: e for e in load_manifest()}
    out = {"window_ms": W_MS, "lambda": LAM, "mu": MU}
    for name, fn in (("schema_baseline", lambda: schema_baseline(split)),
                     ("defect_signal", lambda: defect_signal(split, episodes)),
                     ("hidden_predicate", lambda: hidden_predicate(split, episodes)),
                     ("break_even_reference", lambda: break_even_reference(split)),
                     ("grounding_extra", lambda: grounding_extra(episodes)),
                     ("recheck", lambda: recheck(split)),
                     ("drift_curve", lambda: drift_curve(split))):
        out[name] = fn()
        print(f"[review] {name} done", flush=True)
        (RESULTS_DIR / "paper2_review_analyses.json").write_text(json.dumps(out, indent=2, default=float) + "\n")
    print(f"[review] wrote {RESULTS_DIR / 'paper2_review_analyses.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
