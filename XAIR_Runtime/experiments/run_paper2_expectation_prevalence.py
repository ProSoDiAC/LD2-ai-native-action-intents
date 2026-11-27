#!/usr/bin/env python3
"""Reference-cell analyses without drift Monte Carlo, and under other defect prevalences.
Output: paper2_expectation_prevalence.json.

At the reference cell (w = 4 s, p_drift = 0.5, tau = 250 ms, capture anchoring) a drawn drift
always precedes submission, because tau is below every evidence age. The expected outcome of a
frame is therefore p * outcome(drift) + (1 - p) * outcome(no drift), and no drift seeds are needed.

  expectation   exact expected U, rates, contrasts, and lambda* per producer, with frame-cluster
                and category-cluster bootstrap intervals; Monte Carlo error of the five-seed
                estimates used in the main replay
  composition   adverse publications split into stale and misgrounded, safety-relevant
                (permissive) and availability-only (over-conservative), and malformed
  (The detector is included only at a 1.5 s evidence age: at its own 65 ms latency a 250 ms
  drift offset follows submission, which the exact two-case expectation does not cover.)
  prevalence    the same quantities with frames reweighted to a defect prevalence of
                1, 5, 10, 25, or 50 percent, and for each use case alone
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.paper2_common import LAMBDA_HAZARD, MU_WRONGFUL_REVOKE, RESULTS_DIR, load_manifest, trial_reward  # noqa: E402
from experiments.paper2_splits import load_split  # noqa: E402
from experiments.run_b2_validity_frontier import load_replay_records, replay_one  # noqa: E402
from experiments.run_paper2_latency_factorial import with_age  # noqa: E402
from experiments.run_paper2_revision_analyses import RANK  # noqa: E402

P_DRIFT = 0.5
W_MS = 4000
LAM, MU = LAMBDA_HAZARD, MU_WRONGFUL_REVOKE
GATES = ("direct", "freshness_only", "schema_freshness", "xair")
PREVALENCES = (0.01, 0.05, 0.10, 0.25, 0.50)
N_BOOT = 10000
FIELDS = ("success", "adverse", "stale", "misgrounded", "safety", "conservative", "malformed",
          "wrongful", "valid_correct")


def outcome(rec, gate: str, drift: bool) -> dict:
    s = replay_one(rec, gate=gate, anchor="capture", freshness_ms=W_MS, invalid_at_submit=drift)
    adverse = bool(s["hazardous_publish"])
    conservative = adverse and (
        (s["stale_publish"] and RANK.get(s["model_action"], 0) > 1)
        or (s["unsafe_publish"] and RANK.get(s["model_action"], -1) >= RANK.get(s["gt_action"], 0)))
    return {
        "success": float(s["successful_actuation"]),
        "adverse": float(adverse),
        "stale": float(s["stale_publish"]),
        "misgrounded": float(s["unsafe_publish"]),
        "safety": float(adverse and not conservative),
        "conservative": float(conservative),
        "malformed": float(bool(s["published"]) and not rec.gate_schema_valid),
        "wrongful": float(s["wrongful_revoke"]),
        "valid_correct": float(bool(s["successful_actuation"]) or bool(s["wrongful_revoke"])),
        # released on valid context, and correct among those (accuracy behind Proposition 1(b))
        "valid_ctx": float(s["context_valid_at_eval"] is True),
        "released_valid": float(bool(s["published"]) and s["context_valid_at_eval"] is True),
        "released_valid_correct": float(bool(s["published"]) and s["context_valid_at_eval"] is True
                                        and bool(s["grounding_correct"])),
        "reward": trial_reward(s),
    }


def expected_frames(recs) -> dict[str, dict[str, np.ndarray]]:
    """Per path, arrays over frames of expected indicators and reward."""
    out = {}
    for g in GATES:
        rows = []
        for r in recs:
            d, n = outcome(r, g, True), outcome(r, g, False)
            rows.append({k: P_DRIFT * d[k] + (1 - P_DRIFT) * n[k] for k in d})
        out[g] = {k: np.array([row[k] for row in rows]) for k in rows[0]}
    vc = out["xair"]["valid_correct"]
    out["revoke_all"] = {k: np.zeros(len(recs)) for k in out["xair"]}
    out["revoke_all"]["wrongful"] = vc.copy()
    out["revoke_all"]["valid_correct"] = vc.copy()
    out["revoke_all"]["reward"] = -MU * vc
    return out


def summarize(fr, w: np.ndarray, idx: np.ndarray) -> dict:
    """Weighted means over frames; idx = bootstrap index matrix (rows = replicates)."""
    def wmean(a, ii=None):
        if ii is None:
            return float((a * w).sum() / w.sum())
        return (a[ii] * w[ii]).sum(axis=1) / w[ii].sum(axis=1)

    res = {}
    for g, a in fr.items():
        point = {k: wmean(a[k]) for k in (*FIELDS, "reward")}
        point["U"] = point.pop("reward")
        point["WRR"] = point["wrongful"] / point["valid_correct"] if point["valid_correct"] else 0.0
        ci = {}
        for k, key in (("U", "reward"), ("SAR", "success"), ("Adverse", "adverse")):
            b = wmean(a[key], idx)
            ci[k] = [float(x) for x in np.quantile(b, [0.025, 0.975])]
        res[g] = {**point, "ci95": ci}
    for a_, b_ in (("xair", "freshness_only"), ("xair", "schema_freshness"), ("xair", "revoke_all")):
        d = fr[a_]["reward"] - fr[b_]["reward"]
        bd = wmean(d, idx)
        res[f"{a_}_minus_{b_}"] = {"delta": wmean(d), "ci95": [float(x) for x in np.quantile(bd, [0.025, 0.975])]}
    s, h = fr["xair"]["success"], fr["xair"]["adverse"]
    sb, hb = wmean(s, idx), wmean(h, idx)
    with np.errstate(divide="ignore", invalid="ignore"):
        lam_b = np.where(hb > 0, (1 + MU) * sb / hb, np.inf)
    lam = (1 + MU) * wmean(s) / wmean(h) if wmean(h) > 0 else float("inf")
    res["lambda_star_xair"] = {"value": lam, "ci95": [float(x) for x in np.quantile(lam_b, [0.025, 0.975])]}
    return res


def boot_index(n: int, seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, n, size=(N_BOOT, n))


def category_index(cats: list[str], seed: int = 0) -> np.ndarray:
    """Resample categories with replacement; a replicate keeps all frames of each drawn category."""
    uniq = sorted(set(cats))
    members = {c: np.array([i for i, x in enumerate(cats) if x == c]) for c in uniq}
    rng = np.random.default_rng(seed)
    reps = []
    for _ in range(2000):
        draw = rng.integers(0, len(uniq), size=len(uniq))
        reps.append(np.concatenate([members[uniq[j]] for j in draw]))
    m = max(len(r) for r in reps)
    # pad with a repeated index of zero weight: use -1 markers replaced by the first index and masked
    return reps, m


def category_ci(fr, cats) -> dict:
    reps, _ = category_index(cats)
    out = {}
    for a_, b_ in (("xair", "freshness_only"), ("xair", "revoke_all")):
        d = fr[a_]["reward"] - fr[b_]["reward"]
        vals = np.array([d[r].mean() for r in reps])
        out[f"{a_}_minus_{b_}"] = [float(x) for x in np.quantile(vals, [0.025, 0.975])]
    return out


def mc_error(recs, fr) -> dict:
    """Standard error of the five-seed Monte Carlo estimate of U and of the XAIR contrasts."""
    out = {}
    for g in ("freshness_only", "xair"):
        dvals = np.array([outcome(r, g, True)["reward"] - outcome(r, g, False)["reward"] for r in recs])
        var = (P_DRIFT * (1 - P_DRIFT) * dvals ** 2).mean()
        out[g] = float(np.sqrt(var / (len(recs) * 5)))
    return out


def weights(eps: list[dict], prevalence: float | None, use_case: str | None = None) -> np.ndarray:
    w = np.ones(len(eps))
    if use_case is not None:
        w = np.array([1.0 if e["use_case"] == use_case else 0.0 for e in eps])
    if prevalence is not None:
        d = np.array([1.0 if e["defect_present"] else 0.0 for e in eps])
        nd, nn = (w * d).sum(), (w * (1 - d)).sum()
        if nd == 0 or nn == 0:
            raise ValueError(
                "cannot reweight to a defect prevalence when one class has zero selected "
                f"frames (defective={nd}, nominal={nn})"
            )
        w = w * np.where(d == 1, prevalence / nd, (1 - prevalence) / nn)
    return w


NOMINAL = {"uc1_triage": "ACCEPT", "uc2_restart": "RESUME", "uc3_speed": "ACCEPT",
           "uc4_conflict": "RESUME", "uc5_safety": "ACCEPT"}


def always_nominal(recs, episodes):
    """A perception-free producer: always the nominal action of the use case, the integrator-declared
    predicate line.state == 'RUN', well formed, at the primary model's evidence ages."""
    from experiments.run_b2_validity_frontier import ReplayRecord
    out = []
    for r in recs:
        c = object.__new__(ReplayRecord)
        for k in ReplayRecord.__slots__:
            setattr(c, k, getattr(r, k))
        ep = episodes[r.frame_id]
        c.model_action = NOMINAL[ep["use_case"]]
        c.grounding_correct = c.model_action == c.gt_action
        c.gate_schema_valid = c.schema_valid = True
        c.n_preconditions = 1
        c.preconds_hold_nominal = True
        # the drift patch falsifies line.state except for the UC5 defect patch
        c.preconds_hold_drifted = "line" not in (ep.get("drift_patch") or {})
        out.append(c)
    return out


def five_seed_split(recs) -> dict:
    """Adverse publications of the main (five-seed) replay split into safety-relevant and conservative."""
    from experiments.run_paper2_stats import generate_trials

    out = {}
    for g in GATES:
        ts = generate_trials(recs, gate=g, freshness_ms=W_MS)
        cons = sum(1 for t in ts if t["hazardous_publish"] and (
            (t["stale_publish"] and RANK.get(t["model_action"], 0) > 1)
            or (t["unsafe_publish"] and RANK.get(t["model_action"], -1) >= RANK.get(t["gt_action"], 0))))
        adv = sum(1 for t in ts if t["hazardous_publish"])
        out[g] = {"adverse": adv / len(ts), "safety": (adv - cons) / len(ts), "conservative": cons / len(ts)}
    return out


def nominal_minus_7b(frames_by: dict) -> dict:
    """Paired frame-cluster contrast of full-XAIR utility: perception-free nominal producer minus 7B."""
    ids_n, eps, fr_n = frames_by["always_nominal_7b_ages"]
    ids_q, _, fr_q = frames_by["qwen7b_blind"]
    assert ids_n == ids_q, "the nominal producer must share the 7B frames"
    d = fr_n["xair"]["reward"] - fr_q["xair"]["reward"]
    idx = boot_index(len(d))
    out = {}
    for pv in (*PREVALENCES,):
        w = weights(eps, pv)
        boots = (d[idx] * w[idx]).sum(axis=1) / w[idx].sum(axis=1)
        out[f"{pv:g}"] = {"delta": float((d * w).sum() / w.sum()),
                          "ci95": [float(x) for x in np.quantile(boots, [0.025, 0.975])]}
    return out


def released_valid_defect_check(episodes: dict, split: dict) -> dict:
    """Accuracy and coverage of full-XAIR releases on valid context, with the emitted preconditions on
    defect.* kept (main replay) and dropped (they are checked against the ground-truth flag)."""
    from experiments.run_paper2_review_analyses import load_raw, records_from

    def drop(ep, rec):
        rec["preconditions"] = [p for p in rec.get("preconditions") or [] if not p.strip().startswith("defect.")]

    out = {}
    for m in ("qwen2.5vl:7b", "llama3.2-vision:11b"):
        raw = load_raw("phase_p", {m}, {"blind"})
        out[m] = {}
        for name, kw in (("kept", {}), ("dropped", {"rec_fn": drop})):
            recs = sorted(records_from(raw, episodes, split, **kw), key=lambda r: r.frame_id)
            x, eps = expected_frames(recs)["xair"], [episodes[r.frame_id] for r in recs]
            res = {}
            for pv in PREVALENCES:
                w = weights(eps, pv)
                rv, rvc, vc_ = ((x[k] * w).sum() for k in ("released_valid", "released_valid_correct", "valid_ctx"))
                res[f"{pv:g}"] = {"accuracy": float(rvc / rv) if rv > 0 else None,
                                  "coverage": float(rv / vc_) if vc_ > 0 else None}
            out[m][name] = res
    return out


def main() -> int:
    split = load_split()
    episodes = {e["frame_id"]: e for e in load_manifest()}
    producers = {
        "qwen7b_blind": [r for r in load_replay_records("phase_p", models={"qwen2.5vl:7b"}, variants={"blind"},
                                                        use_repaired=False) if split.get(r.frame_id) == "test"],
        "qwen3b_blind": [r for r in load_replay_records("phase_p", models={"qwen2.5vl:3b"}, variants={"blind"},
                                                        use_repaired=False) if split.get(r.frame_id) == "test"],
        "gemma3_blind": [r for r in load_replay_records("phase_p", models={"gemma3:12b"}, variants={"blind"},
                                                        use_repaired=False) if split.get(r.frame_id) == "test"],
        "llama_blind": [r for r in load_replay_records("phase_p", models={"llama3.2-vision:11b"}, variants={"blind"},
                                                       use_repaired=False) if split.get(r.frame_id) == "test"],
    }
    det = [r for r in load_replay_records("detector_patchcore", models=None, variants=None, use_repaired=False)
           if split.get(r.frame_id) == "test"]
    producers["always_nominal_7b_ages"] = always_nominal(producers["qwen7b_blind"], episodes)
    producers["detector_age1500ms"] = with_age(det, 1500.0)
    out = {"p_drift": P_DRIFT, "window_ms": W_MS, "lambda": LAM, "mu": MU, "producers": {}}
    frames_by: dict = {}
    for name, recs in producers.items():
        recs = sorted(recs, key=lambda r: r.frame_id)
        eps = [episodes[r.frame_id] for r in recs]
        fr = expected_frames(recs)
        frames_by[name] = ([r.frame_id for r in recs], eps, fr)
        idx = boot_index(len(recs))
        acc = np.array([1.0 if r.grounding_correct else 0.0 for r in recs])
        nominal = np.array([0.0 if e["defect_present"] else 1.0 for e in eps])
        res = {"n_frames": len(recs), "balanced": summarize(fr, np.ones(len(recs)), idx),
               "category_cluster_ci": category_ci(fr, [e["category"] for e in eps]),
               "mc_se_five_seeds": mc_error(recs, fr), "prevalence": {}, "use_case": {}}
        for pv in PREVALENCES:
            w = weights(eps, pv)
            r_ = summarize(fr, w, idx)
            r_["accuracy"] = float((acc * w).sum() / w.sum())
            r_["always_nominal_accuracy"] = float((nominal * w).sum() / w.sum())
            x = fr["xair"]
            rv, rvc, vc_ = ((x[k] * w).sum() for k in ("released_valid", "released_valid_correct", "valid_ctx"))
            r_["xair_released_valid"] = {"accuracy": float(rvc / rv) if rv > 0 else None,
                                         "coverage": float(rv / vc_) if vc_ > 0 else None,
                                         "exceeds_5_7": bool(rv > 0 and rvc / rv > 5 / 7)}
            res["prevalence"][f"{pv:g}"] = r_
        for uc in sorted({e["use_case"] for e in eps}):
            w = weights(eps, None, uc)
            keep = w > 0
            sub_idx = boot_index(int(keep.sum()), seed=1)
            sub = {g: {k: v[keep] for k, v in a.items()} for g, a in fr.items()}
            r_ = summarize(sub, np.ones(int(keep.sum())), sub_idx)
            r_["accuracy"] = float(acc[keep].mean())
            res["use_case"][uc] = r_
        out["producers"][name] = res
        print(f"[expectation] {name} done", flush=True)
    out["nominal_minus_7b_xair"] = nominal_minus_7b(frames_by)
    out["five_seed_adverse_split_7b"] = five_seed_split(producers["qwen7b_blind"])
    out["xair_released_valid_defect_preconditions"] = released_valid_defect_check(episodes, split)
    path = RESULTS_DIR / "paper2_expectation_prevalence.json"
    path.write_text(json.dumps(out, indent=2, default=float) + "\n")
    print(f"[expectation] wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
