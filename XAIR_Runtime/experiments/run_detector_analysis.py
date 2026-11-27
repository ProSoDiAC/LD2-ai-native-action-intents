#!/usr/bin/env python3
"""Turn the controlled detector producer's scores into intents and replay them through the B2 gate.

Inputs (from ``detector_producer.py``): per-frame image scores, 56x56 anomaly maps and
measured inference latencies. Per category, two thresholds are calibrated on the
*training split only*: the image-score threshold that maximizes balanced accuracy of
defect detection, and the anomalous-area cut that best separates minor from major
defects. Test frames are then mapped to an action with the cell's use-case table, the
same table that defines ground truth, so the producer is correct whenever its estimate
of the defect state is.

The detector emits no preconditions; as in the harness-supplied baseline of B2, each
intent carries the cell predicate ``line.state == 'RUN'``.

Outputs:
  perception_cache/detector_patchcore.jsonl   decisions in the Phase-P cache format
  detector_analysis.json                      detection quality and the B2 comparison
"""

from __future__ import annotations

import importlib.util
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments import paper2_common  # noqa: E402
from experiments.paper2_common import RESULTS_DIR, load_manifest  # noqa: E402
from experiments.paper2_splits import load_split  # noqa: E402
from experiments.run_b2_validity_frontier import cache_path, load_replay_records  # noqa: E402
from experiments.run_paper2_stats import (  # noqa: E402
    bootstrap_utility_frames, generate_trials, paired_frame_contrast,
)

DET = RESULTS_DIR / "detector"
TAG = "detector_patchcore"
MODEL = "patchcore-wrn50"
PREDICATE = "line.state == 'RUN'"
GATES = ("direct", "freshness_only", "xair")


def gt_actions() -> dict:
    path = ROOT / "experiments" / "datasets" / "manufacturing-a1" / "scripts" / "build_manifest.py"
    spec = importlib.util.spec_from_file_location("build_manifest", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod.GT_ACTIONS


def best_threshold(scores: np.ndarray, labels: np.ndarray) -> float:
    """Threshold maximizing balanced accuracy (ties: the lowest threshold)."""
    best, best_t = -1.0, float(scores.max()) + 1e-6
    for t in np.unique(scores):
        pred = scores >= t
        tpr = (pred & labels).sum() / max(1, labels.sum())
        tnr = (~pred & ~labels).sum() / max(1, (~labels).sum())
        ba = (tpr + tnr) / 2
        if ba > best:
            best, best_t = ba, float(t)
    return best_t


def area(amap: np.ndarray, thr: float) -> float:
    return float((amap.astype(np.float32) >= thr).mean())


def iso(t: datetime) -> str:
    return t.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


def _reweight(trials: list[dict], lam: float) -> list[dict]:
    mu = paper2_common.MU_WRONGFUL_REVOKE
    return [dict(t, reward=float(t["successful_actuation"]) - lam * float(t["hazardous_publish"])
                 - mu * float(t["wrongful_revoke"])) for t in trials]


def regime_grid(test, vlm_ages: dict, episodes: dict) -> dict:
    """XAIR against revoke-all and freshness-only over evidence age, drift offset, window, lambda.

    Evidence ages: the detector's own, a uniform 1.5 s (slow but inside every window, and
    independent of source), and the 7B model's observed ages (source-confounded).
    """
    from experiments.run_paper2_latency_factorial import with_age

    ages = {"detector": None, "uniform_1500ms": lambda r: 1500.0, "vlm_7b": lambda r: vlm_ages[r.frame_id]}
    out: dict = {}
    for age_label, fn in ages.items():
        recs = test
        if fn is not None:
            recs = with_age(test, 0.0)
            for r in recs:
                r.latency_ms = fn(r)
        for tau in (0, 250, 2000):
            for w in (2000, 4000, 8000):
                base = {g: generate_trials(recs, gate=g, freshness_ms=w, drift_offset_ms=tau)
                        for g in ("freshness_only", "xair")}
                for lam in (1.0, 2.0, 5.0):
                    fo, xa = _reweight(base["freshness_only"], lam), _reweight(base["xair"], lam)
                    ra = [dict(t, reward=-paper2_common.MU_WRONGFUL_REVOKE *
                               (bool(t["successful_actuation"]) or bool(t["wrongful_revoke"]))) for t in xa]
                    cell = {"U_xair": sum(t["reward"] for t in xa) / len(xa),
                            "U_freshness_only": sum(t["reward"] for t in fo) / len(fo),
                            "U_revoke_all": sum(t["reward"] for t in ra) / len(ra)}
                    for name, other in (("revoke_all", ra), ("freshness_only", fo)):
                        cell[f"xair_minus_{name}"] = paired_frame_contrast(
                            xa, other, fields=("reward",), n_boot=2000)["fields"]["U"]
                    for src in ("visa", "mvtec_ad"):
                        sel = [i for i, t in enumerate(xa) if episodes[t["frame_id"]]["source_dataset"] == src]
                        cell[f"xair_minus_revoke_all_{src}"] = paired_frame_contrast(
                            [xa[i] for i in sel], [ra[i] for i in sel], fields=("reward",), n_boot=2000)["fields"]["U"]
                    out[f"{age_label}_tau{tau}_w{w}_lambda{lam:g}"] = cell
    return out


def main() -> int:
    episodes = {e["frame_id"]: e for e in load_manifest()}
    split = load_split()
    table = gt_actions()
    scores = {json.loads(line)["frame_id"]: json.loads(line) for line in
              (DET / "patchcore_scores.jsonl").read_text().splitlines() if line.strip()}
    npz = np.load(DET / "patchcore_maps.npz")
    maps = dict(zip(npz["frame_id"].tolist(), npz["maps"]))

    cats = sorted({(e["source_dataset"], e["category"]) for e in episodes.values()})
    calib, pred = {}, {}
    for src, cat in cats:
        fids = [f for f, e in episodes.items() if (e["source_dataset"], e["category"]) == (src, cat)]
        train = [f for f in fids if split.get(f) == "train"]
        s = np.array([scores[f]["image_score"] for f in train])
        y = np.array([bool(episodes[f]["defect_present"]) for f in train])
        t_img = best_threshold(s, y)
        defect_train = [f for f in train if episodes[f]["defect_present"]]
        a = np.array([area(maps[f], t_img) for f in defect_train])
        major = np.array([episodes[f]["severity"] == "major" for f in defect_train])
        t_area = best_threshold(a, major) if len(defect_train) and major.any() and (~major).any() else 0.0
        calib[f"{src}/{cat}"] = {"image_threshold": t_img, "area_cut": t_area, "n_train": len(train)}
        for f in fids:
            if scores[f]["image_score"] < t_img:
                state = "none"
            else:
                state = "major" if area(maps[f], t_img) >= t_area else "minor"
            pred[f] = state

    now = datetime(2026, 9, 27, tzinfo=timezone.utc)
    lines = []
    for f, e in sorted(episodes.items()):
        lat = float(scores[f]["latency_ms"])
        action = table[e["use_case"]][pred[f]]
        lines.append(json.dumps({
            "frame_id": f, "use_case": e["use_case"], "model": MODEL, "prompt_variant": "detector",
            "source_dataset": e["source_dataset"], "category": e["category"],
            "capture_ts": iso(now), "emitted_ts": iso(now + timedelta(milliseconds=lat)), "latency_ms": lat,
            "action": action, "gt_action": e["ground_truth_action"], "action_valid": True,
            "parse_ok": True, "schema_valid": True, "schema_valid_repaired": True,
            "precondition_syntax_ok": True, "preconditions": [PREDICATE], "preconditions_repaired": [PREDICATE],
            "defect_judgement": pred[f] != "none", "defect_present": bool(e["defect_present"]),
            "predicted_severity": pred[f], "image_score": scores[f]["image_score"], "error": None,
        }) + "\n")
    cache_path(TAG).write_text("".join(lines))

    recs = load_replay_records(TAG, models=None, variants=None, use_repaired=False)
    test = [r for r in recs if split.get(r.frame_id) == "test"]
    ytest = [(pred[r.frame_id] != "none", bool(episodes[r.frame_id]["defect_present"])) for r in test]
    tp = sum(p and y for p, y in ytest)
    tn = sum(not p and not y for p, y in ytest)
    pos = sum(y for _, y in ytest)
    lat = sorted(float(scores[r.frame_id]["latency_ms"]) for r in test)
    out = {
        "calibration": calib,
        "test": {
            "n": len(test),
            "grounding": sum(r.grounding_correct for r in test) / len(test),
            "defect_balanced_accuracy": (tp / pos + tn / (len(test) - pos)) / 2,
            "severity_exact": sum(pred[r.frame_id] == episodes[r.frame_id]["severity"] for r in test) / len(test),
            "grounding_by_use_case": {
                uc: sum(r.grounding_correct for r in test if r.use_case == uc) / max(1, sum(r.use_case == uc for r in test))
                for uc in sorted({r.use_case for r in test})
            },
            "latency_ms": {"p50": lat[len(lat) // 2], "p95": lat[int(0.95 * (len(lat) - 1))]},
        },
        "b2": {},
    }
    # Reference cell of B2, and the same cell with the observed 7B evidence ages, so that
    # the controlled producer also faces the latency the VLM pays.
    vlm = {r.frame_id: r.latency_ms for r in load_replay_records(
        "phase_p", models={"qwen2.5vl:7b"}, variants={"blind"}, use_repaired=False)}
    for label, ages in (("detector_latency", None), ("vlm_7b_latency", vlm)):
        rs = test
        if ages is not None:
            from experiments.run_paper2_latency_factorial import with_age
            rs = with_age(test, 0.0)
            for r in rs:
                r.latency_ms = ages[r.frame_id]
        cell = {}
        trials = {g: generate_trials(rs, gate=g, freshness_ms=4000, drift_offset_ms=250) for g in GATES}
        for g, ts in trials.items():
            opp = sum(bool(t["successful_actuation"]) or bool(t["wrongful_revoke"]) for t in ts)
            cell[g] = {**bootstrap_utility_frames(ts, n_boot=1000, seed=0),
                       "SAR": sum(bool(t["successful_actuation"]) for t in ts) / len(ts),
                       "hazard": sum(bool(t["hazardous_publish"]) for t in ts) / len(ts),
                       "WRR": sum(bool(t["wrongful_revoke"]) for t in ts) / opp if opp else 0.0}
        xa = trials["xair"]
        revoke = [dict(t, successful_actuation=0, hazardous_publish=0,
                       wrongful_revoke=int(bool(t["successful_actuation"]) or bool(t["wrongful_revoke"])),
                       reward=-paper2_common.MU_WRONGFUL_REVOKE * (bool(t["successful_actuation"]) or bool(t["wrongful_revoke"])))
                  for t in xa]
        cell["revoke_all"] = bootstrap_utility_frames(revoke, n_boot=1000, seed=0)
        cell["xair_minus_freshness_only"] = paired_frame_contrast(xa, trials["freshness_only"], n_boot=10000)
        cell["xair_minus_revoke_all"] = paired_frame_contrast(xa, revoke, fields=("reward",), n_boot=10000)
        cell["xair_minus_direct"] = paired_frame_contrast(xa, trials["direct"], n_boot=10000)
        out["b2"][label] = cell
    out["grid"] = regime_grid(test, vlm, episodes)
    (RESULTS_DIR / "detector_analysis.json").write_text(json.dumps(out, indent=2) + "\n")
    t = out["test"]
    print(f"[detector] test grounding {t['grounding']:.3f}, defect BA {t['defect_balanced_accuracy']:.3f}, "
          f"severity {t['severity_exact']:.3f}, latency p50 {t['latency_ms']['p50']:.0f} ms")
    for label, cell in out["b2"].items():
        print(label, {g: round(cell[g]["U"], 3) for g in (*GATES, "revoke_all")})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
