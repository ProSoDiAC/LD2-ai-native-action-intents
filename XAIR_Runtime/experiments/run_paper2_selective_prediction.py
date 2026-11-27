#!/usr/bin/env python3
"""Selective prediction: can an emission-time signal tell which VLM actions are correct?

Output: results/paper2_selective_prediction.json

Signals available to a gate at emission, without ground truth:
  confidence   the model's self-reported confidence
  speed        negative latency (faster = more confident?)
  agreement    number of the other four models that emit the same action on the frame
  vote_share   share of the five models that emit the plurality action (ties broken by model order)
For each signal: AUROC for predicting correctness (frame bootstrap, all frames with a
decision from the model), and the accuracy a producer would need for publication to pay,
a > lambda / (lambda + 1 + mu) = 5/7 (Proposition 1b), against what the best
operating point reaches. Thresholds are chosen on training frames; the test frames
(612) are only evaluated.
"""

from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.paper2_common import RESULTS_DIR  # noqa: E402
from experiments.paper2_splits import load_split  # noqa: E402
from experiments.run_paper2_review_analyses import MODELS, load_raw  # noqa: E402

LAMBDA, MU = 5.0, 1.0
A_STAR = LAMBDA / (LAMBDA + 1 + MU)
RNG = np.random.default_rng(0)


def auroc(score: np.ndarray, y: np.ndarray) -> float:
    pos, neg = score[y == 1], score[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    allv = np.concatenate([pos, neg])
    ranks = allv.argsort().argsort().astype(float)
    # average ranks over ties
    _, inv, cnt = np.unique(allv, return_inverse=True, return_counts=True)
    sums = np.bincount(inv, weights=ranks)
    ranks = (sums / cnt)[inv] + 1.0
    return float((ranks[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def boot_auc(score, y, n=2000):
    idx = RNG.integers(0, len(y), size=(n, len(y)))
    vals = np.array([auroc(score[i], y[i]) for i in idx])
    vals = vals[~np.isnan(vals)]
    return [float(np.quantile(vals, 0.025)), float(np.quantile(vals, 0.975))]


def main() -> None:
    split = load_split()
    raw = load_raw("phase_p", set(MODELS), {"blind"})
    by_frame: dict[str, dict[str, dict]] = defaultdict(dict)
    for r in raw:
        by_frame[r["frame_id"]][r["model"]] = r
    full = {f: d for f, d in by_frame.items() if len(d) == len(MODELS)}

    rows = []  # one per (frame, model)
    for f, d in full.items():
        votes = Counter(d[m]["action"] for m in MODELS)
        top = max(votes.values())
        plural = next(d[m]["action"] for m in MODELS if votes[d[m]["action"]] == top)
        for m in MODELS:
            r = d[m]
            rows.append({
                "frame": f, "model": m, "part": split.get(f), "uc": r["use_case"],
                "correct": int(r["action"] == r["gt_action"]),
                "conf": (float(r["confidence"]) if r.get("confidence") is not None else float("nan")),
                "speed": -float(r["latency_ms"]),
                "agree": sum(d[o]["action"] == r["action"] for o in MODELS if o != m),
                "vote_share": votes[r["action"]] / len(MODELS),
                "plural_correct": int(plural == r["gt_action"]),
                "is_plural": int(r["action"] == plural),
            })

    out = {"n_frames_all_five_models": len(full), "a_star": A_STAR, "per_model": {}, "plurality_vote": {}}
    for m in MODELS:
        rs = [x for x in rows if x["model"] == m]
        y = np.array([x["correct"] for x in rs])
        entry = {"n": len(rs), "accuracy": float(y.mean()), "auroc": {}}
        for sig in ("conf", "speed", "agree"):
            s = np.array([x[sig] for x in rs], dtype=float)
            ok = ~np.isnan(s)
            entry["auroc"][sig] = {"auroc": auroc(s[ok], y[ok]), "ci95": boot_auc(s[ok], y[ok]), "n_used": int(ok.sum())}
        # selective accuracy by agreement level on the test frames
        te = [x for x in rs if x["part"] == "test"]
        lv = {}
        for k in range(5):
            sub = [x for x in te if x["agree"] >= k]
            lv[str(k)] = {"coverage": len(sub) / len(te), "accuracy": float(np.mean([x["correct"] for x in sub])) if sub else None}
        entry["test_accuracy_by_min_agreement"] = lv
        # best operating point chosen on train: smallest coverage-weighted accuracy above A_STAR
        tr = [x for x in rs if x["part"] == "train"]
        best = None
        for k in range(5):
            sub = [x for x in tr if x["agree"] >= k]
            if sub and np.mean([x["correct"] for x in sub]) > A_STAR and len(sub) / len(tr) >= 0.05:
                best = k
                break
        entry["train_selected_min_agreement"] = best
        if best is not None:
            sub = [x for x in te if x["agree"] >= best]
            entry["test_at_selected"] = {"coverage": len(sub) / len(te),
                                         "accuracy": float(np.mean([x["correct"] for x in sub])) if sub else None}
        # threshold search over every signal and a train-only logistic combination
        def feats(xs):
            return np.array([[0.0 if np.isnan(x["conf"]) else x["conf"], x["speed"], x["agree"]] for x in xs], dtype=float)
        ytr = np.array([x["correct"] for x in tr]); yte = np.array([x["correct"] for x in te])
        scores = {}
        for j, sig in enumerate(("conf", "speed", "agree")):
            scores[sig] = (feats(tr)[:, j], feats(te)[:, j])
        Ftr, Fte = feats(tr), feats(te)
        mu, sd = Ftr.mean(0), Ftr.std(0) + 1e-9
        Ztr, Zte = (Ftr - mu) / sd, (Fte - mu) / sd
        w = np.zeros(3); b = 0.0
        for _ in range(500):
            pr = 1 / (1 + np.exp(-(Ztr @ w + b)))
            w -= 0.1 * (Ztr.T @ (pr - ytr) / len(ytr)); b -= 0.1 * float((pr - ytr).mean())
        scores["logistic"] = (Ztr @ w + b, Zte @ w + b)
        sel = {}
        for sig, (str_, ste) in scores.items():
            chosen = None
            for thr in np.unique(str_):
                msk = str_ >= thr
                if msk.mean() >= 0.05 and ytr[msk].mean() > A_STAR:
                    chosen = thr  # lowest threshold (largest coverage) that clears A*
                    break
            if chosen is None:
                sel[sig] = None
            else:
                mte = ste >= chosen
                sel[sig] = {"test_coverage": float(mte.mean()),
                            "test_accuracy": float(yte[mte].mean()) if mte.any() else None,
                            "n_test_selected": int(mte.sum())}
        entry["threshold_search_train_selected"] = sel
        out["per_model"][m] = entry

    # plurality vote of the five models (one decision per frame)
    seen = {}
    for x in rows:
        seen[x["frame"]] = x
    fr = list(seen.values())
    y = np.array([x["plural_correct"] for x in fr])
    out["plurality_vote"]["accuracy_all"] = float(y.mean())
    te = [x for x in fr if x["part"] == "test"]
    out["plurality_vote"]["accuracy_test"] = float(np.mean([x["plural_correct"] for x in te]))
    share = {}
    for x in fr:
        pass
    # accuracy of the plurality action as a function of how many models back it
    votes_by_frame = {}
    for f, d in full.items():
        c = Counter(d[m]["action"] for m in MODELS)
        votes_by_frame[f] = max(c.values())
    for k in (2, 3, 4, 5):
        sub = [x for x in fr if votes_by_frame[x["frame"]] >= k]
        share[str(k)] = {"coverage": len(sub) / len(fr),
                         "accuracy": float(np.mean([x["plural_correct"] for x in sub])) if sub else None}
    out["plurality_vote"]["by_min_backing_models_all_frames"] = share
    # by use case
    uc = {}
    for u in sorted({x["uc"] for x in fr}):
        sub = [x for x in fr if x["uc"] == u]
        uc[u] = float(np.mean([x["plural_correct"] for x in sub]))
    out["plurality_vote"]["accuracy_by_use_case"] = uc

    path = RESULTS_DIR / "paper2_selective_prediction.json"
    path.write_text(json.dumps(out, indent=2))
    print(json.dumps({"plurality": out["plurality_vote"], "per_model": {m: {k: v for k, v in e.items() if k != "test_accuracy_by_min_agreement"} for m, e in out["per_model"].items()}}, indent=1))


if __name__ == "__main__":
    main()
