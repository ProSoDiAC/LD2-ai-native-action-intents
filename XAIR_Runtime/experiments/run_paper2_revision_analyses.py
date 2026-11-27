#!/usr/bin/env python3
"""
Supplementary analyses added in the 2026-09 academic review (all offline, CPU).

Each block answers one reviewer concern and is written to
``results/paper2_revision_analyses.json``:

  b1_diagnostics        blind grounding vs chance and vs the task-aware majority, per-UC
                        action diversity and Cohen's kappa, defect-judgement TPR/FPR,
                        self-catch / false-block on syntactically valid emissions only
  latency_by_source     inference latency by image source (VisA vs MVTec AD)
  b2_by_source          headline-cell utility per gate, stratified by image source
  b2_other_cells        w in {2000, 8000} and tau = 4000 ms (drift after most inferences)
  b2_harness_echo       XAIR with the single harness precondition line.state == 'RUN'
  b2_xair_gain_sources  hazards avoided by XAIR vs freshness-only: schema vs precondition
  b2_revoke_all_reference  utility of never publishing, and XAIR minus it, per B2 cell
  b2_hazard_construct   utility when over-conservative publications cost mu, not lambda
  b3_reselection        best-fixed / oracle / learners re-selected and re-trained under
                        other cost weights and drift probabilities
  category_holdout      held-out-category grounding vs the disjoint training categories
  b5_structure          retry-stale vs single-shot identity check

Usage:
    python3 experiments/run_paper2_revision_analyses.py --tag phase_p
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments import paper2_common  # noqa: E402
from experiments.paper2_common import (  # noqa: E402
    RESULTS_DIR,
    drift_patch,
    load_manifest,
    nominal_context,
)
from experiments.paper2_splits import filter_by_split, load_split  # noqa: E402
from experiments.rl.budget_env import ACTIONS, feature_vector  # noqa: E402
from experiments.rl.policies import FixedBudget, LinUCB, QLearning  # noqa: E402
from experiments.run_b2_validity_frontier import (  # noqa: E402
    load_replay_records,
    merge_context,
)
from experiments.run_paper2_stats import (  # noqa: E402
    GATES,
    HEADLINE,
    SEEDS,
    bootstrap_utility_frames,
    generate_trials,
    paired_frame_contrast,
)
from xair.core.context_validator import ContextValidator  # noqa: E402

PRIMARY = HEADLINE["model"]
N_PAIRED = 10000  # paired frame-cluster resamples, as in run_paper2_stats.py
# Conservativeness rank: a publication is "permissive" when it lets the line do more
# than ground truth allows; the reverse is an over-conservative (availability) error.
RANK = {
    "ACCEPT": 0, "RESUME": 0, "SLOW_DOWN": 1, "HOLD_FOR_OPERATOR": 2,
    "REJECT_TO_BIN": 2, "STOP": 3, "E_STOP": 4,
}


def _binom_two_sided(k: int, n: int, p: float = 0.5) -> float:
    try:
        from scipy.stats import binomtest

        return float(binomtest(k, n, p).pvalue)
    except ImportError:  # normal approximation
        z = (k - n * p) / math.sqrt(n * p * (1 - p))
        return math.erfc(abs(z) / math.sqrt(2))


def _mcnemar_exact(b: int, c: int) -> float:
    n = b + c
    return 1.0 if n == 0 else min(1.0, _binom_two_sided(min(b, c), n, 0.5))


def majority_mcnemar_by_tiebreak(items: list[tuple[str, str, bool]]) -> dict:
    """Paired test of the model against every task-aware majority rule.

    ``items`` holds (use_case, gt_action, model_correct). Where the most frequent
    ground-truth action of a use case is tied, each tied action defines one majority
    rule; all combinations are tested.
    """
    from itertools import product

    ucs = sorted({uc for uc, _, _ in items})
    tops = {}
    for uc in ucs:
        counts = Counter(gt for u, gt, _ in items if u == uc)
        best = max(counts.values())
        tops[uc] = sorted(a for a, c in counts.items() if c == best)
    rules = []
    for choice in product(*(tops[uc] for uc in ucs)):
        maj = dict(zip(ucs, choice))
        b = sum(1 for uc, gt, ok in items if ok and gt != maj[uc])
        c = sum(1 for uc, gt, ok in items if gt == maj[uc] and not ok)
        rules.append({"majority": {uc: maj[uc] for uc in ucs if len(tops[uc]) > 1},
                      "b_model_only": b, "c_majority_only": c, "p_exact": _mcnemar_exact(b, c)})
    ps = [r["p_exact"] for r in rules]
    return {"rules": rules, "p_min": min(ps), "p_max": max(ps)}


def _kappa(pairs: list[tuple[str, str]]) -> float:
    n = len(pairs)
    if n == 0:
        return float("nan")
    po = sum(a == b for a, b in pairs) / n
    ca, cb = Counter(a for a, _ in pairs), Counter(b for _, b in pairs)
    pe = sum(ca[k] * cb.get(k, 0) for k in ca) / (n * n)
    return 0.0 if pe >= 1.0 else (po - pe) / (1 - pe)


def _two_prop(k1: int, n1: int, k2: int, n2: int) -> dict:
    p1, p2 = k1 / n1, k2 / n2
    p = (k1 + k2) / (n1 + n2)
    se0 = math.sqrt(p * (1 - p) * (1 / n1 + 1 / n2))
    z = (p1 - p2) / se0 if se0 else 0.0
    se = math.sqrt(p1 * (1 - p1) / n1 + p2 * (1 - p2) / n2)
    return {
        "p1": p1, "n1": n1, "p2": p2, "n2": n2, "diff": p1 - p2,
        "diff_ci95": [p1 - p2 - 1.96 * se, p1 - p2 + 1.96 * se],
        "z": z, "p_value": math.erfc(abs(z) / math.sqrt(2)),
    }


# ---------------------------------------------------------------------------
# B1
# ---------------------------------------------------------------------------


def b1_diagnostics(records, episodes) -> dict:
    out: dict = {}
    by_model = defaultdict(list)
    for r in records:
        by_model[r.model].append(r)
    for model, recs in sorted(by_model.items()):
        n = len(recs)
        k = sum(r.grounding_correct for r in recs)
        # Task-aware majority: most frequent GT action within the use case (evaluation labels).
        maj = {}
        for uc in {r.use_case for r in recs}:
            # Ties (UC2/UC4 are 200/200) are broken by action name so the baseline is deterministic;
            # the majority accuracy (50.0%) does not depend on the tie-break, the paired test does.
            counts = Counter(r.gt_action for r in recs if r.use_case == uc)
            maj[uc] = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]
        maj_ok = [r.gt_action == maj[r.use_case] for r in recs]
        b = sum(1 for r, m in zip(recs, maj_ok) if r.grounding_correct and not m)
        c = sum(1 for r, m in zip(recs, maj_ok) if m and not r.grounding_correct)
        per_uc = {}
        for uc in sorted({r.use_case for r in recs}):
            s = [r for r in recs if r.use_case == uc]
            per_uc[uc] = {
                "n": len(s),
                "accuracy": sum(r.grounding_correct for r in s) / len(s),
                "distinct_actions": len({r.model_action for r in s}),
                "modal_action": Counter(r.model_action for r in s).most_common(1)[0],
                "cohen_kappa": _kappa([(r.model_action, r.gt_action) for r in s]),
            }
        out[model] = {
            "n": n,
            "accuracy": k / n,
            "p_vs_0.5_binomial": _binom_two_sided(k, n, 0.5),
            "majority_accuracy": sum(maj_ok) / n,
            "vs_majority_mcnemar": {"b_model_only": b, "c_majority_only": c, "p_exact": _mcnemar_exact(b, c)},
            "vs_majority_mcnemar_all_tiebreaks": majority_mcnemar_by_tiebreak(
                [(r.use_case, r.gt_action, bool(r.grounding_correct)) for r in recs]),
            "per_use_case": per_uc,
        }
    return out


def b1_judgement_and_catch(tag: str, episodes: dict) -> dict:
    """Defect-judgement rates and self-catch/false-block split by precondition syntax."""
    from experiments.perception_cache import cache_path
    from xair.ai.structured_intent import precondition_syntax_ok

    latest = {}
    for line in cache_path(tag).read_text().splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        if rec.get("error") or rec.get("prompt_variant") != "blind":
            continue
        latest[(rec["frame_id"], rec["model"], rec.get("use_case"))] = rec
    per = defaultdict(lambda: Counter())
    for (fid, model, _), rec in latest.items():
        ep = episodes.get(fid)
        if ep is None:
            continue
        c = per[model]
        present = bool(ep.get("defect_present"))
        judged = rec.get("defect_judgement")
        if judged is not None:
            c["tp"] += bool(judged) and present
            c["fn"] += (not judged) and present
            c["fp"] += bool(judged) and not present
            c["tn"] += (not judged) and not present
        exprs = list(rec.get("preconditions") or [])
        valid = [e for e in exprs if precondition_syntax_ok(e)]
        syntax_ok = len(valid) == len(exprs)
        correct = rec.get("action") == rec.get("gt_action")
        if not valid and exprs:
            nominal_hold = False
        elif not valid:
            nominal_hold = True
        else:
            v = ContextValidator(nominal_context(ep))
            nominal_hold = syntax_ok and all(v._check(e)[0] for e in valid)
        refs_defect = any("defect." in e for e in exprs)
        key = "valid_syntax" if syntax_ok else "invalid_syntax"
        c[f"{key}:n_wrong"] += not correct
        c[f"{key}:n_correct"] += correct
        c[f"{key}:self_catch"] += (not correct) and not nominal_hold
        c[f"{key}:false_block"] += correct and not nominal_hold
        c["references_defect_absent"] += refs_defect
    out = {}
    for model, c in sorted(per.items()):
        tpr = c["tp"] / max(1, c["tp"] + c["fn"])
        fpr = c["fp"] / max(1, c["fp"] + c["tn"])
        row = {"TPR": tpr, "FPR": fpr, "balanced_accuracy": (tpr + 1 - fpr) / 2,
               "emissions_referencing_defect_absent": c["references_defect_absent"]}
        for key in ("valid_syntax", "invalid_syntax"):
            nw, nc = c[f"{key}:n_wrong"], c[f"{key}:n_correct"]
            row[key] = {
                "n": nw + nc,
                "self_catch": c[f"{key}:self_catch"] / nw if nw else None,
                "false_block": c[f"{key}:false_block"] / nc if nc else None,
            }
        vs = row["valid_syntax"]
        if vs["self_catch"] is not None and vs["false_block"] is not None:
            vs["discrimination_self_minus_false"] = vs["self_catch"] - vs["false_block"]
        out[model] = row
    return out


# ---------------------------------------------------------------------------
# B2
# ---------------------------------------------------------------------------


def latency_by_source(records, episodes) -> dict:
    out: dict = {}
    for model in sorted({r.model for r in records}):
        out[model] = {}
        for src in ("visa", "mvtec_ad"):
            lat = sorted(r.latency_ms for r in records
                         if r.model == model and episodes[r.frame_id].get("source_dataset") == src)
            if not lat:
                continue
            out[model][src] = {
                "n": len(lat),
                "median_ms": lat[len(lat) // 2],
                **{f"frac_over_{w}": sum(x > w for x in lat) / len(lat) for w in (2000, 4000, 8000)},
            }
    return out


def _rates(trials: list[dict]) -> dict:
    n = max(1, len(trials))
    return {
        "SAR": sum(t["successful_actuation"] for t in trials) / n,
        "hazard": sum(t["hazardous_publish"] for t in trials) / n,
        "publish_rate": sum(t["published"] for t in trials) / n,
        "n_trials": len(trials),
    }


def b2_by_source(test_primary, episodes, n_boot) -> dict:
    out: dict = {}
    for src in ("visa", "mvtec_ad"):
        sub = [r for r in test_primary if episodes[r.frame_id].get("source_dataset") == src]
        cell = {}
        trials = {}
        for gate in GATES:
            trials[gate] = generate_trials(sub, gate=gate, freshness_ms=HEADLINE["freshness_ms"])
            cell[gate] = {**_rates(trials[gate]),
                          **bootstrap_utility_frames(trials[gate], n_boot=n_boot, seed=0)}
        cell["xair_minus_freshness_only"] = paired_frame_contrast(
            trials["xair"], trials["freshness_only"], n_boot=N_PAIRED)
        out[src] = {"n_frames": len({r.frame_id for r in sub}), "gates": cell}
    return out


def b2_other_cells(test_primary, n_boot) -> dict:
    out: dict = {}
    for label, w, tau in (("w2000_tau250", 2000, 250), ("w8000_tau250", 8000, 250),
                          ("w4000_tau4000", 4000, 4000)):
        trials = {g: generate_trials(test_primary, gate=g, freshness_ms=w, drift_offset_ms=tau)
                  for g in GATES}
        out[label] = {
            "gates": {g: {**_rates(ts), **bootstrap_utility_frames(ts, n_boot=n_boot, seed=0)}
                      for g, ts in trials.items()},
            "xair_minus_freshness_only": paired_frame_contrast(
                trials["xair"], trials["freshness_only"], n_boot=N_PAIRED),
            "xair_minus_direct": paired_frame_contrast(trials["xair"], trials["direct"], n_boot=N_PAIRED),
        }
    return out


def b2_harness_echo(test_primary, episodes, n_boot) -> dict:
    """XAIR where the only precondition is the harness-supplied line.state == 'RUN'."""
    expr = "line.state == 'RUN'"
    hold = {}
    for fid in {r.frame_id for r in test_primary}:
        ep = episodes[fid]
        nominal = nominal_context(ep)
        drifted = merge_context(nominal, drift_patch(ep))
        hold[fid] = (ContextValidator(nominal)._check(expr)[0], ContextValidator(drifted)._check(expr)[0])

    class _Echo:
        def __init__(self, rec):
            self.__dict__.update({k: getattr(rec, k) for k in rec.__slots__})
            self.preconds_hold_nominal, self.preconds_hold_drifted = hold[rec.frame_id]
            self.n_preconditions = 1

    echo = [_Echo(r) for r in test_primary]
    t_echo = generate_trials(echo, gate="xair", freshness_ms=HEADLINE["freshness_ms"])
    t_xair = generate_trials(test_primary, gate="xair", freshness_ms=HEADLINE["freshness_ms"])
    return {
        "harness_echo_xair": {**_rates(t_echo), **bootstrap_utility_frames(t_echo, n_boot=n_boot, seed=0)},
        "model_emitted_xair": {**_rates(t_xair), **bootstrap_utility_frames(t_xair, n_boot=n_boot, seed=0)},
        "model_minus_echo": paired_frame_contrast(t_xair, t_echo, n_boot=N_PAIRED),
        "drift_patch_changes_line_state_fraction": sum(not h[1] for h in hold.values()) / len(hold),
    }


def b2_revoke_all_reference(test_primary, n_boot) -> dict:
    """Utility of never publishing, per B2 cell, and XAIR minus that reference.

    Revoking every intent scores -mu on each correct, admissible intent (the trials that
    XAIR scores as successful actuation or wrongful revocation) and 0 elsewhere.
    """
    out: dict = {}
    cells = (("reference", HEADLINE["freshness_ms"], HEADLINE["drift_offset_ms"]),
             ("w2000_tau250", 2000, 250), ("w8000_tau250", 8000, 250), ("w4000_tau4000", 4000, 4000))
    for label, w, tau in cells:
        xa = generate_trials(test_primary, gate="xair", freshness_ms=w, drift_offset_ms=tau)
        revoke = []
        for t in xa:
            opportunity = bool(t["successful_actuation"]) or bool(t["wrongful_revoke"])
            revoke.append(dict(t, successful_actuation=0, hazardous_publish=0,
                               wrongful_revoke=int(opportunity),
                               reward=-paper2_common.MU_WRONGFUL_REVOKE * opportunity))
        fo = generate_trials(test_primary, gate="freshness_only", freshness_ms=w, drift_offset_ms=tau)
        out[label] = {
            "U_revoke_all": sum(t["reward"] for t in revoke) / len(revoke),
            "revoke_all_bootstrap": bootstrap_utility_frames(revoke, n_boot=n_boot, seed=0),
            "revoke_all_minus_freshness_only": paired_frame_contrast(revoke, fo, fields=("reward",),
                                                                     n_boot=N_PAIRED),
            "admissible_correct_share": sum(t["wrongful_revoke"] for t in revoke) / len(revoke),
            # Re-weighted means over the same trials: revoke-all costs -mu per opportunity at any lambda.
            "U_by_lambda": {
                str(lam): {
                    "xair": sum(float(t["successful_actuation"]) - lam * float(t["hazardous_publish"])
                                - paper2_common.MU_WRONGFUL_REVOKE * float(t["wrongful_revoke"])
                                for t in xa) / len(xa),
                    "revoke_all": sum(t["reward"] for t in revoke) / len(revoke),
                }
                for lam in (1.0, 2.0, 5.0, 10.0)
            },
            "xair_minus_revoke_all": paired_frame_contrast(xa, revoke, fields=("reward",), n_boot=N_PAIRED),
        }
    return out


def b4_best_minus_primary() -> dict:
    """Paired frame-cluster contrast of the best non-oracle B4 selector with always-7B (p50 setting)."""
    by = defaultdict(list)
    path = RESULTS_DIR / "b4_eval_trials.jsonl"
    for line in path.read_text().splitlines():
        r = json.loads(line)
        if r["setting"] == "p50":
            by[r["router"]].append(r)
    return {
        "always_llama_minus_always_7b": paired_frame_contrast(
            by["static:llama3.2-vision:11b"], by["static:qwen2.5vl:7b"], n_boot=N_PAIRED),
    }


def b2_xair_gain_sources(test_primary) -> dict:
    w = HEADLINE["freshness_ms"]
    fo = generate_trials(test_primary, gate="freshness_only", freshness_ms=w)
    xa = generate_trials(test_primary, gate="xair", freshness_ms=w)
    reasons = Counter()
    for a, b in zip(fo, xa):
        assert a["frame_id"] == b["frame_id"] and a["seed"] == b["seed"]
        if a["hazardous_publish"] and not b["hazardous_publish"]:
            reasons[b["reason"]] += 1
    total = sum(reasons.values())
    lenient_schema_invalid = sum(1 for a, b in zip(fo, xa) if b["reason"] == "schema_invalid" and a["published"])
    lenient_schema_invalid_success = sum(
        1 for a, b in zip(fo, xa) if b["reason"] == "schema_invalid" and a["successful_actuation"])
    return {
        "hazards_avoided": total,
        "by_xair_reason": dict(reasons),
        "share": {k: v / total for k, v in reasons.items()} if total else {},
        "schema_invalid_published_by_freshness_only": lenient_schema_invalid,
        "schema_invalid_counted_as_success_by_freshness_only": lenient_schema_invalid_success,
        # Why full XAIR revokes context-valid intents at the reference cell (WRR composition).
        "xair_wrongful_revoke_reasons": dict(Counter(b["reason"] for b in xa if b["wrongful_revoke"])),
    }


def _reward_conservative(t: dict) -> float:
    """Over-conservative publications (stale or misgrounded) cost mu instead of lambda."""
    permissive = RANK.get(t["model_action"], -1) < RANK.get(t["gt_action"], 0)
    if t["successful_actuation"]:
        return 1.0
    if t["stale_publish"]:
        return -paper2_common.LAMBDA_HAZARD if RANK.get(t["model_action"], 0) <= 1 else -paper2_common.MU_WRONGFUL_REVOKE
    if t["unsafe_publish"]:
        return -paper2_common.LAMBDA_HAZARD if permissive else -paper2_common.MU_WRONGFUL_REVOKE
    if t["wrongful_revoke"]:
        return -paper2_common.MU_WRONGFUL_REVOKE
    return 0.0


def b2_hazard_construct(test_primary, n_boot) -> dict:
    out: dict = {}
    w = HEADLINE["freshness_ms"]
    trials = {g: generate_trials(test_primary, gate=g, freshness_ms=w) for g in GATES}
    for g, ts in trials.items():
        conservative = [t for t in ts if t["hazardous_publish"] and (
            (t["stale_publish"] and RANK.get(t["model_action"], 0) > 1)
            or (t["unsafe_publish"] and RANK.get(t["model_action"], -1) >= RANK.get(t["gt_action"], 0)))]
        alt = [dict(t, reward=_reward_conservative(t)) for t in ts]
        out[g] = {
            "hazards": sum(t["hazardous_publish"] for t in ts),
            "over_conservative_hazards": len(conservative),
            "over_conservative_share": len(conservative) / max(1, sum(t["hazardous_publish"] for t in ts)),
            "U_headline": sum(t["reward"] for t in ts) / len(ts),
            "U_conservative_aware": bootstrap_utility_frames(alt, n_boot=n_boot, seed=0),
        }
        trials[g] = alt
    out["xair_minus_freshness_only_conservative_aware"] = paired_frame_contrast(
        trials["xair"], trials["freshness_only"], fields=("reward",), n_boot=N_PAIRED)
    return out


# ---------------------------------------------------------------------------
# B3 re-selection under other cost weights / drift
# ---------------------------------------------------------------------------


def b3_reselection(tag: str, split) -> dict:
    from experiments.run_b3_validity_budget import eval_frozen, paired_records, run_oracle, run_online

    pairs = paired_records(tag)
    fid = lambda p: p[0].frame_id  # noqa: E731
    train = filter_by_split(pairs, split="train", frame_id_fn=fid, assignment=split)
    test = filter_by_split(pairs, split="test", frame_id_fn=fid, assignment=split)
    n_feat = len(feature_vector(train[0][1], train[0][0], privileged_severity=False))
    old_lambda = paper2_common.LAMBDA_HAZARD
    # Grounding accuracy of the pooled five-model test decisions the budgets act on.
    out: dict = {"test_pool": {"n": len(test),
                               "grounding": sum(bool(r.grounding_correct) for r, _ in test) / len(test)}}
    try:
        for lam in (1.0, 2.0, 5.0):
            paper2_common.LAMBDA_HAZARD = lam
            for p in (0.1, 0.25, 0.5):
                setting = {"p_drift": p, "drift_offset_ms": 250, "label": f"p{p}"}
                cell = defaultdict(list)
                for seed in SEEDS:
                    scored = []
                    for action in ACTIONS:
                        ev = eval_frozen(train, FixedBudget(action), setting=setting, seed=seed)
                        scored.append((ev["mean_reward"], action))
                    scored.sort(key=lambda x: x[0], reverse=True)
                    best = scored[0][1]
                    te = eval_frozen(test, FixedBudget(best), setting=setting, seed=seed)
                    cell["best_fixed"].append(te["mean_reward"])
                    cell["best_fixed_action"].append(best.name)
                    cell["best_fixed_WRR"].append(te["summary"].get("WRR", 0.0))
                    cell["oracle"].append(run_oracle(test, setting=setting, seed=seed)["mean_reward"])
                    if p == 0.5:
                        for learner in (LinUCB(n_feat, alpha=1.0),
                                        QLearning(epsilon=0.15, alpha=0.25, gamma=0.0, seed=seed)):
                            run_online(train, learner, epochs=2, seed=seed, learn=True)
                            ev = eval_frozen(test, learner, setting=setting, seed=seed)
                            cell[learner.name].append(ev["mean_reward"])
                            cell[f"{learner.name}_WRR"].append(ev["summary"].get("WRR", 0.0))
                out[f"lambda{lam:g}_p{p:g}"] = {
                    k: (Counter(v).most_common() if k == "best_fixed_action" else sum(v) / len(v))
                    for k, v in cell.items()
                }
    finally:
        paper2_common.LAMBDA_HAZARD = old_lambda
    out["analytic_note"] = (
        "With tau below every inference latency, a scheduled drift always precedes "
        "submission. A lenient publication then has expected reward <= (1-p) - lambda*p, "
        "a revocation >= -(1-p)*mu; at lambda=5, p=0.5 revoking dominates for any grounding "
        "accuracy."
    )
    return out


def category_holdout(records) -> dict:
    path = RESULTS_DIR / "paper2_frame_split_category_holdout.json"
    if not path.is_file():
        return {"missing": str(path)}
    assign = json.loads(path.read_text())["assignment"]
    prim = [r for r in records if r.model == PRIMARY]
    held = [r for r in prim if assign.get(r.frame_id) == "test"]
    seen = [r for r in prim if assign.get(r.frame_id) == "train"]
    return {
        "held_out_categories_vs_training_categories": _two_prop(
            sum(r.grounding_correct for r in held), len(held),
            sum(r.grounding_correct for r in seen), len(seen)),
        "note": "Disjoint frame sets. The VLM is zero-shot, so this compares category "
                "difficulty; it does not test generalisation of the VLM.",
    }


def b5_structure() -> dict:
    path = RESULTS_DIR / "b5_eval_trials.jsonl"
    if not path.is_file():
        return {"missing": str(path)}
    by = defaultdict(dict)
    for line in path.read_text().splitlines():
        t = json.loads(line)
        if t.get("setting") == "headline":
            by[t["policy"]][(t["frame_id"], t["seed"])] = t
    ss, rs = by.get("single_shot", {}), by.get("retry_stale", {})
    same = sum(1 for k in ss if k in rs and all(
        ss[k][f] == rs[k][f] for f in ("successful_actuation", "hazardous_publish")))
    return {"retry_stale_same_outcome_as_single_shot": same, "n_episodes": len(ss)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", default="phase_p")
    parser.add_argument("--n-boot", type=int, default=1000)
    args = parser.parse_args()
    random.seed(0)

    episodes = {e["frame_id"]: e for e in load_manifest()}
    split = load_split()
    records = load_replay_records(args.tag, models=None, variants={"blind"}, use_repaired=False)
    test_primary = [r for r in records if r.model == PRIMARY and split.get(r.frame_id) == "test"]

    out: dict = {"headline_config": HEADLINE, "n_boot": args.n_boot}
    print("[rev] B1 diagnostics", flush=True)
    out["b1_diagnostics"] = b1_diagnostics(records, episodes)
    out["b1_judgement_and_catch"] = b1_judgement_and_catch(args.tag, episodes)
    print("[rev] latency by source / B2 stratified", flush=True)
    out["latency_by_source"] = latency_by_source(records, episodes)
    out["b2_by_source"] = b2_by_source(test_primary, episodes, args.n_boot)
    print("[rev] B2 other cells / echo / gain sources / hazard construct", flush=True)
    out["b2_other_cells"] = b2_other_cells(test_primary, args.n_boot)
    out["b2_harness_echo"] = b2_harness_echo(test_primary, episodes, args.n_boot)
    out["b2_xair_gain_sources"] = b2_xair_gain_sources(test_primary)
    out["b2_revoke_all_reference"] = b2_revoke_all_reference(test_primary, args.n_boot)
    out["b4_best_minus_primary"] = b4_best_minus_primary()
    out["b2_hazard_construct"] = b2_hazard_construct(test_primary, args.n_boot)
    print("[rev] B3 re-selection", flush=True)
    out["b3_reselection"] = b3_reselection(args.tag, split)
    out["category_holdout"] = category_holdout(records)
    out["b5_structure"] = b5_structure()

    path = RESULTS_DIR / "paper2_revision_analyses.json"
    path.write_text(json.dumps(out, indent=2, default=str))
    print(f"[rev] wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
