#!/usr/bin/env python3
"""Benign-race replay: does the gateway recheck revoke intents whose context changed harmlessly?

Output: paper2_benign_races.json. Replays the preserved decisions; no inference is run.

A queueing delay q separates validation (t_v, at the evidence age) from release (t_g = t_v + q),
as in the queueing-delay replay of run_paper2_review_analyses.py. One context change happens at
T ~ Exp(r) after evidence capture and belongs to one of three classes:

  invalidating        line.state leaves RUN: the integrator-declared predicate becomes false
  predicate-preserving a write to line.state that keeps it RUN: the read-set version changes,
                      the predicate still holds
  irrelevant          a write to a path outside the intent's read set

Every intent carries the integrator-declared predicate line.state == 'RUN' (as in the
declared-predicate arm), so line.state is in every read set. The context is invalid at release
only after an invalidating change. Release policies, all with the age check of the gate:

  validate_only  checks at t_v only
  global_version revokes if any context version changed in the queue
  readset_version revokes if a read-set version changed in the queue
  semantic       re-evaluates the predicates at t_g
  full_xair      semantic and read-set version (Algorithm 1)
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.paper2_common import RESULTS_DIR, Measurement, classify, trial_reward  # noqa: E402
from experiments.paper2_splits import load_split  # noqa: E402
from experiments.run_b2_validity_frontier import gate_publishes  # noqa: E402
from experiments.run_paper2_latency_factorial import with_age  # noqa: E402
from experiments.run_paper2_review_analyses import W_MS, contrast, rates, test_records  # noqa: E402
from experiments.run_paper2_stats import SEEDS  # noqa: E402

CLASSES = ("invalidating", "predicate_preserving", "irrelevant")
POLICIES = ("validate_only", "global_version", "readset_version", "semantic", "full_xair")
RATES = (0.2, 1.0)
QUEUES_MS = (250, 1000, 2000)


def release(policy: str, *, passed_v: bool, age_ok_g: bool, race_in_queue: bool, cls: str,
            pred_g: bool) -> bool:
    if not passed_v:
        return False
    if policy == "validate_only":
        return True
    if not age_ok_g:
        return False
    readset_changed = race_in_queue and cls in ("invalidating", "predicate_preserving")
    if policy == "global_version":
        return not race_in_queue
    if policy == "readset_version":
        return not readset_changed
    if policy == "semantic":
        return pred_g
    return pred_g and not readset_changed  # full_xair


def replay(recs, cls: str, r_rate: float, q_ms: int) -> dict[str, list[dict]]:
    arms: dict[str, list[dict]] = {p: [] for p in POLICIES}
    for seed in SEEDS:
        for rec in recs:
            rng = random.Random(f"{seed}:{rec.frame_id}")
            t_change = rng.expovariate(r_rate) * 1000.0
            t_v, t_g = rec.latency_ms, rec.latency_ms + q_ms
            invalid_by_v = cls == "invalidating" and t_change <= t_v
            invalid_by_g = cls == "invalidating" and t_change <= t_g
            race_in_queue = t_v < t_change <= t_g
            # the model's own preconditions see the nominal context unless the change invalidates
            pred_v = rec.preconds_hold_nominal and not invalid_by_v
            pred_g = rec.preconds_hold_nominal and not invalid_by_g
            passed_v, _ = gate_publishes("xair", elapsed_ms=t_v, freshness_ms=W_MS, preconds_hold=pred_v,
                                         schema_valid=rec.gate_schema_valid)
            valid = not invalid_by_g
            for policy in POLICIES:
                published = release(policy, passed_v=passed_v, age_ok_g=t_g <= W_MS,
                                    race_in_queue=race_in_queue, cls=cls, pred_g=pred_g)
                s = classify(gt_action=rec.gt_action, model_action=rec.model_action, published=published,
                             measurement=Measurement(valid, valid, 1, 1))
                s.update(frame_id=rec.frame_id, seed=seed)
                s["reward"] = trial_reward(s)
                arms[policy].append(s)
    return arms


def main() -> int:
    split = load_split()
    producers = {
        "qwen7b_blind_declared": test_records("phase_p", {"qwen2.5vl:7b"}, {"blind"}, split),
        "detector_age1500ms": with_age(test_records("detector_patchcore", None, None, split), 1500.0),
    }
    out: dict = {"window_ms": W_MS, "classes": list(CLASSES), "policies": list(POLICIES), "cells": {}}
    for pname, recs in producers.items():
        cells = out["cells"].setdefault(pname, {})
        for r_rate in RATES:
            for q_ms in QUEUES_MS:
                for cls in CLASSES:
                    arms = replay(recs, cls, r_rate, q_ms)
                    cell = {p: rates(ts) for p, ts in arms.items()}
                    cell["readset_version_minus_semantic"] = contrast(arms["readset_version"], arms["semantic"])
                    cell["full_xair_minus_semantic"] = contrast(arms["full_xair"], arms["semantic"])
                    cell["global_version_minus_semantic"] = contrast(arms["global_version"], arms["semantic"])
                    cell["full_xair_minus_validate_only"] = contrast(arms["full_xair"], arms["validate_only"])
                    cell["change_in_queue_share"] = sum(
                        1 for seed in SEEDS for rec in recs
                        if rec.latency_ms < random.Random(f"{seed}:{rec.frame_id}").expovariate(r_rate) * 1000.0
                        <= rec.latency_ms + q_ms) / (len(SEEDS) * len(recs))
                    cells[f"r{r_rate:g}_q{q_ms}_{cls}"] = cell
                    print(pname, r_rate, q_ms, cls,
                          {p: round(cell[p]["U"], 3) for p in POLICIES}, flush=True)
    path = RESULTS_DIR / "paper2_benign_races.json"
    path.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
    print("wrote", path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
