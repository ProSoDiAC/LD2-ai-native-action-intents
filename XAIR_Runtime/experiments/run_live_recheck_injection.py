#!/usr/bin/env python3
"""Live gate check with a context change injected between validation (t_v) and release (t_g).

Closes the gap that the earlier live consistency check injected no drift between t_v and t_g.
Per trial: nominal context -> POST intent (t_v) -> inject one change -> wait q -> POST commit (t_g).
Changes: none, invalidating (line.state leaves RUN), predicate-preserving (robot.speed 0.05->0.1 under robot.speed <= 0.5),
irrelevant (a path outside the read set). Commit scopes: readset (default XAIR) and truth (semantic).
Plus an age case: no change, window shorter than the queue. Output: results/live_recheck_injection.json.
"""
from __future__ import annotations
import argparse, json, sys, time, uuid
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.paper2_common import RESULTS_DIR, _post  # noqa: E402

NOMINAL = {"line": {"state": "RUN"}, "gripper": {"state": "OPEN"}, "robot": {"speed": 0.05}, "cell": {"interlock": "CLEAR", "aux": 0}}
CHANGES = {
    "none": None,
    "invalidating": {"line": {"state": "PAUSED"}},
    "predicate_preserving": {"robot": {"speed": 0.1}},
    "irrelevant": {"cell": {"aux": 1}},
}


def intent(window_ms: int, expr: str = "line.state == 'RUN'") -> dict:
    return {"id": str(uuid.uuid4()), "source": "ai",
            "timestamp_decision": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z",
            "freshness_window_ms": window_ms,
            "preconditions": [{"expr": expr}],
            "payload": {"action_type": "RESUME", "target_entity": "line_1"},
            "extensions": {"model_id": "live-inject", "inference_latency_ms": 0, "evidence": {"frame_id": "inject"}}}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--xair", default="http://127.0.0.1:18080")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--queue-ms", type=int, default=50)
    a = ap.parse_args()
    out = {"xair": a.xair, "n_per_cell": a.n, "queue_ms": a.queue_ms, "cells": {}}
    plan = [(c, s, 4000, a.queue_ms) for c in CHANGES for s in ("readset", "truth")]
    plan += [("none_age_expired", s, 100, a.queue_ms + 150) for s in ("readset", "truth")]
    for change, scope, window, wait in plan:
        res = {}
        for _ in range(a.n):
            _post(f"{a.xair}/v1/context/snapshot", NOMINAL)
            r = _post(f"{a.xair}/v1/intents", intent(window, "robot.speed <= 0.5" if change == "predicate_preserving" else "line.state == 'RUN'"))
            if r.get("outcome") != "EXECUTE":
                res["not_authorized"] = res.get("not_authorized", 0) + 1
                continue
            patch = CHANGES.get(change)
            if patch:
                _post(f"{a.xair}/v1/context/snapshot", patch)
            time.sleep(wait / 1000)
            c = _post(f"{a.xair}/v1/intents/{r['id']}/commit", {"scope": scope})
            res[c["status"]] = res.get(c["status"], 0) + 1
        out["cells"][f"{change}/{scope}"] = res
        print(change, scope, res, flush=True)
    (RESULTS_DIR / "live_recheck_injection.json").write_text(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
