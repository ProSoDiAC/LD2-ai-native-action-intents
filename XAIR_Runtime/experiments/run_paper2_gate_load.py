#!/usr/bin/env python3
"""Load and concurrency test of the XAIR gate behind its HTTP API.

Phase 1 (throughput): closed-loop clients submit an intent and commit it; reports
throughput and submit+commit latency per concurrency level.
Phase 2 (concurrency correctness): the same clients run while a writer flips
``line.state`` between RUN and STOP. An intent whose precondition is
``line.state == 'RUN'`` must not be committed if a change of that path was
acknowledged strictly between the end of its submission and the start of its
commit (a *certain* violation, expected 0). Also reports how many intents were
revoked at validation, revoked at commit, and committed.

Needs the gate and a Redis instance, e.g.
  REDIS_URL=redis://127.0.0.1:16399/0 XAIR_SCHEMA=action-intent-v1-ai.json \
  uvicorn xair.adapters.http_server:app --port 18391
and PAPER2_XAIR=http://127.0.0.1:18391.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import sys
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import http.client
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = Path(__file__).resolve().parent / "results"
URL = os.environ.get("PAPER2_XAIR", "http://127.0.0.1:18391")


def body() -> dict:
    return {
        "id": str(uuid.uuid4()),
        "source": "ai",
        "timestamp_decision": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z",
        "freshness_window_ms": 5000,
        "preconditions": [{"expr": "line.state == 'RUN'"}],
        "safety_constraints": [{"expr": "gripper.state == 'OPEN'"}],
        "payload": {"action_type": "RESUME", "target_entity": f"line_{uuid.uuid4().hex[:8]}"},  # distinct target: no resource conflict
        "extensions": {"model_id": "load", "inference_latency_ms": 0, "evidence": {"frame_id": "load"}},
    }


def pct(v: list[float], q: float) -> float:
    s = sorted(v)
    return s[min(len(s) - 1, max(0, int(round(q * (len(s) - 1)))))] if s else 0.0


class Sess:
    """Keep-alive HTTP/1.1 client (stdlib) with one persistent connection per instance."""

    def __init__(self) -> None:
        u = urlparse(URL)
        self.c = http.client.HTTPConnection(u.hostname, u.port, timeout=30)

    def post(self, path: str, payload: dict) -> tuple[int, dict]:
        self.c.request("POST", path, json.dumps(payload), {"Content-Type": "application/json"})
        r = self.c.getresponse()
        return r.status, json.loads(r.read().decode())


def set_ctx(sess: Sess, state: str, aux: int | None = None) -> None:
    doc = {"line": {"state": state}, "gripper": {"state": "OPEN"}}
    if aux is not None:
        doc["aux"] = {"counter": aux}
    sess.post("/v1/context/snapshot", doc)


def one(sess: Sess, hold_ms: float = 0.0) -> dict:
    b = body()
    t0 = time.perf_counter(); w0 = time.time()
    _, rec = sess.post("/v1/intents", b)
    t1 = time.perf_counter(); w1 = time.time()
    out = {"validated": rec.get("state") == "AUTHORIZED", "val_reason": rec.get("reason"),
           "sub_start": w0, "sub_end": w1, "state": rec.get("state")}
    if out["validated"]:
        if hold_ms:
            time.sleep(hold_ms / 1000.0)  # queueing delay between validation and commit
        w2 = time.time()
        status, cj = sess.post(f"/v1/intents/{b['id']}/commit", {})
        w3 = time.time()
        out.update(com_start=w2, com_end=w3, committed=(status == 200 and bool(cj.get("released"))), commit_reason=cj.get("reason"))
    out["lat_ms"] = (time.perf_counter() - t0) * 1000.0
    out["http_ms"] = (t1 - t0) * 1000.0
    return out


def phase1(levels: list[int], per_level: int) -> list[dict]:
    rows = []
    for c in levels:
        sess = [Sess() for _ in range(c)]
        set_ctx(sess[0], "RUN")
        n_each = max(1, per_level // c)
        lat: list[float] = []
        lock = threading.Lock()

        def worker(i: int) -> None:
            loc = []
            for _ in range(n_each):
                loc.append(one(sess[i])["lat_ms"])
            with lock:
                lat.extend(loc)

        t0 = time.perf_counter()
        with ThreadPoolExecutor(c) as ex:
            list(ex.map(worker, range(c)))
        dt = time.perf_counter() - t0
        rows.append({"clients": c, "intents": len(lat), "seconds": round(dt, 3),
                     "throughput_per_s": round(len(lat) / dt, 1),
                     "p50_ms": round(pct(lat, .5), 2), "p95_ms": round(pct(lat, .95), 2),
                     "p99_ms": round(pct(lat, .99), 2), "mean_ms": round(statistics.mean(lat), 2)})
        print(rows[-1], flush=True)
    return rows


def phase2(c: int, per_client: int, flip_hz: float, hold_ms: float, target: str = "read") -> dict:
    """target: read (flip line.state, a path the intents read), irrelevant (bump aux.counter, never read), none (no writer)."""
    sess = [Sess() for _ in range(c)]
    wsess = Sess()
    set_ctx(wsess, "RUN")
    writes: list[tuple[float, float, str]] = []  # (start, end, new_state)
    stop = threading.Event()

    def writer() -> None:
        state, n = "RUN", 0
        while not stop.is_set() and target != "none":
            n += 1
            if target == "read":
                state = "STOP" if state == "RUN" else "RUN"
            s = time.time(); set_ctx(wsess, state, n); e = time.time()
            if target == "read":
                writes.append((s, e, state))
            time.sleep(max(0.0, 1.0 / flip_hz - (e - s)))
        while not stop.is_set():
            time.sleep(0.01)

    results: list[dict] = []
    lock = threading.Lock()

    def worker(i: int) -> None:
        loc = [one(sess[i], hold_ms) for _ in range(per_client)]
        with lock:
            results.extend(loc)

    th = threading.Thread(target=writer); th.start()
    with ThreadPoolExecutor(c) as ex:
        list(ex.map(worker, range(c)))
    stop.set(); th.join()

    validated = [r for r in results if r["validated"]]
    committed = [r for r in validated if r.get("committed")]
    reasons: dict[str, int] = {}
    for r in results:
        if not r["validated"]:
            key = str(r.get("val_reason"))
            reasons[key] = reasons.get(key, 0) + 1
    commit_reasons: dict[str, int] = {}
    for r in validated:
        key = str(r.get("commit_reason"))
        commit_reasons[key] = commit_reasons.get(key, 0) + 1
    exposed = certain = 0
    for r in validated:
        # a change acknowledged fully inside (submission end, commit start) must have been seen at commit
        if any(st >= r["sub_end"] and e <= r["com_start"] for st, e, _ in writes):
            exposed += 1
            certain += bool(r.get("committed"))
    return {"clients": c, "intents": len(results), "writes": len(writes), "flip_hz": flip_hz, "hold_ms": hold_ms, "writer": target,
            "revoked_at_validation": len(results) - len(validated),
            "validated": len(validated), "committed": len(committed),
            "revoked_at_commit": len(validated) - len(committed),
            "validation_revocation_reasons": reasons, "commit_reasons": commit_reasons, "exposed_to_certain_change": exposed, "certain_violations": certain}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--levels", default="1,4,16,64")
    ap.add_argument("--per-level", type=int, default=2000)
    ap.add_argument("--p2-clients", type=int, default=8)
    ap.add_argument("--p2-per-client", type=int, default=500)
    ap.add_argument("--flip-hz", type=float, default=100.0)
    ap.add_argument("--hold-ms", type=float, default=20.0)
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--out", default="paper2_gate_load.json")
    a = ap.parse_args()
    levels = [int(x) for x in a.levels.split(",")]
    p1 = phase1(levels, a.per_level)
    p2 = phase2(a.p2_clients, a.p2_per_client, a.flip_hz, a.hold_ms)
    print(p2, flush=True)
    controls = []
    for target in ("none", "irrelevant", "read"):
        for rep in range(a.reps):
            row = phase2(a.p2_clients, a.p2_per_client // 2, a.flip_hz, a.hold_ms, target)
            row["rep"] = rep
            controls.append(row)
            print(row, flush=True)
    out = {"host": {"python": platform.python_version(), "machine": platform.machine(), "cpus": os.cpu_count()},
           "gate": "single uvicorn worker, Redis context store, HTTP on localhost",
           "phase1_throughput": p1, "phase2_concurrency": p2, "phase3_controls": controls}
    (RESULTS_DIR / a.out).write_text(json.dumps(out, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
