#!/usr/bin/env python3
"""Live check with an OPC UA cell in the loop (frames replayed as the camera, the VLM run live).

Per trial: nominal context -> live 7B inference on a test frame (capture time is the anchor) ->
a Poisson drift event writes line.state=PAUSED through the OPC UA server while the model is still
thinking -> the intent is submitted to the XAIR HTTP gate twice (freshness-only: no preconditions;
full XAIR: the model's emitted preconditions) and committed after a short queue. Ground truth for
validity is the line.state read back from the OPC UA node at release. The OPC UA server and bridge
come from the LD1 copy in live_cell_ld1clone/; the gate is this repository's.
Output: results/paper2_live_opcua.json (per-trial rows and per-path summary).
Audit fields per row (do not alter the protocol): emitted preconditions (strings), the full gate
responses (submit and commit) of both gated paths, the line.state value read at each path's release,
and wall-clock timestamps of capture, submit, commit, release read and the drift write.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import random
import sys
import time
import uuid
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.paper2_common import DATASET_ROOT, RESULTS_DIR, _post, load_manifest  # noqa: E402
from experiments.paper2_splits import load_split  # noqa: E402
from xair.ai.ollama_client import OllamaClient  # noqa: E402
from xair.ai.structured_intent import StructuredIntentProducer  # noqa: E402

MODEL = "qwen2.5vl:7b"
VARIANT = "blind_v2"
LAMBDA, MU = 5.0, 1.0


def iso(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


def nominal(ep: dict) -> dict:
    return {"line": {"state": "RUN"}, "gripper": {"state": "OPEN"}, "robot": {"speed": 0.05},
            "defect": {"absent": not ep.get("defect_present", False)}}


def intent(res, preconds: list[str], window_ms: int, target: str = "line_1") -> dict:
    return {"id": str(uuid.uuid4()), "source": "ai", "timestamp_decision": res.capture_ts,
            "freshness_window_ms": window_ms,
            "preconditions": [{"expr": e} for e in preconds],
            "payload": {"action_type": res.action, "target_entity": target},
            "extensions": {"model_id": MODEL, "inference_latency_ms": res.latency_ms,
                           "evidence": {"frame_id": res.frame_id}}}


def classify(released: bool, valid: bool, correct: bool) -> str:
    if released:
        return "success" if valid and correct else ("stale" if not valid else "misgrounded")
    return "forgone" if valid and correct else "correct_revoke"


def reward(kind: str) -> float:
    return {"success": 1.0, "stale": -LAMBDA, "misgrounded": -LAMBDA, "forgone": -MU}.get(kind, 0.0)


def contrasts(ok_rows: list[dict], n_boot: int = 10000) -> dict:
    """Paired frame bootstrap of U differences (one drift draw per frame)."""
    import numpy as np

    def u(r, path):
        if path == "revoke_all":
            return reward("forgone" if r["valid_at_release"] and r["correct"] else "correct_revoke")
        return reward(r[path + "_kind"])

    rng = np.random.default_rng(0)
    idx = rng.integers(0, len(ok_rows), size=(n_boot, len(ok_rows)))
    res = {}
    for a, b in (("xair", "freshness_only"), ("xair", "revoke_all"), ("xair", "direct")):
        d = np.array([u(r, a) - u(r, b) for r in ok_rows])
        boots = d[idx].mean(axis=1)
        res[f"{a}_minus_{b}"] = {"delta": float(d.mean()), "ci95": [float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975))]}
    return res


async def main_async(a) -> int:
    from asyncua import Client

    split = load_split()
    eps = [e for e in load_manifest() if split.get(e["frame_id"]) == "test"]
    random.Random(a.seed).shuffle(eps)
    eps = eps[: a.n]
    producer = StructuredIntentProducer(OllamaClient())
    rng = random.Random(a.seed + 1)
    rows = []
    async with Client(url=f"opc.tcp://127.0.0.1:{a.port}/xair/hil/") as cli:
        idx = await cli.get_namespace_index("http://xair/hil")
        line = cli.get_node(f"ns={idx};i=1")
        state_node = await line.get_child(f"{idx}:State")
        method = await line.get_child(f"{idx}:SetLineState")

        async def set_state(s: str) -> None:
            await line.call_method(method, s)

        for i, ep in enumerate(eps):
            await set_state("RUN")
            _post(f"{a.xair}/v1/context/snapshot", nominal(ep))
            delay = rng.expovariate(a.rate)

            drift_ts: dict = {}

            async def drift():
                await asyncio.sleep(delay)
                drift_ts["start"] = iso(datetime.utcnow())
                await set_state("PAUSED")
                drift_ts["done"] = iso(datetime.utcnow())

            task = asyncio.create_task(drift())
            t0 = time.monotonic()
            res = await asyncio.to_thread(
                producer.produce, ep, variant=VARIANT, use_case=ep["use_case"],
                image_path=DATASET_ROOT / ep["path"], model=MODEL)
            gt = ep["use_cases"][ep["use_case"]]["ground_truth_action"]
            if res.error or not res.action:
                task.cancel()
                rows.append({"frame_id": ep["frame_id"], "error": res.error or "no action"})
                continue
            preconds = list(res.preconditions or [])
            correct = res.action == gt
            pre_state = await state_node.read_value()
            pre_valid = pre_state == "RUN"
            pre_read_ts = iso(datetime.utcnow())

            async def run_path(path: str, pre: list[str]) -> dict:
                """Submit, queue, commit and read the node at this path's own release instant."""
                # one target per path: the gate serialises commits per target, and the paths must not block each other
                it = intent(res, pre, a.window_ms, target=f"line_1_{path}")
                ts = {"submit": iso(datetime.utcnow())}
                r = await asyncio.to_thread(_post, f"{a.xair}/v1/intents", it)
                ts["submit_response"] = iso(datetime.utcnow())
                ok = r.get("outcome") == "EXECUTE"
                await asyncio.sleep(a.queue_ms / 1000)
                out = {path + "_reason": r.get("reason") or r.get("outcome"),
                       path + "_preconditions": list(pre), path + "_gate_submit": r}
                if ok:
                    ts["commit"] = iso(datetime.utcnow())
                    c = await asyncio.to_thread(_post, f"{a.xair}/v1/intents/{r['id']}/commit", {"scope": "readset"})
                    ts["commit_response"] = iso(datetime.utcnow())
                    ok = c.get("status") == "committed"
                    out[path + "_commit"] = c.get("status")
                    out[path + "_gate_commit"] = c
                out[path] = ok
                rel_state = await state_node.read_value()
                ts["release_read"] = iso(datetime.utcnow())
                out[path + "_line_state_at_release"] = rel_state
                out[path + "_valid"] = rel_state == "RUN"
                out[path + "_timestamps"] = ts
                return out

            # Both paths run concurrently on the same drifting world; each is judged against the
            # node value read right after its own commit, not after the other path.
            parts = await asyncio.gather(run_path("freshness_only", []), run_path("xair", preconds))
            sub = {k: v for part in parts for k, v in part.items()}
            task.cancel()
            row = {"frame_id": ep["frame_id"], "use_case": ep["use_case"], "action": res.action, "gt": gt,
                   "correct": correct, "valid_before_submit": pre_valid,
                   "valid_at_release": sub["xair_valid"], "latency_ms": res.latency_ms,
                   "emitted_preconditions": preconds, "capture_ts": res.capture_ts,
                   "line_state_before_submit": pre_state, "before_submit_read_ts": pre_read_ts,
                   "drift_timestamps": dict(drift_ts), "drift_fired": bool(drift_ts.get("done")),
                   "drift_delay_ms": delay * 1000, "elapsed_ms": (time.monotonic() - t0) * 1000, **sub}
            row["synchronised"] = (pre_valid == sub["freshness_only_valid"] == sub["xair_valid"])
            for path in ("direct", "freshness_only", "xair"):
                released = True if path == "direct" else bool(sub[path])
                v = row["valid_before_submit"] if path == "direct" else sub[path + "_valid"]
                row[path + "_kind"] = classify(released, v, correct)
            rows.append(row)
            print(f"[{i + 1}/{len(eps)}] lat={res.latency_ms:.0f}ms valid={row['valid_at_release']} sync={row['synchronised']} xair={sub['xair']} fresh={sub['freshness_only']}", flush=True)
    ok_rows = [r for r in rows if "error" not in r]
    summary = {}
    for path in ("direct", "freshness_only", "xair", "revoke_all"):
        kinds = [("forgone" if (r["valid_at_release"] and r["correct"]) else "correct_revoke") if path == "revoke_all"
                 else r[path + "_kind"] for r in ok_rows]
        n = len(kinds)
        summary[path] = {"n": n, **{k: kinds.count(k) / n for k in ("success", "stale", "misgrounded", "forgone", "correct_revoke")},
                         "U": sum(reward(k) for k in kinds) / n}
    sync_rows = [r for r in ok_rows if r["synchronised"]]
    sync_summary = {}
    for path in ("direct", "freshness_only", "xair", "revoke_all"):
        kinds = [("forgone" if (r["valid_at_release"] and r["correct"]) else "correct_revoke") if path == "revoke_all"
                 else r[path + "_kind"] for r in sync_rows]
        n = len(kinds)
        sync_summary[path] = {"n": n, "U": sum(reward(k) for k in kinds) / n if n else None}
    out = {"design": "concurrent paths, per-path release-time validity", "synchronised_only": {"n": len(sync_rows), "summary": sync_summary, "contrasts": contrasts(sync_rows) if sync_rows else None},
           "contrasts": contrasts(ok_rows), "model": MODEL, "variant": VARIANT, "rate_per_s": a.rate, "window_ms": a.window_ms, "queue_ms": a.queue_ms,
           "n_requested": a.n, "n_ok": len(ok_rows), "summary": summary, "rows": rows}
    path = RESULTS_DIR / a.out
    path.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps({"summary": summary, "contrasts": out["contrasts"]}, indent=2))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--xair", default="http://127.0.0.1:18090")
    ap.add_argument("--port", type=int, default=4841)
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--rate", type=float, default=0.2, help="Poisson drift rate, 1/s")
    ap.add_argument("--window-ms", type=int, default=4000)
    ap.add_argument("--queue-ms", type=int, default=50)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default="paper2_live_opcua.json")
    return asyncio.run(main_async(ap.parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())
