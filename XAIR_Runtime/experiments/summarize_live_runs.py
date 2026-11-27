#!/usr/bin/env python3
"""Aggregate the repeated live checks into the summaries quoted in the manuscript.

Reads the per-run outputs of ``run_b2_fidelity.py --seed s --out-suffix _seed{s}`` and
``run_gate_latency_bench.py --out-suffix _run{r}`` and writes ``b2_fidelity_runs.json`` and
``gate_latency_runs.json``. Offline only: it never contacts the live stack.

    python3 experiments/summarize_live_runs.py
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.paper2_common import RESULTS_DIR  # noqa: E402


def _run_index(path: Path, prefix: str) -> int:
    return int(path.stem.removeprefix(prefix))


def _fidelity_run(path: Path) -> dict:
    s = json.loads(path.read_text())["summary"]
    rows = list(csv.DictReader(path.with_suffix(".csv").open()))
    discordant = [
        {
            "frame_id": r["frame_id"], "gate": r["gate"], "anchor": r["anchor"],
            "latency_ms": float(r["latency_ms"]), "freshness_ms": float(r["freshness_ms"]),
            "offline_published": r["offline_published"] == "True",
            "live_reason": r["live_reason"],
        }
        for r in rows if r["agree"] != "True"
    ]
    return {
        "model": s["model"], "seed": s["seed"], "n_frames": s["n_frames"],
        "n_comparisons": s["n_comparisons"], "disagreements": s["disagreements"],
        "disagreement_rate": s["disagreement_rate"],
        "max_distance_to_window_ms": max(
            (abs(d["freshness_ms"] - d["latency_ms"]) for d in discordant), default=None),
        "discordant": discordant, "provenance": s.get("provenance"),
    }


def fidelity() -> dict:
    runs = [_fidelity_run(p) for p in sorted(RESULTS_DIR.glob("b2_fidelity_seed[0-9]*.json"),
                                              key=lambda p: _run_index(p, "b2_fidelity_seed"))]
    all_models = [_fidelity_run(p) for p in sorted(RESULTS_DIR.glob("b2_fidelity_allframes_*.json"))]
    first = json.loads((RESULTS_DIR / "b2_fidelity_seed1.json").read_text())["summary"]
    return {
        "config": {k: first[k] for k in ("model", "freshness_ms", "p_drift", "drift_offset_ms", "n_frames")},
        "disagreements": [r["disagreements"] for r in runs],
        "max_rate": max(r["disagreement_rate"] for r in runs),
        "max_distance_to_window_ms": max((r["max_distance_to_window_ms"] or 0.0) for r in runs),
        "runs": runs,
        # One run per model over 1,990 frames: every decision family the offline suites replay,
        # including the out-of-vocabulary actions that the AI schema now rejects live.
        "all_models": {r["model"]: {k: r[k] for k in ("n_comparisons", "disagreements", "disagreement_rate",
                                                     "max_distance_to_window_ms")} for r in all_models},
    }


def latency() -> dict:
    runs = []
    for path in sorted(RESULTS_DIR.glob("gate_latency_run[0-9]*.json"), key=lambda p: _run_index(p, "gate_latency_run")):
        d = json.loads(path.read_text())
        runs.append({
            "run": _run_index(path, "gate_latency_run"), "n": d["n"], "run_utc": d["run_utc"],
            "p50_ms": d["latency_ms"]["p50"], "p95_ms": d["latency_ms"]["p95"],
            **{k: d.get(k) for k in ("host", "loadavg_1m", "xair_version", "code_commit",
                                     "runtime_or_gateway_modified")},
        })
    return {
        "mode": "xair (optimistic gate recheck), adapter to XAIR over HTTP",
        "p50_range_ms": [min(r["p50_ms"] for r in runs), max(r["p50_ms"] for r in runs)],
        "p95_range_ms": [min(r["p95_ms"] for r in runs), max(r["p95_ms"] for r in runs)],
        "runs": runs,
    }


def main() -> int:
    for name, payload in (("b2_fidelity_runs.json", fidelity()), ("gate_latency_runs.json", latency())):
        (RESULTS_DIR / name).write_text(json.dumps(payload, indent=2) + "\n")
        print(f"[live-summary] wrote {name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
