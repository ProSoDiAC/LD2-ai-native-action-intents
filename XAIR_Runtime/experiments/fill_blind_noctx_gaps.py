#!/usr/bin/env python3
"""Fill residual qwen2.5vl:7b blind_noctx gaps with 768px images."""

from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path

from experiments.paper2_common import DATASET_ROOT, load_manifest
from experiments.perception_cache import CacheKey, load_cache, plan_jobs
from xair.ai.ollama_client import OllamaClient
from xair.ai.structured_intent import StructuredIntentProducer

ROOT = Path(__file__).resolve().parents[2]
CACHE = Path("experiments/results/perception_cache/phase_p.jsonl")
ERR = CACHE.with_suffix(".errors.jsonl")


def pending_jobs():
    _, done = load_cache(CACHE)
    jobs = plan_jobs(
        load_manifest(),
        models=["qwen2.5vl:7b"],
        variants=["blind_noctx"],
        primary_model="qwen2.5vl:7b",
        limit=None,
    )
    return [
        (ep, m, v)
        for ep, m, v in jobs
        if CacheKey(ep["frame_id"], m, v, ep["use_case"]) not in done
    ]


def bounce_ollama() -> None:
    # Ollama runs on this host (the L40 node); -n fails fast instead of prompting.
    subprocess.run(
        ["sudo", "-n", "systemctl", "restart", "ollama"],
        check=False,
        capture_output=True,
    )
    time.sleep(6)


def main() -> int:
    # Force 768 for these historically hang-prone frames.
    orig = OllamaClient.encode_image
    OllamaClient.encode_image = staticmethod(
        lambda path, max_side=None: orig(path, max_side=768)
    )

    pending = pending_jobs()
    print(f"[fill] pending={len(pending)}", flush=True)
    client = OllamaClient()
    producer = StructuredIntentProducer(client)
    ok = fail = 0

    for i, (ep, model, variant) in enumerate(pending, 1):
        img = DATASET_ROOT / ep["path"]
        result = producer.produce(
            ep,
            variant=variant,
            use_case=ep["use_case"],
            image_path=img,
            model=model,
        )
        if result.error and (
            "timed out" in result.error.lower() or "500" in result.error
        ):
            try:
                client.unload(model)
            except Exception:
                pass
            time.sleep(1)
            result = producer.produce(
                ep,
                variant=variant,
                use_case=ep["use_case"],
                image_path=img,
                model=model,
            )

        record = result.to_json()
        record["gt_action"] = ep["use_cases"][ep["use_case"]]["ground_truth_action"]
        record["severity"] = ep.get("severity")
        record["category"] = ep.get("category")
        record["source_dataset"] = ep.get("source_dataset")
        record["defect_present"] = ep.get("defect_present")
        record["gap_fill"] = "blind_noctx_768"

        if result.error:
            fail += 1
            ERR.parent.mkdir(parents=True, exist_ok=True)
            with ERR.open("a") as es:
                es.write(json.dumps(record) + "\n")
            print(
                f"[fill] FAIL {i}/{len(pending)} {ep['frame_id']}: {result.error[:100]}",
                flush=True,
            )
            if fail % 2 == 0:
                bounce_ollama()
                client = OllamaClient()
                producer = StructuredIntentProducer(client)
            continue

        with CACHE.open("a") as sink:
            sink.write(json.dumps(record) + "\n")
        ok += 1
        print(
            f"[fill] OK {i}/{len(pending)} {ep['frame_id']} "
            f"{result.latency_ms/1000:.1f}s schema={result.schema_valid}",
            flush=True,
        )

    remaining = len(pending_jobs())
    summary = {"ok": ok, "fail": fail, "remaining": remaining}
    print(json.dumps(summary), flush=True)
    Path("experiments/results/perception_cache/blind_noctx_gap_fill.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    return 0 if remaining == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
