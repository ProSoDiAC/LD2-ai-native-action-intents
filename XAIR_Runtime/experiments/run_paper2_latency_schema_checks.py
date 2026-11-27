"""Checks behind two review points: the VisA/MVTec latency gap and the 7B schema-validity drop.

Reads the cached 7B perception records only. Prints image size, retry share and latency per
image source for the blind campaign, and the origin of invalid intents in the original and
the corrected-prompt campaigns.
"""
import io
import json
import re
import statistics as st
from collections import defaultdict
from datetime import datetime
from pathlib import Path

from PIL import Image

HERE = Path(__file__).resolve().parent
CACHE = HERE / "results" / "perception_cache"
DATA = HERE / "datasets" / "manufacturing-a1"
MODEL = "qwen2.5vl:7b"


def load(name, variant):
    with open(CACHE / name) as fh:
        rows = (json.loads(line) for line in fh)
        return [r for r in rows if r["model"] == MODEL and r["prompt_variant"] == variant]


def gap_ms(r):
    f = lambda t: datetime.fromisoformat(t.replace("Z", "+00:00"))
    return (f(r["emitted_ts"]) - f(r["capture_ts"])).total_seconds() * 1000


def latency_by_source():
    frames = {}
    with open(DATA / "manifest.jsonl") as fh:
        for line in fh:
            r = json.loads(line)
            frames[r["frame_id"]] = r
    acc = defaultdict(lambda: defaultdict(list))
    for r in load("phase_p.jsonl", "blind"):
        f = frames[r["frame_id"]]
        im = Image.open(DATA / f["path"]).convert("RGB")
        im.thumbnail((1024, 1024))
        buf = io.BytesIO()
        im.save(buf, "JPEG", quality=90)
        a = acc[f["source_dataset"]]
        a["lat"].append(r["latency_ms"])
        a["retry"].append(gap_ms(r) - r["latency_ms"] > 1000)
        a["mp"].append(im.size[0] * im.size[1] / 1e6)
        a["kb"].append(len(buf.getvalue()) / 1e3)
        a["t"].append(r["capture_ts"])
    for src, a in acc.items():
        n = len(a["lat"])
        print(f"{src}: n={n} median latency {st.median(a['lat']):.0f} ms, "
              f">4 s {sum(x > 4000 for x in a['lat']) / n:.2f}, retries {sum(a['retry']) / n:.3f}, "
              f"median {st.median(a['mp']):.2f} MP / {st.median(a['kb']):.0f} KB, "
              f"window {min(a['t'])[:16]} to {max(a['t'])[:16]}")


def schema_origin():
    for name, variant in (("phase_p.jsonl", "blind"), ("phase_p_v2.jsonl", "blind_v2")):
        rows = load(name, variant)
        bad = [r for r in rows if not r["schema_valid"]]
        bare = [r for r in bad if any(not re.search(r"==|!=|<|>", p) for p in r["preconditions"])]
        per_uc = defaultdict(list)
        for r in rows:
            per_uc[r["use_case"]].append(r["schema_valid"])
        print(f"{variant}: invalid {len(bad)}, with a bare-path precondition {len(bare)}, "
              f"parse_ok {sum(r['parse_ok'] for r in rows) / len(rows):.2f}, "
              f"action_valid {sum(r['action_valid'] for r in rows) / len(rows):.2f}")
        print("  validity per use case:", {k: round(sum(v) / len(v), 2) for k, v in sorted(per_uc.items())})


if __name__ == "__main__":
    latency_by_source()
    schema_origin()
