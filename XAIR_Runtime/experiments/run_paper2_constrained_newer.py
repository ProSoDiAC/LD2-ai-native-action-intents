#!/usr/bin/env python3
"""Grammar-constrained decoding (7B, blind prompt) and a newer VLM (Qwen3-VL 8B), on the 612 test
frames, replayed through the B2 reference cell. Output: paper2_constrained_newer.json.

Both arms ran after the main campaign on a shared accelerator, so besides their own evidence ages
they are replayed at the primary 7B model's original ages, so that only the decision changes.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.paper2_common import RESULTS_DIR, load_manifest  # noqa: E402
from experiments.paper2_splits import load_split  # noqa: E402
from experiments.run_b2_validity_frontier import mcnemar_exact_p  # noqa: E402
from experiments.run_paper2_review_analyses import (  # noqa: E402
    break_even,
    contrast,
    load_raw,
    records_from,
    ref_trials,
    revoke_all,
    summary,
)

ARMS = {
    "qwen7b_blind": ("phase_p", "qwen2.5vl:7b", "blind"),
    "qwen7b_constrained": ("phase_p_constrained", "qwen2.5vl:7b", "blind_constrained"),
    "qwen3vl_8b": ("phase_p_qwen3", "qwen3-vl:8b-instruct", "blind"),
    "qwen7b_ref_constrained": ("phase_p_ref_constrained", "qwen2.5vl:7b", "blind_ref_constrained"),
}
HARNESS = "line.state == 'RUN'"


def evaluate(recs) -> dict:
    ts = {g: ref_trials(recs, g) for g in ("freshness_only", "schema_freshness", "xair")}
    ra = revoke_all(ts["xair"])
    return {
        "freshness_only": summary(ts["freshness_only"]),
        "xair": summary(ts["xair"]),
        "xair_minus_freshness_only": contrast(ts["xair"], ts["freshness_only"]),
        "xair_minus_schema_freshness": contrast(ts["xair"], ts["schema_freshness"]),
        "xair_minus_revoke_all": contrast(ts["xair"], ra),
        "break_even": break_even(ts["xair"]),
    }


def main() -> int:
    split = load_split()
    episodes = {e["frame_id"]: e for e in load_manifest()}
    raws = {k: [r for r in load_raw(tag, {m}, {v}) if split.get(r["frame_id"]) == "test"]
            for k, (tag, m, v) in ARMS.items() if (RESULTS_DIR / "perception_cache" / f"{tag}.jsonl").is_file()}
    base_age = {r.frame_id: r.latency_ms for r in records_from(raws["qwen7b_blind"], episodes, split)}
    base_ok = {r["frame_id"]: r["action"] == r["gt_action"] for r in raws["qwen7b_blind"]}
    out = {}
    for k, raw in raws.items():
        recs = records_from(raw, episodes, split)
        ok = {r["frame_id"]: r["action"] == r["gt_action"] for r in raw}
        shared = sorted(set(ok) & set(base_ok))
        b = sum(1 for f in shared if ok[f] and not base_ok[f])
        c = sum(1 for f in shared if base_ok[f] and not ok[f])
        lat = sorted(r.latency_ms for r in recs)
        res = {
            "n": len(raw),
            "accuracy": sum(ok.values()) / len(ok),
            "schema_valid": sum(bool(r.get("schema_valid")) for r in raw) / len(raw),
            "median_age_ms": lat[len(lat) // 2],
            "share_age_over_4s": sum(x > 4000 for x in lat) / len(lat),
            "vs_7b_blind": {"arm_only": b, "7b_only": c, "mcnemar_exact_p": mcnemar_exact_p(b, c)},
            "single_harness_predicate_share": sum(1 for r in raw if (r.get("preconditions") or []) == [HARNESS]) / len(raw),
            "own_ages": evaluate(recs),
        }
        matched = [r for r in recs if r.frame_id in base_age]
        for r in matched:
            r.latency_ms = base_age[r.frame_id]
        res["original_7b_ages"] = evaluate(matched)

        def harness(ep, rec):
            rec["preconditions"] = [HARNESS]

        hrec = [r for r in records_from(raw, episodes, split, rec_fn=harness) if r.frame_id in base_age]
        for r in hrec:
            r.latency_ms = base_age[r.frame_id]
        ht = ref_trials(hrec, "xair")
        et = ref_trials(matched, "xair")
        res["emitted_minus_harness_original_ages"] = contrast(et, ht)

        # Review 2026-11: a precondition on defect.absent is checked against a ground-truth flag the
        # model never sees, so repeat the comparison with those preconditions dropped.
        def nodefect(ep, rec):
            rec["preconditions"] = [p for p in (rec.get("preconditions") or []) if "defect" not in str(p)]

        nrec = [r for r in records_from(raw, episodes, split, rec_fn=nodefect) if r.frame_id in base_age]
        for r in nrec:
            r.latency_ms = base_age[r.frame_id]
        nt = ref_trials(nrec, "xair")
        res["no_defect_preconditions_original_ages"] = {
            "share_with_defect_precondition": sum(
                1 for r in raw if any("defect" in str(p) for p in (r.get("preconditions") or []))) / len(raw),
            "xair": summary(nt),
            "xair_minus_revoke_all": contrast(nt, revoke_all(nt)),
            "break_even": break_even(nt),
            "minus_harness": contrast(nt, ht),
            "xair_minus_freshness_only": contrast(nt, ref_trials(nrec, "freshness_only")),
            "xair_minus_schema_freshness": contrast(nt, ref_trials(nrec, "schema_freshness")),
        }
        out[k] = res
        print(f"[constrained/newer] {k}: acc={res['accuracy']:.3f} schema={res['schema_valid']:.3f}", flush=True)
    path = RESULTS_DIR / "paper2_constrained_newer.json"
    path.write_text(json.dumps(out, indent=2, default=float) + "\n")
    print(f"[constrained/newer] wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
