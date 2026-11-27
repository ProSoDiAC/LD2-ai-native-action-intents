#!/usr/bin/env python3
"""Figure: break-even weight lambda* of full XAIR against revoke-all, per producer.

Reads paper2_review_analyses.json, paper2_constrained_newer.json, and paper2_audit_extras.json.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.paper2_common import RESULTS_DIR  # noqa: E402
from experiments.plot_paper2_figures import OKABE, apply_style, save  # noqa: E402


def main() -> int:
    apply_style()
    rev = json.loads((RESULTS_DIR / "paper2_review_analyses.json").read_text())["break_even_reference"]
    new = json.loads((RESULTS_DIR / "paper2_constrained_newer.json").read_text())
    aud = json.loads((RESULTS_DIR / "paper2_audit_extras.json").read_text())
    vlm, det = OKABE["blue"], OKABE["vermillion"]
    rows = [  # label, break-even record, color
        ("Gemma 3 12B", rev["gemma3:12b"]["xair"], vlm),
        ("Qwen3-VL 8B", new["qwen3vl_8b"]["own_ages"]["break_even"], vlm),
        ("Llama 3.2-Vision 11B", rev["llama3.2-vision:11b"]["xair"], vlm),
        ("Qwen2.5-VL 3B", rev["qwen2.5vl:3b"]["xair"], vlm),
        ("Qwen2.5-VL 7B, constrained", new["qwen7b_constrained"]["no_defect_preconditions_original_ages"]["break_even"], vlm),
        ("Qwen2.5-VL 7B", rev["qwen2.5vl:7b"]["xair"], vlm),
        ("Qwen3-VL 8B, declared predicate", aud["qwen3_harness"]["own_ages"]["break_even"], vlm),
        ("Detector, 1.5 s evidence age", rev["detector_age1500ms"]["xair"], det),
        ("Detector, own latency", rev["detector_own_latency"]["xair"], det),
    ]
    rows.sort(key=lambda r: r[1]["lambda_star"])
    fig, ax = plt.subplots(figsize=(6.3, 2.6))
    for y, (label, rec, color) in enumerate(rows):
        x, (lo, hi) = rec["lambda_star"], rec["ci95"]
        ax.plot([lo, hi], [y, y], color=color, linewidth=2.2, solid_capstyle="butt")
        ax.plot([x], [y], marker="o", markersize=6, color=color, markeredgecolor="white", zorder=3)
        ax.text(hi * 1.06, y, f"{x:.2f}", va="center", ha="left", fontsize=8, color="#333333")
    ax.axvline(5, color="#52514e", linewidth=1.1, linestyle="--")
    ax.text(5 * 1.03, len(rows) - 0.4, r"default $\lambda{=}5$", fontsize=8, color="#52514e", va="center")
    ax.set_yticks(range(len(rows)), [r[0] for r in rows], fontsize=8.5)
    ax.set_xscale("log")
    ax.set_xticks([0.5, 1, 2, 5, 10], ["0.5", "1", "2", "5", "10"])
    ax.set_xlim(0.45, 14)
    ax.set_ylim(-0.6, len(rows) - 0.1)
    ax.set_xlabel(r"Break-even weight $\lambda^{*}$ (release pays for $\lambda<\lambda^{*}$)", fontsize=9)
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    save(fig, ROOT.parent / "journal" / "figures", "break_even")
    print("wrote break_even.pdf")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
