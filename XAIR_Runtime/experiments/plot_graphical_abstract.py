#!/usr/bin/env python3
"""Graphical abstract: problem -> release-time gate -> outcomes, plus one small gain-vs-drift chart.

Reads ``paper2_drift_timing.json`` (run_paper2_drift_timing.py); plots the utility gain of full XAIR
over freshness-only release (VLM producer) at the Poisson drift rates. Writes
``graphical_abstract.{png,pdf}`` to --out.

    python experiments/plot_graphical_abstract.py --out ../journal/submission
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.paper2_common import RESULTS_DIR  # noqa: E402

BLUE, VERM, ORANGE, SKY, INK, MUTED = "#0072B2", "#D55E00", "#E69F00", "#56B4E9", "#1f2933", "#52606d"
RATES = (("poisson_0.05", 0.05), ("poisson_0.2", 0.2), ("poisson_1", 1.0))
PRODUCER = "qwen7b_blind"


def box(ax, x, y, w, h, title, body, face, edge, tfs=19.5):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.0,rounding_size=0.12",
                                facecolor=face, edgecolor=edge, linewidth=2.2, zorder=2))
    ax.text(x + w / 2, y + h * (0.66 if body else 0.5), title, ha="center", va="center", fontsize=tfs,
            fontweight="bold", color=INK, zorder=3)
    if body:
        ax.text(x + w / 2, y + h * 0.30, body, ha="center", va="center", fontsize=14.9, color=MUTED,
                zorder=3, linespacing=1.3)


def arrow(ax, p0, p1, color=MUTED):
    ax.annotate("", xy=p1, xytext=p0, zorder=1,
                arrowprops=dict(arrowstyle="-|>,head_length=0.7,head_width=0.35", color=color, lw=2.4,
                                shrinkA=0, shrinkB=0))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=RESULTS_DIR / "figures")
    args = ap.parse_args()
    cells = json.loads((RESULTS_DIR / "paper2_drift_timing.json").read_text())["cells"][PRODUCER]
    plt.rcParams.update({"font.family": "DejaVu Sans", "pdf.fonttype": 42, "ps.fonttype": 42})

    fig = plt.figure(figsize=(10.0, 4.0))
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_xlim(0, 25)
    ax.set_ylim(0, 10)
    ax.axis("off")

    # 1 problem
    box(ax, 0.4, 3.2, 5.0, 3.6, "AI action", "from aged or\nstale evidence", "#e8f4fc", SKY)
    ax.text(2.9, 7.7, "Problem", ha="center", fontsize=17.2, color=MUTED, style="italic")
    # 2 gate
    box(ax, 7.4, 2.8, 5.4, 4.4, "Release-time\nvalidation", "is the evidence\nstill valid now?", "#dcebf6", BLUE)
    ax.text(10.1, 7.9, "Gate", ha="center", fontsize=17.2, color=MUTED, style="italic")
    # 3 outcomes
    box(ax, 14.3, 6.1, 3.6, 1.9, "Released:\nsuccessful", "", "#dcebf6", BLUE, tfs=13.8)
    box(ax, 14.3, 3.7, 3.6, 1.9, "Released:\nadverse", "", "#fff3cc", "#a07800", tfs=13.8)
    box(ax, 14.3, 1.3, 3.6, 1.9, "Revoked", "", "#fbe9dc", VERM, tfs=13.8)
    ax.text(16.1, 8.5, "Outcomes", ha="center", fontsize=17.2, color=MUTED, style="italic")
    arrow(ax, (5.5, 5.0), (7.3, 5.0))
    arrow(ax, (12.9, 6.4), (14.2, 7.0), BLUE)
    arrow(ax, (12.9, 5.0), (14.2, 4.65), "#a07800")
    arrow(ax, (12.9, 3.6), (14.2, 2.3), VERM)

    # small chart
    cx = fig.add_axes((0.815, 0.24, 0.16, 0.50))
    xs = [r for _, r in RATES]
    gain = [cells[s]["xair"]["U"] - cells[s]["freshness_only"]["U"] for s, _ in RATES]
    cx.plot(xs, gain, color=BLUE, marker="o", lw=2.6, ms=8)
    cx.set_xscale("log")
    cx.set_xticks(xs, [f"{r:g}" for r in xs])
    cx.minorticks_off()
    cx.set_ylim(0, max(gain) * 1.15)
    cx.set_xlabel("Drift rate (per s)", fontsize=12.6)
    cx.set_ylabel("Gain vs. freshness-only", fontsize=12.6)
    cx.tick_params(labelsize=11.5)
    cx.grid(color="#e5e5e5")
    cx.spines[["top", "right"]].set_visible(False)
    fig.text(0.865, 0.84, "Gain grows with drift", ha="center", fontsize=14.3, fontweight="bold", color=INK)
    args.out.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(args.out / f"graphical_abstract.{ext}", dpi=300)
    print(f"[ga] wrote {args.out}/graphical_abstract.{{png,pdf}}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
