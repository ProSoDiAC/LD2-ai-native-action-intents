#!/usr/bin/env python3
"""Figure: full-XAIR contrasts as a function of the Poisson drift rate (paper2_review_analyses.json)."""

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
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

PRODUCERS = {
    "qwen7b_blind": ("Qwen2.5-VL 7B (observed ages)", OKABE["blue"], "o", 0.007),
    "detector_age1500ms": ("Detector (1.5 s age)", OKABE["vermillion"], "s", 0.0092),
}
PANELS = (
    ("xair_minus_freshness_only", "(a) Full XAIR $-$ freshness-only"),
    ("xair_minus_revoke_all", "(b) Full XAIR $-$ revoke-all"),
)


def main() -> int:
    apply_style()
    d = json.loads((RESULTS_DIR / "paper2_review_analyses.json").read_text())["drift_curve"]["cells"]
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.9))
    for ax, (key, title) in zip(axes, PANELS):
        for prod, (label, color, marker, jx) in PRODUCERS.items():
            cells = d[prod]
            rs = sorted((float(k.split("_")[1]), v) for k, v in cells.items() if k.startswith("poisson_"))
            x = [r for r, _ in rs]
            y = [v[key]["delta"] for _, v in rs]
            lo = [v[key]["ci95"][0] for _, v in rs]
            hi = [v[key]["ci95"][1] for _, v in rs]
            ax.fill_between(x, lo, hi, color=color, alpha=0.18, linewidth=0)
            ax.plot(x, y, color=color, marker=marker, markersize=5, linewidth=2, label=label)
            js = cells["job_shop_operating"][key]
            ax.errorbar([jx], [js["delta"]], yerr=[[js["delta"] - js["ci95"][0]], [js["ci95"][1] - js["delta"]]],
                        color=color, marker=marker, markersize=6, markerfacecolor="white", linewidth=1.5,
                        capsize=3)
        ax.axhline(0, color="#52514e", linewidth=1.0)
        ax.axvline(0.013, color="#b5b4ae", linewidth=0.8, linestyle=":")
        ax.set_xscale("log")
        ax.set_xticks([0.008, 0.02, 0.05, 0.2, 1, 5])
        ax.set_xticklabels(["job\nshop", "0.02", "0.05", "0.2", "1", "5"])
        ax.set_xlabel(r"Drift rate $r$ (s$^{-1}$)")
        ax.set_ylabel(r"$\Delta U$ ($\lambda{=}5$)")
        ax.set_title(title, fontsize=10, loc="left")
    handles = [Line2D([], [], color=c, marker=m, linewidth=2, label=lab) for lab, c, m, _ in PRODUCERS.values()]
    handles += [Line2D([], [], color="#52514e", marker="o", markerfacecolor="white", linestyle="none",
                       label="job-shop log (operating)"),
                Patch(facecolor="#52514e", alpha=0.18, label="95% band")]
    fig.legend(handles=handles, loc="upper center", ncol=4, frameon=False, fontsize=8, bbox_to_anchor=(0.5, 1.07))
    fig.tight_layout()
    save(fig, ROOT.parent / "journal" / "figures", "drift_rate_curve")
    print("wrote drift_rate_curve.pdf")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
