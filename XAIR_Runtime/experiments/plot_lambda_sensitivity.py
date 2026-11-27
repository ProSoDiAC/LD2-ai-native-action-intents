#!/usr/bin/env python3
"""Figure: ordering of the publication paths as the adverse-publication weight lambda varies.

Utility is linear in lambda, U = success - lambda * adverse - mu * wrongful, so each path is a straight
line. Panel (a) is the balanced corpus, panel (b) the corpus reweighted to 10% defect prevalence,
both for the primary 7B model at the reference cell (exact per-frame expectations).

Reads paper2_expectation_prevalence.json.
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

PATHS = [  # key, label, color, linestyle
    ("direct", "Direct", OKABE["vermillion"], ":"),
    ("freshness_only", "Freshness-only", OKABE["orange"], "-."),
    ("schema_freshness", "Schema+freshness", OKABE["sky"], (0, (5, 2))),
    ("xair", "Full XAIR", OKABE["blue"], "-"),
    ("revoke_all", "Revoke-all", "#52514e", "--"),
]
LAMBDAS = [0.0, 10.0]


def utility(rec: dict, lam: float, mu: float) -> float:
    return rec["success"] - lam * rec["adverse"] - mu * rec["wrongful"]


def main() -> int:
    apply_style()
    data = json.loads((RESULTS_DIR / "paper2_expectation_prevalence.json").read_text())
    mu = float(data["mu"])
    prod = data["producers"]["qwen7b_blind"]
    panels = [
        ("(a) Balanced corpus", prod["balanced"]),
        ("(b) 10% defect prevalence", prod["prevalence"]["0.1"]),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(6.3, 2.6), sharey=True)
    for ax, (title, rates) in zip(axes, panels):
        for key, label, color, style in PATHS:
            ys = [utility(rates[key], lam, mu) for lam in LAMBDAS]
            ax.plot(LAMBDAS, ys, color=color, linestyle=style, linewidth=1.8, label=label)
        # lambda* of full XAIR against revoke-all: (1 + mu) * success / adverse
        x = rates["xair"]
        lam_star = (1 + mu) * x["success"] / x["adverse"]
        if lam_star <= LAMBDAS[1]:
            y_star = utility(x, lam_star, mu)
            ax.plot([lam_star], [y_star], marker="o", markersize=5.5, color=OKABE["blue"],
                    markeredgecolor="white", zorder=4)
            ax.annotate(rf"$\lambda^{{*}}{{=}}{lam_star:.2f}$", (lam_star, y_star),
                        textcoords="offset points", xytext=(6, -12), fontsize=8, color="#333333")
        ax.axvline(5, color="#52514e", linewidth=0.9, linestyle=":")
        ax.set_xlim(*LAMBDAS)
        ax.set_ylim(-6.0, 0.6)
        ax.set_xlabel(r"Weight $\lambda$ of an adverse release", fontsize=9)
        ax.set_title(title, fontsize=9)
    axes[0].set_ylabel("Utility $U$ ($\\mu{=}1$)", fontsize=9)
    axes[0].text(5.12, -5.8, r"default $\lambda{=}5$", fontsize=7.5, color="#52514e")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=5, fontsize=8, frameon=False,
               bbox_to_anchor=(0.5, -0.08))
    fig.tight_layout()
    save(fig, ROOT.parent / "journal" / "figures", "lambda_sensitivity")
    print("wrote lambda_sensitivity.pdf")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
