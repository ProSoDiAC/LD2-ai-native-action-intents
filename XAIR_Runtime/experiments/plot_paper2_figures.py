#!/usr/bin/env python3
"""Generate the Paper 2 manuscript figures (architecture, anchoring, dataset, B2–B5)."""

from __future__ import annotations

import argparse
import ast
import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "experiments" / "results"
# The manuscript lives in the (gitignored) top-level journal/ folder; fall back to
# results/figures when it is absent, e.g. in a public checkout.
PAPER_FIG = (
    ROOT.parent / "journal" / "figures"
    if (ROOT.parent / "journal").is_dir()
    else ROOT / "experiments" / "results" / "figures"
)

GATE_ORDER = ("direct", "freshness_only", "xair")
GATE_LABEL = {
    "direct": "Direct",
    "freshness_only": "Freshness-only",
    "xair": "Full XAIR",
}
# Okabe-Ito palette (colour-blind safe), with distinct markers/line styles as well.
OKABE = {
    "blue": "#0072B2", "orange": "#E69F00", "green": "#009E73", "vermillion": "#D55E00",
    "sky": "#56B4E9", "purple": "#CC79A7", "yellow": "#F0E442", "black": "#000000",
}
GATE_COLOR = {
    "direct": OKABE["vermillion"],
    "freshness_only": OKABE["orange"],
    "xair": OKABE["blue"],
}
GATE_MARKER = {"direct": "o", "freshness_only": "s", "xair": "^"}
GATE_LINESTYLE = {"direct": ":", "freshness_only": "--", "xair": "-"}
HEADLINE = dict(freshness_ms=4000, p_drift=0.5, drift_offset_ms=250, anchor="capture")


def apply_style() -> None:
    plt.rcParams.update(
        {
            "font.size": 11,
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "legend.fontsize": 9,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.28,
            "grid.linestyle": "--",
            # TrueType (Type 42) fonts: IEEE PDF eXpress rejects Type 3.
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def save(fig, dest: Path, stem: str) -> None:
    # IEEE style: titles belong in the caption, not in the image.
    if fig._suptitle is not None:
        fig._suptitle.set_text("")
    dest.mkdir(parents=True, exist_ok=True)
    fig.savefig(dest / f"{stem}.pdf", bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)


def parse_ci(raw: str) -> tuple[float, float]:
    try:
        lo, hi = ast.literal_eval(raw)
        return float(lo), float(hi)
    except (ValueError, SyntaxError, TypeError):
        return 0.0, 1.0


def load_b2_grid(path: Path) -> list[dict]:
    rows = []
    with path.open() as fh:
        for row in csv.DictReader(fh):
            row["freshness_ms"] = int(row["freshness_ms"])
            row["p_drift"] = float(row["p_drift"])
            row["drift_offset_ms"] = int(float(row["drift_offset_ms"]))
            row["SAR"] = float(row["SAR"])
            row["SER"] = float(row["SER"])
            row["WRR"] = float(row["WRR"])
            row["utility"] = float(row["utility"])
            row["hazardous_publish_rate"] = float(row["hazardous_publish_rate"])
            row["SAR_ci95"] = parse_ci(row["SAR_ci95"])
            row["SER_ci95"] = parse_ci(row["SER_ci95"])
            rows.append(row)
    return rows


def filter_rows(rows: list[dict], **kwargs) -> list[dict]:
    out = rows
    for key, val in kwargs.items():
        out = [r for r in out if r.get(key) == val]
    return out


def plot_validity_frontier(rows: list[dict], dest: Path) -> None:
    """SAR, SER, and utility vs freshness window at the headline volatility setting."""
    subset = filter_rows(
        rows,
        anchor="capture",
        p_drift=HEADLINE["p_drift"],
        drift_offset_ms=HEADLINE["drift_offset_ms"],
    )

    fig, axes = plt.subplots(1, 3, figsize=(6.3, 2.6))
    metrics = [
        ("SAR", "SAR", "SAR_ci95"),
        ("hazardous_publish_rate", "Adverse", "hazardous_publish_ci95"),
        ("utility", r"Utility $U$ ($\lambda{=}5$, $\mu{=}1$)", None),
    ]
    for panel, ax, (field, ylabel, ci_field) in zip("abc", axes, metrics):
        ax.set_title(f"({panel})", loc="left", fontsize=10)
        for gate in GATE_ORDER:
            pts = sorted(
                [r for r in subset if r["gate"] == gate],
                key=lambda r: r["freshness_ms"],
            )
            if not pts:
                continue
            xs = [r["freshness_ms"] for r in pts]
            ys = [r[field] for r in pts]
            ax.plot(
                xs,
                ys,
                marker=GATE_MARKER[gate],
                linestyle=GATE_LINESTYLE[gate],
                linewidth=2,
                label=GATE_LABEL[gate],
                color=GATE_COLOR[gate],
            )
        if field == "utility":
            # Revoke-all forgoes every context-valid correct intent: U = -mu * SAR(direct), constant in w.
            direct = sorted([r for r in subset if r["gate"] == "direct"], key=lambda r: r["freshness_ms"])
            ax.plot([r["freshness_ms"] for r in direct], [-r["SAR"] for r in direct], color="#555555",
                    linestyle="-.", linewidth=1.6, marker="v", markersize=4, label="Revoke-all")
        ax.set_xscale("log")
        ax.set_xlabel("Window $w$ (ms)", fontsize=9)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.set_xticks([100, 500, 2000, 8000], ["100", "500", "2k", "8k"])
        ax.tick_params(axis="x", labelsize=8)
    handles, labels = axes[2].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 0.0))
    fig.suptitle(
        (
            "Validity frontier (capture anchor, "
            f"$p_{{\\mathrm{{drift}}}}={HEADLINE['p_drift']}$, "
            f"$\\tau={HEADLINE['drift_offset_ms']}\\,\\mathrm{{ms}}$)"
        ),
        fontsize=12,
        y=1.02,
    )
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    save(fig, dest, "b2_validity_frontier")


def plot_anchoring_timeline(dest: Path) -> None:
    """Schematic: capture vs emission anchoring on a latency timeline."""
    fig, axes = plt.subplots(2, 1, figsize=(4.4, 2.7), sharex=True)
    t_capture, t_emit, t_submit = 0.0, 3.2, 3.5
    t_drift = 1.0
    freshness = 2.0

    for ax, anchor, title in zip(
        axes,
        ("capture", "emission"),
        (
            "(a) capture",
            "(b) emission",
        ),
    ):
        ax.set_ylim(0, 1.6)
        ax.axvline(t_capture, color="#2c3e50", linewidth=2.2, label="Evidence $t_e$")
        ax.axvline(t_emit, color="#8e44ad", linewidth=2.0, linestyle="--", label="Emission $t_d$")
        ax.axvline(t_drift, color=OKABE["vermillion"], linewidth=1.8, linestyle=":", label="Drift at $t_e{+}\\tau$")
        start = t_capture if anchor == "capture" else t_emit
        ax.axvspan(
            start,
            start + freshness,
            color=OKABE["green"],
            alpha=0.25,
            label="Freshness window $w$",
            zorder=0,
        )
        ax.axvline(t_submit, color="#2980b9", linewidth=2.2, label="Validation $t_v$")
        ax.annotate(
            "",
            xy=(t_emit, 0.35),
            xytext=(t_capture, 0.35),
            arrowprops=dict(arrowstyle="<->", color="#566573", lw=1.2),
        )
        ax.text((t_capture + t_emit) / 2, 0.42, r"$\Delta_{\mathrm{inf}}$", ha="center", fontsize=7.5, color="#566573")
        elapsed = (t_submit - start) if anchor == "capture" else max(0.0, t_submit - t_emit)
        stale = elapsed > freshness
        ax.text(
            min(t_submit + 0.15, 4.6),
            0.4,
            f"elapsed $= {elapsed:.1f}\\,\\mathrm{{s}}$"
            + (" (aged)" if stale else " (fresh)"),
            fontsize=7.5,
            color="#1a5276",
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="#2980b9", linewidth=1.0),
        )
        ax.set_yticks([])
        ax.text(6.15, 1.33, title, fontsize=7.5, ha="right", va="center", fontweight="bold")
        ax.set_xlim(-0.2, 6.2)
    axes[1].set_xlabel("Time since evidence capture (s)", fontsize=8)
    for ax in axes:
        ax.tick_params(labelsize=7.5)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, fontsize=6.8, frameon=False,
               bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(rect=(0, 0, 1, 0.80))
    save(fig, dest, "fig_anchoring_timeline")


def _box(ax, x, y, w, h, text, *, facecolor, edgecolor, fontsize=9, fontweight="normal"):
    from matplotlib.patches import FancyBboxPatch

    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.02,rounding_size=0.08",
        facecolor=facecolor,
        edgecolor=edgecolor,
        linewidth=1.8,
        zorder=2,
    )
    ax.add_patch(patch)
    ax.text(
        x + w / 2,
        y + h / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        fontweight=fontweight,
        zorder=3,
        multialignment="center",
        color="#1c2833",
    )
    return patch


def _arrow(ax, start, end, *, color="#2c3e50", style="-|>", lw=1.5, text="", text_offset=(0, 0.12)):
    ax.annotate(
        "",
        xy=end,
        xytext=start,
        arrowprops=dict(arrowstyle=style, color=color, lw=lw, shrinkA=4, shrinkB=4),
        zorder=1,
    )
    if text:
        mx = (start[0] + end[0]) / 2 + text_offset[0]
        my = (start[1] + end[1]) / 2 + text_offset[1]
        ax.text(
            mx,
            my,
            text,
            ha="center",
            va="bottom",
            fontsize=8.5,
            color=color,
            fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.15", facecolor="white", edgecolor="none", alpha=0.9),
        )


def plot_architecture_diagram(dest: Path) -> None:
    """End-to-end architecture: proposal, contract, publication gate, outcomes, plant context."""
    from matplotlib.patches import FancyBboxPatch

    ink, muted = "#1f2933", "#52606d"
    pal = {  # face, edge
        "input": ("#f5f7fa", "#7b8794"),
        "model": ("#e8f4fc", "#56B4E9"),
        "contract": ("#fdf3dc", "#E69F00"),
        "gate": ("#dcebf6", "#0072B2"),
        "budget": ("#f6e9f0", "#CC79A7"),
        "publish": ("#dcebf6", "#0072B2"),
        "revoke": ("#fbe9dc", "#D55E00"),
        "plant": ("#f5f7fa", "#3e4c59"),
    }

    fig, ax = plt.subplots(figsize=(6.3, 3.0))
    ax.set_xlim(0, 15.5)
    ax.set_ylim(-0.05, 6.55)
    ax.axis("off")

    def box(x, y, w, h, title, body="", kind="input", dashed=False, tfs=8.2):
        face, edge = pal[kind]
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.12",
                                    facecolor=face, edgecolor=edge, linewidth=1.1,
                                    linestyle="--" if dashed else "-", zorder=2))
        cy = y + h / 2
        n = body.count("\n") + 1 if body else 0
        if body:
            ax.text(x + w / 2, cy + 0.11 * n + 0.07, title, ha="center", va="center", fontsize=tfs,
                    fontweight="bold", color=ink, zorder=3)
            ax.text(x + w / 2, cy - 0.16, body, ha="center", va="center",
                    fontsize=6.6, color=muted, zorder=3, linespacing=1.3)
        else:
            ax.text(x + w / 2, cy, title, ha="center", va="center", fontsize=8.2,
                    fontweight="bold", color=ink, zorder=3)

    def path(points, color=muted, label="", at=None, lw=1.1, ha="center"):
        xs, ys = zip(*points)
        ax.plot(xs, ys, color=color, lw=lw, zorder=1, solid_capstyle="butt", solid_joinstyle="miter")
        ax.annotate("", xy=points[-1], xytext=points[-2], zorder=1,
                    arrowprops=dict(arrowstyle="-|>,head_length=0.35,head_width=0.18", color=color,
                                    lw=lw, shrinkA=0, shrinkB=0))
        if label:
            ax.text(at[0], at[1], label, ha=ha, va="center", fontsize=7.4, color=color, zorder=4)

    # Offline-replay region: the perception cache stands in for live inference in the evaluation.
    ax.add_patch(FancyBboxPatch((2.25, 3.45), 5.9, 2.85, boxstyle="round,pad=0.02,rounding_size=0.15",
                                facecolor="none", edgecolor="#9aa5b1", linewidth=0.9,
                                linestyle=(0, (4, 3)), zorder=0))
    ax.text(2.4, 6.05, "offline: perception-cache replay",
            ha="left", va="center", fontsize=7.8, color="#7b8794", style="italic")

    # Proposal row
    box(0.05, 4.55, 2.0, 1.1, "Camera", "frame at $t_e$", "input")
    box(2.4, 4.2, 2.6, 1.6, "Proposer", "VLM or agent,\nblind prompt", "model")
    box(5.4, 4.2, 2.6, 1.6, "AIS intent", "action, preconditions,\nevidence, anchor", "contract")
    box(8.4, 3.95, 2.6, 2.1, "XAIR gate", "validate $t_v$,\nrecheck $t_g$", "gate")
    box(11.4, 5.0, 1.55, 0.95, "Publish", "", "publish")
    box(13.2, 4.85, 2.25, 1.25, "Middleware", "publish at $t_m$", "plant")
    box(11.4, 3.6, 1.55, 0.95, "Revoke", "", "revoke")
    box(13.2, 3.6, 2.25, 0.95, "Audit log", "", "input")

    path([(2.05, 5.1), (2.4, 5.1)])
    path([(5.0, 5.0), (5.4, 5.0)])
    path([(8.0, 5.0), (8.4, 5.0)])
    path([(11.0, 5.47), (11.4, 5.47)], color=pal["publish"][1])
    path([(12.95, 5.47), (13.2, 5.47)], color=pal["publish"][1])
    path([(11.0, 4.07), (11.4, 4.07)], color=pal["revoke"][1])
    path([(12.95, 4.07), (13.2, 4.07)], color=pal["revoke"][1])

    # Context row
    box(0.05, 1.65, 2.0, 1.1, "MES/OPC UA", "plant state", "input", tfs=7.2)
    box(5.4, 1.75, 2.6, 0.95, "Validity budget", "window $w$, strictness", "budget")
    box(9.4, 1.55, 2.6, 1.15, "Plant context", "snapshot $C(t)$", "plant")
    box(2.4, 0.0, 2.6, 0.95, "Drift", "drift $p$, $\\tau$, rate $r$", "revoke")

    path([(2.05, 2.45), (2.18, 2.45), (2.18, 3.65), (4.65, 3.65), (4.65, 4.2)], label="prompt context",
         at=(2.3, 3.88), ha="left")
    path([(2.05, 1.95), (2.18, 1.95), (2.18, 1.2), (10.1, 1.2), (10.1, 1.55)], label="state updates",
         at=(6.2, 1.38))
    path([(5.0, 0.48), (11.3, 0.48), (11.3, 1.55)], color=pal["revoke"][1], label="context patch",
         at=(8.1, 0.66))
    path([(6.7, 4.2), (6.7, 2.7)], color=pal["budget"][1], label="features", at=(6.85, 3.2), ha="left")
    path([(8.0, 2.22), (8.95, 2.22), (8.95, 3.95)], color=pal["budget"][1],
         label="$w$", at=(9.1, 3.2), ha="left")
    path([(10.45, 2.7), (10.45, 3.95)], color=pal["plant"][1], label="read $C$, $\\nu_R$", at=(10.6, 3.2),
         ha="left")

    fig.tight_layout(pad=0.2)
    save(fig, dest, "fig_architecture")


def plot_b4_utility_vs_volatility(table_csv: Path, dest: Path) -> None:
    if not table_csv.is_file():
        return
    keep = {
        "oracle": ("Oracle", OKABE["black"], "-", "o"),
        "static:llama3.2-vision:11b": ("Always Llama 3.2-Vision", OKABE["purple"], "-", "D"),
        "cap:8000ms": ("Latency cap 8 s", OKABE["green"], ":", "v"),
        "static:qwen2.5vl:7b": ("Always Qwen 7B", OKABE["blue"], "-", "s"),
        "static:qwen2.5vl:32b": ("Always Qwen 32B", OKABE["orange"], "-.", "^"),
        "cascade:vol": ("Cascade", OKABE["black"], (0, (1, 1)), "P"),
        "static:qwen2.5vl:3b": ("Always Qwen 3B", OKABE["sky"], "--", "X"),
    }
    order = ["p0", "p25", "p50", "p75"]
    xs = [0.0, 0.25, 0.5, 0.75]
    series: dict[str, list[float]] = {k: [] for k in keep}
    with table_csv.open() as fh:
        rows = list(csv.DictReader(fh))
    for setting in order:
        for router in keep:
            match = [r for r in rows if r["router"] == router and r["setting"] == setting]
            series[router].append(float(match[0]["utility"]) if match else float("nan"))
    fig, ax = plt.subplots(figsize=(6.6, 3.9))
    for router, (label, color, ls, mk) in keep.items():
        big = router == "cascade:vol"
        ax.plot(xs, series[router], marker=mk, linewidth=2.4 if big else 1.8, markersize=9 if big else 6,
                label=label, color=color, linestyle=ls, zorder=5 if big else 2,
                markerfacecolor="white" if big else color)
    ax.set_xlabel(r"Plant volatility $p_{\mathrm{drift}}$")
    ax.set_ylabel("Utility $U$")
    ax.set_xticks(xs)
    ax.legend(fontsize=8.5, ncol=2, frameon=True)
    fig.tight_layout()
    # Documentation figure (docs/SECONDARY_RESULTS.md), not a manuscript figure.
    dest.mkdir(parents=True, exist_ok=True)
    fig.savefig(dest / "b4_utility_vs_volatility.png", dpi=150, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)

def plot_dataset_collage(dest: Path) -> None:
    """Nominal vs defective examples from VisA (CC BY 4.0)."""
    try:
        from PIL import Image
    except ImportError:
        print("[paper2-fig] skip dataset collage: Pillow missing")
        return
    root = ROOT / "experiments" / "datasets" / "manufacturing-a1"
    manifest = root / "manifest.jsonl"
    if not manifest.is_file() or not (root / "frames").is_dir():
        print("[paper2-fig] skip dataset collage: image frames not downloaded")
        return
    rows = [json.loads(l) for l in manifest.read_text().splitlines() if l.strip()]
    by: dict[tuple, dict[str, list]] = defaultdict(lambda: {"ok": [], "def": []})
    for r in rows:
        key = (r.get("source_dataset"), r.get("category"))
        bucket = "def" if r.get("defect_present") else "ok"
        by[key][bucket].append(r)
    # VisA only (CC BY 4.0): MVTec AD images are CC BY-NC-SA and are not reproduced in figures.
    prefer = [
        ("visa", "pcb1"),
        ("visa", "cashew"),
        ("visa", "capsules"),
        ("visa", "fryum"),
    ]
    picked = [k for k in prefer if by[k]["ok"] and by[k]["def"]][:4]
    if len(picked) < 4:
        return
    # Sized to the text width (6.3 in), like every other figure in the paper.
    fig, axes = plt.subplots(2, 4, figsize=(6.3, 3.5))
    for col, key in enumerate(picked):
        src, cat = key
        for row_i, kind in enumerate(("ok", "def")):
            ax = axes[row_i][col]
            cand = sorted(by[key][kind], key=lambda r: r["frame_id"])
            rec = cand[len(cand) // 3]
            img = Image.open(root / rec["path"]).convert("RGB")
            w, h = img.size
            side = min(w, h)
            left, top = (w - side) // 2, (h - side) // 2
            img = img.crop((left, top, left + side, top + side)).resize(
                (256, 256), Image.Resampling.LANCZOS
            )
            ax.imshow(np.asarray(img))
            ax.set_xticks([])
            ax.set_yticks([])
            ax.grid(False)
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_color(OKABE["vermillion"] if kind == "def" else OKABE["blue"])
                spine.set_linewidth(2.0)
            src_lab = "MVTec AD" if src == "mvtec_ad" else "VisA"
            if row_i == 0:
                ax.set_title(f"{src_lab}\n{cat.replace('_', ' ')}", fontsize=14, pad=3)
            if col == 0:
                ax.set_ylabel(
                    "Nominal" if kind == "ok" else "Defective",
                    fontsize=14,
                    fontweight="bold",
                )
            if kind == "def":
                sev = rec.get("severity") or "minor"
                ax.text(
                    0.03,
                    0.97,
                    sev,
                    transform=ax.transAxes,
                    va="top",
                    ha="left",
                    fontsize=13,
                    color="white",
                    bbox=dict(
                        boxstyle="round,pad=0.2",
                        facecolor=OKABE["vermillion"] if sev == "major" else OKABE["orange"],
                        edgecolor="none",
                        alpha=0.85,
                    ),
                )
    fig.tight_layout(pad=0.3)
    save(fig, dest, "dataset_collage")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=PAPER_FIG)
    parser.add_argument("--b2-csv", type=Path, default=RESULTS / "b2_validity_frontier.csv")
    args = parser.parse_args()
    apply_style()

    if args.b2_csv.is_file():
        rows = load_b2_grid(args.b2_csv)
        plot_validity_frontier(rows, args.out)
        print(f"[paper2-fig] B2 plots from {len(rows)} grid cells -> {args.out}")
    else:
        print(f"[paper2-fig] skip B2: {args.b2_csv} missing")

    plot_dataset_collage(args.out)
    plot_anchoring_timeline(args.out)
    plot_architecture_diagram(args.out)
    plot_b4_utility_vs_volatility(RESULTS / "b4_routing_table.csv",
                                  Path(__file__).resolve().parents[2] / "docs" / "figures")
    print(f"[paper2-fig] schematic + B1--B5 extras -> {args.out}")


if __name__ == "__main__":
    main()
