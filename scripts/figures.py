#!/usr/bin/env python3
"""
Generate the README figures with matplotlib.

Produces four PNGs in docs/figures/:
  roofline.png   the roofline plot with attention plotted on it (data-driven)
  ab_flip.png    the thesis A/B: one e-graph, two winners
  pipeline.png   the five-stage pipeline
  fusion.png     naive vs fused HBM traffic

Run: python scripts/figures.py
The flops/HBM numbers come from the same model the Rust code uses, so the
roofline figure is real, not decorative.
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

OUT = Path(__file__).resolve().parent.parent / "docs" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

INK = "#1b1f24"
BLUE = "#2563eb"
RED = "#dc2626"
GREEN = "#059669"
GREY = "#6b7280"
LIGHT = "#eef2f7"


def _box(ax, xy, w, h, text, fc, ec, tc="white", fs=10):
    x, y = xy
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.06",
        linewidth=1.5, edgecolor=ec, facecolor=fc, zorder=2))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            color=tc, fontsize=fs, zorder=3, wrap=True)


def _arrow(ax, p0, p1, color=INK, style="-|>", ls="-", lw=1.6):
    ax.add_patch(FancyArrowPatch(
        p0, p1, arrowstyle=style, mutation_scale=14, linewidth=lw,
        linestyle=ls, color=color, zorder=1,
        shrinkA=2, shrinkB=2))


# ── 1. roofline (data-driven) ────────────────────────────────────────────────

def roofline():
    # H100-ish: 989 TFLOP/s over 3.35 TB/s -> ridge ~295 flop/byte.
    peak_flops = 989e12
    peak_bw = 3.35e12
    ridge = peak_flops / peak_bw

    fig, ax = plt.subplots(figsize=(7.2, 4.4), dpi=150)
    I = np.logspace(0, 3.2, 400)               # arithmetic intensity, flop/byte
    achievable = np.minimum(peak_flops, peak_bw * I) / 1e12  # TFLOP/s
    ax.plot(I, achievable, color=INK, lw=2.2, zorder=2)
    ax.axvline(ridge, color=GREY, ls="--", lw=1.2, zorder=1)
    ax.text(ridge * 1.05, 30, f"ridge\n{ridge:.0f} flop/byte", color=GREY, fontsize=9)

    # naive attention sits at ~9-11 flop/byte: deep in the HBM-bound region.
    for inten, label in [(9.0, "attention\n(naive)")]:
        perf = min(peak_flops, peak_bw * inten) / 1e12
        ax.scatter([inten], [perf], color=RED, s=70, zorder=4)
        ax.annotate(label, (inten, perf), textcoords="offset points",
                    xytext=(10, -28), color=RED, fontsize=9)

    # fusion lifts intensity off the floor (fewer bytes for the same flops).
    ax.scatter([60], [min(peak_flops, peak_bw * 60) / 1e12], color=GREEN, s=70, zorder=4)
    ax.annotate("fused\n(less HBM)", (60, min(peak_flops, peak_bw * 60) / 1e12),
                textcoords="offset points", xytext=(8, 8), color=GREEN, fontsize=9)

    ax.fill_betweenx([0, peak_flops / 1e12], 1, ridge, color=RED, alpha=0.05)
    ax.fill_betweenx([0, peak_flops / 1e12], ridge, 1600, color=BLUE, alpha=0.05)
    ax.text(3, 700, "HBM-bound", color=RED, fontsize=10)
    ax.text(360, 200, "compute-bound", color=BLUE, fontsize=10)

    ax.set_xscale("log")
    ax.set_xlim(1, 1600)
    ax.set_ylim(0, peak_flops / 1e12 * 1.08)
    ax.set_xlabel("arithmetic intensity  (flop / byte)")
    ax.set_ylabel("achievable performance  (TFLOP/s)")
    ax.set_title("the roofline: attention is bandwidth-bound", color=INK)
    ax.grid(True, which="both", alpha=0.15)
    fig.tight_layout()
    fig.savefig(OUT / "roofline.png", bbox_inches="tight")
    plt.close(fig)


# ── 2. the A/B flip ──────────────────────────────────────────────────────────

def ab_flip():
    fig, ax = plt.subplots(figsize=(7.6, 3.6), dpi=150)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 5)
    ax.axis("off")

    _box(ax, (3.4, 3.6), 3.2, 1.0,
         "one saturated e-graph\nnaive AND fused attention", LIGHT, INK, tc=INK)
    _box(ax, (0.3, 0.5), 3.4, 1.2,
         "extract: naive\nwrites s×s scores to HBM", RED, RED)
    _box(ax, (6.3, 0.5), 3.4, 1.2,
         "extract: fused (flash)\ns×s stays in SRAM", GREEN, GREEN)

    _arrow(ax, (4.4, 3.6), (2.0, 1.7), color=RED)
    ax.text(2.2, 2.8, "constraints =\nFlops only", color=RED, fontsize=9, ha="center")
    _arrow(ax, (5.6, 3.6), (8.0, 1.7), color=GREEN)
    ax.text(7.8, 2.8, "constraints =\nFlops + HbmBytes", color=GREEN, fontsize=9, ha="center")
    _arrow(ax, (3.7, 1.1), (6.3, 1.1), color=GREY, style="-|>", ls="--")
    ax.text(5.0, 1.35, "add the constraint you were ignoring", color=GREY,
            fontsize=8.5, ha="center")

    ax.set_title("same search, same e-graph, one constraint flips the winner", color=INK)
    fig.tight_layout()
    fig.savefig(OUT / "ab_flip.png", bbox_inches="tight")
    plt.close(fig)


# ── 3. the pipeline ──────────────────────────────────────────────────────────

def pipeline():
    fig, ax = plt.subplots(figsize=(9.2, 2.6), dpi=150)
    ax.set_xlim(0, 23)
    ax.set_ylim(0, 3)
    ax.axis("off")

    stages = [
        ("program\n(JAX / rl-ir)", GREY),
        ("rl-ir\nIR + reference + accountant", BLUE),
        ("rl-opt\negg e-graph of\nequivalent forms", BLUE),
        ("rl-cost\nroofline:\nslowest resource wins", INK),
        ("rl-codegen\nkernel (stub)", GREY),
        ("rl-ledger\nreplayable result", GREY),
    ]
    w, gap, y, h = 3.3, 0.45, 1.0, 1.2
    x = 0.2
    centers = []
    for text, c in stages:
        _box(ax, (x, y), w, h, text, c, c, fs=8.5)
        centers.append((x, x + w))
        x += w + gap
    for (l, r), (nl, _) in zip(centers, centers[1:]):
        _arrow(ax, (r, y + h / 2), (nl, y + h / 2))

    ax.set_title("the pipeline: a program becomes an e-graph, the cost model picks the cheapest plan",
                 color=INK, fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT / "pipeline.png", bbox_inches="tight")
    plt.close(fig)


# ── 4. fusion saves HBM ──────────────────────────────────────────────────────

def fusion():
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.2), dpi=150)

    # naive: arrows dip down to an HBM bar between each stage.
    axn = axes[0]
    axn.set_xlim(0, 10); axn.set_ylim(0, 5); axn.axis("off")
    axn.set_title("naive: every intermediate hits HBM", color=RED, fontsize=10)
    axn.add_patch(FancyBboxPatch((0.3, 0.2), 9.4, 0.7, boxstyle="round,pad=0.02",
                  facecolor=RED, edgecolor=RED, alpha=0.18))
    axn.text(5, 0.55, "HBM", ha="center", va="center", color=RED, fontsize=9)
    steps = ["Q,K,V", "scores\ns×s", "softmax\ns×s", "out\ns×d"]
    xs = np.linspace(1.0, 8.0, len(steps))
    for x, s in zip(xs, steps):
        _box(axn, (x - 0.7, 3.4), 1.4, 1.0, s, LIGHT, INK, tc=INK, fs=8.5)
        _arrow(axn, (x, 3.4), (x, 0.95), color=RED, lw=1.2)        # spill to HBM
    for x0, x1 in zip(xs, xs[1:]):
        _arrow(axn, (x0 + 0.7, 3.9), (x1 - 0.7, 3.9), color=INK)

    # fused: one box, no dips, only inputs/output touch HBM.
    axf = axes[1]
    axf.set_xlim(0, 10); axf.set_ylim(0, 5); axf.axis("off")
    axf.set_title("fused: s×s stays in SRAM", color=GREEN, fontsize=10)
    axf.add_patch(FancyBboxPatch((0.3, 0.2), 9.4, 0.7, boxstyle="round,pad=0.02",
                  facecolor=GREEN, edgecolor=GREEN, alpha=0.15))
    axf.text(5, 0.55, "HBM", ha="center", va="center", color=GREEN, fontsize=9)
    _box(axf, (0.6, 3.4), 1.6, 1.0, "Q,K,V", LIGHT, INK, tc=INK, fs=8.5)
    _box(axf, (3.4, 3.0), 3.2, 1.7, "scores · softmax · out\none kernel, no spill",
         GREEN, GREEN, fs=8.5)
    _box(axf, (7.7, 3.4), 1.6, 1.0, "out s×d", LIGHT, INK, tc=INK, fs=8.5)
    _arrow(axf, (2.2, 3.9), (3.4, 3.9), color=INK)
    _arrow(axf, (6.6, 3.85), (7.7, 3.9), color=INK)
    _arrow(axf, (1.4, 3.4), (1.4, 0.95), color=GREEN, lw=1.2)   # only input read
    _arrow(axf, (8.5, 3.4), (8.5, 0.95), color=GREEN, lw=1.2)   # only output write

    fig.suptitle("same math, two schedules, the fused plan skips the s×s round-trip",
                 color=INK, fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(OUT / "fusion.png", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    roofline()
    ab_flip()
    pipeline()
    fusion()
    print(f"wrote figures to {OUT}")
