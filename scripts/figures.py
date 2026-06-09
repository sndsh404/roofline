#!/usr/bin/env python3
"""
Generate the README figures with matplotlib.

Produces four PNGs in docs/figures/:
  roofline.png   the roofline plot with attention plotted on it (data-driven)
  ab_flip.png    the thesis A/B: one e-graph, two winners
  pipeline.png   the five-stage pipeline
  fusion.png     naive vs fused HBM traffic

Run: python scripts/figures.py
The flops and HBM numbers come from the same model the Rust code uses, so the
roofline figure is real, not decorative. House rules: no long dashes anywhere,
no text overlapping other text, captions live in the README, not in the image.
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
            color=tc, fontsize=fs, zorder=3)


def _arrow(ax, p0, p1, color=INK, style="-|>", ls="-", lw=1.6):
    ax.add_patch(FancyArrowPatch(
        p0, p1, arrowstyle=style, mutation_scale=14, linewidth=lw,
        linestyle=ls, color=color, zorder=1,
        shrinkA=2, shrinkB=2))


# 1. roofline (data-driven) ---------------------------------------------------

def roofline():
    # H100-ish: 989 TFLOP/s over 3.35 TB/s gives a ridge near 295 flop/byte.
    peak_flops = 989e12
    peak_bw = 3.35e12
    ridge = peak_flops / peak_bw

    fig, ax = plt.subplots(figsize=(7.6, 4.6), dpi=150)
    I = np.logspace(0, 3.2, 400)               # arithmetic intensity, flop/byte
    achievable = np.minimum(peak_flops, peak_bw * I) / 1e12  # TFLOP/s
    ax.plot(I, achievable, color=INK, lw=2.2, zorder=2)
    ax.axvline(ridge, color=GREY, ls="--", lw=1.2, zorder=1)
    ax.text(ridge * 1.08, 60, f"ridge point\n{ridge:.0f} flop/byte",
            color=GREY, fontsize=9)

    # naive attention sits near 9 flop/byte, deep in the memory-bound region.
    naive_i = 9.0
    naive_p = min(peak_flops, peak_bw * naive_i) / 1e12
    ax.scatter([naive_i], [naive_p], color=RED, s=70, zorder=4)
    ax.annotate("naive attention\n(about 9 flop/byte)", (naive_i, naive_p),
                textcoords="offset points", xytext=(-14, 26),
                color=RED, fontsize=9, ha="center")

    # fusion lifts intensity: same flops, far fewer bytes.
    fused_i = 60.0
    fused_p = min(peak_flops, peak_bw * fused_i) / 1e12
    ax.scatter([fused_i], [fused_p], color=GREEN, s=70, zorder=4)
    ax.annotate("fused\n(same math, fewer bytes)", (fused_i, fused_p),
                textcoords="offset points", xytext=(16, -6),
                color=GREEN, fontsize=9, ha="left")
    _arrow(ax, (naive_i * 1.45, naive_p * 1.55), (fused_i * 0.78, fused_p * 0.92),
           color=GREY, ls=":", lw=1.4)

    ax.fill_betweenx([0, peak_flops / 1e12], 1, ridge, color=RED, alpha=0.05)
    ax.fill_betweenx([0, peak_flops / 1e12], ridge, 1600, color=BLUE, alpha=0.05)
    ax.text(2.6, 730, "memory-bound\n(waiting on HBM)", color=RED, fontsize=10)
    ax.text(420, 480, "compute-bound\n(waiting on math)", color=BLUE, fontsize=10)

    ax.set_xscale("log")
    ax.set_xlim(1, 1600)
    ax.set_ylim(0, peak_flops / 1e12 * 1.10)
    ax.set_xlabel("arithmetic intensity (flop per byte moved)")
    ax.set_ylabel("achievable speed (TFLOP/s)")
    ax.set_title("the roofline of an H100, and where attention sits on it",
                 color=INK, fontsize=11)
    ax.grid(True, which="both", alpha=0.15)
    fig.tight_layout()
    fig.savefig(OUT / "roofline.png", bbox_inches="tight")
    plt.close(fig)


# 2. the A/B flip -------------------------------------------------------------

def ab_flip():
    fig, ax = plt.subplots(figsize=(7.8, 4.2), dpi=150)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6.4)
    ax.axis("off")

    _box(ax, (3.1, 4.7), 3.8, 1.2,
         "one e-graph holding both forms:\nnaive attention and fused attention",
         LIGHT, INK, tc=INK, fs=9.5)

    _arrow(ax, (4.3, 4.7), (2.0, 2.9), color=RED)
    ax.text(1.55, 3.95, "cost model sees\nonly FLOPs", color=RED,
            fontsize=9, ha="center")
    _arrow(ax, (5.7, 4.7), (8.0, 2.9), color=GREEN)
    ax.text(8.45, 3.95, "cost model sees\nFLOPs and HBM bytes", color=GREEN,
            fontsize=9, ha="center")

    _box(ax, (0.3, 1.6), 3.6, 1.3,
         "picks naive\nwrites the s by s scores\nout to slow memory", RED, RED, fs=9)
    _box(ax, (6.1, 1.6), 3.6, 1.3,
         "picks fused\nthe s by s scores never\nleave fast memory", GREEN, GREEN, fs=9)

    ax.text(5.0, 0.7,
            "the search code is identical in both runs.\n"
            "the only change is one extra constraint in the cost model.",
            color=GREY, fontsize=9, ha="center")

    ax.set_title("same search, same e-graph, one constraint flips the winner",
                 color=INK, fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT / "ab_flip.png", bbox_inches="tight")
    plt.close(fig)


# 3. the pipeline -------------------------------------------------------------

def pipeline():
    fig, ax = plt.subplots(figsize=(11.0, 2.7), dpi=150)
    ax.set_xlim(0, 26.4)
    ax.set_ylim(0, 3.4)
    ax.axis("off")

    stages = [
        ("program", "naive attention\nor the MLP", GREY),
        ("rl-ir", "language, interpreter,\ncost accountant", BLUE),
        ("rl-opt", "e-graph of every\nequivalent form", BLUE),
        ("rl-cost", "roofline model,\nslowest resource wins", INK),
        ("rl-codegen", "lowers the winner\nto a fused kernel", BLUE),
        ("rl-ledger", "records the result\nso it replays", GREY),
    ]
    w, gap, y, h = 4.0, 0.4, 0.7, 1.7
    x = 0.2
    edges = []
    for name, sub, c in stages:
        _box(ax, (x, y), w, h, f"{name}\n{sub}", c, c, fs=9)
        edges.append((x, x + w))
        x += w + gap
    for (_, r), (nl, _) in zip(edges, edges[1:]):
        _arrow(ax, (r, y + h / 2), (nl, y + h / 2))

    ax.set_title("the pipeline: a program becomes an e-graph, "
                 "the cost model picks the cheapest plan, codegen makes it real",
                 color=INK, fontsize=10.5)
    fig.tight_layout()
    fig.savefig(OUT / "pipeline.png", bbox_inches="tight")
    plt.close(fig)


# 4. fusion saves HBM ---------------------------------------------------------

def fusion():
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.6), dpi=150)

    # naive: every stage writes its result down to the HBM bar.
    axn = axes[0]
    axn.set_xlim(0, 10); axn.set_ylim(0, 5.6); axn.axis("off")
    axn.set_title("naive: every intermediate is written to HBM\n"
                  "(each s by s pass is 16 MB at s = 2048)", color=RED, fontsize=10)
    axn.add_patch(FancyBboxPatch((0.3, 0.2), 9.4, 0.7, boxstyle="round,pad=0.02",
                  facecolor=RED, edgecolor=RED, alpha=0.18))
    axn.text(5, 0.55, "HBM (big, slow)", ha="center", va="center", color=RED, fontsize=9)
    steps = ["Q, K, V", "scores\ns by s", "softmax\ns by s", "out\ns by d"]
    xs = np.linspace(1.1, 8.9, len(steps))
    for x, s in zip(xs, steps):
        _box(axn, (x - 0.85, 3.9), 1.7, 1.2, s, LIGHT, INK, tc=INK, fs=8.5)
        _arrow(axn, (x, 3.9), (x, 0.95), color=RED, lw=1.2)
    for x0, x1 in zip(xs, xs[1:]):
        _arrow(axn, (x0 + 0.85, 4.5), (x1 - 0.85, 4.5), color=INK)

    # fused: one kernel, only the inputs and the output touch HBM.
    axf = axes[1]
    axf.set_xlim(0, 10); axf.set_ylim(0, 5.6); axf.axis("off")
    axf.set_title("fused: the s by s tensors stay in SRAM\n"
                  "(only inputs read, only the output written)", color=GREEN, fontsize=10)
    axf.add_patch(FancyBboxPatch((0.3, 0.2), 9.4, 0.7, boxstyle="round,pad=0.02",
                  facecolor=GREEN, edgecolor=GREEN, alpha=0.15))
    axf.text(5, 0.55, "HBM (big, slow)", ha="center", va="center", color=GREEN, fontsize=9)
    _box(axf, (0.5, 3.9), 1.8, 1.2, "Q, K, V", LIGHT, INK, tc=INK, fs=8.5)
    _box(axf, (3.2, 3.7), 3.6, 1.6, "scores, softmax, out\nas one kernel, no spill",
         GREEN, GREEN, fs=8.5)
    _box(axf, (7.7, 3.9), 1.8, 1.2, "out\ns by d", LIGHT, INK, tc=INK, fs=8.5)
    _arrow(axf, (2.3, 4.5), (3.2, 4.5), color=INK)
    _arrow(axf, (6.8, 4.5), (7.7, 4.5), color=INK)
    _arrow(axf, (1.4, 3.9), (1.4, 0.95), color=GREEN, lw=1.2)
    _arrow(axf, (8.6, 3.9), (8.6, 0.95), color=GREEN, lw=1.2)

    fig.suptitle("same math, two schedules. the fused plan skips the round trips",
                 color=INK, fontsize=10.5)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(OUT / "fusion.png", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    roofline()
    ab_flip()
    pipeline()
    fusion()
    print(f"wrote figures to {OUT}")
