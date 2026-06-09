#!/usr/bin/env python3
"""
Generate the README figures with matplotlib.

Produces four PNGs in docs/figures/:
  roofline.png   the roofline plot with attention plotted on it (data-driven)
  ab_flip.png    the thesis A/B: one e-graph, two winners
  pipeline.png   the six-stage pipeline
  fusion.png     naive vs fused HBM traffic

Run: python scripts/figures.py

House style, matched to the blog diagrams: hand-drawn xkcd mode, black and
white only, handwritten font (Humor Sans, committed at docs/fonts, with Comic
Sans as the installed fallback), faint notebook ruling behind everything,
lowercase titles, no long dashes, no text on top of other text. Series are
told apart by line weight and dashing, never by color.
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "docs" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

FONT = ROOT / "docs" / "fonts" / "HumorSans.ttf"
if FONT.exists():
    font_manager.fontManager.addfont(str(FONT))

INK = "#1b1f24"
GREY = "#8a8f98"
RULE = "#d4d4d4"


def _ruling(fig, n=9):
    """Faint horizontal notebook lines behind the whole figure."""
    bg = fig.add_axes([0, 0, 1, 1], zorder=-10)
    bg.set_xlim(0, 1)
    bg.set_ylim(0, 1)
    bg.axis("off")
    for y in np.linspace(0.06, 0.94, n):
        bg.plot([0.01, 0.99], [y, y], color=RULE, lw=1.0, solid_capstyle="round")


def _box(ax, xy, w, h, text, fs=11, lw=2.2):
    x, y = xy
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
        linewidth=lw, edgecolor=INK, facecolor="white", zorder=2))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            color=INK, fontsize=fs, zorder=3)


def _arrow(ax, p0, p1, ls="-", lw=2.0):
    ax.add_patch(FancyArrowPatch(
        p0, p1, arrowstyle="-|>", mutation_scale=16, linewidth=lw,
        linestyle=ls, color=INK, zorder=1, shrinkA=2, shrinkB=2))


# 1. roofline (data-driven) ---------------------------------------------------

def roofline():
    # H100-ish: 989 TFLOP/s over 3.35 TB/s gives a ridge near 295 flop/byte.
    peak_flops = 989e12
    peak_bw = 3.35e12
    ridge = peak_flops / peak_bw

    fig, ax = plt.subplots(figsize=(8.0, 4.8), dpi=150)
    _ruling(fig)
    I = np.logspace(0, 3.2, 200)
    achievable = np.minimum(peak_flops, peak_bw * I) / 1e12
    ax.plot(I, achievable, color=INK, lw=2.6, zorder=2)
    ax.axvline(ridge, color=INK, ls="--", lw=1.4, zorder=1)
    ax.text(ridge * 1.12, 80, "ridge point\n295 flop/byte", color=INK, fontsize=10)

    naive_i = 9.0
    naive_p = min(peak_flops, peak_bw * naive_i) / 1e12
    ax.plot([naive_i], [naive_p], "o", color=INK, ms=9, zorder=4)
    ax.annotate("naive attention\nabout 9 flop/byte", (naive_i, naive_p),
                textcoords="offset points", xytext=(-16, 30),
                color=INK, fontsize=10, ha="center")

    fused_i = 60.0
    fused_p = min(peak_flops, peak_bw * fused_i) / 1e12
    ax.plot([fused_i], [fused_p], "o", color=INK, ms=9, mfc="white", zorder=4)
    ax.annotate("fused, same math\njust fewer bytes", (fused_i, fused_p),
                textcoords="offset points", xytext=(20, -8),
                color=INK, fontsize=10, ha="left")
    _arrow(ax, (naive_i * 1.5, naive_p * 1.6), (fused_i * 0.72, fused_p * 0.78),
           ls=":", lw=1.6)

    ax.text(2.7, 740, "memory bound\n(waiting on HBM)", color=GREY, fontsize=11)
    ax.text(430, 470, "compute bound\n(waiting on math)", color=GREY, fontsize=11)

    ax.set_xscale("log")
    ax.set_xlim(1, 1600)
    ax.set_ylim(0, peak_flops / 1e12 * 1.12)
    ax.set_xlabel("arithmetic intensity (flop per byte moved)")
    ax.set_ylabel("achievable speed (TFLOP/s)")
    ax.set_title("the roofline of an H100, and where attention sits on it",
                 color=INK, fontsize=13)
    fig.savefig(OUT / "roofline.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)


# 2. the A/B flip -------------------------------------------------------------

def ab_flip():
    fig, ax = plt.subplots(figsize=(8.2, 4.4), dpi=150)
    _ruling(fig)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6.4)
    ax.axis("off")

    _box(ax, (3.0, 4.7), 4.0, 1.2,
         "one e-graph holding both forms:\nnaive and fused attention", fs=10)

    _arrow(ax, (4.2, 4.7), (2.0, 2.9))
    ax.text(1.45, 3.95, "cost model sees\nonly flops", color=INK,
            fontsize=10, ha="center")
    _arrow(ax, (5.8, 4.7), (8.0, 2.9))
    ax.text(8.55, 3.95, "cost model sees\nflops and hbm bytes", color=INK,
            fontsize=10, ha="center")

    _box(ax, (0.3, 1.5), 3.6, 1.4,
         "picks naive\nwrites the s by s scores\nout to slow memory", fs=9.5)
    _box(ax, (6.1, 1.5), 3.6, 1.4,
         "picks fused\nthe scores never leave\nfast memory", fs=9.5, lw=3.0)

    ax.text(5.0, 0.55,
            "the search code is identical in both runs.\n"
            "the only change is one extra constraint in the cost model.",
            color=GREY, fontsize=9.5, ha="center")

    ax.set_title("same search, same e-graph, one constraint flips the winner",
                 color=INK, fontsize=13)
    fig.savefig(OUT / "ab_flip.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)


# 3. the pipeline -------------------------------------------------------------

def pipeline():
    fig, ax = plt.subplots(figsize=(11.5, 2.9), dpi=150)
    _ruling(fig, n=6)
    ax.set_xlim(0, 28.8)
    ax.set_ylim(0, 3.6)
    ax.axis("off")

    stages = [
        ("program", "naive attention\nor the MLP"),
        ("rl-ir", "language, interpreter,\ncost accountant"),
        ("rl-opt", "e-graph of every\nequivalent form"),
        ("rl-cost", "roofline model,\nslowest resource wins"),
        ("rl-codegen", "lowers the winner\nto a fused kernel"),
        ("rl-ledger", "records the result\nso it replays"),
    ]
    w, gap, y, h = 4.0, 0.8, 0.6, 1.9
    x = 0.3
    edges = []
    for name, sub in stages:
        _box(ax, (x, y), w, h, f"{name}\n{sub}", fs=9)
        edges.append((x, x + w))
        x += w + gap
    for (_, r), (nl, _) in zip(edges, edges[1:]):
        _arrow(ax, (r, y + h / 2), (nl, y + h / 2))

    ax.set_title("a program becomes an e-graph, the cost model picks the "
                 "cheapest plan, codegen makes it real", color=INK, fontsize=12)
    fig.savefig(OUT / "pipeline.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)


# 4. fusion saves HBM ---------------------------------------------------------

def fusion():
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 3.9), dpi=150)
    _ruling(fig, n=7)

    axn = axes[0]
    axn.set_xlim(0, 10); axn.set_ylim(0, 5.8); axn.axis("off")
    axn.set_title("naive: every step writes to HBM\n(each s by s pass is 16 MB at s = 2048)",
                  color=INK, fontsize=10.5)
    axn.add_patch(FancyBboxPatch((0.3, 0.2), 9.4, 0.8, boxstyle="round,pad=0.02",
                  facecolor="white", edgecolor=INK, linewidth=2.2))
    axn.text(5, 0.6, "HBM (big, slow)", ha="center", va="center", color=INK, fontsize=10)
    steps = ["Q, K, V", "scores\ns by s", "softmax\ns by s", "out\ns by d"]
    xs = np.linspace(1.15, 8.85, len(steps))
    for x, s in zip(xs, steps):
        _box(axn, (x - 0.88, 4.0), 1.76, 1.3, s, fs=9)
        _arrow(axn, (x, 4.0), (x, 1.1), lw=1.7)
    for x0, x1 in zip(xs, xs[1:]):
        _arrow(axn, (x0 + 0.88, 4.65), (x1 - 0.88, 4.65))

    axf = axes[1]
    axf.set_xlim(0, 10); axf.set_ylim(0, 5.8); axf.axis("off")
    axf.set_title("fused: the s by s tensors stay in SRAM\n(only inputs read, only the output written)",
                  color=INK, fontsize=10.5)
    axf.add_patch(FancyBboxPatch((0.3, 0.2), 9.4, 0.8, boxstyle="round,pad=0.02",
                  facecolor="white", edgecolor=INK, linewidth=2.2))
    axf.text(5, 0.6, "HBM (big, slow)", ha="center", va="center", color=INK, fontsize=10)
    _box(axf, (0.5, 4.0), 1.8, 1.3, "Q, K, V", fs=9)
    _box(axf, (3.2, 3.8), 3.6, 1.7, "scores, softmax, out\nas one kernel,\nnothing spills", fs=9, lw=3.0)
    _box(axf, (7.7, 4.0), 1.8, 1.3, "out\ns by d", fs=9)
    _arrow(axf, (2.3, 4.65), (3.2, 4.65))
    _arrow(axf, (6.8, 4.65), (7.7, 4.65))
    _arrow(axf, (1.4, 4.0), (1.4, 1.1), lw=1.7)
    _arrow(axf, (8.6, 4.0), (8.6, 1.1), lw=1.7)

    fig.savefig(OUT / "fusion.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    with plt.xkcd(scale=1.1, length=110, randomness=2):
        plt.rcParams["figure.facecolor"] = "white"
        roofline()
        ab_flip()
        pipeline()
        fusion()
    print(f"wrote figures to {OUT}")
