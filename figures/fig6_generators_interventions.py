"""Figure 6: accuracy by task type per generator (a)-(c) and controlled
interventions I1-I3 (d).

Data come from Table 11 (main results) and Table 12 (interventions).
Hatched bars (No-history in (a)-(c), GPT-6 Sol in (d)) use the mosaic fill
from mosaic_style.py.

    python figures/fig6_generators_interventions.py [--out-dir figures]
"""

import argparse

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from mosaic_style import (INK, TEXT, GRID, WHITE, LEGEND_FILL, MosaicFill,
                          HandlerMosaic, mosaic_rect, setup_fonts,
                          add_common_args, save)

ERR = "#2b2b2b"
GREEN = "#6aaa6e"    # Oracle
ORANGE = "#f0a35a"   # System-retrieved
SLATE = "#7a93ad"    # I1
BLUE = "#5a8bbf"     # I2
LIGHT_GREEN = "#9ed39a"  # I3
NOHIST_EDGE = "#9a9a9a"
NOHIST_BG = "#f2f2f2"

FIG_W, FIG_H = 453.6, 298.8  # points (6.3 x 4.15 in)
YMAX = 110


# ---------------------------------------------------------------- data
TASKS = ["Info.\nextract.", "Multi-\nsession", "Temporal\nreason.",
         "Knowl.\nupdate", "Abs-\ntention"]

# Table 11: No-history, Full history, Oracle, Retrieved, Retrieved s.d.,
# Oracle - Retrieved gap (computed before rounding).
MAIN = {
    "(a)  Qwen3.8-27B": dict(
        nohist=[6.7, 3.3, 2.4, 1.4, 0.0],
        full=[62.0, 41.3, 37.8, 33.3, 53.3],
        oracle=[88.7, 71.9, 66.1, 61.1, 76.7],
        retr=[76.7, 53.7, 45.7, 44.4, 63.3],
        sd=[0.7, 0.0, 0.0, 0.0, 0.0],
        gap=[12.0, 18.2, 20.5, 16.7, 13.3]),
    "(b)  DeepSeek-V4-Pro": dict(
        nohist=[8.7, 4.1, 3.1, 1.4, 0.0],
        full=[70.0, 52.1, 48.0, 43.1, 66.7],
        oracle=[92.7, 81.0, 76.4, 72.2, 86.7],
        retr=[82.7, 65.3, 58.3, 55.6, 73.3],
        sd=[0.0, 0.8, 0.8, 0.0, 0.0],
        gap=[10.0, 15.7, 18.1, 16.7, 13.3]),
    "(c)  GPT-6 Sol": dict(
        nohist=[10.0, 5.0, 3.9, 2.8, 0.0],
        full=[76.0, 61.2, 56.7, 51.4, 76.7],
        oracle=[96.0, 87.6, 84.3, 80.6, 93.3],
        retr=[88.0, 75.2, 67.7, 65.3, 83.3],
        sd=[0.7, 0.0, 0.0, 0.0, 0.0],
        gap=[8.0, 12.4, 16.5, 15.3, 10.0]),
}

# Table 12: (label, colour, Qwen, DeepSeek, GPT-6 Sol)
INTERVENTIONS = [
    ("Oracle\nbaseline", GREEN, 74.2, 82.4, 88.6),
    ("I1\n+distractors", SLATE, 63.2, 74.2, 83.4),
    ("I2\nkey first", BLUE, 74.2, 82.4, 88.6),
    ("I2\nkey middle", BLUE, 65.8, 74.8, 83.2),
    ("I2\nkey last", BLUE, 71.0, 79.2, 85.8),
    ("Retrieved\nbaseline", ORANGE, 57.8, 67.8, 76.2),
    ("I3\n+back-filled", LIGHT_GREEN, 69.2, 77.4, 85.2),
]


# ---------------------------------------------------------------- plot
def style_axes(ax, ylabel):
    ax.set_ylim(0, YMAX)
    ax.set_yticks(range(0, 101, 20))
    ax.set_axisbelow(True)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(TEXT)
        ax.spines[side].set_linewidth(0.9)
    ax.tick_params(colors=TEXT, width=0.9, length=3)
    ax.tick_params(axis="y", labelsize=7.0)
    if ylabel:
        ax.set_ylabel("Accuracy (%)", fontsize=7.0, color=TEXT)
    else:
        ax.tick_params(axis="y", labelleft=False)


def panel_main(ax, title, d, ylabel):
    w = 0.2
    x = np.arange(len(TASKS))
    for xi, v in zip(x, d["nohist"]):
        mosaic_rect(ax, (xi - 2 * w, 0), w, v, NOHIST_EDGE, NOHIST_BG,
                    NOHIST_EDGE, 0.8)
    ax.bar(x - 0.5 * w, d["full"], w, facecolor=WHITE, edgecolor=TEXT,
           linewidth=0.9)
    ax.bar(x + 0.5 * w, d["oracle"], w, color=GREEN, edgecolor=GREEN,
           linewidth=0.5)
    ax.bar(x + 1.5 * w, d["retr"], w, color=ORANGE, edgecolor=ORANGE,
           linewidth=0.5)
    sd = np.array(d["sd"])
    has_sd = sd > 0
    ax.errorbar(x[has_sd] + 1.5 * w, np.array(d["retr"])[has_sd],
                yerr=sd[has_sd], fmt="none", ecolor=ERR, elinewidth=0.8,
                capsize=1.6, capthick=0.8, zorder=3)
    for xi, o, r, s, g in zip(x, d["oracle"], d["retr"], sd, d["gap"]):
        ax.text(xi + w, max(o, r + s) + 3.0, f"−{g:.1f}",
                ha="center", va="bottom", fontsize=6.3, color=TEXT)
    ax.set_xticks(x)
    ax.set_xticklabels(TASKS, fontsize=6.1, linespacing=0.93)
    ax.margins(x=0.05)
    style_axes(ax, ylabel)
    ax.set_title(title, fontsize=7.6, color=INK, pad=4)


def panel_interventions(ax):
    w = 0.25
    x = np.arange(len(INTERVENTIONS))
    for xi, (_, col, q, ds, g) in zip(x, INTERVENTIONS):
        ax.bar(xi - w, q, w, facecolor=WHITE, edgecolor=col, linewidth=1.0)
        ax.bar(xi, ds, w, color=col, edgecolor=col, linewidth=1.0)
        mosaic_rect(ax, (xi + w / 2, 0), w, g, col, WHITE, col, 1.0)
        for dx, v in ((-w, q), (0, ds), (w, g)):
            ax.text(xi + dx, v + 1.5, f"{v:.1f}", ha="center", va="bottom",
                    fontsize=6.0, color=TEXT)
    ax.set_xticks(x)
    ax.set_xticklabels([lab for lab, *_ in INTERVENTIONS], fontsize=6.4,
                       linespacing=0.93)
    ax.margins(x=0.05)
    style_axes(ax, ylabel=True)
    ax.set_title("(d)  Controlled interventions", loc="left", fontsize=7.6,
                 color=INK, pad=4)

    proxies = [Rectangle((0, 0), 1, 1, facecolor=WHITE, edgecolor=TEXT,
                         linewidth=0.9),
               Rectangle((0, 0), 1, 1, facecolor=LEGEND_FILL,
                         edgecolor=LEGEND_FILL, linewidth=1.0),
               Rectangle((0, 0), 1, 1)]
    ax.legend(proxies, ["open: Qwen3.8-27B", "filled: DeepSeek-V4-Pro",
                        "hatched: GPT-6 Sol"],
              handler_map={proxies[2]: HandlerMosaic(TEXT, WHITE, TEXT, 0.9)},
              loc="lower right", bbox_to_anchor=(1.0, 1.0), ncol=3,
              frameon=False, fontsize=6.6, labelcolor=INK, handlelength=1.6,
              columnspacing=1.4)


def main():
    ap = argparse.ArgumentParser()
    add_common_args(ap, "fig6_generators_interventions")
    args = ap.parse_args()
    MosaicFill.cell = args.cell
    setup_fonts()

    fig = plt.figure(figsize=(FIG_W / 72, FIG_H / 72))

    def rect(x, y, w, h):
        return [x / FIG_W, y / FIG_H, w / FIG_W, h / FIG_H]

    for i, (title, d) in enumerate(MAIN.items()):
        ax = fig.add_axes(rect(32.65 + i * 146.27, 166.97, 126.10, 88.48))
        panel_main(ax, title, d, ylabel=(i == 0))
    panel_interventions(fig.add_axes(rect(32.65, 25.40, 418.64, 88.48)))

    proxies = [Rectangle((0, 0), 1, 1),
               Rectangle((0, 0), 1, 1, facecolor=WHITE, edgecolor=TEXT,
                         linewidth=0.9),
               Rectangle((0, 0), 1, 1, facecolor=GREEN, edgecolor=GREEN,
                         linewidth=0.5),
               Rectangle((0, 0), 1, 1, facecolor=ORANGE, edgecolor=ORANGE,
                         linewidth=0.5)]
    fig.legend(proxies, ["No-history", "Full history",
                         "Oracle (annotated sessions)", "System-retrieved"],
               handler_map={proxies[0]: HandlerMosaic(
                   NOHIST_EDGE, NOHIST_BG, NOHIST_EDGE, 0.8)},
               loc="lower left", bbox_to_anchor=rect(32.85, 282.1, 0, 0)[:2],
               ncol=4, frameon=False, fontsize=6.8, labelcolor=INK,
               handlelength=1.6, columnspacing=1.6, borderaxespad=0.0)
    fig.text(32.65 / FIG_W, 276.4 / FIG_H,
             "number above each group: Retrieved $-$ Oracle (points);  "
             "error bars: s.d. over 3 retrieval seeds (not drawn when "
             "s.d. = 0);  decoding temperature 0",
             fontsize=6.0, color=TEXT, ha="left", va="baseline")

    save(fig, args)


if __name__ == "__main__":
    main()
