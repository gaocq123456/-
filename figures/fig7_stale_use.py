"""Figure 7: stale-evidence use per memory system, coloured by R tier
(Table 1); open / filled / mosaic bars are Qwen3.8-27B / DeepSeek-V4-Pro /
GPT-6 Sol.

Rates come from Table 4; error bars are the s.d. over three retrieval seeds.
GPT-6 Sol bars use the mosaic fill from mosaic_style.py.

    python figures/fig7_stale_use.py [--out-dir figures]
"""

import argparse

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.transforms import ScaledTranslation, blended_transform_factory

from mosaic_style import (INK, TEXT, GRID, WHITE, LEGEND_FILL, MosaicFill,
                          HandlerMosaic, mosaic_rect, setup_fonts,
                          add_common_args, save)

ERR = "#3a3a3a"
TIERS = {
    "explicit": ("#6aaa6e", "R explicit (●)", "25.5 / 18.0 / 14.0"),
    "partial": ("#f0a35a", "R partial: overwrite / delete (◐)",
                "42.6 / 35.0 / 29.0"),
    "absent": ("#dc6e5f", "R absent (○)", "52.0 / 41.0 / 35.1"),
    "control": ("#9a9a9a", "control: full history, no memory mgmt.",
                "58.1 / 47.0 / 40.9"),
}

# (system, R tier, (rate, s.d.) for Qwen / DeepSeek / GPT-6 Sol)
SYSTEMS = [
    ("APEX-MEM", "explicit", (24.1, 2.4), (16.9, 2.1), (13.2, 2.2)),
    ("Zep", "explicit", (26.9, 2.6), (19.1, 2.2), (14.8, 1.8)),
    ("Mem0", "partial", (41.2, 2.5), (33.8, 2.3), (28.1, 2.1)),
    ("Memory-R1", "partial", (43.9, 2.7), (36.2, 2.4), (29.9, 1.9)),
    ("TiMem", "absent", (49.1, 2.7), (37.9, 2.5), (33.2, 2.5)),
    ("MemoryOS", "absent", (51.8, 2.6), (41.1, 2.3), (34.9, 2.4)),
    ("LightMem", "absent", (55.2, 2.5), (43.9, 2.3), (37.1, 2.9)),
    ("Full history\n(no memory\nmanagement)", "control",
     (58.1, 0.0), (47.0, 0.0), (40.9, 0.0)),
]

SEPARATION = ("●–○ separation: 26.5 / 23.0 / 21.1 points "
              "(Qwen / DeepSeek / GPT-6 Sol)")

FIG_W, FIG_H = 218.16, 277.2  # points (3.03 x 3.85 in)
BAR_H = 0.24
BAR_STEP = 0.26


def main():
    ap = argparse.ArgumentParser()
    add_common_args(ap, "fig7_stale_use")
    args = ap.parse_args()
    MosaicFill.cell = args.cell
    setup_fonts()

    fig = plt.figure(figsize=(FIG_W / 72, FIG_H / 72))

    def rect(x, y, w, h):
        return [x / FIG_W, y / FIG_H, w / FIG_W, h / FIG_H]

    ax = fig.add_axes(rect(46.90, 31.88, 113.44, 187.11))

    for i, (_, tier, *rates) in enumerate(SYSTEMS):
        col = TIERS[tier][0]
        for k, (v, sd) in enumerate(rates):
            y = i + (k - 1) * BAR_STEP
            xy = (0, y - BAR_H / 2)
            if k == 0:
                ax.add_patch(Rectangle(xy, v, BAR_H, facecolor=WHITE,
                                       edgecolor=col, linewidth=0.9))
            elif k == 1:
                ax.add_patch(Rectangle(xy, v, BAR_H, facecolor=col,
                                       edgecolor=col, linewidth=0.9))
            else:
                mosaic_rect(ax, xy, v, BAR_H, col, WHITE, col, 0.9)
            if sd > 0:
                ax.errorbar(v, y, xerr=sd, fmt="none", ecolor=ERR,
                            elinewidth=0.6, capsize=1.3, capthick=0.6,
                            zorder=3)
            ax.text(v + sd + 0.9, y, f"{v:.1f}%", ha="left", va="center",
                    fontsize=5.2, color=TEXT)

    ax.set_xlim(0, 70)
    ax.set_ylim(len(SYSTEMS) - 0.45, -0.55)
    ax.set_xticks(range(0, 71, 10))
    ax.set_yticks(range(len(SYSTEMS)))
    ax.set_yticklabels([s[0] for s in SYSTEMS], fontsize=6.3,
                       linespacing=0.93)
    ax.set_axisbelow(True)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(TEXT)
        ax.spines[side].set_linewidth(0.9)
    ax.tick_params(colors=TEXT, width=0.9)
    ax.tick_params(axis="x", length=3, labelsize=6.2)
    ax.tick_params(axis="y", length=2.5)
    ax.set_xlabel("Stale-use rate: answers grounded in\n"
                  "a superseded version (%)", fontsize=6.4, color=TEXT,
                  linespacing=0.93)

    # tier means in a column right of the axes
    right = blended_transform_factory(ax.transAxes, ax.transData) + \
        ScaledTranslation(3.25 / 72, 0, fig.dpi_scale_trans)
    for tier, (col, _, means) in TIERS.items():
        rows = [i for i, s in enumerate(SYSTEMS) if s[1] == tier]
        ax.text(1.0, sum(rows) / len(rows), means, transform=right,
                ha="left", va="center", fontsize=5.4, color=col)
    ax.text(1.0, -0.55, "group mean\nQwen/DS/GPT",
            transform=right + ScaledTranslation(0, 1.88 / 72,
                                                fig.dpi_scale_trans),
            ha="left", va="bottom", fontsize=5.4, color=TEXT)

    fig.text(0.5, 269.03 / FIG_H, SEPARATION, ha="center", va="baseline",
             fontsize=6.0, color=TEXT)

    legend_kw = dict(loc="upper left", frameon=False, fontsize=5.8,
                     labelcolor=INK, handlelength=1.3, handleheight=0.766,
                     labelspacing=0.25, borderaxespad=0.0)
    tier_handles = [Rectangle((0, 0), 1, 1, facecolor=col, linewidth=0)
                    for col, _, _ in TIERS.values()]
    fig.legend(tier_handles, [lab for _, lab, _ in TIERS.values()],
               bbox_to_anchor=rect(2.90, 261.83, 0, 0)[:2], **legend_kw)

    gen_handles = [Rectangle((0, 0), 1, 1, facecolor=WHITE, edgecolor=TEXT,
                             linewidth=0.8),
                   Rectangle((0, 0), 1, 1, facecolor=LEGEND_FILL,
                             edgecolor=LEGEND_FILL, linewidth=1.0),
                   Rectangle((0, 0), 1, 1)]
    fig.legend(gen_handles, ["open: Qwen3.8-27B", "filled: DeepSeek-V4-Pro",
                             "hatched: GPT-6 Sol"],
               handler_map={gen_handles[2]: HandlerMosaic(TEXT, WHITE, TEXT,
                                                          0.8)},
               bbox_to_anchor=rect(133.80, 261.83, 0, 0)[:2], **legend_kw)

    save(fig, args)


if __name__ == "__main__":
    main()
