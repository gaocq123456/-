"""Re-layout of the two result figures in the ORIGINAL style (Comic Neue, original colours).

Hatched bars (No-history, GPT-6 Sol) use the mosaic fill from mosaic_style.py."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from matplotlib.patches import Patch
import numpy as np
from pathlib import Path
from mosaic_style import HandlerMosaic, mosaic_rect

HERE = Path(__file__).resolve().parent
fm.fontManager.addfont(str(HERE / 'fonts' / 'ComicNeue-Regular.ttf'))
INK = '#1e1e1e'; INK2 = '#555555'; GRID = '#ededed'
GREEN = '#6aaa6e'; ORANGE = '#f0a35a'; RED = '#dc6e5f'; GRAY = '#9a9a9a'; SLATE = '#7a93ad'; BLUE = '#5a8bbf'; LGREEN = '#9ed39a'
plt.rcParams.update({
    'font.family': ['Comic Neue', 'DejaVu Sans'], 'font.size': 7, 'pdf.fonttype': 42,
    'axes.spines.top': False, 'axes.spines.right': False, 'axes.linewidth': 0.9, 'axes.edgecolor': INK2,
    'xtick.color': INK2, 'ytick.color': INK2, 'xtick.labelcolor': INK2, 'ytick.labelcolor': INK2,
    'xtick.major.width': 0.9, 'ytick.major.width': 0.9, 'xtick.major.size': 3, 'ytick.major.size': 3,
    'text.color': INK, 'axes.labelcolor': INK2, 'hatch.linewidth': 0.6,
})
OUT = str(HERE) + '/'

# =========================================================== results detail (full text width)
tasks = ['Info.\nextract.', 'Multi-\nsession', 'Temporal\nreason.', 'Knowl.\nupdate', 'Abs-\ntention']
data = {
    'Qwen3.8-27B': dict(noh=[6.7, 3.3, 2.4, 1.4, 0.0], full=[62.0, 41.3, 37.8, 33.3, 53.3],
                        orc=[88.7, 71.9, 66.1, 61.1, 76.7], ret=[76.7, 53.7, 45.7, 44.4, 63.3], sd=[0.7, 0, 0, 0, 0]),
    'DeepSeek-V4-Pro': dict(noh=[8.7, 4.1, 3.1, 1.4, 0.0], full=[70.0, 52.1, 48.0, 43.1, 66.7],
                            orc=[92.7, 81.0, 76.4, 72.2, 86.7], ret=[82.7, 65.3, 58.3, 55.6, 73.3], sd=[0, 0.8, 0.8, 0, 0]),
    'GPT-6 Sol': dict(noh=[10.0, 5.0, 3.9, 2.8, 0.0], full=[76.0, 61.2, 56.7, 51.4, 76.7],
                      orc=[96.0, 87.6, 84.3, 80.6, 93.3], ret=[88.0, 75.2, 67.7, 65.3, 83.3], sd=[0.7, 0, 0, 0, 0]),
}
gaps = {'Qwen3.8-27B': [12.0, 18.2, 20.5, 16.7, 13.3], 'DeepSeek-V4-Pro': [10.0, 15.7, 18.1, 16.7, 13.3],
        'GPT-6 Sol': [8.0, 12.4, 16.5, 15.3, 10.0]}

fig = plt.figure(figsize=(6.3, 4.15))
gs = fig.add_gridspec(2, 3, height_ratios=[1, 1.0], hspace=0.6, wspace=0.16, left=0.072, right=0.995, top=0.855, bottom=0.085)
x = np.arange(5); w = 0.2
for p, (gen, d) in enumerate(data.items()):
    ax = fig.add_subplot(gs[0, p])
    xs = [x + (k - 1.5) * w for k in range(4)]
    for xi, v in zip(xs[0], d['noh']):
        mosaic_rect(ax, (xi - w / 2, 0), w, v, GRAY, '#f2f2f2', GRAY, 0.8)
    ax.bar(xs[1], d['full'], w, facecolor='white', edgecolor=INK2, lw=0.9, label='Full history', zorder=2)
    ax.bar(xs[2], d['orc'], w, facecolor=GREEN, edgecolor=GREEN, lw=0.5, label='Oracle (annotated sessions)', zorder=2)
    ax.bar(xs[3], d['ret'], w, facecolor=ORANGE, edgecolor=ORANGE, lw=0.5, label='System-retrieved', zorder=2)
    for xi, v, s in zip(xs[3], d['ret'], d['sd']):
        if s > 0:
            ax.errorbar(xi, v, yerr=s, color='#2b2b2b', lw=0.8, capsize=1.6, capthick=0.8, zorder=3)
    for xi, g, o in zip(x, gaps[gen], d['orc']):
        ax.text(xi + 0.2, o + 3, f'−{g:.1f}', ha='center', va='bottom', fontsize=6.3, color=INK2)
    ax.set_ylim(0, 110); ax.set_yticks([0, 20, 40, 60, 80, 100])
    ax.set_xticks(x); ax.set_xticklabels(tasks, fontsize=6.1, linespacing=0.95)
    ax.yaxis.grid(True, color=GRID, lw=0.8, zorder=0); ax.set_axisbelow(True)
    ax.set_title(f'({"abc"[p]})  {gen}', fontsize=7.6, color=INK, pad=4)
    if p == 0:
        ax.set_ylabel('Accuracy (%)', fontsize=7)
    else:
        ax.tick_params(axis='y', labelleft=False)
h, l = fig.axes[0].get_legend_handles_labels()
noh = Patch(label='No-history')
fig.legend([noh] + h, ['No-history'] + l, loc='upper left', ncol=4, frameon=False, bbox_to_anchor=(0.065, 1.0), handlelength=1.6, columnspacing=1.6, fontsize=6.8,
           handler_map={noh: HandlerMosaic(GRAY, '#f2f2f2', GRAY, 0.8)})
fig.text(0.072, 0.925, 'number above each group: Retrieved $-$ Oracle (points);  error bars: s.d. over 3 retrieval seeds (not drawn when s.d. = 0);  decoding temperature 0',
         fontsize=6, color=INK2)

ax = fig.add_subplot(gs[1, :])
groups = ['Oracle\nbaseline', 'I1\n+distractors', 'I2\nkey first', 'I2\nkey middle', 'I2\nkey last', 'Retrieved\nbaseline', 'I3\n+back-filled']
gcol = [GREEN, SLATE, BLUE, BLUE, BLUE, ORANGE, LGREEN]
vals = [[74.2, 82.4, 88.6], [63.2, 74.2, 83.4], [74.2, 82.4, 88.6], [65.8, 74.8, 83.2], [71.0, 79.2, 85.8], [57.8, 67.8, 76.2], [69.2, 77.4, 85.2]]
xg = np.arange(len(groups)); wg = 0.25
for gi, (c, v) in enumerate(zip(gcol, vals)):
    for k in range(3):
        xi = xg[gi] + (k - 1) * wg
        if k == 0:
            ax.bar(xi, v[k], wg, facecolor='white', edgecolor=c, lw=1.0, zorder=2)
        elif k == 1:
            ax.bar(xi, v[k], wg, facecolor=c, edgecolor=c, lw=1.0, zorder=2)
        else:
            mosaic_rect(ax, (xi - wg / 2, 0), wg, v[k], c, 'white', c, 1.0)
        lab = f'{v[k]:.1f}'
        ax.text(xi, v[k] + 1.5, lab, ha='center', va='bottom', fontsize=6, color=INK2)
ax.set_ylim(0, 110); ax.set_yticks([0, 20, 40, 60, 80, 100]); ax.set_ylabel('Accuracy (%)', fontsize=7)
ax.set_xticks(xg); ax.set_xticklabels(groups, fontsize=6.4, linespacing=0.95)
ax.yaxis.grid(True, color=GRID, lw=0.8, zorder=0); ax.set_axisbelow(True)
ax.set_title('(d)  Controlled interventions', fontsize=7.6, color=INK, loc='left', pad=4)
gh = [Patch(facecolor='white', edgecolor=INK2, lw=0.9, label='open: Qwen3.8-27B'),
      Patch(facecolor='#8a8a8a', edgecolor='#8a8a8a', label='filled: DeepSeek-V4-Pro'),
      Patch(label='hatched: GPT-6 Sol')]
ax.legend(handles=gh, loc='lower right', bbox_to_anchor=(1.0, 1.0), ncol=3, frameon=False, fontsize=6.6, handlelength=1.6, columnspacing=1.4,
          handler_map={gh[2]: HandlerMosaic(INK2, 'white', INK2, 0.9)})
fig.savefig(OUT + 'results_detail.pdf')
fig.savefig(OUT + 'results_detail_preview.png', dpi=220)

# =========================================================== stale use (single column)
systems = ['APEX-MEM', 'Zep', 'Mem0', 'Memory-R1', 'TiMem', 'MemoryOS', 'LightMem', 'Full history\n(no memory\nmanagement)']
tier = [0, 0, 1, 1, 2, 2, 2, 3]
TC = [GREEN, ORANGE, RED, GRAY]
mean = [[24.1, 16.9, 13.2], [26.9, 19.1, 14.8], [41.2, 33.8, 28.1], [43.9, 36.2, 29.9], [49.1, 37.9, 33.2], [51.8, 41.1, 34.9], [55.2, 43.9, 37.1], [58.1, 47.0, 40.9]]
# s.d. over three retrieval seeds (Full-history control has none)
sd = [[2.4, 2.1, 2.2], [2.6, 2.2, 1.8], [2.5, 2.3, 2.1], [2.7, 2.4, 1.9], [2.7, 2.5, 2.5], [2.6, 2.3, 2.4], [2.5, 2.3, 2.9], [0, 0, 0]]
fig, ax = plt.subplots(figsize=(3.03, 3.85))
fig.subplots_adjust(left=0.215, right=0.735, top=0.79, bottom=0.115)
bh = 0.26
ypos = np.arange(len(systems))[::-1].astype(float)
for i, s in enumerate(systems):
    c = TC[tier[i]]
    for k in range(3):
        y = ypos[i] + (1 - k) * bh
        v = mean[i][k]
        if k == 0:
            ax.barh(y, v, bh * 0.92, facecolor='white', edgecolor=c, lw=0.9, zorder=2)
        elif k == 1:
            ax.barh(y, v, bh * 0.92, facecolor=c, edgecolor=c, lw=0.9, zorder=2)
        else:
            mosaic_rect(ax, (0, y - bh * 0.92 / 2), v, bh * 0.92, c, 'white', c, 0.9)
        e = sd[i][k]
        if e > 0:
            ax.errorbar(v, y, xerr=e, color='#3a3a3a', lw=0.6, capsize=1.3, capthick=0.6, zorder=3)
        ax.text(v + e + 0.9, y, f'{v:.1f}%', va='center', fontsize=5.2, color=INK2)
ax.set_yticks(ypos); ax.set_yticklabels(systems, fontsize=6.3, linespacing=0.9)
ax.set_xlim(0, 70); ax.set_xticks([0, 10, 20, 30, 40, 50, 60, 70])
ax.set_ylim(ypos[-1] - 0.55, ypos[0] + 0.55)
ax.xaxis.grid(True, color=GRID, lw=0.8, zorder=0); ax.set_axisbelow(True)
ax.tick_params(axis='x', labelsize=6.2); ax.tick_params(axis='y', length=2.5)
ax.set_xlabel('Stale-use rate: answers grounded in\na superseded version (%)', fontsize=6.4, linespacing=0.95)
# group means on the right, in tier colour
means = [('25.5 / 18.0 / 14.0', 0, 1), ('42.6 / 35.0 / 29.0', 2, 3), ('52.0 / 41.0 / 35.1', 4, 6), ('58.1 / 47.0 / 40.9', 7, 7)]
for t, (txt, a, b) in enumerate(means):
    yc = (ypos[a] + ypos[b]) / 2
    ax.text(72, yc, txt, va='center', ha='left', fontsize=5.4, color=TC[t], clip_on=False)
ax.text(72, ypos[0] + 0.62, 'group mean\nQwen/DS/GPT', va='bottom', ha='left', fontsize=5.4, color=INK2, clip_on=False, linespacing=0.95)
# legends (tier colours + generator encodings) and separation note
th = [Patch(facecolor=GREEN, label='R explicit (●)'), Patch(facecolor=ORANGE, label='R partial: overwrite / delete (◐)'),
      Patch(facecolor=RED, label='R absent (○)'), Patch(facecolor=GRAY, label='control: full history, no memory mgmt.')]
gh = [Patch(facecolor='white', edgecolor=INK2, lw=0.8, label='open: Qwen3.8-27B'),
      Patch(facecolor='#8a8a8a', edgecolor='#8a8a8a', label='filled: DeepSeek-V4-Pro'),
      Patch(label='hatched: GPT-6 Sol')]
l1 = fig.legend(handles=th, loc='upper left', bbox_to_anchor=(0.0, 0.955), ncol=1, frameon=False, fontsize=5.8, handlelength=1.3, handleheight=0.8, labelspacing=0.25)
l2 = fig.legend(handles=gh, loc='upper left', bbox_to_anchor=(0.6, 0.955), ncol=1, frameon=False, fontsize=5.8, handlelength=1.3, handleheight=0.8, labelspacing=0.25,
                handler_map={gh[2]: HandlerMosaic(INK2, 'white', INK2, 0.8)})
fig.text(0.5, 0.99, '●–○ separation: 26.5 / 23.0 / 21.1 points (Qwen / DeepSeek / GPT-6 Sol)', ha='center', va='top', fontsize=6, color=INK2)
fig.savefig(OUT + 'stale_use.pdf')
fig.savefig(OUT + 'stale_use_preview.png', dpi=260)
print('ok')
