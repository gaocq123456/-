"""Mosaic fill for hatched bars: small square tiles stepped along the
diagonal, used instead of plain hatch lines (fully vector in PDF output)."""

import numpy as np
from matplotlib.artist import Artist
from matplotlib.colors import to_rgb
from matplotlib.legend_handler import HandlerBase
from matplotlib.patches import Rectangle
from matplotlib.path import Path
from matplotlib.transforms import IdentityTransform, TransformedPath

# Mosaic tile geometry, in points.  Tiles sit on a grid anchored to the
# figure origin, so neighbouring bars share one continuous pattern.  Every
# PERIOD-th diagonal of the grid is filled, which gives a staircase of
# squares leaning like "/".
MOSAIC_CELL = 1.0
MOSAIC_PERIOD = 3
# Staircase tiles use a softened tint of the bar colour; the tiles between
# two staircases get a faint checker of the same colour.
MOSAIC_TINT = 0.7
MOSAIC_HALFTONE = 0.12


def mix(color, bg, t):
    """Opaque blend: t=1 gives color, t=0 gives bg."""
    c, b = np.array(to_rgb(color)), np.array(to_rgb(bg))
    return tuple(b + (c - b) * t)


class MosaicFill(Artist):
    """Square-tile diagonal fill clipped to a reference patch.

    The tiles are computed at draw time in device space, so their size is a
    fixed number of points regardless of the axes scale or output format.
    """

    cell = MOSAIC_CELL

    def __init__(self, patch, color, bg, period=MOSAIC_PERIOD,
                 tint=MOSAIC_TINT, halftone=MOSAIC_HALFTONE):
        super().__init__()
        self._patch = patch
        self._color = mix(color, bg, tint)
        self._half = mix(color, bg, halftone) if halftone else None
        self._period = period
        self.set_zorder(patch.get_zorder() + 0.01)

    @staticmethod
    def _squares(ii, jj, c):
        x, y = ii * c, jj * c
        polys = np.stack([np.c_[x, y], np.c_[x + c, y],
                          np.c_[x + c, y + c], np.c_[x, y + c]], axis=1)
        return Path.make_compound_path_from_polys(polys)

    def draw(self, renderer):
        if not self.get_visible():
            return
        bb = self._patch.get_window_extent(renderer)
        if bb.width <= 0 or bb.height <= 0:
            return
        c = renderer.points_to_pixels(self.cell)
        ii, jj = np.meshgrid(np.arange(np.floor(bb.x0 / c), np.ceil(bb.x1 / c)),
                             np.arange(np.floor(bb.y0 / c), np.ceil(bb.y1 / c)))
        ii, jj = ii.ravel(), jj.ravel()
        phase = (ii - jj) % self._period

        gc = renderer.new_gc()
        gc.set_linewidth(0)
        gc.set_antialiased(True)
        gc.set_snap(False)
        gc.set_clip_rectangle(self._patch.clipbox)
        gc.set_clip_path(TransformedPath(self._patch.get_path(),
                                         self._patch.get_transform()))
        renderer.open_group("mosaic", gid=self.get_gid())
        if self._half is not None:
            # checker in the gaps: alternate tiles of the off-diagonals
            sel = (phase != 0) & ((ii + jj) % 2 == 0)
            if sel.any():
                renderer.draw_path(gc, self._squares(ii[sel], jj[sel], c),
                                   IdentityTransform(), self._half)
        sel = phase == 0
        renderer.draw_path(gc, self._squares(ii[sel], jj[sel], c),
                           IdentityTransform(), self._color)
        renderer.close_group("mosaic")
        gc.restore()


def mosaic_rect(ax, xy, width, height, color, bg, edgecolor, lw, zorder=2):
    """Mosaic bar = background fill, mosaic tiles, then the outline on top."""
    face = Rectangle(xy, width, height, facecolor=bg, edgecolor="none",
                     linewidth=0, zorder=zorder)
    ax.add_patch(face)
    tiles = MosaicFill(face, color, bg)
    ax.add_artist(tiles)
    tiles.set_clip_box(ax.bbox)
    ax.add_patch(Rectangle(xy, width, height, facecolor="none",
                           edgecolor=edgecolor, linewidth=lw,
                           zorder=zorder + 0.02))


class HandlerMosaic(HandlerBase):
    """Legend handle drawn with the mosaic fill."""

    def __init__(self, color, bg, edgecolor, lw):
        super().__init__()
        self._style = color, bg, edgecolor, lw

    def create_artists(self, legend, orig_handle, xdescent, ydescent,
                       width, height, fontsize, trans):
        color, bg, edgecolor, lw = self._style
        xy = (-xdescent, -ydescent)
        face = Rectangle(xy, width, height, facecolor=bg, edgecolor="none",
                         linewidth=0, transform=trans)
        edge = Rectangle(xy, width, height, facecolor="none",
                         edgecolor=edgecolor, linewidth=lw, transform=trans)
        return [face, MosaicFill(face, color, bg), edge]
