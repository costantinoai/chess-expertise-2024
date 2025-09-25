#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Shared helpers to plot fsaverage surface overlays with optional contours.

Functions here are analysis-agnostic and reusable across MVPA/GLM/Neurosynth.
"""
from __future__ import annotations

from typing import Optional, Tuple
import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
from matplotlib import colorbar as mcolorbar
from nilearn.datasets import load_fsaverage, load_fsaverage_data
from nilearn.plotting import plot_surf_stat_map, plot_surf_contours


def plot_left_fsaverage_grid(
    overlay_lh: np.ndarray,
    label_map_lh: Optional[np.ndarray] = None,
    title: str = "",
    out_path: Optional[str] = None,
    color_range: Tuple[float, float] = (-1.0, 1.0),
    cmap_positive: str = "Reds",
    cmap_negative: str = "coolwarm",
    add_colorbar: bool = True,
    add_contours: bool = True,
    contour_color: str = "k",
    contour_linewidth: float = 0.3,
) -> Tuple[plt.Figure, Optional[str]]:
    """Plot a 2x3 grid of fsaverage LH views with optional ROI contours.

    Top row: lateral, medial, ventral on inflated mesh, with curvature background.
    Bottom row: dorsal on flat mesh. Optional colorbar placed on the right.

    Parameters
    - overlay_lh: vertex-wise data for the left hemisphere (n_vertices,).
    - label_map_lh: integer label map for LH (same vertices); used for contours if provided.
    - title: figure suptitle.
    - out_path: if set, path to save PNG.
    - color_range: (vmin, vmax) for shared scale.
    - cmap_positive/cmap_negative: colormaps for nonnegative / signed data.
    - add_colorbar: whether to draw a vertical colorbar on the right.
    - add_contours: whether to add ROI contours from label_map_lh.
    - contour_color/contour_linewidth: styling for contours.
    """
    vmin, vmax = color_range
    cmap_used = cmap_negative if vmin < 0 else cmap_positive

    fsavg = load_fsaverage("fsaverage")
    surf_inflated = fsavg["inflated"]
    surf_flat = fsavg["flat"]
    bg_map = load_fsaverage_data(mesh="fsaverage", data_type="curvature")

    fig = plt.figure(figsize=(10, 8))
    gs = fig.add_gridspec(
        nrows=2, ncols=3, height_ratios=[1.0, 1.5], wspace=0.05, hspace=0.03
    )
    ax_lateral = fig.add_subplot(gs[0, 0], projection="3d")
    ax_medial = fig.add_subplot(gs[0, 1], projection="3d")
    ax_ventral = fig.add_subplot(gs[0, 2], projection="3d")
    ax_dorsal = fig.add_subplot(gs[1, :], projection="3d")

    plt.subplots_adjust(left=0.05, right=0.9, top=0.93, bottom=0.06)

    # Optional colorbar on the right
    if add_colorbar:
        cax = fig.add_axes([0.92, 0.25, 0.012, 0.25])
        norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
        cb = mcolorbar.ColorbarBase(cax, cmap=cmap_used, norm=norm, orientation="vertical")
        cb.set_ticks([vmin, 0, vmax])
        cb.set_ticklabels([f"{vmin:.2f}", "0", f"{vmax:.2f}"])
        cb.ax.tick_params(labelsize=12)

    # Top: three inflated views
    for ax, view in [(ax_lateral, "lateral"), (ax_medial, "medial"), (ax_ventral, "ventral")]:
        plot_surf_stat_map(
            surf_mesh=surf_inflated,
            stat_map=overlay_lh,
            bg_map=bg_map,
            hemi="left",
            view=view,
            cmap=cmap_used,
            vmin=vmin,
            vmax=vmax,
            colorbar=False,
            figure=fig,
            axes=ax,
            darkness=0.8,
            bg_on_data=True,
        )
        if add_contours and label_map_lh is not None:
            # Use all positive labels as levels
            levels = [int(x) for x in np.unique(label_map_lh) if int(x) > 0]
            plot_surf_contours(
                surf_mesh=surf_inflated,
                roi_map=label_map_lh,
                view=view,
                hemi="left",
                figure=fig,
                axes=ax,
                colors=contour_color,
                levels=levels,
                linewidths=contour_linewidth,
            )

    # Bottom: flat dorsal
    f1 = plot_surf_stat_map(
        surf_mesh=surf_flat,
        stat_map=overlay_lh,
        hemi="left",
        view="dorsal",
        cmap=cmap_used,
        vmin=vmin,
        vmax=vmax,
        colorbar=False,
        figure=fig,
        axes=ax_dorsal,
        darkness=0.2,
    )
    if add_contours and label_map_lh is not None:
        levels = [int(x) for x in np.unique(label_map_lh) if int(x) > 0]
        plot_surf_contours(
            surf_mesh=surf_flat,
            roi_map=label_map_lh,
            view="dorsal",
            hemi="left",
            figure=f1,
            axes=ax_dorsal,
            colors=contour_color,
            levels=levels,
            linewidths=contour_linewidth,
        )

    if title:
        fig.suptitle(title, fontsize=16, y=0.98)
    plt.tight_layout()

    if out_path:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        fig.savefig(out_path, dpi=300, bbox_inches="tight")

    return fig, out_path

