#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  4 21:22:01 2025

@author: costantino_ai
"""
import os
import matplotlib.pyplot as plt
import nilearn.surface as surf
from nilearn.plotting import plot_surf_stat_map
import matplotlib.gridspec as gridspec
from nilearn.datasets import fetch_surf_fsaverage


def plot_fsaverage_hemisphere(overlay, hemisphere, title="", out_path=None):
    """
    Plot a single hemisphere (left or right) in fsaverage space.
    Layout:
        ┌─────────┬─────────┬─────────┐
        │ Lateral │ Medial  │ Ventral │
        ├─────────┴─────────┴─────────┤
        │          Flat               │
        └─────────────────────────────┘

    Parameters
    ----------
    overlay : numpy.ndarray
        Array of shape (n_vertices,) for the given hemisphere.
    hemisphere : {"left", "right"}
        Which hemisphere to plot.
    title : str, optional
        Figure title.
    out_path : str or None, optional
        Where to save the figure. If None, the figure is not saved.
    """

    if hemisphere not in {"left", "right"}:
        raise ValueError("hemisphere must be 'left' or 'right'")

    if overlay is None:
        raise ValueError(f"No overlay provided for {hemisphere} hemisphere.")

    # ---------------------------------------------------------------------
    # Load fsaverage surfaces for the selected hemisphere

    fsaverage = fetch_surf_fsaverage(mesh="fsaverage")
    surf_inflated = fsaverage[f"infl_{hemisphere}"]
    surf_flat = fsaverage[f"flat_{hemisphere}"]
    bg_map = fsaverage[f"sulc_{hemisphere}"]

    # ---------------------------------------------------------------------

    fig = plt.figure(figsize=(10, 8))

    # Grid layout: 2 rows × 3 columns (top), 1 row × 3 columns (bottom)
    gs = gridspec.GridSpec(
        nrows=2, ncols=3, height_ratios=[1.0, 1.5], wspace=0.05, hspace=0.03
    )

    # -- Top row (3 subplots: lateral, medial, ventral) --
    ax_lateral = fig.add_subplot(gs[0, 0], projection="3d")
    ax_medial = fig.add_subplot(gs[0, 1], projection="3d")
    ax_ventral = fig.add_subplot(gs[0, 2], projection="3d")

    # -- Bottom row (single subplot spanning all columns: flat view) --
    ax_flat = fig.add_subplot(gs[1, :], projection="3d")

    # --------------------
    # Plot inflated views
    # --------------------
    plot_surf_stat_map(
        surf_mesh=surf_inflated,
        stat_map=overlay,
        bg_map=bg_map,
        hemi=hemisphere,
        view="lateral",
        colorbar=False,
        figure=fig,
        axes=ax_lateral,
        darkness=0.8,
    )

    plot_surf_stat_map(
        surf_mesh=surf_inflated,
        stat_map=overlay,
        bg_map=bg_map,
        hemi=hemisphere,
        view="medial",
        colorbar=False,
        figure=fig,
        axes=ax_medial,
        darkness=0.8,
    )

    plot_surf_stat_map(
        surf_mesh=surf_inflated,
        stat_map=overlay,
        bg_map=bg_map,
        hemi=hemisphere,
        view="ventral",
        colorbar=False,
        figure=fig,
        axes=ax_ventral,
        darkness=0.8,
    )

    # --------------------
    # Plot flat view (with colorbar)
    # --------------------
    plot_surf_stat_map(
        surf_mesh=surf_flat,
        stat_map=overlay,
        hemi=hemisphere,
        view="dorsal",
        colorbar=True,  # Only add colorbar here
        figure=fig,
        axes=ax_flat,
        darkness=0.7,
    )

    # ----------------------
    # Final adjustments
    # ----------------------
    fig.suptitle(title, fontsize=16, y=0.98)

    if out_path is not None:
        out_path = os.path.abspath(out_path)
        fig.savefig(out_path, dpi=300, bbox_inches="tight")

    plt.show()

    return fig, out_path


# --- Load Surface Data ---
# Update the file path as needed
surf_map = "/data/projects/chess/data/BIDS/derivatives/fmriprep-SPM-surf_smoothed-NO_GS-FD-HMP_brainmasked/fsaverage/fmriprep-SPM-fsaverage/GLM/sub-01/exp/beta_0005.gii"

# Load surface-based statistical data (GIFTI format)
stat_data = surf.load_surf_data(surf_map)

plot_fsaverage_hemisphere(
    stat_data[: len(stat_data) // 2],
    "left",
    title="Left Hemisphere",
    out_path="lh_plot.png",
)
plot_fsaverage_hemisphere(
    stat_data[len(stat_data) // 2 :],
    "right",
    title="Right Hemisphere",
    out_path="rh_plot.png",
)
