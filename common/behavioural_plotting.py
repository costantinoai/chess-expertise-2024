#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Shared behavioural plotting utilities (RDM heatmaps, colorbar)."""
from __future__ import annotations

from typing import Sequence
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Rectangle


def plot_shared_colorbar(cmap, vmin=-18, vmax=18, out_dir: str | None = None, filename: str = "symmetric_colorbar.png"):
    """Render a standalone colorbar image with given colormap and limits."""
    fig, ax = plt.subplots(figsize=(6, 1))
    fig.patch.set_facecolor('white')
    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    cbar = plt.colorbar(
        plt.cm.ScalarMappable(norm=norm, cmap=cmap),
        cax=plt.axes([0.1, 0.5, 0.8, 0.15]), orientation='horizontal'
    )
    cbar.set_label("Dissimilarity", fontsize=14)
    cbar.ax.tick_params(labelsize=12)
    ax.set_axis_off()
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        save_path = os.path.join(out_dir, filename)
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0.1, facecolor='white')
    plt.show()


def plot_rdm_heatmap(
    rdm: np.ndarray,
    bold_title: str,
    expertise_label: str,
    strategies: Sequence[int] | Sequence[str],
    strat_colors: Sequence[str],
    strat_alphas: Sequence[float],
    colormap: str = "RdPu",
    vmin: float = 0,
    vmax: float = 18,
):
    """Plot an RDM as a heatmap with strategy group bars along axes.

    The strategy bars are derived from `strategies`, `strat_colors`, and `strat_alphas`.
    """
    fig, ax = plt.subplots(figsize=(10, 8), facecolor='white')
    sns.heatmap(rdm, annot=False, fmt="d", cmap=colormap, vmin=vmin, vmax=vmax, cbar=False, ax=ax, square=True)
    ax.set_aspect('equal')

    ticks = []
    prev = None
    for i, lab in enumerate(strategies):
        if lab != prev:
            ticks.append(i)
            prev = lab
    ticks.append(len(strategies))

    for idx, start in enumerate(ticks[:-1]):
        end = ticks[idx + 1]
        width = end - start
        color = strat_colors[start]
        alpha = strat_alphas[start]
        ax.add_patch(Rectangle((start, -0.01), width, -0.0005 * len(rdm), color=color, alpha=alpha, ec=None, transform=ax.get_xaxis_transform(), clip_on=False))
        ax.add_patch(Rectangle((-0.01, start), -0.0005 * len(rdm), width, color=color, alpha=alpha, ec=None, transform=ax.get_yaxis_transform(), clip_on=False))

    title_text = f"{bold_title}\n{expertise_label}"
    ax.set_title(title_text, fontsize=24, pad=25)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.tight_layout()
    plt.show()

