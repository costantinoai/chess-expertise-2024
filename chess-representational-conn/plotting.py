# -*- coding: utf-8 -*-
"""Plotting helpers for representational connectivity matrices."""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from modules import MANAGER


def plot_connectivity_matrix(matrix: np.ndarray,
                              title: str,
                              out_path: str,
                              mask: np.ndarray | None = None,
                              vmin: float = -1.0,
                              vmax: float = 1.0,
                              cmap: str = "coolwarm"):
    """Plot a square connectivity matrix with ROI labels."""
    roi_names = [r.region_long_name for r in MANAGER.rois]
    order = np.argsort([r.cortex_id for r in MANAGER.rois])
    matrix = matrix[order][:, order]
    if mask is not None:
        mask = mask[order][:, order]
        plot_mat = np.where(mask, matrix, np.nan)
    else:
        plot_mat = matrix

    plt.figure(figsize=(12, 10))
    sns.heatmap(plot_mat, vmin=vmin, vmax=vmax, cmap=cmap,
                square=True, cbar=True,
                xticklabels=np.array(roi_names)[order],
                yticklabels=np.array(roi_names)[order])
    plt.xticks(rotation=90)
    plt.title(title)
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=300)
    plt.close()

