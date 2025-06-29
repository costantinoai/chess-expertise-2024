#!/usr/bin/env python3
"""Plotting helpers for manifold analysis."""

from __future__ import annotations

import os
from typing import Iterable

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

from . import logger

sns.set_style("whitegrid")


def plot_subject_roi(
    radius: float, dimension: float, subject: str, roi: int, out_dir: str
) -> None:
    """Save a simple scatter plot for a subject/ROI."""
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.scatter(dimension, radius, color="tab:blue")
    ax.set_xlabel("Dimension")
    ax.set_ylabel("Radius")
    ax.set_title(f"Sub {subject} ROI {roi}")
    os.makedirs(out_dir, exist_ok=True)
    fig.savefig(os.path.join(out_dir, f"sub-{subject}_roi-{roi}.png"), dpi=300)
    plt.close(fig)


def plot_group_comparison(df: pd.DataFrame, out_dir: str, metric: str) -> None:
    """Plot boxplots comparing groups for a metric across ROIs."""
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.boxplot(data=df, x="ROI", y=metric, hue="group", ax=ax)
    ax.set_title(f"Group comparison: {metric}")
    plt.xticks(rotation=90)
    os.makedirs(out_dir, exist_ok=True)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"group_{metric}.png"), dpi=300)
    plt.close(fig)


def plot_group_heatmap(values: np.ndarray, rois: Iterable[int], metric: str, out_dir: str) -> None:
    """Plot heatmap of mean metric per ROI for experts and nonexperts."""
    fig, ax = plt.subplots(figsize=(6, max(4, len(rois) * 0.25)))
    sns.heatmap(
        values,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        yticklabels=rois,
        xticklabels=["expert", "nonexpert"],
        ax=ax,
    )
    ax.set_xlabel("Group")
    ax.set_ylabel("ROI")
    ax.set_title(metric.capitalize())
    os.makedirs(out_dir, exist_ok=True)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"heatmap_{metric}.png"), dpi=300)
    plt.close(fig)
