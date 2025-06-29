#!/usr/bin/env python3
"""Core manifold analysis functions."""

from __future__ import annotations

import numpy as np
from joblib import Parallel, delayed
from scipy.stats import ttest_ind
from statsmodels.stats.multitest import multipletests

from . import logger

try:
    from neural_manifolds_replicaMFT.manifold_analysis_correlation import (
        manifold_analysis_corr,
    )
except Exception:  # pragma: no cover - package may not be installed in tests
    manifold_analysis_corr = None
    logger.warning("neural_manifolds_replicaMFT package not available.")


def compute_manifold(data: np.ndarray, labels: np.ndarray) -> tuple[float, float]:
    """Run manifold analysis and return radius and dimension."""
    # Drop NaNs
    valid_mask = ~np.isnan(data).any(axis=0)
    data = data[:, valid_mask]
    # Drop zero-variance features
    var = np.var(data, axis=0)
    data = data[:, var > 0]
    if data.size == 0 or manifold_analysis_corr is None:
        return np.nan, np.nan
    res = manifold_analysis_corr(data, labels)
    radius = float(res.get("radius", np.nan))
    dimension = float(res.get("dimension", np.nan))
    return radius, dimension


def fdr_ttest(
    group1: np.ndarray, group2: np.ndarray, labels: np.ndarray, alpha: float = 0.05
):
    """Return DataFrame with t-test and FDR correction across ROIs."""
    import pandas as pd

    n_rois = group1.shape[1]
    tvals = np.zeros(n_rois)
    pvals = np.ones(n_rois)
    for i in range(n_rois):
        g1 = group1[:, i]
        g2 = group2[:, i]
        g1 = g1[~np.isnan(g1)]
        g2 = g2[~np.isnan(g2)]
        if len(g1) < 2 or len(g2) < 2:
            continue
        t_stat, p_val = ttest_ind(np.sort(g1), np.sort(g2), nan_policy="omit")
        tvals[i] = t_stat
        pvals[i] = p_val
    reject, p_fdr, _, _ = multipletests(pvals, alpha=alpha, method="fdr_bh")
    df = pd.DataFrame(
        {
            "ROI": labels,
            "t_stat": tvals,
            "p_val": pvals,
            "p_fdr": p_fdr,
            "significant": reject,
        }
    )
    return df
