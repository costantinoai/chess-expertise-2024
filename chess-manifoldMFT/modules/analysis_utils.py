#!/usr/bin/env python3
"""Core manifold analysis functions."""

from typing import Union, List, Tuple, Optional
import numpy as np
import pandas as pd
from scipy.stats import ttest_ind
from statsmodels.stats.multitest import multipletests

from . import logger

try:
    from mftma.manifold_analysis_correlation import (
        manifold_analysis_corr,
    )
except Exception:  # pragma: no cover - package may not be installed in tests
    manifold_analysis_corr = None
    logger.warning("neural_manifolds_replicaMFT package not available.")


def compute_manifold(
    data: Optional[np.ndarray] = None,
    labels: Optional[np.ndarray] = None,
    *,
    manifolds: Optional[List[np.ndarray]] = None,
    kappa: float = 0,
    n_t: int = 200,
) -> Tuple[float, float, float]:
    """Run manifold analysis and return mean capacity, radius and dimension.

    Parameters
    ----------
    data : ndarray, optional
        2D array of shape ``(n_obs, n_features)``. ``labels`` must also be
        provided in this case.
    labels : ndarray, optional
        Integer labels assigning each observation in ``data`` to a manifold.
    manifolds : list of ndarray, optional
        Precomputed list where each element has shape ``(n_features, n_samples)``
        representing a single manifold. If provided, ``data`` and ``labels`` are
        ignored.
    kappa : float, optional
        Margin parameter passed to :func:`manifold_analysis_corr`.
    n_t : int, optional
        Number of Gaussian vectors used in the analysis.
    """

    if manifolds is None:
        if data is None or labels is None:
            raise ValueError("data and labels must be provided when manifolds is None")

        # Drop NaNs  
        valid_mask = ~np.isnan(data).any(axis=0)  # type: ignore
        data = data[:, valid_mask]
        # Drop zero-variance features
        var = np.var(data, axis=0)
        data = data[:, var > 0]  # type: ignore
        if data.size == 0 or manifold_analysis_corr is None:  # type: ignore
            return np.nan, np.nan, np.nan

        manifolds = []
        for lab in np.unique(labels):
            mask = labels == lab
            # transpose so voxels/features are rows as expected by the library
            manifolds.append(data[mask].T)  # type: ignore
    else:
        if len(manifolds) == 0 or manifold_analysis_corr is None:
            return np.nan, np.nan, np.nan

        # assume all manifolds share the same feature dimension
        n_feat = manifolds[0].shape[0]
        valid_mask = np.ones(n_feat, dtype=bool)
        for m in manifolds:
            if m.shape[0] != n_feat:
                raise ValueError("All manifolds must have the same number of features")
            valid_mask &= ~np.isnan(m).any(axis=1)  # type: ignore
            valid_mask &= np.var(m, axis=1) > 0
        manifolds = [m[valid_mask] for m in manifolds]
        if manifolds[0].size == 0:
            return np.nan, np.nan, np.nan

    a_vec, r_vec, d_vec, _, _ = manifold_analysis_corr(manifolds, kappa, n_t)

    capacity = float(1 / np.mean(1 / np.asarray(a_vec)))  # type: ignore
    radius = float(np.mean(r_vec))
    dimension = float(np.mean(d_vec))
    return capacity, radius, dimension


def fdr_ttest(
    group1: np.ndarray, group2: np.ndarray, labels: np.ndarray, alpha: float = 0.05
) -> "pd.DataFrame":
    """Return DataFrame with t-test and FDR correction across ROIs."""
    import pandas as pd

    n_rois = group1.shape[1]
    tvals = np.zeros(n_rois)
    pvals = np.ones(n_rois)
    for i in range(n_rois):
        g1 = group1[:, i]
        g2 = group2[:, i]
        g1 = g1[~np.isnan(g1)]  # type: ignore
        g2 = g2[~np.isnan(g2)]  # type: ignore
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
