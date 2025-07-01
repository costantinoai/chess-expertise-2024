#!/usr/bin/env python3
"""Core manifold analysis functions."""

from typing import Union, List, Tuple, Optional
import numpy as np
import pandas as pd
from scipy.stats import ttest_ind
from statsmodels.stats.multitest import multipletests

from . import logger

from mftma.manifold_analysis_correlation import manifold_analysis_corr
from mftma.alldata_dimension_analysis import alldata_dimension_analysis

def compute_manifold(
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


    # assume all manifolds share the same feature dimension
    n_feat = manifolds[0].shape[0]
    valid_mask = np.ones(n_feat, dtype=bool)
    for m in manifolds:
        if m.shape[0] != n_feat:
            raise ValueError("All manifolds must have the same number of features")
        valid_mask &= ~np.isnan(m).any(axis=1)  # type: ignore
        # valid_mask &= np.var(m, axis=1) > 0 # FIXME: this does not work for single vectors!
    manifolds_clean = [m[valid_mask] for m in manifolds]
    if manifolds_clean[0].size == 0:
        return np.nan, np.nan, np.nan

    # Perform Manifold Analysis
    # Core manifold analysis
    a, r, d, rho0, K = manifold_analysis_corr(manifolds_clean, kappa, n_t)
    # Additional dimension analyses
    D_pr, D_ev, D_feat = alldata_dimension_analysis(manifolds_clean, perc=0.9)

    # Aggregate results
    manifold_results = {
        "alpha_M": 1.0 / np.mean(1.0 / a),
        "R_M": np.mean(r),
        "D_M": np.mean(d),
        "rho_center": rho0,
        "D_participation_ratio": D_pr,
        "D_explained_variance": D_ev,
        "D_feature": D_feat,
    }


    return manifold_results

def build_metric_array(df, subjects, rois, metric):
    import numpy as np
    arr = np.full((len(subjects), len(rois)), np.nan)
    for i, subj in enumerate(subjects):
        for j, roi in enumerate(rois):
            val = df[(df["subject"]==subj) & (df["roi"]==roi)][metric]
            if not val.empty:
                arr[i, j] = val.values[0]
    return arr

# --- Helper: FDR t-test function with full stats (moved from main_noavg.py) ---
def fdr_ttest(group1_vals, group2_vals, roi_labels, alpha=0.05):
    import numpy as np
    import pandas as pd
    from statsmodels.stats.multitest import multipletests
    from scipy.stats import ttest_ind, t
    results = []
    for i, roi in enumerate(roi_labels):
        g1 = group1_vals[:, i]
        g2 = group2_vals[:, i]
        g1 = g1[~np.isnan(g1)]
        g2 = g2[~np.isnan(g2)]
        if len(g1) < 2 or len(g2) < 2:
            results.append({
                "ROI_Label": roi, "t_stat": np.nan, "p_val": np.nan, "dof": np.nan,
                "cohen_d": np.nan, "mean_diff": np.nan, "ci95_low": np.nan, "ci95_high": np.nan
            })
            continue
        res = ttest_ind(g1, g2, equal_var=False)
        diff = np.mean(g1) - np.mean(g2)
        se = np.sqrt(np.var(g1, ddof=1)/len(g1) + np.var(g2, ddof=1)/len(g2))
        dof = (np.var(g1, ddof=1)/len(g1) + np.var(g2, ddof=1)/len(g2))**2 / (
            ((np.var(g1, ddof=1)/len(g1))**2)/(len(g1)-1) + ((np.var(g2, ddof=1)/len(g2))**2)/(len(g2)-1)
        )
        ci = t.interval(0.95, dof, loc=diff, scale=se)
        pooled_sd = np.sqrt(((len(g1)-1)*np.var(g1, ddof=1) + (len(g2)-1)*np.var(g2, ddof=1)) / (len(g1)+len(g2)-2))
        d = diff / pooled_sd if pooled_sd > 0 else np.nan
        results.append({
            "ROI_Label": roi, "t_stat": res.statistic, "p_val": res.pvalue, "dof": dof,
            "cohen_d": d, "mean_diff": diff, "ci95_low": ci[0], "ci95_high": ci[1]
        })
    df = pd.DataFrame(results)
    corrected_p = df["p_val"].copy()
    corrected_p[df["p_val"].isna()] = 1.0
    reject, pval_fdr, _, _ = multipletests(corrected_p, alpha=alpha, method='fdr_bh')
    df["p_val_fdr"] = pval_fdr
    df["significant_fdr"] = reject
    df["significant"] = df["p_val"] < 0.05
    return df
