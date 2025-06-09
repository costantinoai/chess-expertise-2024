#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun  7 22:18:33 2025

@author: costantino_ai
"""

import numpy as np
from scipy.stats import t, norm
from statsmodels.stats.multitest import fdrcorrection
import pingouin as pg
from modules.config import TERM_ORDER
import pandas as pd
from nilearn import image

def split_and_convert_t_to_z(t_map: np.ndarray, dof: int):
    """
    Take a 3D t-map and produce two 3D z-maps:
      - pos_z is the one‐tailed z for t>0
      - neg_z is the one‐tailed z for t<0, with all values flipped positive

    Any voxels ≤0 in the positive map (or ≤0 in the neg map after flip) become z=0.
    """
    # 1) split t into directional components
    t_pos = np.where(t_map > 0, t_map, 0.0)      # positive t’s only
    t_neg = np.where(t_map < 0, -t_map, 0.0)     # flipped absolute negative t’s

    # 2) compute one‐tailed p = P(T > t)
    p_pos = t.sf(t_pos, df=dof)
    p_neg = t.sf(t_neg, df=dof)

    # 3) invert to z so that P(Z > z) = p
    #    norm.isf(p) == the positive z such that upper‐tail area = p
    z_pos = norm.isf(p_pos)
    z_neg = norm.isf(p_neg)

    # 4) optional: force zero where t was zero, to avoid tiny numerical z’s
    z_pos[t_pos == 0] = 0.0
    z_neg[t_neg == 0] = 0.0

    return z_pos, z_neg


def remove_useless_data(data: np.ndarray, dim: int = 2):
    """
    Remove "useless" rows or columns from a 2D array by:
      1) Removing any slice containing NaN or Inf,
      2) Removing any slice that is entirely zeros.

    Parameters
    ----------
    data : np.ndarray, shape (n_samples, n_features)
        2D array to be cleaned.
    dim : int, {1, 2}, default=2
        If dim == 2, operate on features (columns): drop columns
        with any NaN/Inf or all zeros.
        If dim == 1, operate on samples (rows): drop rows
        with any NaN/Inf or all zeros.

    Returns
    -------
    data_clean : np.ndarray
        The filtered array, preserving the other axis.
    keep_mask : np.ndarray of bools
        Boolean mask for retained rows (if dim==1) or columns (if dim==2).
    """
    if data.ndim != 2:
        raise ValueError("remove_useless_data expects a 2D array.")
    if dim not in (1, 2):
        raise ValueError("dim must be 1 (rows) or 2 (columns).")

    # Determine axis: axis=0 for columns, axis=1 for rows
    axis = 1 if dim == 1 else 0

    # 1) Mask out any slice with NaN or Inf
    finite_mask = np.all(np.isfinite(data), axis=axis)

    # 2) Mask out any slice that is entirely zeros
    if dim == 2:
        zero_mask = np.all(data == 0, axis=0)
    else:
        zero_mask = np.all(data == 0, axis=1)

    # Combine: keep finite & not all-zero
    keep_mask = finite_mask & (~zero_mask)

    # Apply mask
    if dim == 2:
        data_clean = data[:, keep_mask]
    else:
        data_clean = data[keep_mask, :]

    return data_clean, keep_mask


def compute_all_zmap_correlations(z_pos, z_neg, term_maps, ref_img,
                                  n_boot=10000, fdr_alpha=0.05,
                                  ci_alpha=0.05, random_state=None):
    """
    Compute correlation statistics for positive and negative z-maps against
    term-maps and estimate their difference using a bootstrap approach.

    Parameters
    ----------
    z_pos, z_neg : np.ndarray
        Positive and negative z-maps to correlate with the term maps.
    term_maps : dict
        Mapping from term name to NIfTI file path.
    ref_img : nibabel.Nifti1Image
        Reference image (for resampling term maps).
    n_boot : int, optional
        Number of bootstrap samples for the individual correlations.
    fdr_alpha : float, optional
        Alpha level for FDR-correction of the p-values.
    ci_alpha : float, optional
        Alpha level for confidence intervals on the difference of correlations.
    random_state : int or None, optional
        Seed for the random number generator used in bootstrapping.

    Returns:
    --------
    df_pos : pd.DataFrame
        Correlation stats for the positive z-map.
    df_neg : pd.DataFrame
        Correlation stats for the negative z-map.
    df_diff : pd.DataFrame
        Difference stats between positive and negative z-maps.
    """
    records_pos = []
    records_neg = []
    records_diff = []

    rng = np.random.default_rng(random_state)

    flat_pos = z_pos.ravel()
    flat_neg = z_neg.ravel()

    for term, path in term_maps.items():
        tpl = image.resample_to_img(image.load_img(path), ref_img,
                                    force_resample=True, copy_header=True)
        flat_t = tpl.get_fdata().ravel()

        # Valid voxel mask
        stacked = np.vstack([flat_pos, flat_neg, flat_t]).T
        ok = np.all(np.isfinite(stacked), axis=1)
        ok &= (stacked.max(axis=1) - stacked.min(axis=1)) > 0
        x, y, t = stacked[ok, 0], stacked[ok, 1], stacked[ok, 2]

        # --- POSITIVE MAP ---
        res_pos = pg.corr(x=t, y=x, method='pearson',
                          bootstraps=n_boot, confidence=0.95,
                          method_ci='percentile', alternative='two-sided')
        r_pos = res_pos['r'].iloc[0]
        p_pos = res_pos['p-val'].iloc[0]
        ci_lo_pos, ci_hi_pos = res_pos['CI95%'].iloc[0]
        records_pos.append((term, r_pos, ci_lo_pos, ci_hi_pos, p_pos))

        # --- NEGATIVE MAP ---
        res_neg = pg.corr(x=t, y=y, method='pearson',
                          bootstraps=n_boot, confidence=0.95,
                          method_ci='percentile', alternative='two-sided')
        r_neg = res_neg['r'].iloc[0]
        p_neg = res_neg['p-val'].iloc[0]
        ci_lo_neg, ci_hi_neg = res_neg['CI95%'].iloc[0]
        records_neg.append((term, r_neg, ci_lo_neg, ci_hi_neg, p_neg))

        # --- DIFFERENCE (bootstrap) ---
        n = len(x)
        boot_diffs = np.empty(n_boot)
        for bi in range(n_boot):
            idx = rng.integers(0, n, size=n)
            r_pos_b = np.corrcoef(t[idx], x[idx])[0, 1]
            r_neg_b = np.corrcoef(t[idx], y[idx])[0, 1]
            boot_diffs[bi] = r_pos_b - r_neg_b

        boot_diffs.sort()
        lo_r = np.percentile(boot_diffs, 100 * ci_alpha / 2)
        hi_r = np.percentile(boot_diffs, 100 * (1 - ci_alpha / 2))

        diff_obs = r_pos - r_neg
        tail_low = np.mean(boot_diffs <= 0)
        tail_high = np.mean(boot_diffs >= 0)
        p_diff = 2 * min(tail_low, tail_high)

        records_diff.append((term, r_pos, r_neg, diff_obs, lo_r, hi_r, p_diff))

    # Assemble DataFrames
    df_pos = pd.DataFrame(records_pos, columns=['term', 'r', 'CI_low', 'CI_high', 'p_raw'])
    df_neg = pd.DataFrame(records_neg, columns=['term', 'r', 'CI_low', 'CI_high', 'p_raw'])
    df_diff = pd.DataFrame(records_diff,
                           columns=['term', 'r_pos', 'r_neg', 'r_diff', 'CI_low', 'CI_high', 'p_raw'])

    # FDR correction
    for df in [df_pos, df_neg, df_diff]:
        rej, p_fdr = fdrcorrection(df['p_raw'], alpha=fdr_alpha)
        df['p_fdr'] = p_fdr
        df['sig'] = rej

    return df_pos, df_neg, df_diff

import os
def save_latex_correlation_tables(df_pos, df_neg, diff_df, run_id, out_dir):
    """
    Save and print LaTeX tables for positive/negative/difference z-map correlations.

    Parameters
    ----------
    df_pos : pd.DataFrame
        Correlations and stats from positive z-map.
    df_neg : pd.DataFrame
        Correlations and stats from negative z-map.
    diff_df : pd.DataFrame
        Difference in correlations between positive and negative z-maps.
    run_id : str
        Identifier for the run (used in titles and filenames).
    out_dir : str
        Directory to save LaTeX tables.
    """
    os.makedirs(out_dir, exist_ok=True)

    def format_and_save(df, columns, caption, label, fname):
        latex_str = (
            df[columns]
            .round(3)
            .sort_values(by=columns[1], ascending=False)
            .to_latex(index=False, escape=True, caption=caption, label=label)
        )
        out_path = os.path.join(out_dir, fname)
        with open(out_path, "w") as f:
            f.write(latex_str)
        print(f"\n=== {caption} ===\n")
        print(latex_str)

    # Table 1: Positive z-map correlations
    format_and_save(
        df_pos,
        columns=['term', 'r', 'CI_low', 'CI_high', 'p_fdr'],
        caption=f"{run_id} — Positive z-map: correlations with each term map.",
        label=f"tab:{run_id}_pos",
        fname=f"{run_id}_positive_zmap.tex"
    )

    # Table 2: Negative z-map correlations
    format_and_save(
        df_neg,
        columns=['term', 'r', 'CI_low', 'CI_high', 'p_fdr'],
        caption=f"{run_id} — Negative z-map: correlations with each term map.",
        label=f"tab:{run_id}_neg",
        fname=f"{run_id}_negative_zmap.tex"
    )

    # Table 3: Differences
    format_and_save(
        diff_df,
        columns=['term', 'r_diff', 'CI_low', 'CI_high', 'p_fdr'],
        caption=f"{run_id} — Difference in correlations (positive - negative).",
        label=f"tab:{run_id}_diff",
        fname=f"{run_id}_difference_zmap.tex"
    )
