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


def compute_zmap_correlations(z_map, term_maps, ref_img, n_boot=10000, fdr_alpha=0.05):
    """
    Compute Pearson correlations (with bootstrap confidence intervals and FDR correction)
    between a z-map and a set of term-maps.

    Parameters
    ----------
    z_map : numpy.ndarray
        3D array of z-values for each voxel.
    term_maps : dict
        Mapping term name -> filepath for its NIfTI image.
    ref_img : nibabel image
        Reference image for resampling term maps.
    n_boot : int, optional
        Number of bootstrap samples for CI estimation (default=10000).
    fdr_alpha : float, optional
        Alpha level for FDR correction (default=0.05).

    Returns
    -------
    pandas.DataFrame
        DataFrame with columns ['Term', 'r', 'CI_low', 'CI_high', 'p_raw', 'p_fdr', 'sig'],
        indexed in the fixed TERM_ORDER.
    """
    import pandas as pd
    from nilearn import image

    records = []
    # flatten the z_map once
    flat_z = z_map.ravel()

    for term, path in term_maps.items():
        # resample term image to match reference
        tpl_img = image.resample_to_img(image.load_img(path), ref_img,
                                        force_resample=True, copy_header=True)
        flat_t = tpl_img.get_fdata().ravel()

        # stack and remove non-finite or constant voxels
        stacked = np.vstack([flat_z, flat_t]).T
        # mask out rows where either is nan/inf or where variance is zero
        finite_mask = np.all(np.isfinite(stacked), axis=1)
        nonconst_mask = (stacked.max(axis=1) - stacked.min(axis=1)) > 0
        good_mask = finite_mask & nonconst_mask
        x = stacked[good_mask, 0]
        y = stacked[good_mask, 1]

        # compute bootstrap Pearson correlation via pingouin
        res = pg.corr(x=x, y=y, method='pearson', bootstraps=n_boot,
                      confidence=0.95, method_ci='percentile')
        r = res['r'].iloc[0]
        p_raw = res['p-val'].iloc[0]
        ci_low, ci_high = res['CI95%'].iloc[0]

        records.append((term, r, ci_low, ci_high, p_raw))

    # assemble DataFrame
    df = pd.DataFrame(records, columns=['Term', 'r', 'CI_low', 'CI_high', 'p_raw'])
    # enforce fixed order
    df['Term'] = pd.Categorical(df['Term'], categories=TERM_ORDER, ordered=True)
    df = df.sort_values('Term').reset_index(drop=True)

    # FDR correction across terms
    rej, p_fdr = fdrcorrection(df['p_raw'], alpha=fdr_alpha)
    df['p_fdr'] = p_fdr
    df['sig'] = rej

    return df

def compute_difference_stats(z_pos: np.ndarray,
                             z_neg: np.ndarray,
                             term_maps: dict,
                             ref_img,
                             n_boot: int = 10000,
                             fdr_alpha: float = 0.05) -> pd.DataFrame:
    """
    Compute bootstrap-based statistical comparison of correlations between
    z_pos vs term maps and z_neg vs term maps.

    Returns a DataFrame with columns:
      ['Term', 'r_pos', 'r_neg', 'r_diff', 'CI_low', 'CI_high', 'p_raw', 'p_fdr', 'sig']
    """
    records = []
    flat_zpos = z_pos.ravel()
    flat_zneg = z_neg.ravel()

    for term, path in term_maps.items():
        # load & resample
        tpl = image.resample_to_img(
            image.load_img(path), ref_img,
            force_resample=True, copy_header=True
        )
        flat_t = tpl.get_fdata().ravel()

        # mask
        mask = np.isfinite(flat_zpos) & np.isfinite(flat_zneg) & np.isfinite(flat_t)
        x = flat_zpos[mask]
        y = flat_zneg[mask]
        t = flat_t[mask]

        # observed
        r_pos = np.corrcoef(x, t)[0,1]
        r_neg = np.corrcoef(y, t)[0,1]
        diff_obs = r_pos - r_neg

        # bootstrap diffs
        diffs = np.empty(n_boot)
        n = len(x)
        for i in range(n_boot):
            idx = np.random.randint(0, n, n)
            diffs[i] = np.corrcoef(x[idx], t[idx])[0,1] - np.corrcoef(y[idx], t[idx])[0,1]

        ci_low, ci_high = np.percentile(diffs, [2.5, 97.5])
        p_raw = np.mean(np.abs(diffs) >= np.abs(diff_obs))

        records.append((term, r_pos, r_neg, diff_obs, ci_low, ci_high, p_raw))

    df = pd.DataFrame(
        records,
        columns=['Term','r_pos','r_neg','r_diff','CI_low','CI_high','p_raw']
    )
    df['Term'] = pd.Categorical(df['Term'], categories=TERM_ORDER, ordered=True)
    df = df.sort_values('Term').reset_index(drop=True)

    rej, p_fdr = fdrcorrection(df['p_raw'], alpha=fdr_alpha)
    df['p_fdr'] = p_fdr
    df['sig'] = rej

    return df
