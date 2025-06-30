#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Statistical helper functions for correlation analysis.

This module contains utilities to convert T-maps into one-tailed Z-maps,
clean voxel data, compute correlation coefficients between brain maps and
Neurosynth term maps and estimate the difference between correlations via
bootstrap resampling.
"""

import numpy as np
from scipy.stats import t, norm
from statsmodels.stats.multitest import fdrcorrection
import pingouin as pg
import pandas as pd
from nilearn import image
from joblib import Parallel, delayed
from tqdm import tqdm
import os

def get_brain_mask(ref):
    """
    Loads the ICBM152 GM brain mask and resamples it to match the reference image.
    Binarizes the mask such that voxels > 0.25 are True.

    Parameters
    ----------
    ref : Nifti1Image
        The reference image to which the mask should be resampled.

    Returns
    -------
    binary_mask : Nifti1Image
        A binarized brain mask in the same space as `ref`.
    """
    from nilearn import datasets, image
    from nilearn.image import math_img
    import nibabel as nib

    # Fetch and load the ICBM152 GM mask
    mni = datasets.fetch_icbm152_2009()
    brain_mask = nib.load(mni['gm'])

    # Resample to the reference image
    resampled_mask = image.resample_to_img(brain_mask, ref, interpolation='nearest')

    # Threshold and binarize the mask
    binary_mask = math_img('img > 0.25', img=resampled_mask)
    return binary_mask

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


# def remove_useless_data(data: np.ndarray, brain_mask_flat=None):
#     """Remove problematic voxels from stacked maps.

#     Parameters
#     ----------
#     data : np.ndarray, shape (n_maps, n_voxels)
#         Array containing different maps stacked by rows.

#     Returns
#     -------
#     data_clean : np.ndarray
#         Data with unusable voxels removed.
#     keep_mask : np.ndarray of bool
#         Boolean mask indicating voxels that were kept.
#     """
#     if data.ndim != 2:
#         raise ValueError("remove_useless_data expects a 2D array")

#     finite_mask = np.all(np.isfinite(data), axis=0)
#     const_mask = np.ptp(np.round(data, 2), axis=0) == 0
#     keep_mask = finite_mask & (~const_mask) if brain_mask_flat == None else finite_mask & (~const_mask) & brain_mask_flat
#     return data[:, keep_mask], keep_mask

def remove_useless_data(data: np.ndarray, brain_mask_flat: np.ndarray = None):
    """Remove problematic voxels from stacked maps.

    Parameters
    ----------
    data : np.ndarray, shape (n_maps, n_voxels)
        Array containing different maps stacked by rows.
    brain_mask_flat : np.ndarray of bool, optional
        1D boolean brain mask to restrict analysis to valid brain voxels.

    Returns
    -------
    data_clean : np.ndarray
        Data with unusable voxels removed.
    keep_mask : np.ndarray of bool
        Boolean mask indicating voxels that were kept.
    """
    if data.ndim != 2:
        raise ValueError("remove_useless_data expects a 2D array")

    n_voxels = data.shape[1]
    print(f"Initial number of voxels: {n_voxels}")

    # Mask: finite values across all maps
    finite_mask = np.all(np.isfinite(data), axis=0)
    print(f"Voxels with all finite values: {np.sum(finite_mask)} "
          f"({n_voxels - np.sum(finite_mask)} removed)")

    # Mask: variance > 0 across maps (remove flat voxels)
    variance = np.var(data, axis=0)
    var_thresh = 1e-5
    low_variance_mask = variance < var_thresh
    print(f"Voxels with variance >= {var_thresh}: {np.sum(~low_variance_mask)} "
          f"({np.sum(low_variance_mask)} removed)")

    # Combine masks
    keep_mask = finite_mask & (~low_variance_mask)

    # Apply brain mask if provided
    if brain_mask_flat is not None:
        if brain_mask_flat.shape[0] != data.shape[1]:
            raise ValueError("brain_mask_flat must have shape (n_voxels,)")
        brain_mask_flat = brain_mask_flat.astype(bool)
        print(f"Voxels in brain mask: {np.sum(brain_mask_flat)} "
              f"({n_voxels - np.sum(brain_mask_flat)} excluded outside brain)")
        keep_mask &= brain_mask_flat

    # Final count
    print(f"Final number of voxels retained: {np.sum(keep_mask)} "
          f"({n_voxels - np.sum(keep_mask)} total removed)")

    return data[:, keep_mask], keep_mask



def compute_all_zmap_correlations(z_pos, z_neg, term_maps, ref_img,
                                  n_boot=10000, fdr_alpha=0.05,
                                  ci_alpha=0.05, random_state=42,
                                  n_jobs=1):
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
    n_jobs : int, optional
        Number of parallel jobs for bootstrap resampling of the correlation
        difference. ``-1`` uses all available cores.

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
    flat_mask = get_brain_mask(ref_img).get_fdata().ravel()

    for term, path in term_maps.items():
        # Resample the term map onto the same grid as the subject map so that
        # voxel-wise operations are valid.
        tpl = image.resample_to_img(image.load_img(path), ref_img,
                                    force_resample=True, copy_header=True)
        flat_t = tpl.get_fdata().ravel()

        # --------------------------------------------------------------
        # Mask out any voxels that are non-finite or identical across all
        # three maps.  The latter effectively removes regions that carry no
        # variance and would otherwise bias the correlation.
        # --------------------------------------------------------------
        stacked, _ = remove_useless_data(np.vstack([flat_pos, flat_neg, flat_t]), flat_mask)
        # x, y, t = stacked

        x, y, t = flat_pos, flat_neg, flat_t

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
        try:
            ci_lo_neg, ci_hi_neg = res_neg['CI95%'].iloc[0]
        except:
            ci_lo_neg, ci_hi_neg = np.nan, np.nan

        records_neg.append((term, r_neg, ci_lo_neg, ci_hi_neg, p_neg))

        # --- DIFFERENCE (bootstrap) ---
        n = len(x)

        # The difference in correlations does not have a simple analytic
        # standard error when the correlations are dependent.  We therefore
        # estimate its sampling distribution by bootstrap resampling the voxels.

        def _boot_one(seed):
            """Draw one bootstrap sample and compute the correlation difference."""
            sub_rng = np.random.default_rng(seed)
            idx = sub_rng.integers(0, n, size=n)
            r_pos_b = np.corrcoef(t[idx], x[idx])[0, 1]
            r_neg_b = np.corrcoef(t[idx], y[idx])[0, 1]
            return (r_pos_b, r_neg_b, r_pos_b - r_neg_b)

        # Draw ``n_boot`` bootstrap samples. Multiprocessing is used when
        # ``n_jobs`` is not 1 to speed up the resampling step.
        seeds = rng.integers(0, 2**32 - 1, size=n_boot)
        if n_jobs == 1:
            res = np.array([_boot_one(s) for s in tqdm(seeds, desc="Bootstrapping")])
        else:
            res = np.array(
                Parallel(n_jobs=n_jobs)(
                    delayed(_boot_one)(s) for s in tqdm(seeds, desc="Bootstrapping")
                )
            )
        _, _, boot_diffs = res[:, 0],res[:, 1],res[:, 2]
        boot_diffs.sort()
        # Percentile-based confidence interval of the difference
        lo_r = np.percentile(boot_diffs, 100 * ci_alpha / 2)
        hi_r = np.percentile(boot_diffs, 100 * (1 - ci_alpha / 2))

        diff_obs = np.mean(boot_diffs)

        # Two-sided p-value: probability that the bootstrap difference crosses 0
        tail_low = np.mean(boot_diffs <= 0)
        tail_high = np.mean(boot_diffs >= 0)
        p_diff = 2 * min(tail_low, tail_high)

        records_diff.append((term, r_pos, r_neg, diff_obs, lo_r, hi_r, p_diff))

    # Assemble DataFrames
    df_pos = pd.DataFrame(records_pos, columns=['term', 'r', 'CI_low', 'CI_high', 'p_raw'])
    df_neg = pd.DataFrame(records_neg, columns=['term', 'r', 'CI_low', 'CI_high', 'p_raw'])
    df_diff = pd.DataFrame(records_diff,
                            columns=['term', 'r_pos', 'r_neg', 'r_diff', 'CI_low', 'CI_high', 'p_raw'])

    # Apply Benjamini-Hochberg FDR correction across all terms
    for df in [df_pos, df_neg, df_diff]:
        rej, p_fdr = fdrcorrection(df['p_raw'], alpha=fdr_alpha)
        df['p_fdr'] = p_fdr
        df['sig'] = rej

    return df_pos, df_neg, df_diff

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
