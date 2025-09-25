#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ROI-level second-level summaries (Univariate or RSA).

Provides a single `run_analysis` function to compute per-ROI group
differences (Experts vs. Novices), apply FDR correction, plot flat
surface overlays, plot glass brains from ROI T-values, and export a
compact LaTeX table. Designed to be called from thin runner scripts.
"""
from __future__ import annotations

import os
import logging
from typing import Dict

import numpy as np
import pandas as pd
from nilearn.image import load_img
from nilearn.maskers import NiftiLabelsMasker
from nilearn.datasets import fetch_surf_fsaverage, fetch_atlas_harvard_oxford

from common.stats_utils import fdr_correction
from glm.modules.roi_utils import (
    get_region_label,
    compute_confidence_intervals,
    get_label_at_com,
    paths_for,
)
from glm.modules.plot_utils import (
    plot_sig_rois_on_glasser_surface,
    plot_glass_brain_from_df,
)
from glm.modules.glm_utils import export_diff_stats_to_latex


logger = logging.getLogger(__name__)


def run_analysis(
    data_dir: str,
    contrasts: Dict[str, str],
    atlas_path: str,
    mode: str,
    out_dir: str,
) -> str:
    """Run ROI-level summaries for a set of contrasts.

    Parameters
    ----------
    data_dir : str
        Base directory for subject-level images (univariate GLM or RSA maps).
    contrasts : dict[str, str]
        Mapping from filename (e.g., 'con_0001.nii' or 'checkmate.nii.gz') to
        human-readable label.
    atlas_path : str
        Path to the Glasser bilateral atlas in MNI space.
    mode : str
        Either 'univ' for univariate GLM or 'rsa' for RSA searchlight.
    out_dir : str
        Output directory where figures and tables will be written.

    Returns
    -------
    str
        Absolute path to the output directory used.
    """
    os.makedirs(out_dir, exist_ok=True)
    logger.info("Output directory: %s", out_dir)

    # Load atlas image and ROI labels (exclude background = 0)
    atlas_img = load_img(atlas_path)
    roi_labels = np.unique(atlas_img.get_fdata())[1:]

    # Surface templates and annotation files
    fsav = fetch_surf_fsaverage("fsaverage")
    from config import DERIVATIVES_PATH
    annot_files = {
        "left": os.path.join(str(DERIVATIVES_PATH), "fastsurfer", "fsaverage", "label", "lh.HCPMMP1.annot"),
        "right": os.path.join(str(DERIVATIVES_PATH), "fastsurfer", "fsaverage", "label", "rh.HCPMMP1.annot"),
    }

    # ROI mean extraction
    masker = NiftiLabelsMasker(labels_img=atlas_img, standardize=False, strategy="mean", verbose=0)

    # Harvard–Oxford for CoM anatomical labeling
    atlas_ho = fetch_atlas_harvard_oxford("cort-maxprob-thr25-2mm")
    atlas_img_ho = load_img(atlas_ho.maps)
    atlas_data_ho = atlas_img_ho.get_fdata()
    affine_ho = atlas_img_ho.affine
    atlas_labels_ho = atlas_ho.labels

    # Cache Glasser data and affine for coordinate transforms
    glasser_data = atlas_img.get_fdata()
    affine_glasser = atlas_img.affine

    from config import EXPERTS as EXP, NONEXPERTS as NOV

    for fname, label in contrasts.items():
        logger.info("Processing: %s", label)

        # Load per-group subject files
        exp_paths = paths_for(EXP, mode, data_dir, fname)
        nov_paths = paths_for(NOV, mode, data_dir, fname)

        # Extract ROI means across subjects
        X_exp = np.squeeze([masker.fit_transform(p) for p in exp_paths])
        X_nov = np.squeeze([masker.fit_transform(p) for p in nov_paths])

        # Welch/CI per ROI
        mean_diff, t_vals, p_vals, cis = compute_confidence_intervals(X_exp, X_nov)
        _, p_vals_fdr = fdr_correction(p_vals, alpha=0.05, method="fdr_bh")

        # Consolidate results
        df = pd.DataFrame(
            {
                "ROI_idx": roi_labels,
                "Mean_Diff": mean_diff,
                "T_Diff": t_vals,
                "P_Diff": p_vals_fdr,
                "CI_Low": cis[0],
                "CI_High": cis[1],
            }
        )
        df["ROI"] = [get_region_label(r) for r in df["ROI_idx"]]
        df["CenterOfMass_Label"] = [
            get_label_at_com(r, glasser_data, affine_glasser, atlas_data_ho, affine_ho, atlas_labels_ho)
            for r in df["ROI_idx"]
        ]

        # Plot all ROIs and significant subset; save under out_dir
        analysis_label = "RSA searchlight" if mode == "rsa" else "Univariate"
        plot_sig_rois_on_glasser_surface(
            df,
            glasser_annot_files=annot_files,
            fsaverage=fsav,
            cmap="cold_hot",
            title=f"{analysis_label} | {label}",
            threshold=1e-5,
            out_dir=out_dir,
        )
        df_sig = df[df["P_Diff"] < 0.05]
        plot_sig_rois_on_glasser_surface(
            df_sig,
            glasser_annot_files=annot_files,
            fsaverage=fsav,
            cmap="cold_hot",
            title=f"{analysis_label} | {label} (FDR p < .05)",
            threshold=1e-5,
            out_dir=out_dir,
        )

        # Glass brain projections
        plot_glass_brain_from_df(df_input=df, atlas_img=atlas_img, title=f"{analysis_label} | {label}", out_dir=out_dir)
        plot_glass_brain_from_df(df_input=df_sig, atlas_img=atlas_img, title=f"{analysis_label} | {label} (FDR p < .05)", out_dir=out_dir)

        # Export LaTeX table (compact)
        latex_path = os.path.join(out_dir, f"{mode}_{os.path.splitext(os.path.basename(fname))[0]}_diff_only_table.tex")
        export_diff_stats_to_latex(
            df,
            out_path=latex_path,
            value_cols=("ROI", "Mean_Diff", "T_Diff", "P_Diff", "CI_Low", "CI_High"),
            caption=f"{analysis_label} | {label}: Experts > Novices",
            label=f"tab:{mode}_{os.path.splitext(os.path.basename(fname))[0]}_diff_only",
        )

    return os.path.abspath(out_dir)

