#!/usr/bin/env python3
"""Run ROI-wise manifold analysis comparing experts and novices."""

import os
from typing import Tuple

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from modules import logger, EXPERTS, NONEXPERTS
from modules.io_utils import load_all_betas, load_atlas
from modules.analysis_utils import compute_manifold, build_metric_array, fdr_ttest
from modules.plot_utils import (
    plot_group_comparison,
    plot_group_heatmap,
    plot_all_group_manifold_plots,
)

AVG_RUNS = False
PARALLEL= False

def process_subject(
    subject: str,
    atlas_data: np.ndarray,
    rois: np.ndarray,
    out_dir: str,
    roi_n_jobs: int = 1
) -> dict:
    """Compute manifold metrics for each ROI for a single subject."""
        
    logger.info(f"Processing subject: {subject}")
    betas_dict, labels = load_all_betas(subject, avg_runs=AVG_RUNS)
    logger.info(f"Loaded betas for subject {subject}: conditions={labels}")

    # ROI metrics arrays
    def roi_metrics(roi: int):
        logger.info(f"Subject {subject}: Processing ROI {roi}")
        mask = atlas_data == roi
        manifolds = []

        for cond in labels:
            logger.info(f"Subject {subject}, ROI {roi}: Processing condition {cond}")
            # extract runs × voxels within this ROI
            if type(betas_dict[cond]) == list:
                condition_vectors = np.stack([beta_img[mask] for beta_img in betas_dict[cond]], axis=0)  # shape = (n_runs, n_voxels)
            else:
                # add a dimension for runs
                condition_vectors = betas_dict[cond][mask][np.newaxis, :]  #

            # identify columns (voxels) to keep: no NaNs and variance > 0
            has_nan = np.isnan(condition_vectors).any(axis=0)
            valid_cols = ~has_nan

            logger.info(f"Subject {subject}, ROI {roi}, Condition {cond}: roi_runs shape {condition_vectors.shape} --> valid {np.sum(valid_cols)}")

            # filter out bad voxels
            filtered = condition_vectors[:, valid_cols]   # now (n_runs, n_valid_voxels)

            # if type(betas_dict[cond]) == list:
            #     # scale: observations are rows, features are columns
            #     scaler = StandardScaler()
            #     filtered = scaler.fit_transform(filtered)  # (n_runs, n_valid_voxels)

            # compute_manifold expects features × observations
            manifolds.append(filtered.T)

        logger.info(f"Subject {subject}, ROI {roi}: Computing manifold metrics")
        manifold_results = compute_manifold(manifolds=manifolds)

        logger.info(f"Subject {subject}, ROI {roi}: manifold analysis completed")
        return manifold_results
    
    if roi_n_jobs == 1:
        logger.info(f"Subject {subject}: Processing ROIs sequentially")
        results = {
            roi: roi_metrics(roi)
            for roi in rois
        }
    else:
        logger.info(f"Subject {subject}: Processing ROIs in parallel (n_jobs={roi_n_jobs})")
        parallel_rois_results = Parallel(n_jobs=roi_n_jobs)(
            delayed(roi_metrics)(roi) for roi in rois
        )
        results = dict(zip(rois, parallel_rois_results))

    logger.info(f"Subject {subject}: Finished processing all ROIs")
    return results


def run_group(
    subs,
    atlas_data,
    rois,
    out_dir,
    subj_n_jobs: int = -1,
    roi_n_jobs: int = 1
):
    """Process a group of subjects using joblib.Parallel if ``subj_n_jobs`` > 1."""
    logger.info(f"Running group analysis for {len(subs)} subjects")
    def _run(s):
        return process_subject(s, atlas_data, rois, out_dir, roi_n_jobs)

    if subj_n_jobs == 1:
        logger.info("Processing subjects sequentially")
        results = {
            s: process_subject(s, atlas_data, rois, out_dir, roi_n_jobs)
            for s in subs
        }
    else:
        logger.info(f"Processing subjects in parallel (n_jobs={subj_n_jobs})")
        parallel_subs_results = Parallel(n_jobs=subj_n_jobs)(
            delayed(process_subject)(s, atlas_data, rois, out_dir, roi_n_jobs)
            for s in subs
        )
        results = dict(zip(subs, parallel_subs_results))

    logger.info("Finished group processing")
    return results

logger.info("Loading atlas")
atlas_data, rois = load_atlas()
out_dir = "manifold_results"
os.makedirs(out_dir, exist_ok=True)
logger.info(f"Output directory: {out_dir}")
 
# fully parallel across ROIs and subjects by default:
logger.info("Processing experts group")
exp_dict = run_group(
    EXPERTS, atlas_data, rois, out_dir,
    subj_n_jobs=-1 if PARALLEL else 1, roi_n_jobs=1
)
logger.info("Processing non-experts group")
nov_dict = run_group(
    NONEXPERTS, atlas_data, rois, out_dir,
    subj_n_jobs=-1 if PARALLEL else 1, roi_n_jobs=1
)

def flatten_metrics(data_dict, label):
    return [
        {"subject": subj, "roi": roi, **metrics, "expertise": label}
        for subj, roi_dict in data_dict.items()
        for roi, metrics in roi_dict.items()
    ]

# Flatten and merge
full_df = pd.DataFrame(flatten_metrics(exp_dict, "Expert") + flatten_metrics(nov_dict, "Novice"))

# --- Group-level plotting and stats ---
metrics_to_plot = [
    "alpha_M", "R_M", "D_M", "rho_center", "D_participation_ratio", "D_explained_variance", "D_feature"
]
rois = full_df["roi"].unique()
exp_subs = full_df[full_df["expertise"]=="Expert"]["subject"].unique()
nov_subs = full_df[full_df["expertise"]=="Novice"]["subject"].unique()
roi_name_map = {i: f"ROI {i}" for i in rois}
plot_all_group_manifold_plots(
    full_df, out_dir, metrics_to_plot, rois, roi_name_map, exp_subs, nov_subs
)
