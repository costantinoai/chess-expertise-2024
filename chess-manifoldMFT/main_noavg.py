#!/usr/bin/env python3
"""
Run ROI-wise manifold analysis comparing experts and novices.

This script loads subject data, computes manifold metrics for each ROI, and generates group-level plots comparing experts and novices.
"""

# =====================
# Imports
# =====================
import os
from typing import Tuple, Dict, Any, List

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from modules import logger, EXPERTS, NONEXPERTS
from modules.io_utils import load_all_betas, load_atlas, assign_manifold_labels
from modules.analysis_utils import compute_manifold
from modules.plot_utils import plot_all_group_manifold_plots

# =====================
# Constants
# =====================
AVG_RUNS = True  # Whether to average runs when loading betas
PARALLEL = True  # Whether to use parallel processing
MANIFOLD_LABELS = 'strategy'  # Manifold assignment strategy ('strategy', 'stimuli', etc.)
EXPERT_LABEL = "Expert"
NOVICE_LABEL = "Novice"
OUT_DIR = "manifold_results"

# =====================
# Functions
# =====================
def process_subject(
    subject: str,
    atlas_data: np.ndarray,
    rois: np.ndarray,
    out_dir: str,
    roi_n_jobs: int = 1,
    manifold_strategy: str = 'stimuli'
) -> Dict[int, Dict[str, Any]]:
    """
    Compute manifold metrics for each ROI for a single subject.

    Args:
        subject (str): Subject identifier.
        atlas_data (np.ndarray): Atlas data array.
        rois (np.ndarray): Array of ROI indices.
        out_dir (str): Output directory.
        roi_n_jobs (int): Number of parallel jobs for ROIs.
        manifold_strategy (str): Manifold assignment strategy.

    Returns:
        Dict[int, Dict[str, Any]]: Dictionary mapping ROI to computed metrics.
    """
    logger.info(f"Processing subject: {subject}")
    betas_dict, labels = load_all_betas(subject, avg_runs=AVG_RUNS)
    logger.info(f"Loaded betas for subject {subject}: conditions={labels}")

    # Assign manifold grouping according to strategy
    manifold_assignments, manifold_names = assign_manifold_labels(labels, manifold_strategy)

    def roi_metrics(roi: int) -> Dict[str, Any]:
        """Compute metrics for a single ROI."""
        logger.info(f"Subject {subject}: Processing ROI {roi}")
        mask = atlas_data == roi
        condition_vectors_list = []
        for cond in labels:
            logger.info(f"Subject {subject}, ROI {roi}: Processing condition {cond}")
            if isinstance(betas_dict[cond], list):
                condition_vectors = np.stack([beta_img[mask] for beta_img in betas_dict[cond]], axis=0)
            else:
                condition_vectors = betas_dict[cond][mask][np.newaxis, :]
            # Identify columns (voxels) to keep: no NaNs and variance > 0
            has_nan = np.isnan(condition_vectors).any(axis=0)
            valid_cols = ~has_nan
            logger.info(f"Subject {subject}, ROI {roi}, Condition {cond}: roi_runs shape {condition_vectors.shape} --> valid {np.sum(valid_cols)}")
            # Filter out bad voxels
            filtered = condition_vectors[:, valid_cols]   # (n_runs, n_valid_voxels)
            condition_vectors_list.append(filtered)
        # Group by manifold assignment
        manifolds = []
        for m_idx in sorted(set(manifold_assignments)):
            if m_idx == -1:
                continue
            # Find all conditions assigned to this manifold
            cond_indices = [i for i, x in enumerate(manifold_assignments) if x == m_idx]
            cond_names = [labels[i] for i in cond_indices]
            logger.info(f"Subject {subject}, ROI {roi}: Manifold {m_idx} ({manifold_names[m_idx] if m_idx < len(manifold_names) else m_idx}) will include conditions: {cond_names}")
            # Stack all runs for these conditions
            if cond_indices:
                all_runs = np.concatenate([condition_vectors_list[i] for i in cond_indices], axis=0)
                logger.info(f"Subject {subject}, ROI {roi}: Manifold {m_idx} shape: {all_runs.shape}")
                manifolds.append(all_runs.T)  # features x observations
        logger.info(f"Subject {subject}, ROI {roi}: Computing manifold metrics")
        manifold_results = compute_manifold(manifolds=manifolds)
        logger.info(f"Subject {subject}, ROI {roi}: manifold analysis completed")
        return manifold_results

    if roi_n_jobs == 1:
        logger.info(f"Subject {subject}: Processing ROIs sequentially")
        results = {
            int(roi): roi_metrics(int(roi))
            for roi in np.atleast_1d(rois)
        }
    else:
        logger.info(f"Subject {subject}: Processing ROIs in parallel (n_jobs={roi_n_jobs})")
        parallel_rois_results = Parallel(n_jobs=roi_n_jobs)(
            delayed(roi_metrics)(int(roi)) for roi in np.atleast_1d(rois)
        )
        results = dict(zip([int(r) for r in np.atleast_1d(rois)], parallel_rois_results))

    logger.info(f"Subject {subject}: Finished processing all ROIs")
    return results


def run_group(
    subs: List[str],
    atlas_data: np.ndarray,
    rois: np.ndarray,
    out_dir: str,
    subj_n_jobs: int = -1,
    roi_n_jobs: int = 1,
    manifold_strategy: str = 'stimuli'
) -> Dict[str, Dict[int, Dict[str, Any]]]:
    """
    Process a group of subjects using joblib.Parallel if ``subj_n_jobs`` > 1.

    Args:
        subs (List[str]): List of subject identifiers.
        atlas_data (np.ndarray): Atlas data array.
        rois (np.ndarray): Array of ROI indices.
        out_dir (str): Output directory.
        subj_n_jobs (int): Number of parallel jobs for subjects.
        roi_n_jobs (int): Number of parallel jobs for ROIs.
        manifold_strategy (str): Manifold assignment strategy.

    Returns:
        Dict[str, Dict[int, Dict[str, Any]]]: Results for all subjects.
    """
    logger.info(f"Running group analysis for {len(subs)} subjects")
    def _run(s):
        return process_subject(s, atlas_data, rois, out_dir, roi_n_jobs, manifold_strategy)

    if subj_n_jobs == 1:
        logger.info("Processing subjects sequentially")
        results = {
            s: process_subject(s, atlas_data, rois, out_dir, roi_n_jobs, manifold_strategy)
            for s in subs
        }
    else:
        logger.info(f"Processing subjects in parallel (n_jobs={subj_n_jobs})")
        parallel_subs_results = Parallel(n_jobs=subj_n_jobs)(
            delayed(process_subject)(s, atlas_data, rois, out_dir, roi_n_jobs, manifold_strategy)
            for s in subs
        )
        results = dict(zip(subs, parallel_subs_results))

    logger.info("Finished group processing")
    return results


def flatten_metrics(data_dict: Dict[str, Dict[int, Dict[str, Any]]], label: str) -> List[Dict[str, Any]]:
    """
    Flatten nested metrics dictionary for DataFrame construction.

    Args:
        data_dict (Dict): Nested dictionary of metrics.
        label (str): Expertise label.

    Returns:
        List[Dict[str, Any]]: List of flattened metric dicts.
    """
    return [
        {"subject": subj, "roi": roi, **metrics, "expertise": label}
        for subj, roi_dict in data_dict.items()
        for roi, metrics in roi_dict.items()
    ]

# =====================
# Main Execution
# =====================
if __name__ == "__main__":
    logger.info("Loading atlas")
    atlas_data, rois = load_atlas()
    os.makedirs(OUT_DIR, exist_ok=True)
    logger.info(f"Output directory: {OUT_DIR}")

    # Fully parallel across ROIs and subjects by default
    logger.info("Processing experts group")
    exp_dict = run_group(
        EXPERTS, atlas_data, rois, OUT_DIR,
        subj_n_jobs=-1 if PARALLEL else 1, roi_n_jobs=1,
        manifold_strategy=MANIFOLD_LABELS
    )
    logger.info("Processing non-experts group")
    nov_dict = run_group(
        NONEXPERTS, atlas_data, rois, OUT_DIR,
        subj_n_jobs=-1 if PARALLEL else 1, roi_n_jobs=1,
        manifold_strategy=MANIFOLD_LABELS
    )

    # Flatten and merge results
    full_df = pd.DataFrame(
        flatten_metrics(exp_dict, EXPERT_LABEL) + flatten_metrics(nov_dict, NOVICE_LABEL)
    )

    # --- Group-level plotting and stats ---
    metrics_to_plot = [
        "alpha_M", "R_M", "D_M", "rho_center", "D_participation_ratio", "D_explained_variance", "D_feature"
    ]
    rois_list = list(full_df["roi"].unique().tolist())
    exp_subs = list(full_df[full_df["expertise"] == EXPERT_LABEL]["subject"].unique().tolist())
    nov_subs = list(full_df[full_df["expertise"] == NOVICE_LABEL]["subject"].unique().tolist())
    roi_name_map = {i: f"ROI {i}" for i in rois_list}
    plot_all_group_manifold_plots(
        full_df, OUT_DIR, metrics_to_plot, rois_list, roi_name_map, exp_subs, nov_subs
    )

