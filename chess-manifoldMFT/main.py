#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Standalone script to perform manifold analysis on fMRI ROI data,
with extensive logging for reproducibility and debugging.

This script:
  - Loads averaged beta images per condition from each subject's SPM.mat
  - Extracts ROI voxel patterns (observations × features)
  - Runs representational manifold analysis per ROI & subject
  - Aggregates metrics and performs group-level statistics (experts vs. non-experts)
  - Saves results (CSVs & plots) and a copy of this script for provenance
"""

import logging  # for configurable log messages
import os  # for filesystem operations
import shutil  # to copy this script for provenance
from datetime import datetime  # to timestamp outputs
import re  # for regex-based parsing
import numpy as np  # numerical arrays
import pandas as pd  # dataframes
import seaborn as sns  # high-level plotting
import matplotlib.pyplot as plt  # low-level plotting
import nibabel as nib  # NIfTI I/O
import scipy.io as sio  # MATLAB .mat I/O
# from scipy.stats import ttest_ind  # independent t-test
# from statsmodels.stats.multitest import multipletests  # FDR correction
from sklearn.random_projection import (
    GaussianRandomProjection,
)  # optional dimensionality reduction
from joblib import Parallel, delayed  # parallel processing

# manifold-analysis core functions
from mftma.manifold_analysis_correlation import manifold_analysis_corr
from mftma.alldata_dimension_analysis import alldata_dimension_analysis

from modules import BASE_GLM_PATH, ATLAS_FILE, EXPERTS, NONEXPERTS

# ---------------------- CONFIGURATION ---------------------- #

# Base path to GLM outputs (SPM.mat directories)
BASE_PATH = BASE_GLM_PATH

# Filename of the SPM.mat within each subject's folder
SPM_FILENAME = "SPM.mat"

# Path to ROI atlas (integer labels in NIfTI)
ATLAS_FILE = ATLAS_FILE

# Root directory for all output (CSVs, plots, provenance script)
OUTPUT_ROOT = "results"

# Subject identifiers by group
EXPERT_SUBJECTS = EXPERTS
NONEXPERT_SUBJECTS = NONEXPERTS
ALL_SUBJECTS = EXPERT_SUBJECTS + NONEXPERT_SUBJECTS

# Dimensionality reduction threshold
PROJECTION_DIM = 5000  # if features > this, apply random projection

# Statistical thresholds
ALPHA_FDR = 0.05  # false discovery rate threshold

# Parallel processing toggle
N_JOBS = 1  # -1 means use all available cores

# ---------------------- LOGGING SETUP ---------------------- #

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------- UTILITY FUNCTIONS ---------------------- #


def create_run_id() -> str:
    """
    Generate a unique run ID based on the current timestamp.
    Returns:
        str: Timestamp in 'YYYYMMDD-HHMMSS' format.
    """
    now = datetime.now()  # current date-time
    return now.strftime("%Y%m%d-%H%M%S")


def save_script_to_file(script_path: str, out_directory: str):
    """
    Copy this script file to the output directory for provenance.
    Args:
        script_path (str): Path to this script (__file__).
        out_directory (str): Directory where script copy will be saved.
    """
    os.makedirs(out_directory, exist_ok=True)  # ensure target exists
    dest_path = os.path.join(out_directory, os.path.basename(script_path))
    shutil.copy(script_path, dest_path)
    logger.info("Copied script to '%s' for provenance", dest_path)


def get_spm_info(subject_id: str) -> dict:
    """
    Load and average beta images per condition from a subject's SPM.mat.
    Args:
        subject_id (str): Subject identifier (e.g., "01").
    Returns:
        dict: Mapping condition_name -> Nifti1Image (average image).
    Raises:
        FileNotFoundError: If SPM.mat is missing.
    """
    spm_mat_path = os.path.join(BASE_PATH, f"sub-{subject_id}", "exp", SPM_FILENAME)
    if not os.path.isfile(spm_mat_path):
        raise FileNotFoundError(f"Missing SPM.mat at {spm_mat_path}")
    logger.debug("[%s] Loading SPM.mat from '%s'", subject_id, spm_mat_path)
    mat = sio.loadmat(spm_mat_path, struct_as_record=False, squeeze_me=True)
    SPM = mat["SPM"]  # MATLAB struct
    betas = SPM.Vbeta  # array of beta entries
    names = SPM.xX.name  # regressor names
    root = os.path.join(BASE_PATH, f"sub-{subject_id}", "exp")  # root directory for images

    # Regex to extract condition labels from regressor names
    pattern = re.compile(r"Sn\(\d+\)\s+(.*?)\*bf\(1\)")
    cond_indices = {}
    for idx, name in enumerate(names):
        m = pattern.match(name)
        if not m:
            continue
        cond = m.group(1)
        cond_indices.setdefault(cond, []).append(idx)

    # Average images per condition
    averaged = {}
    for cond, indices in cond_indices.items():
        logger.debug(
            "[%s] Averaging %d runs for condition '%s'", subject_id, len(indices), cond
        )
        sum_data = None
        affine = header = None
        for i in indices:
            entry = betas[i]
            fname = getattr(entry, "fname", None) or getattr(entry, "filename")
            img = nib.load(os.path.join(root, fname))
            data = img.get_fdata(dtype=np.float32)
            if sum_data is None:
                sum_data = np.zeros_like(data, dtype=np.float32)
                affine, header = img.affine, img.header
            sum_data += data
        avg_data = sum_data / float(len(indices))
        averaged[cond] = nib.Nifti1Image(avg_data, affine=affine, header=header)
    logger.info("[%s] Extracted %d conditions from SPM.mat", subject_id, len(averaged))
    return averaged


def load_roi_voxel_data(subject_id: str, atlas_data: np.ndarray, roi_list: list) -> tuple:
    """
    Extract voxel patterns for each ROI & condition for a subject.
    Args:
        subject_id (str): Subject ID.
        atlas_data (ndarray): 3D array of ROI labels.
        roi_list (list): Integer ROI labels to extract.
    Returns:
        tuple: (roi_dict, conditions)
            roi_dict: {roi_label: ndarray(obs × features)}
            conditions: sorted list of condition names
    """
    logger.info("[%s] Extracting ROI voxel data", subject_id)
    averaged = get_spm_info(subject_id)  # dict cond -> Nifti1Image
    conditions = sorted(averaged.keys())  # sorted condition names
    n_cond = len(conditions)  # number of observations

    # Initialize container for each ROI: obs × voxels
    roi_dict = {}
    for roi in roi_list:
        mask = atlas_data == roi  # boolean mask
        n_vox = int(mask.sum())
        roi_dict[roi] = np.zeros((n_cond, n_vox), dtype=np.float32)

    # Populate ROI arrays with data per condition
    for c_idx, cond in enumerate(conditions):
        img_data = averaged[cond].get_fdata(dtype=np.float32)
        for roi in roi_list:
            mask = atlas_data == roi
            roi_dict[roi][c_idx, :] = img_data[mask]
    logger.debug("[%s] Completed extraction for %d ROIs", subject_id, len(roi_list))
    return roi_dict, conditions


def run_manifold_for_matrix(mat_feat_x_obs: np.ndarray, n_reps: int = 1) -> dict:
    """
    Compute manifold metrics on a features × observations matrix.
    Args:
        mat_feat_x_obs (ndarray): features × observations.
        n_reps (int): Number of repetitions for randomization.
    Returns:
        dict: {'alpha_M', 'R_M', 'D_M', 'rho_center', 'D_participation_ratio',
               'D_explained_variance', 'D_feature'}
    """
    n_feat, n_obs = mat_feat_x_obs.shape
    # Random projection if too many features
    if n_feat > PROJECTION_DIM:
        logger.debug(
            "Applying Gaussian random projection from %d dims to %d",
            n_feat,
            PROJECTION_DIM,
        )
        proj = GaussianRandomProjection(n_components=PROJECTION_DIM, random_state=0)
        mat_feat_x_obs = proj.fit_transform(mat_feat_x_obs)

    # Build list of single-observation arrays for manifold_analysis_corr
    X_list = [mat_feat_x_obs[:, [i]] for i in range(n_obs)]
    # Core manifold analysis
    a, r, d, rho0, _ = manifold_analysis_corr(X_list, 0, 200, n_reps=10)
    # Additional dimension analyses
    D_pr, D_ev, D_feat = alldata_dimension_analysis(X_list, perc=0.9)
    # Aggregate results
    return {
        "alpha_M": 1.0 / np.mean(1.0 / a),
        "R_M": np.mean(r),
        "D_M": np.mean(d),
        "rho_center": rho0,
        "D_participation_ratio": D_pr,
        "D_explained_variance": D_ev,
        "D_feature": D_feat,
    }


def process_subject(subject_id: str, atlas_data: np.ndarray, roi_list: list) -> list:
    """
    Process one subject: extract ROI data, run manifold analysis per ROI.
    Args:
        subject_id (str): Subject ID.
        atlas_data (ndarray): 3D atlas labels.
        roi_list (list): ROI label list.
    Returns:
        list: Records of metrics per ROI for this subject.
    """
    from collections import defaultdict
    logger.info("[%s] Starting manifold analysis", subject_id)
    roi_obs_dict, cats_raw = load_roi_voxel_data(subject_id, atlas_data, roi_list)
    classes = [int(c[0]=="C") for c in cats_raw]
    # classes = list(range(1, 21)) * 2
    # classes = list(range(1, 41))

    # Initialize result containers
    roi_results = {}  # To store results per ROI

    # Loop over ROIs
    for roi, obs_feats in roi_obs_dict.items():
        logger.info("[%s] Starting manifold analysis (%s ROIs)", subject_id, roi)

        # if roi > 2:
        #     continue
        feats_obs = obs_feats.T  # features × observations

        valid_mask = ~np.all(np.isnan(feats_obs), axis=1)  # Drop features (rows) that are all-NaN
        feats_clean = feats_obs[valid_mask, :]  # shape: (features_kept × observations)

        # Group observations by class
        class_to_obs = defaultdict(list)
        for obs_idx, class_label in enumerate(classes):
            class_to_obs[class_label].append(feats_clean[:, obs_idx])  # Add each observation to its class

        # Build per-class manifolds
        X_list = []
        for class_label, obs_list in class_to_obs.items():
            # if len(obs_list) >= 2:  # Only keep classes with enough samples
            obs_array = np.stack(obs_list, axis=1)  # shape: (features_kept × num_samples)
            X_list.append(obs_array)

        # Skip if not enough manifolds
        if len(X_list) == 0:
            logging.warning("Skipping ROI %s: no classes with >= %d samples.", roi, 2)
            continue

        # print size of the list and of each matrix inside the list. each matrix represents the observations for a manifold
        logger.info("Manifold observations per class:")
        for i, obs in enumerate(X_list):
            logger.info("  Class %d: %s features * observations", i, obs.shape)

        # Perform Manifold Analysis
        # Core manifold analysis
        a, r, d, rho0, K = manifold_analysis_corr(X_list, 0, 200, n_reps=10)
        # Additional dimension analyses
        D_pr, D_ev, D_feat = alldata_dimension_analysis(X_list, perc=0.9)

        # Aggregate results
        roi_results[roi] = {
            "alpha_M": 1.0 / np.mean(1.0 / a),
            "R_M": np.mean(r),
            "D_M": np.mean(d),
            "rho_center": rho0,
            "D_participation_ratio": D_pr,
            "D_explained_variance": D_ev,
            "D_feature": D_feat,
        }

    logger.info("[%s] Finished manifold analysis (%d ROIs)", subject_id, len(roi_results))
    return roi_results


# ---------------------- RUN PIPELINE ---------------------- #
if __name__ == "__main__":
    logging.info("Running script")
    # 1) Create run-specific output directory
    run_id = f"{create_run_id()}_manifold-visual"
    out_dir = os.path.join(OUTPUT_ROOT, run_id)
    os.makedirs(out_dir, exist_ok=True)
    logger.info("Output directory created: '%s'", out_dir)

    # 2) Save a copy of this script for provenance
    save_script_to_file(__file__, out_dir)

    # 3) Load atlas data and define ROI list
    logger.info("Loading atlas from '%s'", ATLAS_FILE)
    atlas_img = nib.load(ATLAS_FILE)
    atlas_data = atlas_img.get_fdata().astype(int)
    roi_labels = sorted([lbl for lbl in np.unique(atlas_data) if lbl != 0])
    logger.info("Found %d ROIs in atlas", len(roi_labels))

    # 4) Parallel processing of all subjects
    logger.info("Launching parallel processing for %d subjects", len(ALL_SUBJECTS))

    # 5) Process all subjects, either in parallel or serially
    if USE_PARALLEL:
        logger.info("Processing subjects in parallel (n_jobs=%s)", N_JOBS)

        # First, run Parallel while preserving both subject and result
        results = Parallel(n_jobs=N_JOBS, backend="loky")(
            delayed(lambda subj: (subj, process_subject(subj, atlas_data, roi_labels)))(subj)
            for subj in ALL_SUBJECTS
        )

        # Now convert list of (subj, result) pairs into a dictionary
        nested_records = dict(results)

    else:
        logger.info("Processing subjects serially")

        nested_records = {}
        for subj in ALL_SUBJECTS:
            nested_records[subj] = process_subject(subj, atlas_data, roi_labels)

    # Flatten nested structure manually
    flat_records = []
    for sub, outer_dict in nested_records.items():
        for roi, metrics in outer_dict.items():
            flat_record = {
                'roi_index': roi,
                'sub': sub,
                'exp': sub in EXPERT_SUBJECTS,  # True if idx is in EXPERT_SUBJECTS, else False
                **metrics                  # unpack the inner dictionary (metrics)
            }
            flat_records.append(flat_record)

    # Now create the DataFrame
    df = pd.DataFrame(flat_records)
    metrics_csv = os.path.join(out_dir, f"{run_id}_manifold_metrics.csv")
    df.to_csv(metrics_csv, index=False)
    logger.info("Saved manifold metrics table to '%s'", metrics_csv)

    # # 6) Group-level statistics (α_M experts vs non-experts per ROI)
    # logger.info("Performing group-level statistics on α_M")
    # stats = []
    # for roi in roi_labels:
    #     vals_exp = df[(df.roi == roi) & (df.exp)]["alpha_M"]
    #     vals_non = df[(df.roi == roi) & (~df.exp)]["alpha_M"]
    #     t_stat, p_val = ttest_ind(vals_exp, vals_non, nan_policy="omit")
    #     stats.append((roi, p_val))
    # rois, pvals = zip(*stats)
    # rej, pvals_fdr, _, _ = multipletests(pvals, alpha=ALPHA_FDR, method="fdr_bh")
    # stats_df = pd.DataFrame(
    #     {"roi": rois, "pval": pvals, "pval_fdr": pvals_fdr, "significant": rej}
    # )
    # stats_csv = os.path.join(out_dir, f"{run_id}_stats_alpha_M.csv")
    # stats_df.to_csv(stats_csv, index=False)
    # logger.info("Saved statistics table to '%s'", stats_csv)

    # 8) Plotting: line plots for all manifold metrics by ROI, with hue for expertise (True/False)
    logger.info("Generating line plots for all manifold metrics by expertise")
    sns.set(style="whitegrid")

    # List all metrics to plot
    metrics_to_plot = [
        "alpha_M",
        "R_M",
        "D_M",
        "rho_center",
        "D_participation_ratio",
        "D_explained_variance",
        "D_feature",
    ]

    # Determine subplot grid size (4 columns, rows as needed)
    n_metrics = len(metrics_to_plot)
    n_cols = 4
    n_rows = int(np.ceil(n_metrics / n_cols))

    # Create figure and axes
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 4, n_rows * 4), sharey=False)
    axes = axes.flatten()  # flatten to 1D for easy iteration

    # Loop over each metric and corresponding axis
    for ax, metric in zip(axes, metrics_to_plot):
        # Draw lineplot: ROI on x-axis, metric on y-axis, hue by expertise
        sns.lineplot(
            data=df,
            x="roi_index",
            y=metric,
            hue="exp",         # separate lines for expert (True) vs non-expert (False)
            ci=95,             # 95% confidence interval around the mean
            marker="o",        # circle markers at each ROI
            err_style="band",  # shaded error bands
            ax=ax
        )
        # Title and axis labels
        ax.set_title(f"{metric} by ROI and Expertise")
        ax.set_xlabel("ROI")
        ax.set_ylabel(metric)
        # Rotate x-tick labels for readability
        ax.tick_params(axis="x", rotation=90)
        # Move legend inside subplot
        ax.legend(title="Expertise", loc="upper right", fontsize="small", title_fontsize="small")

    # Remove any unused subplots if metrics < n_rows*n_cols
    for unused_ax in axes[n_metrics:]:
        fig.delaxes(unused_ax)

    # Adjust layout and save
    fig.tight_layout()
    lineplots_path = os.path.join(out_dir, f"{run_id}_manifold_lineplots.png")
    fig.savefig(lineplots_path)
    logger.info("Saved line plots to '%s'", lineplots_path)
import logging
