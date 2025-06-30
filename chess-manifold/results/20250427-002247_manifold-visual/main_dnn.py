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
from scipy.stats import ttest_ind  # independent t-test
from statsmodels.stats.multitest import multipletests  # FDR correction
from sklearn.random_projection import (
    GaussianRandomProjection,
)  # optional dimensionality reduction
from joblib import Parallel, delayed  # parallel processing

# manifold-analysis core functions
from mftma.manifold_analysis_correlation import manifold_analysis_corr
from mftma.alldata_dimension_analysis import alldata_dimension_analysis

# ---------------------- CONFIGURATION ---------------------- #

# Base path to GLM outputs (SPM.mat directories)
BASE_PATH = "/data/projects/chess/data/BIDS/derivatives/fmriprep-SPM_smoothed-NO_GS-FD-HMP_brainmasked/MNI/fmriprep-SPM-MNI/GLM"

# Root directory for all output (CSVs, plots, provenance script)
OUTPUT_ROOT = "results"

# Dimensionality reduction threshold
PROJECTION_DIM = 5000  # if features > this, apply random projection

# Statistical thresholds
ALPHA_FDR = 0.05  # false discovery rate threshold

# Parallel processing toggle
N_JOBS = -1  # -1 means use all available cores

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



# ---------------------- RUN PIPELINE ---------------------- #

# 1) Create run-specific output directory
run_id = f"{create_run_id()}_manifold-visual"
out_dir = os.path.join(OUTPUT_ROOT, run_id)
os.makedirs(out_dir, exist_ok=True)
logger.info("Output directory created: '%s'", out_dir)

# 2) Save a copy of this script for provenance
save_script_to_file(__file__, out_dir)

from collections import defaultdict
import numpy as np

# Load the .npz files
acts_trained_npz = np.load("activations_trained.npz", allow_pickle=True)
acts_untrained_npz = np.load("activations_untrained.npz", allow_pickle=True)

# Convert loaded data into normal Python dictionaries
acts_trained = {key: acts_trained_npz[key].item() for key in acts_trained_npz.files}  # .item() because it's a nested object
acts_untrained = {key: acts_untrained_npz[key].item() for key in acts_untrained_npz.files}

# classes = [c[0]=="C" for c in cats_raw]
# classes = list(range(1, 21)) * 2  # Example: 20 categories duplicated
# classes = list(range(1, 41))  # Alternative: 40 categories

# Initialize result containers
nested_records = {}

# Loop over subjects and ROIs
for subject_id, acts in zip(["exp", "nonexp"], [acts_trained, acts_untrained]):
    roi_results = {}  # To store results per ROI
    for roi, roi_data in acts.items():
        if roi != "08_body_spatial.0":
            continue

        if "value" in roi:
            continue
        logger.info("[%s] Starting manifold analysis (%s ROI)", subject_id, roi)
        all_activations = {}
        for stim_id, stim_info in roi_data.items():
            activation = stim_info['activation']  # Extract only the activation
            activation_flat = activation.reshape(-1)  # Flatten (C × H × W) into (features,)
            all_activations[int(stim_info["stim_id"])] = activation_flat

        # Stack into (features × observations)
        # feats_obs: features × observations, ordered by sorted stim_ids
        stim_ids = sorted(all_activations.keys())  # sort the stimulus ids

        feats_obs = np.stack(
            [all_activations[stim_id] for stim_id in stim_ids],  # stack activations in order
            axis=0
        )

        # 1. Drop features (rows) that are all-NaN
        valid_mask = ~np.all(np.isnan(feats_obs), axis=1)
        feats_obs = feats_obs[valid_mask, :]

        # 2. Drop observations (columns) that are all-zero
        nonzero_mask = ~np.all(feats_obs == 0, axis=1)
        feats_clean = feats_obs[nonzero_mask, :]

        # If N is greater than 5000, do the random projection to 5000 features
        X = [d.reshape(d.shape[0], -1) for d in feats_clean]
        N = X[0].shape[0]
        if N > 5000:
            print("Projecting {}".format(roi))
            M = np.random.randn(5000, N)
            M /= np.sqrt(np.sum(M*M, axis=1, keepdims=True))
            X_list = [np.matmul(M, d) for d in X]

        # # Group observations by class
        # class_to_obs = defaultdict(list)
        # for obs_idx, class_label in enumerate(stim_ids):
        #     class_to_obs[class_label].append(feats_clean[:, obs_idx])  # Add each observation to its class

        # # Build per-class manifolds
        # X_list = []
        # for class_label, obs_list in class_to_obs.items():
        #     # if len(obs_list) >= 2:  # Only keep classes with enough samples
        #     obs_array = np.stack(obs_list, axis=1)  # shape: (features_kept × num_samples)
        #     X_list.append(obs_array)

        # # Skip if not enough manifolds
        # if len(X_list) == 0:
        #     print(f"Skipping ROI {roi}: no classes with >= {2} samples.")
        #     continue

        # Perform Manifold Analysis
        # Core manifold analysis
        a, r, d, rho0, K = manifold_analysis_corr(X_list, 0, 300, n_reps=1)
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

    nested_records[subject_id] = roi_results
    logger.info("[%s] Finished manifold analysis (%d ROIs)", subject_id, len(roi_results))


# Flatten nested structure manually
flat_records = []
for sub, outer_dict in nested_records.items():
    for roi, metrics in outer_dict.items():
        flat_record = {
            'roi_index': roi,
            'sub': sub,
            'exp': sub == "exp",  # True if idx is in EXPERT_SUBJECTS, else False
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
