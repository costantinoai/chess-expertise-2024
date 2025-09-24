#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 25 22:31:01 2025

@author: costantino_ai
"""

import os
import re
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import nibabel as nib
import scipy.io as sio

from sklearn.decomposition import PCA
from scipy.stats import ttest_ind
from statsmodels.stats.multitest import multipletests

from joblib import Parallel, delayed
import logging

from modules.helpers import create_run_id, save_script_to_file


##############################################################################
#                                 CONSTANTS
##############################################################################

# Define any constants or paths used throughout the script.
CUSTOM_COLORS = [
    "#c6dbef", "#c6dbef",   # Light blue (2)
    "#2171b5", "#2171b5", "#2171b5",  # Dark blue (3)
    "#a1d99b", "#a1d99b", "#a1d99b", "#a1d99b",  # Light green (4)
    "#00441b", "#00441b", "#00441b",  # Dark green (3)
    "#fbb4b9", "#fbb4b9",  # Pink (2)
    "#cb181d", "#cb181d", "#cb181d", "#cb181d",  # Red (4)
    "#fec44f", "#fec44f", "#fec44f", "#fec44f"   # Gold (4)
]

EXPERT_SUBJECTS = [
    "03", "04", "06", "07", "08", "09",
    "10", "11", "12", "13", "16", "20",
    "22", "23", "24", "29", "30", "33",
    "34", "36"
]

NONEXPERT_SUBJECTS = [
    "01", "02", "15", "17", "18", "19",
    "21", "25", "26", "27", "28", "32",
    "35", "37", "39", "40", "41", "42",
    "43", "44"
]

BASE_PATH = "/data/projects/chess/data/BIDS/derivatives/fmriprep-SPM_smoothed-NO_GS-FD-HMP_brainmasked/MNI/fmriprep-SPM-MNI/GLM"
SPM_FILENAME = "SPM.mat"
ATLAS_FILE = "/data/projects/chess/data/misc/templates/tpl-MNI152NLin2009cAsym_res-02_atlas-Glasser2016_desc-bilateral_resampled.nii"

ALPHA_FDR = 0.05      # alpha level for FDR correction
USE_PARALLEL = False   # Toggle parallelization
use_fdr = True        # Whether to use FDR-corrected p-values for significance markers


##############################################################################
#                                LOGGING SETUP
##############################################################################

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)  # You can set to logging.DEBUG for more detail


##############################################################################
#                            FUNCTION DEFINITIONS
##############################################################################
def get_spm_betas_info(subject_id):
    """
    Returns, for each condition label (e.g. 'C1', 'C2', …), a list of dicts:
      - 'run'            : int, run number
      - 'regressor_name' : str, full regressor name
      - 'beta_path'      : str, absolute path to the .nii file
    """
    # 1. load SPM.mat
    spm_mat_path = os.path.join(BASE_PATH, f"sub-{subject_id}", "exp", "SPM.mat")
    if not os.path.isfile(spm_mat_path):
        raise FileNotFoundError(f"SPM.mat not found at {spm_mat_path}")

    SPM = sio.loadmat(spm_mat_path, struct_as_record=False, squeeze_me=True)["SPM"]
    beta_info       = SPM.Vbeta
    regressor_names = SPM.xX.name
    spm_dir         = getattr(SPM, "swd", os.path.dirname(spm_mat_path))

    # 2. regex to pull out run and condition
    pattern = r"Sn\((\d+)\)\s+(.*?)\*bf\(1\)"

    cond_info = {}
    for idx, full_name in enumerate(regressor_names):
        m = re.match(pattern, full_name)
        if not m:
            continue

        run  = int(m.group(1))
        cond = m.group(2)

        # --- BUG FIXED HERE ---
        entry_struct = beta_info[idx]
        beta_fname   = getattr(entry_struct, "fname", None)
        if beta_fname is None:
            available = [f for f in dir(entry_struct) if not f.startswith("_")]
            raise AttributeError(
                f"Vbeta[{idx}] has no 'fname' field. Available fields: {available}"
            )

        beta_path = os.path.join(spm_dir, beta_fname)

        cond_info.setdefault(cond, []).append({
            "run": run,
            "regressor_name": full_name,
            "beta_path": beta_path
        })

    return cond_info

def get_spm_info(subject_id):
    """
    Loads the subject's SPM.mat structure and returns averaged beta images.

    The function identifies condition-specific beta images (e.g., 'C1', 'C2', etc.)
    across multiple runs and averages them into a single image per condition.

    Parameters
    ----------
    subject_id : str
        Subject identifier (e.g., '001').

    Returns
    -------
    averaged_betas : dict
        Keys are condition names (e.g., 'C1', 'C2', ...), values are in-memory
        Nifti1Image objects representing the average beta image for that condition.
    """
    import re

    # Construct the full path to the SPM.mat file
    spm_mat_path = os.path.join(BASE_PATH, f"sub-{subject_id}", "exp", SPM_FILENAME)
    if not os.path.isfile(spm_mat_path):
        logger.error(f"SPM.mat not found for sub-{subject_id} at: {spm_mat_path}")
        raise FileNotFoundError(f"Could not find SPM.mat at: {spm_mat_path}")

    logger.debug(f"Loading SPM.mat from: {spm_mat_path}")
    spm_dict = sio.loadmat(spm_mat_path, struct_as_record=False, squeeze_me=True)
    SPM = spm_dict["SPM"]

    beta_info = SPM.Vbeta
    regressor_names_full = SPM.xX.name

    # Regex to match condition names of the form "Sn(1) C1*bf(1)"
    pattern = r"Sn\(\d+\)\s+(.*?)\*bf\(1\)"

    # Group beta indices by condition name (C1, C2, etc.)
    condition_dict = {}
    for i, reg_name in enumerate(regressor_names_full):
        match = re.match(pattern, reg_name)
        if match:
            cond_name = match.group(1)  # e.g., "C1"
            if cond_name not in condition_dict:
                condition_dict[cond_name] = []
            condition_dict[cond_name].append(i)

    # For each condition, load the corresponding beta images and average them
    averaged_betas = {}
    spm_dir = SPM.swd if hasattr(SPM, 'swd') else os.path.dirname(spm_mat_path)

    for cond_name, indices in condition_dict.items():
        if not indices:
            continue

        sum_data = None
        affine, header = None, None

        for idx in indices:
            beta_fname = (
                beta_info[idx].fname
                if hasattr(beta_info[idx], 'fname')
                else beta_info[idx].filename
            )
            beta_path = os.path.join(spm_dir, beta_fname)

            img = nib.load(beta_path)
            data = img.get_fdata(dtype=np.float32)

            # Initialize the summation array on the first pass
            if sum_data is None:
                sum_data = np.zeros_like(data, dtype=np.float32)
                affine = img.affine
                header = img.header

            sum_data += data

        # Average the sums across runs
        avg_data = sum_data / float(len(indices))
        avg_img = nib.Nifti1Image(avg_data, affine=affine, header=header)
        averaged_betas[cond_name] = avg_img

    return averaged_betas


def load_roi_voxel_data(subject_id, atlas_data, unique_rois):
    """
    Loads and extracts ROI-level voxel data from averaged beta images for one subject.

    1. Averages the subject's beta images (via `get_spm_info`).
    2. For each ROI, extracts a matrix of voxel intensities for every condition.

    Parameters
    ----------
    subject_id : str
        Subject identifier (e.g., '001').
    atlas_data : ndarray
        3D array containing integer ROI labels for each voxel.
    unique_rois : list of int
        The unique ROI labels in `atlas_data` (non-zero).

    Returns
    -------
    roi_vox_data_lists : dict
        Keys: ROI labels (int). Values: 2D ndarray of shape (n_conditions, n_voxels_in_ROI).
    conditions : list of str
        Sorted list of condition names to index the rows of each matrix in `roi_vox_data_lists`.
    """
    logger.info(f"[Subject {subject_id}] Loading ROI voxel data...")
    averaged_betas = get_spm_info(subject_id)
    conditions = sorted(averaged_betas.keys())
    n_conditions = len(conditions)
    logger.debug(f"[Subject {subject_id}] Found conditions: {conditions}")

    # Initialize a data structure to hold voxel data per ROI
    roi_vox_data_lists = {}
    for roi_label in unique_rois:
        roi_mask = (atlas_data == roi_label)
        n_voxels = np.sum(roi_mask)
        if n_voxels == 0:
            raise ValueError(f"[ERROR] ROI {roi_label} has 0 voxels in the atlas!")
        roi_vox_data_lists[roi_label] = np.zeros((n_conditions, n_voxels), dtype=np.float32)

    # Fill the data structure with averaged beta values
    for cond_idx, cond_name in enumerate(conditions):
        beta_img = averaged_betas[cond_name]
        beta_data = beta_img.get_fdata()

        for roi_label in unique_rois:
            roi_mask = (atlas_data == roi_label)
            roi_vox_values = beta_data[roi_mask]
            roi_vox_data_lists[roi_label][cond_idx, :] = roi_vox_values

    logger.info(f"[Subject {subject_id}] Completed ROI voxel extraction.")
    return roi_vox_data_lists


def compute_entropy_pr_per_roi(roi_data):
    """
    Computes the participation ratio (PR) for a given ROI data matrix.

    Calculation:
    - Computes the PCA on the input data matrix (shape: n_observations x n_voxels).
    - The PR is computed from the explained variance of each component.

    PR = (Sum of variances)^2 / (Sum of variances^2)

    If the ROI data is degenerate (empty or all NaN), returns np.nan.

    Parameters
    ----------
    roi_data : ndarray
        2D array of shape (n_conditions, n_voxels_in_ROI).

    Returns
    -------
    pr_val : float or np.nan
        The participation ratio for this ROI, or NaN if the data is invalid.
    """
    import numpy as np
    from mftma.manifold_analysis_correlation import manifold_analysis_corr
    # Remove any voxels that are entirely NaN
    valid_voxel_mask = ~np.isnan(roi_data).all(axis=0)
    roi_data = roi_data[:, valid_voxel_mask]

    if roi_data.size == 0:
        return np.nan

    pca = PCA().fit(roi_data)
    var = pca.explained_variance_
    if len(var) == 0:
        return np.nan

    pr_val = (var.sum() ** 2) / np.sum(var ** 2)
    return pr_val


def fdr_ttest(group1_vals, group2_vals, roi_labels, alpha=0.05):
    """
    Performs independent two-sample t-tests for each ROI, then applies FDR correction.

    Parameters
    ----------
    group1_vals : ndarray
        2D array of shape (n_subjects_group1, n_rois).
    group2_vals : ndarray
        2D array of shape (n_subjects_group2, n_rois).
    roi_labels : list or ndarray
        Labels corresponding to each column in `group1_vals` and `group2_vals`.
    alpha : float
        Significance level for FDR correction.

    Returns
    -------
    df : pd.DataFrame
        Columns: [ROI_Label, t_stat, p_val, p_val_fdr, significant_fdr, significant]
        - 'significant_fdr': True if p_val_fdr < alpha
        - 'significant': True if raw p_val < 0.05
    """
    n_rois = group1_vals.shape[1]
    tvals = np.zeros(n_rois)
    pvals = np.ones(n_rois)

    for i in range(n_rois):
        g1 = group1_vals[:, i]
        g2 = group2_vals[:, i]
        g1 = g1[~np.isnan(g1)]
        g2 = g2[~np.isnan(g2)]

        if len(g1) < 2 or len(g2) < 2:
            tvals[i] = 0.0
            pvals[i] = 1.0
        else:
            t_stat, p_val = ttest_ind(np.sort(g1), np.sort(g2), nan_policy='omit')
            tvals[i] = t_stat
            pvals[i] = p_val

    # FDR correction
    reject, pvals_fdr, _, _ = multipletests(pvals, alpha=alpha, method='fdr_bh')

    df = pd.DataFrame({
        "ROI_Label": roi_labels,
        "t_stat": tvals,
        "p_val": pvals,
        "p_val_fdr": pvals_fdr,
        "significant_fdr": reject,
        "significant": pvals < 0.05,
    })
    return df


def plot_all_rois_with_significance(
    stat_df,
    measure_name,
    group_means_df,
    expert_vals,
    nonexpert_vals,
    output_dir,
    use_fdr=True,
    sort_by="roi",
    custom_colors=CUSTOM_COLORS
):
    """
    Plots ROI results for a given measure (e.g., Participation Ratio),
    showing the difference between groups (Experts - Novices) for all ROIs.

    For each ROI:
    1. A bar (or box) shows the distribution of subject-level differences.
    2. Asterisks (*) mark significant ROIs based on FDR or raw p-values.

    Parameters
    ----------
    stat_df : DataFrame
        Contains T-test results (ROI_Label, p_val, p_val_fdr, etc.).
    measure_name : str
        Name of the measure (e.g., "PR").
    group_means_df : DataFrame
        Contains the mean values for experts/nonexperts per ROI (ROI_Label, measure columns).
    expert_vals : ndarray
        Subject-level data for experts, shape (n_experts, n_rois).
    nonexpert_vals : ndarray
        Subject-level data for non-experts, shape (n_nonexperts, n_rois).
    output_dir : str
        Path to the directory where the plot will be saved.
    use_fdr : bool
        Whether to use the FDR-corrected p-values for significance.
    sort_by : str
        "roi" to sort by ROI_Label ascending, or "diff" to sort by group difference.
    custom_colors : list of str or None
        Custom color palette for the bars. If None, Seaborn defaults are used.

    Returns
    -------
    None
        Saves the figure to the specified `output_dir`.
    """
    sns.set_style("white", {'axes.grid': False})

    # A user-friendly mapping from ROI integer labels to textual names
    roi_name_map = {
        1: "Primary Visual",
        2: "Early Visual",
        3: "Dorsal Stream Visual",
        4: "Ventral Stream Visual",
        5: "MT+ Complex",
        6: "Somatosensory and Motor",
        7: "Paracentral Lobular and Mid Cing",
        8: "Premotor",
        9: "Posterior Opercular",
        10: "Early Auditory",
        11: "Auditory Association",
        12: "Insular and Frontal Opercular",
        13: "Medial Temporal",
        14: "Lateral Temporal",
        15: "Temporo-Parieto Occipital Junction",
        16: "Superior Parietal",
        17: "Inferior Parietal",
        18: "Posterior Cing",
        19: "Anterior Cing and Medial Prefrontal",
        20: "Orbital and Polar Frontal",
        21: "Inferior Frontal",
        22: "Dorsolateral Prefrontal"
    }

    # Merge test stats with mean data
    merged = pd.merge(stat_df, group_means_df, on="ROI_Label", how="inner")

    # Collect differences per subject per ROI
    diffs = []
    sig_vals = []
    for _, row in merged.iterrows():
        roi_label = row["ROI_Label"]
        i_roi = np.where(stat_df["ROI_Label"] == roi_label)[0][0]
        expert_data = expert_vals[:, i_roi]
        nonexpert_data = nonexpert_vals[:, i_roi]
        diff_array = np.sort(expert_data) - np.sort(nonexpert_data)
        diffs.append(diff_array)

        # Check significance
        pval_ = row["p_val_fdr"] if use_fdr else row["p_val"]
        sig_vals.append(pval_ < 0.05)

    merged["diff"] = diffs
    merged["is_significant"] = sig_vals
    merged["ROI_Name"] = merged["ROI_Label"].map(roi_name_map)

    # Sort the merged dataframe
    if sort_by == "roi":
        merged = merged.sort_values("ROI_Label")
    elif sort_by == "diff":
        merged = merged.sort_values("diff", ascending=False)

    # Prepare figure
    bar_width = 0.5
    n_rois = merged.shape[0]
    fig_width = max(6, bar_width * n_rois)
    fontsize = min(20, max(6, fig_width * 1.1))

    fig, ax = plt.subplots(figsize=(fig_width, 7))
    x_coords = np.arange(len(merged))

    # Build palette for ROI labels
    if custom_colors is not None:
        roi_labels = merged["ROI_Label"].unique()
        palette_dict = {
            str(roi): custom_colors[i % len(custom_colors)]
            for i, roi in enumerate(roi_labels)
        }
    else:
        palette_dict = sns.color_palette("husl", len(merged["ROI_Label"].unique()))

    # Convert wide diffs data into a long-form DataFrame for plotting
    long_diff_data = []
    for _, row in merged.iterrows():
        for val in row["diff"]:
            long_diff_data.append({
                "ROI_Label": row["ROI_Label"],
                "ROI_Name": row["ROI_Name"],
                "diff": val
            })
    long_df = pd.DataFrame(long_diff_data)
    long_df["ROI_Label"] = long_df["ROI_Label"].astype(str)

    # Plot bar (or box) with the distribution of subject-level differences
    sns.barplot(
        x="ROI_Name", y="diff", hue="ROI_Label", data=long_df,
        palette=palette_dict, ax=ax, legend=False
    )

    # Overlay scatter of the same data
    sns.stripplot(
        x="ROI_Name", y="diff", hue="ROI_Label", data=long_df,
        edgecolor="k", linewidth=0.2, jitter=True, zorder=3,
        size=5, ax=ax, color="grey", alpha=0.3, legend=False
    )

    # Mark significant ROIs with an asterisk
    for i, row in enumerate(merged.itertuples()):
        y_vals = np.array(row.diff)
        if row.is_significant:
            y_mean = np.mean(y_vals)
            offset = 0.1 * (1 if y_mean >= 0 else -1)
            ax.text(
                x_coords[i], y_mean + offset, "*",
                ha="center",
                va="bottom" if y_mean >= 0 else "top",
                fontsize=24, color="red"
            )

    # Format the axes and title
    ax.set_xticks(x_coords)
    ax.set_xticklabels(
        merged["ROI_Name"],
        rotation=30,
        ha="right",
        fontsize=fontsize + 2
    )
    ax.axhline(0, color="black", linestyle="--", linewidth=1)
    ax.set_ylabel("Participation Ratio  Δ", fontsize=fontsize + 2)
    ax.set_xlabel("", fontsize=fontsize + 2)
    ax.set_title(
        f"Participation Ratio (Experts - Novices)\n({'FDR' if use_fdr else 'Raw'} p < 0.05)",
        fontsize=fontsize + 6
    )

    # Clean up spines
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.5)
    ax.spines["bottom"].set_linewidth(0.5)
    ax.spines["bottom"].set_visible(False)
    ax.tick_params(axis="y", labelsize=fontsize + 2)

    # Map each ROI to the color actually used for its bar
    bar_colors = {
        roi: patch.get_facecolor()
        for roi, patch in zip(merged["ROI_Name"].unique(), ax.patches)
    }    # Optionally color ROI labels only if they're significant

    for label_obj in ax.get_xticklabels():
        cortex_name = label_obj.get_text()
        roi_mask = merged["ROI_Name"] == cortex_name
        row_sig = merged.loc[roi_mask, "is_significant"]
        label_obj.set_color("grey" if row_sig.empty or not row_sig.iloc[0] else bar_colors.get(cortex_name, "black"))

    plt.tight_layout()
    fname = os.path.join(output_dir, f"all_rois_{measure_name}_scatter.png")
    plt.savefig(fname, dpi=300)
    plt.show()
    plt.close()


def process_subject(subject_id, atlas_data, unique_rois):
    """
    Pipeline for processing a single subject:
      1) Load & extract ROI-level voxel data.
      2) Compute the Participation Ratio (PR) for each ROI.

    Parameters
    ----------
    subject_id : str
        Subject identifier (e.g., '001').
    atlas_data : ndarray
        3D array of ROI labels (e.g., from a parcellation).
    unique_rois : ndarray
        Unique ROI labels in 'atlas_data'.

    Returns
    -------
    subj_pr : ndarray of shape (n_rois,)
        Contains the PR value for each ROI, in the same order as `unique_rois`.
    """
    logger.info(f"[Subject {subject_id}] Starting process_subject()")
    roi_vox_data = load_roi_voxel_data(subject_id, atlas_data, unique_rois)
    logger.debug(f"[Subject {subject_id}] ROI voxel data loaded.")

    n_rois = len(unique_rois)
    subj_pr = np.full(n_rois, np.nan, dtype=np.float32)

    for ri, roi_label in enumerate(unique_rois):
        data_2d = roi_vox_data[roi_label]
        if data_2d is None or data_2d.size == 0:
            logger.debug(f"[Subject {subject_id}, ROI {roi_label}] Empty or no valid data.")
            subj_pr[ri] = np.nan
        else:
            pr_val = compute_entropy_pr_per_roi(data_2d)
            subj_pr[ri] = pr_val

    logger.info(f"[Subject {subject_id}] Done processing. Returning PR array.")
    return subj_pr


##############################################################################
#                                  MAIN SCRIPT
##############################################################################

def main():
    """
    Main analysis pipeline:
      1) Load atlas and ROI labels.
      2) Process Expert and Non-Expert subjects in parallel.
      3) Compute group-level statistics with FDR correction.
      4) Generate summary CSVs and significance plots.
    """
    logger.info("Starting main pipeline...")

    # Create an output directory with a unique run ID
    output_dir = f"results/{create_run_id()}_manifold"
    os.makedirs(output_dir, exist_ok=True)
    save_script_to_file(output_dir)  # Save this script for reproducibility

    # Load the atlas file
    logger.info("Loading atlas...")
    atlas_img = nib.load(ATLAS_FILE)
    atlas_data = atlas_img.get_fdata().astype(int)
    unique_rois = np.unique(atlas_data)
    unique_rois = unique_rois[unique_rois != 0]
    n_rois = len(unique_rois)
    logger.info(f"Found {n_rois} non-zero ROI labels in the atlas.")

    # Expert subjects
    logger.info("Processing EXPERT subjects...")
    if USE_PARALLEL:
        expert_results = Parallel(n_jobs=-1, verbose=5)(
            delayed(process_subject)(sub_id, atlas_data, unique_rois)
            for sub_id in EXPERT_SUBJECTS
        )
    else:
        expert_results = []
        for sub_id in EXPERT_SUBJECTS:
            expert_results.append(process_subject(sub_id, atlas_data, unique_rois))
    expert_pr_arr = np.array(expert_results)
    logger.info(f"Expert arrays shaped: PR = {expert_pr_arr.shape}")

    # Non-expert subjects
    logger.info("Processing NON-EXPERT subjects...")
    if USE_PARALLEL:
        nonexpert_results = Parallel(n_jobs=-1, verbose=5)(
            delayed(process_subject)(sub_id, atlas_data, unique_rois)
            for sub_id in NONEXPERT_SUBJECTS
        )
    else:
        nonexpert_results = []
        for sub_id in NONEXPERT_SUBJECTS:
            nonexpert_results.append(process_subject(sub_id, atlas_data, unique_rois))
    nonexpert_pr_arr = np.array(nonexpert_results)
    logger.info(f"Non-expert arrays shaped: PR = {nonexpert_pr_arr.shape}")

    # Group-level T-tests
    logger.info("Performing group-level comparisons for PR...")
    pr_stats_df = fdr_ttest(
        expert_pr_arr,
        nonexpert_pr_arr,
        unique_rois,
        alpha=ALPHA_FDR
    )

    # Save stats
    pr_stats_path = os.path.join(output_dir, "roi_pr_stats.csv")
    pr_stats_df.to_csv(pr_stats_path, index=False)
    logger.info(f"Saved PR stats to {pr_stats_path}")

    # Summaries
    expert_pr_mean = np.nanmean(expert_pr_arr, axis=0)
    nonexpert_pr_mean = np.nanmean(nonexpert_pr_arr, axis=0)
    summary_df = pd.DataFrame({
        "ROI_Label": unique_rois,
        "PR_ExpertMean": expert_pr_mean,
        "PR_NonExpertMean": nonexpert_pr_mean
    })
    summary_csv = os.path.join(output_dir, "roi_group_means.csv")
    summary_df.to_csv(summary_csv, index=False)
    logger.info(f"Saved ROI group means to {summary_csv}")

    # Plot all ROIs (PR)
    logger.info("Plotting ROI-wise differences for PR with significance markers...")
    plot_all_rois_with_significance(
        stat_df=pr_stats_df,
        measure_name="PR",
        group_means_df=summary_df,
        expert_vals=expert_pr_arr,
        nonexpert_vals=nonexpert_pr_arr,
        output_dir=output_dir,
        use_fdr=use_fdr,
        sort_by="roi"  # could also use "diff"
    )

    logger.info(f"All done. Results and figures saved in: {output_dir}")


if __name__ == "__main__":
    main()
