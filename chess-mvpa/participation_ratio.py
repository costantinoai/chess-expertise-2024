import os
import re
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from natsort import natsorted

import nibabel as nib
import scipy.io as sio
from scipy.stats import ttest_ind, ttest_1samp
from statsmodels.stats.multitest import fdrcorrection

from sklearn.decomposition import PCA
from statsmodels.stats.multitest import multipletests
from pingouin import compute_effsize

from joblib import Parallel, delayed
import logging

from modules.helpers import create_run_id, save_script_to_file

# Plot styles
sns.set_style("white", {"axes.grid": False})
base_font_size = 28
plt.rcParams.update(
    {
        "font.family": "Ubuntu Condensed",
        "font.size": base_font_size,
        "axes.titlesize": base_font_size * 1.4,  # 36.4 ~ 36
        "axes.labelsize": base_font_size * 1.2,  # 31.2 ~ 31
        "xtick.labelsize": base_font_size,  # 26
        "ytick.labelsize": base_font_size,  # 26
        "legend.fontsize": base_font_size,  # 26
        "figure.figsize": (21, 11),  # wide figures
    }
)



##############################################################################
#                                 CONSTANTS
##############################################################################

# Define any constants or paths used throughout the script.
CUSTOM_COLORS = [
    "#a6cee3", "#a6cee3",  # Early Visual (2)
    "#1f78b4", "#1f78b4", "#1f78b4",  # Intermediate Visual (3)
    "#b2df8a", "#b2df8a", "#b2df8a", "#b2df8a",  # Sensorimotor (4)
    "#33a02c", "#33a02c", "#33a02c",  # Auditory (3)
    "#fb9a99", "#fb9a99",  # Temporal (2)
    "#e31a1c", "#e31a1c", "#e31a1c", "#e31a1c",  # Posterior (4)
    "#fdbf6f", "#fdbf6f", "#fdbf6f", "#fdbf6f"   # Anterior (4)
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
ATLAS_FILE = "/data/projects/chess/data/misc/templates/tpl-MNI152NLin2009cAsym_res-02_atlas-Glasser2016_desc-cortices_bilateral_resampled.nii"

ALPHA_FDR = 0.05      # alpha level for FDR correction
USE_PARALLEL = True   # Toggle parallelization
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
    Performs Welch's t-tests for each ROI and computes full stats:
    - t-statistic
    - p-value
    - FDR-corrected p-value
    - significance (raw and FDR)
    - degrees of freedom
    - Cohen's d (via pingouin)
    - 95% CI for the mean difference

    Parameters
    ----------
    group1_vals : ndarray
        Shape (n_subjects_group1, n_rois)
    group2_vals : ndarray
        Shape (n_subjects_group2, n_rois)
    roi_labels : list or array-like
        ROI labels (1D, length n_rois)
    alpha : float
        Alpha for FDR correction

    Returns
    -------
    df : pd.DataFrame
        Dataframe with columns:
        ['ROI_Label', 't_stat', 'p_val', 'p_val_fdr', 'significant_fdr',
         'significant', 'dof', 'cohen_d', 'mean_diff', 'ci95_low', 'ci95_high']
    """
    results = []

    for i, roi in enumerate(roi_labels):
        g1 = group1_vals[:, i]
        g2 = group2_vals[:, i]

        # Remove NaNs
        g1 = g1[~np.isnan(g1)]
        g2 = g2[~np.isnan(g2)]

        if len(g1) < 2 or len(g2) < 2:
            results.append({
                "ROI_Label": roi,
                "t_stat": np.nan,
                "p_val": np.nan,
                "p_val_fdr": np.nan,
                "significant_fdr": False,
                "significant": False,
                "dof": np.nan,
                "cohen_d": np.nan,
                "mean_diff": np.nan,
                "ci95_low": np.nan,
                "ci95_high": np.nan
            })
            continue

        # Welch's t-test
        res = ttest_ind(g1, g2, equal_var=False)
        ci = res.confidence_interval(confidence_level=0.95)
        mean_diff = np.mean(g1) - np.mean(g2)
        d = compute_effsize(g1, g2, eftype="cohen", paired=False)

        results.append({
            "ROI_Label": roi,
            "t_stat": res.statistic,
            "p_val": res.pvalue,
            "dof": res.df,
            "cohen_d": d,
            "mean_diff": mean_diff,
            "ci95_low": ci.low,
            "ci95_high": ci.high
        })

    df = pd.DataFrame(results)

    # FDR correction on valid p-values
    corrected_p = df["p_val"].copy()
    corrected_p[df["p_val"].isna()] = 1.0  # avoid nan breaking FDR
    reject, pval_fdr, _, _ = multipletests(corrected_p, alpha=alpha, method='fdr_bh')

    df["p_val_fdr"] = pval_fdr
    df["significant_fdr"] = reject
    df["significant"] = df["p_val"] < 0.05

    return df



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


def plot_grouped_roi_bars_with_dots(
    stat_df, measure_name, group_means_df,
    expert_vals, nonexpert_vals, output_dir,
    use_fdr=True, sort_by="roi", custom_colors=CUSTOM_COLORS
):
    """
    Plots ROI-wise bar plots with 95% CI and individual data points for Experts and Non-Experts.
    """

    # === Constants ===
    bar_width = 0.35
    sig_offset = 0.5
    asterisk_fontsize = plt.rcParams["font.size"] * 1.2

    roi_name_map = {
        1: "Primary Visual", 2: "Early Visual", 3: "Dorsal Stream Visual",
        4: "Ventral Stream Visual", 5: "MT+ Complex", 6: "Somatosensory and Motor",
        7: "Paracentral Lobular and Mid Cing", 8: "Premotor", 9: "Posterior Opercular",
        10: "Early Auditory", 11: "Auditory Association", 12: "Insular and Frontal Opercular",
        13: "Medial Temporal", 14: "Lateral Temporal", 15: "Temporo-Parieto Occipital Junction",
        16: "Superior Parietal", 17: "Inferior Parietal", 18: "Posterior Cing",
        19: "Anterior Cing and Medial Prefrontal", 20: "Orbital and Polar Frontal",
        21: "Inferior Frontal", 22: "Dorsolateral Prefrontal"
    }


    merged = pd.merge(stat_df, group_means_df, on="ROI_Label", how="inner")
    merged["ROI_Name"] = merged["ROI_Label"].map(roi_name_map)

    if sort_by == "roi":
        merged = merged.sort_values("ROI_Label", key=natsorted)
    elif sort_by == "diff":
        merged["diff"] = merged["PR_ExpertMean"] - merged["PR_NonExpertMean"]
        merged = merged.sort_values("diff", ascending=False)

    roi_names = merged["ROI_Name"].values
    roi_labels = merged["ROI_Label"].values
    x_coords = np.arange(len(roi_names))

    expert_means, expert_cis = [], []
    nonexpert_means, nonexpert_cis = [], []
    diff_means, diff_cis = [], []
    pvals = []

    # Collect all statistics first
    for roi_label in roi_labels:
        roi_idx = np.where(stat_df["ROI_Label"] == roi_label)[0][0]
        expert_data = expert_vals[:, roi_idx]
        nonexpert_data = nonexpert_vals[:, roi_idx]

        # Drop NaNs
        expert_data = expert_data[~np.isnan(expert_data)]
        nonexpert_data = nonexpert_data[~np.isnan(nonexpert_data)]

        # Group means
        expert_mean = np.mean(expert_data)
        nonexpert_mean = np.mean(nonexpert_data)
        expert_means.append(expert_mean)
        nonexpert_means.append(nonexpert_mean)

        # Confidence intervals for individual groups (1-sample t-tests)
        expert_ci = ttest_1samp(expert_data, popmean=0).confidence_interval(0.95)
        nonexpert_ci = ttest_1samp(nonexpert_data, popmean=0).confidence_interval(0.95)
        expert_cis.append(expert_ci)
        nonexpert_cis.append(nonexpert_ci)

        # Group difference (independent t-test)
        res = ttest_ind(expert_data, nonexpert_data, equal_var=False, nan_policy="omit")
        pvals.append(res.pvalue)
        diff_means.append(expert_mean - nonexpert_mean)
        diff_cis.append(res.confidence_interval(0.95))

    # FDR correction
    pvals = np.array(pvals)
    pvals_fdr = fdrcorrection(pvals, alpha=ALPHA_FDR)[1]
    is_sig = pvals_fdr < ALPHA_FDR if use_fdr else pvals < ALPHA_FDR

    # Color palette
    if custom_colors is not None:
        ordered = merged[["ROI_Label", "ROI_Name"]].drop_duplicates()
        ordered = ordered.sort_values("ROI_Label", key=natsorted)
        palette_dict = {
            row.ROI_Name: custom_colors[i % len(custom_colors)]
            for i, row in enumerate(ordered.itertuples(index=False))
        }
    else:
        palette_dict = {name: c for name, c in zip(roi_names, sns.color_palette("husl", len(roi_names)))}

    fig, ax = plt.subplots()

    for i, roi in enumerate(roi_names):
        color = palette_dict[roi]

        expert_err = [[expert_means[i] - expert_cis[i].low], [expert_cis[i].high - expert_means[i]]]
        nonexpert_err = [[nonexpert_means[i] - nonexpert_cis[i].low], [nonexpert_cis[i].high - nonexpert_means[i]]]

        ax.bar(x_coords[i] - bar_width / 2, expert_means[i], bar_width,
               yerr=expert_err, capsize=4, color=color, edgecolor='black',
               label="Expert" if i == 0 else "", zorder=2)

        ax.bar(x_coords[i] + bar_width / 2, nonexpert_means[i], bar_width,
               yerr=nonexpert_err, capsize=4, color=color, edgecolor='black',
               hatch="//", label="Novice" if i == 0 else "", zorder=2)

        if is_sig[i]:
            y_top = max(expert_cis[i].high, nonexpert_cis[i].high)
            y_line = y_top + sig_offset * 1.2
            y_ast = y_line + sig_offset * 0.05
            ax.plot([x_coords[i] - bar_width / 2, x_coords[i] + bar_width / 2],
                    [y_line, y_line], color="black", linewidth=1.2)
            ax.text(x_coords[i], y_ast, "*", ha="center", va="bottom",
                    fontsize=asterisk_fontsize, zorder=5)

    ax.set_xticks(x_coords)
    ax.set_xticklabels(roi_names, rotation=30, ha="right")
    ax.set_ylabel("Participation Ratio")
    ax.set_title(f"Participation Ratio by ROI ({'FDR' if use_fdr else 'Raw'} p < {ALPHA_FDR})")
    ax.axhline(0, color="black", linestyle="--", linewidth=1)

    for tick, roi, sig in zip(ax.get_xticklabels(), roi_names, is_sig):
        tick.set_color(palette_dict[roi] if sig else "lightgrey")

    y_min = min(ci.low for ci in expert_cis + nonexpert_cis)
    y_max = max(ci.high for ci in expert_cis + nonexpert_cis)
    if np.any(~np.isnan(is_sig)):
        y_min = min(y_min, y_min - sig_offset)
    if np.any(is_sig):
        y_max = max(y_max, y_max + sig_offset)
    ax.set_ylim(y_min - 0.1, y_max + (y_max * 0.1))

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.5)
    ax.spines["bottom"].set_linewidth(0.5)

    ax.legend(loc="upper right", bbox_to_anchor=(1, 1.15), ncol=2, frameon=False)

    plt.tight_layout()
    fname = os.path.join(output_dir, f"roi_bars_{measure_name}_grouped_with_dots.png")
    plt.savefig(fname, dpi=300, bbox_inches="tight")
    plt.show()
    plt.close()
    print(f"Saved: {fname}")


    # === SECOND PLOT: Difference with CI and significance ===

    # Build dataframe for plotting
    diff_data = []
    for i, roi in enumerate(roi_names):
        ci = diff_cis[i]
        diff_data.append({
            "ROI_Label": roi_labels[i],
            "ROI_Name": roi,
            "mean_diff": diff_means[i],
            "ci95_low": ci.low,
            "ci95_high": ci.high,
            "significant_fdr": is_sig[i]
        })

    diff_df = pd.DataFrame(diff_data)

    if sort_by == "roi":
        diff_df = diff_df.sort_values("ROI_Label", key=natsorted)
    elif sort_by == "diff":
        diff_df = diff_df.sort_values("mean_diff", ascending=False)

    x_coords = np.arange(len(diff_df))
    roi_names = diff_df["ROI_Name"].values

    fig, ax = plt.subplots()
    sns.barplot(
        x="ROI_Name", y="mean_diff", data=diff_df,
        palette=[palette_dict[name] for name in roi_names], ax=ax
    )

    # Error bars
    for i, row in enumerate(diff_df.itertuples()):
        ax.errorbar(
            x=i, y=row.mean_diff,
            yerr=[[row.mean_diff - row.ci95_low], [row.ci95_high - row.mean_diff]],
            fmt='none', ecolor='black', elinewidth=1.5, capsize=4, zorder=2
        )

    # Significance stars
    for i, row in enumerate(diff_df.itertuples()):
        if row.significant_fdr:
            if row.mean_diff >= 0:
                y_pos = row.ci95_high + sig_offset
                va = "bottom"
            else:
                y_pos = row.ci95_low - sig_offset
                va = "top"
            ax.text(x_coords[i], y_pos, "*", ha="center", va=va,
                    fontsize=asterisk_fontsize, zorder=5)

    # Axis formatting
    ax.set_xticks(x_coords)
    ax.set_xticklabels(roi_names, rotation=30, ha="right")
    ax.set_ylabel("ΔPR (Experts - Non-Experts)")
    ax.set_title(f"Participation Ratio Differences ({'FDR' if use_fdr else 'Raw'} p < {ALPHA_FDR})", pad=20)
    ax.axhline(0, color="black", linestyle="--", linewidth=1)

    # Color non-significant ticks
    for label in ax.get_xticklabels():
        roi = label.get_text()
        is_sig_roi = diff_df.loc[diff_df["ROI_Name"] == roi, "significant_fdr"].values[0]
        label.set_color("lightgrey" if not is_sig_roi else palette_dict.get(roi, "black"))

    # Y-limits
    # y_min = diff_df["ci95_low"].min() - sig_offset
    # y_max = diff_df["ci95_high"].max() + sig_offset
    ax.set_ylim(-12,12)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.5)
    ax.spines["bottom"].set_linewidth(0.5)
    ax.spines["bottom"].set_visible(False)

    plt.tight_layout()
    fname2 = os.path.join(output_dir, f"roi_diff_{measure_name}_bar_ci.png")
    plt.savefig(fname2, dpi=300, bbox_inches="tight")
    plt.show()
    plt.close()
    print(f"Saved: {fname2}")



##############################################################################
#                                  MAIN SCRIPT
##############################################################################

"""
Main analysis pipeline:
  1) Load atlas and ROI labels.
  2) Process Expert and Non-Expert subjects in parallel.
  3) Compute group-level statistics with FDR correction.
  4) Generate summary CSVs and significance plots.
"""
logger.info("Starting main pipeline...")

# Create an output directory with a unique run ID
output_dir = f"results/{create_run_id()}_participation_ratio"
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

# # Plot all ROIs (PR)
# # Plot difference scatter
# logger.info("Plotting ROI-wise differences for PR with significance markers...")
# plot_all_rois_with_significance(
#     stat_df=pr_stats_df,
#     measure_name="PR",
#     group_means_df=summary_df,
#     expert_vals=expert_pr_arr,
#     nonexpert_vals=nonexpert_pr_arr,
#     output_dir=output_dir,
#     use_fdr=use_fdr,
#     sort_by="roi"
# )

# Plot grouped barplots
logger.info("Plotting grouped barplots for PR with significance markers...")
plot_grouped_roi_bars_with_dots(
    stat_df=pr_stats_df,
    measure_name="PR",
    group_means_df=summary_df,
    expert_vals=expert_pr_arr,
    nonexpert_vals=nonexpert_pr_arr,
    output_dir=output_dir,
    use_fdr=use_fdr,
    sort_by="roi"
)

logger.info(f"All done. Results and figures saved in: {output_dir}")


# Clean and format table for LaTeX output
def format_latex_row(row):
    roi = int(row["ROI_Label"])
    t_stat = f"{row['t_stat']:.2f}"
    dof = f"{row['dof']:.1f}"
    p_val = row["p_val"]
    p_fmt = (
        "$< .001$" if p_val < 0.001 else f"$= {p_val:.3f}$"
    )
    ci_low = f"{row['ci95_low']:.2f}"
    ci_high = f"{row['ci95_high']:.2f}"
    return f"{roi} & $t({dof}) = {t_stat}$ & $p {p_fmt}$ & $[{ci_low}, {ci_high}]$ \\\\"

# Filter and format
rows = []
for _, row in pr_stats_df.iterrows():
    if pd.notnull(row["t_stat"]):
        rows.append(format_latex_row(row))

# LaTeX table construction
latex_table = "\\begin{tabular}{lccc}\n"
latex_table += "\\textbf{ROI} & \\textbf{t-test (df)} & \\textbf{p-value} & \\textbf{95\\% CI} \\\\\n"
latex_table += "\\hline\n"
latex_table += "\n".join(rows)
latex_table += "\n\\end{tabular}"

print(latex_table)
