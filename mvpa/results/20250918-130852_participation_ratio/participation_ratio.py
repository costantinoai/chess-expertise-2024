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

ROI_NAME_MAP = {
    1: "Primary Visual", 2: "Early Visual", 3: "Dorsal Stream Visual",
    4: "Ventral Stream Visual", 5: "MT+ Complex", 6: "Somatosensory and Motor",
    7: "Paracentral Lobular and Mid Cing", 8: "Premotor", 9: "Posterior Opercular",
    10: "Early Auditory", 11: "Auditory Association", 12: "Insular and Frontal Opercular",
    13: "Medial Temporal", 14: "Lateral Temporal", 15: "Temporo-Parieto Occipital Junction",
    16: "Superior Parietal", 17: "Inferior Parietal", 18: "Posterior Cing",
    19: "Anterior Cing and Medial Prefrontal", 20: "Orbital and Polar Frontal",
    21: "Inferior Frontal", 22: "Dorsolateral Prefrontal"
}

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
    Welch tests per ROI + effect sizes + CIs, using centralized helpers.
    Returns a DataFrame with:
      ROI_Label, t_stat, p_val, p_val_fdr, significant_fdr, significant,
      dof, cohen_d, mean_diff, ci95_low, ci95_high
    """
    results = []

    for i, roi in enumerate(roi_labels):
        g1 = group1_vals[:, i]
        g2 = group2_vals[:, i]
        g1 = g1[~np.isnan(g1)]
        g2 = g2[~np.isnan(g2)]

        if g1.size < 2 or g2.size < 2:
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

        # Use centralized Welch helper for mean diff + CI + p + df
        mean_diff, ci_lo, ci_hi, p_raw, df = ci_of_diff_welch(g1, g2, conf=0.95)

        # For t-stat from Welch components (optional; keep SciPy for robustness)
        res = ttest_ind(g1, g2, equal_var=False, nan_policy="omit")

        d = compute_effsize(g1, g2, eftype="cohen", paired=False)

        results.append({
            "ROI_Label": roi,
            "t_stat": float(res.statistic),
            "p_val": float(p_raw),
            "dof": float(df),
            "cohen_d": float(d),
            "mean_diff": float(mean_diff),
            "ci95_low": float(ci_lo),
            "ci95_high": float(ci_hi)
        })

    df = pd.DataFrame(results)

    # FDR (BH)
    corrected_p = df["p_val"].copy()
    corrected_p[df["p_val"].isna()] = 1.0
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


def plot_grouped_roi_bars_with_dots_from_results(
    results_df, measure_name, output_dir,
    use_fdr=True, sort_by="roi", custom_colors=CUSTOM_COLORS
):
    """
    Display-only plotting: reads everything from `results_df`.
    - Group mean CIs: expert_* and novice_* columns
    - Group difference stars: p_fdr (if use_fdr) else p_raw
    No computation of CIs or tests happens here.
    """
    import seaborn as sns
    import matplotlib.pyplot as plt
    import numpy as np

    # Sort
    if sort_by == "roi":
        df = results_df.sort_values("ROI_Label", key=natsorted).copy()
    elif sort_by == "diff":
        df = results_df.sort_values("delta_mean", ascending=False).copy()
    else:
        df = results_df.copy()

    roi_names  = df["ROI_Name"].values
    x_coords   = np.arange(len(df))

    # choose which p to threshold for stars
    if use_fdr:
        pvals = df["p_fdr"].values
    else:
        pvals = df["p_raw"].values

    is_sig = np.array([False if np.isnan(p) else (p < ALPHA_FDR) for p in pvals])

    # color palette
    if custom_colors is not None:
        palette_dict = {name: custom_colors[i % len(custom_colors)] for i, name in enumerate(roi_names)}
    else:
        palette = sns.color_palette("husl", len(roi_names))
        palette_dict = {name: palette[i] for i, name in enumerate(roi_names)}

    # === PLOT 1: bars of group means with CIs ===
    bar_width = 0.35
    sig_offset = 0.5
    asterisk_fontsize = plt.rcParams["font.size"] * 1.2

    fig, ax = plt.subplots()

    for i, name in enumerate(roi_names):
        row = df.iloc[i]
        color = palette_dict[name]

        # Expert bars
        e_mean = row["expert_mean"]
        e_lo   = row["expert_ci_low"]
        e_hi   = row["expert_ci_high"]
        e_err  = ci_to_errbar(e_mean, e_lo, e_hi)

        ax.bar(x_coords[i] - bar_width/2, e_mean, bar_width,
               yerr=e_err, capsize=4, color=color, edgecolor="black",
               label="Expert" if i == 0 else "", zorder=2)

        # Novice bars
        n_mean = row["novice_mean"]
        n_lo   = row["novice_ci_low"]
        n_hi   = row["novice_ci_high"]
        n_err  = ci_to_errbar(n_mean, n_lo, n_hi)

        ax.bar(x_coords[i] + bar_width/2, n_mean, bar_width,
               yerr=n_err, capsize=4, color=color, edgecolor="black", hatch="//",
               label="Novice" if i == 0 else "", zorder=2)

        # significance star based on chosen p
        if is_sig[i]:
            y_top = max(e_hi, n_hi)
            y_line = y_top + sig_offset * 1.2
            y_ast  = y_line + sig_offset * 0.05
            ax.plot([x_coords[i] - bar_width/2, x_coords[i] + bar_width/2],
                    [y_line, y_line], color="black", linewidth=1.2)
            ax.text(x_coords[i], y_ast, "*", ha="center", va="bottom",
                    fontsize=asterisk_fontsize, zorder=5)

    ax.set_xticks(x_coords)
    ax.set_xticklabels(roi_names, rotation=30, ha="right")
    ax.set_ylabel("Participation Ratio")
    ax.set_title(f"Participation Ratio by ROI ({'FDR' if use_fdr else 'Raw'} p < {ALPHA_FDR})")
    ax.axhline(0, color="black", linestyle="--", linewidth=1)

    # tint tick labels by significance
    for tick, name, sig in zip(ax.get_xticklabels(), roi_names, is_sig):
        tick.set_color(palette_dict[name] if sig else "lightgrey")

    # y-lims from CIs
    y_min = np.nanmin(np.r_[df["expert_ci_low"].values,  df["novice_ci_low"].values])
    y_max = np.nanmax(np.r_[df["expert_ci_high"].values, df["novice_ci_high"].values])
    if np.any(is_sig):
        y_max = y_max + sig_offset
    ax.set_ylim(y_min - 0.1, y_max + (abs(y_max) * 0.1))

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

    # === PLOT 2: Δ bars with CIΔ (already computed in pr_stats_df) ===
    fig, ax = plt.subplots()
    sns.barplot(
        x="ROI_Name", y="delta_mean", data=df,
        palette=[palette_dict[name] for name in roi_names], ax=ax
    )

    for i, row in df.iterrows():
        d_mean = row["delta_mean"]
        d_lo   = row["delta_ci_low"]
        d_hi   = row["delta_ci_high"]
        ax.errorbar(
            x=i, y=d_mean,
            yerr=[[d_mean - d_lo], [d_hi - d_mean]],
            fmt='none', ecolor='black', elinewidth=1.5, capsize=4, zorder=2
        )

    # significance stars for delta
    for i, row in df.iterrows():
        if is_sig[i]:
            if row["delta_mean"] >= 0:
                y_pos = row["delta_ci_high"] + sig_offset
                va = "bottom"
            else:
                y_pos = row["delta_ci_low"] - sig_offset
                va = "top"
            ax.text(i, y_pos, "*", ha="center", va=va,
                    fontsize=asterisk_fontsize, zorder=5)

    ax.set_xticks(x_coords)
    ax.set_xticklabels(roi_names, rotation=30, ha="right")
    ax.set_ylabel("ΔPR (Experts - Novices)")
    ax.set_title(f"Participation Ratio Differences ({'FDR' if use_fdr else 'Raw'} p < {ALPHA_FDR})", pad=20)
    ax.axhline(0, color="black", linestyle="--", linewidth=1)

    # tint tick labels by significance
    for label in ax.get_xticklabels():
        name = label.get_text()
        i = np.where(roi_names == name)[0][0]
        label.set_color("lightgrey" if not is_sig[i] else palette_dict.get(name, "black"))

    # let the data define the window (avoid fixed -12..12 which may be inappropriate for PR)
    y_min = np.nanmin(df["delta_ci_low"].values)
    y_max = np.nanmax(df["delta_ci_high"].values)
    ax.set_ylim(y_min - 0.1, y_max + 0.1)

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

# --- CI helpers (centralized) ---
def ci_of_mean(data_1d, conf=0.95):
    """
    Mean and 95% CI using the same method as your plots:
    SciPy one-sample t against 0 -> .confidence_interval(conf).
    Returns (mean, ci_low, ci_high) or (nan, nan, nan) if empty.
    """
    x = np.asarray(data_1d)
    x = x[~np.isnan(x)]
    if x.size == 0:
        return np.nan, np.nan, np.nan
    mean_val = float(np.mean(x))
    ci = ttest_1samp(x, popmean=0).confidence_interval(confidence_level=conf)
    return mean_val, float(ci.low), float(ci.high)

def ci_of_diff_welch(x, y, conf=0.95):
    """
    Mean difference CI via Welch t-test (matches your current difference CIs).
    Returns (mean_diff, ci_low, ci_high, p, df).
    """
    x = x[~np.isnan(x)]
    y = y[~np.isnan(y)]
    if x.size < 2 or y.size < 2:
        return np.nan, np.nan, np.nan, np.nan, np.nan
    res = ttest_ind(x, y, equal_var=False, nan_policy="omit")
    ci = res.confidence_interval(confidence_level=conf)
    mean_diff = float(np.mean(x) - np.mean(y))
    return mean_diff, float(ci.low), float(ci.high), float(res.pvalue), float(res.df)

def ci_to_errbar(mean_val, ci_low, ci_high):
    """
    Convert mean+CI to symmetric matplotlib errorbar tuple [[lower],[upper]].
    """
    if np.any(np.isnan([mean_val, ci_low, ci_high])):
        return [[np.nan], [np.nan]]
    return [[mean_val - ci_low], [ci_high - mean_val]]


def make_pr_multicolumn_table_from_results(
    results_df, output_dir,
    filename_tex="roi_pr_table.tex",
    filename_csv="roi_pr_table.csv",
    include_fdr=True,
    print_to_console=True
):
    """
    Display-only LaTeX table builder. Reads everything from `results_df`.
    No testing or CI computation happens here.
    """
    def fmt_ci(lo, hi):
        if np.any(np.isnan([lo, hi])):
            return "[--, --]"
        return f"[{lo:.2f}, {hi:.2f}]"

    def fmt_val(v):
        return "--" if np.isnan(v) else f"{v:.2f}"

    def fmt_p(p):
        if np.isnan(p): return "--"
        if p < 0.001: return "$<.001$"
        return f"$={p:.3f}$"

    rows = []
    for _, r in results_df.iterrows():
        rows.append({
            "ROI": r["ROI_Name"],
            "Expert_mean": fmt_val(r["expert_mean"]),
            "Expert_CI":   fmt_ci(r["expert_ci_low"], r["expert_ci_high"]),
            "Novice_mean": fmt_val(r["novice_mean"]),
            "Novice_CI":   fmt_ci(r["novice_ci_low"], r["novice_ci_high"]),
            "Delta_mean":  fmt_val(r["delta_mean"]),
            "Delta_CI":    fmt_ci(r["delta_ci_low"], r["delta_ci_high"]),
            "p":           fmt_p(r["p_fdr"] if include_fdr else r["p_raw"])
        })
    df_out = pd.DataFrame(rows)

    p_label = "$p_{\\mathrm{FDR}}$" if include_fdr else "$p$"
    header = (
        "\\begin{table}[p]\n\\centering\n"
        "\\resizebox{\\linewidth}{!}{%\n"
        "\\begin{tabular}{lcc|cc|cc|c}\n\\toprule\n"
        "\\multirow{2}{*}{ROI}\n"
        "  & \\multicolumn{2}{c|}{Experts}\n"
        "  & \\multicolumn{2}{c|}{Novices}\n"
        "  & \\multicolumn{2}{c|}{Experts$-$Novices}\n"
        f"  & {p_label} \\\\\n"
        "  & Mean & 95\\% CI"
        "  & Mean & 95\\% CI"
        "  & $\\Delta$ & 95\\% CI"
        "  &  \\\\\n\\midrule\n"
    )

    body = "\n".join(
        f"{r['ROI']} & {r['Expert_mean']} & {r['Expert_CI']} "
        f"& {r['Novice_mean']} & {r['Novice_CI']} "
        f"& {r['Delta_mean']} & {r['Delta_CI']} "
        f"& {r['p']} \\\\"
        for _, r in df_out.iterrows()
    )
    p_caption = "FDR-corrected $p$" if include_fdr else "raw $p$"
    footer = (
        "\n\\bottomrule\n\\end{tabular}\n}\n"
        "\\caption{Participation Ratio (PR): group means (95\\% CI), group differences (Welch; 95\\% CI), "
        + f"and {p_caption} per ROI." + "}\n"
        "\\label{tab:pr_multicolumn}\n\\end{table}\n"
    )



    latex_table = header + body + footer
    if print_to_console:
        print(latex_table)

    tex_out = os.path.join(output_dir, filename_tex)
    with open(tex_out, "w") as f:
        f.write(latex_table)
    print(f"Saved LaTeX table: {tex_out}")

    csv_out = os.path.join(output_dir, filename_csv)
    df_out.to_csv(csv_out, index=False)
    print(f"Saved CSV: {csv_out}")

    return df_out, latex_table

def build_pr_results_df(expert_vals, nonexpert_vals, roi_labels, pr_stats_df, roi_name_map=ROI_NAME_MAP):
    """
    Create a single, consolidated dataframe with EVERYTHING needed for plots/tables.
    No hypothesis tests are run here (those are already in pr_stats_df).
    This only adds descriptive group means + 95% CIs and ROI names,
    then merges with the Welch results you already computed (Δ, CIΔ, p, p_FDR).

    Returns
    -------
    results_df : pd.DataFrame with columns:
        ROI_Label, ROI_Name,
        expert_mean, expert_ci_low, expert_ci_high,
        novice_mean, novice_ci_low, novice_ci_high,
        delta_mean, delta_ci_low, delta_ci_high,
        p_raw, p_fdr, significant_raw, significant_fdr
    """
    rows = []
    # We rely on your existing helper for CI of the mean
    for i, roi in enumerate(roi_labels):
        X = expert_vals[:, i]
        Y = nonexpert_vals[:, i]
        X = X[~np.isnan(X)]
        Y = Y[~np.isnan(Y)]

        # Descriptive CIs for each group (NO testing here)
        e_mean, e_lo, e_hi = ci_of_mean(X)
        n_mean, n_lo, n_hi = ci_of_mean(Y)

        rows.append({
            "ROI_Label": int(roi),
            "ROI_Name": roi_name_map.get(int(roi), f"ROI {int(roi)}"),
            "expert_mean": e_mean, "expert_ci_low": e_lo, "expert_ci_high": e_hi,
            "novice_mean": n_mean, "novice_ci_low": n_lo, "novice_ci_high": n_hi,
        })

    desc_df = pd.DataFrame(rows)

    # Rename pr_stats_df columns to a stable schema and merge (no new tests here)
    stats_df = pr_stats_df.rename(columns={
        "mean_diff": "delta_mean",
        "ci95_low": "delta_ci_low",
        "ci95_high": "delta_ci_high",
        "p_val": "p_raw",
        "p_val_fdr": "p_fdr",
    })

    keep_cols = [
        "ROI_Label", "delta_mean", "delta_ci_low", "delta_ci_high",
        "p_raw", "p_fdr", "significant", "significant_fdr"
    ]
    stats_df = stats_df[keep_cols].copy()

    results_df = desc_df.merge(stats_df, on="ROI_Label", how="left")

    # For convenience in plotting later:
    results_df["sig_for_plot"] = results_df["significant_fdr"].fillna(False).astype(bool)

    # Ensure stable ROI order
    results_df = results_df.sort_values("ROI_Label", key=natsorted).reset_index(drop=True)
    return results_df


##############################################################################
#                                  MAIN SCRIPT
##############################################################################

"""
Main analysis pipeline:
  1) Load atlas and ROI labels.
  2) Process Expert and Novice subjects in parallel.
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

# Novice subjects
logger.info("Processing Novice subjects...")
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
logger.info(f"Novice arrays shaped: PR = {nonexpert_pr_arr.shape}")

# Group-level T-tests
logger.info("Performing group-level comparisons for PR...")
pr_stats_df = fdr_ttest(
    expert_pr_arr,
    nonexpert_pr_arr,
    unique_rois,
    alpha=ALPHA_FDR
)

# Consolidate in a single df for exp, novices, diff
pr_results_df = build_pr_results_df(
    expert_vals=expert_pr_arr,
    nonexpert_vals=nonexpert_pr_arr,
    roi_labels=unique_rois,
    pr_stats_df=pr_stats_df,
    roi_name_map=ROI_NAME_MAP
)

# save it once for reuse
pr_stats_path = os.path.join(output_dir, "roi_pr_results_consolidated.csv")
pr_results_df.to_csv(pr_stats_path, index=False)
logger.info(f"Saved PR stats to {pr_stats_path}")

# # Plot all ROIs (PR)
logger.info("Plotting grouped barplots for PR with significance markers...")
plot_grouped_roi_bars_with_dots_from_results(
    results_df=pr_results_df,
    measure_name="PR",
    output_dir=output_dir,
    use_fdr=use_fdr,
    sort_by="roi",
    custom_colors=CUSTOM_COLORS
)

# Make multi-column LaTeX + CSV table for PR
make_pr_multicolumn_table_from_results(
    results_df=pr_results_df,
    output_dir=output_dir,
    include_fdr=True,
    filename_tex="roi_pr_table.tex",
    filename_csv="roi_pr_table.csv",
    print_to_console=True
)

logger.info(f"Figures and tables saved in: {output_dir}")
