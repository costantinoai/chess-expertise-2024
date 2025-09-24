# ==============================
# Participation Ratio (PR) pipeline
# ==============================
# This script:
#  1) loads SPM first-level betas,
#  2) extracts ROI-wise voxel matrices,
#  3) computes Participation Ratio (PR) per ROI per subject,
#  4) runs Welch tests (Experts vs Novices) with CIs and FDR,
#  5) consolidates results (group means + CIs + Δ + CIs + p, p_FDR),
#  6) plots figures from the consolidated results only,
#  7) builds a LaTeX multi-column table from the consolidated results.
#
# Design principle: analysis functions do all computation ONCE.
# Plotting/reporting functions only READ precomputed results (no re-computation).
# ==============================

import os                           # paths / saving
import re                           # regex for SPM regressor parsing
import numpy as np                  # numerics
import pandas as pd                 # tabular data / CSVs
import seaborn as sns               # plotting styles
import matplotlib.pyplot as plt     # plotting
from natsort import natsorted       # natural sort for ROI order

import nibabel as nib               # load NIfTI
import scipy.io as sio              # load SPM.mat (MATLAB)
from scipy.stats import ttest_ind, ttest_1samp  # Welch test + mean CI

from sklearn.decomposition import PCA           # PR from PCA spectrum
from statsmodels.stats.multitest import multipletests  # FDR (BH)
from pingouin import compute_effsize            # Cohen's d

from joblib import Parallel, delayed            # parallel subjects
import logging                                  # logging

from modules.helpers import create_run_id, save_script_to_file  # reproducibility

# ------------------------------
# Plot styles for consistent figures across the paper
# ------------------------------
sns.set_style("white", {"axes.grid": False})
base_font_size = 28
plt.rcParams.update({
    "font.family": "Ubuntu Condensed",
    "font.size": base_font_size,
    "axes.titlesize": base_font_size * 1.4,
    "axes.labelsize": base_font_size * 1.2,
    "xtick.labelsize": base_font_size,
    "ytick.labelsize": base_font_size,
    "legend.fontsize": base_font_size,
    "figure.figsize": (21, 11)
})

# ------------------------------
# Constants / configuration
# ------------------------------
CUSTOM_COLORS = [  # 22 entries to color ROIs by coarse families
    "#a6cee3", "#a6cee3",                    # Early Visual (2)
    "#1f78b4", "#1f78b4", "#1f78b4",         # Intermediate Visual (3)
    "#b2df8a", "#b2df8a", "#b2df8a", "#b2df8a",  # Sensorimotor (4)
    "#33a02c", "#33a02c", "#33a02c",         # Auditory (3)
    "#fb9a99", "#fb9a99",                    # Temporal (2)
    "#e31a1c", "#e31a1c", "#e31a1c", "#e31a1c",  # Posterior (4)
    "#fdbf6f", "#fdbf6f", "#fdbf6f", "#fdbf6f"   # Anterior (4)
]

# Subject lists (string IDs used in paths)
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

# Coarse ROI names (Glasser families merged bilaterally)
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

# Data locations
BASE_PATH = "/data/projects/chess/data/BIDS/derivatives/fmriprep-SPM_smoothed-NO_GS-FD-HMP_brainmasked/MNI/fmriprep-SPM-MNI/GLM"
SPM_FILENAME = "SPM.mat"
ATLAS_FILE = "/data/projects/chess/data/misc/templates/tpl-MNI152NLin2009cAsym_res-02_atlas-Glasser2016_desc-cortices_bilateral_resampled.nii"

# Stats settings
ALPHA_FDR = 0.05       # Benjamini–Hochberg FDR level
USE_PARALLEL = True    # parallelize across subjects
use_fdr = True         # use FDR-corrected p for stars/tables

# ------------------------------
# Logging config
# ------------------------------
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)  # set to DEBUG for more verbosity

# ==============================
# Helpers: load SPM betas & build ROI matrices
# ==============================
def get_spm_info(subject_id):
    """
    Load SPM.mat for a subject, find all condition regressors ("C1"..."C40"),
    and average the corresponding beta images across runs, returning a dict
    { 'C1': Nifti1Image, ..., 'C40': Nifti1Image }.

    Parameters
    ----------
    subject_id : str
        Subject code (e.g., '03').

    Returns
    -------
    averaged_betas : dict[str, nib.Nifti1Image]
        One averaged beta per condition.
    """
    # Path to subject's SPM.mat
    spm_mat_path = os.path.join(BASE_PATH, f"sub-{subject_id}", "exp", SPM_FILENAME)
    if not os.path.isfile(spm_mat_path):
        logger.error(f"SPM.mat not found for sub-{subject_id}: {spm_mat_path}")
        raise FileNotFoundError(spm_mat_path)

    # Load SPM struct
    spm_dict = sio.loadmat(spm_mat_path, struct_as_record=False, squeeze_me=True)
    SPM = spm_dict["SPM"]
    beta_info = SPM.Vbeta               # list-like of beta file refs
    regressor_names_full = SPM.xX.name  # names like "Sn(1) C1*bf(1)"
    pattern = r"Sn\(\d+\)\s+(.*?)\*bf\(1\)"  # capture "C1", "C2", ...

    # Map condition -> list of beta indices (across runs)
    condition_dict = {}
    for i, reg_name in enumerate(regressor_names_full):
        m = re.match(pattern, reg_name)
        if not m:
            continue
        cond_name = m.group(1)  # e.g., "C1"
        condition_dict.setdefault(cond_name, []).append(i)

    # Load and average the betas for each condition
    averaged_betas = {}
    spm_dir = SPM.swd if hasattr(SPM, 'swd') else os.path.dirname(spm_mat_path)
    for cond_name, indices in condition_dict.items():
        if not indices:  # safety
            continue
        sum_data, affine, header = None, None, None
        for idx in indices:
            # SPM stores filename in .fname (older versions may differ; we guard above)
            beta_fname = beta_info[idx].fname if hasattr(beta_info[idx], 'fname') else beta_info[idx].filename
            beta_path = os.path.join(spm_dir, beta_fname)
            img = nib.load(beta_path)
            data = img.get_fdata(dtype=np.float32)
            if sum_data is None:
                sum_data = np.zeros_like(data, dtype=np.float32)
                affine, header = img.affine, img.header
            sum_data += data
        avg_data = sum_data / float(len(indices))
        averaged_betas[cond_name] = nib.Nifti1Image(avg_data, affine=affine, header=header)

    return averaged_betas


def load_roi_voxel_data(subject_id, atlas_data, unique_rois):
    """
    Extract a (conditions x voxels) matrix for each ROI for one subject.

    Steps:
      1) average betas per condition (get_spm_info),
      2) for each ROI, grab beta values at ROI voxels for every condition.

    Returns
    -------
    roi_vox_data : dict[int, np.ndarray]
        roi_label -> array shape (n_conditions, n_voxels_in_roi)
    """
    logger.info(f"[Subject {subject_id}] Extracting ROI voxel matrices...")
    averaged_betas = get_spm_info(subject_id)
    conditions = sorted(averaged_betas.keys())  # ensure consistent order across subjects

    roi_vox_data = {}
    for roi_label in unique_rois:
        # Boolean ROI mask
        roi_mask = (atlas_data == roi_label)
        n_voxels = int(np.sum(roi_mask))
        if n_voxels == 0:
            raise ValueError(f"ROI {roi_label} has 0 voxels in atlas.")

        # Pre-allocate matrix (conditions x voxels)
        mat = np.zeros((len(conditions), n_voxels), dtype=np.float32)
        for ci, cname in enumerate(conditions):
            beta_data = averaged_betas[cname].get_fdata()
            mat[ci, :] = beta_data[roi_mask]
        roi_vox_data[roi_label] = mat

    logger.info(f"[Subject {subject_id}] ROI extraction done.")
    return roi_vox_data


# ==============================
# PR computation + statistics (Welch + FDR)
# ==============================
def compute_entropy_pr_per_roi(roi_data):
    """
    Compute Participation Ratio (PR) from the PCA spectrum of a ROI matrix.

    PR = (sum(λ))^2 / sum(λ^2)
    where λ are eigenvalues (explained variance) of the covariance.

    Parameters
    ----------
    roi_data : np.ndarray
        shape (n_conditions, n_voxels)

    Returns
    -------
    float
        PR value (np.nan if degenerate).
    """
    # Drop columns (voxels) that are entirely NaN
    valid = ~np.isnan(roi_data).all(axis=0)
    roi_data = roi_data[:, valid]
    if roi_data.size == 0:
        return np.nan

    # PCA on conditions x voxels
    var = PCA().fit(roi_data).explained_variance_
    if var.size == 0:
        return np.nan
    return float((var.sum() ** 2) / np.sum(var ** 2))


# --- Centralized CI helpers (reused everywhere) ---
def ci_of_mean(data_1d, conf=0.95):
    """
    Mean and 95% CI using SciPy's one-sample t CI (against 0).
    This yields mean ± t_{n-1,α/2} * SE(mean).

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
    Mean difference CI via Welch t-test (unequal variances).
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
    Convert mean+CI to symmetric errorbar tuple [[lower],[upper]] for Matplotlib.
    """
    if np.any(np.isnan([mean_val, ci_low, ci_high])):
        return [[np.nan], [np.nan]]
    return [[mean_val - ci_low], [ci_high - mean_val]]


def fdr_ttest(group1_vals, group2_vals, roi_labels, alpha=0.05):
    """
    Welch tests per ROI + effect sizes + CIs.
    This is the ONLY place where group-difference tests are computed.

    Returns
    -------
    pd.DataFrame with columns:
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
            # Not enough data to test
            results.append({
                "ROI_Label": int(roi),
                "t_stat": np.nan, "p_val": np.nan, "p_val_fdr": np.nan,
                "significant_fdr": False, "significant": False,
                "dof": np.nan, "cohen_d": np.nan,
                "mean_diff": np.nan, "ci95_low": np.nan, "ci95_high": np.nan
            })
            continue

        # Welch diff CI + p + df (single source of truth)
        mean_diff, ci_lo, ci_hi, p_raw, df = ci_of_diff_welch(g1, g2, conf=0.95)

        # For transparency, also record t-stat from SciPy's test
        welch = ttest_ind(g1, g2, equal_var=False, nan_policy="omit")

        # Cohen's d (unpaired)
        d = compute_effsize(g1, g2, eftype="cohen", paired=False)

        results.append({
            "ROI_Label": int(roi),
            "t_stat": float(welch.statistic),
            "p_val": float(p_raw),
            "dof": float(df),
            "cohen_d": float(d),
            "mean_diff": float(mean_diff),
            "ci95_low": float(ci_lo),
            "ci95_high": float(ci_hi)
        })

    df = pd.DataFrame(results)

    # Benjamini–Hochberg FDR correction on p-values
    corrected_p = df["p_val"].fillna(1.0).to_numpy()
    reject, pval_fdr, _, _ = multipletests(corrected_p, alpha=alpha, method='fdr_bh')

    df["p_val_fdr"] = pval_fdr
    df["significant_fdr"] = reject
    df["significant"] = df["p_val"] < 0.05

    return df


# ==============================
# Subject-level pipeline: compute PRs for all ROIs
# ==============================
def process_subject(subject_id, atlas_data, unique_rois):
    """
    For one subject:
      - build ROI voxel matrices
      - compute PR per ROI
    Returns an array of shape (n_rois,) aligned with unique_rois order.
    """
    logger.info(f"[Subject {subject_id}] Start")
    roi_vox_data = load_roi_voxel_data(subject_id, atlas_data, unique_rois)

    subj_pr = np.full(len(unique_rois), np.nan, dtype=np.float32)
    for ri, roi_label in enumerate(unique_rois):
        mat = roi_vox_data[roi_label]
        subj_pr[ri] = compute_entropy_pr_per_roi(mat) if (mat is not None and mat.size) else np.nan

    logger.info(f"[Subject {subject_id}] Done")
    return subj_pr


# ==============================
# Consolidate: descriptive CIs + merge stats (no new computations)
# ==============================
def build_pr_results_df(expert_vals, nonexpert_vals, roi_labels, pr_stats_df, roi_name_map=ROI_NAME_MAP):
    """
    Create a single dataframe with EVERYTHING needed downstream:
      - Descriptive group means + their 95% CIs (Experts, Novices) [no tests]
      - Group differences + CIs + p + p_FDR (from pr_stats_df)
      - ROI names and natural ordering

    Returns
    -------
    results_df : pd.DataFrame with columns:
        ROI_Label, ROI_Name,
        expert_mean, expert_ci_low, expert_ci_high,
        novice_mean, novice_ci_low, novice_ci_high,
        delta_mean, delta_ci_low, delta_ci_high,
        p_raw, p_fdr, significant, significant_fdr
    """
    # Compute descriptive CIs per group per ROI
    rows = []
    for i, roi in enumerate(roi_labels):
        X = expert_vals[:, i]
        Y = nonexpert_vals[:, i]
        X = X[~np.isnan(X)]
        Y = Y[~np.isnan(Y)]
        e_mean, e_lo, e_hi = ci_of_mean(X)
        n_mean, n_lo, n_hi = ci_of_mean(Y)
        rows.append({
            "ROI_Label": int(roi),
            "ROI_Name": roi_name_map.get(int(roi), f"ROI {int(roi)}"),
            "expert_mean": e_mean, "expert_ci_low": e_lo, "expert_ci_high": e_hi,
            "novice_mean": n_mean, "novice_ci_low": n_lo, "novice_ci_high": n_hi
        })
    desc_df = pd.DataFrame(rows)

    # Merge Welch test results from pr_stats_df (Δ, CIΔ, p, p_FDR)
    stats_df = pr_stats_df.rename(columns={
        "mean_diff": "delta_mean",
        "ci95_low": "delta_ci_low",
        "ci95_high": "delta_ci_high",
        "p_val": "p_raw",
        "p_val_fdr": "p_fdr",
    })[[
        "ROI_Label", "delta_mean", "delta_ci_low", "delta_ci_high",
        "p_raw", "p_fdr", "significant", "significant_fdr"
    ]].copy()

    results_df = desc_df.merge(stats_df, on="ROI_Label", how="left")
    results_df = results_df.sort_values("ROI_Label", key=natsorted).reset_index(drop=True)
    return results_df


# ==============================
# Plotting (display-only; no computations)
# ==============================
def plot_pr_combined_panel_from_results(
    results_df,
    measure_name,
    output_dir,
    use_fdr=True,
    sort_by="roi",
    custom_colors=CUSTOM_COLORS,
    fig_title=None,
):
    """
    Single, manuscript-ready 2-panel figure:
      Top:   Expert vs Novice mean bars with 95% CIs + significance markers
      Bottom: Δ = Expert - Novice bars with 95% CIΔ + significance markers

    Reads only from `results_df` (no statistical computation here).
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import os

    # ---------- Order data ----------
    if sort_by == "roi":
        df = results_df.sort_values("ROI_Label", key=natsorted).copy()
    elif sort_by == "diff":
        df = results_df.sort_values("delta_mean", ascending=False).copy()
    else:
        df = results_df.copy()

    roi_names = df["ROI_Name"].values
    x = np.arange(len(df))

    # ---------- Significance selection ----------
    pvals = df["p_fdr"].values if use_fdr else df["p_raw"].values
    is_sig = np.array([False if np.isnan(p) else (p < ALPHA_FDR) for p in pvals])

    # ---------- Palette ----------
    if custom_colors is not None:
        palette_dict = {name: custom_colors[i % len(custom_colors)] for i, name in enumerate(roi_names)}
    else:
        palette = sns.color_palette("husl", len(roi_names))
        palette_dict = {name: palette[i] for i, name in enumerate(roi_names)}

    # ---------- Figure layout (taller; equal panel heights) ----------
    fig = plt.figure(constrained_layout=True, figsize=(18, 14))  # same width, taller height
    gs = fig.add_gridspec(nrows=2, ncols=1, height_ratios=[2, 3])
    ax_top = fig.add_subplot(gs[0, 0])
    ax_bot = fig.add_subplot(gs[1, 0], sharex=ax_top)

    # Common styling
    bar_width = 0.35
    capsize = 4
    sig_offset = 0.5
    asterisk_fontsize = plt.rcParams["font.size"] * 1.2

    # Keep handles for a single, figure-level legend (so it won't overlap)
    legend_handles = []
    legend_labels = []

    # ---------- TOP: group means ----------
    for i, name in enumerate(roi_names):
        row = df.iloc[i]
        color = palette_dict[name]

        # Expert
        e_mean, e_lo, e_hi = row["expert_mean"], row["expert_ci_low"], row["expert_ci_high"]
        e_err = ci_to_errbar(e_mean, e_lo, e_hi)
        h_exp = ax_top.bar(
            x[i] - bar_width/2, e_mean, bar_width,
            yerr=e_err, capsize=capsize,
            color=color, edgecolor="black", zorder=2,
        )
        # Save one handle only
        if i == 0:
            legend_handles.append(h_exp[0])
            legend_labels.append("Experts")

        # Novice
        n_mean, n_lo, n_hi = row["novice_mean"], row["novice_ci_low"], row["novice_ci_high"]
        n_err = ci_to_errbar(n_mean, n_lo, n_hi)
        h_nov = ax_top.bar(
            x[i] + bar_width/2, n_mean, bar_width,
            yerr=n_err, capsize=capsize,
            color=color, edgecolor="black", hatch="//", zorder=2,
        )
        if i == 0:
            legend_handles.append(h_nov[0])
            legend_labels.append("Novices")

        # Stars for significance
        if is_sig[i] and np.isfinite(e_hi) and np.isfinite(n_hi):
            y_top = max(e_hi, n_hi)
            y_line = y_top + sig_offset * 1.2
            y_ast  = y_line + sig_offset * 0.05
            ax_top.plot([x[i] - bar_width/2, x[i] + bar_width/2], [y_line, y_line],
                        color="black", linewidth=1.2)
            ax_top.text(x[i], y_ast, "*", ha="center", va="bottom",
                        fontsize=asterisk_fontsize, zorder=5)

    # Axes + style
    ax_top.set_ylabel(f"Mean {measure_name} (±95% CI)")  # concise label
    ax_top.axhline(0, color="black", linestyle="--", linewidth=1)
    ax_top.tick_params(axis="x", labelbottom=False)  # bottom panel holds the x labels

    # Y-limits from CIs with buffer
    y_min = np.nanmin(np.r_[df["expert_ci_low"].values, df["novice_ci_low"].values])
    y_max = np.nanmax(np.r_[df["expert_ci_high"].values, df["novice_ci_high"].values])
    if np.any(is_sig):
        y_max += sig_offset
    ax_top.set_ylim(y_min - 0.1, y_max + abs(y_max) * 0.1)

    # Cosmetics
    for spine in ["top", "right"]:
        ax_top.spines[spine].set_visible(False)

    # ---------- BOTTOM: Δ bars + CIΔ ----------
    for i, row in df.iterrows():
        name = row["ROI_Name"]
        color = palette_dict[name]
        d_mean, d_lo, d_hi = row["delta_mean"], row["delta_ci_low"], row["delta_ci_high"]

        # Bar
        ax_bot.bar(x[i], d_mean, width=0.6, color=color, edgecolor="black", zorder=2)

        # CI whiskers
        if np.all(np.isfinite([d_mean, d_lo, d_hi])):
            ax_bot.errorbar(
                i, d_mean,
                yerr=[[d_mean - d_lo], [d_hi - d_mean]],
                fmt='none', ecolor='black', elinewidth=1.5, capsize=capsize, zorder=3
            )

        # Stars
        if is_sig[i] and np.isfinite(d_lo) and np.isfinite(d_hi):
            if d_mean >= 0:
                y_pos, va = d_hi + sig_offset, "bottom"
            else:
                y_pos, va = d_lo - sig_offset, "top"
            ax_bot.text(i, y_pos, "*", ha="center", va=va,
                        fontsize=asterisk_fontsize, zorder=5)

    # Axes + style
    ax_bot.set_ylabel(f"Δ{measure_name} (±95% CI)")  # concise label
    ax_bot.set_ylabel(f"Δ{measure_name} (±95% CI)")  # concise label
    ax_bot.axhline(0, color="black", linestyle="--", linewidth=1)
    ax_bot.set_xticks(x)
    ax_bot.set_xticklabels(roi_names, rotation=30, ha="right")
    ax_bot.yaxis.set_major_locator(plt.MultipleLocator(5))

    # Tint bottom tick labels by significance
    for lbl in ax_bot.get_xticklabels():
        name = lbl.get_text()
        idx = np.where(roi_names == name)[0][0]
        lbl.set_color(palette_dict[name] if is_sig[idx] else "lightgrey")

    # Y-limits
    y_min = np.nanmin(df["delta_ci_low"].values)
    y_max = np.nanmax(df["delta_ci_high"].values)
    ax_bot.set_ylim(y_min - 1, y_max + 1)

    # Cosmetics
    for spine in ["top", "right", "bottom"]:
        ax_bot.spines[spine].set_visible(False)

    # ---------- Figure title + legend ----------
    # if fig_title is None:
    #     fig_title = f"{measure_name}: Group means and differences across ROIs ({'FDR' if use_fdr else 'raw'} p < {ALPHA_FDR})"
    # fig.suptitle(fig_title, fontsize=plt.rcParams["axes.titlesize"] * 1.05, y=0.99)

    # One figure-level legend below panels (no overlap)
    fig.legend(
        handles=legend_handles, labels=legend_labels,
        loc="upper center", bbox_to_anchor=(0.5, 0.005),
        frameon=False, ncol=2
    )
    # Make a bit of room for the legend at the bottom
    plt.subplots_adjust(bottom=0.08, hspace=0.3)

    # ---------- Save ----------
    fname = os.path.join(output_dir, f"roi_{measure_name}_combined_panel.png")
    plt.savefig(fname, dpi=300, bbox_inches="tight")
    plt.show()
    plt.close()
    print(f"Saved: {fname}")


# ==============================
# LaTeX table (display-only; no computations)
# ==============================
def make_pr_multicolumn_table_from_results(
    results_df, output_dir,
    filename_tex="roi_pr_table.tex",
    filename_csv="roi_pr_table.csv",
    include_fdr=True,
    print_to_console=True
):
    """
    Build and save a LaTeX multi-column table (and CSV) directly from results_df.

    Columns:
      ROI | Experts: Mean, 95% CI | Novices: Mean, 95% CI | Δ: Mean, 95% CI | p or p_FDR
    """
    # small format helpers
    def fmt_ci(lo, hi):
        return "[--, --]" if np.any(np.isnan([lo, hi])) else f"[{lo:.2f}, {hi:.2f}]"

    def fmt_val(v):
        return "--" if np.isnan(v) else f"{v:.2f}"

    def fmt_p(p):
        if np.isnan(p): return "--"
        return "$<.001$" if p < 0.001 else f"$={p:.3f}$"

    # tabular rows
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

    # header (avoid f-string `{}` collisions with LaTeX braces)
    p_label = "$p_{\\mathrm{FDR}}$" if include_fdr else "$p$"
    header = (
        "\\begin{table}[p]\n\\centering\n"
        "\\resizebox{\\linewidth}{!}{%\n"
        "\\begin{tabular}{lcc|cc|cc|c}\n\\toprule\n"
        "\\multirow{2}{*}{ROI}\n"
        "  & \\multicolumn{2}{c|}{Experts}\n"
        "  & \\multicolumn{2}{c|}{Novices}\n"
        "  & \\multicolumn{2}{c|}{Experts$-$Novices}\n"
        "  & " + p_label + " \\\\\n"
        "  & Mean & 95\\% CI"
        "  & Mean & 95\\% CI"
        "  & $\\Delta$ & 95\\% CI"
        "  &  \\\\\n\\midrule\n"
    )

    # body
    body = "\n".join(
        f"{r['ROI']} & {r['Expert_mean']} & {r['Expert_CI']} "
        f"& {r['Novice_mean']} & {r['Novice_CI']} "
        f"& {r['Delta_mean']} & {r['Delta_CI']} "
        f"& {r['p']} \\\\"
        for _, r in df_out.iterrows()
    )

    # footer + caption
    p_caption = "FDR-corrected $p$" if include_fdr else "raw $p$"
    footer = (
        "\n\\bottomrule\n\\end{tabular}\n}\n"
        "\\caption{Participation Ratio (PR): group means (95\\% CI), group differences (Welch; 95\\% CI), "
        + f"and {p_caption} per ROI." + "}\n"
        "\\label{tab:pr_multicolumn}\n\\end{table}\n"
    )

    latex_table = header + body + footer

    # print + save
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


# ==============================
# MAIN
# ==============================
if __name__ == "__main__":
    logger.info("PR pipeline started.")

    # Output folder with unique ID + save a copy of this script for provenance
    output_dir = f"results/{create_run_id()}_participation_ratio"
    os.makedirs(output_dir, exist_ok=True)
    save_script_to_file(output_dir)

    # Load atlas (Glasser coarse families, resampled to data space)
    logger.info("Loading atlas...")
    atlas_img = nib.load(ATLAS_FILE)
    atlas_data = atlas_img.get_fdata().astype(int)
    unique_rois = np.unique(atlas_data)
    unique_rois = unique_rois[unique_rois != 0]  # remove background
    logger.info(f"Found {len(unique_rois)} non-zero ROI labels.")

    # Compute PR arrays for Experts / Novices (rows=subjects, cols=ROIs)
    logger.info("Processing EXPERTS...")
    if USE_PARALLEL:
        expert_pr_arr = np.array(Parallel(n_jobs=-1, verbose=5)(
            delayed(process_subject)(sid, atlas_data, unique_rois) for sid in EXPERT_SUBJECTS
        ))
    else:
        expert_pr_arr = np.array([process_subject(sid, atlas_data, unique_rois) for sid in EXPERT_SUBJECTS])

    logger.info("Processing NOVICES...")
    if USE_PARALLEL:
        nonexpert_pr_arr = np.array(Parallel(n_jobs=-1, verbose=5)(
            delayed(process_subject)(sid, atlas_data, unique_rois) for sid in NONEXPERT_SUBJECTS
        ))
    else:
        nonexpert_pr_arr = np.array([process_subject(sid, atlas_data, unique_rois) for sid in NONEXPERT_SUBJECTS])

    # Welch tests + CIs + FDR by ROI
    logger.info("Running group comparisons (Welch + FDR)...")
    pr_stats_df = fdr_ttest(expert_pr_arr, nonexpert_pr_arr, unique_rois, alpha=ALPHA_FDR)

    # Consolidate descriptive CIs + stats into a single results table
    pr_results_df = build_pr_results_df(
        expert_vals=expert_pr_arr,
        nonexpert_vals=nonexpert_pr_arr,
        roi_labels=unique_rois,
        pr_stats_df=pr_stats_df,
        roi_name_map=ROI_NAME_MAP
    )

    # Save consolidated CSV once, and reuse everywhere
    consolidated_csv = os.path.join(output_dir, "roi_pr_results_consolidated.csv")
    pr_results_df.to_csv(consolidated_csv, index=False)
    logger.info(f"Saved consolidated PR results: {consolidated_csv}")

    # Figures (read-only from consolidated df)
    logger.info("Plotting figures...")
    plot_pr_combined_panel_from_results(
        results_df=pr_results_df,
        measure_name="PR",
        output_dir=output_dir,
        use_fdr=use_fdr,
        sort_by="roi",
        custom_colors=CUSTOM_COLORS
    )

    # LaTeX table (read-only from consolidated df)
    make_pr_multicolumn_table_from_results(
        results_df=pr_results_df,
        output_dir=output_dir,
        include_fdr=True,
        filename_tex="roi_pr_table.tex",
        filename_csv="roi_pr_table.csv",
        print_to_console=True
    )

    logger.info(f"All done. Figures and tables saved in: {output_dir}")
