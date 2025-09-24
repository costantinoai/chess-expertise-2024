"""
Participation Ratio (PR) pipeline — refactored, typed, and documented
=====================================================================

This module:
  1) loads SPM first-level betas,
  2) extracts ROI-wise voxel matrices,
  3) computes Participation Ratio (PR) per ROI per subject,
  4) runs Welch tests (Experts vs Novices) with CIs and FDR,
  5) consolidates results (group means + CIs + Δ + CIs + p, p_FDR),
  6) plots figures from the consolidated results only (no recomputation),
  7) builds a LaTeX multi-column table from the consolidated results.

Design principles
-----------------
- Analysis functions compute things **once**.
- Plotting/reporting functions are **read-only** (accept precomputed results).
- Clear boundaries between IO, analysis, and presentation.
- All configuration centralized in a dataclass (`Config`).
- Extensive type hints and docstrings for maintainability.
- Minimal code duplication; small utilities shared across call sites.
- Robustness: explicit errors with helpful messages; logging around IO.

Author: <you>
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import logging
import re

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import nibabel as nib
import scipy.io as sio
from scipy.stats import ttest_ind, ttest_1samp

from sklearn.decomposition import PCA
from statsmodels.stats.multitest import multipletests
from pingouin import compute_effsize

from joblib import Parallel, delayed

# Local helpers
from modules.helpers import create_run_id, save_script_to_file

# ------------------------------
# Plot styles for consistent figures across the paper
# ------------------------------
sns.set_style("white", {"axes.grid": False})
_BASE_FONT_SIZE = 28
plt.rcParams.update({
    "font.family": "Ubuntu Condensed",
    "font.size": _BASE_FONT_SIZE,
    "axes.titlesize": _BASE_FONT_SIZE * 1.4,
    "axes.labelsize": _BASE_FONT_SIZE * 1.2,
    "xtick.labelsize": _BASE_FONT_SIZE,
    "ytick.labelsize": _BASE_FONT_SIZE,
    "legend.fontsize": _BASE_FONT_SIZE,
    "figure.figsize": (21, 11),
})

# ------------------------------
# Configuration & constants
# ------------------------------
@dataclass(frozen=True)
class Config:
    """Central configuration for paths, subjects, atlas, and stats."""

    # Data locations
    base_path: Path = Path("/data/projects/chess/data/BIDS/derivatives/fmriprep-SPM_smoothed-NO_GS-FD-HMP_brainmasked/MNI/fmriprep-SPM-MNI/GLM")
    spm_filename: str = "SPM.mat"
    atlas_file: Path = Path("/data/projects/chess/data/misc/templates/tpl-MNI152NLin2009cAsym_res-02_atlas-Glasser2016_desc-cortices_bilateral_resampled.nii")

    # Subject lists (string IDs used in paths)
    expert_subjects: Tuple[str, ...] = (
        "03", "04", "06", "07", "08", "09",
        "10", "11", "12", "13", "16", "20",
        "22", "23", "24", "29", "30", "33",
        "34", "36",
    )
    nonexpert_subjects: Tuple[str, ...] = (
        "01", "02", "15", "17", "18", "19",
        "21", "25", "26", "27", "28", "32",
        "35", "37", "39", "40", "41", "42",
        "43", "44",
    )

    # Stats settings
    alpha_fdr: float = 0.05
    use_parallel: bool = True
    n_jobs: int = -1

    # Plotting
    custom_colors: Tuple[str, ...] = (
        # 22 entries to color ROIs by coarse families
        "#a6cee3", "#a6cee3",                    # Early Visual (2)
        "#1f78b4", "#1f78b4", "#1f78b4",         # Intermediate Visual (3)
        "#b2df8a", "#b2df8a", "#b2df8a", "#b2df8a",  # Sensorimotor (4)
        "#33a02c", "#33a02c", "#33a02c",         # Auditory (3)
        "#fb9a99", "#fb9a99",                    # Temporal (2)
        "#e31a1c", "#e31a1c", "#e31a1c", "#e31a1c",  # Posterior (4)
        "#fdbf6f", "#fdbf6f", "#fdbf6f", "#fdbf6f"   # Anterior (4)
    )

    # ROI names (Glasser families merged bilaterally)
    roi_name_map: Mapping[int, str] = field(default_factory=lambda: {
        1: "Primary Visual", 2: "Early Visual", 3: "Dorsal Stream Visual",
        4: "Ventral Stream Visual", 5: "MT+ Complex", 6: "Somatosensory and Motor",
        7: "Paracentral Lobular and Mid Cing", 8: "Premotor", 9: "Posterior Opercular",
        10: "Early Auditory", 11: "Auditory Association", 12: "Insular and Frontal Opercular",
        13: "Medial Temporal", 14: "Lateral Temporal", 15: "Temporo-Parieto Occipital Junction",
        16: "Superior Parietal", 17: "Inferior Parietal", 18: "Posterior Cing",
        19: "Anterior Cing and Medial Prefrontal", 20: "Orbital and Polar Frontal",
        21: "Inferior Frontal", 22: "Dorsolateral Prefrontal",
    })


# ------------------------------
# Logging config
# ------------------------------
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")


# ------------------------------
# Utilities
# ------------------------------
def _assert_file_exists(path: Path, label: str) -> None:
    if not path.is_file():
        msg = f"{label} not found: {path}"
        logger.error(msg)
        raise FileNotFoundError(msg)


def _spm_dir_from_swd(spm_obj, fallback: Path) -> Path:
    """Extract SPM working directory robustly across SPM versions."""
    swd = getattr(spm_obj, "swd", None)
    return Path(swd) if swd else fallback


# ==============================
# Helpers: load SPM betas & build ROI matrices
# ==============================
def get_spm_condition_betas(subject_id: str, cfg: Config) -> Dict[str, nib.Nifti1Image]:
    """Load SPM.mat for one subject and return averaged beta image per condition.

    Parameters
    ----------
    subject_id : str
        Subject code (e.g., '03').
    cfg : Config
        Global configuration.

    Returns
    -------
    dict[str, nib.Nifti1Image]
        Mapping from condition label (e.g., 'C1') to averaged beta NIfTI.
    """
    spm_mat_path = cfg.base_path / f"sub-{subject_id}" / "exp" / cfg.spm_filename
    _assert_file_exists(spm_mat_path, "SPM.mat")

    spm_dict = sio.loadmat(spm_mat_path.as_posix(), struct_as_record=False, squeeze_me=True)
    SPM = spm_dict["SPM"]
    beta_info = SPM.Vbeta
    regressor_names: Sequence[str] = SPM.xX.name

    pattern = re.compile(r"Sn\(\d+\)\s+(.*?)\*bf\(1\)")

    # Map condition -> beta indices (across runs)
    condition_to_indices: Dict[str, List[int]] = {}
    for i, reg_name in enumerate(regressor_names):
        m = pattern.match(reg_name)
        if m:
            cond = m.group(1)
            condition_to_indices.setdefault(cond, []).append(i)

    spm_dir = _spm_dir_from_swd(SPM, spm_mat_path.parent)
    averaged: Dict[str, nib.Nifti1Image] = {}

    for cond, idxs in condition_to_indices.items():
        if not idxs:
            continue
        sum_data: Optional[np.ndarray] = None
        affine = header = None
        for idx in idxs:
            beta_fname = getattr(beta_info[idx], "fname", None) or getattr(beta_info[idx], "filename")
            beta_path = spm_dir / beta_fname
            img = nib.load(beta_path.as_posix())
            data = img.get_fdata(dtype=np.float32)
            if sum_data is None:
                sum_data = np.zeros_like(data, dtype=np.float32)
                affine, header = img.affine, img.header
            sum_data += data
        assert sum_data is not None
        avg = sum_data / float(len(idxs))
        averaged[cond] = nib.Nifti1Image(avg, affine=affine, header=header)

    return averaged


def extract_roi_voxel_matrices(
    subject_id: str,
    atlas_data: np.ndarray,
    roi_labels: np.ndarray,
    cfg: Config,
) -> Dict[int, np.ndarray]:
    """For one subject, build a (conditions × voxels) matrix per ROI.

    Steps:
      1) average betas per condition (get_spm_condition_betas),
      2) for each ROI, gather beta values at ROI voxels for every condition.

    Returns
    -------
    dict[int, np.ndarray]
        roi_label -> array shape (n_conditions, n_voxels_in_roi)
    """
    logger.info(f"[Subject {subject_id}] Extracting ROI voxel matrices…")
    averaged_betas = get_spm_condition_betas(subject_id, cfg)
    conditions = sorted(averaged_betas.keys())

    roi_data: Dict[int, np.ndarray] = {}
    for roi_label in roi_labels:
        mask = atlas_data == roi_label
        n_vox = int(mask.sum())
        if n_vox == 0:
            raise ValueError(f"ROI {roi_label} has 0 voxels in atlas.")
        mat = np.zeros((len(conditions), n_vox), dtype=np.float32)
        for ci, cname in enumerate(conditions):
            beta_vals = averaged_betas[cname].get_fdata()
            mat[ci, :] = beta_vals[mask]
        roi_data[int(roi_label)] = mat

    logger.info(f"[Subject {subject_id}] ROI extraction done.")
    return roi_data


# ==============================
# PR computation + statistics (Welch + FDR)
# ==============================
def participation_ratio(roi_matrix: np.ndarray) -> float:
    """Compute PR from PCA spectrum of a ROI matrix (conditions × voxels).

    PR = (sum(λ))^2 / sum(λ^2)

    Returns np.nan if matrix is degenerate after NaN-only column removal.
    """
    if roi_matrix.size == 0:
        return float("nan")
    valid = ~np.isnan(roi_matrix).all(axis=0)
    X = roi_matrix[:, valid]
    if X.size == 0:
        return float("nan")
    var = PCA().fit(X).explained_variance_
    if var.size == 0:
        return float("nan")
    return float((var.sum() ** 2) / np.sum(var ** 2))


def mean_and_ci(x: np.ndarray, conf: float = 0.95) -> Tuple[float, float, float]:
    """Return (mean, ci_low, ci_high). Uses SciPy's one-sample t CI against 0."""
    x = np.asarray(x)
    x = x[~np.isnan(x)]
    if x.size == 0:
        return (np.nan, np.nan, np.nan)
    mean_val = float(np.mean(x))
    ci = ttest_1samp(x, popmean=0).confidence_interval(confidence_level=conf)
    return (mean_val, float(ci.low), float(ci.high))


def welch_diff_with_ci(
    x: np.ndarray, y: np.ndarray, conf: float = 0.95
) -> Tuple[float, float, float, float, float, float]:
    """Return (mean_x, mean_y, mean_diff, ci_low, ci_high, p_val, df)."""
    x = x[~np.isnan(x)]
    y = y[~np.isnan(y)]
    if x.size < 2 or y.size < 2:
        return (np.nan, np.nan, np.nan, np.nan, np.nan, np.nan)
    res = ttest_ind(x, y, equal_var=False, nan_policy="omit")
    ci = res.confidence_interval(confidence_level=conf)
    mean_diff = float(np.mean(x) - np.mean(y))
    return (float(np.mean(x)), float(np.mean(y)), mean_diff, float(ci.low), float(ci.high), float(res.pvalue))


def ci_to_errbar(mean_val: float, ci_low: float, ci_high: float) -> List[List[float]]:
    """Convert mean+CI to Matplotlib's symmetric errorbar [[lower],[upper]]."""
    if np.any(np.isnan([mean_val, ci_low, ci_high])):
        return [[np.nan], [np.nan]]
    return [[mean_val - ci_low], [ci_high - mean_val]]


def per_roi_welch_and_fdr(
    expert_vals: np.ndarray,
    novice_vals: np.ndarray,
    roi_labels: Sequence[int],
    alpha: float,
) -> pd.DataFrame:
    """Welch tests per ROI + effect sizes + CIs + BH-FDR.

    Returns a DataFrame with columns:
      ROI_Label, t_stat, p_val, p_val_fdr, significant_fdr, significant,
      dof, cohen_d, mean_diff, ci95_low, ci95_high
    """
    records: List[Dict[str, float]] = []
    for i, roi in enumerate(roi_labels):
        g1 = expert_vals[:, i]
        g2 = novice_vals[:, i]
        g1 = g1[~np.isnan(g1)]
        g2 = g2[~np.isnan(g2)]

        if g1.size < 2 or g2.size < 2:
            records.append({
                "ROI_Label": int(roi),
                "t_stat": np.nan, "p_val": np.nan, "dof": np.nan,
                "cohen_d": np.nan, "mean_diff": np.nan,
                "ci95_low": np.nan, "ci95_high": np.nan,
            })
            continue

        res = ttest_ind(g1, g2, equal_var=False, nan_policy="omit")
        ci = res.confidence_interval(confidence_level=0.95)
        d = compute_effsize(g1, g2, eftype="cohen", paired=False)

        records.append({
            "ROI_Label": int(roi),
            "t_stat": float(res.statistic),
            "p_val": float(res.pvalue),
            "dof": float(res.df),
            "cohen_d": float(d),
            "mean_diff": float(np.mean(g1) - np.mean(g2)),
            "ci95_low": float(ci.low),
            "ci95_high": float(ci.high),
        })

    df = pd.DataFrame.from_records(records)

    # Benjamini–Hochberg FDR on p-values
    pvals = df["p_val"].fillna(1.0).to_numpy()
    reject, pval_fdr, _, _ = multipletests(pvals, alpha=alpha, method="fdr_bh")
    df["p_val_fdr"] = pval_fdr
    df["significant_fdr"] = reject
    df["significant"] = df["p_val"] < 0.05
    return df


# ==============================
# Subject-level pipeline
# ==============================
def process_subject(subject_id: str, atlas_data: np.ndarray, roi_labels: np.ndarray, cfg: Config) -> np.ndarray:
    """Compute PR per ROI for one subject. Returns shape (n_rois,)."""
    logger.info(f"[Subject {subject_id}] Start")
    roi_mats = extract_roi_voxel_matrices(subject_id, atlas_data, roi_labels, cfg)
    pr = np.full(len(roi_labels), np.nan, dtype=np.float32)
    for idx, roi in enumerate(roi_labels):
        mat = roi_mats[int(roi)]
        pr[idx] = participation_ratio(mat) if mat is not None and mat.size else np.nan
    logger.info(f"[Subject {subject_id}] Done")
    return pr


# ==============================
# Consolidation (no new computations)
# ==============================
def consolidate_results(
    expert_vals: np.ndarray,
    novice_vals: np.ndarray,
    roi_labels: np.ndarray,
    stats_df: pd.DataFrame,
    roi_name_map: Mapping[int, str],
) -> pd.DataFrame:
    """Create a single dataframe with EVERYTHING needed downstream.

    Columns:
      ROI_Label, ROI_Name,
      expert_mean, expert_ci_low, expert_ci_high,
      novice_mean, novice_ci_low, novice_ci_high,
      delta_mean, delta_ci_low, delta_ci_high,
      p_raw, p_fdr, significant, significant_fdr
    """
    rows: List[Dict[str, float | str]] = []
    for i, roi in enumerate(roi_labels):
        X = expert_vals[:, i]
        Y = novice_vals[:, i]
        e_mean, e_lo, e_hi = mean_and_ci(X)
        n_mean, n_lo, n_hi = mean_and_ci(Y)
        rows.append({
            "ROI_Label": int(roi),
            "ROI_Name": roi_name_map.get(int(roi), f"ROI {int(roi)}"),
            "expert_mean": e_mean, "expert_ci_low": e_lo, "expert_ci_high": e_hi,
            "novice_mean": n_mean, "novice_ci_low": n_lo, "novice_ci_high": n_hi,
        })
    desc_df = pd.DataFrame(rows)

    stats = stats_df.rename(columns={
        "mean_diff": "delta_mean",
        "ci95_low": "delta_ci_low",
        "ci95_high": "delta_ci_high",
        "p_val": "p_raw",
        "p_val_fdr": "p_fdr",
    })[[
        "ROI_Label", "delta_mean", "delta_ci_low", "delta_ci_high",
        "p_raw", "p_fdr", "significant", "significant_fdr",
    ]].copy()

    out = desc_df.merge(stats, on="ROI_Label", how="left")
    return out.sort_values("ROI_Label").reset_index(drop=True)


# ==============================
# Plotting (display-only; no computations)
# ==============================
def plot_pr_combined_panel(
    results_df: pd.DataFrame,
    measure_name: str,
    output_dir: Path,
    alpha_fdr: float,
    use_fdr: bool = True,
    sort_by: str = "roi",  # {"roi", "diff"}
    custom_colors: Optional[Sequence[str]] = None,
    fig_title: Optional[str] = None,
) -> Path:
    """Two-panel figure: group means (95% CI) and Δ with CIΔ.

    Read-only: consumes `results_df` only.
    Returns the path to the saved PNG.
    """
    from natsort import natsorted

    if sort_by == "roi":
        df = results_df.sort_values("ROI_Label", key=natsorted).copy()
    elif sort_by == "diff":
        df = results_df.sort_values("delta_mean", ascending=False).copy()
    else:
        df = results_df.copy()

    roi_names = df["ROI_Name"].values
    x = np.arange(len(df))

    pvals = df["p_fdr"].values if use_fdr else df["p_raw"].values
    is_sig = np.array([False if np.isnan(p) else (p < alpha_fdr) for p in pvals])

    # Palette per ROI name
    if custom_colors is not None:
        palette_dict = {name: custom_colors[i % len(custom_colors)] for i, name in enumerate(roi_names)}
    else:
        palette = sns.color_palette("husl", len(roi_names))
        palette_dict = {name: palette[i] for i, name in enumerate(roi_names)}

    fig = plt.figure(constrained_layout=True, figsize=(18, 14))
    gs = fig.add_gridspec(nrows=2, ncols=1, height_ratios=[2, 3])
    ax_top = fig.add_subplot(gs[0, 0])
    ax_bot = fig.add_subplot(gs[1, 0], sharex=ax_top)

    bar_width = 0.35
    capsize = 4
    sig_offset = 0.5
    asterisk_fs = plt.rcParams["font.size"] * 1.2

    legend_handles: List[plt.Artist] = []
    legend_labels: List[str] = []

    # --- TOP: group means ---
    for i, name in enumerate(roi_names):
        row = df.iloc[i]
        color = palette_dict[name]

        e_mean, e_lo, e_hi = row["expert_mean"], row["expert_ci_low"], row["expert_ci_high"]
        e_err = ci_to_errbar(e_mean, e_lo, e_hi)
        h_exp = ax_top.bar(x[i] - bar_width/2, e_mean, bar_width, yerr=e_err, capsize=capsize,
                            color=color, edgecolor="black", zorder=2)
        if i == 0:
            legend_handles.append(h_exp[0])
            legend_labels.append("Experts")

        n_mean, n_lo, n_hi = row["novice_mean"], row["novice_ci_low"], row["novice_ci_high"]
        n_err = ci_to_errbar(n_mean, n_lo, n_hi)
        h_nov = ax_top.bar(x[i] + bar_width/2, n_mean, bar_width, yerr=n_err, capsize=capsize,
                            color=color, edgecolor="black", hatch="//", zorder=2)
        if i == 0:
            legend_handles.append(h_nov[0])
            legend_labels.append("Novices")

        if is_sig[i] and np.isfinite(e_hi) and np.isfinite(n_hi):
            y_top = max(e_hi, n_hi)
            y_line = y_top + sig_offset * 1.2
            y_ast = y_line + sig_offset * 0.05
            ax_top.plot([x[i] - bar_width/2, x[i] + bar_width/2], [y_line, y_line], color="black", linewidth=1.2)
            ax_top.text(x[i], y_ast, "*", ha="center", va="bottom", fontsize=asterisk_fs, zorder=5)

    ax_top.set_ylabel(f"Mean {measure_name} (±95% CI)")
    ax_top.axhline(0, color="black", linestyle="--", linewidth=1)
    ax_top.tick_params(axis="x", labelbottom=False)

    y_min = np.nanmin(np.r_[df["expert_ci_low"].values, df["novice_ci_low"].values])
    y_max = np.nanmax(np.r_[df["expert_ci_high"].values, df["novice_ci_high"].values])
    if np.any(is_sig):
        y_max += sig_offset
    ax_top.set_ylim(y_min - 0.1, y_max + abs(y_max) * 0.1)

    for spine in ["top", "right"]:
        ax_top.spines[spine].set_visible(False)

    # --- BOTTOM: deltas ---
    for i, row in df.iterrows():
        name = row["ROI_Name"]
        color = palette_dict[name]
        d_mean, d_lo, d_hi = row["delta_mean"], row["delta_ci_low"], row["delta_ci_high"]

        ax_bot.bar(x[i], d_mean, width=0.6, color=color, edgecolor="black", zorder=2)
        if np.all(np.isfinite([d_mean, d_lo, d_hi])):
            ax_bot.errorbar(i, d_mean, yerr=[[d_mean - d_lo], [d_hi - d_mean]], fmt="none",
                            ecolor="black", elinewidth=1.5, capsize=capsize, zorder=3)

        if is_sig[i] and np.isfinite(d_lo) and np.isfinite(d_hi):
            if d_mean >= 0:
                y_pos, va = d_hi + sig_offset, "bottom"
            else:
                y_pos, va = d_lo - sig_offset, "top"
            ax_bot.text(i, y_pos, "*", ha="center", va=va, fontsize=asterisk_fs, zorder=5)

    ax_bot.set_ylabel(f"Δ{measure_name} (±95% CI)")
    ax_bot.axhline(0, color="black", linestyle="--", linewidth=1)
    ax_bot.set_xticks(x)
    ax_bot.set_xticklabels(roi_names, rotation=30, ha="right")
    ax_bot.yaxis.set_major_locator(plt.MultipleLocator(5))

    for lbl in ax_bot.get_xticklabels():
        name = lbl.get_text()
        idx = np.where(roi_names == name)[0][0]
        lbl.set_color(palette_dict[name] if is_sig[idx] else "lightgrey")

    y_min = np.nanmin(df["delta_ci_low"].values)
    y_max = np.nanmax(df["delta_ci_high"].values)
    ax_bot.set_ylim(y_min - 1, y_max + 1)

    for spine in ["top", "right", "bottom"]:
        ax_bot.spines[spine].set_visible(False)

    if fig_title:
        fig.suptitle(fig_title, fontsize=plt.rcParams["axes.titlesize"] * 1.05, y=0.99)

    ax_top.legend(handles=legend_handles, labels=legend_labels, loc="upper center",
                  bbox_to_anchor=(0.5, 1.25), frameon=False, ncol=2)

    roi_family_labels = [
        "Early Visual", "Intermediate Visual", "Sensorimotor", "Auditory",
        "Temporal", "Posterior", "Anterior",
    ]
    roi_family_colors = [
        "#a6cee3", "#1f78b4", "#b2df8a", "#33a02c",
        "#fb9a99", "#e31a1c", "#fdbf6f",
    ]
    family_handles = [plt.Rectangle((0, 0), 0.5, 1, color=c, edgecolor="black") for c in roi_family_colors]
    leg = fig.legend(handles=family_handles, labels=roi_family_labels, loc="lower center",
               bbox_to_anchor=(0.5, -0.05), frameon=True, ncol=7, prop={"size": 20})
    leg.get_frame().set_edgecolor("black")

    plt.subplots_adjust(top=0.85, bottom=0.12, hspace=0.3)

    output_dir.mkdir(parents=True, exist_ok=True)
    fname = output_dir / f"roi_{measure_name}_combined_panel.png"
    plt.savefig(fname.as_posix(), dpi=300, bbox_inches="tight")
    plt.show()
    plt.close()
    logger.info(f"Saved figure: {fname}")
    return fname


# ==============================
# LaTeX table (display-only; no computations)
# ==============================
def build_pr_table(
    results_df: pd.DataFrame,
    output_dir: Path,
    filename_tex: str = "roi_pr_table.tex",
    filename_csv: str = "roi_pr_table.csv",
    include_fdr: bool = True,
    print_to_console: bool = True,
) -> Tuple[pd.DataFrame, str, Path, Path]:
    """Build and save a LaTeX multi-column table (and CSV) from results_df.

    Columns: ROI | Experts: Mean, 95% CI | Novices: Mean, 95% CI | Δ: Mean, 95% CI | p or p_FDR

    Returns the DataFrame used for CSV, the LaTeX string, and the paths written.
    """
    def fmt_ci(lo: float, hi: float) -> str:
        return "[--, --]" if np.any(np.isnan([lo, hi])) else f"[{lo:.2f}, {hi:.2f}]"

    def fmt_val(v: float) -> str:
        return "--" if np.isnan(v) else f"{v:.2f}"

    def fmt_p(p: float) -> str:
        if np.isnan(p):
            return "--"
        return "$<.001$" if p < 0.001 else f"$={p:.3f}$"

    rows = []
    for _, r in results_df.iterrows():
        rows.append({
            "ROI": r["ROI_Name"],
            "Expert_mean": fmt_val(r["expert_mean"]),
            "Expert_CI": fmt_ci(r["expert_ci_low"], r["expert_ci_high"]),
            "Novice_mean": fmt_val(r["novice_mean"]),
            "Novice_CI": fmt_ci(r["novice_ci_low"], r["novice_ci_high"]),
            "Delta_mean": fmt_val(r["delta_mean"]),
            "Delta_CI": fmt_ci(r["delta_ci_low"], r["delta_ci_high"]),
            "p": fmt_p(r["p_fdr"] if include_fdr else r["p_raw"]),
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
        f"  & {p_label} \\\\\
"
        "  & Mean & 95\\% CI"
        "  & Mean & 95\\% CI"
        "  & $\\Delta$ & 95\\% CI"
        "  &  \\\\n\\midrule\n"
    )

    body = "\n".join(
        f"{r['ROI']} & {r['Expert_mean']} & {r['Expert_CI']} "
        f"& {r['Novice_mean']} & {r['Novice_CI']} "
        f"& {r['Delta_mean']} & {r['Delta_CI']} "
        f"& {r['p']} \\\\" for _, r in df_out.iterrows()
    )

    p_caption = "FDR-corrected $p$" if include_fdr else "raw $p$"
    footer = (
        "\n\\bottomrule\n\\end{tabular}\n}\n"
        "\\caption{Participation Ratio (PR): group means (95\\% CI), group differences (Welch; 95\\% CI), "
        f"and {p_caption} per ROI." "}\n"
        "\\label{tab:pr_multicolumn}\n\\end{table}\n"
    )

    latex_table = header + body + footer

    if print_to_console:
        print(latex_table)

    output_dir.mkdir(parents=True, exist_ok=True)
    tex_out = output_dir / filename_tex
    csv_out = output_dir / filename_csv
    tex_out.write_text(latex_table)
    df_out.to_csv(csv_out, index=False)

    logger.info(f"Saved LaTeX table: {tex_out}")
    logger.info(f"Saved CSV: {csv_out}")
    return df_out, latex_table, tex_out, csv_out


# ==============================
# MAIN
# ==============================
def main(cfg: Optional[Config] = None) -> None:
    cfg = cfg or Config()
    logger.info("PR pipeline started.")

    # Output folder with unique ID + save a copy of this script for provenance
    output_dir = Path("results") / f"{create_run_id()}_participation_ratio"
    output_dir.mkdir(parents=True, exist_ok=True)
    save_script_to_file(output_dir)

    # Load atlas
    logger.info("Loading atlas…")
    _assert_file_exists(cfg.atlas_file, "Atlas NIfTI")
    atlas_img = nib.load(cfg.atlas_file.as_posix())
    atlas_data = atlas_img.get_fdata().astype(int)
    roi_labels = np.unique(atlas_data)
    roi_labels = roi_labels[roi_labels != 0]
    logger.info(f"Found {len(roi_labels)} non-zero ROI labels.")

    # Compute PR arrays for Experts / Novices (rows=subjects, cols=ROIs)
    def _run_group(sub_ids: Sequence[str]) -> np.ndarray:
        if cfg.use_parallel:
            return np.array(Parallel(n_jobs=cfg.n_jobs, verbose=5)(
                delayed(process_subject)(sid, atlas_data, roi_labels, cfg) for sid in sub_ids
            ))
        return np.array([process_subject(sid, atlas_data, roi_labels, cfg) for sid in sub_ids])

    logger.info("Processing EXPERTS…")
    expert_pr = _run_group(cfg.expert_subjects)

    logger.info("Processing NOVICES…")
    novice_pr = _run_group(cfg.nonexpert_subjects)

    # Welch tests + CIs + FDR by ROI
    logger.info("Running group comparisons (Welch + FDR)…")
    stats_df = per_roi_welch_and_fdr(expert_pr, novice_pr, roi_labels, alpha=cfg.alpha_fdr)

    # Consolidate
    results_df = consolidate_results(expert_pr, novice_pr, roi_labels, stats_df, cfg.roi_name_map)

    # Save consolidated CSV once
    consolidated_csv = output_dir / "roi_pr_results_consolidated.csv"
    results_df.to_csv(consolidated_csv, index=False)
    logger.info(f"Saved consolidated PR results: {consolidated_csv}")

    # Figures (read-only from consolidated df)
    logger.info("Plotting figures…")
    _ = plot_pr_combined_panel(
        results_df=results_df,
        measure_name="PR",
        output_dir=output_dir,
        alpha_fdr=cfg.alpha_fdr,
        use_fdr=True,
        sort_by="roi",
        custom_colors=cfg.custom_colors,
    )

    # LaTeX table (read-only from consolidated df)
    _ = build_pr_table(
        results_df=results_df,
        output_dir=output_dir,
        include_fdr=True,
        filename_tex="roi_pr_table.tex",
        filename_csv="roi_pr_table.csv",
        print_to_console=True,
    )

    logger.info(f"All done. Figures and tables saved in: {output_dir}")


if __name__ == "__main__":  # pragma: no cover
    main()
