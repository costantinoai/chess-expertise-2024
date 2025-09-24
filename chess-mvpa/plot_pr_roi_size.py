import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import nibabel as nib
import scipy.io as sio
from scipy.stats import linregress
from joblib import Parallel, delayed
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.multitest import fdrcorrection
from matplotlib.patches import Patch
from matplotlib.colors import LinearSegmentedColormap
import matplotlib as mpl

from modules.helpers import create_run_id, save_script_to_file


# Make custom colormap
def make_brain_cmap():
    center = plt.cm.RdPu(0)[:3]
    neg = np.linspace([0.0, 0.5, 0.7], center, 256)
    pos = plt.cm.RdPu(np.linspace(0, 1, 256))[:, :3]
    return LinearSegmentedColormap.from_list("custom_brain", np.vstack((neg, pos)))


BRAIN_CMAP = make_brain_cmap()

# Plot styles
sns.set_style("white", {"axes.grid": False})
base_font_size = 22
plt.rcParams.update(
    {
        "font.family": "Ubuntu Condensed",
        "font.size": base_font_size,
        "axes.titlesize": base_font_size * 1.4,  # 36.4 ~ 36
        "axes.labelsize": base_font_size * 1.2,  # 31.2 ~ 31
        "xtick.labelsize": base_font_size,  # 26
        "ytick.labelsize": base_font_size,  # 26
        "legend.fontsize": base_font_size,  # 26
        "figure.figsize": (12, 9),  # wide figures
    }
)
plt.rcParams['axes.labelsize']
# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
CUSTOM_COLORS = [
    "#c6dbef",
    "#c6dbef",
    "#2171b5",
    "#2171b5",
    "#2171b5",
    "#a1d99b",
    "#a1d99b",
    "#a1d99b",
    "#a1d99b",
    "#00441b",
    "#00441b",
    "#00441b",
    "#fbb4b9",
    "#fbb4b9",
    "#cb181d",
    "#cb181d",
    "#cb181d",
    "#cb181d",
    "#fec44f",
    "#fec44f",
    "#fec44f",
    "#fec44f",
]

ROI_NAME_MAP = {
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
    22: "Dorsolateral Prefrontal",
}

EXPERT_SUBJECTS = [
    "03",
    "04",
    "06",
    "07",
    "08",
    "09",
    "10",
    "11",
    "12",
    "13",
    "16",
    "20",
    "22",
    "23",
    "24",
    "29",
    "30",
    "33",
    "34",
    "36",
]
NONEXPERT_SUBJECTS = [
    "01",
    "02",
    "15",
    "17",
    "18",
    "19",
    "21",
    "25",
    "26",
    "27",
    "28",
    "32",
    "35",
    "37",
    "39",
    "40",
    "41",
    "42",
    "43",
    "44",
]

ELOS = [
    2041,
    1850,
    1751,
    2241,
    1941,
    1969,
    2100,
    2051,
    2150,
    1824,
    1881,
    2269,
    2010,
    2100,
    2094,
    1879,
    2232,
    2073,
    2083,
    2189,
]

BASE_PATH = "/data/projects/chess/data/BIDS/derivatives/fmriprep-SPM_smoothed-NO_GS-FD-HMP_brainmasked/MNI/fmriprep-SPM-MNI/GLM"
SPM_FILENAME = "SPM.mat"
ATLAS_FILE = "/data/projects/chess/data/misc/templates/tpl-MNI152NLin2009cAsym_res-02_atlas-Glasser2016_desc-cortices_bilateral_resampled.nii"

# -----------------------------------------------------------------------------
# Functions
# -----------------------------------------------------------------------------


def get_spm_info(subject_id):
    path = os.path.join(BASE_PATH, f"sub-{subject_id}", "exp", SPM_FILENAME)
    spm = sio.loadmat(path, struct_as_record=False, squeeze_me=True)["SPM"]
    betas = spm.Vbeta
    names = spm.xX.name
    pattern = r"Sn\(\d+\)\s+(.*?)\*bf\(1\)"
    cond_dict = {}
    for i, name in enumerate(names):
        match = re.match(pattern, name)
        if match:
            cond = match.group(1)
            cond_dict.setdefault(cond, []).append(i)
    spm_dir = getattr(spm, "swd", os.path.dirname(path))
    result = {}
    for cond, indices in cond_dict.items():
        data_sum = None
        for idx in indices:
            img = nib.load(os.path.join(spm_dir, betas[idx].fname))
            data = img.get_fdata(dtype=np.float32)
            if data_sum is None:
                data_sum = np.zeros_like(data)
                affine, header = img.affine, img.header
            data_sum += data
        result[cond] = nib.Nifti1Image(data_sum / len(indices), affine, header)
    return result


def load_roi_voxel_data(subject_id, atlas_data, unique_rois):
    betas = get_spm_info(subject_id)
    conditions = sorted(betas.keys())
    output = {
        roi: np.zeros((len(conditions), np.sum(atlas_data == roi)), dtype=np.float32)
        for roi in unique_rois
    }
    for i, cond in enumerate(conditions):
        data = betas[cond].get_fdata()
        for roi in unique_rois:
            output[roi][i, :] = data[atlas_data == roi]
    return output


def compute_entropy_pr(roi_data):
    valid = ~np.isnan(roi_data).all(axis=0)
    cleaned = roi_data[:, valid]
    if cleaned.shape[1] < 2:
        return np.nan, 0
    pca = PCA().fit(cleaned)
    var = pca.explained_variance_
    return (var.sum() ** 2) / np.sum(var**2), cleaned.shape[1]


def process_subject(subject_id, atlas_data, unique_rois):
    roi_data = load_roi_voxel_data(subject_id, atlas_data, unique_rois)
    pr, vox = [], []
    for roi in unique_rois:
        p, v = compute_entropy_pr(roi_data[roi])
        pr.append(p)
        vox.append(v)
    return np.array(pr), np.array(vox)


def plot_voxelcount_vs_pr(df, y_col, title_prefix, fname, out_dir):
    slope, intercept, r, p, _ = linregress(df["VoxelCount"], df[y_col])
    title = f"{title_prefix} (r = {r:.2f}, p = {p:.3f})"
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.scatter(
        df["VoxelCount"],
        df[y_col],
        c=df["Color"],
        edgecolor="black",
        s=250,
        linewidth=0,
        alpha=0.7,
    )
    ax.plot(
        df["VoxelCount"],
        intercept + slope * df["VoxelCount"],
        color="black",
        linestyle="--",
        linewidth=1,
    )
    # for _, row in df.iterrows():
    #     ax.text(row["VoxelCount"], row[y_col], row["ROI_Name"], ha="left", alpha=0.75)
    ax.set_xlabel("Average (Experts and Novices) Number of Voxels in ROI")
    ax.set_ylabel("Participation Ratio (Experts - Non-Experts)")
    ax.set_title(title, pad=20)
    # ax.tick_params(labelsize=12)
    # === SPINES & GRID ===
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.5)
    ax.spines["bottom"].set_linewidth(0.5)

    plt.tight_layout()
    path = os.path.join(out_dir, fname)
    plt.savefig(path, dpi=300)
    plt.show()
    logging.info("Saved: %s", path)


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

output_dir = f"results/{create_run_id()}_pr_voxelcount"
os.makedirs(output_dir, exist_ok=True)
save_script_to_file(output_dir)

atlas_data = nib.load(ATLAS_FILE).get_fdata().astype(int)
unique_rois = np.unique(atlas_data)
unique_rois = unique_rois[unique_rois != 0]

expert_results = Parallel(n_jobs=-1)(
    delayed(process_subject)(sub, atlas_data, unique_rois) for sub in EXPERT_SUBJECTS
)
nonexpert_results = Parallel(n_jobs=-1)(
    delayed(process_subject)(sub, atlas_data, unique_rois) for sub in NONEXPERT_SUBJECTS
)

expert_pr = np.array([r[0] for r in expert_results])
expert_vox = np.array([r[1] for r in expert_results])
nonexpert_pr = np.array([r[0] for r in nonexpert_results])
nonexpert_vox = np.array([r[1] for r in nonexpert_results])

df = pd.DataFrame(
    {
        "ROI_Label": unique_rois,
        "ROI_Name": [ROI_NAME_MAP.get(roi, f"ROI {roi}") for roi in unique_rois],
        "Color": CUSTOM_COLORS[: len(unique_rois)],
        "VoxelCount": np.mean(np.vstack([expert_vox, nonexpert_vox]), axis=0),
        "PR_Expert": np.nanmean(expert_pr, axis=0),
        "PR_NonExpert": np.nanmean(nonexpert_pr, axis=0),
    }
)
df["PR_Diff"] = df["PR_Expert"] - df["PR_NonExpert"]

plot_voxelcount_vs_pr(
    df, "PR_Expert", "Experts: PR vs ROI Size", "experts_voxel_vs_pr.png", output_dir
)
plot_voxelcount_vs_pr(
    df,
    "PR_NonExpert",
    "Non-Experts: PR vs ROI Size",
    "nonexperts_voxel_vs_pr.png",
    output_dir,
)
plot_voxelcount_vs_pr(
    df,
    "PR_Diff",
    "PR Difference vs. Average ROI Size",
    "diff_voxel_vs_pr.png",
    output_dir,
)

def plot_feature_importance_from_pca(clf, roi_names, top_n=None):
    """
    Plot the contribution of each ROI to the classifier decision in PCA space.
    Weights are projected back to original ROI space.

    Parameters
    ----------
    clf : Trained LogisticRegression classifier in PCA space
    roi_names : dict mapping ROI indices to ROI labels
    top_n : number of top-weighted ROIs to plot (if None, plot all)
    """

    weights_orig_space = clf.coef_[0]  # Already in ROI space
    abs_weights = np.abs(weights_orig_space)

    if len(weights_orig_space) != len(roi_names):
        raise ValueError("Mismatch between PCA features and ROI names")

    # Sort weights and limit to top N
    sorted_idx = np.argsort(abs_weights)[::-1]
    if top_n is None:
        top_n = len(roi_names)
    top_idx = sorted_idx[:top_n]
    top_weights = weights_orig_space[top_idx]
    top_names = [roi_names[i + 1] for i in top_idx]

    # Color by direction: green for Expert, red for Non-Expert
    colors = ["#cb181d" if w > 0 else "#238b45" for w in top_weights]

    # Symmetric x-limits
    xmax = max(abs(top_weights)) * 1.1

    # Create bar plot
    # fig, ax = plt.subplots()

    # Get default figsize and scale height
    default_figsize = plt.rcParams["figure.figsize"]
    fig_width, fig_height = default_figsize
    fig, ax = plt.subplots(figsize=(fig_width, fig_height * 0.7))  # or any multiplier you like

    sns.barplot(
        x=top_weights,
        y=top_names,
        palette=colors,
        edgecolor=None,
        alpha=0.7,
        ax=ax,
        # width=0.4
    )
    ax.set_xlim(-xmax, xmax)

    # Title and labels (extra pad for legend space)
    ax.set_title("Top 10 ROI Contributions to Group Classification", pad=40)
    ax.set_xlabel("Weight in Original ROI Space")
    ax.set_ylabel("")

    # Move y-axis labels to the right and set color
    ax.yaxis.tick_right()
    ax.yaxis.set_label_position("right")
    ax.tick_params(axis='y', colors='gray', length=0)  # remove tick marks

    from matplotlib.font_manager import FontProperties
    # Use a font known to support Unicode arrows (like DejaVu Sans)
    arrow_font = FontProperties(family="DejaVu Sans", size=17
)

    # Arrow text using a separate font
    ax.text(
        0.5, 1.05,
        "← Higher dimensionality predictive of Novice   Higher dimensionality predictive of Expert →",
        ha="center",
        va="center",
        transform=ax.transAxes,
        fontproperties=arrow_font,
        color="black"
    )


    # Remove any grid lines
    ax.grid(False)

    # Remove top/right spines
    sns.despine()
    plt.tight_layout()
    plt.show()

# === PCA Projection with Classification ===
def plot_pr_pca_projection(
    expert_pr,
    nonexpert_pr,
    output_dir,
    ROI_NAME_MAP,
    title="PCA of Participation Ratios Across Subjects",
):
    """
    Run PCA on PR data, plot 2D projection, and classify with logistic regression.

    Returns
    -------
    pca : fitted PCA object
    clf : trained LogisticRegression
    coords : PCA-transformed 2D coordinates
    """

    # Use Ubuntu Condensed font
    mpl.rcParams["font.family"] = "Ubuntu Condensed"

    all_pr = np.vstack([expert_pr, nonexpert_pr])
    group_labels = np.array(
        ["Expert"] * len(expert_pr) + ["Non-Expert"] * len(nonexpert_pr)
    )
    binary_labels = (group_labels == "Expert").astype(int)

    scaler = StandardScaler()
    all_pr_scaled = scaler.fit_transform(all_pr)

    pca = PCA(n_components=2)
    coords = pca.fit_transform(all_pr_scaled)
    explained = pca.explained_variance_ratio_ * 100

    clf_2d = LogisticRegression(random_state=42).fit(coords, binary_labels)
    clf_full = LogisticRegression(random_state=42).fit(all_pr_scaled, binary_labels)

    # Compute bounds with padding
    x_min, x_max = coords[:, 0].min(), coords[:, 0].max()
    y_min, y_max = coords[:, 1].min(), coords[:, 1].max()
    x_range = x_max - x_min
    y_range = y_max - y_min
    x_min_p, x_max_p = x_min - 0.1 * x_range, x_max + 0.1 * x_range
    y_min_p, y_max_p = y_min - 0.1 * y_range, y_max + 0.1 * y_range

    # Create meshgrid with padded limits
    xx, yy = np.meshgrid(
        np.linspace(x_min_p, x_max_p, 300), np.linspace(y_min_p, y_max_p, 300)
    )
    Z = clf_2d.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

    # Plot
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.contourf(xx, yy, Z, alpha=0.15, cmap=plt.cm.RdYlGn, levels=[-1, 0.5, 2])
    colors = {"Expert": "green", "Non-Expert": "red"}
    for group in np.unique(group_labels):
        mask = group_labels == group
        ax.scatter(
            coords[mask, 0],
            coords[mask, 1],
            c=colors[group],
            s=250,
            alpha=0.7,
            label=group,
        )

    # Expand axis limits slightly
    ax.set_xlim(x_min_p, x_max_p)
    ax.set_ylim(y_min_p, y_max_p)
    ax.grid(False)

    ax.set_xlabel(f"PCA Dimension 1 ({explained[0]:.1f}% var)")
    ax.set_ylabel(f"PCA Dimension 2 ({explained[1]:.1f}% var)")
    ax.set_title(title, pad=20)
    ax.legend(title="Group", loc="best", frameon=False)
    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, "pr_subject_pca.png"), dpi=300)
    plt.show()

    return pca, clf_2d, clf_full, coords


# === Correlation between ELO and PR (per ROI) ===
def plot_elo_roi_correlation(elo_data, expert_pr, roi_names):
    stats = [
        {
            "ROI": roi,
            "r": linregress(elo_data, expert_pr[:, i]).rvalue,
            "p": linregress(elo_data, expert_pr[:, i]).pvalue,
        }
        for i, roi in enumerate(roi_names)
    ]
    df = pd.DataFrame(stats)
    reject, p_fdr = fdrcorrection(df["p"].values)
    df["p_fdr"] = p_fdr
    df["significant"] = reject
    df["roi_num"] = list(range(len(roi_names)))
    df = df.sort_values("roi_num")

    plt.figure()
    barplot = sns.barplot(
        x="ROI",
        y="r",
        data=df,
        palette=sns.diverging_palette(10, 240, n=len(df), center="light"),
        edgecolor="black",
    )
    plt.axhline(0, color="black", linestyle="--", linewidth=1)
    for i, row in df.iterrows():
        if row["significant"]:
            barplot.text(
                x=df.index.get_loc(i),
                y=row["r"] + 0.02 * np.sign(row["r"]),
                s="*",
                ha="center",
                va="bottom" if row["r"] > 0 else "top",
                fontsize=14,
            )
    plt.title("ELO vs Dimensionality (PR) by ROI\n(* FDR-corrected p < 0.05)")
    plt.xlabel("ROI")
    plt.ylabel("Correlation (r)")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.show()


# === Top N scatter plots: ELO vs PR ===
def plot_top_roi_scatter_grid(elo_values, expert_pr, roi_names, top_n=4):
    stats = [
        {
            "ROI": roi_names[i],
            "r": linregress(elo_values, expert_pr[:, i]).rvalue,
            "p": linregress(elo_values, expert_pr[:, i]).pvalue,
            "idx": i,
        }
        for i in range(expert_pr.shape[1])
    ]
    df = pd.DataFrame(stats).sort_values("r", key=abs, ascending=False).head(top_n)

    fig, axes = plt.subplots(1, top_n, figsize=(5 * top_n, 5), sharey=True)
    for ax, row in zip(axes, df.itertuples()):
        sns.regplot(
            x=elo_values,
            y=expert_pr[:, row.idx],
            ax=ax,
            color="royalblue",
            line_kws={"color": "black"},
        )
        ax.set_title(f"{row.ROI}\nr = {row.r:.2f}, p = {row.p:.3f}")
        ax.set_xlabel("ELO")
        ax.set_ylabel("PR")
    plt.tight_layout()
    plt.show()


def plot_pr_and_pca_heatmaps(
    pr_data, expert_subjects, nonexpert_subjects, roi_names, pca, n_components=2
):
    """
    Plot PR heatmap (subject x ROI) and PCA loadings (component x ROI) in a single figure.

    Parameters
    ----------
    pr_data : ndarray of shape (n_subjects, n_rois)
    expert_subjects : list of expert subject IDs
    nonexpert_subjects : list of non-expert subject IDs
    roi_names : list of ROI names in order
    pca : fitted PCA object
    n_components : number of PCA components to show
    """

    # Prepare PR dataframe
    subject_ids = expert_subjects + nonexpert_subjects
    df_pr = pd.DataFrame(pr_data, index=subject_ids, columns=roi_names)
    df_pr["Group"] = ["Expert"] * len(expert_subjects) + ["Non-Expert"] * len(
        nonexpert_subjects
    )

    # Prepare PCA loadings dataframe
    loadings = pca.components_[:n_components]  # shape: (n_components, n_rois)
    df_pca = pd.DataFrame(loadings, columns=roi_names)
    df_pca.index = [f"PC{i+1}" for i in range(n_components)]


    # Get default figsize and scale height
    default_figsize = plt.rcParams["figure.figsize"]
    fig_width, fig_height = default_figsize

    # Set up figure
    height_ratios = [len(subject_ids), n_components]
    fig, axes = plt.subplots(
        nrows=2,
        ncols=1,
        figsize=(fig_width, fig_height * 2),
        sharex=True,
        gridspec_kw={"height_ratios": height_ratios},
    )

    # --- PR Heatmap ---
    sns.heatmap(
        df_pr.drop(columns="Group"),
        ax=axes[0],
        cmap="mako",
        cbar=False,
        xticklabels=False,  # Hide x tick labels here
        yticklabels=False,
    )
    axes[0].set_ylabel("Subjects (Experts on top)")
    axes[0].set_title("Participation Ratios by Subject and ROI")

    # Draw line after expert rows
    axes[0].axhline(len(expert_subjects), color="black", linewidth=1.5)

    # --- PCA Loadings Heatmap ---
    max_abs = np.abs(df_pca.values).max()  # Find max absolute loading

    sns.heatmap(
        df_pca,
        ax=axes[1],
        cmap=BRAIN_CMAP,
        center=0,
        vmin=-max_abs,
        vmax=max_abs,
        cbar=False,
        xticklabels=True,
        yticklabels=True,
    )
    # Add thick black border
    for spine in axes[1].spines.values():
        spine.set_edgecolor("black")
        spine.set_linewidth(2.5)  # adjust thickness

    # axes[1].set_xlabel("ROI")
    # axes[1].set_ylabel("Principal Component")
    axes[1].set_title("ROI Contributions to PCA Components")

    # Fix x-tick alignment
    xticks = np.arange(len(roi_names)) + 0.5
    axes[1].set_xticks(xticks)
    axes[1].set_xticklabels(roi_names, rotation=30, ha="right", color="gray")

    plt.tight_layout()
    plt.show()


# === Run full analysis ===
ordered_roi_names = [ROI_NAME_MAP[k] for k in sorted(ROI_NAME_MAP.keys())]

pca, clf_2d, clf_full, coords = plot_pr_pca_projection(
    expert_pr=expert_pr,
    nonexpert_pr=nonexpert_pr,
    output_dir="figures",
    ROI_NAME_MAP=ROI_NAME_MAP,
    title="PCA of Participation Ratios Across Subjects",
)

all_pr = np.vstack([expert_pr, nonexpert_pr])
plot_pr_and_pca_heatmaps(
    pr_data=all_pr,
    expert_subjects=EXPERT_SUBJECTS,
    nonexpert_subjects=NONEXPERT_SUBJECTS,
    roi_names=ordered_roi_names,
    pca=pca,
    n_components=2,
)

plot_feature_importance_from_pca(clf=clf_full, roi_names=ROI_NAME_MAP, top_n=10)

plot_elo_roi_correlation(elo_data=ELOS, expert_pr=expert_pr, roi_names=ordered_roi_names)

plot_top_roi_scatter_grid(
    elo_values=ELOS, expert_pr=expert_pr, roi_names=ordered_roi_names, top_n=5
)
import logging
