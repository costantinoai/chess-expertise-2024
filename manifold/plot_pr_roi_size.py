import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from viz_utils import make_brain_cmap
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

from common_utils import create_run_id, save_script_to_file
from config import GLM_BASE_PATH, ATLAS_CORTICES, EXPERTS, NONEXPERTS


BRAIN_CMAP = make_brain_cmap()

# Plot styles
sns.set_style("white", {"axes.grid": False})
base_font_size = 22
plt.rcParams.update(
    {
        "font.family": "Ubuntu Condensed",
        "font.size": base_font_size,
        "axes.titlesize": base_font_size * 1.4,
        "axes.labelsize": base_font_size * 1.2,
        "xtick.labelsize": base_font_size,
        "ytick.labelsize": base_font_size,
        "legend.fontsize": base_font_size,
        "figure.figsize": (12, 9),
    }
)

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
CUSTOM_COLORS = [
    "#c6dbef", "#c6dbef", "#2171b5", "#2171b5", "#2171b5",
    "#a1d99b", "#a1d99b", "#a1d99b", "#a1d99b",
    "#00441b", "#00441b", "#00441b",
    "#fbb4b9", "#fbb4b9",
    "#cb181d", "#cb181d", "#cb181d", "#cb181d",
    "#fec44f", "#fec44f", "#fec44f", "#fec44f",
]

ROI_NAME_MAP = {
    1: "Primary Visual", 2: "Early Visual", 3: "Dorsal Stream Visual",
    4: "Ventral Stream Visual", 5: "MT+ Complex", 6: "Somatosensory and Motor",
    7: "Paracentral Lobular and Mid Cing", 8: "Premotor", 9: "Posterior Opercular",
    10: "Early Auditory", 11: "Auditory Association", 12: "Insular and Frontal Opercular",
    13: "Medial Temporal", 14: "Lateral Temporal", 15: "Temporo-Parieto Occipital Junction",
    16: "Superior Parietal", 17: "Inferior Parietal", 18: "Posterior Cing",
    19: "Anterior Cing and Medial Prefrontal", 20: "Orbital and Polar Frontal",
    21: "Inferior Frontal", 22: "Dorsolateral Prefrontal",
}

EXPERT_SUBJECTS = list(EXPERTS)
NONEXPERT_SUBJECTS = list(NONEXPERTS)

ELOS = [
    2041, 1850, 1751, 2241, 1941, 1969, 2100, 2051, 2150, 1824,
    1881, 2269, 2010, 2100, 2094, 1879, 2232, 2073, 2083, 2189,
]

BASE_PATH = str(GLM_BASE_PATH)
SPM_FILENAME = "SPM.mat"
ATLAS_FILE = str(ATLAS_CORTICES)


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
        affine = header = None
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
    ax.scatter(df["VoxelCount"], df[y_col], c=df["Color"], edgecolor="black", s=70)
    ax.set_xlabel("Voxel Count (ROI size)")
    ax.set_ylabel(y_col)
    ax.set_title(title)
    for i, row in df.iterrows():
        ax.text(row["VoxelCount"], row[y_col], row["ROI_Label"], fontsize=8, ha="left", va="bottom")
    os.makedirs(out_dir, exist_ok=True)
    plt.savefig(os.path.join(out_dir, fname), dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    run_id = create_run_id()
    out_dir = os.path.join("results", f"{run_id}_pr_vs_roi_size")
    os.makedirs(out_dir, exist_ok=True)
    save_script_to_file(out_dir)

    atlas_img = nib.load(ATLAS_FILE)
    atlas_data = atlas_img.get_fdata().astype(int)
    unique_rois = np.unique(atlas_data)
    unique_rois = unique_rois[unique_rois != 0]

    expert_data = Parallel(n_jobs=-1)(delayed(process_subject)(sid, atlas_data, unique_rois) for sid in EXPERT_SUBJECTS)
    novice_data = Parallel(n_jobs=-1)(delayed(process_subject)(sid, atlas_data, unique_rois) for sid in NONEXPERT_SUBJECTS)
    expert_pr = np.vstack([x[0] for x in expert_data])
    novice_pr = np.vstack([x[0] for x in novice_data])
    voxel_counts = np.nanmean(np.vstack([x[1] for x in expert_data + novice_data]), axis=0)

    df = pd.DataFrame({
        "ROI": unique_rois.astype(int),
        "ROI_Label": [ROI_NAME_MAP.get(int(r), str(int(r))) for r in unique_rois],
        "VoxelCount": voxel_counts,
    })
    df["PR_Expert"] = np.nanmean(expert_pr, axis=0)
    df["PR_NonExpert"] = np.nanmean(novice_pr, axis=0)
    df["PR_Diff"] = df["PR_Expert"] - df["PR_NonExpert"]
    df["Color"] = [CUSTOM_COLORS[(i - 1) % len(CUSTOM_COLORS)] for i in df["ROI"]]

    plot_voxelcount_vs_pr(df, "PR_Expert", "Experts: PR vs ROI Size", "experts_voxel_vs_pr.png", out_dir)
    plot_voxelcount_vs_pr(df, "PR_NonExpert", "Non-Experts: PR vs ROI Size", "nonexperts_voxel_vs_pr.png", out_dir)
    plot_voxelcount_vs_pr(df, "PR_Diff", "PR Difference vs ROI Size", "diff_voxel_vs_pr.png", out_dir)

    # Optional: simple 2D PCA classification from PR (subject x ROI)
    all_pr = np.vstack([expert_pr, novice_pr])
    labels = np.array([1] * len(expert_pr) + [0] * len(novice_pr))
    pca = PCA(n_components=2)
    X2 = pca.fit_transform(all_pr)
    clf = LogisticRegression()
    clf.fit(X2, labels)
    w = clf.coef_[0]
    b = clf.intercept_[0]
    xx, yy = np.meshgrid(
        np.linspace(X2[:, 0].min() - 1, X2[:, 0].max() + 1, 200),
        np.linspace(X2[:, 1].min() - 1, X2[:, 1].max() + 1, 200),
    )
    Z = (w[0] * xx + w[1] * yy + b)
    plt.figure(figsize=(8, 6))
    plt.contour(xx, yy, Z, levels=[0], colors="k", linestyles="--")
    plt.scatter(X2[: len(expert_pr), 0], X2[: len(expert_pr), 1], c="#4CAF50", label="Experts")
    plt.scatter(X2[len(expert_pr):, 0], X2[len(expert_pr):, 1], c="#F44336", label="Novices")
    plt.legend()
    plt.title("PCA of Participation Ratios Across Subjects")
    plt.savefig(os.path.join(out_dir, "pr_pca_classification.png"), dpi=300, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    main()

