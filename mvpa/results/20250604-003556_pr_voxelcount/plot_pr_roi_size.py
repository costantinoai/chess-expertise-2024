import os
import re
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import nibabel as nib
import scipy.io as sio
from sklearn.decomposition import PCA
from scipy.stats import linregress
from joblib import Parallel, delayed
from modules.helpers import create_run_id, save_script_to_file

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
CUSTOM_COLORS = [
    "#c6dbef", "#c6dbef", "#2171b5", "#2171b5", "#2171b5",
    "#a1d99b", "#a1d99b", "#a1d99b", "#a1d99b",
    "#00441b", "#00441b", "#00441b",
    "#fbb4b9", "#fbb4b9",
    "#cb181d", "#cb181d", "#cb181d", "#cb181d",
    "#fec44f", "#fec44f", "#fec44f", "#fec44f"
]

ROI_NAME_MAP = {
    1: "Primary Visual", 2: "Early Visual", 3: "Dorsal Stream Visual", 4: "Ventral Stream Visual",
    5: "MT+ Complex", 6: "Somatosensory and Motor", 7: "Paracentral Lobular and Mid Cing",
    8: "Premotor", 9: "Posterior Opercular", 10: "Early Auditory", 11: "Auditory Association",
    12: "Insular and Frontal Opercular", 13: "Medial Temporal", 14: "Lateral Temporal",
    15: "Temporo-Parieto Occipital Junction", 16: "Superior Parietal", 17: "Inferior Parietal",
    18: "Posterior Cing", 19: "Anterior Cing and Medial Prefrontal",
    20: "Orbital and Polar Frontal", 21: "Inferior Frontal", 22: "Dorsolateral Prefrontal"
}

EXPERT_SUBJECTS = ["03", "04", "06", "07", "08", "09", "10", "11", "12", "13", "16", "20", "22", "23", "24", "29", "30", "33", "34", "36"]
NONEXPERT_SUBJECTS = ["01", "02", "15", "17", "18", "19", "21", "25", "26", "27", "28", "32", "35", "37", "39", "40", "41", "42", "43", "44"]

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
    output = {roi: np.zeros((len(conditions), np.sum(atlas_data == roi)), dtype=np.float32) for roi in unique_rois}
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
    return (var.sum() ** 2) / np.sum(var ** 2), cleaned.shape[1]

def process_subject(subject_id, atlas_data, unique_rois):
    roi_data = load_roi_voxel_data(subject_id, atlas_data, unique_rois)
    pr, vox = [], []
    for roi in unique_rois:
        p, v = compute_entropy_pr(roi_data[roi])
        pr.append(p)
        vox.append(v)
    return np.array(pr), np.array(vox)

def plot_voxelcount_vs_pr(df, y_col, title_prefix, fname, out_dir):
    sns.set_style("white", {'axes.grid': False})
    slope, intercept, r, p, _ = linregress(df["VoxelCount"], df[y_col])
    title = f"{title_prefix} (r = {r:.2f}, p = {p:.3f})"
    fig, ax = plt.subplots(figsize=(10, 10))
    sns.set_style("whitegrid")
    ax.scatter(df["VoxelCount"], df[y_col], c=df["Color"], edgecolor="black", s=250, linewidth=0, alpha=0.7)
    ax.plot(df["VoxelCount"], intercept + slope * df["VoxelCount"], color="black", linestyle="--", linewidth=1)
    # for _, row in df.iterrows():
    #     ax.text(row["VoxelCount"], row[y_col], row["ROI_Name"], fontsize=8, ha="left", alpha=0.75)
    ax.set_xlabel("Average (Experts and Novices) Number of Voxels in ROI", fontsize=14)
    ax.set_ylabel("Participation Ratio (Experts - Non-Experts)", fontsize=14)
    ax.set_title(title, fontsize=21, pad=20)
    ax.tick_params(labelsize=12)
    # === SPINES & GRID ===
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.5)
    ax.spines["bottom"].set_linewidth(0.5)

    plt.tight_layout()
    path = os.path.join(out_dir, fname)
    plt.savefig(path, dpi=300)
    plt.show()
    print(f"Saved: {path}")

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

output_dir = f"results/{create_run_id()}_pr_voxelcount"
os.makedirs(output_dir, exist_ok=True)
save_script_to_file(output_dir)

atlas_data = nib.load(ATLAS_FILE).get_fdata().astype(int)
unique_rois = np.unique(atlas_data)
unique_rois = unique_rois[unique_rois != 0]

expert_results = Parallel(n_jobs=-1)(delayed(process_subject)(sub, atlas_data, unique_rois) for sub in EXPERT_SUBJECTS)
nonexpert_results = Parallel(n_jobs=-1)(delayed(process_subject)(sub, atlas_data, unique_rois) for sub in NONEXPERT_SUBJECTS)

expert_pr = np.array([r[0] for r in expert_results])
expert_vox = np.array([r[1] for r in expert_results])
nonexpert_pr = np.array([r[0] for r in nonexpert_results])
nonexpert_vox = np.array([r[1] for r in nonexpert_results])

df = pd.DataFrame({
    "ROI_Label": unique_rois,
    "ROI_Name": [ROI_NAME_MAP.get(roi, f"ROI {roi}") for roi in unique_rois],
    "Color": CUSTOM_COLORS[:len(unique_rois)],
    "VoxelCount": np.mean(np.vstack([expert_vox, nonexpert_vox]), axis=0),
    "PR_Expert": np.nanmean(expert_pr, axis=0),
    "PR_NonExpert": np.nanmean(nonexpert_pr, axis=0)
})
df["PR_Diff"] = df["PR_Expert"] - df["PR_NonExpert"]

plot_voxelcount_vs_pr(df, "PR_Expert", "Experts: PR vs ROI Size", "experts_voxel_vs_pr.png", output_dir)
plot_voxelcount_vs_pr(df, "PR_NonExpert", "Non-Experts: PR vs ROI Size", "nonexperts_voxel_vs_pr.png", output_dir)
plot_voxelcount_vs_pr(df, "PR_Diff", "PR Difference vs. Average ROI Size", "diff_voxel_vs_pr.png", output_dir)


import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import MDS
from sklearn.preprocessing import StandardScaler

# -----------------------------------------------------------------------------
# MDS Plot of PR Vectors for Experts and Non-Experts
# -----------------------------------------------------------------------------

def plot_pr_mds(expert_pr, nonexpert_pr, output_dir, roi_labels=None, title="MDS of Participation Ratios Across Subjects"):
    # Combine all subject PR vectors
    all_pr = np.vstack([expert_pr, nonexpert_pr])
    group_labels = np.array(["Expert"] * len(expert_pr) + ["Non-Expert"] * len(nonexpert_pr))

    # Remove ROIs with NaNs in any subject (to avoid MDS errors)
    valid_mask = ~np.any(np.isnan(all_pr), axis=0)
    all_pr_cleaned = all_pr[:, valid_mask]

    # Optionally z-score features (ROIs) for stability
    all_pr_scaled = StandardScaler().fit_transform(all_pr_cleaned)

    # Run MDS
    mds = MDS(n_components=2, random_state=42, dissimilarity='euclidean')
    mds_coords = mds.fit_transform(all_pr_scaled)

    # Plot
    fig, ax = plt.subplots(figsize=(8, 8))
    colors = {"Expert": "green", "Non-Expert": "red"}
    for group in np.unique(group_labels):
        mask = group_labels == group
        ax.scatter(
            mds_coords[mask, 0],
            mds_coords[mask, 1],
            c=colors[group],
            label=group,
            edgecolor="black",
            s=100,
            alpha=0.8
        )

    ax.set_xlabel("MDS Dimension 1", fontsize=14)
    ax.set_ylabel("MDS Dimension 2", fontsize=14)
    ax.set_title(title, fontsize=21, pad=20)
    ax.legend(title="Group")
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    path = os.path.join(output_dir, "pr_subject_mds.png")
    plt.savefig(path, dpi=300)
    plt.show()
    print(f"Saved MDS plot: {path}")

# -----------------------------------------------------------------------------
# Call the function with your data
# -----------------------------------------------------------------------------

plot_pr_mds(expert_pr, nonexpert_pr, output_dir)

def plot_pr_pca(expert_pr, nonexpert_pr, output_dir, title="PCA of Participation Ratios Across Subjects"):
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    import numpy as np
    import os
    sns.set_style("white", {'axes.grid': False})

    # === Combine data ===
    all_pr = np.vstack([expert_pr, nonexpert_pr])
    group_labels = np.array(["Expert"] * len(expert_pr) + ["Non-Expert"] * len(nonexpert_pr))

    # === Remove ROIs with any NaN across subjects ===
    valid_mask = ~np.any(np.isnan(all_pr), axis=0)
    all_pr_cleaned = all_pr[:, valid_mask]

    # === Standardize features (ROIs) ===
    all_pr_scaled = StandardScaler().fit_transform(all_pr_cleaned)

    # === PCA projection ===
    pca = PCA(n_components=2)
    pca_coords = pca.fit_transform(all_pr_scaled)
    explained = pca.explained_variance_ratio_ * 100

    # === Plotting ===
    fig, ax = plt.subplots(figsize=(10, 10))
    colors = {"Expert": "green", "Non-Expert": "red"}
    for group in np.unique(group_labels):
        mask = group_labels == group
        ax.scatter(
            pca_coords[mask, 0],
            pca_coords[mask, 1],
            c=colors[group],
            s=250,
            alpha=0.7,
            label=group
        )

    ax.set_xlabel(f"PCA Dimension 1 ({explained[0]:.1f}% var)", fontsize=14)
    ax.set_ylabel(f"PCA Dimension 2 ({explained[1]:.1f}% var)", fontsize=14)
    ax.set_title(title, fontsize=21, pad=20)
    ax.tick_params(labelsize=12)

    # === Spines, grid, layout ===
    ax.set_title(title, fontsize=21, pad=20)
    ax.tick_params(labelsize=12)
    # === SPINES & GRID ===
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.5)
    ax.spines["bottom"].set_linewidth(0.5)


    # === Legend ===
    ax.legend(title="Group", fontsize=21, title_fontsize=21, loc="best", frameon=False)

    # === Save and show ===
    plt.tight_layout()
    out_path = os.path.join(output_dir, "pr_subject_pca.png")
    plt.savefig(out_path, dpi=300)
    plt.show()
    print(f"Saved PCA plot: {out_path}")


plot_pr_pca(expert_pr, nonexpert_pr, output_dir)

def plot_pr_tsne(expert_pr, nonexpert_pr, output_dir, title="t-SNE of Participation Ratios Across Subjects"):
    import os
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.manifold import TSNE
    from sklearn.preprocessing import StandardScaler

    # === Combine PR data ===
    all_pr = np.vstack([expert_pr, nonexpert_pr])
    group_labels = np.array(["Expert"] * len(expert_pr) + ["Non-Expert"] * len(nonexpert_pr))

    # === Remove ROIs with any NaNs across subjects ===
    valid_mask = ~np.any(np.isnan(all_pr), axis=0)
    all_pr_cleaned = all_pr[:, valid_mask]

    # === Standardize (z-score) across ROIs ===
    all_pr_scaled = StandardScaler().fit_transform(all_pr_cleaned)

    # === t-SNE Projection ===
    tsne = TSNE(n_components=2, perplexity=10, random_state=42, init='pca', n_iter=1000)
    tsne_coords = tsne.fit_transform(all_pr_scaled)

    # === Plot ===
    fig, ax = plt.subplots(figsize=(10, 10))
    colors = {"Expert": "green", "Non-Expert": "red"}
    for group in np.unique(group_labels):
        mask = group_labels == group
        ax.scatter(
            tsne_coords[mask, 0],
            tsne_coords[mask, 1],
            c=colors[group],
            edgecolor="black",
            s=250,
            alpha=0.8,
            label=group
        )

    ax.set_xlabel("t-SNE Dimension 1", fontsize=14)
    ax.set_ylabel("t-SNE Dimension 2", fontsize=14)
    ax.set_title(title, fontsize=21, pad=20)
    ax.tick_params(labelsize=12)

    # === Spines & Grid ===
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.5)
    ax.spines["bottom"].set_linewidth(0.5)

    # === Legend ===
    ax.legend(title="Group", fontsize=21, title_fontsize=21, loc="best", frameon=False)

    # === Save ===
    plt.tight_layout()
    out_path = os.path.join(output_dir, "pr_subject_tsne.png")
    plt.savefig(out_path, dpi=300)
    plt.show()
    print(f"Saved t-SNE plot: {out_path}")

plot_pr_tsne(expert_pr, nonexpert_pr, output_dir)
