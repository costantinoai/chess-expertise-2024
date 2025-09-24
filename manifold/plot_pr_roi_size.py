import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from viz_utils import make_brain_cmap
from joblib import Parallel, delayed
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from matplotlib.colors import LinearSegmentedColormap

from common.common_utils import create_run_id, save_script_to_file
from config import ATLAS_CORTICES, EXPERTS, NONEXPERTS
from meta import ROI_NAME_MAP, ROI_COLORS, apply_plot_style
from manifold.modules.pr_utils import (
    process_subject,
    plot_voxelcount_vs_pr,
)


BRAIN_CMAP = make_brain_cmap()

# Plot styles
apply_plot_style(22)

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
EXPERT_SUBJECTS = list(EXPERTS)
NONEXPERT_SUBJECTS = list(NONEXPERTS)
ATLAS_FILE = str(ATLAS_CORTICES)


## All function bodies moved to manifold.modules.pr_utils


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
    df["Color"] = [ROI_COLORS[(i - 1) % len(ROI_COLORS)] for i in df["ROI"]]

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
