import os
import re
import numpy as np
import nibabel as nib
import scipy.io as sio
import matplotlib.pyplot as plt
from scipy.stats import linregress
from joblib import Parallel, delayed
from sklearn.decomposition import PCA

from config import GLM_BASE_PATH, ATLAS_CORTICES, EXPERTS, NONEXPERTS
from meta import ROI_NAME_MAP, ROI_COLORS


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
    for _, row in df.iterrows():
        ax.text(row["VoxelCount"], row[y_col], row["ROI_Label"], fontsize=8, ha="left", va="bottom")
    os.makedirs(out_dir, exist_ok=True)
    plt.savefig(os.path.join(out_dir, fname), dpi=300, bbox_inches="tight")
    plt.close(fig)

