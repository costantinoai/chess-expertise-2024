# -*- coding: utf-8 -*-
"""Representational connectivity analysis utilities."""

import os
import re
import logging
import sys
import numpy as np
import nibabel as nib
import scipy.io as sio
from scipy.spatial.distance import pdist, squareform
from scipy.stats import ttest_1samp, ttest_ind
from statsmodels.stats.multitest import multipletests

# Add the parent directory of 'modules' to sys.path
sys.path.append('/home/eik-tb/OneDrive_andreaivan.costantino@kuleuven.be/GitHub/chess-expertise-2024/chess-mvpa')

# Reuse constants and ROI manager from the MVPA module
from modules import (
    MANAGER,
    EXPERT_SUBJECTS,
    NONEXPERT_SUBJECTS,
)

# Paths copied from the MVPA scripts
BASE_PATH = "/data/projects/chess/data/BIDS/derivatives/fmriprep-SPM_smoothed-NO_GS-FD-HMP_brainmasked/MNI/fmriprep-SPM-MNI/GLM"
SPM_FILENAME = "SPM.mat"
ATLAS_FILE = "/data/projects/chess/data/misc/templates/tpl-MNI152NLin2009cAsym_res-02_atlas-Glasser2016_desc-cortices_bilateral_resampled.nii"

logger = logging.getLogger(__name__)


def load_spm_betas(subject_id: str):
    """Load and average beta images from SPM for one subject."""
    spm_mat = os.path.join(BASE_PATH, f"sub-{subject_id}", "exp", SPM_FILENAME)
    if not os.path.isfile(spm_mat):
        raise FileNotFoundError(f"SPM.mat not found at {spm_mat}")

    spm = sio.loadmat(spm_mat, struct_as_record=False, squeeze_me=True)["SPM"]
    beta_info = spm.Vbeta
    regressor_names = spm.xX.name

    pattern = r"Sn\(\d+\)\s+(.*?)\*bf\(1\)"
    cond_dict: dict[str, list[int]] = {}
    for i, name in enumerate(regressor_names):
        m = re.match(pattern, name)
        if m:
            cond = m.group(1)
            cond_dict.setdefault(cond, []).append(i)

    spm_dir = spm.swd if hasattr(spm, "swd") else os.path.dirname(spm_mat)
    averaged = {}
    for cond, indices in cond_dict.items():
        sum_data = None
        affine = None
        header = None
        for idx in indices:
            fname = getattr(beta_info[idx], "fname", getattr(beta_info[idx], "filename", None))
            img = nib.load(os.path.join(spm_dir, fname))
            data = img.get_fdata(dtype=np.float32)
            if sum_data is None:
                sum_data = np.zeros_like(data, dtype=np.float32)
                affine = img.affine
                header = img.header
            sum_data += data
        averaged[cond] = nib.Nifti1Image(sum_data / len(indices), affine, header)
    return averaged


def extract_roi_data(betas: dict, atlas_data: np.ndarray, roi_labels: list[int]):
    """Extract voxel patterns per ROI for all conditions."""
    conditions = sorted(betas.keys())
    roi_data = {
        roi: np.zeros((len(conditions), int(np.sum(atlas_data == roi))), dtype=np.float32)
        for roi in roi_labels
    }
    for ci, cond in enumerate(conditions):
        data = betas[cond].get_fdata()
        for roi in roi_labels:
            mask = atlas_data == roi
            roi_data[roi][ci, :] = data[mask]
    return roi_data, conditions


def compute_rdm(X: np.ndarray) -> np.ndarray:
    """Compute a Pearson correlation distance RDM."""
    dist = pdist(X, metric="correlation")  # 1 - r
    return squareform(dist)


def _flatten_upper(mat: np.ndarray) -> np.ndarray:
    idx = np.triu_indices_from(mat, k=1)
    return mat[idx]


def compute_subject_matrix(subject_id: str, atlas_path: str = ATLAS_FILE):
    """Return the representational connectivity matrix for one subject."""
    logger.info(f"Processing subject {subject_id}")
    atlas = nib.load(atlas_path).get_fdata().astype(int)
    roi_labels = [r.region_id for r in MANAGER.rois]
    betas = load_spm_betas(subject_id)
    roi_data, _ = extract_roi_data(betas, atlas, roi_labels)
    rdms = {roi: compute_rdm(data) for roi, data in roi_data.items()}
    n_rois = len(roi_labels)
    conn = np.zeros((n_rois, n_rois), dtype=np.float32)
    for i, r1 in enumerate(roi_labels):
        v1 = _flatten_upper(rdms[r1])
        for j, r2 in enumerate(roi_labels):
            v2 = _flatten_upper(rdms[r2])
            if np.all(np.isnan(v1)) or np.all(np.isnan(v2)):
                conn[i, j] = np.nan
            else:
                conn[i, j] = np.corrcoef(v1, v2)[0, 1]
    return conn, roi_labels


def fisher_z(r: np.ndarray) -> np.ndarray:
    """Fisher r-to-z transform."""
    return np.arctanh(np.clip(r, -0.999999, 0.999999))


def group_statistics(mats: list[np.ndarray]):
    """Compute mean matrix and one-sample t-test across subjects."""
    data = np.stack(mats, axis=0)
    mean = np.nanmean(data, axis=0)
    tstat, pvals = ttest_1samp(data, 0, nan_policy="omit")
    reject, pvals_fdr, _, _ = multipletests(pvals.flatten(), method="fdr_bh")
    reject = reject.reshape(pvals.shape)
    pvals_fdr = pvals_fdr.reshape(pvals.shape)
    return mean, tstat, pvals, pvals_fdr, reject


def group_difference(expert_mats: list[np.ndarray], novice_mats: list[np.ndarray]):
    """Difference between two groups."""
    a = np.stack(expert_mats)
    b = np.stack(novice_mats)
    diff_mean = np.nanmean(a, axis=0) - np.nanmean(b, axis=0)
    tstat, pvals = ttest_ind(a, b, axis=0, nan_policy="omit")
    reject, pvals_fdr, _, _ = multipletests(pvals.flatten(), method="fdr_bh")
    reject = reject.reshape(pvals.shape)
    pvals_fdr = pvals_fdr.reshape(pvals.shape)
    return diff_mean, tstat, pvals, pvals_fdr, reject
