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
from sklearn.covariance import GraphicalLasso

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


from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# def compute_subject_matrix(subject_id: str, atlas_path: str = ATLAS_FILE):
#     """Return the representational connectivity matrix for one subject using Graphical Lasso with PCA."""
#     logger.info(f"Processing subject {subject_id}")

#     # Load atlas and extract ROI labels (excluding background/0)
#     atlas = nib.load(atlas_path).get_fdata().astype(int)
#     roi_labels = np.unique(atlas)[1:]

#     # Load subject betas
#     betas = load_spm_betas(subject_id)

#     # Extract voxelwise data per ROI
#     roi_data, _ = extract_roi_data(betas, atlas, roi_labels)

#     # Compute RDMs after removing NaN or constant columns
#     rdms = {}
#     for roi, data in roi_data.items():
#         valid_cols = ~np.isnan(data).any(axis=0) & (np.nanstd(data, axis=0) > 1e-5)
#         clean_data = data[:, valid_cols]
#         rdms[roi] = compute_rdm(clean_data)

#     # Prepare matrix of flattened RDMs
#     flat_rdms = []
#     valid_rois = []
#     for roi in roi_labels:
#         vec = _flatten_upper(rdms[roi])
#         if not np.all(np.isnan(vec)):
#             flat_rdms.append(vec)
#             valid_rois.append(roi)

#     if len(flat_rdms) < 2:
#         raise ValueError("Not enough valid ROIs for graphical lasso.")

#     # Stack and standardize
#     X = np.vstack(flat_rdms)
#     X = StandardScaler().fit_transform(X)

#     # Apply PCA to reduce dimensionality (at most n_rois - 1)
#     X = PCA(n_components=22).fit_transform(X)

#     # Fit Graphical Lasso
#     model = GraphicalLasso(alpha=0.001, max_iter=500)
#     model.fit(X)

#     return model.precision_, np.array(roi_labels)

from scipy.stats import spearmanr

def compute_subject_matrix(subject_id: str, atlas_path: str = ATLAS_FILE):
    """Return the lower-triangle (no diagonal) Spearman connectivity matrix for one subject."""
    logger.info(f"Processing subject {subject_id}")

    # Load atlas and extract ROI labels (excluding background/0)
    atlas = nib.load(atlas_path).get_fdata().astype(int)
    roi_labels = np.unique(atlas)[1:]

    # Load subject betas
    betas = load_spm_betas(subject_id)

    # Extract voxelwise data per ROI
    roi_data, _ = extract_roi_data(betas, atlas, roi_labels)

    # Compute RDMs after cleaning
    rdms = {}
    for roi, data in roi_data.items():
        valid_cols = ~np.isnan(data).any(axis=0) & (np.nanstd(data, axis=0) > 1e-5)
        clean_data = data[:, valid_cols]
        if clean_data.shape[1] > 1:
            rdms[roi] = compute_rdm(clean_data)

    # Flatten RDMs
    flat_rdms = []
    valid_rois = []
    for roi in roi_labels:
        if roi in rdms:
            vec = _flatten_upper(rdms[roi])
            if not np.all(np.isnan(vec)):
                flat_rdms.append(vec)
                valid_rois.append(roi)

    if len(flat_rdms) < 2:
        raise ValueError("Not enough valid ROIs for correlation matrix.")

    # Stack into matrix (n_rois x n_features)
    X = np.vstack(flat_rdms)

    # Compute full Spearman correlation matrix
    rho, _ = spearmanr(X, axis=1)
    n = len(valid_rois)

    # Create matrix and mask upper triangle and diagonal
    mat = np.full((n, n), np.nan)
    i_lower = np.tril_indices(n, k=-1)
    mat[i_lower] = rho[i_lower]

    return mat, np.array(valid_rois)


def fisher_z(r: np.ndarray) -> np.ndarray:
    """Fisher r-to-z transform."""
    return np.arctanh(np.clip(r, -0.999999, 0.999999))
# def fisher_z(r: np.ndarray) -> np.ndarray:
#     """Fisher r-to-z transform."""
#     return r


def group_statistics(mats: list[np.ndarray]):
    """Compute mean matrix and one-sample t-test across subjects, testing only lower triangle (no diagonal)."""
    data = np.stack(mats, axis=0)
    mean = np.nanmean(data, axis=0)

    # Get lower triangle indices (excluding diagonal)
    n = mean.shape[0]
    i_lower = np.tril_indices(n, k=-1)

    # Extract lower triangle values across subjects
    data_lt = data[:, i_lower[0], i_lower[1]]

    # T-test on lower triangle values only
    tstat_vec, pval_vec = ttest_1samp(data_lt, popmean=0, nan_policy="omit")

    # FDR correction
    reject_vec, pvals_fdr_vec, _, _ = multipletests(pval_vec, method="fdr_bh")

    # Fill full matrices with NaN
    tstat = np.full((n, n), np.nan)
    pvals = np.full((n, n), np.nan)
    pvals_fdr = np.full((n, n), np.nan)
    reject = np.zeros((n, n), dtype=bool)

    # Map results to lower triangle
    tstat[i_lower] = tstat_vec
    pvals[i_lower] = pval_vec
    pvals_fdr[i_lower] = pvals_fdr_vec
    reject[i_lower] = reject_vec

    return mean, tstat, pvals, pvals_fdr, reject

def group_difference(expert_mats: list[np.ndarray], novice_mats: list[np.ndarray]):
    """Compute mean difference and t-test between groups, testing only lower triangle (no diagonal)."""
    a = np.stack(expert_mats)
    b = np.stack(novice_mats)

    diff_mean = np.nanmean(a, axis=0) - np.nanmean(b, axis=0)

    # Get lower triangle indices
    n = diff_mean.shape[0]
    i_lower = np.tril_indices(n, k=-1)

    # Extract values across subjects
    a_lt = a[:, i_lower[0], i_lower[1]]
    b_lt = b[:, i_lower[0], i_lower[1]]

    # T-test
    tstat_vec, pval_vec = ttest_ind(a_lt, b_lt, axis=0, nan_policy="omit")

    # FDR correction
    reject_vec, pvals_fdr_vec, _, _ = multipletests(pval_vec, method="fdr_bh")

    # Fill full matrices with NaN
    tstat = np.full((n, n), np.nan)
    pvals = np.full((n, n), np.nan)
    pvals_fdr = np.full((n, n), np.nan)
    reject = np.zeros((n, n), dtype=bool)

    # Map results to lower triangle
    tstat[i_lower] = tstat_vec
    pvals[i_lower] = pval_vec
    pvals_fdr[i_lower] = pvals_fdr_vec
    reject[i_lower] = reject_vec

    return diff_mean, tstat, pvals, pvals_fdr, reject
