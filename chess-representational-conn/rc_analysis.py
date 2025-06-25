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

# Paths copied from the MVPA scripts
BASE_PATH = "/data/projects/chess/data/BIDS/derivatives/fmriprep-SPM_smoothed-NO_GS-FD-HMP_brainmasked/MNI/fmriprep-SPM-MNI/GLM"
SPM_FILENAME = "SPM.mat"
ATLAS_FILE = "/data/projects/chess/data/misc/templates/tpl-MNI152NLin2009cAsym_res-02_atlas-Glasser2016_desc-cortices_bilateral_resampled.nii"

logger = logging.getLogger(__name__)


def load_spm_betas(subject_id: str):
    """
    Load and average beta images from SPM for a given subject.

    This function loads the `SPM.mat` file to identify regressors,
    groups beta images by condition name, and averages beta images
    across runs or sessions for each condition.

    Parameters
    ----------
    subject_id : str
        Subject identifier (e.g., '01' for sub-01).

    Returns
    -------
    averaged : dict[str, nib.Nifti1Image]
        Dictionary mapping condition names to averaged beta NIfTI images.
    """
    # Construct path to subject's SPM.mat file
    spm_mat = os.path.join(BASE_PATH, f"sub-{subject_id}", "exp", SPM_FILENAME)

    if not os.path.isfile(spm_mat):
        raise FileNotFoundError(f"SPM.mat not found at {spm_mat}")

    # Load the SPM.mat file and extract the SPM struct
    spm = sio.loadmat(spm_mat, struct_as_record=False, squeeze_me=True)["SPM"]

    # Get beta file info and regressor names
    beta_info = spm.Vbeta
    regressor_names = spm.xX.name

    # Pattern to extract condition names (e.g., "Sn(1) condition*bf(1)" → "condition")
    pattern = r"Sn\(\d+\)\s+(.*?)\*bf\(1\)"
    cond_dict: dict[str, list[int]] = {}

    # Match and group beta indices by condition name
    for i, name in enumerate(regressor_names):
        m = re.match(pattern, name)
        if m:
            cond = m.group(1)
            cond_dict.setdefault(cond, []).append(i)

    # Determine the directory where beta images are stored
    spm_dir = spm.swd if hasattr(spm, "swd") else os.path.dirname(spm_mat)

    averaged = {}

    # For each condition, average its associated beta images
    for cond, indices in cond_dict.items():
        sum_data = None
        affine = None
        header = None

        for idx in indices:
            # SPM versions differ: use either `fname` or `filename`
            fname = getattr(beta_info[idx], "fname", getattr(beta_info[idx], "filename", None))
            img = nib.load(os.path.join(spm_dir, fname))
            data = img.get_fdata(dtype=np.float32)

            # Initialize accumulator and metadata
            if sum_data is None:
                sum_data = np.zeros_like(data, dtype=np.float32)
                affine = img.affine
                header = img.header

            sum_data += data  # accumulate beta image

        # Average the accumulated beta images and store result
        averaged[cond] = nib.Nifti1Image(sum_data / len(indices), affine, header)

    return averaged

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from typing import Union

def extract_roi_data(
    betas: dict,
    atlas_data: np.ndarray,
    roi_labels: list[int],
    reduce_dim: Union[int, float, None] = None,
    center: bool = True
):
    """
    Extract voxel patterns per ROI for all conditions, with optional centering and dimensionality reduction.

    Parameters
    ----------
    betas : dict
        Dictionary mapping condition names to Nifti1Image beta maps.
    atlas_data : np.ndarray
        3D atlas data with ROI labels.
    roi_labels : list of int
        List of ROI labels to extract.
    reduce_dim : int, float, or None
        If int: number of PCA components to keep.
        If float (0 < value < 1): amount of explained variance to retain.
        If None: no dimensionality reduction.
    center : bool
        If True, subtract the mean activation pattern across conditions for each ROI
        (i.e., center each column by its mean across rows).

    Returns
    -------
    roi_data : dict
        Dictionary mapping ROI labels to condition-by-feature matrices (cleaned, possibly centered and PCA-reduced).
    conditions : list of str
        List of condition names, sorted alphabetically.
    """
    conditions = sorted(betas.keys())
    roi_data = {}

    for roi in roi_labels:
        n_voxels = int(np.sum(atlas_data == roi))
        data_matrix = np.zeros((len(conditions), n_voxels), dtype=np.float32)

        # Fill matrix: rows = conditions, columns = voxels
        for ci, cond in enumerate(conditions):
            data = betas[cond].get_fdata()
            mask = atlas_data == roi
            data_matrix[ci, :] = data[mask]

        # --- Sanitize columns ---
        valid_cols = ~np.isnan(data_matrix).any(axis=0) & (np.nanstd(data_matrix, axis=0) > 1e-5)
        clean_data = data_matrix[:, valid_cols]

        # --- Optional centering: remove mean pattern across conditions ---
        if center:
            clean_data = clean_data - np.mean(clean_data, axis=0, keepdims=True)

        # --- Optional PCA ---
        if reduce_dim is not None:
            scaler = StandardScaler()
            clean_data = scaler.fit_transform(clean_data)

            if isinstance(reduce_dim, int):
                pca = PCA(n_components=reduce_dim)
            elif isinstance(reduce_dim, float) and 0 < reduce_dim < 1:
                pca = PCA(n_components=reduce_dim)
            else:
                raise ValueError("reduce_dim must be None, a positive integer, or a float between 0 and 1.")

            clean_data = pca.fit_transform(clean_data)

        roi_data[roi] = clean_data

    return roi_data, conditions

def compute_rdm(X: np.ndarray) -> np.ndarray:
    """Compute a Pearson correlation distance RDM."""
    dist = pdist(X, metric="sqeuclidean")  # 1 - r
    return squareform(dist)


def _flatten_upper(mat: np.ndarray) -> np.ndarray:
    idx = np.triu_indices_from(mat, k=1)
    return mat[idx]

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
    roi_data, _ = extract_roi_data(betas, atlas, roi_labels, reduce_dim=None, center=True)

    # Compute RDMs after cleaning
    rdms = {}
    for roi, data in roi_data.items():
        rdms[roi] = compute_rdm(data)

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
    tstat_vec, pval_vec = ttest_ind(a_lt, b_lt, axis=0, nan_policy="omit", equal_var=False)

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
