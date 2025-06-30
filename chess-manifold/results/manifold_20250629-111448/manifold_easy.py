#!/usr/bin/env python3  # Use Python 3 interpreter
# -*- coding: utf-8 -*-  # Define file encoding
"""
Created on Sun Jun 29 10:39:56 2025  # Script creation timestamp

@author: costantino_ai  # Script author
"""

import os  # Filesystem operations
import re  # Regular expressions
import shutil  # File copying
from datetime import datetime  # Timestamps
from typing import List, Dict  # Type hints

import numpy as np  # Numerical arrays
import pandas as pd  # Dataframes
import nibabel as nib  # NIfTI I/O
import scipy.io as sio  # MATLAB .mat I/O
from scipy.spatial import procrustes, distance_matrix  # Procrustes and distance computations
from scipy.stats import ttest_ind  # Statistical tests

from sklearn.decomposition import PCA  # Principal component analysis
from sklearn.preprocessing import StandardScaler  # Data standardization
from sklearn.metrics import pairwise_distances  # Pairwise distance metrics

import matplotlib.pyplot as plt  # 2D plotting
import seaborn as sns  # High-level plotting
from mpl_toolkits.mplot3d import Axes3D  # 3D plotting

from joblib import Parallel, delayed  # Parallel processing
from scipy.cluster.hierarchy import linkage, dendrogram  # Hierarchical clustering
import matplotlib.animation as animation  # Animations

import logging  # Logging utility

# ---------------------- CONFIGURATION ---------------------- #
BASE_PATH = "/data/projects/chess/data/BIDS/derivatives/fmriprep-SPM_smoothed-NO_GS-FD-HMP_brainmasked/MNI/fmriprep-SPM-MNI/GLM"  # Base path for GLM outputs
SPM_FILENAME = "SPM.mat"  # Name of the SPM.mat file
ATLAS_FILE = "/home/eik-tb/.../glasser_cortex_bilateral.nii"  # Path to ROI atlas
OUTPUT_ROOT = "results"  # Root directory for all outputs

EXPERT_SUBJECTS = ["03","04","06","07","08","09","10","11","12","13","16","20","22","23","24","29","30","33","34","36"]  # Expert IDs
NONEXPERT_SUBJECTS = ["01","02","15","17","18","19","21","25","26","27","28","32","35","37","39","40","41","42","43","44"]  # Novice IDs
ALL_SUBJECTS = EXPERT_SUBJECTS + NONEXPERT_SUBJECTS  # Combined subject list

N_JOBS = -1  # Number of parallel jobs (-1 uses all cores)
sns.set(style="whitegrid")  # Seaborn style for plots

# ---------------------- LOGGING SETUP ---------------------- #
logging.basicConfig(
    level=logging.INFO,  # Log INFO and above
    format="%(asctime)s [%(levelname)s] %(message)s",  # Format
    datefmt="%Y-%m-%d %H:%M:%S"  # Date format
)
logger = logging.getLogger(__name__)  # Module logger

# ---------------------- UTILITY FUNCTIONS ---------------------- #
def create_run_id() -> str:  # Generate a unique run ID
    now = datetime.now()  # Current time
    return now.strftime("%Y%m%d-%H%M%S")  # Format as string


def save_script(script_path: str, out_dir: str):  # Save a copy of this script
    os.makedirs(out_dir, exist_ok=True)  # Ensure output directory exists
    dest = os.path.join(out_dir, os.path.basename(script_path))  # Destination path
    shutil.copy(script_path, dest)  # Copy file
    logger.info("Script copied to %s", dest)  # Log action


def get_spm_betas(subject: str) -> Dict[str, nib.Nifti1Image]:  # Load and average SPM betas
    spm_mat = os.path.join(BASE_PATH, f"sub-{subject}", "exp", SPM_FILENAME)  # Path
    mat = sio.loadmat(spm_mat, struct_as_record=False, squeeze_me=True)  # Load .mat
    SPM = mat['SPM']  # Extract struct
    betas = SPM.Vbeta  # Beta entries
    names = SPM.xX.name  # Condition names
    root = getattr(SPM, 'swd', os.path.dirname(spm_mat))  # Source dir
    pattern = re.compile(r"Sn\(\d+\)\s+(.*?)\*bf\(1\)")  # Regex
    cond_idx = {}  # Condition indices map
    for i, nm in enumerate(names):  # Loop names
        m = pattern.match(nm)  # Match
        if not m:  # Skip non-match
            continue
        cond = m.group(1)  # Extract condition label
        cond_idx.setdefault(cond, []).append(i)  # Append index
    averaged = {}  # Storage
    for cond, idxs in cond_idx.items():  # For each condition
        acc, affine, hdr = None, None, None  # Init
        for i in idxs:  # Loop beta indices
            entry = betas[i]  # Entry
            fname = getattr(entry,'fname',None) or getattr(entry,'filename')  # Filename
            img = nib.load(os.path.join(root, fname))  # Load NIfTI
            data = img.get_fdata(dtype=np.float32)  # Data
            if acc is None:  # First image
                acc = np.zeros_like(data)  # Accumulator
                affine, hdr = img.affine, img.header  # Metadata
            acc += data  # Sum
        averaged[cond] = nib.Nifti1Image(acc/len(idxs), affine, hdr)  # Average
    return averaged  # Return dict


def extract_roi_data(subject: str, atlas: np.ndarray, rois: List[int]) -> (Dict[int, np.ndarray], List[str]):  # ROI extraction
    imgs = get_spm_betas(subject)  # Condition images
    conds = sorted(imgs.keys())  # Sorted conditions
    n = len(conds)  # Number observations
    data = {}  # ROI map
    for roi in rois:  # For each ROI label
        mask = atlas == roi  # Boolean mask
        nvox = mask.sum()  # Count voxels
        arr = np.zeros((n, nvox), dtype=np.float32)  # Init array
        for i, cond in enumerate(conds):  # Loop conditions
            vol = imgs[cond].get_fdata(dtype=np.float32)  # Load data
            arr[i, :] = vol[mask]  # Extract voxels
        data[roi] = arr  # Store
    return data, conds  # Return data and labels


def process_subject(subject: str, atlas: np.ndarray, rois: List[int]) -> (str, Dict[int, np.ndarray], np.ndarray):  # Single subject
    roi_data, conds = extract_roi_data(subject, atlas, rois)  # Get ROI data
    labels = np.array([int(c[0] == 'C') for c in conds])  # Binary labels
    return subject, roi_data, labels  # Return tuple

# ---------------------- PARALLEL EXTRACTION ---------------------- #
def parallel_extraction(subjects: List[str], atlas: np.ndarray, rois: List[int]) -> (Dict[str, Dict[int, np.ndarray]], Dict[str, np.ndarray]):  # Parallel load
    results = Parallel(n_jobs=N_JOBS)(delayed(process_subject)(s, atlas, rois) for s in subjects)  # Parallel
    data, labs = {}, {}  # Init
    for subj, roi_data, labels in results:  # Collect
        data[subj] = roi_data  # ROI dict
        labs[subj] = labels  # Labels
    return data, labs  # Return

# ---------------------- MANIFOLD HELPERS ---------------------- #
def participation_ratio(data: np.ndarray) -> (float, np.ndarray):  # Manifold dimensionality
    cov = np.cov(data.T)  # Covariance
    eig = np.linalg.eigvalsh(cov)  # Eigenvalues
    pr = (eig.sum()**2) / (eig**2).sum()  # Participation ratio
    return pr, eig / eig.sum()  # Return ratio and normed eigs


def cluster_metrics(data: np.ndarray, labels: np.ndarray) -> (float, float, np.ndarray):  # Cluster metrics
    uniq = np.unique(labels)  # Unique labels
    centroids = np.array([data[labels==u].mean(0) for u in uniq])  # Centroids
    compact = np.mean([np.linalg.norm(data[labels==u] - c, axis=1).mean() for u, c in zip(uniq, centroids)])  # Intra
    dmat = distance_matrix(centroids, centroids)  # Inter-centroid
    separation = dmat[np.triu_indices(len(centroids), k=1)].mean()  # Off-diagonals mean
    return compact, separation, centroids  # Return


def procrustes_align(matrices: List[np.ndarray]) -> List[np.ndarray]:  # Align via Procrustes
    ref = matrices[0]  # Reference
    aligned = [ref]  # Store
    for M in matrices[1:]:  # For others
        _, Y, _ = procrustes(ref, M)  # Align
        aligned.append(Y)  # Append
    return aligned  # Return list

# ---------------------- COMPUTE METRICS per ROI ---------------------- #
def compute_metrics_per_roi(data: Dict[str, Dict[int, np.ndarray]], labels: Dict[str, np.ndarray], groups: Dict[str, str], n_comp: int = 10):
    records = []  # Metrics list
    exp_align, nov_align = {}, {}  # Align buffers

    for subj, roi_dict in data.items():  # Loop subjects
        grp = groups[subj]  # Expert/novice
        for roi, mat in roi_dict.items():  # Loop ROIs
            X = StandardScaler().fit_transform(mat)  # Standardize
            red = PCA(n_components=n_comp).fit_transform(X)  # PCA
            pr, _ = participation_ratio(red)  # Dimensionality
            comp, sep, _ = cluster_metrics(red, labels[subj])  # Cluster

            records.append({'subject': subj, 'group': grp, 'roi': roi, 'participation_ratio': pr, 'compactness': comp, 'separation': sep})  # Append

            if grp == 'expert':  # Store for alignment
                exp_align.setdefault(roi, []).append(red)
            else:
                nov_align.setdefault(roi, []).append(red)

    # Align per ROI
    aligned_expert = {roi: procrustes_align(mats) for roi, mats in exp_align.items()}  # Map
    aligned_novice = {roi: procrustes_align(mats) for roi, mats in nov_align.items()}  # Map

    return pd.DataFrame(records), aligned_expert, aligned_novice  # Return

# ---------------------- PLOTTING per ROI ---------------------- #
def plot_pr_histogram_per_roi(metrics_df: pd.DataFrame, out_dir: str):  # PR histogram
    for roi in sorted(metrics_df['roi'].unique()):  # Loop ROIs
        df = metrics_df[metrics_df['roi'] == roi]  # Subset
        plt.figure(figsize=(6,4))  # Figure
        sns.histplot(data=df, x='participation_ratio', hue='group', element='step', stat='density', common_norm=False)  # Plot
        plt.title(f"ROI {roi} Participation Ratio")  # Title
        plt.xlabel("Participation Ratio")  # X label
        plt.ylabel("Density")  # Y label
        plt.tight_layout()  # Layout
        plt.savefig(os.path.join(out_dir, f"roi_{roi}_pr_hist.png"))  # Save
        plt.close()  # Close


def plot_centroid_shift_per_roi(aligned_ex: Dict[int,List[np.ndarray]], aligned_nv: Dict[int,List[np.ndarray]], labels: np.ndarray, out_dir: str):  # Centroid shifts
    for roi in aligned_ex.keys():  # Loop ROIs
        me = np.mean(np.stack(aligned_ex[roi]), axis=0)  # Expert mean
        mn = np.mean(np.stack(aligned_nv[roi]), axis=0)  # Novice mean
        uniq = np.unique(labels)  # Labels
        exp_cent = np.array([me[labels==u].mean(0) for u in uniq])  # Expert centroids
        nov_cent = np.array([mn[labels==u].mean(0) for u in uniq])  # Novice

        plt.figure(figsize=(5,5))  # Figure
        for i,u in enumerate(uniq):  # Each label
            plt.plot([exp_cent[i,0], nov_cent[i,0]], [exp_cent[i,1], nov_cent[i,1]], 'k--', alpha=0.6)  # Line
            plt.scatter(exp_cent[i,0], exp_cent[i,1], color='blue', alpha=0.8)  # Expert point
            plt.scatter(nov_cent[i,0], nov_cent[i,1], color='red', alpha=0.8)  # Novice
        plt.title(f"ROI {roi} Centroid Shift")  # Title
        plt.xlabel("Dim 1")  # X label
        plt.ylabel("Dim 2")  # Y label
        plt.tight_layout()  # Layout
        plt.savefig(os.path.join(out_dir, f"roi_{roi}_centroid_shift.png"))  # Save
        plt.close()  # Close


def plot_3d_manifold_per_roi(aligned: Dict[int,List[np.ndarray]], labels: np.ndarray, group_name: str, out_dir: str):  # 3D manifold
    for roi, mats in aligned.items():  # Loop ROIs
        mean_emb = np.mean(np.stack(mats), axis=0)  # Mean embedding
        fig = plt.figure(figsize=(6,5))  # Figure
        ax = fig.add_subplot(111, projection='3d')  # 3D axis
        for u in np.unique(labels):  # Each label
            idx = labels==u  # Mask
            ax.scatter(mean_emb[idx,0], mean_emb[idx,1], mean_emb[idx,2], label=f"{u}", alpha=0.6)  # Scatter
        ax.set_title(f"{group_name} ROI {roi} 3D Manifold")  # Title
        ax.set_xlabel("Dim 1")  # Labels
        ax.set_ylabel("Dim 2")
        ax.set_zlabel("Dim 3")
        ax.legend()  # Legend
        plt.tight_layout()  # Layout
        plt.savefig(os.path.join(out_dir, f"{group_name.lower()}_roi_{roi}_3d.png"))  # Save
        plt.close()  # Close


def plot_dendrogram_per_roi(data: Dict[str, Dict[int,np.ndarray]], labels: Dict[str,np.ndarray], roi: int, out_dir: str):  # Dendrogram
    # Use first subject as representative
    subj = list(data.keys())[0]  # Example
    X = StandardScaler().fit_transform(data[subj][roi])  # Data
    d = pairwise_distances(X, metric='euclidean')  # Distance
    Z = linkage(d, method='average')  # Linkage
    plt.figure(figsize=(8,4))  # Figure
    dendrogram(Z, labels=[f"{l}" for l in labels[subj]], leaf_rotation=90)  # Plot
    plt.title(f"ROI {roi} Dendrogram (Subj {subj})")  # Title
    plt.tight_layout()  # Layout
    plt.savefig(os.path.join(out_dir, f"roi_{roi}_dendrogram.png"))  # Save
    plt.close()  # Close

# ---------------------- MAIN EXECUTION ---------------------- #
if __name__ == '__main__':
    run_id = create_run_id()  # Run ID
    out_dir = os.path.join(OUTPUT_ROOT, f"manifold_{run_id}")  # Output directory
    os.makedirs(out_dir, exist_ok=True)  # Create
    save_script(__file__, out_dir)  # Save script

    atlas_img = nib.load(ATLAS_FILE)  # Load atlas
    atlas = atlas_img.get_fdata().astype(int)  # Numpy array
    rois = [r for r in np.unique(atlas) if r != 0]  # ROI labels
    groups = {**{s:'expert' for s in EXPERT_SUBJECTS}, **{s:'novice' for s in NONEXPERT_SUBJECTS}}  # Map

    data, labs = parallel_extraction(ALL_SUBJECTS, atlas, rois)  # Load data
    metrics_df, aligned_exp, aligned_nov = compute_metrics_per_roi(data, labs, groups)  # Compute metrics

    metrics_df.to_csv(os.path.join(out_dir, 'metrics.csv'), index=False)  # Save metrics
    logger.info("Metrics saved to %s", out_dir)  # Log

    # Generate plots per ROI
    plot_pr_histogram_per_roi(metrics_df, out_dir)  # PR histograms
    plot_centroid_shift_per_roi(aligned_exp, aligned_nov, labs[ALL_SUBJECTS[0]], out_dir)  # Centroid shifts
    plot_3d_manifold_per_roi(aligned_exp, labs[ALL_SUBJECTS[0]], 'Expert', out_dir)  # 3D expert
    plot_3d_manifold_per_roi(aligned_nov, labs[ALL_SUBJECTS[0]], 'Novice', out_dir)  # 3D novice
    for roi in rois:  # Dendrograms
        plot_dendrogram_per_roi(data, labs, roi, out_dir)

    logger.info("All figures saved.")  # Done message
