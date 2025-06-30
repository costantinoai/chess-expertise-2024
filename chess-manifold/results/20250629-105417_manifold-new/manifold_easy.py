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
from typing import List, Dict
import numpy as np  # Numerical arrays
import pandas as pd  # Dataframes
import nibabel as nib  # NIfTI I/O
import scipy.io as sio  # MATLAB .mat I/O
from scipy.spatial import procrustes, distance_matrix  # Procrustes and distance computations
from scipy.stats import ttest_ind  # Statistical tests

from sklearn.decomposition import PCA  # Principal component analysis
from sklearn.preprocessing import StandardScaler  # Data standardization

import matplotlib.pyplot as plt  # Plotting
import seaborn as sns  # High-level plotting

import logging  # Logging utility

# ---------------------- CONFIGURATION ---------------------- #
BASE_PATH = "/data/projects/chess/data/BIDS/derivatives/fmriprep-SPM_smoothed-NO_GS-FD-HMP_brainmasked/MNI/fmriprep-SPM-MNI/GLM"  # Base path for GLM outputs
SPM_FILENAME = "SPM.mat"  # Name of the SPM.mat file
ATLAS_FILE = "/home/eik-tb/OneDrive_andreaivan.costantino@kuleuven.be/GitHub/chess-expertise-2024/chess-rois/results/glasser_cortex_bilateral/glasser_cortex_bilateral.nii"  # ROI atlas path
OUTPUT_ROOT = "results"  # Root directory for outputs

EXPERT_SUBJECTS = [  # List of expert subject IDs
    "03", "04", "06", "07", "08", "09", "10", "11", "12", "13",
    "16", "20", "22", "23", "24", "29", "30", "33", "34", "36",
]
NONEXPERT_SUBJECTS = [  # List of novice subject IDs
    "01", "02", "15", "17", "18", "19", "21", "25", "26", "27",
    "28", "32", "35", "37", "39", "40", "41", "42", "43", "44",
]
ALL_SUBJECTS = EXPERT_SUBJECTS + NONEXPERT_SUBJECTS  # Combined subject list

PROJECTION_DIM = 5000  # Threshold for random projection
ALPHA_FDR = 0.05  # FDR threshold
N_JOBS = 1  # Parallel jobs (-1 uses all cores)

sns.set(style="whitegrid")  # Seaborn style

# ---------------------- LOGGING SETUP ---------------------- #
logging.basicConfig(
    level=logging.INFO,  # Show INFO and above
    format="%(asctime)s [%(levelname)s] %(message)s",  # Log format
    datefmt="%Y-%m-%d %H:%M:%S",  # Date format
)
logger = logging.getLogger(__name__)  # Module-level logger

# ---------------------- UTILITY FUNCTIONS ---------------------- #
def create_run_id() -> str:  # Generate unique run ID
    now = datetime.now()  # Get current datetime
    return now.strftime("%Y%m%d-%H%M%S")  # Return formatted string


def save_script_to_file(script_path: str, out_directory: str):  # Copy script for provenance
    os.makedirs(out_directory, exist_ok=True)  # Ensure output directory exists
    dest = os.path.join(out_directory, os.path.basename(script_path))  # Destination path
    shutil.copy(script_path, dest)  # Copy file
    logger.info("Copied script to '%s'", dest)  # Log action


def get_spm_info(subject_id: str) -> dict:  # Load and average beta images
    spm_mat = os.path.join(BASE_PATH, f"sub-{subject_id}", "exp", SPM_FILENAME)  # Path to SPM.mat
    if not os.path.isfile(spm_mat):  # Check existence
        raise FileNotFoundError(f"Missing SPM.mat at {spm_mat}")  # Error if missing
    mat = sio.loadmat(spm_mat, struct_as_record=False, squeeze_me=True)  # Load .mat file
    SPM = mat["SPM"]  # Extract struct
    betas = SPM.Vbeta  # Beta entries
    names = SPM.xX.name  # Regressor names
    root = getattr(SPM, 'swd', os.path.dirname(spm_mat))  # Source directory
    pattern = re.compile(r"Sn\(\d+\)\s+(.*?)\*bf\(1\)")  # Regex for conditions
    cond_indices = {}  # Dict for indices
    for idx, nm in enumerate(names):  # Loop regressor names
        m = pattern.match(nm)  # Match regex
        if not m:  # Skip non-matches
            continue
        cond = m.group(1)  # Extract condition label
        cond_indices.setdefault(cond, []).append(idx)  # Add index
    averaged = {}  # Store averaged images
    for cond, inds in cond_indices.items():  # Per condition
        sum_data = None  # Initialize sum
        for i in inds:  # Loop indices
            entry = betas[i]  # Beta entry
            fname = getattr(entry, 'fname', None) or getattr(entry, 'filename')  # Filename
            img = nib.load(os.path.join(root, fname))  # Load image
            data = img.get_fdata(dtype=np.float32)  # Get data
            if sum_data is None:  # First image
                sum_data = np.zeros_like(data, dtype=np.float32)  # Init accumulator
                affine, header = img.affine, img.header  # Save metadata
            sum_data += data  # Accumulate
        avg = sum_data / len(inds)  # Compute average
        averaged[cond] = nib.Nifti1Image(avg, affine=affine, header=header)  # Store NIfTI
    logger.info("[%s] Extracted %d conditions", subject_id, len(averaged))  # Log info
    return averaged  # Return dict


def load_roi_voxel_data(subject_id: str, atlas_data: np.ndarray, roi_list: list) -> tuple:  # Extract ROI data
    imgs = get_spm_info(subject_id)  # Load condition images
    conds = sorted(imgs.keys())  # Sorted conditions
    n_cond = len(conds)  # Number of observations
    roi_dict = {}  # Store ROI arrays
    for roi in roi_list:  # For each ROI
        mask = atlas_data == roi  # Boolean mask
        n_vox = mask.sum()  # Count voxels
        roi_dict[roi] = np.zeros((n_cond, n_vox), dtype=np.float32)  # Init array
    for i, cond in enumerate(conds):  # For each condition
        data = imgs[cond].get_fdata(dtype=np.float32)  # Load data
        for roi in roi_list:  # Per ROI
            mask = atlas_data == roi  # ROI mask
            roi_dict[roi][i, :] = data[mask]  # Extract voxels
    logger.info("[%s] Loaded ROI data for %d conditions", subject_id, n_cond)  # Log info
    return roi_dict, conds  # Return data and labels


def compute_participation_ratio(data: np.ndarray) -> tuple:  # Compute manifold dimensionality
    cov = np.cov(data.T)  # Covariance matrix
    eig = np.linalg.eigvalsh(cov)  # Eigenvalues
    pr = (eig.sum() ** 2) / (eig**2).sum()  # Participation ratio
    return pr, eig / eig.sum()  # Return PR and normalized eigvals


def run_procrustes_alignment(Xs: List[np.ndarray]) -> List[np.ndarray]:  # Align via Procrustes
    ref = Xs[0]  # Reference matrix
    aligned = [ref]  # Store aligned
    for X in Xs[1:]:  # For each other
        _, Y, _ = procrustes(ref, X)  # Align
        aligned.append(Y)  # Append result
    return aligned  # Return aligned list


def compute_cluster_metrics(data: np.ndarray, labels: np.ndarray) -> tuple:  # Cluster metrics
    uniq = np.unique(labels)  # Unique labels
    cent = np.array([data[labels==u].mean(axis=0) for u in uniq])  # Centroids
    # Compactness = mean intra-cluster dist
    comp = np.mean([np.linalg.norm(data[labels==u] - c, axis=1).mean() for u, c in zip(uniq, cent)])
    # Inter-cluster = mean centroid dist
    dmat = distance_matrix(cent, cent)  # Pairwise
    inter = dmat[np.triu_indices(len(cent), k=1)].mean()  # Upper triangle mean
    return comp, inter, cent  # Return metrics


def run_subject_level_analysis(subject_ids: list, group_assignments: dict, atlas_data: np.ndarray, roi_labels: list) -> tuple:  # Subject data extraction
    subject_data, subject_labels = {}, {}  # Initialize containers
    for sid in subject_ids:  # For each subject
        logger.info("[%s] Extracting subject data", sid)  # Log
        roi_dict, conds = load_roi_voxel_data(sid, atlas_data, roi_labels)  # ROI data
        classes = np.array([int(c[0]=="C") for c in conds])  # Binary labels
        subject_data[sid], subject_labels[sid] = [], []  # Init lists
        for roi in roi_labels:  # For each ROI
            obs = roi_dict[roi]  # Observations × voxels
            mask = ~np.isnan(obs).any(axis=0)  # Drop NaN obs
            obs = obs[:, mask]  # Apply mask
            if obs.shape[0] < 2:  # Skip if too few obs
                continue
            subject_data[sid].append(obs)  # Store data
            subject_labels[sid].append(classes)  # Store labels
    return subject_data, subject_labels  # Return dicts

# ---------------------- CORE PIPELINE ---------------------- #
def analyze_manifolds(subject_data: Dict[str, List[np.ndarray]], subject_labels: Dict[str, List[np.ndarray]], group_assignments: Dict[str, str], n_components=10) -> tuple:  # Main analysis
    pca = PCA(n_components=n_components)  # PCA instance
    metrics = {'subject':[], 'group':[], 'participation_ratio':[], 'cluster_compactness':[], 'inter_cluster_distance':[]}  # Metrics df
    evals_list = []  # Explained variance
    expert_emb, novice_emb = [], []  # Embeddings
    for sid, runs in subject_data.items():  # Per subject
        data = np.vstack(runs)  # Stack runs
        labels = np.concatenate(subject_labels[sid])  # Stack labels
        data = StandardScaler().fit_transform(data)  # Standardize
        red = pca.fit_transform(data)  # PCA reduction
        pr, eig_norm = compute_participation_ratio(red)  # Dimensionality
        comp, inter, _ = compute_cluster_metrics(red, labels)  # Cluster metrics
        metrics['subject'].append(sid)  # Append subject
        metrics['group'].append(group_assignments[sid])  # Append group
        metrics['participation_ratio'].append(pr)  # Append PR
        metrics['cluster_compactness'].append(comp)  # Append compactness
        metrics['inter_cluster_distance'].append(inter)  # Append inter-cluster
        evals_list.append(eig_norm)  # Append eigvals
        if group_assignments[sid]=='expert':  # Expert group
            expert_emb.append(red)  # Store embedding
        else:  # Novice group
            novice_emb.append(red)  # Store embedding
    exp_al = run_procrustes_alignment(expert_emb)  # Align experts
    nov_al = run_procrustes_alignment(novice_emb)  # Align novices
    return pd.DataFrame(metrics), exp_al, nov_al, np.vstack(evals_list)  # Return results

# ---------------------- PLOTTING FUNCTIONS ---------------------- #
def plot_metric_distributions(df: pd.DataFrame, metric: str, title: str=None):  # Violin+strip plot
    plt.figure(figsize=(8,6))  # Figure size
    sns.violinplot(x="group", y=metric, data=df, inner=None, alpha=0.5)  # Violin
    sns.stripplot(x="group", y=metric, data=df, jitter=True, color='k', alpha=0.7)  # Points
    g1 = df[df['group']=='expert'][metric]  # Expert values
    g2 = df[df['group']=='novice'][metric]  # Novice values
    t, p = ttest_ind(g1, g2, equal_var=False)  # T-test
    plt.title(f"{title or metric} (p={p:.3f})")  # Title
    plt.xlabel('Group')  # X label
    plt.ylabel(metric)  # Y label
    plt.tight_layout()  # Layout
    plt.show()  # Display


def plot_mean_aligned_embeddings(aligned: List[np.ndarray], labels: np.ndarray, group: str):  # Mean manifold
    mean_emb = np.stack(aligned).mean(axis=0)  # Mean across subjects
    plt.figure(figsize=(6,5))  # Figure
    for u in np.unique(labels):  # Each class
        idx = labels==u  # Mask
        plt.scatter(mean_emb[idx,0], mean_emb[idx,1], label=f"Label {u}", alpha=0.6)  # Plot
    plt.title(f"{group.capitalize()} Mean Manifold")  # Title
    plt.xlabel('Dim 1')  # X label
    plt.ylabel('Dim 2')  # Y label
    plt.legend()  # Legend
    plt.tight_layout()  # Layout
    plt.show()  # Display


def plot_explained_variance_by_group(evals: np.ndarray, groups: np.ndarray):  # Explained variance
    df = pd.DataFrame(evals)  # Dataframe
    df['group'] = groups  # Add group
    melt = df.melt(id_vars='group', var_name='Component', value_name='Variance')  # Melt
    plt.figure(figsize=(8,5))  # Figure
    sns.lineplot(data=melt, x='Component', y='Variance', hue='group', marker='o')  # Line plot
    plt.title('Explained Variance by Component')  # Title
    plt.tight_layout()  # Layout
    plt.show()  # Display

# ---------------------- MAIN EXECUTION ---------------------- #
run_id = create_run_id()  # Unique ID
out_dir = os.path.join(OUTPUT_ROOT, f"{run_id}_manifold-new")  # Output dir
os.makedirs(out_dir, exist_ok=True)  # Create
logger.info("Output directory: %s", out_dir)  # Log
save_script_to_file(__file__, out_dir)  # Save script

atlas_img = nib.load(ATLAS_FILE)  # Load atlas
atlas_data = atlas_img.get_fdata().astype(int)  # Get data
roi_labels = sorted([l for l in np.unique(atlas_data) if l!=0])  # ROI list
logger.info("Loaded %d ROIs", len(roi_labels))  # Log

subs = ALL_SUBJECTS  # All subjects
groups = {s:'expert' for s in EXPERT_SUBJECTS}  # Expert mapping
groups.update({s:'novice' for s in NONEXPERT_SUBJECTS})  # Novice mapping

subj_data, subj_labels = run_subject_level_analysis(subs, groups, atlas_data, roi_labels)  # Extract data
metrics_df, exp_align, nov_align, eigs = analyze_manifolds(subj_data, subj_labels, groups)  # Analyze

plot_metric_distributions(metrics_df, 'participation_ratio', 'Participation Ratio by Group')  # Plot PR
plot_metric_distributions(metrics_df, 'cluster_compactness', 'Cluster Compactness by Group')  # Plot compactness
plot_metric_distributions(metrics_df, 'inter_cluster_distance', 'Inter-Cluster Distance')  # Plot inter-dist
plot_explained_variance_by_group(eigs, metrics_df['group'].values)  # Plot variance

# Example aligned plot for first subject in each group
first_exp = EXPERT_SUBJECTS[0]  # First expert
first_nov = NONEXPERT_SUBJECTS[0]  # First novice
plot_mean_aligned_embeddings(exp_align, np.concatenate(subj_labels[first_exp]), 'expert')  # Expert manifold
plot_mean_aligned_embeddings(nov_align, np.concatenate(subj_labels[first_nov]), 'novice')  # Novice manifold

metrics_df.to_csv(os.path.join(out_dir, 'group_metrics.csv'), index=False)  # Save metrics
logger.info("Saved metrics to CSV")  # Log completion
