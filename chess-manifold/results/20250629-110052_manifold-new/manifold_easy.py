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
N_JOBS = -1  # Parallel jobs (-1 uses all cores)

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

from joblib import Parallel, delayed

def _process_one_subject(sid: str, atlas_data: np.ndarray, roi_labels: list) -> tuple:
    """Process a single subject: extract ROI data and condition labels."""
    try:
        logger.info("[%s] Extracting subject data", sid)
        roi_dict, conds = load_roi_voxel_data(sid, atlas_data, roi_labels)
        classes = np.array([int(c[0] == "C") for c in conds])

        subj_data, subj_labels = [], []
        for roi in roi_labels:
            obs = roi_dict[roi]
            mask = ~np.isnan(obs).any(axis=0)
            obs = obs[:, mask]
            if obs.shape[0] < 2:
                continue
            subj_data.append(obs)
            subj_labels.append(classes)

        return sid, subj_data, subj_labels
    except Exception as e:
        logger.error("[%s] Error during processing: %s", sid, str(e))
        return sid, None, None

def run_subject_level_analysis_parallel(subject_ids: list, group_assignments: dict, atlas_data: np.ndarray, roi_labels: list, n_jobs: int = -1) -> tuple:
    """Run subject-level analysis in parallel."""
    results = Parallel(n_jobs=n_jobs)(
        delayed(_process_one_subject)(sid, atlas_data, roi_labels) for sid in subject_ids
    )

    subject_data = {}
    subject_labels = {}
    for sid, data, labels in results:
        if data is not None and labels is not None:
            subject_data[sid] = data
            subject_labels[sid] = labels
        else:
            logger.warning("[%s] Skipped due to processing error or empty data", sid)

    return subject_data, subject_labels


from sklearn.manifold import MDS

def plot_participation_histogram(metrics_df):
    plt.figure(figsize=(8, 5))
    sns.histplot(data=metrics_df, x='participation_ratio', hue='group', element='step', stat='density', common_norm=False)
    plt.title("Participation Ratio Distribution by Group")
    plt.xlabel("Participation Ratio")
    plt.ylabel("Density")
    plt.tight_layout()
    plt.show()

def plot_centroid_shift(exp_aligned, nov_aligned, labels):
    """Assumes same stimuli labels across aligned subjects"""
    mean_exp = np.mean(np.stack(exp_aligned), axis=0)
    mean_nov = np.mean(np.stack(nov_aligned), axis=0)
    unique_labels = np.unique(labels)

    exp_centroids = np.array([mean_exp[labels == l].mean(axis=0) for l in unique_labels])
    nov_centroids = np.array([mean_nov[labels == l].mean(axis=0) for l in unique_labels])

    plt.figure(figsize=(6, 6))
    for i, l in enumerate(unique_labels):
        plt.plot([exp_centroids[i, 0], nov_centroids[i, 0]],
                 [exp_centroids[i, 1], nov_centroids[i, 1]],
                 'k--', alpha=0.6)
        plt.scatter(exp_centroids[i, 0], exp_centroids[i, 1], color='blue', label='Expert' if i==0 else "", alpha=0.8)
        plt.scatter(nov_centroids[i, 0], nov_centroids[i, 1], color='red', label='Novice' if i==0 else "", alpha=0.8)
    plt.legend()
    plt.title("Cluster Centroid Shifts (Expert vs. Novice)")
    plt.xlabel("Dim 1")
    plt.ylabel("Dim 2")
    plt.tight_layout()
    plt.show()

def plot_pca_trajectory(subject_data, subject_labels, group_assignments):
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))
    for i, group in enumerate(["expert", "novice"]):
        group_data = []
        group_labels = []
        for subj in subject_data:
            if group_assignments[subj] != group:
                continue
            X = np.vstack(subject_data[subj])
            y = np.concatenate(subject_labels[subj])
            group_data.append(StandardScaler().fit_transform(X))
            group_labels.append(y)
        X_all = np.vstack(group_data)
        y_all = np.concatenate(group_labels)
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_all)
        axs[i].scatter(X_pca[:, 0], X_pca[:, 1], c=y_all, cmap='coolwarm', alpha=0.6)
        axs[i].set_title(f"{group.capitalize()} PCA Projection")
        axs[i].set_xlabel("PC1")
        axs[i].set_ylabel("PC2")
    plt.suptitle("PCA Trajectories of Stimuli Across Groups")
    plt.tight_layout()
    plt.show()

def plot_individual_manifolds(subject_data, subject_labels, group_assignments, n_subjects=5):
    """Plot a few individual subject manifolds using MDS."""
    subjects_by_group = {
        "expert": [s for s in subject_data if group_assignments[s] == "expert"][:n_subjects],
        "novice": [s for s in subject_data if group_assignments[s] == "novice"][:n_subjects],
    }

    fig, axes = plt.subplots(nrows=2, ncols=n_subjects, figsize=(4*n_subjects, 8), sharex=True, sharey=True)
    for i, group in enumerate(["expert", "novice"]):
        for j, subj in enumerate(subjects_by_group[group]):
            data = np.vstack(subject_data[subj])
            labels = np.concatenate(subject_labels[subj])
            mds = MDS(n_components=2, dissimilarity='euclidean', random_state=42)
            proj = mds.fit_transform(StandardScaler().fit_transform(data))
            ax = axes[i, j]
            for label in np.unique(labels):
                idx = labels == label
                ax.scatter(proj[idx, 0], proj[idx, 1], label=f'Class {label}', alpha=0.7)
            ax.set_title(f"{group.capitalize()} - Subj {subj}")
            ax.axis('off')
    plt.suptitle("Individual Subject Manifold Structures (MDS)", fontsize=16)
    plt.tight_layout()
    plt.show()

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

subj_data, subj_labels = run_subject_level_analysis_parallel(
    subject_ids=subs,
    group_assignments=groups,
    atlas_data=atlas_data,
    roi_labels=roi_labels,
    n_jobs=N_JOBS  # from config
)

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

plot_individual_manifolds(subj_data, subj_labels, groups)
plot_pca_trajectory(subj_data, subj_labels, groups)
plot_centroid_shift(exp_align, nov_align, np.concatenate(subj_labels[EXPERT_SUBJECTS[0]]))  # or your common label set
plot_participation_histogram(metrics_df)

from mpl_toolkits.mplot3d import Axes3D

def plot_3d_aligned_manifold(aligned_data, labels, group_name):
    """3D scatter plot of average aligned manifold."""
    mean_embedding = np.mean(np.stack(aligned_data), axis=0)
    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, projection='3d')
    for label in np.unique(labels):
        idx = labels == label
        ax.scatter(mean_embedding[idx, 0], mean_embedding[idx, 1], mean_embedding[idx, 2],
                   label=f"Label {label}", alpha=0.6)
    ax.set_title(f"3D Aligned Manifold - {group_name}")
    ax.set_xlabel("Dim 1")
    ax.set_ylabel("Dim 2")
    ax.set_zlabel("Dim 3")
    ax.legend()
    plt.tight_layout()
    plt.show()

from scipy.cluster.hierarchy import linkage, dendrogram
from sklearn.metrics import pairwise_distances

def plot_rdm_dendrogram(subject_data, subject_labels, subject_id):
    """Plots a dendrogram from the average RDM of a subject."""
    X = np.vstack(subject_data[subject_id])
    y = np.concatenate(subject_labels[subject_id])
    dists = pairwise_distances(StandardScaler().fit_transform(X), metric='euclidean')
    Z = linkage(dists, method='average')

    plt.figure(figsize=(10, 5))
    dendrogram(Z, labels=[f"L{l}" for l in y], leaf_rotation=90)
    plt.title(f"Dendrogram of Stimuli (Subj {subject_id})")
    plt.tight_layout()
    plt.show()

import matplotlib.animation as animation

def animate_pca_trajectory(subject_data, subject_labels, subject_id, save_path=None):
    X = np.vstack(subject_data[subject_id])
    y = np.concatenate(subject_labels[subject_id])
    X_std = StandardScaler().fit_transform(X)
    X_pca = PCA(n_components=2).fit_transform(X_std)

    fig, ax = plt.subplots(figsize=(6, 5))
    scat = ax.scatter([], [], c=[], cmap='viridis', s=60, alpha=0.7)

    def update(frame):
        ax.clear()
        ax.set_xlim(X_pca[:, 0].min(), X_pca[:, 0].max())
        ax.set_ylim(X_pca[:, 1].min(), X_pca[:, 1].max())
        ax.set_title(f"Stimuli PCA Evolution - Frame {frame}")
        ax.scatter(X_pca[:frame, 0], X_pca[:frame, 1], c=y[:frame], cmap='viridis', s=60, alpha=0.6)

    ani = animation.FuncAnimation(fig, update, frames=len(X_pca), interval=200)
    if save_path:
        ani.save(save_path, fps=5)
    plt.show()

# 3D view of expert/novice manifolds
plot_3d_aligned_manifold(exp_align, np.concatenate(subj_labels[EXPERT_SUBJECTS[0]]), 'Expert')
plot_3d_aligned_manifold(nov_align, np.concatenate(subj_labels[NONEXPERT_SUBJECTS[0]]), 'Novice')

# Dendrogram from one representative subject
plot_rdm_dendrogram(subj_data, subj_labels, EXPERT_SUBJECTS[0])

# Optional animation (saves .gif if path is given)
animate_pca_trajectory(subj_data, subj_labels, EXPERT_SUBJECTS[0], save_path=os.path.join(out_dir, "expert_pca.gif"))


metrics_df.to_csv(os.path.join(out_dir, 'group_metrics.csv'), index=False)  # Save metrics
logger.info("Saved metrics to CSV")  # Log completion
