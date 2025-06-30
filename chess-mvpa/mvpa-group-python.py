import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from nilearn import image, plotting
from nilearn.glm.second_level import SecondLevelModel
from nilearn.datasets import fetch_atlas_harvard_oxford
from nilearn.glm.thresholding import fdr_threshold
from nilearn.reporting import get_clusters_table

# ---------------------- CUSTOM VISUALS ----------------------
plt.rcParams['font.family'] = 'Ubuntu Condensed'

# Custom brain colormap
def make_brain_cmap():
    center = plt.cm.RdPu(0)[:3]
    neg = np.linspace([0.0, 0.5, 0.7], center, 256)
    pos = plt.cm.RdPu(np.linspace(0, 1, 256))[:, :3]
    return LinearSegmentedColormap.from_list('custom_brain', np.vstack((neg, pos)))

BRAIN_CMAP = make_brain_cmap()

# ---------------------- CONFIGURATION ----------------------
EXPERT_SUBJECTS = [
    "03", "04", "06", "07", "08", "09", "10", "11", "12", "13",
    "16", "20", "22", "23", "24", "29", "30", "33", "34", "36"
]
NONEXPERT_SUBJECTS = [
    "01", "02", "15", "17", "18", "19", "21", "25", "26", "27",
    "28", "32", "35", "37", "39", "40", "41", "42", "43", "44"
]
DATA_DIR = '/home/eik-tb/Desktop/mvpa_searchlight'
RESULTS_DIR = os.path.join(DATA_DIR, 'results-mvpa_searchlight')
os.makedirs(RESULTS_DIR, exist_ok=True)

# ---------------------- DATA LOADING ----------------------
def find_nifti_files(data_dir, pattern='searchlight_checkmate'):
    nii_files = []
    for root, dirs, files in os.walk(data_dir):
        for file in files:
            if file.endswith('.nii.gz') and 'sub-' in file and pattern in file:
                nii_files.append(os.path.join(root, file))
    nii_files.sort()
    return nii_files


def split_by_group(nii_files, expert_ids, nonexpert_ids):
    expert_files = [f for f in nii_files if any(f'sub-{sid}' in f for sid in expert_ids)]
    nonexpert_files = [f for f in nii_files if any(f'sub-{sid}' in f for sid in nonexpert_ids)]
    return expert_files, nonexpert_files

nii_files = find_nifti_files(DATA_DIR)
expert_files, nonexpert_files = split_by_group(nii_files, EXPERT_SUBJECTS, NONEXPERT_SUBJECTS)

# ---------------------- THRESHOLDING ----------------------
ALPHA = 0.001    # significance level
APPLY_FDR = False  # whether to apply FDR correction per tail


def compute_thresholds(z_map, alpha=ALPHA):
    """
    Compute thresholds for positive and negative tails independently using FDR or uncorrected alpha.
    Returns (neg_thr, pos_thr).
    """
    data = z_map.get_fdata()
    # Positive tail
    pos_vals = data[data > 0]
    # Negative tail (flip sign)
    neg_vals = -data[data < 0]

    if APPLY_FDR:
        pos_thr = fdr_threshold(pos_vals, alpha)
        neg_thr = fdr_threshold(neg_vals, alpha)
    else:
        from scipy.stats import norm
        pos_thr = norm.isf(alpha)
        neg_thr = pos_thr
    return neg_thr, pos_thr

# ---------------------- MODEL FITTING ----------------------
all_files = expert_files + nonexpert_files
group_labels = [1]*len(expert_files) + [-1]*len(nonexpert_files)
design_matrix = pd.DataFrame({'group': group_labels})

# Plot design matrix
plotting.plot_design_matrix(design_matrix)

second_level_model = SecondLevelModel(n_jobs=-1, smoothing_fwhm=3.0)
second_level_model = second_level_model.fit(all_files, design_matrix=design_matrix)

# Between-group contrast (experts > non-experts)
contrast = second_level_model.compute_contrast('group', output_type='z_score')
neg_thr, pos_thr = compute_thresholds(contrast)

# Plot
plotting.plot_glass_brain(
    contrast,
    display_mode='lyrz',
    colorbar=True,
    cmap=BRAIN_CMAP,
    symmetric_cbar=True,
    threshold=pos_thr,
    title=f'Experts > Non-Experts (z > {pos_thr:.2f})'
)


# ---------------------- WITHIN-GROUP ANALYSIS ----------------------
def fit_and_plot_group(files, label):
    model = SecondLevelModel(n_jobs=-1, smoothing_fwhm=3.0)
    design = pd.DataFrame({label: [1]*len(files)})
    model = model.fit(files, design_matrix=design)
    z_map = model.compute_contrast(label, output_type='z_score')
    neg_thr, pos_thr = compute_thresholds(z_map)

    plotting.plot_glass_brain(
        z_map,
        display_mode='lyrz',
        colorbar=True,
        cmap=BRAIN_CMAP,
        symmetric_cbar=True,
        threshold=pos_thr,
        title=f'{label.capitalize()} (z > {pos_thr:.2f})'
    )

    return z_map

z_expert = fit_and_plot_group(expert_files, 'expert')
z_nonexpert = fit_and_plot_group(nonexpert_files, 'nonexpert')


import nibabel as nib

# ---------------------- CLUSTER REPORTING ----------------------
atlas = fetch_atlas_harvard_oxford('cort-maxprob-thr25-2mm')
atlas_img = atlas.maps
atlas_resampled = image.resample_to_img(atlas_img, contrast, interpolation='nearest')

atlas_data = atlas_resampled.get_fdata()
atlas_labels = atlas.labels


# --- Extract clusters & annotate ---
clusters = get_clusters_table(
    contrast,
    stat_threshold=pos_thr,
    cluster_threshold=10,
    two_sided=True,
    return_label_maps=False
)

# Convert to DataFrame
df_clust = clusters.copy()
# Anatomical label lookup
labels = []
for x, y, z in zip(df_clust['X'], df_clust['Y'], df_clust['Z']):
    ijk = np.round(nib.affines.apply_affine(
        np.linalg.inv(atlas_resampled.affine), [x, y, z]
    )).astype(int)
    try:
        idx = atlas_data[tuple(ijk)]
        labels.append(atlas_labels[int(idx)])
    except Exception:
        labels.append('Unknown')


df_clust['Anat_Label'] = labels
print(df_clust)

# # Generate tables for each map
# for name, z_map in [('expert_vs_nonexpert', contrast),
#                     ('expert_within', z_expert),
#                     ('nonexpert_within', z_nonexpert)]:
#     neg_thr, pos_thr = compute_thresholds(z_map)
    # for thr, tail in [(pos_thr, 'pos'), (-neg_thr, 'neg')]:
    #     clusters = get_clusters_table(z_map, atlas_resampled, stat_threshold=thr if thr != np.inf else 0)
    #     print(clusters)
        # clusters.to_csv(os.path.join(RESULTS_DIR, f'{name}_{tail}_clusters.csv'), index=False)

print('Analysis complete. Results saved to', RESULTS_DIR)
