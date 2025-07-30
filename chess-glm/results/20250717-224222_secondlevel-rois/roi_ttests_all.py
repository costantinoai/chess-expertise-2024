#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 17 17:53:35 2025

@author: costantino_ai
"""

import os
import numpy as np
import pandas as pd
from glob import glob
from scipy.stats import ttest_ind
from nilearn import image, plotting
from nilearn.image import load_img
from nilearn.maskers import NiftiLabelsMasker
from nilearn.datasets import fetch_surf_fsaverage
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from statsmodels.stats.multitest import fdrcorrection
import nibabel.freesurfer as fs
from scipy.ndimage import center_of_mass
from nilearn.image import coord_transform
from nilearn.datasets import fetch_atlas_harvard_oxford

# Plot styles
sns.set_style("white")
base_font_size = 22
plt.rcParams.update(
    {
        "font.family": "Ubuntu Condensed",
        "font.size": base_font_size,
        "axes.titlesize": base_font_size * 1.4,  # 36.4 ~ 36
        "axes.labelsize": base_font_size * 1.2,  # 31.2 ~ 31
        "xtick.labelsize": base_font_size,  # 26
        "ytick.labelsize": base_font_size,  # 26
        "legend.fontsize": base_font_size,  # 26
        "figure.figsize": (12, 9),  # wide figures
    }
)

REGIONS_LABELS = (
    (1, "Primary Visual Cortex"),
    (2, "Medial Superior Temporal Area"),
    (3, "Sixth Visual Area"),
    (4, "Second Visual Area"),
    (5, "Third Visual Area"),
    (6, "Fourth Visual Area"),
    (7, "Eighth Visual Area"),
    (8, "Primary Motor Cortex"),
    (9, "Primary Sensory Cortex"),
    (10, "Frontal Eye Fields"),
    (11, "Premotor Eye Field"),
    (12, "Area 55b"),
    (13, "Area V3A"),
    (14, "RetroSplenial Complex"),
    (15, "Parieto-Occipital Sulcus Area 2"),
    (16, "Seventh Visual Area"),
    (17, "IntraParietal Sulcus Area 1"),
    (18, "Fusiform Face Complex"),
    (19, "Area V3B"),
    (20, "Area Lateral Occipital 1"),
    (21, "Area Lateral Occipital 2"),
    (22, "Posterior InferoTemporal complex"),
    (23, "Middle Temporal Area"),
    (24, "Primary Auditory Cortex"),
    (25, "PeriSylvian Language Area"),
    (26, "Superior Frontal Language Area"),
    (27, "PreCuneus Visual Area"),
    (28, "Superior Temporal Visual Area"),
    (29, "Medial Area 7P"),
    (30, "Area 7m"),
    (31, "Parieto-Occipital Sulcus Area 1"),
    (32, "Area 23d"),
    (33, "Area ventral 23 a+b"),
    (34, "Area dorsal 23 a+b"),
    (35, "Area 31p ventral"),
    (36, "Area 5m"),
    (37, "Area 5m ventral"),
    (38, "Area 23c"),
    (39, "Area 5L"),
    (40, "Dorsal Area 24d"),
    (41, "Ventral Area 24d"),
    (42, "Lateral Area 7A"),
    (43, "Supplementary and Cingulate Eye Field"),
    (44, "Area 6m anterior"),
    (45, "Medial Area 7A"),
    (46, "Lateral Area 7P"),
    (47, "Area 7PC"),
    (48, "Area Lateral IntraParietal ventral"),
    (49, "Ventral IntraParietal Complex"),
    (50, "Medial IntraParietal Area"),
    (51, "Area 1"),
    (52, "Area 2"),
    (53, "Area 3a"),
    (54, "Dorsal area 6"),
    (55, "Area 6mp"),
    (56, "Ventral Area 6"),
    (57, "Area Posterior 24 prime"),
    (58, "Area 33 prime"),
    (59, "Anterior 24 prime"),
    (60, "Area p32 prime"),
    (61, "Area a24"),
    (62, "Area dorsal 32"),
    (63, "Area 8BM"),
    (64, "Area p32"),
    (65, "Area 10r"),
    (66, "Area 47m"),
    (67, "Area 8Av"),
    (68, "Area 8Ad"),
    (69, "Area 9 Middle"),
    (70, "Area 8B Lateral"),
    (71, "Area 9 Posterior"),
    (72, "Area 10d"),
    (73, "Area 8C"),
    (74, "Area 44"),
    (75, "Area 45"),
    (76, "Area 47l (47 lateral)"),
    (77, "Area anterior 47r"),
    (78, "Rostral Area 6"),
    (79, "Area IFJa"),
    (80, "Area IFJp"),
    (81, "Area IFSp"),
    (82, "Area IFSa"),
    (83, "Area posterior 9-46v"),
    (84, "Area 46"),
    (85, "Area anterior 9-46v"),
    (86, "Area 9-46d"),
    (87, "Area 9 anterior"),
    (88, "Area 10v"),
    (89, "Area anterior 10p"),
    (90, "Polar 10p"),
    (91, "Area 11l"),
    (92, "Area 13l"),
    (93, "Orbital Frontal Complex"),
    (94, "Area 47s"),
    (95, "Area Lateral IntraParietal dorsal"),
    (96, "Area 6 anterior"),
    (97, "Inferior 6-8 Transitional Area"),
    (98, "Superior 6-8 Transitional Area"),
    (99, "Area 43"),
    (100, "Area OP4/PV"),
    (101, "Area OP1/SII"),
    (102, "Area OP2-3/VS"),
    (103, "Area 52"),
    (104, "RetroInsular Cortex"),
    (105, "Area PFcm"),
    (106, "Posterior Insular Area 2"),
    (107, "Area TA2"),
    (108, "Frontal OPercular Area 4"),
    (109, "Middle Insular Area"),
    (110, "Pirform Cortex"),
    (111, "Anterior Ventral Insular Area"),
    (112, "Anterior Agranular Insula Complex"),
    (113, "Frontal OPercular Area 1"),
    (114, "Frontal OPercular Area 3"),
    (115, "Frontal OPercular Area 2"),
    (116, "Area PFt"),
    (117, "Anterior IntraParietal Area"),
    (118, "Entorhinal Cortex"),
    (119, "PreSubiculum"),
    (120, "Hippocampus"),
    (121, "ProStriate Area"),
    (122, "Perirhinal Ectorhinal Cortex"),
    (123, "Area STGa"),
    (124, "ParaBelt Complex"),
    (125, "Auditory 5 Complex"),
    (126, "ParaHippocampal Area 1"),
    (127, "ParaHippocampal Area 3"),
    (128, "Area STSd anterior"),
    (129, "Area STSd posterior"),
    (130, "Area STSv posterior"),
    (131, "Area TG dorsal"),
    (132, "Area TE1 anterior"),
    (133, "Area TE1 posterior"),
    (134, "Area TE2 anterior"),
    (135, "Area TF"),
    (136, "Area TE2 posterior"),
    (137, "Area PHT"),
    (138, "Area PH"),
    (139, "Area TemporoParietoOccipital Junction 1"),
    (140, "Area TemporoParietoOccipital Junction 2"),
    (141, "Area TemporoParietoOccipital Junction 3"),
    (142, "Dorsal Transitional Visual Area"),
    (143, "Area PGp"),
    (144, "Area IntraParietal 2"),
    (145, "Area IntraParietal 1"),
    (146, "Area IntraParietal 0"),
    (147, "Area PF opercular"),
    (148, "Area PF Complex"),
    (149, "Area PFm Complex"),
    (150, "Area PGi"),
    (151, "Area PGs"),
    (152, "Area V6A"),
    (153, "VentroMedial Visual Area 1"),
    (154, "VentroMedial Visual Area 3"),
    (155, "ParaHippocampal Area 2"),
    (156, "Area V4t"),
    (157, "Area FST"),
    (158, "Area V3CD"),
    (159, "Area Lateral Occipital 3"),
    (160, "VentroMedial Visual Area 2"),
    (161, "Area 31pd"),
    (162, "Area 31a"),
    (163, "Ventral Visual Complex"),
    (164, "Area 25"),
    (165, "Area s32"),
    (166, "posterior OFC Complex"),
    (167, "Area Posterior Insular 1"),
    (168, "Insular Granular Complex"),
    (169, "Area Frontal Opercular 5"),
    (170, "Area posterior 10p"),
    (171, "Area posterior 47r"),
    (172, "Area TG Ventral"),
    (173, "Medial Belt Complex"),
    (174, "Lateral Belt Complex"),
    (175, "Auditory 4 Complex"),
    (176, "Area STSv anterior"),
    (177, "Area TE1 Middle"),
    (178, "Para-Insular Area"),
    (179, "Area anterior 32 prime"),
    (180, "Area posterior 24"),
)

EXPERTS = [
    "03",
    "04",
    "06",
    "07",
    "08",
    "09",
    "10",
    "11",
    "12",
    "13",
    "16",
    "20",
    "22",
    "23",
    "24",
    "29",
    "30",
    "33",
    "34",
    "36",
]
NOVICES = [
    "01",
    "02",
    "15",
    "17",
    "18",
    "19",
    "21",
    "25",
    "26",
    "27",
    "28",
    "32",
    "35",
    "37",
    "39",
    "40",
    "41",
    "42",
    "43",
    "44",
]

from datetime import datetime
import shutil, inspect

def create_run_id() -> str:
    """
    Create a unique run ID based on the current timestamp.

    Returns:
        str: A string representing the current date and time in the format "YYYYMMDD-HHMMSS".
    """
    now = datetime.now()
    return now.strftime("%Y%m%d-%H%M%S")

def save_script_to_file(output_directory):
    """
    Save the calling script to a specified output directory.

    This function obtains the filename of the script that directly calls this function
    (i.e., the "caller frame") and copies that script to a target directory, providing
    reproducibility by capturing the exact code used in the analysis.

    Parameters
    ----------
    output_directory : str
        Path to the directory where the script file will be copied.

    Returns
    -------
    None
    """
    caller_frame = inspect.stack()[1]  # Stack frame of the caller
    script_file = caller_frame.filename
    script_file_out = os.path.join(output_directory, os.path.basename(script_file))
    shutil.copy(script_file, script_file_out)

OUTPATH = f"results/{create_run_id()}_secondlevel-rois"
os.makedirs(OUTPATH)
save_script_to_file(OUTPATH)

# Make custom colormap
def make_brain_cmap():
    center = plt.cm.RdPu(0)[:3]
    neg = np.linspace([0.0, 0.5, 0.7], center, 256)
    pos = plt.cm.RdPu(np.linspace(0, 1, 256))[:, :3]
    return LinearSegmentedColormap.from_list("custom_brain", np.vstack((neg, pos)))


BRAIN_CMAP = make_brain_cmap()


def get_region_label(region_number):
    region_dict = dict(REGIONS_LABELS)
    return region_dict.get(region_number, f"Unknown region ({region_number})")


# Star p-values
def stars(p):
    if p < 0.001:
        return "***"
    elif p < 0.01:
        return "**"
    elif p < 0.05:
        return "*"
    else:
        return ""


# Plot function for Glasser surface
def plot_sig_rois_on_glasser_surface(
    sig_df,
    hemisphere,
    glasser_annot_file,
    fsaverage,
    cmap="cold_hot",
    threshold=0,
    title="",
):
    """
    Plot Glasser parcellation with T_Diff values from sig_df on the specified hemisphere.

    Parameters
    ----------
    sig_df : pd.DataFrame
        Must contain 'ROI' (region name) and 'T_Diff' columns. The index must match region indices in .annot.
    hemisphere : str
        'left' or 'right'
    glasser_annot_file : str
        Path to FreeSurfer .annot file for the chosen hemisphere (e.g., 'lh.HCPMMP1.annot').
    fsaverage : dict
        Dict returned by nilearn.datasets.fetch_surf_fsaverage().
    cmap : str or matplotlib colormap
        Colormap to use.
    threshold : float or None
        Optional threshold to apply to values.
    """
    # Load annotation
    labels, ctab, region_names = fs.read_annot(glasser_annot_file)

    # Initialize array for texture values (NaNs everywhere)
    n_vertices = labels.shape[0]
    texture = np.full(n_vertices, 0.0)

    total_vertices_updated = 0

    for i, row in sig_df.iterrows():
        label_idx = int(row.ROI_idx)
        t_value = row["T_Diff"]

        mask = labels == label_idx
        n_vertices = mask.sum()

        texture[mask] = t_value
        total_vertices_updated += n_vertices

    # Pick correct mesh
    mesh = fsaverage["flat_left"] if hemisphere == "left" else fsaverage["flat_right"]
    bg_map = fsaverage["sulc_left"] if hemisphere == "left" else fsaverage["sulc_right"]

    # Plot
    plotting.plot_surf_stat_map(
        mesh,
        texture,
        hemi=hemisphere,
        bg_map=bg_map,
        cmap=cmap,
        threshold=threshold,
        colorbar=True,
        title=title,
        darkness=0.7,
        bg_on_data=True,
        engine="plotly",
        view="dorsal",
        output_file=f"{os.path.join(OUTPATH, title)}_surf.png"
    )

def paths_for(subject_ids, mode, data_dir, fname):
    paths = []
    for subject_id in subject_ids:
        subject_folder = os.path.join(data_dir, f"sub-{subject_id}")

        if mode == "rsa":
            # For RSA: filename is like sub-03_searchlight_checkmate.nii.gz
            filename = f"sub-{subject_id}_searchlight_{fname}"
        else:
            # For univariate: filename is just con_0001.nii
            filename = f"exp/{fname}"

        full_pattern = os.path.join(subject_folder, filename)
        matched_files = glob(full_pattern)

        if not matched_files:
            raise FileNotFoundError(f"No file found for pattern: {full_pattern}")

        paths.append(matched_files[0])

    return paths


def export_diff_stats_to_latex(df, label, output_path):
    df = df.copy()
    df = df[df["P_Diff"] < 0.05].sort_values("T_Diff", ascending=False)
    df["Star_Diff"] = df["P_Diff"].apply(stars)

    df = df[
        [
            "ROI",
            "CenterOfMass_Label",
            "Mean_Diff",
            "T_Diff",
            "CI_Low",
            "CI_High",
            "P_Diff",
            "Star_Diff",
        ]
    ]

    column_format = "llrrr@{}r@{}rl"
    headers = [
        "ROI",
        "Harvard-Oxford Label",
        "$M_{\\text{diff}}$",
        "$t$",
        "\\multicolumn{2}{c}{95\\% CI}",
        "$p$",
        "Sig.",
    ]
    subheaders = ["", "", "", "", "LL", "UL", "", ""]

    # Create LaTeX header manually
    header_row = " & ".join(headers) + " \\\\"
    subheader_row = " & ".join(subheaders) + " \\\\"
    midrule = "\\cmidrule(lr){5-6}"

    # Convert dataframe to LaTeX (no header)
    body = df.to_latex(
        index=False,
        escape=False,
        float_format="%.3f",
        header=False,
        column_format=column_format,
    )

    # Assemble table
    table = f"""\\begin{{table}}[ht]
\\centering
\\caption{{{label}: Experts > Novices}}
\\label{{tab:{label.replace(' ', '_').lower()}}}
\\resizebox{{\\linewidth}}{{!}}{{%
\\begin{{tabular}}{{{column_format}}}
\\toprule
{header_row}
{subheader_row}
{midrule}
{body.strip()}
\\bottomrule
\\end{{tabular}}
}}
\\end{{table}}"""

    with open(output_path, "w") as f:
        f.write(table)

    print(table)
    print(f"LaTeX table saved to: {output_path}")


def compute_confidence_intervals(X1, X2):
    """
    Compute mean difference, t-value, p-value, and 95% CI for each ROI.
    """
    means = X1.mean(0) - X2.mean(0)
    t_results = ttest_ind(X1, X2, axis=0)
    cis = t_results.confidence_interval()
    return means, t_results.statistic, t_results.pvalue, cis


def get_label_at_com(
    roi_val, glasser_data, affine_glasser, atlas_data_ho, affine_ho, atlas_labels_ho
):
    try:
        # Step 1: Create binary mask for the target ROI in the Glasser atlas
        roi_mask = glasser_data == roi_val
        if not np.any(roi_mask):
            return "No ROI voxels"

        # Step 2: Get voxel coordinates for this ROI
        coords = np.argwhere(roi_mask)

        # Step 3: Convert to MNI space using Glasser affine
        mni_coords = np.array(
            [coord_transform(x, y, z, affine_glasser) for x, y, z in coords]
        )

        # Step 4: Keep only voxels in the left hemisphere (MNI x < 0)
        left_coords = coords[mni_coords[:, 0] < 0]
        if len(left_coords) == 0:
            return "No LH voxels"

        # Step 5: Create a binary mask limited to LH voxels
        lh_mask = np.zeros_like(glasser_data, dtype=bool)
        lh_mask[tuple(left_coords.T)] = True

        # Step 6: Compute center of mass in voxel space
        com_voxel = center_of_mass(lh_mask)
        if np.any(np.isnan(com_voxel)):
            return "Invalid CoM"

        # Step 7: Convert center of mass to MNI space
        com_mni = coord_transform(*com_voxel, affine_glasser)

        # Step 8: Convert MNI coordinate to voxel space of HO atlas
        inv_affine_ho = np.linalg.inv(affine_ho)
        com_voxel_ho = coord_transform(*com_mni, inv_affine_ho)
        x, y, z = [int(round(v)) for v in com_voxel_ho]

        # Step 9: Ensure voxel indices are within bounds
        x = np.clip(x, 0, atlas_data_ho.shape[0] - 1)
        y = np.clip(y, 0, atlas_data_ho.shape[1] - 1)
        z = np.clip(z, 0, atlas_data_ho.shape[2] - 1)

        # Step 10: Lookup the HO atlas label at the voxel
        label_idx = int(atlas_data_ho[x, y, z])
        if label_idx == 0:
            return "Unlabeled"
        return (
            atlas_labels_ho[label_idx]
            if label_idx < len(atlas_labels_ho)
            else f"Unknown label {label_idx}"
        )

    except Exception as e:
        return f"Error: {str(e)}"


def plot_glass_brain_from_df(
    df_input, atlas_img, title, cmap=BRAIN_CMAP, base_font_size=22
):
    """
    Create and display a glass brain plot from a DataFrame with ROI and T_Diff values.

    Parameters
    ----------
    df_input : pd.DataFrame
        Must contain 'ROI' and 'T_Diff' columns.
    atlas_img : nibabel.Nifti1Image
        Atlas image where voxel values match ROI labels.
    title : str
        Title for the plot.
    cmap : matplotlib colormap, optional
        Colormap to use (default is BRAIN_CMAP).
    threshold : float, optional
        Optional value threshold to mask low effects (default is 0).
    base_font_size : int, optional
        Font size for the title (default is 22).
    """
    atlas_arr = atlas_img.get_fdata()
    diff_map = np.zeros_like(atlas_arr)

    for _, row in df_input.iterrows():
        roi_label = row["ROI_idx"]
        t_value = row["T_Diff"]
        diff_map[atlas_arr == roi_label] = t_value

    sig_diff_img = image.new_img_like(atlas_img, diff_map)

    plotting.plot_glass_brain(
        sig_diff_img,
        title=title,
        cmap=cmap,
        colorbar=True,
        symmetric_cbar=True,
        plot_abs=False,
        output_file=f"{os.path.join(OUTPATH, title)}_glass.png"
    )

# === Main Analysis ===
def run_analysis(DATA_DIR, CONTRASTS, atlas_path, mode):
    # Load atlas image and get ROI labels
    atlas_img = load_img(atlas_path)
    roi_labels = np.unique(atlas_img.get_fdata())[1:]

    # Fetch fsaverage surface template (used for surface plotting)
    fsav = fetch_surf_fsaverage("fsaverage")
    annot_file = "/data/projects/chess/data/BIDS/derivatives/fastsurfer/fsaverage/label/lh.HCPMMP1.annot"

    # Initialize masker for extracting ROI means
    masker = NiftiLabelsMasker(
        labels_img=atlas_img, standardize=False, strategy="mean", verbose=0
    )

    # Load Harvard-Oxford atlas for anatomical labeling
    atlas_ho = fetch_atlas_harvard_oxford("cort-maxprob-thr25-2mm")
    atlas_img_ho = load_img(atlas_ho.maps)
    atlas_data_ho = atlas_img_ho.get_fdata()
    affine_ho = atlas_img_ho.affine
    atlas_labels_ho = atlas_ho.labels

    # Cache Glasser data and affine for coordinate transforms
    glasser_data = atlas_img.get_fdata()
    affine_glasser = atlas_img.affine

    for fname, label in CONTRASTS.items():
        print(f"Processing: {label}")

        # Load subject data for each group
        exp_paths = paths_for(EXPERTS, mode, DATA_DIR, fname)
        nov_paths = paths_for(NOVICES, mode, DATA_DIR, fname)

        # Apply masker across subjects
        X_exp = np.squeeze([masker.fit_transform(p) for p in exp_paths])
        X_nov = np.squeeze([masker.fit_transform(p) for p in nov_paths])

        # Compute stats and confidence intervals for group difference
        mean_diff, t_vals, p_vals, cis = compute_confidence_intervals(X_exp, X_nov)
        p_vals_fdr = fdrcorrection(p_vals)[1]

        # Organize results in dataframe
        df = pd.DataFrame({
            "ROI_idx": roi_labels,
            "Mean_Diff": mean_diff,
            "T_Diff": t_vals,
            "P_Diff": p_vals_fdr,
            "CI_Low": cis[0],
            "CI_High": cis[1],
        })

        # Add ROI and anatomical labels
        df["ROI"] = [get_region_label(r) for r in df["ROI_idx"]]
        df["CenterOfMass_Label"] = [
            get_label_at_com(
                r, glasser_data, affine_glasser, atlas_data_ho, affine_ho, atlas_labels_ho
            )
            for r in df["ROI_idx"]
        ]

        # Plot all ROIs (left hemisphere surface and glass brain)
        plot_sig_rois_on_glasser_surface(
            df,
            hemisphere="left",
            glasser_annot_file=annot_file,
            fsaverage=fsav,
            cmap=BRAIN_CMAP,
            title=f"{label} Exp vs Nov",
        )

        # Plot FDR-significant ROIs only
        df_sig = df[df["P_Diff"] < 0.05]
        plot_sig_rois_on_glasser_surface(
            df_sig,
            hemisphere="left",
            glasser_annot_file=annot_file,
            fsaverage=fsav,
            cmap=BRAIN_CMAP,
            title=f"{label} Exp vs Nov (FDR < .05)",
        )

        # Glass brain projection for all ROIs and significant subset
        plot_glass_brain_from_df(
            df_input=df, atlas_img=atlas_img, title=f"{label} Exp vs Nov (all ROIs)"
        )
        plot_glass_brain_from_df(
            df_input=df_sig, atlas_img=atlas_img, title=f"{label} Exp vs Nov (FDR < .05)"
        )

        # Export APA-style LaTeX table with CI and stars
        latex_path = f"{OUTPATH}/{mode}_{fname}_diff_only_table.tex"
        export_diff_stats_to_latex(df, label, latex_path)

# Paths
ATLAS_PATH = "/data/projects/chess/data/misc/templates/tpl-MNI152NLin2009cAsym_res-02_atlas-Glasser2016_desc-bilateral_resampled.nii"

# Run univariate
UNIV_DATA_DIR = "/data/projects/chess/data/BIDS/derivatives/fmriprep-SPM_smoothed-4_GS-FD-HMP_brainmasked/MNI/fmriprep-SPM-MNI/GLM"
UNIV_CONTRASTS = {
    "con_0001.nii": "Checkmate > Non-checkmate",
    "con_0002.nii": "All > Rest",
}
run_analysis(UNIV_DATA_DIR, UNIV_CONTRASTS, ATLAS_PATH, mode="univ")

# Run RSA
RSA_DATA_DIR = "/data/projects/chess/data/BIDS/derivatives/rsa_searchlight"
RSA_CONTRASTS = {
    "checkmate.nii.gz": "Checkmate",
    "strategy.nii.gz": "Strategy",
    "visualSimilarity.nii.gz": "Visual Similarity",
}
run_analysis(RSA_DATA_DIR, RSA_CONTRASTS, ATLAS_PATH, mode="rsa")
