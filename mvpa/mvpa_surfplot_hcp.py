#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 18 21:50:52 2025

@author: costantino_ai

This script loads group-level T-test (or correlation) results and ROI annotations,
then creates bar plots for each requested contrast and regressor. It displays the mean difference
(or correlation coefficient), confidence intervals, and significance markers (FDR-corrected)
for each ROI.

Overall Workflow:
1) Define input directories of MVPA results and ROI annotation files.
2) Load the pickled group-level statistics (analysis_results).
3) For each requested contrast and regressor, prepare a sorted data structure.
4) Create bar plots showing mean differences (or coefficients), confidence intervals, and significance.
5) Save each bar plot to disk, separated by contrast type and analysis method.
"""

import os
import math
import glob
import numpy as np
import pickle
import pandas as pd
import matplotlib.pyplot as plt
from nilearn.datasets import load_fsaverage, load_fsaverage_data
import matplotlib.colors as mcolors
import matplotlib.colorbar as cbar
from typing import Tuple
from nilearn.plotting import plot_surf_stat_map, plot_surf_contours

from modules.helpers import create_run_id
from modules import (
    MANAGER,
    CORTICES_GROUPS_CMAPS,
    ROIManager,
    HCPMMP1_LH_LABELS,
    HCPMMP1_LH_NAMES_STR
)

regressor_mapping = {
    "checkmate": "Checkmate vs. Non-checkmate boards",
    "stimuli_half": "Pairwise checkmate boards",
    "stimuli": "Pairwise all boards",
    "motif_half": "Motifs (Checkmate boards only)",
    "check_n_half": "Number of moves to checkmate (Checkmate boards only)",
    "side_half": "King position (L/R, Checkmate boards only)",
    "categories_half": "Strategy (Checkmate boards only)",
    "categories": "Strategy (all stimuli)",
    "visualStimuli": "Visually similar pairs",
    "total_pieces_half": "Total number of pieces (Checkmate boards only)",
    "legal_moves_half": "Number of available legal moves (Checkmate boards only)",
    "total_pieces": "Total number of pieces",
    "legal_moves": "Number of available legal moves",
    "difficulty_half": "Board difficulty (Checkmate boards only)",
    "first_piece_half": "First piece to move (Checkmate boards only)",
    "checkmate_piece_half": "Checkmate piece (Checkmate boards only)",
}


def load_glasser_surf():
    from nilearn.datasets import load_fsaverage
    from nilearn.surface import SurfaceImage
    import matplotlib.colors as mcolors

    def load_freesurfer_lut(file_path):
        """Load a FreeSurfer-style LUT file and create a Matplotlib colormap."""
        lut_data = []
        label_to_color = {}

        with open(file_path, "r") as f:
            for line in f:
                if line.strip() and not line.startswith(
                    "#"
                ):  # Ignore comments and empty lines
                    parts = line.split()
                    if len(parts) >= 6:  # Expected format: index, label, R, G, B, A
                        _ = int(parts[0])
                        label = parts[1]
                        r, g, b, a = map(int, parts[2:6])
                        color = (r / 255.0, g / 255.0, b / 255.0, a / 255.0)
                        lut_data.append(color)
                        label_to_color[label] = color

        cmap = mcolors.ListedColormap(lut_data)
        return cmap, label_to_color

    data = {
        "left": "/data/projects/chess/data/BIDS/derivatives/fastsurfer/fsaverage/label/lh.HCPMMP1.annot",
        "right": "/data/projects/chess/data/BIDS/derivatives/fastsurfer/fsaverage/label/rh.HCPMMP1.annot",
    }

    cmaps = {
        "left": load_freesurfer_lut(
            "/data/projects/chess/data/misc/lh_HCPMMP1_color_table.txt"
        )[0],
        "right": load_freesurfer_lut(
            "/data/projects/chess/data/misc/rh_HCPMMP1_color_table.txt"
        )[0],
    }

    fsaverage = load_fsaverage("fsaverage")

    glasser_atlas = SurfaceImage(
        mesh=fsaverage["flat"],
        data=data,
    )

    return glasser_atlas, cmaps


def format_contrast(s):
    """
    Convert underscore ("_") to space and "vs" to hyphen ("-"), then capitalize.
    Example: "experts_vs_nonexperts" -> "Experts - Nonexperts".
    """
    s = s.replace("_", " ")  # Replace underscores with spaces
    s = s.replace("vs", "-")  # Replace "vs" with "-"
    return " ".join(word.capitalize() for word in s.split())  # Capitalize each word

def build_significance_overlay(
    stats_df: pd.DataFrame,
    manager: ROIManager,
) -> Tuple[Tuple[np.ndarray, np.ndarray], float, float]:
    """
    Constructs overlay maps for the left and right hemispheres based on statistical values.

    For each region of interest (ROI), the function assigns the mean value from the provided
    `stats_df` to the overlay map if the ROI is significant. If no ROIs are found for a specific
    region, an error is raised.

    Parameters
    ----------
    stats_df : pd.DataFrame
        DataFrame containing statistics for each ROI. It must have ROI names as the index and
        include a column "mean-chance" with mean values for each ROI.

    manager : ROIManager
        Object managing ROI information, including mapping region names to region IDs.

    lh_labels : np.ndarray
        Array of labels corresponding to the left hemisphere ROIs.

    rh_labels : np.ndarray
        Array of labels corresponding to the right hemisphere ROIs.

    Returns
    -------
    overlays : Tuple[np.ndarray, np.ndarray]
        Tuple containing overlay arrays for the left and right hemispheres.

    Raises
    ------
    ValueError
        If no ROIs are found for a specified region in the input data.
    """
    # Determine the hemisphere (left or right) and hierarchical level (ROI, cortex, lobe)
    hemisphere = "l"
    hierarchical_level = "cortex" if max(stats_df.index) < 22 else "ROI"

    # Initialize overlays for the left and right hemispheres with NaNs
    overlay = np.full_like(HCPMMP1_LH_LABELS, np.nan, dtype=float)

    # Extract significant ROI names from the DataFrame index
    significant_rois_df = stats_df[stats_df["fdr_reject"] == True]

    for idx, significant_roi_row in significant_rois_df.iterrows():

        significant_roi_name_clean = significant_roi_row["roi"]

        # Extract the mean value for the current ROI from the DataFrame
        mean_value = significant_roi_row["mean_diff"]

        if hierarchical_level == "cortex":
            # Get all ROIs under the cortex name
            selected_rois, _ = manager.get_by_filter(
                hemisphere=hemisphere,
                cortex=significant_roi_name_clean,
            )

            if len(selected_rois) == 0:
                raise ValueError(f"No ROIs found for significant cortex: {significant_roi_name_clean}")

            unique_cortices = np.unique([roi.cortex for roi in selected_rois])
            assert len(unique_cortices) == 1

            for roi in selected_rois:
                region_label = HCPMMP1_LH_NAMES_STR.index(roi.region_name.upper())
                overlay[HCPMMP1_LH_LABELS == region_label] = mean_value

        elif hierarchical_level == "ROI":
            region_name = f"L_{significant_roi_name_clean}_ROI".upper()

            try:
                region_label = HCPMMP1_LH_NAMES_STR.index(region_name)
            except ValueError:
                raise ValueError(f"ROI not found: {region_name}")

            overlay[HCPMMP1_LH_LABELS == region_label] = mean_value

        else:
            raise ValueError(f"Unsupported hierarchical level: {hierarchical_level}")

    # Return the overlays for both hemispheres
    return overlay


regressor_mapping = {
    "checkmate": "Checkmate vs. Non-checkmate boards",
    "stimuli_half": "Pairwise checkmate boards",
    "stimuli": "Pairwise all boards",
    "motif_half": "Motifs (Checkmate boards only)",
    "check_n_half": "Number of moves to checkmate (Checkmate boards only)",
    "side_half": "King position (L/R, Checkmate boards only)",
    "categories_half": "Strategy (Checkmate boards only)",
    "categories": "Strategy (all stimuli)",
    "visualStimuli": "Visually similar pairs",
    "total_pieces_half": "Total number of pieces (Checkmate boards only)",
    "legal_moves_half": "Number of available legal moves (Checkmate boards only)",
    "total_pieces": "Total number of pieces",
    "legal_moves": "Number of available legal moves",
    "difficulty_half": "Board difficulty (Checkmate boards only)",
    "first_piece_half": "First piece to move (Checkmate boards only)",
    "checkmate_piece_half": "Checkmate piece (Checkmate boards only)",
}

# ----------------------------------------------------------------------------
# List of tuples where each tuple has:
#  (1) The path to the pickled group-level results,
#  (2) The path to the corresponding ROI .tsv annotation file
# ----------------------------------------------------------------------------
MVPA_RESULTS_PATHS = [
    # (
    #     "/data/projects/chess/data/BIDS/derivatives/mvpa/20250402-230003_glasser_regions_bilateral",
    #     "/home/eik-tb/OneDrive_andreaivan.costantino@kuleuven.be/GitHub/chess-expertise-2024/chess-rois/results/glasser_regions_bilateral",
    # ),
    (
        "/data/projects/chess/data/BIDS/derivatives/mvpa/20250402-191833_glasser_cortices_bilateral",
        "/home/eik-tb/OneDrive_andreaivan.costantino@kuleuven.be/GitHub/chess-expertise-2024/chess-rois/results/glasser_cortex_bilateral",
    ),
]

# You can optionally specify which regressors you want to keep; if empty, all are used
regressors_to_keep = ["checkmate", "categories", "visualStimuli"]

# Contrasts you want to plot
# contrasts = ["experts_vs_nonexperts",  "experts_vs_chance", "nonexperts_vs_chance"]
contrasts = ["experts_vs_nonexperts"]

# Analyses (e.g., "svm" or "rsa_corr")
# analyses = ["svm", "rsa_corr"]
analyses = ["rsa_corr"]

# Select the colormap depending on sign of data range
cmap_used = "RdPu"

# MAIN SCRIPT
for analysis in analyses:
    # -------------------------------------------------------------------------
    # Loop over each pair of directories: one with group results, one with ROI labels
    # -------------------------------------------------------------------------
    for analysis_results_path, roi_annotation_path in MVPA_RESULTS_PATHS:

        # ---------------------------------------------------------------------
        # Find the pickle file in the group results directory
        # ---------------------------------------------------------------------
        pkl_files = glob.glob(
            os.path.join(analysis_results_path, analysis, "*group/*.pkl")
        )

        if len(pkl_files) > 1:
            raise ValueError(
                "More than one file found in the group directory. Please check."
            )
        elif len(pkl_files) == 0:
            raise ValueError("No pickle file found in the group directory.")
        else:
            # We have exactly one pickle file
            analysis_results_pickle = pkl_files[0]
            group_path = os.path.dirname(analysis_results_pickle)

        # ---------------------------------------------------------------------
        # Find the .tsv ROI annotation file in roi_annotation_path
        # ---------------------------------------------------------------------
        roi_annotation_files = glob.glob(os.path.join(roi_annotation_path, "*.tsv"))

        if len(roi_annotation_files) > 1:
            raise ValueError(
                "More than one .tsv file found in the ROI annotation directory."
            )
        elif len(roi_annotation_files) == 0:
            raise ValueError(
                "No .tsv annotation file found in the ROI annotation directory."
            )
        else:
            # We have exactly one annotation file
            roi_annotation_file = roi_annotation_files[0]

        # ---------------------------------------------------------------------
        # Load the ROI annotation data
        # ---------------------------------------------------------------------
        roi_df = pd.read_csv(roi_annotation_file, sep="\t")

        # Determine the measure string for labeling
        measure_string = "Decoding Accuracy" if analysis == "svm" else "Coefficient"

        # ---------------------------------------------------------------------
        # Load the saved analysis results (pickled dictionary)
        # ---------------------------------------------------------------------
        with open(analysis_results_pickle, "rb") as f:
            analysis_results = pickle.load(f)

        # ---------------------------------------------------------------------
        # For each requested contrast, we create plots and save them
        # ---------------------------------------------------------------------
        for contrast in contrasts:

            # Create a subfolder for each contrast
            contrast_dir = os.path.join(group_path, contrast)
            os.makedirs(contrast_dir, exist_ok=True)

            # Create a subfolder for bar plots, each run is uniquely identified
            barplots_directory = os.path.join(contrast_dir, f"{create_run_id()}_surfplots")
            os.makedirs(barplots_directory, exist_ok=False)

            # Nice formatting of the contrast for titles
            formatted_contrast = format_contrast(contrast)

            # -----------------------------------------------------------------
            # Extract only the data for this particular contrast
            # -----------------------------------------------------------------
            # We want to filter `analysis_results` such that we only keep items where
            # the key is the same as the contrast (e.g., "experts_vs_nonexperts").
            # This yields a dictionary of regressors -> ROI stats.
            contrast_results_dict = {
                regressor: rois
                for comp_key, reg_dict in analysis_results.items()
                if comp_key == contrast  # match the requested contrast key
                for regressor, rois in reg_dict.items()
            }

            # -----------------------------------------------------------------
            # Determine which regressors to process. If `regressors_to_keep` is non-empty,
            # we only keep those. Otherwise, we take all regressors.
            # -----------------------------------------------------------------
            all_regressors = contrast_results_dict.keys()
            if len(regressors_to_keep) > 0:
                regressors = [r for r in all_regressors if r in regressors_to_keep]
            else:
                regressors = list(all_regressors)

            # -----------------------------------------------------------------
            # Compute global y-axis limits across all regressors for consistency
            # -----------------------------------------------------------------
            mins = []
            maxs = []

            for regressor in regressors:
                single_regressor_data = contrast_results_dict[regressor]
                df = pd.DataFrame.from_dict(single_regressor_data, orient="index")
                df = df[df["fdr_reject"] == True]

                min_val = (df["mean_diff"]).min()
                max_val = (df["mean_diff"]).max()

                # Round to nearest decimal for aesthetics
                # e.g., floor to 1 decimal place for the lower limit, ceil for the upper limit
                vmin = 0 if np.isnan(min_val) or np.isinf(min_val) else math.floor(min_val * 1000) / 1000
                vmax = 0 if np.isnan(max_val) or np.isinf(max_val) else math.ceil(max_val * 1000) / 1000

                mins.append(vmin)
                maxs.append(vmax)

            vmin = np.min(mins)
            vmax = np.max(maxs)
            # vmin = 0
            # vmax = 0.27 if "chance" in contrast else .10

            # -----------------------------------------------------------------
            # Now generate bar plots for each regressor
            # -----------------------------------------------------------------
            for regressor in regressors:
                single_regressor_data = contrast_results_dict[regressor]

                # We'll use this for the plot title
                title = (
                    f"{formatted_contrast} | {measure_string} difference | "
                    f"{regressor.replace('_', ' ').capitalize()}"
                )

                # -------------------------------------------------------------
                # Convert the single_regressor_data dictionary into a DataFrame
                # -------------------------------------------------------------
                df = pd.DataFrame.from_dict(single_regressor_data, orient="index")
                df = df.reset_index().rename(columns={"index": "roi"})

                # Extract the confidence interval bounds
                df["ci_low"] = df["CI95"].apply(lambda x: x[0])
                df["ci_high"] = df["CI95"].apply(lambda x: x[1])

                # Create a mapping from ROI names to colors
                roi_color_map = dict(zip(roi_df["region_name"], roi_df["color"]))

                # Create an ordering map for the ROIs if available
                if "order" in roi_df.columns and not roi_df["order"].isna().all():
                    roi_order_map = dict(zip(roi_df["region_name"], roi_df["order"]))
                else:
                    roi_order_map = dict(zip(roi_df["region_name"], roi_df["region_id"]))

                # Merge color and order information into the DataFrame
                df["color"] = df["roi"].map(roi_color_map)
                df["order"] = df["roi"].map(roi_order_map)
                df["roi"] = df["roi"].str.replace("_", " ")

                # Sort by 'order' to ensure consistent ROI ordering
                df = df.sort_values("order")

                # Extract the ordered ROI names
                ordered_rois = list(df["roi"].values)

                # Convert the ROI column to a categorical to preserve sorting
                df["roi"] = pd.Categorical(
                    df["roi"], categories=ordered_rois, ordered=True
                )
                df.sort_values("roi", inplace=True)

                ######

                # -- fsaverage surfaces (only need left hemisphere if we're plotting left) --
                fsaverage_meshes = load_fsaverage("fsaverage")
                surf_inflated = fsaverage_meshes["pial"]  # (coords, faces)
                surf_flat = fsaverage_meshes["flat"]  # (coords, faces)

                # Just ensure glasser_surf is a 1D array (n_vertices,).
                glasser_surf, glasser_cmaps = load_glasser_surf()
                colors = [
                    next(color for group, color in CORTICES_GROUPS_CMAPS if group == r.cortex_group)
                    for r in MANAGER.rois
                    if r.hemisphere == "L"
                ]
                labels = [r.region_long_name for r in MANAGER.rois if r.hemisphere == "L"]

                # Create an overaly containing only significant ROIs
                overlay = build_significance_overlay(df, MANAGER)

                # Create figure with custom size
                fig = plt.figure(figsize=(10, 8))

                # Create a gridspec with 2 rows and 3 columns
                # Adjust height_ratios so the bottom row is larger
                # wspace/hspace reduce white space horizontally/vertically
                gs = fig.add_gridspec(
                    nrows=2,
                    ncols=3,
                    height_ratios=[1.0, 1.5],  # top row : bottom row
                    wspace=0.05,  # reduce space between columns
                    hspace=0.03,  # reduce space between rows
                )

                # Top row (3 subplots: lateral, medial, ventral)
                ax_lateral = fig.add_subplot(gs[0, 0], projection="3d")
                ax_medial = fig.add_subplot(gs[0, 1], projection="3d")
                ax_ventral = fig.add_subplot(gs[0, 2], projection="3d")

                # Bottom row (spans all 3 columns)
                ax_dorsal = fig.add_subplot(gs[1, :], projection="3d")

                # Optionally adjust margins around the whole figure
                plt.subplots_adjust(
                    left=0.05,  # space on the left
                    right=0.88,  # space on the right
                    top=0.93,  # space at the top
                    bottom=0.04,  # space at the bottom
                )

                bg_map = load_fsaverage_data(mesh="fsaverage", data_type="curvature")

                # ------------------------------
                # Create a colorbar to the right
                # ------------------------------
                # ------------------------------

                cbar_label = "Decoding Accuracy Δ"

                cax = fig.add_axes([0.95, 0.25, 0.01, 0.25])  # [left, bottom, width, height]

                # Create a normalizer with vmin and vmax
                norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

                # Create colorbar
                cb = cbar.ColorbarBase(cax, cmap=cmap_used, norm=norm, orientation="vertical")

                # Set ticks: only vmin, 0, vmax
                cb.set_ticks([vmin, 0, vmax])
                cb.set_ticklabels([f"{vmin:.2f}", "0", f"{vmax:.2f}"])

                # Adjust tick label size
                cb.ax.tick_params(labelsize=14)

                # Add colorbar label
                cb.set_label(cbar_label, fontsize=16)

                # Adjust layout
                plt.subplots_adjust(right=0.9)

                # ------------------------------
                # Plot each view, no colorbar on top row
                # ------------------------------
                plot_surf_stat_map(
                    surf_mesh=surf_inflated,
                    stat_map=overlay,
                    bg_map=bg_map,
                    hemi="left",
                    view="lateral",
                    cmap=cmap_used,
                    vmin=vmin,
                    vmax=vmax,
                    colorbar=False,
                    figure=fig,
                    axes=ax_lateral,
                    darkness=0.8,  # how dark the background is
                    bg_on_data=True,
                )
                # plot_surf_contours(
                #     surf_mesh=surf_inflated,
                #     roi_map=glasser_surf,
                #     view="lateral",
                #     hemi="left",
                #     figure=fig,
                #     axes=ax_lateral,
                #     colors=colors,
                #     labels=labels,
                #     levels=list(range(1, 181)),
                #     # legend=True,
                # )

                plot_surf_stat_map(
                    surf_mesh=surf_inflated,
                    stat_map=overlay,
                    bg_map=bg_map,
                    hemi="left",
                    view="medial",
                    cmap=cmap_used,
                    vmin=vmin,
                    vmax=vmax,
                    colorbar=False,
                    figure=fig,
                    axes=ax_medial,
                    darkness=0.8,  # how dark the background is
                    bg_on_data=True,
                )
                # plot_surf_contours(
                #     surf_mesh=surf_inflated,
                #     roi_map=glasser_surf,
                #     view="medial",
                #     hemi="left",
                #     figure=fig,
                #     axes=ax_medial,
                #     colors=colors,
                #     labels=labels,
                #     levels=list(range(1, 181)),
                #     # legend=True,
                # )

                plot_surf_stat_map(
                    surf_mesh=surf_inflated,
                    stat_map=overlay,
                    bg_map=bg_map,
                    hemi="left",
                    view="ventral",
                    cmap=cmap_used,
                    vmin=vmin,
                    vmax=vmax,
                    colorbar=False,
                    figure=fig,
                    axes=ax_ventral,
                    darkness=0.8,  # how dark the background is
                    bg_on_data=True,
                )
                # plot_surf_contours(
                #     surf_mesh=surf_inflated,
                #     roi_map=glasser_surf,
                #     view="ventral",
                #     hemi="left",
                #     figure=fig,
                #     axes=ax_ventral,
                #     colors=colors,
                #     labels=labels,
                #     levels=list(range(1, 181)),
                #     # legend=True,
                # )

                # ------------------------------
                # Bottom: dorsal view with the ONLY colorbar
                f1 = plot_surf_stat_map(
                    surf_mesh=surf_flat,
                    stat_map=overlay,
                    # bg_map=bg_map,
                    hemi="left",
                    view="dorsal",
                    cmap=cmap_used,
                    vmin=vmin,
                    vmax=vmax,
                    colorbar=False,  # only colorbar here
                    figure=fig,
                    axes=ax_dorsal,
                    darkness=0.2,  # how dark the background is
                    # bg_on_data=True,
                )
                plot_surf_contours(
                    surf_mesh=surf_flat,
                    roi_map=glasser_surf,
                    view="dorsal",
                    hemi="left",
                    figure=f1,
                    axes=ax_dorsal,
                    colors=colors,
                    labels=labels,
                    levels=list(range(1, 181)),
                    # legend=True,
                )


                fig.suptitle(title, fontsize=16, y=0.98)
                plt.tight_layout()

                # -------------------------------------------------------------
                # Save the figure
                # -------------------------------------------------------------
                output_filename = os.path.join(barplots_directory, title + "_surfplot.png")
                plt.savefig(output_filename)
                # plt.show()
                plt.close()
