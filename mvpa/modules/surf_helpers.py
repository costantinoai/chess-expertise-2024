#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 18 16:28:32 2025

@author: costantino_ai
"""

import os
import pickle
import nibabel as nib
from typing import Tuple, List
import numpy as np
import pandas as pd
from surfplot import Plot
from neuromaps.datasets import fetch_fsaverage
import matplotlib.image as mpimg
from matplotlib.colors import LinearSegmentedColormap
import logging
from logging_utils import setup_logging
from modules import (
    plt,
    LH_ANNOT,
    RH_ANNOT,
    ROIS_CSV,
    LEFT_LUT,
    RIGHT_LUT,
    P_ALPHA,
    FDR_ALPHA,
    HCPMMP1_LH_LABELS,
    HCPMMP1_RH_LABELS,
    MANAGER,
    ROIManager,
    CORTICES_GROUPS_CMAPS
)
from modules.helpers import save_script_to_file, filter_significant_ROIs

def plot_significant_surface_overlays(
    ttest_results_path,
    select_corrected_p_values=True,
    ROIS_CSV=ROIS_CSV,
    LEFT_LUT=LEFT_LUT,
    RIGHT_LUT=RIGHT_LUT,
    LH_ANNOT=LH_ANNOT,
    RH_ANNOT=RH_ANNOT,
    P_ALPHA=P_ALPHA,
    FDR_ALPHA=FDR_ALPHA,
    out_root=None,
):
    """
    Load T-test results from disk, filter and plot significant overlays on the surface,
    and generate single multi-panel figures (2×N) showing Non-Experts vs. Experts.

    Parameters
    ----------
    ttest_results_path : str
        Path to the pickled dictionary containing T-test results.
    ROIS_CSV : str
        Path to a CSV file used by ROIManager to map ROI indices to metadata.
    LEFT_LUT, RIGHT_LUT : str
        Paths to LUT files used by ROIManager for left and right hemispheres.
    LH_ANNOT, RH_ANNOT : str
        Paths to FreeSurfer annotation files for left and right hemispheres.
    P_ALPHA : float, optional
        p-value threshold for significance (default 0.05).
    FDR_ALPHA : float, optional
        Alpha level for FDR correction (default 0.05).
    out_root : str or None, optional
        Root directory to save all generated figures. If None, defaults to
        the directory containing 'ttest_results_path'.

    Returns
    -------
    None
        The function writes out individual hemisphere-overlay figures (one per regressor/expertise)
        and also writes out 2×N multi-panel figures summarizing them.
    """

    # If out_root is not provided, use the directory of the T-test results
    if out_root is None:
        out_root = os.path.join(os.path.dirname(ttest_results_path))

    # Create the output directory if needed
    os.makedirs(out_root, exist_ok=True)
    out_text_file = os.path.join(out_root, 'mvpa_logs_surfaces.log')
    setup_logging(log_file=out_text_file)

        # 1) Load results from disk
        with open(ttest_results_path, "rb") as f:
            results_dict = pickle.load(f)

        # Loop over each analysis, level
        for analysis in results_dict.keys():

            for level in results_dict[analysis].keys():
                if (level == "region") and ("svm" in analysis):

                    # Extract the slice of the dictionary for this analysis and level
                    sliced_dict = results_dict[analysis][level].copy()

                    # Get global maximum and minimum across regressors and exp levels
                    logging.info(f"ANALYSIS: {analysis}, LEVEL: {level}")
                    filtered_dict, global_max, measure_has_negative, max_error = filter_significant_ROIs(
                        sliced_dict, select_corrected_p_values, P_ALPHA, FDR_ALPHA
                    )
                    # global_min = -global_max if measure_has_negative else 0.0
                    global_max = .3
                    global_min = 0

                    # We'll store figure paths in a structure to build a grid later:
                    # figures_for_grid[regressor_name] = {False: path, True: path}
                    figures_for_grid = {}

                    # 7) For each regressor and expertise level, build the overlay and plot
                    for regressor_name, expertise_dict in filtered_dict.items():
                        for expertise_bool, figure_stats_df in expertise_dict.items():

                            # Build overlay
                            overlay_lh, overlay_rh = build_significance_overlay(
                                figure_stats_df,
                                manager=MANAGER,
                                lh_labels=HCPMMP1_LH_LABELS,
                                rh_labels=HCPMMP1_RH_LABELS,
                            )


                            # Setup labels for the figure
                            corr_str = (
                                f"FDR corrected (p<.{FDR_ALPHA})"
                                if select_corrected_p_values
                                else f"Uncorrected (p<.{P_ALPHA})"
                            )
                            expertise_str = "Experts" if expertise_bool else "Novices"

                            # Build output directory and filename
                            figure_out_dir = os.path.join(out_root, "group", level)
                            os.makedirs(figure_out_dir, exist_ok=True)

                            filename = (
                                f"{analysis}__{level}__{regressor_name}"
                                f"__{'experts' if expertise_bool else 'novices'}"
                                f"__{'fdr' if select_corrected_p_values else 'uncorrected'}_newflat.png"
                            )
                            figure_out_path = os.path.join(figure_out_dir, filename)

                            # Figure title
                            figure_title = (
                                f"{analysis.replace('_', ' ').upper()} - "
                                f"{corr_str} - "
                                f"{regressor_name.capitalize()} - "
                                f"{expertise_str}"
                            )

                            truncated_cmap = LinearSegmentedColormap.from_list(
                                "truncated_seismic", plt.cm.seismic(np.linspace(0.5, 1.0, 256))
                            )

                            # 8) Plot the fsaverage overlay (individual figure)
                            fig, saved_path = plot_fsaverage_overlay(
                                # (overlay_lh, overlay_rh), # the rh is a copy of lh. see build_significance_overlay()
                                (overlay_lh),
                                title=figure_title,
                                out_path=figure_out_path,
                                color_range=(global_min, global_max),
                                # views=("lateral", "medial", "ventral"),
                                # zoom=1.2,
                                cmap_positive=truncated_cmap,
                                cmap_negative="seismic",
                            )

                            # Save the path in the dictionary for the multi-panel grid
                            if regressor_name not in figures_for_grid:
                                figures_for_grid[regressor_name] = {}
                            figures_for_grid[regressor_name][expertise_bool] = saved_path

                    # 9) Once we've plotted everything, create a single multi-panel figure
                    #    (2×N: row=0 => Non-Experts, row=1 => Experts, columns => regressors)
                    multi_fig_title = (
                        f"{analysis.replace('_', ' ').upper()} {level.capitalize()} ({corr_str})"
                    )
                    multi_fig_filename = f"{analysis}_{level}_{'fdr' if select_corrected_p_values else 'uncorrected'}_grid.png"
                    multi_fig_out_path = os.path.join(figure_out_dir, multi_fig_filename)

                    # 10) Build the 2×N grid figure
                    _ = plot_experts_vs_non_experts_grid(
                        figures_for_grid, title=multi_fig_title, out_path=multi_fig_out_path
                    )

                    # Save the current script in output dir for future reference
                    save_script_to_file(figure_out_dir)

    return results_dict


def build_significance_overlay(
    stats_df: pd.DataFrame,
    manager: ROIManager,
    lh_labels: np.ndarray,
    rh_labels: np.ndarray,
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

    # Initialize overlays for the left and right hemispheres with NaNs
    overlay_lh = np.full_like(lh_labels, np.nan, dtype=float)
    overlay_rh = np.full_like(rh_labels, np.nan, dtype=float)

    # Extract significant ROI names from the DataFrame index
    significant_roi_names = stats_df.index

    for significant_roi_name in significant_roi_names:

        # Determine the hemisphere (left or right) and hierarchical level (ROI, cortex, lobe)
        hemisphere = significant_roi_name[0]
        hierarchical_level = significant_roi_name.split("_")[-1]

        # Clean up the significant ROI name based on the hierarchical level
        if hierarchical_level == "ROI":
            significant_roi_name_clean = significant_roi_name[2:].replace("_ROI", "")
        elif hierarchical_level == "cortex":
            significant_roi_name_clean = significant_roi_name[2:].replace("_cortex", "")
        elif hierarchical_level == "lobe":
            significant_roi_name_clean = significant_roi_name[2:].replace("_lobe", "")
        else:
            raise ValueError(
                f"Unexpected hierarchical level in ROI name: {significant_roi_name}"
            )

        # Retrieve ROIs using the manager's filter method based on the hierarchical level
        selected_rois, _ = manager.get_by_filter(
            hemisphere=hemisphere,
            lobe=significant_roi_name_clean if hierarchical_level == "lobe" else None,
            cortex=significant_roi_name_clean if hierarchical_level == "cortex" else None,
            region=significant_roi_name_clean if hierarchical_level == "ROI" else None,
        )

        # Raise an error if no ROIs are found for the given criteria
        if len(selected_rois) == 0:
            raise ValueError(
                f"No ROIs found for significant region: {significant_roi_name}"
            )

        # Extract the mean value for the current ROI from the DataFrame
        mean_value = stats_df.loc[significant_roi_name, "mean-chance"]

        # Iterate through the selected ROIs and assign values to the overlay maps
        for selected_roi in selected_rois:
            if hemisphere.lower() == "l":
                # Assign values to the left hemisphere overlay
                overlay_lh[lh_labels == selected_roi.region_id] = mean_value
                overlay_rh[rh_labels == selected_roi.region_id] = mean_value

            elif hemisphere.lower() == "r":
                # Assign values to the right hemisphere overlay
                overlay_rh[rh_labels == selected_roi.region_id] = mean_value

    # Return the overlays for both hemispheres
    return overlay_lh, overlay_rh


# def build_ROI_choice_overlay(
#     selected_rois: list,
#     lh_labels: np.ndarray,
#     rh_labels: np.ndarray,
# ) -> Tuple[Tuple[np.ndarray, np.ndarray], float, float]:
#     """
#     Builds overlay maps for the left and right hemispheres based on statistical values.

#     For each region of interest (ROI), the function assigns the mean value from the provided
#     stats_df to the overlay map if the ROI is significant. If not, the value is set to 0.

#     Parameters
#     ----------
#     stats_df : pd.DataFrame
#         DataFrame containing statistics for each ROI. It must have ROI names as the index and
#         include a column "mean-chance" with mean values for each ROI.

#     manager : ROIManager
#         Object managing ROI information, including mapping region names to region IDs.

#     lh_labels : np.ndarray
#         Array of labels corresponding to the left hemisphere ROIs.

#     rh_labels : np.ndarray
#         Array of labels corresponding to the right hemisphere ROIs.

#     Returns
#     -------
#     overlays : Tuple[np.ndarray, np.ndarray]
#         Tuple containing overlay arrays for the left and right hemispheres.
#     """

#     # Initialize overlays for the left and right hemispheres with zeros
#     overlay_lh = np.full_like(lh_labels, np.nan, dtype=float)
#     overlay_rh = np.full_like(rh_labels, np.nan, dtype=float)

#     # Paint each ROI's region_id
#     for roi in selected_rois:
#         if roi.region_id is None:
#             continue
#         if roi.hemisphere.upper() == "L":
#             overlay_lh[lh_labels == roi.region_id] = roi.region_id
#             overlay_rh[rh_labels == roi.region_id] = roi.region_id
#         else:
#             overlay_rh[rh_labels == roi.region_id] = roi.region_id

#     valid_lh = overlay_lh[~np.isnan(overlay_lh)]
#     valid_rh = overlay_rh[~np.isnan(overlay_rh)]
#     if len(valid_lh) == 0 and len(valid_rh) == 0:
#         # means no valid data
#         return overlay_lh, overlay_rh, float("inf"), float("-inf")

#     global_min = float(
#         min(
#             valid_lh.min() if len(valid_lh) else np.inf,
#             valid_rh.min() if len(valid_rh) else np.inf,
#         )
#     )
#     global_max = float(
#         max(
#             valid_lh.max() if len(valid_lh) else -np.inf,
#             valid_rh.max() if len(valid_rh) else -np.inf,
#         )
#     )

#     # Return the overlays as a tuple
#     return (overlay_lh, overlay_rh), global_max, global_min


# def plot_fsaverage_overlay(
#     overlays,
#     title="",
#     out_path=None,
#     color_range=(-1, 1),
#     views=("lateral", "medial", "ventral"),
#     brightness=0.8,
#     size=(1200, 1500),
#     zoom=1.2,
#     cmap_positive="Reds",
#     cmap_negative="coolwarm",
# ):
#     """
#     Plot an overlay (left and/or right hemisphere) on the fsaverage inflated surface.

#     Parameters
#     ----------
#     overlays : tuple or list
#         Tuple (overlay_lh, overlay_rh). Each overlay can be:
#          - an array of shape (n_vertices,) corresponding to vertex-wise data
#          - None, if that hemisphere should not be displayed
#     title : str, optional
#         Title of the figure. Default: "" (no title).
#     out_path : str or None, optional
#         If provided, the figure will be saved to this path. If None, the figure is not saved.
#         Default: None.
#     color_range : tuple, optional
#         (min_val, max_val) for color scaling. Default: (-1, 1).
#     views : tuple, optional
#         Views to display. Default: ("lateral", "medial", "ventral").
#     brightness : float, optional
#         Surface brightness for Plot. Default: 0.6.
#     size : tuple, optional
#         (width_px, height_px) for Plot. Default: (500, 700).
#     zoom : float, optional
#         Zoom factor for Plot. Default: 1.2.
#     cmap_positive : str, optional
#         Colormap to use if data has only nonnegative values. Default: "Reds".
#     cmap_negative : str, optional
#         Colormap to use if data has negative values. Default: "coolwarm".
#     show_figure : bool, optional
#         If True, display the figure after building. Default: True.

#     Returns
#     -------
#     fig : matplotlib.figure.Figure
#         The created figure object.
#     saved_path : str or None
#         The path where the figure was saved, or None if not saved.
#     """

#     try:
#         overlay_lh, overlay_rh = overlays
#     except:
#         overlay_lh = overlays
#         overlay_rh = None


#     # Quick check: if both are None, there's nothing to plot
#     if overlay_lh is None and overlay_rh is None:
#         raise ValueError("Both left and right overlays cannot be None.")

#     # Fetch surfaces
#     surfaces = fetch_fsaverage(density="164k")
#     lh_inflated, rh_inflated = surfaces["inflated"]
#     lh_sulc, rh_sulc = surfaces["sulc"]

#     # Decide which hemispheres to pass to Plot
#     if (overlay_lh is not None) and (overlay_rh is not None):

#         p = Plot(
#             lh_inflated,
#             rh_inflated,
#             views=views,
#             brightness=brightness,
#             size=size,
#             zoom=zoom,
#         )
#         # Shading
#         p.add_layer(
#             {"left": lh_sulc, "right": rh_sulc},
#             cmap="binary_r",
#             cbar=False,
#             alpha=0.5,
#         )
#         # Overlay
#         cmap_used = cmap_negative if color_range[0] < 0 else cmap_positive
#         p.add_layer(
#             {"left": overlay_lh, "right": overlay_rh},
#             cmap=cmap_used,
#             color_range=color_range,
#             alpha=1,
#         )

#         p.add_layer(
#             {"left": overlay_lh, "right": overlay_rh},
#             cmap="gray",
#             alpha=1,
#             color_range=(0, 100),
#             as_outline=True,
#             cbar=False,
#         )

#     elif overlay_lh is not None:
#         # Only left hemisphere
#         p = Plot(
#             lh_inflated,
#             views=("lateral", "medial", "ventral"),
#             brightness=brightness,
#             size=(1000*3, 300*3),
#             zoom=zoom,
#         )
#         # Shading
#         p.add_layer(
#             {"left": lh_sulc},
#             cmap="binary_r",
#             cbar=False,
#             alpha=0.5,
#         )
#         # Overlay
#         cmap_used = cmap_negative if color_range[0] < 0 else cmap_positive
#         p.add_layer(
#             {"left": overlay_lh},
#             cmap=cmap_used,
#             color_range=color_range,
#             alpha=1,
#         )

#         p.add_layer(
#             {"left": overlay_lh},
#             cmap="gray",
#             alpha=1,
#             color_range=(0, 100),
#             as_outline=True,
#             cbar=False,
#         )

#     else:
#         # Only right hemisphere
#         p = Plot(
#             rh_inflated,
#             views=views,
#             brightness=brightness,
#             size=size,
#             zoom=zoom,
#         )
#         p.add_layer(
#             {"right": rh_sulc},
#             cmap="binary_r",
#             cbar=False,
#             alpha=0.5,
#         )
#         cmap_used = cmap_negative if color_range[0] < 0 else cmap_positive
#         p.add_layer(
#             {"right": overlay_rh},
#             cmap=cmap_used,
#             color_range=color_range,
#             alpha=1,
#         )

#         p.add_layer(
#             {"right": overlay_rh}, cmap="gray", alpha=1, as_outline=True, cbar=False
#         )

#     # Build the figure
#     cbar_kws = dict(outer_labels_only=True, pad=0.02, n_ticks=2, fontsize=20)
#     fig = p.build(cbar_kws=cbar_kws)

#     # Add a label to the colorbar
#     if len(fig.axes) > 1:
#         xlabel = "Accuracy - Chance (%)" if "SVM" in title else "Coefficient"
#         fig.axes[1].set_xlabel(xlabel, fontstyle="italic")

#     # Add title
#     if title:
#         fig.suptitle(title, y=0.9)

#     # Show figure
#     plt.show()

#     # Save figure if out_path is given
#     saved_path = None
#     if out_path is not None:
#         saved_path = os.path.abspath(out_path)
#         fig.savefig(saved_path, dpi=300, bbox_inches="tight")

#     plt.show()

#     return fig, saved_path

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
                if line.strip() and not line.startswith("#"):  # Ignore comments and empty lines
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

    from config import DERIVATIVES_PATH, MISC_PATH
    data = {
        "left": os.path.join(str(DERIVATIVES_PATH), "fastsurfer", "fsaverage", "label", "lh.HCPMMP1.annot"),
        "right": os.path.join(str(DERIVATIVES_PATH), "fastsurfer", "fsaverage", "label", "rh.HCPMMP1.annot"),
    }

    cmaps = {
        "left": load_freesurfer_lut(os.path.join(str(MISC_PATH), "lh_HCPMMP1_color_table.txt"))[0],
        "right": load_freesurfer_lut(os.path.join(str(MISC_PATH), "rh_HCPMMP1_color_table.txt"))[0],
    }

    fsaverage = load_fsaverage("fsaverage")

    glasser_atlas = SurfaceImage(
        mesh=fsaverage["flat"],
        data=data,
    )

    return glasser_atlas, cmaps


import numpy as np
from nilearn.plotting import plot_surf_stat_map, plot_surf_contours
from nilearn.datasets import load_fsaverage
from nilearn.datasets import load_fsaverage_data

def plot_fsaverage_overlay(
    overlays,
    title="",
    out_path=None,
    color_range=(-1, 1),
    cmap_positive="Reds",
    cmap_negative="coolwarm",
    views=None
):
    """
    Plot a grid of four inflated views (lateral, medial, ventral on top,
    dorsal on the bottom) for the left hemisphere in fsaverage space,
    with contours overlaid.

    Parameters
    ----------
    overlays : array or (array, None)
        - An array of shape (n_vertices,) for the left hemisphere data.
          If using a tuple, the second entry is ignored (right hemisphere).
    title : str, optional
        Suptitle for the entire figure.
    out_path : str or None, optional
        Path to save the figure. If None, figure is not saved.
    color_range : (float, float), optional
        (vmin, vmax) for shared color scale across all subplots.
    cmap_positive : str, optional
        Colormap if data are nonnegative or you just prefer a positive colormap.
    cmap_negative : str, optional
        Colormap if the data include negative values.
    """


    # Unpack overlays for potential future extension,
    # but we'll only use the left hemisphere here.
    try:
        overlay_lh, _ = overlays
    except:
        overlay_lh = overlays

    if overlay_lh is None:
        raise ValueError("No overlay for the left hemisphere. Cannot plot.")

    # Select the colormap depending on sign of data range
    vmin, vmax = color_range
    cmap_used = cmap_negative if vmin < 0 else cmap_positive

    # -- fsaverage surfaces (only need left hemisphere if we're plotting left) --
    fsaverage_meshes = load_fsaverage("fsaverage")
    surf_inflated = fsaverage_meshes["inflated"]  # (coords, faces)
    surf_flat = fsaverage_meshes["flat"]  # (coords, faces)

    # Just ensure glasser_surf is a 1D array (n_vertices,).
    glasser_surf, glasser_cmaps = load_glasser_surf()
    colors = [next(color for group, color in CORTICES_GROUPS_CMAPS if group == r.cortex_group) for r in MANAGER.rois if r.hemisphere == "L"]
    labels = [r.region_long_name for r in MANAGER.rois if r.hemisphere == "L"]

    # Create figure with custom size
    fig = plt.figure(figsize=(10, 8))

    # Create a gridspec with 2 rows and 3 columns
    # Adjust height_ratios so the bottom row is larger
    # wspace/hspace reduce white space horizontally/vertically
    gs = fig.add_gridspec(
        nrows=2, ncols=3,
        height_ratios=[1.0, 1.5],  # top row : bottom row
        wspace=0.05,               # reduce space between columns
        hspace=0.03                # reduce space between rows
    )

    # Top row (3 subplots: lateral, medial, ventral)
    ax_lateral = fig.add_subplot(gs[0, 0], projection="3d")
    ax_medial  = fig.add_subplot(gs[0, 1], projection="3d")
    ax_ventral = fig.add_subplot(gs[0, 2], projection="3d")

    # Bottom row (spans all 3 columns)
    ax_dorsal = fig.add_subplot(gs[1, :], projection="3d")

    # Optionally adjust margins around the whole figure
    plt.subplots_adjust(
        left=0.05,   # space on the left
        right=0.88,  # space on the right
        top=0.93,    # space at the top
        bottom=0.04  # space at the bottom
    )


    bg_map=load_fsaverage_data(mesh="fsaverage", data_type="sulcal")

    import matplotlib.colorbar as cbar
    import matplotlib.colors as mcolors

    # ------------------------------
    # Create a colorbar to the right
    # ------------------------------
    # ------------------------------
    cbar_label = "Coefficient"
    if "SVM" in title:
        cbar_label = "Accuracy - Chance (%)"

    cax = fig.add_axes([0.95, 0.25, 0.01, 0.25])  # [left, bottom, width, height]

    # Create a normalizer with vmin and vmax
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

    # Create colorbar
    cb = cbar.ColorbarBase(
        cax,
        cmap=cmap_used,
        norm=norm,
        orientation="vertical"
    )

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
        stat_map=overlay_lh,
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
        # bg_on_data=True,
    )
    plot_surf_contours(
        surf_mesh=surf_inflated,
        roi_map=glasser_surf,
        view="lateral",
        hemi="left",
        figure=fig,
        axes=ax_lateral,
        colors=colors,
        labels=labels,
        levels=list(range(1, 181)),
        # legend=True,
    )

    plot_surf_stat_map(
        surf_mesh=surf_inflated,
        stat_map=overlay_lh,
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
        # bg_on_data=True,
    )
    plot_surf_contours(
        surf_mesh=surf_inflated,
        roi_map=glasser_surf,
        view="medial",
        hemi="left",
        figure=fig,
        axes=ax_medial,
        colors=colors,
        labels=labels,
        levels=list(range(1, 181)),
        # legend=True,
    )

    plot_surf_stat_map(
        surf_mesh=surf_inflated,
        stat_map=overlay_lh,
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
        # bg_on_data=True,
    )
    plot_surf_contours(
        surf_mesh=surf_inflated,
        roi_map=glasser_surf,
        view="ventral",
        hemi="left",
        figure=fig,
        axes=ax_ventral,
        colors=colors,
        labels=labels,
        levels=list(range(1, 181)),
        # legend=True,
    )

    # ------------------------------
    # Bottom: dorsal view with the ONLY colorbar
    f1 = plot_surf_stat_map(
        surf_mesh=surf_flat,
        stat_map=overlay_lh,
        # bg_map=bg_map,
        hemi="left",
        view="dorsal",
        cmap=cmap_used,
        vmin=vmin,
        vmax=vmax,
        colorbar=False,  # only colorbar here
        figure=fig,
        axes=ax_dorsal,
        darkness=0.7,  # how dark the background is
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

    # ------------------------------
    # Suptitle, saving, showing
    # ------------------------------
    fig.suptitle(title, fontsize=16, y=0.98)
    plt.tight_layout()

    if out_path:
        out_path = os.path.abspath(out_path)
        fig.savefig(out_path, dpi=300, bbox_inches="tight")

    plt.show()

    return fig, out_path



def plot_experts_vs_non_experts_grid(
    figures_dict,
    title="",
    out_path=None,
):
    """
    Arrange saved brain-plot figures in a 2×N grid:
      - Non-Experts on the top row
      - Experts on the bottom row
      - Columns are regressors

    Parameters
    ----------
    figures_dict : dict
        {
          "regressor1": {False: "/path/to/non_experts_fig.png",
                         True:  "/path/to/experts_fig.png"},
          "regressor2": {...},
          ...
        }
        Each regressor has two keys, False (non-experts) and True (experts),
        mapping to the saved figure paths.
    title : str
        Title for the entire multi-panel figure.
    out_path : str or None
        If provided, the figure is saved to this path.
    figsize : tuple
        Matplotlib figure size in inches (width, height). You may want to
        scale it by the number of columns, e.g. (4 * num_regressors, 6).
    regressor_fontsize : int
        Font size for regressor titles on each column.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The created multi-panel figure.

    Notes
    -----
    - This function loads the images from disk (using the paths stored in figures_dict)
      and places them into the subplots via `imshow`.
    - By default, each subplot has `axis('off')`.
    """
    # Number of regressors = number of columns
    regressors = list(figures_dict.keys())
    n_regressors = len(regressors)

    # We know we want 2 rows: 0 (non-experts), 1 (experts)
    # Adjust figsize so it scales with the number of columns
    # if you'd like. For example:
    figsize = plt.rcParams["figure.figsize"]
    dynamic_figsize = (4 * n_regressors, figsize[1])

    fig, axs = plt.subplots(
        nrows=2, ncols=n_regressors, figsize=dynamic_figsize, squeeze=False
    )

    for c, regressor_name in enumerate(regressors):
        # Each regressor sub-dict: {False: path, True: path}
        expertise_dict = figures_dict[regressor_name]

        # Non-experts = expertise_bool == False => row 0
        # Experts = expertise_bool == True => row 1
        for expertise_bool, fig_path in expertise_dict.items():
            row_idx = 0 if not expertise_bool else 1

            # Load the image
            if fig_path is not None and os.path.isfile(fig_path):
                img = mpimg.imread(fig_path)
                axs[row_idx][c].imshow(img)
            else:
                # If the path doesn't exist, display a placeholder
                axs[row_idx][c].text(
                    0.5,
                    0.5,
                    "No image",
                    ha="center",
                    va="center",
                    transform=axs[row_idx][c].transAxes,
                )

            axs[row_idx][c].axis("off")

        # Optionally label the column with the regressor name
        # Place text at the top row (row=0), or use a suptitle if you prefer.
        axs[0][c].set_title(regressor_name, fontsize=20)

    # Add a global title if provided
    if title:
        fig.suptitle(title)

    plt.tight_layout()

    # # Save figure if out_path is given
    # if out_path is not None:
    #     fig.savefig(out_path, dpi=300, bbox_inches="tight")

    plt.show()

    return fig


# def plot_roi_selection_overlay(
#     freesurfer_path,
#     lh_annotation,
#     rh_annotation,
#     manager,
#     out_path=None,
#     show_plot=True,
#     views=("lateral", "medial", "ventral"),
#     **filters,
# ) -> None:
#     """
#     1. Select ROIs from manager using filters.
#     2. Build (overlay_lh, overlay_rh) with region_id, ignoring non-selected.
#     3. Use the shared plot_fs_surface_grid for final visualization.

#     If out_path is None, the figure won't be saved, only shown if show_plot=True.
#     """

#     def build_selection_overlay(
#         lh_labels: np.ndarray, rh_labels: np.ndarray, selected_rois: List[ROI]
#     ) -> Tuple[np.ndarray, np.ndarray, float, float]:
#         """
#         Build a single (overlay_lh, overlay_rh) pair for a given set of selected ROIs,
#         assigning each ROI's region_id or some integer, while non-selected are np.nan.

#         Returns
#         -------
#         overlay_lh, overlay_rh : np.ndarray
#             Float arrays shaped like lh_labels, rh_labels.
#         global_min, global_max : float
#             The min/max of the valid (non-nan) entries.
#         """
#         overlay_lh = np.full_like(lh_labels, np.nan, dtype=float)
#         overlay_rh = np.full_like(rh_labels, np.nan, dtype=float)

#         # Paint each ROI's region_id
#         for roi in selected_rois:
#             if roi.region_id is None:
#                 continue
#             if roi.hemisphere.upper() == "L":
#                 overlay_lh[lh_labels == roi.region_id] = roi.region_id
#                 overlay_rh[rh_labels == roi.region_id] = roi.region_id
#             else:
#                 overlay_rh[rh_labels == roi.region_id] = roi.region_id

#         valid_lh = overlay_lh[~np.isnan(overlay_lh)]
#         valid_rh = overlay_rh[~np.isnan(overlay_rh)]
#         if len(valid_lh) == 0 and len(valid_rh) == 0:
#             # means no valid data
#             return overlay_lh, overlay_rh, float("inf"), float("-inf")

#         global_min = float(
#             min(
#                 valid_lh.min() if len(valid_lh) else np.inf,
#                 valid_rh.min() if len(valid_rh) else np.inf,
#             )
#         )
#         global_max = float(
#             max(
#                 valid_lh.max() if len(valid_lh) else -np.inf,
#                 valid_rh.max() if len(valid_rh) else -np.inf,
#             )
#         )
#         return overlay_lh, overlay_rh, global_min, global_max

#     # 1) Filter ROIs
#     selected_rois, _ = manager.get_by_filter(**filters)
#     if not selected_rois:
#         logging.warning(f"No ROIs found with filters={filters}. Nothing to plot.")
#         return

#     # 2) Load .annot
#     lh_labels, lh_ctab, lh_names = nib.freesurfer.read_annot(str(lh_annotation))
#     rh_labels, rh_ctab, rh_names = nib.freesurfer.read_annot(str(rh_annotation))

#     # 3) Build single overlay
#     overlay_lh, overlay_rh, gmin, gmax = build_selection_overlay(
#         lh_labels, rh_labels, selected_rois
#     )
#     if gmin == float("inf") and gmax == float("-inf"):
#         logging.warning("All selected ROIs have no assigned region_id. Aborting.")
#         return

#     overlays = [(overlay_lh, overlay_rh)]

#     # Fetch surfaces
#     surfaces = fetch_fsaverage(density="164k")
#     lh_inflated, rh_inflated = surfaces["inflated"]
#     lh_sulc, rh_sulc = surfaces["sulc"]

#     p = Plot(
#         lh_inflated,
#         rh_inflated,
#         views=views,
#         brightness=0.8,
#         size=(1200, 1500),
#         zoom=1.2,
#     )
#     # Shading
#     p.add_layer(
#         {"left": lh_sulc, "right": rh_sulc},
#         cmap="binary_r",
#         cbar=False,
#         alpha=0.5,
#     )
#     # Overlay
#     p.add_layer(
#         {"left": overlays[0], "right": overlays[1]},
#         color_range=(1, 180),
#         alpha=1,
#     )

#     p.add_layer(
#         {"left": overlay_lh, "right": overlay_rh},
#         cmap="gray",
#         alpha=1,
#         color_range=(0, 100),
#         as_outline=True,
#         cbar=False,
#     )
