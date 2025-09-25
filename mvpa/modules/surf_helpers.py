#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Surface overlay helpers for MVPA figures.

Builds hemisphere overlays from FreeSurfer HCP-MMP1 annotations plus centralized
ROI TSV metadata (rois/sets/*). ROIManager/LUT logic removed. Supports ROI- and
cortex-level overlays; lobe overlays are intentionally not implemented.
"""

from __future__ import annotations

import os
import pickle
import logging
from typing import Tuple

import numpy as np
import pandas as pd
import nibabel as nib
import matplotlib.image as mpimg
from matplotlib.colors import LinearSegmentedColormap

from common.logging_utils import setup_logging
from common.surface_plotting import plot_left_fsaverage_grid

from modules import (
    plt,
    LH_ANNOT,
    RH_ANNOT,
    P_ALPHA,
    FDR_ALPHA,
    HCPMMP1_LH_LABELS,
    HCPMMP1_RH_LABELS,
    HCPMMP1_LH_NAMES,
    HCPMMP1_RH_NAMES,
    CORTICES_GROUPS_CMAPS,
)
from rois.meta import get_roi_info
from modules.helpers import save_script_to_file, filter_significant_ROIs


def plot_significant_surface_overlays(
    ttest_results_path: str,
    select_corrected_p_values: bool = True,
    LH_ANNOT: str = LH_ANNOT,
    RH_ANNOT: str = RH_ANNOT,
    P_ALPHA: float = P_ALPHA,
    FDR_ALPHA: float = FDR_ALPHA,
    out_root: str | None = None,
):
    """Load group results, build ROI/cortex overlays, and save fsaverage plots.

    Writes per-regressor figures and a 2×N grid (Novices vs Experts) by regressor.
    """
    if out_root is None:
        out_root = os.path.join(os.path.dirname(ttest_results_path))
    os.makedirs(out_root, exist_ok=True)
    setup_logging(log_file=os.path.join(out_root, "mvpa_logs_surfaces.log"))

    with open(ttest_results_path, "rb") as f:
        results_dict = pickle.load(f)

    for analysis in results_dict.keys():
        for level in results_dict[analysis].keys():
            if (level == "region") and ("svm" in analysis):
                sliced = results_dict[analysis][level].copy()

                logging.info("ANALYSIS=%s LEVEL=%s", analysis, level)
                _, global_max, measure_has_negative, max_error = filter_significant_ROIs(
                    sliced, select_corrected_p_values, P_ALPHA, FDR_ALPHA
                )
                global_max = 0.3
                global_min = 0.0

                figures_for_grid: dict[str, dict[bool, str | None]] = {}

                for regressor_name, expertise_dict in sliced.items():
                    for expertise_bool, figure_stats_df in expertise_dict.items():
                        overlay_lh, overlay_rh = build_significance_overlay(
                            figure_stats_df,
                            lh_labels=HCPMMP1_LH_LABELS,
                            rh_labels=HCPMMP1_RH_LABELS,
                        )

                        corr_str = (
                            f"FDR corrected (p<.{FDR_ALPHA})"
                            if select_corrected_p_values
                            else f"Uncorrected (p<.{P_ALPHA})"
                        )
                        expertise_str = "Experts" if expertise_bool else "Novices"

                        figure_out_dir = os.path.join(out_root, "group", level)
                        os.makedirs(figure_out_dir, exist_ok=True)
                        filename = (
                            f"{analysis}__{level}__{regressor_name}__"
                            f"{'experts' if expertise_bool else 'novices'}__"
                            f"{'fdr' if select_corrected_p_values else 'uncorrected'}_newflat.png"
                        )
                        figure_out_path = os.path.join(figure_out_dir, filename)

                        figure_title = (
                            f"{analysis.replace('_', ' ').upper()} - {corr_str} - "
                            f"{regressor_name.capitalize()} - {expertise_str}"
                        )

                        truncated_cmap = LinearSegmentedColormap.from_list(
                            "truncated_seismic", plt.cm.seismic(np.linspace(0.5, 1.0, 256))
                        )

                        _, saved_path = plot_fsaverage_overlay(
                            (overlay_lh),
                            title=figure_title,
                            out_path=figure_out_path,
                            color_range=(global_min, global_max),
                            cmap_positive=truncated_cmap,
                            cmap_negative="seismic",
                        )

                        if regressor_name not in figures_for_grid:
                            figures_for_grid[regressor_name] = {}
                        figures_for_grid[regressor_name][expertise_bool] = saved_path

                multi_fig_title = (
                    f"{analysis.replace('_', ' ').upper()} {level.capitalize()} "
                    f"({'FDR' if select_corrected_p_values else 'Uncorrected'})"
                )
                multi_fig_out_path = os.path.join(figure_out_dir, f"{analysis}_{level}_grid.png")
                _ = plot_experts_vs_non_experts_grid(figures_for_grid, title=multi_fig_title, out_path=multi_fig_out_path)
                save_script_to_file(figure_out_dir)

    return results_dict


def build_significance_overlay(
    stats_df: pd.DataFrame,
    lh_labels: np.ndarray,
    rh_labels: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Construct per-hemisphere overlays based on significant ROI statistics.

    Expects stats_df index to contain names like 'L_V1_ROI' or 'R_V1_cortex'.
    Uses ROI TSV (rois/meta) for cortex-level aggregation via fs_name.
    """
    overlay_lh = np.full_like(lh_labels, np.nan, dtype=float)
    overlay_rh = np.full_like(rh_labels, np.nan, dtype=float)

    roi_info = get_roi_info("glasser_regions_bilateral")
    name_to_idx_lh = {name: i for i, name in enumerate(HCPMMP1_LH_NAMES)}
    name_to_idx_rh = {name: i for i, name in enumerate(HCPMMP1_RH_NAMES)}

    for significant_roi_name in stats_df.index:
        hemisphere = significant_roi_name[0]
        hierarchical_level = significant_roi_name.split("_")[-1]

        if hierarchical_level == "ROI":
            roi_clean = significant_roi_name[2:].replace("_ROI", "")
        elif hierarchical_level == "cortex":
            roi_clean = significant_roi_name[2:].replace("_cortex", "")
        elif hierarchical_level == "lobe":
            # not implemented by design
            logging.warning("Lobe overlays not implemented; skipping %s", significant_roi_name)
            continue
        else:
            logging.warning("Unexpected overlay level in '%s'", significant_roi_name)
            continue

        mean_value = stats_df.loc[significant_roi_name, "mean-chance"]

        if hierarchical_level == "ROI":
            fs_name = f"{hemisphere}_{roi_clean}_ROI".encode()
            if hemisphere.lower() == "l":
                idx = name_to_idx_lh.get(fs_name)
                if idx is not None:
                    overlay_lh[lh_labels == idx] = mean_value
                else:
                    logging.warning("FS name not found in LH annot: %s", fs_name)
                idx_r = name_to_idx_rh.get(fs_name.replace(b"L_", b"R_"))
                if idx_r is not None:
                    overlay_rh[rh_labels == idx_r] = mean_value
            else:
                idx = name_to_idx_rh.get(fs_name)
                if idx is not None:
                    overlay_rh[rh_labels == idx] = mean_value
                else:
                    logging.warning("FS name not found in RH annot: %s", fs_name)

        elif hierarchical_level == "cortex":
            target_cortex = roi_clean.replace("_", " ").strip().lower()
            for rid, cortex in roi_info.id_to_cortex.items():
                if cortex.strip().lower() != target_cortex:
                    continue
                fsn = roi_info.id_to_fsname.get(rid, "")
                if not fsn:
                    continue
                if hemisphere.lower() == "l" and fsn.startswith("L_"):
                    idx = name_to_idx_lh.get(fsn.encode())
                    if idx is not None:
                        overlay_lh[lh_labels == idx] = mean_value
                elif hemisphere.lower() == "r" and fsn.startswith("R_"):
                    idx = name_to_idx_rh.get(fsn.encode())
                    if idx is not None:
                        overlay_rh[rh_labels == idx] = mean_value

    return overlay_lh, overlay_rh


def plot_fsaverage_overlay(
    overlays,
    title: str = "",
    out_path: str | None = None,
    color_range=(-1, 1),
    cmap_positive="Reds",
    cmap_negative="coolwarm",
    views=None,
):
    """Delegate fsaverage left-hemisphere grid plotting to shared helper."""
    try:
        overlay_lh, _ = overlays
    except Exception:
        overlay_lh = overlays
    return plot_left_fsaverage_grid(
        overlay_lh=overlay_lh,
        label_map_lh=HCPMMP1_LH_LABELS,
        title=title,
        out_path=out_path,
        color_range=color_range,
        cmap_positive=cmap_positive,
        cmap_negative=cmap_negative,
        add_colorbar=True,
        add_contours=True,
    )


def plot_experts_vs_non_experts_grid(
    figures_dict: dict[str, dict[bool, str | None]],
    title: str = "",
    out_path: str | None = None,
):
    """Arrange saved brain-plot images in a 2×N grid (Novices top, Experts bottom)."""
    regressors = list(figures_dict.keys())
    n_regressors = len(regressors)
    figsize = plt.rcParams["figure.figsize"]
    dynamic_figsize = (4 * n_regressors, figsize[1])

    fig, axs = plt.subplots(nrows=2, ncols=n_regressors, figsize=dynamic_figsize, squeeze=False)
    for c, regressor_name in enumerate(regressors):
        expertise_dict = figures_dict[regressor_name]
        for expertise_bool, fig_path in expertise_dict.items():
            row_idx = 0 if not expertise_bool else 1
            if fig_path is not None and os.path.isfile(fig_path):
                img = mpimg.imread(fig_path)
                axs[row_idx][c].imshow(img)
            else:
                axs[row_idx][c].text(0.5, 0.5, "No image", ha="center", va="center", transform=axs[row_idx][c].transAxes)
            axs[row_idx][c].axis("off")
        axs[0][c].set_title(regressor_name, fontsize=20)

    if title:
        fig.suptitle(title)
    plt.tight_layout()
    if out_path is not None:
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.show()
    return fig

