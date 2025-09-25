#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MVPA plotting helpers.

This module orchestrates MVPA plotting and delegates actual plotting
to shared utilities under `common/` to keep styles/logic DRY.
"""

import os
import numpy as np
import pickle
import pandas as pd
import logging
from modules import plt
from common.logging_utils import setup_logging
from modules import (
    FDR_ALPHA,
    P_ALPHA,
    CORTICES_NAMES,
    CORTICES_GROUPS_CMAPS,
)
from modules.helpers import filter_significant_ROIs, save_script_to_file
from common.plotting_utils import plot_mvpa_barplot as shared_plot_mvpa_barplot


def plot_group_level_mvpa(ttest_results_path,
                          plot_corrected=True):
    """
    Plot MVPA group-level results across different levels and analyses, using
    the new dictionary-of-dataframes structure. One figure per (analysis, level, expertise).

    Parameters:
        ttest_results_path (str): Path to the t-test results pickle file.
        plot_corrected (bool): If True, use the FDR-corrected p-values for star annotations.

    Returns:
        None
    """

    corr_str = (
        f"FDR corrected (p<{FDR_ALPHA})"
        if plot_corrected
        else f"Uncorrected (p<{P_ALPHA})"
    )

    # Output root
    output_root = os.path.dirname(ttest_results_path)
    os.makedirs(output_root, exist_ok=True)
    # Keep a copy of the script alongside outputs for provenance
    save_script_to_file(output_root)

    # 1) Load your dictionary of results from disk
    with open(ttest_results_path, "rb") as f:
        results_dict = pickle.load(f)

    out_text_file = os.path.join(output_root, 'mvpa_logs_barplots.log')

    # Ensure logging goes both to console and to file for this run
    setup_logging(log_file=out_text_file)

    # 2) Iterate over analyses and hierarchical levels
    for analysis in results_dict.keys():
        for level in results_dict[analysis].keys():

                # Only do the cortex level
                if level == "cortex" and "svm" in analysis:
                    logging.info(f"ANALYSIS: {analysis}, LEVEL: {level}")

                    # Slice the dictionary for (analysis, level)
                    sliced_dict = results_dict[analysis][level].copy()
                    if "categories" in sliced_dict:
                        del sliced_dict["categories"]


                    # 3) Filter or compute global min/max for consistent y-limits
                    _, global_max, measure_has_negative, max_error = filter_significant_ROIs(
                        sliced_dict, plot_corrected, P_ALPHA, FDR_ALPHA
                    )

                    # global_max = max_error + (0.1*max_error)
                    # global_min = -max_error if measure_has_negative else 0.0

                    global_max = .2 if "svm" in analysis else .5
                    global_min = -.05
                    # global_max = .2 if "svm" in analysis else .5
                    # global_min = 0

                    # Determine cortex order from predefined groups
                    ordered_regions = tuple(name for name, _ in CORTICES_GROUPS_CMAPS)

                    # 4) Output directory for saved figures
                    output_dir = os.path.join(output_root, "group", level)
                    os.makedirs(output_dir, exist_ok=True)

                    # 5) Plot separately for each expertise level
                    for expertise_bool in [True, False]:

                        # Build the “aggregator_data” that the new plot function needs:
                        # aggregator_data[hue_value][expertise_bool] = sub_df
                        # where hue_value is effectively your "regressor".
                        aggregator_data = {}
                        for regressor in sliced_dict:

                            # For safety, skip if this regressor doesn't have the requested expertise
                            if expertise_bool not in sliced_dict[regressor]:
                                continue

                            sub_df = sliced_dict[regressor][expertise_bool].copy()

                            # Rename the indices in sub_df using the mapping
                            sub_df = sub_df.rename(index=CORTICES_NAMES)

                            # If needed, reorder the sub_df to match your manager's region order
                            # so that bars come out in a sensible order on the x-axis:
                            sub_df = sub_df.reindex(ordered_regions, fill_value=np.nan)

                            aggregator_data[regressor] = sub_df

                        # Skip if no data found for that expertise
                        if not aggregator_data:
                            continue


                        for regressor in aggregator_data.keys():
                            # Construct a title for a single-regressor figure
                            fig_title = (
                                f"Regressor: {regressor} | "
                                f"Analysis: {analysis.capitalize()} | "
                                f"Level: {level.capitalize()} | "
                                f"Expertise: {'Expert' if expertise_bool else 'Novice'} | "
                                f"{corr_str}"
                            )

                            single_regressor_data = {
                                regressor: aggregator_data[regressor]
                            }
                            _ = shared_plot_mvpa_barplot(
                                data=single_regressor_data,
                                x_hue="target",
                                y_col="mean-chance",
                                x_groups="region",
                                title=fig_title,
                                x_groups_order=ordered_regions,
                                hue_order=[regressor],
                                out_dir=output_dir,
                                vmin=global_min,
                                vmax=global_max,
                                use_corrected_p=plot_corrected,
                            )

                        # Construct a title for the figure
                        fig_title = (
                            f"Analysis: {analysis.capitalize()} | "
                            f"Level: {level.capitalize()} | "
                            f"Expertise: {'Expert' if expertise_bool else 'Novice'} | "
                            f"{corr_str}"
                        )

                        # 6) Now call your specialized plot function
                        _ = shared_plot_mvpa_barplot(
                            data=aggregator_data,
                            x_hue="target",
                            y_col="mean-chance",
                            x_groups="region",
                            title=fig_title,
                            x_groups_order=ordered_regions,  # if you want to order ROI names
                            hue_order=list(aggregator_data.keys()),  # the different regressors
                            out_dir=output_dir,
                            vmin=global_min,
                            vmax=global_max,
                            use_corrected_p=plot_corrected,  # uses p_corrected or p_uncorrected
                        )

        logging.info("All plots generated successfully!")
    return output_root

def plot_mvpa_barplot(*args, **kwargs):
    """Thin wrapper that delegates MVPA bar plotting to `common.plotting_utils.plot_mvpa_barplot`.

    Use this from MVPA callers to keep a single source of truth for plotting.
    """
    from common.plotting_utils import plot_mvpa_barplot as _plot
    return _plot(*args, **kwargs)
