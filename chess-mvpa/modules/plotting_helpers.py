#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 21 17:59:03 2025

@author: costantino_ai
"""

import os
import numpy as np
import pickle
import pandas as pd
import seaborn as sns
from modules.helpers import OutputLogger
# from modules.roi_manager import ROIManager
from modules import plt,logging
from modules import (
    ROIS_CSV,
    LEFT_LUT,
    RIGHT_LUT,
    CONTRAST_MAP,
    MANAGER,
    FDR_ALPHA,
    P_ALPHA,
    CORTICES_NAMES
)
from modules.helpers import filter_significant_ROIs


def plot_group_level_mvpa(ttest_results_path,
                          ROIS_CSV=ROIS_CSV,
                          LEFT_LUT=LEFT_LUT,
                          RIGHT_LUT=RIGHT_LUT,
                          CONTRAST_MAP=CONTRAST_MAP,
                          plot_corrected=True):
    """
    Plot MVPA group-level results across different levels and analyses, using
    the new dictionary-of-dataframes structure. One figure per (analysis, level, expertise).

    Parameters:
        ttest_results_path (str): Path to the t-test results pickle file.
        ROIS_CSV (str): Path to the ROI CSV file.
        LEFT_LUT (str): Path to the left hemisphere lookup table.
        RIGHT_LUT (str): Path to the right hemisphere lookup table.
        CONTRAST_MAP (dict): Dictionary mapping contrasts to chance levels.
        plot_corrected (bool): If True, use the FDR-corrected p-values for star annotations.

    Returns:
        None
    """

    corr_str = (
        f"FDR corrected (p<{FDR_ALPHA})"
        if plot_corrected
        else f"Uncorrected (p<{P_ALPHA})"
    )

    # 1) Load your dictionary of results from disk
    with open(ttest_results_path, "rb") as f:
        results_dict = pickle.load(f)

    output_root = os.path.dirname(ttest_results_path)
    os.makedirs(output_root, exist_ok=True)
    out_text_file = os.path.join(output_root, 'mvpa_logs_barplots.log')

    with OutputLogger(True, out_text_file):

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

                    # get x-groups order from manager
                    ordered_regions = MANAGER.get_ordered_cortices_names()

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
                            # Construct a title for the figure
                            fig_title = (
                                f"Regressor: {regressor} | "
                                f"Analysis: {analysis.capitalize()} | "
                                f"Level: {level.capitalize()} | "
                                f"Expertise: {'Expert' if expertise_bool else 'Novice'} | "
                                f"{corr_str}"
                            )

                            single_regressor_data = {key: value for key, value in aggregator_data.items() if key == regressor}
                            _ = plot_mvpa_barplot(
                                data=single_regressor_data,
                                x_hue="target",
                                y_col="mean-chance",
                                x_groups="region",
                                title=fig_title,
                                x_groups_order=ordered_regions,  # if you want to order ROI names
                                hue_order=list(single_regressor_data.keys()),  # the different regressors
                                out_dir=output_dir,
                                vmin=global_min,
                                vmax=global_max,
                                use_corrected_p=plot_corrected,  # uses p_corrected or p_uncorrected
                            )

                        # Construct a title for the figure
                        fig_title = (
                            f"Analysis: {analysis.capitalize()} | "
                            f"Level: {level.capitalize()} | "
                            f"Expertise: {'Expert' if expertise_bool else 'Novice'} | "
                            f"{corr_str}"
                        )

                        # 6) Now call your specialized plot function
                        _ = plot_mvpa_barplot(
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

        print("All plots generated successfully!")
    return output_root

def plot_mvpa_barplot(
    data,
    x_hue,
    y_col,
    x_groups=None,
    chance_level=0.0,
    title="",
    hue_order=None,
    x_groups_order=None,
    out_dir=None,
    plot_points=False,
    use_corrected_p=True,
    vmin=None,
    vmax=None

):
    """
    Plot MVPA bar plot with statistical annotations.

    Parameters:
    - data (pd.DataFrame): Input data for plotting.
    - x_hue (str): Column for hue categories (e.g., true labels), shown in the legend.
    - y_col (str): Column for y-axis values (e.g., accuracy).
    - x_groups (str or None): Column for x-axis groups (stacked bars), or None if ungrouped.
    - chance_level (float): Reference chance level for significance testing.
    - title (str): Title of the plot.
    - hue_order (list or None): Order of hue categories for plotting and operations.
    - x_groups_order (list or None): Order of x-axis groups for plotting.

    Returns:
    - stats_results (list): A list of dictionaries with t-test results for each category or group.
    """

    def _build_long_dataframe(data):
        """
        Build a long-form DataFrame from the nested dictionary structure:
            data[hue][expertise_bool] = pd.DataFrame
        The 'target' column is used for the x-axis.

        Parameters
        ----------
        data : dict
            {hue_value -> {True/False -> DataFrame}}
        hue_order : list or None
            The order of the hue_value keys. If None, uses list(data.keys()).
        x_groups_order : list or None
            The order of the targets (ROI labels). If None, determined from the data.

        Returns
        -------
        df_sorted : pd.DataFrame
            Long-form DataFrame with columns:
              ["hue", "expertise", "target", "mean-chance", "ci_low", "ci_high",
               "p_uncorrected", "p_corrected"] (+ any others present in the sub-DataFrame).
            Rows are sorted by x_groups_order if provided.
        """
        # Extract and concatenate all DataFrames from the dictionary
        df_list = []
        for key, df in data.items():
            df_list.append(df)

        # Concatenate all DataFrames to merge them
        df_all = pd.concat(df_list, axis=0)

        # Ensure the index is unique or use explicit column
        df_sorted = df_all.reset_index()  # Converts index into a regular column
        if "region" not in df_sorted.columns:
            df_sorted.rename(columns={"index": "region"}, inplace=True)  # Rename for clarity

        return df_sorted

    def _annotate_bar(stats_result, line_center, _ax, use_corrected_p):
        """
        Annotate bars with significance markers based on t-test results.

        Parameters:
        - stats_results (list): List of t-test results.
        - line_center (list): List of tuples (line_x, line_y, line_lower, line_upper, group).
        - _ax (matplotlib.axes.Axes): Axis object for the plot.
        """
        chosen_p="p_corrected" if use_corrected_p == True else "p_corrected"

        try:
            line_x, line_y, group, facecolor = line_center
        except:
            return

        p_value = stats_result[chosen_p]
        ci_low = stats_result["ci_low"]
        ci_high = stats_result["ci_high"]
        mean = stats_result["mean-chance"]

        yerr = mean-ci_low

        assert mean == line_y

        # Plot error bars
        _ax.errorbar(
            line_x,
            line_y,
            yerr=yerr,
            color="black",
            linewidth=1.5,
            capsize=4,
            zorder=2
        )

        significance = (
            "***"
            if p_value < 0.001
            else "**" if p_value < 0.01 else "*" if p_value < 0.05 else None
        )
        if significance:
            top = (mean+yerr) + ((mean+yerr)*0.05)
            _ax.text(line_x, top, significance, ha="center", color="black")

    def get_bars(_ax):
        # Filter out bars with height or width equal to 0 (e.g., legend placeholders)
        bars = [
            bar for bar in _ax.patches if bar.get_width() != 0
        ]
        return bars

    def find_xmajortick_for_bar(_ax, bar):
        """
        Find the x-major tick label corresponding to a given bar element.

        Parameters:
        - ax: matplotlib Axes object containing the bar chart.
        - bar: The bar element (matplotlib Rectangle object).

        Returns:
        - The x-major tick label (string) if found, otherwise None.
        """
        # Calculate the bar's x-center
        bar_center = bar.get_center()[0]

        # Get x-tick positions and labels
        x_ticks = _ax.get_xticks()
        x_tick_labels = [tick.get_text() for tick in _ax.get_xticklabels()]

        # Find the index of the closest x-tick
        closest_index = np.argmin([abs(bar_center - tick) for tick in x_ticks])

        return x_tick_labels[closest_index]

    def get_bars_coordinates(_ax):
        """
        Calculate bar details for annotation, ensuring alignment with x_groups_order and hue_order.

        Parameters:
        - ax (matplotlib.axes.Axes): Axis object containing the bars.

        Returns:
        - List of tuples (bar_x, bar_y, ci_upper, ci_lower, group_tuple) for each bar:
          bar_x: x-coordinate of the bar center
          bar_y: y-coordinate (height) of the bar
          ci_upper: upper limit of the confidence interval
          ci_lower: lower limit of the confidence interval
          group_tuple: tuple of (hue, group) or (hue, None) if ungrouped
        """

        # Filter out bars with height or width equal to 0 (e.g., legend placeholders)
        bars = get_bars(_ax)

        # Get hues (categories) from the legend
        hues = tuple(
            [label.get_text() for label in _ax.get_legend().get_texts()]
            if _ax.get_legend()
            else [label.get_text() for label in _ax.get_xmajorticklabels()]
        )

        # Calculate the number of bars per hue
        bars_per_hue = len(bars) // len(hues)

        # Create a list of tuples: (x, y, y_upper, y_lower, index)
        bar_tuples = [
            (
                bars[i].get_x()
                + bars[i].get_width() / 2,  # x-coordinate (center of the bar)
                bars[i].get_height(),  # y-coordinate (height of the bar)
                (
                    find_xmajortick_for_bar(_ax, bars[i]),  # x_major tick (ROI)
                    hues[i // bars_per_hue],  # Hue assignment based on index
                ),
                bars[i].get_facecolor(),
            )
            for i in range(len(bars))
        ]

        return bar_tuples

    # Determine figure size dynamically based on the number of x_groups
    figsize = plt.rcParams["figure.figsize"]
    fig, ax = plt.subplots(figsize=(figsize[0]*3, figsize[1]*1.5))

    # Plot the bars with seaborn
    x_group_col = x_groups if x_groups else x_hue
    x_groups_order = x_groups_order if x_groups else hue_order

    sns.barplot(
        data=_build_long_dataframe(data),
        x=x_group_col,
        y=y_col,
        hue=x_hue,
        palette="colorblind",
        errorbar=None,
        err_kws={"linewidth": 2},
        capsize=0.2,
        ax=ax,
        order=x_groups_order,
        hue_order=hue_order,
        alpha=.2 if plot_points else .9,
        zorder=1
    )

    # Get coordinates for bars and error bars
    bar_tuples = get_bars_coordinates(ax)

    # Define the order of bars based on provided arguments
    bars_order = (
        [(x, h) for x in x_groups_order for h in hue_order ]
        if x_groups_order != hue_order
        else [(h, h) for h in hue_order]
    )

    if plot_points:
        # Add individual data points
        sns.stripplot(
            data=data,
            x=x_group_col,
            y=y_col,
            hue=x_hue,
            dodge=True if x_groups else False,  # Aligns points based on hue
            palette="colorblind",
            order=x_groups_order,
            hue_order=hue_order,
            alpha=0.5,  # Make points slightly transparent
            ax=ax,
            jitter=True,  # Adds some random noise to spread the points
            size=15,
            zorder=-1
        )

    if x_group_col is None:
        x_group_col = x_hue

    # Annotate bars
    for x_region, hue_regressor in bars_order:

        stats_result_group = data[hue_regressor].loc[x_region]

        # Find the corresponding bar in the plot
        bar_tuple = next(
            (bar_tuple for bar_tuple in bar_tuples if bar_tuple[-2] == (x_region, hue_regressor)),
            None,
        )

        logging.debug(bar_tuple)

        # Annotate the bar with the relevant stats
        _annotate_bar(stats_result_group, bar_tuple, ax, use_corrected_p)

    if plot_points:
        # Avoid duplicate legend entries
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles[:len(hue_order)], labels[:len(hue_order)])

    # Add chance level reference line
    x_measure_str = "Accuracy-Chance" if "Svm" in title else "Coefficient"

    ax.axhline(chance_level, color="black", linestyle="--", linewidth=1)
    ax.set_xlabel("Regions of Interest")
    ax.set_ylabel(x_measure_str)
    ax.set_title(title, pad=30)
    ax.set_ylim((vmin,vmax))
    ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.5)
    ax.spines["bottom"].set_linewidth(0.5)

    for label_obj in ax.get_xticklabels():
        cortex_name = label_obj.get_text()
        if len(hue_order) > 1:
            this_color = MANAGER.get_by_filter(cortex=cortex_name)[0][0].cortex_color
        else:
            this_color = "black"
        label_obj.set_color(this_color)

    # Define the groups and corresponding colors for the custom legend
    groups = [
        ("Early Visual", "#a6cee3"),
        ("Intermediate Visual", "#1f78b4"),
        ("Sensorimotor", "#b2df8a"),
        ("Auditory", "#33a02c"),
        ("Temporal", "#fb9a99"),
        ("Posterior", "#e31a1c"),
        ("Anterior", "#fdbf6f")
    ]

    # Remove the default legend if hue_order has only one element.
    if isinstance(hue_order, list) and len(hue_order) == 1:
        palette = [MANAGER.get_by_filter(cortex=cortex_name)[0][0].cortex_color for cortex_name in data[list(data.keys())[0]].index]

        # Now override the colors of the bars with your own palette.
        # This assumes that the number of bars equals the length of your my_palette.
        for i, bar in enumerate(ax.patches):
            # You might need to adjust the indexing if you have nested bars (i.e., dodge=True)
            bar.set_facecolor(palette[i % len(palette)])

        ax.legend_.remove()  # Remove the Seaborn default legend

        # Legend placement variables
        legend_x = 1.02  # Position outside the plot (right side)
        legend_y = 0.8   # Center the legend vertically
        box_height = 0.04  # Height of each color box (scaled for the plot size)
        text_offset = 0.05  # Offset between box and text

        # Create custom legend inside the same figure
        for i, (group, color) in enumerate(groups):
            y_pos = legend_y - (i - len(groups) / 2) * box_height * 1.5  # Adjust vertical position

            # Add colored rectangles
            ax.add_patch(plt.Rectangle((legend_x, y_pos), 0.03, box_height, color=color,
                                       transform=ax.transAxes, clip_on=False))

            # Add text labels
            ax.text(legend_x + text_offset, y_pos + box_height / 2, group, fontsize=20,
                    verticalalignment='center', transform=ax.transAxes)

    # Show the plot
    plt.tight_layout()

    if out_dir is not None:
        # Save the plot
        filename = title.replace(" ", "_").lower() + ".png"
        plt.savefig(os.path.join(out_dir, filename), dpi=300, bbox_inches="tight")

    plt.show()

    return out_dir
