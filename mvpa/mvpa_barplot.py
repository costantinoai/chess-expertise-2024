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
import seaborn as sns

from modules.helpers import create_run_id


def format_contrast(s):
    """
    Convert underscore ("_") to space and "vs" to hyphen ("-"), then capitalize.
    Example: "experts_vs_nonexperts" -> "Experts - Nonexperts".
    """
    s = s.replace("_", " ")  # Replace underscores with spaces
    s = s.replace("vs", "-")  # Replace "vs" with "-"
    return " ".join(word.capitalize() for word in s.split())  # Capitalize each word


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
    (
        "/data/projects/chess/data/BIDS/derivatives/mvpa/20250402-191243_bilalic_sphere_rois",
        "/home/eik-tb/OneDrive_andreaivan.costantino@kuleuven.be/GitHub/chess-expertise-2024/chess-rois/results/bilalic_sphere_rois",
    ),
    (
        "/data/projects/chess/data/BIDS/derivatives/mvpa/20250402-191833_glasser_cortices_bilateral",
        "/home/eik-tb/OneDrive_andreaivan.costantino@kuleuven.be/GitHub/chess-expertise-2024/chess-rois/results/glasser_cortex_bilateral",
    ),
    (
        "/data/projects/chess/data/BIDS/derivatives/mvpa/20250402-230003_glasser_regions_bilateral",
        "/home/eik-tb/OneDrive_andreaivan.costantino@kuleuven.be/GitHub/chess-expertise-2024/chess-rois/results/glasser_regions_bilateral",
    ),
]

# You can optionally specify which regressors you want to keep; if empty, all are used
regressors_to_keep = []

# Reference chance level (e.g., 0 or 0.5). Tuple for easy extension if needed
chance_level = (0.0,)

# Whether to color the ROI label even if it's non-significant (Default: False)
plot_nonsig_labels = False

# Contrasts you want to plot
contrasts = ["experts_vs_nonexperts", "nonexperts_vs_chance", "experts_vs_chance"]

# Analyses (e.g., "svm" or "rsa_corr")
analyses = ["svm", "rsa_corr"]

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
            barplots_directory = os.path.join(contrast_dir, f"{create_run_id()}_bplots")
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

            # We gather the min and max from each regressor's confidence intervals
            for regressor in regressors:
                single_regressor_data = contrast_results_dict[regressor]
                temp_df = pd.DataFrame.from_dict(single_regressor_data, orient="index")

                # Extract lower/upper CI bounds
                temp_df["ci_low"] = temp_df["CI95"].apply(lambda x: x[0])
                temp_df["ci_high"] = temp_df["CI95"].apply(lambda x: x[1])

                # Fix infinite CI bounds if present
                temp_df["ci_low"] = np.where(
                    np.isinf(temp_df["ci_low"]), temp_df["mean_diff"], temp_df["ci_low"]
                )
                temp_df["ci_high"] = np.where(
                    np.isinf(temp_df["ci_high"]), temp_df["mean_diff"], temp_df["ci_high"]
                )

                # Compute the margin for error bar extension
                min_val = (
                    temp_df["mean_diff"]
                    - (temp_df["mean_diff"] - temp_df["ci_low"])
                    - 0.05
                ).min()
                max_val = (
                    temp_df["mean_diff"]
                    + (temp_df["ci_high"] - temp_df["mean_diff"])
                    + 0.05
                ).max()

                # Round for aesthetics
                y_min = (
                    0
                    if np.isnan(min_val) or np.isinf(min_val)
                    else math.floor(min_val * 1000) / 1000
                )
                y_max = (
                    0
                    if np.isnan(max_val) or np.isinf(max_val)
                    else math.ceil(max_val * 1000) / 1000
                )

                mins.append(y_min)
                maxs.append(y_max)

            # Determine global y-limits
            y_min = np.min(mins)
            y_max = np.max(maxs)

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

                # -------------------------------------------------------------
                # Build a custom color palette from the ROI column
                # -------------------------------------------------------------
                palette = df.set_index("roi")["color"].to_dict()

                # If any color is missing or NaN, use a fallback palette
                if any(pd.isna(color) or color is None for color in palette.values()):
                    palette = "RdPu"

                # -------------------------------------------------------------
                # Adjust figure size and font sizes
                # -------------------------------------------------------------
                bar_width = 0.5
                n_rois = len(ordered_rois)
                fig_width = max(
                    6, bar_width * n_rois
                )  # dynamic width based on number of ROIs

                base_fontsize = 10
                fontsize = min(14, max(6, fig_width * 0.4))  # clamp between 6 and 14

                fig, ax = plt.subplots(figsize=(fig_width, 8))

                # -------------------------------------------------------------
                # Create the barplot (without error bars)
                # -------------------------------------------------------------
                sns.barplot(
                    data=df,
                    x="roi",
                    y="mean_diff",
                    palette=palette,
                    order=ordered_rois,
                    errorbar=None,  # We'll add manual error bars below
                    ax=ax,
                    alpha=0.8,
                    zorder=1,
                )

                # -------------------------------------------------------------
                # Add manual error bars (confidence intervals)
                # -------------------------------------------------------------
                for idx, row in df.iterrows():
                    x_pos = ordered_rois.index(row["roi"])

                    # Calculate error bar lengths
                    lower_error_length = row["mean_diff"] - row["ci_low"]
                    upper_error_length = row["ci_high"] - row["mean_diff"]

                    # Handle infinity if present
                    if np.isinf(lower_error_length):
                        lower_error_length = upper_error_length
                    if np.isinf(upper_error_length):
                        upper_error_length = lower_error_length

                    ax.errorbar(
                        x=x_pos,
                        y=row["mean_diff"],
                        yerr=[[lower_error_length], [upper_error_length]],
                        color="black",
                        linewidth=1.5,
                        capsize=4,
                        zorder=3,
                    )

                # -------------------------------------------------------------
                # Annotate significance on bars (FDR-corrected p-values)
                # -------------------------------------------------------------
                def annotate_bar(row, x_position, axis):
                    """
                    Place asterisks above the bar if the FDR-corrected p-value
                    passes significance thresholds.
                    """
                    p_value = row["p_fdr"]
                    if p_value < 0.001:
                        significance = "***"
                    elif p_value < 0.01:
                        significance = "**"
                    elif p_value < 0.05:
                        significance = "*"
                    else:
                        significance = ""

                    if significance:
                        # Place the text slightly above the upper error bar
                        upper_err = (
                            row["ci_high"] - row["mean_diff"]
                            if not np.isinf(row["ci_high"])
                            else row["mean_diff"] - row["ci_low"]
                        )
                        y_pos = row["mean_diff"] + upper_err + 0.02
                        axis.text(
                            x_position,
                            y_pos,
                            significance,
                            ha="center",
                            va="bottom",
                            color="black",
                            fontsize=fontsize + 4,
                        )

                # Apply annotation for each bar
                for idx, row in df.iterrows():
                    x_position = ordered_rois.index(row["roi"])
                    annotate_bar(row, x_position, ax)

                # -------------------------------------------------------------
                # Set global y-limits across all regressors
                # -------------------------------------------------------------
                ax.set_ylim([y_min, y_max])

                # -------------------------------------------------------------
                # Draw a horizontal line at chance_level
                # (In this code, chance_level is a tuple but only uses index [0])
                # -------------------------------------------------------------
                ax.axhline(chance_level, color="black", linestyle="--", linewidth=1)

                # -------------------------------------------------------------
                # Final formatting of labels, title, spines, ticks
                # -------------------------------------------------------------
                ax.set_xlabel("")
                ax.set_ylabel(f"{measure_string}  Δ", fontsize=fontsize + 2)

                # Use a shorter descriptive string for the title
                title_short = regressor_mapping[regressor] + f" - {analysis}"
                ax.set_title(title_short, pad=30, fontsize=fontsize + 6)

                # Rotate x-axis labels
                ax.set_xticklabels(
                    ax.get_xticklabels(), rotation=30, ha="right", fontsize=fontsize + 2
                )

                # Clean up spines
                ax.spines["top"].set_visible(False)
                ax.spines["right"].set_visible(False)
                ax.spines["left"].set_linewidth(0.5)
                ax.spines["bottom"].set_linewidth(0.5)
                ax.spines["bottom"].set_visible(False)

                # Map each ROI to the color actually used for its bar
                bar_colors = {
                    roi: patch.get_facecolor()
                    for roi, patch in zip(ordered_rois, ax.patches)
                }

                # Optionally color ROI labels only if they're significant
                for label_obj in ax.get_xticklabels():
                    cortex_name = label_obj.get_text()

                    # Check if this ROI was rejected after FDR correction
                    roi_mask = df["roi"] == cortex_name
                    row = df.loc[roi_mask, "fdr_reject"]

                    if not row.empty and bool(row.iloc[0]):
                        this_color = bar_colors.get(cortex_name, "black")
                    else:
                        # If not significant and we're not plotting them, set to grey or white
                        this_color = "grey" if plot_nonsig_labels else "white"

                    label_obj.set_color(this_color)

                from matplotlib.ticker import MultipleLocator

                # Set y-ticks to 0.05 increments (change as needed)
                ax.yaxis.set_major_locator(MultipleLocator(0.05))

                ax.tick_params(axis="x", length=0)  # hide tick lines on x-axis
                ax.tick_params(axis="y", labelsize=fontsize + 2)

                plt.tight_layout()

                # -------------------------------------------------------------
                # Save the figure
                # -------------------------------------------------------------
                output_filename = os.path.join(barplots_directory, title + "_barplot.png")
                plt.savefig(output_filename)
                plt.show()
