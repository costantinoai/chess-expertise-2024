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
import glob
import numpy as np
import pickle
import pandas as pd
import matplotlib.pyplot as plt

from modules.helpers import create_run_id
from modules.text_utils import format_contrast

# Define base font size and update global rcParams
plt.rcParams.update({
    "font.family": "Ubuntu Condensed"
})

## moved to modules.text_utils.format_contrast


regressor_mapping = {
    "checkmate": "Checkmate vs. Non-checkmate boards",
    "stimuli_half": "Pairwise checkmate boards",
    "stimuli": "Pairwise all boards",
    "motif_half": "Motifs (Checkmate boards only)",
    "check_n_half": "Number of moves to checkmate (Checkmate boards only)",
    "side_half": "King position (L-R, Checkmate boards only)",
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
from config import MVPA_RESULTS_ROOT
MVPA_RESULTS_PATHS = [
    (
        os.environ.get("CHESS_MVPA_RESULTS_DIR", os.path.join(str(MVPA_RESULTS_ROOT), "<RUN>_glasser_cortices_bilateral")),
        os.environ.get("CHESS_ROIS_ANNOT_DIR", os.path.join("rois", "results", "glasser_cortex_bilateral")),
    ),
]

# You can optionally specify which regressors you want to keep; if empty, all are used
regressors_to_keep = []

# Reference chance level (e.g., 0 or 0.5). Tuple for easy extension if needed
chance_level = (0.0,)

# Whether to color the ROI label even if it's non-significant (Default: False)
plot_nonsig_labels = True

# Contrasts you want to plot
# contrasts = ["experts_vs_nonexperts", "nonexperts_vs_chance", "experts_vs_chance"]
contrasts = ["experts_vs_nonexperts"]

# Analyses (e.g., "svm" or "rsa_corr")
analyses = ["svm", "rsa_corr"]
# analyses = ["rsa_corr"]

# -----------------------------------------------------------------------------
# Helper Functions
# -----------------------------------------------------------------------------

def find_single_file(directory, pattern, file_description):
    """
    Search for a single file matching the given glob pattern in the specified directory.
    Raise a ValueError if zero or multiple files are found.

    Args:
        directory (str): Base directory to search in.
        pattern (str): Glob pattern (relative to `directory`) to match files.
        file_description (str): Description for error messages (e.g., "pickle", "TSV").

    Returns:
        str: Full path to the single matched file.
    """
    search_path = os.path.join(directory, pattern)
    matched_files = glob.glob(search_path)

    if len(matched_files) > 1:
        raise ValueError(f"More than one {file_description} file found in '{directory}'.")
    elif len(matched_files) == 0:
        raise ValueError(f"No {file_description} file found in '{directory}'.")

    return matched_files[0]


def load_roi_annotations(roi_annotation_path):
    """
    Load ROI annotation data from a single TSV file inside the given directory.

    Args:
        roi_annotation_path (str): Directory containing exactly one .tsv file.

    Returns:
        pandas.DataFrame: DataFrame of ROI annotations (expects columns 'region_name', 'color', and optionally 'order').
    """
    tsv_path = find_single_file(roi_annotation_path, "*.tsv", "TSV annotation")
    roi_df = pd.read_csv(tsv_path, sep="\t")
    return roi_df


def load_analysis_results(pickle_path):
    """
    Unpickle and return the analysis results dictionary.

    Args:
        pickle_path (str): Path to the .pkl file containing a dict of contrasts -> regressors -> ROI stats.

    Returns:
        dict: Nested dictionary from the pickled file.
    """
    with open(pickle_path, "rb") as f:
        return pickle.load(f)


def build_regressor_dataframe(regressor, analysis_results, roi_df, chance_value):
    """
    Construct a long-format DataFrame for a single regressor, combining novices and experts stats,
    mapping ROI colors and ordering.

    Args:
        regressor (str): Name of the regressor to process.
        analysis_results (dict): Nested dict with keys 'nonexperts_vs_chance', 'experts_vs_chance', 'experts_vs_nonexperts'.
        roi_df (pandas.DataFrame): Annotation DataFrame with columns ['region_name', 'color', 'order' (optional)].
        chance_value (float): Numeric chance level to add to 'mean_diff' values.

    Returns:
        pandas.DataFrame: DataFrame with columns [
            'roi', 'group', 'mean', 'ci_low', 'ci_high',
            'p_diff', 'sig_diff', 'color', 'roi_order', 'roi_label'
        ], sorted and ready for plotting.
    """
    # Extract per-ROI dictionaries
    novices_dict = analysis_results["nonexperts_vs_chance"][regressor]
    experts_dict = analysis_results["experts_vs_chance"][regressor]
    diff_dict    = analysis_results["experts_vs_nonexperts"][regressor]

    rows = []
    for roi_name in novices_dict.keys():
        # Novices: add chance to the stored "mean_diff" and CI95 bounds
        nov_mean     = chance_value + novices_dict[roi_name]["mean_diff"]
        nov_ci_low   = chance_value + novices_dict[roi_name]["CI95"][0]
        nov_ci_high  = chance_value + novices_dict[roi_name]["CI95"][1]

        # Experts: same procedure
        exp_mean     = chance_value + experts_dict[roi_name]["mean_diff"]
        exp_ci_low   = chance_value + experts_dict[roi_name]["CI95"][0]
        exp_ci_high  = chance_value + experts_dict[roi_name]["CI95"][1]

        # Experts vs Novices p-value & significance boolean (FDR-corrected)
        p_fdr        = diff_dict[roi_name]["p_fdr"]
        diff_signif  = diff_dict[roi_name]["fdr_reject"]

        # Append row for Novices
        rows.append({
            "roi":      roi_name,
            "group":    "Novices",
            "mean":     nov_mean,
            "ci_low":   nov_ci_low,
            "ci_high":  nov_ci_high,
            "p_diff":   p_fdr,
            "sig_diff": diff_signif,
        })
        # Append row for Experts
        rows.append({
            "roi":      roi_name,
            "group":    "Experts",
            "mean":     exp_mean,
            "ci_low":   exp_ci_low,
            "ci_high":  exp_ci_high,
            "p_diff":   p_fdr,
            "sig_diff": diff_signif,
        })

    df = pd.DataFrame(rows)

    # Map ROI name → color
    roi_color_map = dict(zip(roi_df["region_name"], roi_df["color"]))
    df["color"] = df["roi"].map(roi_color_map)

    # Determine ROI ordering: use 'order' if available, otherwise fallback to 'region_id'
    if "order" in roi_df.columns and not roi_df["order"].isna().all():
        roi_order_map = dict(zip(roi_df["region_name"], roi_df["order"]))
    else:
        roi_order_map = dict(zip(roi_df["region_name"], roi_df.get("region_id", roi_df.index + 1)))
    df["roi_order"] = df["roi"].map(roi_order_map)

    # Map numeric order → human-readable ROI label
    roi_name_map = {
        1: "Primary Visual",
        2: "Early Visual",
        3: "Dorsal Stream Visual",
        4: "Ventral Stream Visual",
        5: "MT+ Complex",
        6: "Somatosensory and Motor",
        7: "Paracentral Lobular and Mid Cing",
        8: "Premotor",
        9: "Posterior Opercular",
        10: "Early Auditory",
        11: "Auditory Association",
        12: "Insular and Frontal Opercular",
        13: "Medial Temporal",
        14: "Lateral Temporal",
        15: "Temporo-Parieto Occipital Junction",
        16: "Superior Parietal",
        17: "Inferior Parietal",
        18: "Posterior Cing",
        19: "Anterior Cing and Medial Prefrontal",
        20: "Orbital and Polar Frontal",
        21: "Inferior Frontal",
        22: "Dorsolateral Prefrontal"
    }
    df["roi_label"] = df["roi_order"].map(roi_name_map)

    # Sort by ROI order, then by group so that 'Experts' rows come before 'Novices'
    group_priority = {"Experts": 0, "Novices": 1}
    df["group_priority"] = df["group"].map(group_priority)
    df = df.sort_values(["roi_order", "group_priority"], ascending=True).reset_index(drop=True)

    # Ensure ROI labels are categorical and maintain their sorted order
    unique_labels = df.drop_duplicates("roi_label")["roi_label"].tolist()
    df["roi_label"] = pd.Categorical(df["roi_label"], categories=unique_labels, ordered=True)

    # Clip any infinite CI bounds to the mean (avoids plotting issues)
    df["ci_low"]  = np.where(np.isinf(df["ci_low"]), df["mean"], df["ci_low"])
    df["ci_high"] = np.where(np.isinf(df["ci_high"]), df["mean"], df["ci_high"])

    return df

def plot_expert_vs_novice(
    df,
    analysis_type,
    regressor_key,
    regressor_mapping,
    output_dir,
    chance_value,
    plot_nonsig_labels=False
):
    """
    Create and save a side-by-side bar plot comparing Experts vs Novices across ROIs,
    following a structured logic with configuration, data preparation, plotting, and styling.

    Args:
        df (pandas.DataFrame): DataFrame from build_regressor_dataframe(), containing columns:
            ['roi_label', 'group', 'mean', 'ci_low', 'ci_high', 'p_diff', 'sig_diff', 'color'].
        analysis_type (str): Identifier for analysis (e.g., "svm" or "rdm").
        regressor_key (str): Key for the current regressor (used for title lookup).
        regressor_mapping (dict): Maps regressor_key to a human-readable string.
        output_dir (str): Directory to save the resulting figure.
        chance_value (float): Numeric chance level to draw a horizontal reference line.
        plot_nonsig_labels (bool): If True, color non-significant ROI labels grey; otherwise keep them black.
    """
    # === CONFIGURATION ===
    fig_width = 21
    fig_height = 11
    tick_fontsize = 12*2
    label_fontsize = 14*2
    title_fontsize = 20*2
    asterisk_fontsize = 18*2.3

    # === MERGE STATS INTO A SUMMARY FOR SIDE-BY-SIDE PLOTTING ===
    # Our df has one row per ROI per group. We want one row per ROI containing:
    #   - roi_label (categorical)
    #   - exp_mean, exp_ci_low, exp_ci_high
    #   - nov_mean, nov_ci_low, nov_ci_high
    #   - p_diff, sig_diff (shared between experts and novices)
    #   - color (shared for both bars)
    # We'll build a new DataFrame named 'summary' with one row per unique ROI.

    roi_labels = df["roi_label"].cat.categories.tolist()
    summary_rows = []
    for roi in roi_labels:
        # Extract expert row
        exp_row = df[(df["roi_label"] == roi) & (df["group"] == "Experts")]
        # Extract novice row
        nov_row = df[(df["roi_label"] == roi) & (df["group"] == "Novices")]

        # Default values
        exp_mean = exp_ci_low = exp_ci_high = np.nan
        nov_mean = nov_ci_low = nov_ci_high = np.nan
        p_diff = None
        sig_diff = False
        color = "#CCCCCC"

        if not exp_row.empty:
            exp_mean    = exp_row["mean"].iloc[0]
            exp_ci_low  = exp_row["ci_low"].iloc[0]
            exp_ci_high = exp_row["ci_high"].iloc[0]
            p_diff      = exp_row["p_diff"].iloc[0]
            sig_diff    = bool(exp_row["sig_diff"].iloc[0])
            color       = exp_row["color"].iloc[0]
        if not nov_row.empty:
            nov_mean    = nov_row["mean"].iloc[0]
            nov_ci_low  = nov_row["ci_low"].iloc[0]
            nov_ci_high = nov_row["ci_high"].iloc[0]
            # (p_diff & sig_diff & color taken from expert row)

        summary_rows.append({
            "ROI_Label":   roi,
            "exp_mean":    exp_mean,
            "exp_ci_low":  exp_ci_low,
            "exp_ci_high": exp_ci_high,
            "nov_mean":    nov_mean,
            "nov_ci_low":  nov_ci_low,
            "nov_ci_high": nov_ci_high,
            "p_diff":      p_diff,
            "sig_diff":    sig_diff,
            "color":       color
        })

    summary = pd.DataFrame(summary_rows)
    # Derive a human-readable ROI name equal to the roi_label (since it's already descriptive)
    summary["ROI_Name"] = summary["ROI_Label"].astype(str)

    # === ORDER BY ROI LABEL (categorical order from df) ===
    # Since roi_labels was taken from the categorical order, summary is already in ROI order.
    # If a different ordering (e.g., by effect size) is desired, modify here.
    # For now, we keep the ROI order as-is.

    # === BUILD A COLOR PALETTE / DICTIONARY FOR BARS ===
    # summary["color"] already has the color for each ROI. We'll convert this to a dict:
    palette_dict = {row.ROI_Name: row.color for row in summary.itertuples(index=False)}

    # === PREPARE PLOTTING DATA ===
    x_coords = np.arange(len(summary))
    roi_names = summary["ROI_Name"].values

    # === CREATE FIGURE ===
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    # === BARPLOT: Experts and Novices side-by-side ===
    bar_width = 0.35
    # Plot expert bars to the left of each x-coordinate
    for i, row in summary.iterrows():
        # Coordinates for each bar
        x_center = x_coords[i]
        # Expert bar
        # Expert bar (70% opaque)
        ax.bar(
            x_center - bar_width/2,
            row.exp_mean,
            width=bar_width,
            color=row.color,
            edgecolor="black",
            linewidth=1,
            alpha=0.7,                 # ← set alpha here
            label="Experts" if i == 0 else ""
        )

        # Novice bar (70% opaque, with hatch)
        ax.bar(
            x_center + bar_width/2,
            row.nov_mean,
            width=bar_width,
            color=row.color,
            edgecolor="black",
            linewidth=1,
            hatch="//",
            alpha=0.7,                 # ← set alpha here
            label="Novices" if i == 0 else ""
        )


    # === ERROR BARS: 95% CI for Experts and Novices ===
    for i, row in summary.iterrows():
        x_center = x_coords[i]
        # Expert CI (correct for infinite bounds if needed)
        e_mean = row.exp_mean
        e_low  = row.exp_ci_low
        e_high = row.exp_ci_high
        if not np.isnan(e_low):
            # Fallback: if bound equals mean or infinite, expand symmetrically
            if np.isinf(e_low) or e_low == e_mean:
                e_low = e_mean - abs(e_high - e_mean)
            if np.isinf(e_high) or e_high == e_mean:
                e_high = e_mean + abs(e_low - e_mean)
            ax.errorbar(
                x=x_center - bar_width/2,
                y=e_mean,
                yerr=[[e_mean - e_low], [e_high - e_mean]],
                fmt='none',
                ecolor='black',
                elinewidth=1.2,
                capsize=0,
                capthick=0,
                zorder=2
            )
        # Novice CI
        n_mean = row.nov_mean
        n_low  = row.nov_ci_low
        n_high = row.nov_ci_high
        if not np.isnan(n_low):
            if np.isinf(n_low) or n_low == n_mean:
                n_low = n_mean - abs(n_high - n_mean)
            if np.isinf(n_high) or n_high == n_mean:
                n_high = n_mean + abs(n_low - n_mean)
            ax.errorbar(
                x=x_center + bar_width/2,
                y=n_mean,
                yerr=[[n_mean - n_low], [n_high - n_mean]],
                fmt='none',
                ecolor='black',
                elinewidth=1.2,
                capsize=0,
                capthick=0,
                zorder=2
            )

    # === SIGNIFICANCE ASTERISKS ===
    sig_offset = 0.05  # vertical offset above CI for star
    for i, row in summary.iterrows():
        if row.sig_diff and (not np.isnan(row.exp_ci_high) or not np.isnan(row.nov_ci_high)):
            # Place the asterisk above the higher of the two CIs
            top_of_cis = max(
                row.exp_ci_high if not np.isnan(row.exp_ci_high) else -np.inf,
                row.nov_ci_high if not np.isnan(row.nov_ci_high) else -np.inf
            )
            y_pos = top_of_cis + sig_offset
            ax.text(
                x_coords[i],
                y_pos,
                "*",
                ha="center",
                va="bottom",
                fontsize=asterisk_fontsize
            )
            ax.plot(
                [x_coords[i] - bar_width / 2, x_coords[i] + bar_width / 2],
                [y_pos, y_pos],
                color="black", linewidth=1.2
            )

    # === AXIS FORMAT ===
    ax.set_xticks(x_coords)
    ax.set_xticklabels(roi_names, rotation=30, ha="right", fontsize=tick_fontsize+5)
    ax.set_ylabel(f"{'Decoding accuracy - Chance' if analysis_type=='svm' else 'RDMs Correlation'} (with 95% CI)",
                  fontsize=label_fontsize)
    ax.set_xlabel("")  # No x-axis label
    human_readable = regressor_mapping.get(regressor_key, regressor_key)
    analysis_type_clean = "RSA" if "rsa" in analysis_type else "SVM Decoding"
    ax.set_title(
        f"Brain-Models {analysis_type_clean} - {human_readable}",
        fontsize=title_fontsize,
        pad=20
    )

    # Horizontal line at chance level
    ax.axhline(chance_value, color="black", linestyle="--", linewidth=1)

    # === SPINES & GRID ===
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.5)
    ax.spines["bottom"].set_visible(False)
    ax.spines["bottom"].set_linewidth(0.5)
    ax.tick_params(axis="y", labelsize=tick_fontsize)

    # === COLOR X-TICK LABELS BASED ON SIGNIFICANCE ===
    # Map roi_name -> bar color from palette_dict
    bar_colors = palette_dict
    for label_obj in ax.get_xticklabels():
        name = label_obj.get_text()
        is_sig = summary.loc[summary["ROI_Name"] == name, "sig_diff"].values[0]
        if not is_sig and plot_nonsig_labels:
            label_obj.set_color("lightgrey")
        else:
            label_obj.set_color(bar_colors.get(name, "black"))

    # === Y-LIMITS: Autoscale from bar heights, CI bounds, and asterisks ===
    # Compute bounds from summary table
    ci_lows = summary[["exp_ci_low", "nov_ci_low"]].values.flatten()
    ci_highs = summary[["exp_ci_high", "nov_ci_high"]].values.flatten()
    # Remove NaNs for calculation
    ci_lows = ci_lows[~np.isnan(ci_lows)]
    ci_highs = ci_highs[~np.isnan(ci_highs)]

    # === Y-LIMITS: Autoscale from bar heights, CI bounds, and asterisks ===
    y_min = -0.01

    if "rsa" in analysis_type and regressor_key == "side_half":
        y_max = 0.5
    elif "svm" in analysis_type and regressor_key == "side_half":
        y_max = 0.4
    else:
        y_max = 0.25  # default

    margin = 0.05 * (y_max - y_min)
    ax.set_ylim(y_min - margin, y_max + margin)

    ax.legend(
        loc="upper right",
        # bbox_to_anchor=(1, 1.15),
        ncol=2,  # optional: lay out legend entries in two columns
        fontsize=24,
        frameon=False
    )

    # === FINALIZE ===
    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)

    # Safe filename
    safe_title = f"Experts_vs_Novices_{analysis_type}_{human_readable}".replace(" ", "_").replace("/", "_")
    fname = os.path.join(output_dir, f"{safe_title}.png")
    plt.savefig(fname, dpi=300, bbox_inches="tight")
    plt.show()
    plt.close(fig)
    logging.info("Figure saved: %s", fname)

# -----------------------------------------------------------------------------
# Top-Level Script (No 'main' function)
# -----------------------------------------------------------------------------

# Assumed pre-defined variables in the environment:
# - analyses: list of analysis identifiers (e.g., ["svm", "rdm"])
# - MVPA_RESULTS_PATHS: list of tuples (analysis_results_path, roi_annotation_path)
# - contrasts: list of contrast keys (e.g., ["experts_vs_nonexperts", "nonexperts_vs_chance", "experts_vs_chance"])
# - regressors_to_keep: list of regressor names to include (empty list means keep all)
# - chance_level: list or tuple where first element is numeric chance value, e.g., [0.0]
# - regressor_mapping: dict mapping regressor keys to human-readable strings
# - plot_nonsig_labels: boolean flag; if True, non-significant ROI labels are colored grey/white
# - create_run_id: function that returns a unique run identifier string

for analysis in analyses:
    # -------------------------------------------------------------------------
    # Loop over each (group results, ROI labels) directory pair
    # -------------------------------------------------------------------------
    for analysis_results_path, roi_annotation_path in MVPA_RESULTS_PATHS:
        # 1) Find the single pickle file in the group results directory
        try:
            pickle_rel_pattern = os.path.join(analysis, "*group", "*.pkl")
            analysis_results_pickle = find_single_file(
                analysis_results_path,
                pickle_rel_pattern,
                "pickle"
            )
        except ValueError as e:
            raise ValueError(f"Error for analysis '{analysis}' in '{analysis_results_path}': {e}")

        # Derive group_path as the directory containing the chosen pickle
        group_path = os.path.dirname(analysis_results_pickle)

        # 2) Load the ROI annotation DataFrame
        roi_df = load_roi_annotations(roi_annotation_path)

        # 3) Load the pickled analysis results dictionary
        analysis_results = load_analysis_results(analysis_results_pickle)

        # ---------------------------------------------------------------------
        # Determine which regressors exist across all contrasts
        # ---------------------------------------------------------------------
        all_regressors = sorted({
            reg
            for contrast_dict in analysis_results.values()
            for reg in contrast_dict.keys()
        })
        if regressors_to_keep:
            # If user specified a subset, filter accordingly
            regressors = [r for r in all_regressors if r in regressors_to_keep]
        else:
            regressors = all_regressors

        # ---------------------------------------------------------------------
        # Prepare output directory for stacked barplots under 'experts_vs_nonexperts'
        # ---------------------------------------------------------------------
        run_id = create_run_id()
        stacked_bplots_dir = os.path.join(
            group_path,
            "experts_vs_nonexperts",
            f"{run_id}_stacked_bplots"
        )
        os.makedirs(stacked_bplots_dir, exist_ok=True)

        # ---------------------------------------------------------------------
        # Iterate over each regressor and generate a barplot if all contrasts exist
        # ---------------------------------------------------------------------
        for reg in regressors:
            required_keys = [
                "nonexperts_vs_chance",
                "experts_vs_chance",
                "experts_vs_nonexperts"
            ]
            # Skip if any required contrast is missing for this regressor
            if not all(
                (key in analysis_results and reg in analysis_results[key])
                for key in required_keys
            ):
                logging.warning("Skipping regressor '%s'—one or more contrasts missing.", reg)
                continue

            # Build a DataFrame combining Novices, Experts, and difference stats
            df_reg = build_regressor_dataframe(
                regressor=reg,
                analysis_results=analysis_results,
                roi_df=roi_df,
                chance_value=chance_level[0]
            )

            # Create and save the Experts vs Novices plot
            plot_expert_vs_novice(
                df=df_reg,
                analysis_type=analysis,
                regressor_key=reg,
                regressor_mapping=regressor_mapping,
                output_dir=stacked_bplots_dir,
                chance_value=chance_level[0],
                plot_nonsig_labels=plot_nonsig_labels
            )
import logging
