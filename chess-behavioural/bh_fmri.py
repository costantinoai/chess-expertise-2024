#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Refactored Chess Expertise Analysis Script with Modular Functions (Enhanced Plot Formatting + Inline Comments)

This version:
- Ensures every line has an inline comment.
- Adds multiprocessing support when MULTIPROCESS flag is set to True.
"""

import os                                                      # Provides functions for interacting with the operating system
import glob                                                    # For file pattern matching (finding .mat files)
import warnings                                                # To suppress or handle warnings
import numpy as np                                             # Numerical computing library
import pandas as pd                                            # DataFrame library for tabular data
import matplotlib as mpl                                        # Core Matplotlib library for plotting
import matplotlib.pyplot as plt                                # Pyplot API for Matplotlib
import seaborn as sns                                          # Statistical plotting library built on Matplotlib
from matplotlib.patches import Rectangle                       # For drawing rectangles on plots
from matplotlib.colors import LinearSegmentedColormap          # To create custom colormaps
import pingouin as pg                                          # Statistical package (used here for bootstrapped correlations)
import mat73                                                   # For loading MATLAB v7.3 .mat files
import scipy.io                                                # For loading MATLAB < v7.3 .mat files
from sklearn.manifold import MDS                               # Multi-dimensional scaling algorithm
from modules.helpers import create_run_id                       # Helper function to create a unique run identifier

import multiprocessing                                         # Python's multiprocessing module for parallel execution


# ----------------------------------------
# GLOBAL SETTINGS
# ----------------------------------------

SF = 1.0                                                      # Scaling factor for all font sizes
FONT_FAMILY = 'Ubuntu'                                        # Global font family
TITLE_FS = 32 * SF                                            # Font size for plot titles
LABEL_FS = 26 * SF                                            # Font size for axis labels
TICK_FS = 20 * SF                                             # Font size for tick labels
COLORBAR_LABEL_FS = 22 * SF                                   # Font size for colorbar labels
COLORBAR_TICK_FS = 18 * SF                                    # Font size for colorbar tick labels

FIGSIZE = (10, 8)                                             # Default figure size (width, height) for all plots

mpl.rcParams['font.family'] = FONT_FAMILY                     # Apply global font family for Matplotlib
mpl.rcParams['font.size'] = TICK_FS                           # Apply default font size for most text
sns.set_style("white")                                        # Use Seaborn's white style for all plots

warnings.filterwarnings("ignore", category=FutureWarning)     # Suppress FutureWarnings
warnings.filterwarnings("ignore", category=UserWarning)       # Suppress UserWarnings

OUT_ROOT = f"./results/{create_run_id()}_bh_fmri"             # Directory to save all output figures
os.makedirs(OUT_ROOT, exist_ok=True)                          # Create the directory if it doesn't already exist

COL_GREEN = '#006400'                                         # Dark green color used for first strategy group
COL_RED = '#8B0000'                                           # Dark red color used for second strategy group

center_color = plt.cm.RdPu(0)[:3]                             # Center color from the RdPu colormap (RGB tuple)
neg_colors = np.linspace([0.0, 0.5, 0.7], center_color, 256)  # Interpolate from dark blue to center color
pos_colors = plt.cm.RdPu(np.linspace(0, 1, 256))[:, :3]       # Interpolate from center color to dark RdPu
CUSTOM_CMAP = LinearSegmentedColormap.from_list(               # Create a custom diverging colormap
    "centered_rdpu_blue",                                      # Name of the new colormap
    np.vstack((neg_colors, pos_colors))                        # Vertically stack negative and positive segments
)

MULTIPROCESS = True                                            # Flag to enable multiprocessing for subject processing


# ----------------------------------------
# STRATEGY COLOR & ALPHA MAPPING
# ----------------------------------------

STRATEGIES = [                                                # Sequence of strategy labels for each stimulus/trial
    1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3,
    4, 4, 4, 5, 5, 5,
    1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3,
    4, 4, 4, 5, 5, 5,
]

def compute_strategy_colors_alphas(strategies):                # Given a list of strategies, return colors & alpha transparencies
    """
    Assign a consistent color (first 5 unique → green, next 5 → red) and increasing alpha transparency
    (0.2 → 1.0) for each block of identical strategy labels in the input list.
    Returns two lists (colors, alphas) of length = len(strategies).
    """
    current = None                                             # Track the current strategy label
    colors = []                                                # List to store assigned colors
    alphas = []                                                # List to store assigned alpha transparencies
    color_idx = 0                                              # Index to count unique strategy blocks

    for strat in strategies:                                   # Iterate through each strategy label in sequence
        if strat != current:                                   # If this is a new strategy block
            if color_idx < 5:                                  # If within the first 5 unique blocks
                color = COL_GREEN                               # Assign dark green
                alpha = (color_idx + 1) / 5.0                   # Alpha increments: 0.2, 0.4, ..., 1.0
            else:                                               # For the next 5 unique blocks
                color = COL_RED                                 # Assign dark red
                alpha = (color_idx + 1 - 5) / 5.0               # Alpha increments: 0.2, 0.4, ..., 1.0
            current = strat                                    # Update current strategy label
            color_idx += 1                                      # Increment unique-block counter

        colors.append(color)                                    # Append chosen color for this position
        alphas.append(alpha)                                    # Append chosen alpha for this position

    return colors, alphas                                       # Return lists of colors and alphas

STRAT_COLORS, STRAT_ALPHAS = compute_strategy_colors_alphas(STRATEGIES)  # Precompute for global use


# ----------------------------------------
# PLOTTING FUNCTIONS
# ----------------------------------------

def plot_shared_colorbar(cmap, vmin=-18, vmax=18):             # Display a standalone colorbar for a diverging colormap
    """
    Display a standalone colorbar for a diverging colormap (range vmin→vmax),
    with ticks and label on the left side.
    Useful when individual heatmaps omit their own colorbars.
    """
    fig, ax = plt.subplots(figsize=FIGSIZE, facecolor='white')  # Create a new figure and axis
    fig.subplots_adjust(left=0.3)                                # Leave space on the left for colorbar

    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)            # Normalize data range for color mapping
    sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)             # Create a ScalarMappable for the colorbar
    sm.set_array([])                                             # Dummy array (required)

    # Add colorbar to the left
    cbar = fig.colorbar(
        sm, ax=ax, fraction=0.05, pad=0.02,
        location="left"
    )
    cbar.set_label("Dissimilarity Value", fontsize=COLORBAR_LABEL_FS)   # Set label text
    cbar.ax.tick_params(labelsize=COLORBAR_TICK_FS)                     # Set tick label size

    ax.set_axis_off()                                            # Hide dummy axis
    save_path = os.path.join(OUT_ROOT, "symmetric_colorbar.png") # Output path
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0.1, facecolor='white')  # Save to file
    plt.show()                                                   # Display figure


def plot_rdm_heatmap(rdm, bold_title, expertise_label, colormap="RdPu", vmin=0, vmax=18):
    """
    Plot a representational dissimilarity matrix (RDM) as a square heatmap without inline colorbar.
    Adds colored rectangles along axes to indicate strategy grouping.
    Title is two lines: bold_title (bold) on line 1, expertise_label on line 2 (normal).
    """
    fig, ax = plt.subplots(figsize=FIGSIZE, facecolor='white') # Create figure and axis with specified size

    sns.heatmap(                                               # Plot heatmap of the RDM
        rdm,
        annot=False,                                          # No annotations in cells
        fmt="d",                                              # Integer format
        cmap=colormap,                                        # Use specified colormap
        vmin=vmin,                                            # Minimum data value for normalization
        vmax=vmax,                                            # Maximum data value for normalization
        cbar=False,                                           # Do not plot inline colorbar
        ax=ax,                                                # Plot on this axis
        square=True                                           # Force square cells
    )
    ax.set_aspect('equal')                                     # Ensure equal aspect ratio (square plot)

    # Identify where strategy label changes to draw colored bars
    ticks = []                                                  # List to store indices where strategy changes
    prev = None                                                 # Track previous strategy
    for i, lab in enumerate(STRATEGIES):                        # Enumerate through STRATEGIES
        if lab != prev:                                         # If this strategy differs from previous
            ticks.append(i)                                     # Record the index
            prev = lab                                          # Update previous strategy
    ticks.append(len(STRATEGIES))                               # Append end index for final block

    for idx, start in enumerate(ticks[:-1]):                     # Iterate through strategy blocks
        end = ticks[idx + 1]                                    # Compute end index of this block
        width = end - start                                     # Width of the block
        color = STRAT_COLORS[start]                             # Color assigned to this block
        alpha = STRAT_ALPHAS[start]                             # Alpha transparency assigned to this block
        rect_x = Rectangle(                                     # Rectangle along x-axis (bottom)
            (start, -0.01), width, -0.0005 * len(rdm),
            color=color, alpha=alpha, ec=None,
            transform=ax.get_xaxis_transform(),                 # Coordinate transform for x-axis
            clip_on=False                                       # Draw outside plot area
        )
        rect_y = Rectangle(                                     # Rectangle along y-axis (left)
            (-0.01, start), -0.0005 * len(rdm), width,
            color=color, alpha=alpha, ec=None,
            transform=ax.get_yaxis_transform(),                 # Coordinate transform for y-axis
            clip_on=False                                       # Draw outside plot area
        )
        ax.add_patch(rect_x)                                     # Add x-axis rectangle to plot
        ax.add_patch(rect_y)                                     # Add y-axis rectangle to plot

    # Build two-line title: first line bold, second line normal
    title_text = f"{bold_title}\n{expertise_label}"
    #title_text = f"{bold_title}"
    ax.set_title(title_text, fontsize=TITLE_FS, pad=25)         # Set title with padding for two lines

    ax.set_xticks([])                                           # Remove x-axis ticks
    ax.set_yticks([])                                           # Remove y-axis ticks

    save_path = os.path.join(OUT_ROOT, bold_title.replace(" ", "_") + f"_{expertise_label}.png")  # Construct filename
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0.1, facecolor='white')  # Save figure to file
    plt.show()                                                  # Display the plot


def plot_mds(dissimilarity_matrix, bold_title, expertise_label):
    """
    Perform Multidimensional Scaling (MDS) on a precomputed dissimilarity matrix and plot the 2D embedding.
    Points are colored/translucent according to global STRAT_COLORS & STRAT_ALPHAS.
    Title is two lines: bold_title (bold) on line 1, expertise_label on line 2 (normal).
    """
    mds = MDS(n_components=2, dissimilarity="precomputed", random_state=42)  # Initialize MDS model
    coords = mds.fit_transform(dissimilarity_matrix)                         # Compute 2D embedding

    fig, ax = plt.subplots(figsize=FIGSIZE, facecolor='white')               # Create figure and axis
    for x, y, color in zip(coords[:, 0], coords[:, 1], STRAT_COLORS):
        ax.scatter(x, y, color=color, marker='o', alpha=0.7, s=200)         # Scatter points with color & alpha

    title_text = f"{bold_title}\n{expertise_label}"
    # Two-line title text
    ax.set_title(title_text, fontsize=TITLE_FS, pad=25)                        # Set plot title
    ax.set_xticks([])                                                          # Remove x-axis ticks
    ax.set_yticks([])                                                          # Remove y-axis ticks
    for spine in ax.spines.values():                                           # Iterate through plot spines
        spine.set_edgecolor('black')                                            # Set spine edge color
        spine.set_linewidth(1.5)       # Set spine line width

    save_path = os.path.join(OUT_ROOT, bold_title.replace(" ", "_") + f"_{expertise_label}.png")  # Filename
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0.1, facecolor='white')  # Save figure
    plt.show()                                                                # Display plot


def plot_choice_frequency(data, expertise_label):
    """
    Create a bar plot showing how often each stimulus (by ID) was chosen in pairwise comparisons.
    Bars are colored/translucent according to global STRAT_COLORS & STRAT_ALPHAS.
    Title is two lines: "Stimulus Selection Frequency" (bold) and expertise_label (normal).
    """
    freq = data['better'].value_counts().sort_index()                          # Compute frequency of 'better' stimuli

    fig, ax = plt.subplots(figsize=FIGSIZE, facecolor='white')                 # Create figure and axis
    bar_plot = sns.barplot(                                                     # Plot bar chart
        x=freq.index,                                                           # X-axis stimulus IDs
        y=freq.values,                                                          # Y-axis selection counts
        palette=STRAT_COLORS[:len(freq)],                                       # Use precomputed colors
        ax=ax                                                                   # Plot on this axis
    )
    for bar, alpha in zip(bar_plot.patches, STRAT_ALPHAS[:len(freq)]):          # Iterate through bars
        bar.set_alpha(alpha)                                                    # Set transparency for each bar

    bold_line = f"Stimulus Selection Frequency\n{expertise_label}"                                  # First line for title (bold)
    title_text = f"{bold_line}"
    # Two-line title
    ax.set_title(title_text, fontsize=TITLE_FS, pad=25)                         # Set title with padding

    ax.set_xlabel('Stimulus ID', fontsize=LABEL_FS)                             # Set x-axis label
    ax.set_ylabel('Selection Count', fontsize=LABEL_FS)                         # Set y-axis label
    ax.set_xticks([])                                                           # Remove x-axis ticks
    ax.tick_params(labelsize=TICK_FS)                                           # Set tick label size

    for spine_loc in ["left", "bottom"]:                                        # Show only left and bottom spines
        ax.spines[spine_loc].set_visible(True)                                   # Make spine visible
        ax.spines[spine_loc].set_linewidth(1.5)                                  # Set spine line width
        ax.spines[spine_loc].set_edgecolor("black")                              # Set spine color
    for spine_loc in ["top", "right"]:                                           # Hide top and right spines
        ax.spines[spine_loc].set_visible(False)                                  # Make spine invisible

    save_path = os.path.join(OUT_ROOT, f"Stimulus_Selection_Frequency_{expertise_label}.png")  # Filename
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0.1, facecolor='white')  # Save figure
    plt.show()                                                                  # Display plot


def plot_model_behavior_correlations(expert_results, novice_results, column_labels):
    """
    Generate a side-by-side barplot of bootstrapped Pearson correlations (with 95% CI)
    for Experts vs. Novices across specified model dimensions.
    Title is two lines: "Behavioral vs. Model Dissimilarity Correlations" (bold) and "(Experts vs. Novices)".
    """
    # Each tuple in expert_results/novice_results: (col, r, p, ci_lower, ci_upper)
    r_exp = [tup[1] for tup in expert_results]                                 # Extract r-values for Experts
    ci_exp = [(tup[3], tup[4]) for tup in expert_results]                       # Extract (ci_lower, ci_upper) for Experts
    r_nov = [tup[1] for tup in novice_results]                                   # Extract r-values for Novices
    ci_nov = [(tup[3], tup[4]) for tup in novice_results]                       # Extract (ci_lower, ci_upper) for Novices

    def ci_to_yerr(ci_list, r_list):                                            # Convert CI tuples to yerr format
        lowers, uppers = [], []                                                  # Initialize lists for lower & upper errors
        for (ci_l, ci_u), r_val in zip(ci_list, r_list):                         # Iterate through each (ci_l, ci_u) and r
            lowers.append(r_val - ci_l)                                          # Lower error = r - ci_lower
            uppers.append(ci_u - r_val)                                          # Upper error = ci_upper - r
        return np.array([lowers, uppers])                                        # Return 2 x N array

    yerr_exp = ci_to_yerr(ci_exp, r_exp)                                          # Compute yerr for Experts
    yerr_nov = ci_to_yerr(ci_nov, r_nov)                                          # Compute yerr for Novices

    new_order_raw = ["visual", "strategy", "check"]                               # Desired ordering of columns
    new_order_pretty = ["Visual\nSimilarity", "Strategy", "Checkmate"]             # Pretty labels for x-axis
    idx_order = [column_labels.index(lbl) for lbl in new_order_raw]               # Indices in original order

    labels_pretty = new_order_pretty                                            # Use pretty labels
    r_exp = [r_exp[i] for i in idx_order]                                         # Reorder r-values for Experts
    r_nov = [r_nov[i] for i in idx_order]                                         # Reorder r-values for Novices
    yerr_exp = yerr_exp[:, idx_order]                                             # Reorder yerr for Experts
    yerr_nov = yerr_nov[:, idx_order]                                             # Reorder yerr for Novices

    x = np.arange(len(labels_pretty))                                             # X positions for bars
    width = 0.35                                                                   # Width of each bar

    fig, ax = plt.subplots(figsize=FIGSIZE, facecolor="white")                    # Create figure and axis
    ax.bar(                                                                        # Plot Experts' bars
        x - width/2, r_exp, width=width,
        yerr=yerr_exp, capsize=5,
        color=COL_GREEN, label="Experts",
        alpha=0.7
    )
    ax.bar(                                                                        # Plot Novices' bars
        x + width/2, r_nov, width=width,
        yerr=yerr_nov, capsize=5,
        color=COL_RED, label="Novices",
        alpha=0.7
    )

    ax.set_xticks(x)                                                               # Set x-axis tick positions
    # ax.set_xticklabels(labels_pretty, rotation=45, ha="right", fontsize=LABEL_FS)  # Set x-axis tick labels
    ax.set_xticklabels(labels_pretty, ha="center", fontsize=LABEL_FS)  # Set x-axis tick labels
    ax.set_ylim(-0.2, 1.0)                                                          # Set y-axis limits
    ax.set_ylabel(r"Pearson $\it{r}$ (95% CI via bootstrapping)", fontsize=LABEL_FS)  # Set y-axis label

    bold_line = "Behavioral-Model RDMs Correlations\nExperts vs. Novices"                     # First line for title (bold)
    title_text = f"{bold_line}"
                                           # Second line for title (normal)
    ax.set_title(title_text, fontsize=TITLE_FS, pad=25)                              # Set plot title

    ax.axhline(0, linestyle="--", color="gray", linewidth=1)                         # Add horizontal zero line
    ax.legend(loc="upper left", frameon=False, fontsize=LABEL_FS)                    # Add legend

    # Add stars for significance (if p < 0.05) above each Experts bar
    # Add stars for significance (if p < 0.05) above each bar (Experts and Novices)
    for i in range(len(labels_pretty)):
        # Extract values for Experts
        _, _, p_exp, _, _ = expert_results[idx_order[i]]
        r_exp_val = r_exp[i]

        # Extract values for Novices
        _, _, p_nov, _, _ = novice_results[idx_order[i]]
        r_nov_val = r_nov[i]

        # Determine star string for Experts
        if p_exp < 0.001:
            stars_exp = "***"
        elif p_exp < 0.01:
            stars_exp = "**"
        elif p_exp < 0.05:
            stars_exp = "*"
        else:
            stars_exp = None

        # Determine star string for Novices
        if p_nov < 0.001:
            stars_nov = "***"
        elif p_nov < 0.01:
            stars_nov = "**"
        elif p_nov < 0.05:
            stars_nov = "*"
        else:
            stars_nov = None

        # Plot stars for Experts
        if stars_exp:
            xpos = x[i] - width / 2                           # X-position of Experts bar
            y_pos = r_exp_val + yerr_exp[1, i] + 0.03         # Y-position just above error bar
            ax.text(
                xpos, y_pos, stars_exp,
                ha="center", va="bottom",
                fontsize=TITLE_FS * 0.6, fontweight="bold"
            )

        # Plot stars for Novices
        if stars_nov:
            xpos = x[i] + width / 2                           # X-position of Novices bar
            y_pos = r_nov_val + yerr_nov[1, i] + 0.03         # Y-position just above error bar
            ax.text(
                xpos, y_pos, stars_nov,
                ha="center", va="bottom",
                fontsize=TITLE_FS * 0.6, fontweight="bold"
            )


    for spine_loc in ["left", "bottom"]:                                             # Customize spines
        ax.spines[spine_loc].set_visible(True)                                        # Show left/bottom spines
        ax.spines[spine_loc].set_linewidth(1.5)                                       # Set spine width
        ax.spines[spine_loc].set_edgecolor("black")                                   # Set spine color
    for spine_loc in ["top", "right"]:                                               # Hide top/right spines
        ax.spines[spine_loc].set_visible(False)                                       # Make spine invisible

    fig.tight_layout()                                                                # Adjust layout
    save_path = os.path.join(OUT_ROOT, "Behavioral_Model_RDMs_Correlations.png")      # Define output path
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0.1, facecolor='white')    # Save the plot
    plt.show()                                                                         # Display plot
                                                                      # Display plot


# ----------------------------------------
# DATA PROCESSING FUNCTIONS
# ----------------------------------------

def load_participants(participants_xlsx_path, sourcedata_root):
    """
    Load participant IDs and expertise statuses from Excel.
    Returns:
        participants_list: list of (sub_id_str, is_expert_bool)
        counts: (num_experts, num_nonexperts)
    """
    df = pd.read_excel(participants_xlsx_path)                            # Read participants Excel file
    df_filtered = df.dropna(subset=["Expert"])                             # Filter where 'Expert' is not NaN
    subs = [                                                                 # Create list of (sub_id, is_expert)
        (f"sub-{int(s):02}", bool(e))                                        # Format sub_id with leading zeros
        for s, e in zip(                                                     # Zip subject IDs with Expert flags
            df_filtered["sub_id"].astype(int).astype(str).str.zfill(2),     # Zero-pad subject IDs to length 2
            df_filtered["Expert"]                                           # Expert boolean flag
        )
    ]
    num_exp = int(df_filtered["Expert"].sum())                              # Count number of experts
    num_non = len(df_filtered) - num_exp                                     # Count number of non-experts
    return subs, (num_exp, num_non)                                          # Return list and counts


def create_pairwise_df(trial_df):
    """
    Convert a subject's trial-level DataFrame into a pairwise comparison DataFrame.
    Each row lists 'better', 'worse', and 'sub_id'.
    """
    button_map = {                                                           # Mapping of button codes for preference
        1: {1: 51, -1: 52, 0: 0},                                              # For first mapping scheme
        2: {-1: 51, 1: 52, 0: 0},                                              # For second mapping scheme
    }
    comp_list = []                                                            # Initialize list for comparison rows

    for i in range(1, len(trial_df)):                                         # Iterate through trials from the second onward
        resp = int(trial_df.iloc[i]["response"])                               # Current trial's response code
        if resp == 0:                                                          # If response is zero (no preference)
            continue                                                            # Skip this trial

        bm_key = int(trial_df.iloc[i]["button_mapping"])                        # Button mapping key for this trial
        mapping = button_map[bm_key]                                           # Get mapping dictionary

        curr_stim = int(trial_df.iloc[i]["stim_id"])                            # Current stimulus ID
        prev_stim = int(trial_df.iloc[i - 1]["stim_id"])                        # Previous stimulus ID

        if resp == mapping[1]:                                                  # If response corresponds to "current preferred"
            better, worse = curr_stim, prev_stim                                # Current is better than previous
        else:                                                                  # Otherwise
            better, worse = prev_stim, curr_stim                                # Previous is better than current

        comp_list.append({                                                     # Append a comparison row
            "better": better,                                                  # Preferred stimulus ID
            "worse": worse,                                                    # Non-preferred stimulus ID
            "sub_id": trial_df.iloc[i]["sub_id"]                               # Subject identifier
        })

    return pd.DataFrame(comp_list)                                             # Return as DataFrame


def compute_symmetric_rdm(pairwise_df, expertise_label, do_plot=False):
    """
    Build a symmetric RDM: RDM[i,j] = | count(i preferred over j) - count(j over i) |.
    If do_plot=True, displays a heatmap with bold title and expertise label.
    Returns:
        rdm_matrix (ndarray)
    """
    counts = pairwise_df.groupby(['better', 'worse']).size().unstack(fill_value=0)  # Count occurrences of each (better, worse) pair
    stimuli = sorted(set(pairwise_df["better"]).union(set(pairwise_df["worse"])))   # Unique stimulus IDs
    n = max(stimuli)                                                                 # Number of stimuli (assuming IDs from 1..n)
    mat = np.zeros((n, n), dtype=int)                                                # Initialize count matrix (n x n)

    for (i, j), c in counts.stack().items():                                          # Iterate through nonzero counts
        mat[i - 1, j - 1] = c                                                          # Fill matrix at (i-1, j-1)

    rdm_mat = np.abs(mat - mat.T)                                                     # Compute symmetric dissimilarity (absolute difference)

    if do_plot:                                                                       # If plotting is requested
        bold = "Behavioral RDM"                                       # Bold title line
        plot_rdm_heatmap(rdm_mat, bold, expertise_label, colormap="RdPu", vmin=0, vmax=18)  # Plot as heatmap

    return rdm_mat                                                                    # Return RDM as ndarray


def compute_directional_dsm(pairwise_df, expertise_label, do_plot=False):
    """
    Build a directional preference DSM: DSM[i,j] = count(i over j) - count(j over i).
    If do_plot=True, displays a heatmap with bold title and expertise label.
    Returns:
        dsm_matrix (ndarray)
    """
    counts = pairwise_df.groupby(['better', 'worse']).size().unstack(fill_value=0)  # Count occurrences of each (better, worse) pair
    stimuli = sorted(set(pairwise_df["better"]).union(set(pairwise_df["worse"])))   # Unique stimulus IDs
    n = max(stimuli)                                                                 # Number of stimuli
    mat = np.zeros((n, n), dtype=int)                                                # Initialize count matrix

    for (i, j), c in counts.stack().items():                                          # Iterate through nonzero counts
        mat[i - 1, j - 1] = c                                                          # Fill matrix at (i-1, j-1)

    dsm_mat = mat - mat.T                                                             # Compute directional DSM (signed difference)

    if do_plot:                                                                       # If plotting is requested
        bold = "Directional Preference"                                        # Bold title line
        plot_rdm_heatmap(dsm_mat, bold, expertise_label, colormap=CUSTOM_CMAP, vmin=-18, vmax=18)  # Plot heatmap

    return dsm_mat                                                                    # Return DSM as ndarray


def process_single_participant(sub_id, is_expert, sourcedata_root, columns):
    """
    Load, filter, and concatenate all 'exp' runs for a given participant.
    Returns:
        trial_df (DataFrame): All valid trials concatenated, or None if none valid.
    """
    participant_dir = os.path.join(sourcedata_root, sub_id, "bh")      # Construct participant directory path
    mat_files = sorted(glob.glob(os.path.join(participant_dir, "*.mat")))  # Find all .mat files
    assert len(mat_files) > 0, f"No .mat files found for {sub_id}"     # Ensure files exist

    single_df = pd.DataFrame([], columns=columns)                      # Initialize empty DataFrame for this subject

    for mat_file in mat_files:                                         # Iterate through each .mat file
        try:
            ts, sid, run, task_ext = os.path.basename(mat_file).split("_")  # Attempt to split filename into parts
        except ValueError:                                             # If splitting into 4 parts fails
            sid, run, task_ext = os.path.basename(mat_file).split("_")      # Split into 3 parts

        task = task_ext.split(".")[0]                                   # Remove file extension to get task name
        if task != "exp":                                               # If this file is not an experiment run
            continue                                                     # Skip it

        try:
            mat = mat73.loadmat(mat_file)                               # Try loading v7.3 .mat file
        except:
            mat = scipy.io.loadmat(mat_file)                            # Fallback to scipy for older .mat

        trials = mat.get("trialList", [])                               # Extract 'trialList' from loaded .mat
        assert len(trials) > 0, f"Empty 'trialList' in {mat_file}"       # Ensure trials are present

        run_df = pd.DataFrame(trials, columns=columns)                   # Create DataFrame for this run
        if run_df["response"].sum() <= 0:                                # If no valid responses in this run
            warnings.warn(f"No valid responses for {sub_id}, run {run}")  # Warn user
            continue                                                     # Skip this run

        single_df = pd.concat([single_df, run_df], ignore_index=True)   # Append run data to subject DataFrame

    if single_df["response"].sum() <= 0:                                # If no valid responses across all runs
        warnings.warn(f"No valid responses across runs for {sub_id}")  # Warn user
        return None                                                     # Return None to indicate skipping

    return single_df                                                    # Return concatenated DataFrame for this subject


def worker_process(sub_info):
    """
    Worker function for multiprocessing. Unpacks (sub_id, is_expert), processes the subject,
    and returns (sub_id, single_df, is_expert, rdm_ind, dsm_ind) or None values if skipped.
    """
    sub_id, is_expert, sourcedata_root, columns = sub_info         # Unpack the tuple of information
    single_df = process_single_participant(sub_id, is_expert, sourcedata_root, columns)  # Process subject
    if single_df is None:                                           # If processing returned None
        return (sub_id, None, is_expert, None, None)                # Return tuple with None placeholders

    pairwise_df = create_pairwise_df(single_df)                     # Create pairwise comparison DataFrame
    rdm_ind = compute_symmetric_rdm(pairwise_df, "Individual", do_plot=False)  # Compute individual RDM (no plot)
    dsm_ind = compute_directional_dsm(pairwise_df, "Individual", do_plot=False)  # Compute individual DSM (no plot)
    return (sub_id, single_df, is_expert, rdm_ind, dsm_ind)          # Return processed results for this subject


# ----------------------------------------
# GROUP-LEVEL ANALYSIS FUNCTIONS
# ----------------------------------------

def load_stimulus_categories(cat_path):
    """
    Load stimulus categories from an Excel file and filter valid stim_id range.
    Returns:
        df_cat (DataFrame with columns ['stim_id', 'check', 'visual', 'strategy'])
    """
    df_cat = pd.read_excel(cat_path)                                      # Read category Excel
    df_cat = df_cat[
        (df_cat["stim_id"] >= 1) &                                        # Filter stim_id >=1
        (df_cat["stim_id"] <= len(df_cat["stim_id"].unique()))            # Filter stim_id <= number of unique
    ].reset_index(drop=True)                                               # Reset index after filtering
    return df_cat[["stim_id", "check", "visual", "strategy"]]             # Return only relevant columns


def correlate_model_behavior(d_group, df_cat, expertise_label):
    """
    For each categorical (and scalar) column in df_cat, compute:
      - Model RDM (0/1 for categorical, abs difference for scalar)
      - Pearson r, p-value, and 95% CI (via Pingouin bootstrap) with d_group.
    Plots each model RDM as a square heatmap with formatted title.
    Returns:
        results_list: list of tuples (col, r, p, ci_lower, ci_upper)
        col_labels: list of column names in the same order.
    """
    cat_cols = ["check", "visual", "strategy"]                            # List of categorical columns to analyze
    scalar_cols = []                                                       # Placeholder for scalar columns if needed

    n_stim = len(df_cat["stim_id"].unique())                               # Number of unique stimuli
    D_trunc = d_group[:n_stim, :n_stim]                                     # Truncate group RDM to match stimuli

    results = []                                                           # List to store correlation results
    labels = []                                                            # List to store column labels

    for col in scalar_cols + cat_cols:                                      # Iterate through each column of interest
        vals = df_cat[col].values                                           # Extract values for this column
        n = len(vals)                                                       # Number of stimuli for this column
        if n == 0:                                                          # If no values (e.g., scalar_cols is empty)
            continue                                                        # Skip to next column

        M = np.zeros((n, n))                                                # Initialize model RDM matrix
        if col in scalar_cols:                                              # If this is a scalar column
            for a in range(n):                                              # Iterate through rows
                for b in range(n):                                          # Iterate through columns
                    M[a, b] = abs(vals[a] - vals[b])                        # Fill with absolute differences
        else:                                                               # Otherwise, categorical column
            for a in range(n):                                              # Iterate through rows
                for b in range(n):                                          # Iterate through columns
                    M[a, b] = 0 if vals[a] == vals[b] else 1                # 0 if same category, 1 if different

        bold_title = f"{col.capitalize()}"       # Title for model RDM plot
        plot_rdm_heatmap(M, bold_title, expertise_label, colormap="RdPu", vmin=0, vmax=1)  # Plot model RDM

        tri_idx = np.tril_indices(n, k=-1)                                   # Indices for lower triangle
        x_vals = M[tri_idx]                                                  # Flatten model RDM lower triangle
        y_vals = D_trunc[tri_idx]                                            # Flatten group RDM lower triangle

        corr_df = pg.corr(                                                   # Use Pingouin to compute correlation
            x_vals, y_vals,
            method='pearson',
            alternative='two-sided',
            bootstrap=10000
        )
        r_val = corr_df['r'].values[0]                                        # Extract Pearson r
        p_val = corr_df['p-val'].values[0]                                    # Extract p-value
        ci_l, ci_u = corr_df['CI95%'].values[0]                               # Extract 95% CI bounds

        results.append((col, r_val, p_val, ci_l, ci_u))                       # Append to results list
        labels.append(col)                                                    # Append column label

    return results, labels                                                  # Return correlation results and labels


def analyze_group(df_group, expertise_label, df_cat):
    """
    Complete analysis pipeline for a single group (Experts or Novices):
      1. Plot choice-frequency.
      2. Compute & plot symmetric RDM.
      3. Compute & plot MDS on RDM.
      4. Compute & plot directional DSM.
      5. Correlate group RDM with model RDMs from df_cat.
    Returns:
        group_rdm: the symmetric RDM for the entire group.
        correlation_results: list of (col, r, p, ci_l, ci_u).
        column_labels: list of column names corresponding to results.
    """
    pairwise_all = create_pairwise_df(df_group)                             # Convert trials to pairwise comparisons
    plot_choice_frequency(pairwise_all, expertise_label)                     # Plot choice-frequency distribution

    group_rdm = compute_symmetric_rdm(pairwise_all, expertise_label, do_plot=True)  # Compute and plot group RDM

    df_for_mds = pd.DataFrame(group_rdm, columns=list(range(group_rdm.shape[0])))     # Create DataFrame for MDS
    plot_mds(df_for_mds, "MDS Embedding of RDM", expertise_label)  # Plot MDS embedding

    _ = compute_directional_dsm(pairwise_all, expertise_label, do_plot=True)    # Compute and plot directional DSM

    corr_results, col_labels = correlate_model_behavior(group_rdm, df_cat, expertise_label)  # Correlate group RDM with model RDMs

    print(f"--- Correlation Results ({expertise_label}) ---")                  # Print header for correlation results
    for col, r_val, p_val, ci_l, ci_u in corr_results:                          # Iterate through results
        print(f"Column: {col}")                                                 # Print column name
        print(f"  r = {r_val:.3f}, p = {p_val:.3e}, 95% CI = [{ci_l:.3f}, {ci_u:.3f}]\n")  # Print stats

    return group_rdm, corr_results, col_labels                                    # Return group RDM and correlation info


# ----------------------------------------
# MAIN EXECUTION FLOW
# ----------------------------------------

participants_list, (num_exp, num_non) = load_participants(                        # Load participants and counts
    participants_xlsx_path="data/participants.xlsx",
    sourcedata_root="/media/costantino_ai/eik-T9/projects_backup/2024_chess-expertise/data/sourcedata"
)
print(f"Number of Experts: {num_exp} | Number of Non-Experts: {num_non}")          # Print participant counts

trial_columns = [                                                                 # Define column names for trial DataFrames
    "sub_id", "run", "run_trial_n", "stim_id",
    "stim_onset_real", "response", "stim_onset_expected",
    "button_mapping"
]

experts_df = pd.DataFrame([], columns=trial_columns)                               # Initialize empty DataFrame for Experts
novices_df = pd.DataFrame([], columns=trial_columns)                                # Initialize empty DataFrame for Novices
experts_rdms = []                                                                  # List to store each Expert's individual RDM
novices_rdms = []                                                                  # List to store each Novice's individual RDM

sourcedata_root = "/media/costantino_ai/eik-T9/projects_backup/2024_chess-expertise/data/sourcedata"  # Root path for source data

if MULTIPROCESS:                                                                   # If multiprocessing is enabled
    n_cpu = max(1, multiprocessing.cpu_count() - 1)                                # Use all but one CPU core
    with multiprocessing.Pool(processes=n_cpu) as pool:                             # Create a pool of worker processes
        args = [                                                                   # Prepare arguments for each subject
            (sub_id, is_expert, sourcedata_root, trial_columns)
            for sub_id, is_expert in participants_list
        ]
        results = pool.map(worker_process, args)                                   # Map worker_process over all subjects

    for sub_id, single_df, is_expert, rdm_ind, dsm_ind in results:                 # Iterate through worker results
        if single_df is None:                                                       # If subject had no valid data
            continue                                                                # Skip to next
        if is_expert:                                                               # If subject is an Expert
            experts_rdms.append(rdm_ind)                                            # Store individual RDM
            experts_df = pd.concat([experts_df, single_df], ignore_index=True)      # Append trials to Experts DataFrame
        else:                                                                       # If subject is a Novice
            novices_rdms.append(rdm_ind)                                            # Store individual RDM
            novices_df = pd.concat([novices_df, single_df], ignore_index=True)      # Append trials to Novices DataFrame

else:                                                                              # If multiprocessing is disabled
    for sub_id, is_expert in participants_list:                                    # Iterate through each subject
        print(f"Processing participant: {sub_id} | Expert: {is_expert}")            # Print subject info
        single_df = process_single_participant(                                     # Process subject sequentially
            sub_id=sub_id,
            is_expert=is_expert,
            sourcedata_root=sourcedata_root,
            columns=trial_columns
        )
        if single_df is None:                                                       # If subject had no valid data
            continue                                                                # Skip to next
        pairwise_df = create_pairwise_df(single_df)                                  # Convert trials to pairwise comparisons
        rdm_ind = compute_symmetric_rdm(pairwise_df, "Individual", do_plot=False)    # Compute individual symmetric RDM
        dsm_ind = compute_directional_dsm(pairwise_df, "Individual", do_plot=False)  # Compute individual directional DSM

        if is_expert:                                                               # If subject is an Expert
            experts_rdms.append(rdm_ind)                                            # Store individual RDM
            experts_df = pd.concat([experts_df, single_df], ignore_index=True)      # Append trials to Experts DataFrame
        else:                                                                       # If subject is a Novice
            novices_rdms.append(rdm_ind)                                            # Store individual RDM
            novices_df = pd.concat([novices_df, single_df], ignore_index=True)      # Append trials to Novices DataFrame

category_df = load_stimulus_categories(                                           # Load stimulus category data
    cat_path="/home/eik-tb/OneDrive_andreaivan.costantino@kuleuven.be/"
             "GitHub/chess-expertise-2024/chess-dataset-vis/data/categories.xlsx"
)

expert_rdm, expert_corrs, expert_cols = analyze_group(                             # Analyze Experts group
    df_group=experts_df,
    expertise_label="Experts",
    df_cat=category_df
)

novice_rdm, novice_corrs, novice_cols = analyze_group(                             # Analyze Novices group
    df_group=novices_df,
    expertise_label="Novices",
    df_cat=category_df
)

if expert_cols != novice_cols:                                                     # Ensure column labels match
    raise ValueError("Mismatch in column labels between expert and novice results!")

plot_model_behavior_correlations(                                                  # Plot final correlation comparison
    expert_results=expert_corrs,
    novice_results=novice_corrs,
    column_labels=expert_cols
)

pretty_labels = ["Visual Similarity", "Strategy", "Checkmate"]                    # Pretty labels for final summary table
stats_df = pd.DataFrame({                                                          # Build summary DataFrame
    "Dimension": pretty_labels,                                                     # Model dimension names
    "r_Experts": [f"{tup[1]:.3f}" for tup in expert_corrs],                         # Expert correlation r-values
    "95% CI Experts": [f"[{tup[3]:.3f}, {tup[4]:.3f}]" for tup in expert_corrs],     # Expert 95% confidence intervals
    "r_Novices": [f"{tup[1]:.3f}" for tup in novice_corrs],                          # Novice correlation r-values
    "95% CI Novices": [f"[{tup[3]:.3f}, {tup[4]:.3f}]" for tup in novice_corrs],      # Novice 95% confidence intervals
})
print("\n=== Behavioral RDM Correlations (Experts vs. Novices) ===\n")               # Print header for summary table
print(stats_df.to_string(index=False))                                             # Print summary table without index

plot_shared_colorbar(CUSTOM_CMAP, vmin=-18, vmax=18)                                # Display the standalone colorbar for DSMs
