#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 15 17:24:31 2025

@author: costantino_ai
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
from modules.helpers import create_run_id                       # Helper function to create a unique run identifier
import scipy.stats
from scipy.spatial.distance import squareform
import multiprocessing                                         # Python's multiprocessing module for parallel execution

# ----------------------------------------
# GLOBAL SETTINGS
# ----------------------------------------

SF = 1.0                                                      # Scaling factor for all font sizes
FONT_FAMILY = 'Ubuntu'                                        # Global font family
TITLE_FS = 26 * 1.4 * SF                                      # Font size for plot titles (≈36.4)
LABEL_FS = 26 * 1.2 * SF                                      # Font size for axis labels (≈31.2)
TICK_FS = 26 * SF                                             # Font size for tick labels (26)
COLORBAR_LABEL_FS = 26 * 1.1 * SF                             # Font size for colorbar labels (≈28.6)
COLORBAR_TICK_FS = 26 * 0.9 * SF                              # Font size for colorbar tick labels (≈23.4)

FIGSIZE = (12, 9)                                             # Default figure size (width, height) for all plots

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
            logging.warning(f"No valid responses for {sub_id}, run {run}")  # Warn user
            continue                                                     # Skip this run

        single_df = pd.concat([single_df, run_df], ignore_index=True)   # Append run data to subject DataFrame

    if single_df["response"].sum() <= 0:                                # If no valid responses across all runs
        logging.warning(f"No valid responses across runs for {sub_id}")  # Warn user
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

def compute_reliabilities_from_trials(trial_df_dict):
    """
    Compute within-group and between-group RDM reliability from raw trial data (not precomputed RDMs).
    Uses leave-one-subject-out strategy with RDMs built from pooled trial data, as in the original script.

    Args:
        trial_df_dict (dict):
            {
                "GroupName": {
                    "sub-01": DataFrame (raw trials),
                    ...
                },
                ...
            }

    Returns:
        dict: {
            "GroupName": {
                "within": [...],
                "between": [...]
            },
            ...
        }
    """
    results = {}

    for group_name, subj_data in trial_df_dict.items():
        group_subs = list(subj_data.keys())
        within_scores = []

        for sub_id in group_subs:
            # Get trials for subject i
            df_i = subj_data[sub_id]
            pairwise_i = create_pairwise_df(df_i)
            rdm_i = compute_symmetric_rdm(pairwise_i, expertise_label="Individual", do_plot=False)

            # Get pooled trials from all other subjects
            others_dfs = [df for sid, df in subj_data.items() if sid != sub_id]
            if not others_dfs:
                continue  # skip leave-one-out if only one subject
            df_others = pd.concat(others_dfs, ignore_index=True)
            pairwise_others = create_pairwise_df(df_others)
            rdm_others = compute_symmetric_rdm(pairwise_others, expertise_label="Group", do_plot=False)

            # Compare lower triangles
            idx = np.tril_indices(rdm_i.shape[0], k=-1)
            vec_i = rdm_i[idx]
            vec_others = rdm_others[idx]
            r, _ = scipy.stats.spearmanr(vec_i, vec_others)
            within_scores.append(r)

        results[group_name] = {"within": within_scores}

    # If multiple groups, compute between-group reliabilities
    if len(trial_df_dict) > 1:
        group_names = list(trial_df_dict.keys())
        for g1 in group_names:
            # vecs1 = results[g1]["within"]
            others = [g2 for g2 in group_names if g2 != g1]
            between_scores = []

            for sub_id, df in trial_df_dict[g1].items():
                pairwise_i = create_pairwise_df(df)
                rdm_i = compute_symmetric_rdm(pairwise_i, expertise_label="Individual", do_plot=False)
                idx = np.tril_indices(rdm_i.shape[0], k=-1)
                vec_i = rdm_i[idx]

                # Average other group RDMs
                dfs_others = [trial_df_dict[g2][sid] for g2 in others for sid in trial_df_dict[g2]]
                df_others = pd.concat(dfs_others, ignore_index=True)
                pairwise_others = create_pairwise_df(df_others)
                rdm_others = compute_symmetric_rdm(pairwise_others, expertise_label="Group", do_plot=False)
                vec_others = rdm_others[idx]

                r, _ = scipy.stats.spearmanr(vec_i, vec_others)
                between_scores.append(r)

            results[g1]["between"] = between_scores

    return results

def plot_reliability_bars(plot_data, out_fig=None, out_csv=None, run_id=None):
    """
    Plot within- and between-group RDM reliability for Experts and Novices,
    with 95% CI error bars, significance asterisks, and between-group comparisons.

    Parameters
    ----------
    plot_data : dict
        Dictionary containing bar plot data and stats:
        - x: np.ndarray of x positions
        - width: float, bar width
        - means_exp, means_nov: mean r values
        - ci_exp, ci_nov: confidence intervals (2 x 2 arrays)
        - summary: dict with individual stats and p-values
        - p_vals_between: dict with p-values for between-group diffs
    out_fig : str
        Path to save the figure.
    out_csv : str
        Path to save the CSV with annotated data.
    run_id : str
        Title identifier for the plot.

    Returns
    -------
    pd.DataFrame
        Annotated DataFrame used for plotting.
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np
    import pandas as pd

    COL_GREEN = "#4daf4a"
    COL_RED = "#e41a1c"
    means_exp = plot_data["means_exp"]
    means_nov = plot_data["means_nov"]
    ci_exp = plot_data["ci_exp"]
    ci_nov = plot_data["ci_nov"]
    summary = plot_data["summary"]
    p_vals_between = plot_data["p_vals_between"]

    terms = ["Within-group", "Between-group"]
    plot_df = pd.DataFrame({
        "Term": terms * 2,
        "Correlation": list(means_exp) + list(means_nov),
        "CI_low": list(means_exp - ci_exp[0]) + list(means_nov - ci_nov[0]),
        "CI_high": list(means_exp + ci_exp[1]) + list(means_nov + ci_nov[1]),
        "Group": ["Experts"] * 2 + ["Novices"] * 2,
    })

    fig, ax = plt.subplots()
    sns.barplot(
        data=plot_df,
        x="Term",
        y="Correlation",
        hue="Group",
        palette=[COL_GREEN, COL_RED],
        dodge=True,
        errorbar=None,
        ax=ax,
    )

    patches = ax.patches
    data_range = np.abs(plot_df["Correlation"]).max() + np.abs(plot_df["CI_high"]).max()
    offset = data_range / 100 if data_range > 0 else 0.005
    group_heights = []

    # Individual error bars and stars
    for idx, row in plot_df.iterrows():
        patch = patches[idx]
        x_c = patch.get_x() + patch.get_width() / 2
        r = row["Correlation"]
        lo = row["CI_low"]
        hi = row["CI_high"]
        err_low = abs(r - lo)
        err_high = abs(hi - r)
        ax.errorbar(
            x_c, r,
            yerr=[[err_low], [err_high]],
            fmt="none", ecolor="k", capsize=5, capthick=1.5
        )
        group = row["Group"]
        label = "within" if row["Term"].lower().startswith("within") else "between"
        p = summary[group][label]["p"]

        if p < 0.001:
            star = "***"
        elif p < 0.01:
            star = "**"
        elif p < 0.05:
            star = "*"
        else:
            star = None

        if star:
            y_star, va = hi + offset, "bottom"
            ax.text(x_c, y_star, star, ha="center", va=va, color="gray", fontsize=18, fontweight="bold")
            group_heights.append(y_star)
        else:
            group_heights.append(hi)

    # Between-group comparisons
    for i, term in enumerate(["within", "between"]):
        p = p_vals_between[term]["p"]
        if p >= 0.05:
            continue

        bar1, bar2 = patches[i], patches[i + 2]
        x1 = bar1.get_x() + bar1.get_width() / 2
        x2 = bar2.get_x() + bar2.get_width() / 2
        max_y = max(group_heights[i], group_heights[i + 2])
        y_line = max_y + offset * 8

        ax.plot([x1, x2], [y_line, y_line], "k-", linewidth=1.5)
        star = "***" if p < 0.001 else ("**" if p < 0.01 else "*")
        ax.text((x1 + x2) / 2, y_line + offset, star, ha="center", va="bottom", color="black")

    ax.set_xlabel("Group", fontweight="bold")
    ax.set_ylabel("Spearman r (95% CI)", fontweight="bold")
    ax.set_title(f"{run_id or 'RDM Reliability'}\nExperts vs. Novices", pad=20)
    ax.set_xticklabels(terms, rotation=0)
    ax.legend(loc="upper left", frameon=False)
    sns.despine(ax=ax, top=True, right=True, left=False, bottom=False)
    ax.spines["left"].set_linewidth(1.0)
    ax.spines["bottom"].set_linewidth(1.0)
    ax.axhline(0, linestyle="--", color="gray", linewidth=1)
    plt.tight_layout()

    # Save outputs
    if out_fig:
        plot_df.to_csv(out_csv, index=False)
        plt.savefig(out_fig, dpi=300, bbox_inches="tight", pad_inches=0.1, facecolor="white")
    plt.show()

    return plot_df


# ----------------------------------------
# MAIN EXECUTION FLOW
# ----------------------------------------
sourcedata_root="/media/costantino_ai/eik-T9/projects_backup/2024_chess-expertise/data/sourcedata"

participants_list, (num_exp, num_non) = load_participants(
    participants_xlsx_path="data/participants.xlsx",
    sourcedata_root=sourcedata_root
)
print(f"Number of Experts: {num_exp} | Number of Non-Experts: {num_non}")

trial_columns = [
    "sub_id", "run", "run_trial_n", "stim_id",
    "stim_onset_real", "response", "stim_onset_expected",
    "button_mapping"
]

experts_df = pd.DataFrame([], columns=trial_columns)
novices_df = pd.DataFrame([], columns=trial_columns)

# Initialize RDM storage in dictionary format
rdms_dict = {
    "Experts": {},
    "Novices": {}
}

import logging

# Define CPU usage
n_cpu = max(1, multiprocessing.cpu_count() - 1)
logging.info(f"Using {n_cpu} CPU cores for parallel processing.")

# Prepare participant arguments
args = [
    (sub_id, is_expert, sourcedata_root, trial_columns)
    for sub_id, is_expert in participants_list
]

# Process data in parallel
logging.info("Starting multiprocessing pool...")
with multiprocessing.Pool(processes=n_cpu) as pool:
    results = pool.map(worker_process, args)

# Handle results
logging.info("Processing subject data...")
for i, (sub_id, single_df, is_expert, rdm_ind, dsm_ind) in enumerate(results):
    if rdm_ind is None:
        continue

    group_label = "Expert" if is_expert else "Novice"

    if single_df is None:
        logging.warning(f"[{group_label} | {sub_id}] Skipped: No valid data found.")
        continue

    if is_expert:
        experts_df = pd.concat([experts_df, single_df], ignore_index=True)
        rdms_dict["Experts"][sub_id] = rdm_ind
    else:
        novices_df = pd.concat([novices_df, single_df], ignore_index=True)
        rdms_dict["Novices"][sub_id] = rdm_ind

    logging.info(f"[{group_label} | {sub_id}] Data processed successfully.")

# Summary
logging.info(f"Finished processing {len(results)} participants.")
logging.info(f"Total valid Experts: {len(rdms_dict['Experts'])}")
logging.info(f"Total valid Novices: {len(rdms_dict['Novices'])}")

# Build trial-level data per subject per group
trial_df_dict = {
    "Experts": {sub_id: df for sub_id, df in zip(experts_df["sub_id"].unique(), [experts_df[experts_df["sub_id"] == sid] for sid in experts_df["sub_id"].unique()])},
    "Novices": {sub_id: df for sub_id, df in zip(novices_df["sub_id"].unique(), [novices_df[novices_df["sub_id"] == sid] for sid in novices_df["sub_id"].unique()])}
}

# Run updated reliability calculation
results = compute_reliabilities_from_trials(trial_df_dict)

# Step 1: Compute group-level stats with Pingouin
summary = {}
for group in ["Experts", "Novices"]:
    summary[group] = {}
    for typ in ["within", "between"]:
        data = pd.Series(results[group][typ])
        ttest = pg.ttest(data, 0, alternative='two-sided')
        row = ttest.iloc[0]
        summary[group][typ] = {
            "mean": data.mean(),
            "ci95%": (row['CI95%'][0], row['CI95%'][1]),
            "p": row['p-val']
        }

# Step 2: Between-group comparisons using independent t-tests
p_vals_between = {}
for typ in ["within", "between"]:
    data_exp = pd.Series(results["Experts"][typ])
    data_nov = pd.Series(results["Novices"][typ])
    ttest = pg.ttest(data_exp, data_nov, paired=False, alternative='two-sided')
    row = ttest.iloc[0]
    p_vals_between[typ] = {
        "p": row["p-val"],
        "ci95%": row["CI95%"]
    }

# Step 3: Prepare data for plotting
labels = ["Within", "Between"]
x = np.arange(len(labels))
width = 0.35

means_exp = [summary["Experts"]["within"]["mean"], summary["Experts"]["between"]["mean"]]
means_nov = [summary["Novices"]["within"]["mean"], summary["Novices"]["between"]["mean"]]

ci_exp = np.array([
    [summary["Experts"][k]["mean"] - summary["Experts"][k]["ci95%"][0],
     summary["Experts"][k]["ci95%"][1] - summary["Experts"][k]["mean"]]
    for k in ["within", "between"]
]).T

ci_nov = np.array([
    [summary["Novices"][k]["mean"] - summary["Novices"][k]["ci95%"][0],
     summary["Novices"][k]["ci95%"][1] - summary["Novices"][k]["mean"]]
    for k in ["within", "between"]
]).T

# Optional: save for use in plotting
plot_data = {
    "x": x,
    "width": width,
    "means_exp": means_exp,
    "means_nov": means_nov,
    "ci_exp": ci_exp,
    "ci_nov": ci_nov,
    "summary": summary,
    "p_vals_between": p_vals_between
}

plot_reliability_bars(plot_data)
