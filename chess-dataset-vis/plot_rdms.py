#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  7 15:26:04 2025

@author: costantino_ai
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.patches import Rectangle
import seaborn as sns
import numpy as np

def plot_rdm(rdm, title="RDM", colormap="RdPu"):
    """
    Plot a representational dissimilarity matrix (RDM) with external colored bars along the axes to group strategies.

    Args:
    - rdm (array-like): The dissimilarity matrix to plot.
    - strategies (list of str): The list of strategy labels corresponding to both the rows and columns of the RDM.
    - title (str): The title of the plot.
    - colormap (str): The colormap for the heatmap.
    """
    # Initialize the plot
    fig, ax = plt.subplots(figsize=(12, 10))
    hm = sns.heatmap(
        rdm,
        annot=False,
        fmt="d",
        cmap=colormap,
        ax=ax,

    )
    hm.collections[0].colorbar.set_label("Dissimilarity", fontsize=24)  # main label


    # Set up tick positions and labels for changes in strategy group
    ticks = []
    tick_labels = []
    prev_label = None

    # Determine the positions where the label changes
    for i, label in enumerate(STRATEGIES):
        if label != prev_label:
            ticks.append(i)
            tick_labels.append(label)
        prev_label = label

    # Adding one more tick for the end of the last block
    ticks.append(len(STRATEGIES))

    # Set ticks and labels
    ax.set_xticks(ticks[:-1], labels=tick_labels, rotation=45, ha='right', fontsize=8)
    ax.set_yticks(ticks[:-1], labels=tick_labels, fontsize=8)

    # Add rectangles for strategy group highlighting
    strategy_colors, strategy_alpha = determine_color_and_alpha(STRATEGIES)

    # Create patches for both axes
    for idx, start in enumerate(ticks[:-1]):
        end = ticks[idx + 1]
        width = end - start
        color = strategy_colors[ticks[idx]]
        alpha = strategy_alpha[ticks[idx]]

        # Create rectangle patches
        # Adding rect_x along the bottom x-axis
        rect_x = Rectangle((start, -0.01), width, -0.0005*len(rdm), color=color, alpha=alpha, ec=None, transform=ax.get_xaxis_transform(), clip_on=False)
        # Adding rect_y along the left y-axis
        rect_y = Rectangle((-0.01, start), -0.0005*len(rdm), width, color=color, alpha=alpha, ec=None, transform=ax.get_yaxis_transform(), clip_on=False)

        ax.add_patch(rect_x)
        ax.add_patch(rect_y)

    # Set ticks and labels
    ax.set_xticks([])
    ax.set_yticks([])

    # Enhance plot titles and labels
    plt.title(title, fontsize=30, pad=20)
    hm.collections[0].colorbar.ax.tick_params(labelsize=20)  # Set colorbar label size to 20

    fname = os.path.join(OUTPUT_DIR, f"{title}.png")
    plt.savefig(fname, bbox_inches='tight', pad_inches=0.1)

    plt.show()

def determine_color_and_alpha(STRATEGIES):
    # Base colors
    colors_green = '#006400'  # Dark green
    colors_red = '#8B0000'    # Dark red

    # Initialize tracking variables
    current_strategy = None
    strategy_colors = []
    strategy_alpha = []
    color_index = 0

    # Loop through strategies
    for strategy in STRATEGIES:
        if strategy != current_strategy:
            # Check if current strategy index is less than 5 for green, else red
            if color_index < 5:  # Assuming exactly 10 strategies
                color = colors_green
                alpha = (color_index + 1) / 5.0  # Increment alpha from 0.2 to 1.0
            else:
                color = colors_red
                alpha = (color_index + 1 - 5) / 5.0  # Normalize and increment alpha
            current_strategy = strategy
            color_index += 1

        strategy_colors.append(color)
        strategy_alpha.append(alpha)

    return strategy_colors, strategy_alpha

def compute_rdm(series: pd.Series, colname: str) -> np.ndarray:
    """
    Given a pandas Series (vector) of length N and its column name,
    return an NxN RDM (numpy array).

    - For 'check-n', 'total_pieces', and 'legal_moves': use absolute difference.
    - All others: use strict 0/1 difference (categorical).
    """
    # Define the numeric columns
    numeric_cols = {'check-n', 'total_pieces', 'legal_moves'}

    # Get values
    vals = series.values

    if colname in numeric_cols:
        try:
            vals = vals.astype(float)
        except ValueError:
            raise ValueError(f"Column '{colname}' is expected to be numeric but cannot be converted.")
        rdm = np.abs(vals[:, None] - vals[None, :])
    else:
        # Convert to integer labels
        vals, _ = pd.factorize(vals)
        rdm = (vals[:, None] != vals[None, :]).astype(float)

    return rdm

# -------------------------------------------------------------------
# 1) Configuration
# -------------------------------------------------------------------

EXCEL_PATH = "./data/categories.xlsx"  # Path to your Excel file
STIM_DIR   = "./data/stimuli"          # Directory containing all stimulus images
OUTPUT_DIR = "./results/output_rdm_figures"        # Where to save the figures

# Make sure the output directory exists:
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# Figure layout parameters:
N_COLS = 20  # we place 20 stimuli per row
N_ROWS = 2   # total 40 stimuli
THUMB_WIDTH  = 20 *2  # width (in inches) for the entire grid figure
THUMB_HEIGHT = 4 *2  # height (in inches) for the entire grid figure

# Border thickness around each stimulus:
BORDER_LW = 5 * 2

regressor_mapping = {
    "check": "Checkmate vs. Non-checkmate boards",
    "stim_id": "Pairwise all boards",
    "motif": "Motifs (Checkmate boards only)",
    "check-n": "Number of moves to checkmate (Checkmate boards only)",
    "side": "King position (L-R, Checkmate boards only)",
    "strategy": "Strategy (all stimuli)",
    "visual": "Visually similar pairs",
    "total_pieces": "Total number of pieces",
    "legal_moves": "Number of available legal moves",
    "difficulty": "Board difficulty (Checkmate boards only)",
    "first_piece": "First piece to move (Checkmate boards only)",
    "checkmate_piece": "Checkmate piece (Checkmate boards only)",
}

STRATEGIES = [
    1,
    1,
    1,
    2,
    2,
    2,
    2,
    3,
    3,
    3,
    3,
    3,
    3,
    3,
    4,
    4,
    4,
    5,
    5,
    5,
    1,
    1,
    1,
    2,
    2,
    2,
    2,
    3,
    3,
    3,
    3,
    3,
    3,
    3,
    4,
    4,
    4,
    5,
    5,
    5,
]

# -------------------------------------------------------------------
# 2) Read Excel data and basic checks
# -------------------------------------------------------------------

df = pd.read_excel(EXCEL_PATH)

# Identify which labels we do not want to plot:
exclude_columns = {'filename', 'fen', 'correct', 'Unnamed: 15', 'side: 0=lfet; 1=right'}
level_columns = [col for col in df.columns if col not in exclude_columns]

# We expect a 'filename' column plus one or more numeric columns.
# The example below assumes exactly 40 rows (first 20 = "cold set", second 20 = "hot set").
# Adapt as needed if your dataset differs.
if df.shape[0] != 40:
    print("WARNING: This script is configured for exactly 40 stimuli. Found:", df.shape[0])

# Identify which labels we do not want to plot:
exclude_columns = {
    'filename',
    'fen',
    'correct',
    'Unnamed: 15',
    'side: 0=lfet; 1=right'
}
level_columns = [col for col in df.columns if col not in exclude_columns]

# Check the number of rows (you said you have exactly 40 stimuli)
if df.shape[0] != 40:
    print("WARNING: This script is configured for exactly 40 stimuli. Found:", df.shape[0])

print("Columns found:", df.columns)
print("Columns to build RDMs for:", level_columns)

for col in level_columns:
    print(f"Computing RDM for column: {col}")
    rdm = compute_rdm(df[col], col)

    plot_rdm(rdm, title=regressor_mapping[col])
