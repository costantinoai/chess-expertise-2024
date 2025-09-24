#!/usr/bin/env python3

"""
Generate visualization of a chess-stimuli dataset with colored borders
indicating various categorical 'levels' (check, check-n, strategy, etc.).
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.patches import Rectangle
import seaborn as sns

# -------------------------------------------------------------------
# 1) Configuration
# -------------------------------------------------------------------

EXCEL_PATH = "./data/categories.xlsx"  # Path to your Excel file
STIM_DIR   = "./data/stimuli"          # Directory containing all stimulus images
OUTPUT_DIR = "./results/output_figures"        # Where to save the figures

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
    import logging
    logging.basicConfig(level=logging.INFO)
    logging.warning("This script is configured for exactly 40 stimuli. Found: %s", df.shape[0])

# -------------------------------------------------------------------
# 3) Helper function to draw all stimuli in one figure (no color outlines)
# -------------------------------------------------------------------

def plot_all_stimuli(df, stim_dir, outpath):
    """
    Create a single figure with all stimuli (2 rows x 20 columns = 40 total).
    Just displays the images (no borders).
    """
    fig, axes = plt.subplots(nrows=N_ROWS, ncols=N_COLS,
                             figsize=(THUMB_WIDTH, THUMB_HEIGHT))

    # Flatten the 2D axes array for easier iteration:
    axes = axes.flatten()

    for i, (idx, row) in enumerate(df.iterrows()):
        ax = axes[i]
        ax.set_axis_off()  # No tick marks
        # Load the image
        img_path = os.path.join(stim_dir, row['filename'])
        if not os.path.isfile(img_path):
            # If image doesn't exist, you may raise an error or skip
            logging.warning("Image not found: %s", img_path)
            continue

        img = mpimg.imread(img_path)
        ax.imshow(img)
        # Optionally, set a small title with the filename (or omit if you prefer)
        # ax.set_title(row['filename'], fontsize=6)

    plt.tight_layout()
    plt.savefig(outpath, dpi=300)
    plt.show()
    plt.close(fig)

# -------------------------------------------------------------------
# 5.1) New: All stimuli with descriptive tags
# -------------------------------------------------------------------
def plot_all_stimuli_with_dynamic_borders(df, stim_dir, outpath):
    """
    Plot all stimuli with discrete, visibly distinct borders based on 'check' and 'strategy'.
    Green shades for checkmate, red for non-checkmate. Strategy level determines shade.
    """
    n_cols = 5
    n_rows = 8
    fig_width = 15
    fig_height = 24
    border_width = 8
    fontsize = 18
    fontname = "Ubuntu Condensed"

    fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols,
                             figsize=(fig_width, fig_height))
    axes = axes.flatten()

    # Suptitle
    plt.suptitle("Full Dataset", fontsize=40, fontname=fontname, weight='bold')

    # Separate checkmate and non-checkmate
    df_check = df[df['check'] == 1].copy()
    df_nocheck = df[df['check'] == 0].copy()

    # Rank strategy values
    df_check['strategy_rank'] = df_check['strategy'].rank(method='dense').astype(int)
    df_nocheck['strategy_rank'] = df_nocheck['strategy'].rank(method='dense').astype(int)

    # Palettes
    check_palette = sns.light_palette("green", n_colors=df_check['strategy_rank'].nunique() + 2)[1:-1]
    nocheck_palette = sns.light_palette("red", n_colors=df_nocheck['strategy_rank'].nunique() + 2)[1:-1]

    check_color_map = {
        rank: color for rank, color in zip(sorted(df_check['strategy_rank'].unique()), check_palette)
    }
    nocheck_color_map = {
        rank: color for rank, color in zip(sorted(df_nocheck['strategy_rank'].unique()), nocheck_palette)
    }

    # Merge strategy rank into main DataFrame
    df = df.copy()
    df['strategy_rank'] = -1
    df.loc[df_check.index, 'strategy_rank'] = df_check['strategy_rank']
    df.loc[df_nocheck.index, 'strategy_rank'] = df_nocheck['strategy_rank']

    for i, (idx, row) in enumerate(df.iterrows()):
        if i >= len(axes): break

        ax = axes[i]
        ax.set_axis_off()

        img_path = os.path.join(stim_dir, row['filename'])
        if not os.path.isfile(img_path):
            logging.warning("Image not found: %s", img_path)
            continue

        img = mpimg.imread(img_path)
        ax.imshow(img)

        # Tag
        try:
            stim_id = idx + 1
            check = "C" if int(row['check']) == 1 else "NC"
            strategy = int(row['strategy']) + 1
            visual = int(row['visual']) + 1
            # tag = f"S{stim_id} • {check} • SY{strategy} • P{visual}"
            tag = fr"$\bf{{S{stim_id}}}$•{check}•SY{strategy}•P{visual}"
        except KeyError as e:
            raise KeyError(f"Missing required column for tag: {e}")

        # Add tag below image using text
        ax.set_title(tag, fontsize=fontsize, pad=4, loc='center')

        # Border
        strat_rank = row['strategy_rank']
        if strat_rank == -1:
            border_color = "gray"
        else:
            border_color = check_color_map[strat_rank] if check == "C" else nocheck_color_map[strat_rank]

        rect = Rectangle((0, 0), 1, 1,
                         transform=ax.transAxes,
                         fill=False,
                         linewidth=border_width,
                         edgecolor=border_color)
        ax.add_patch(rect)

    plt.tight_layout(rect=[0, 0, 1, 0.98])  # Leave room for suptitle
    # plt.tight_layout()  # Leave room for suptitle
    plt.savefig(outpath, dpi=300)
    plt.show()
    plt.close(fig)

# -------------------------------------------------------------------
# 4) Function to color-code one "level" (column) in a new figure
# -------------------------------------------------------------------

def plot_level(df, stim_dir, level_col, outpath):
    """
    For the given level_col in df (e.g. 'check'), create a figure of all stimuli
    outlined in a color that depends on the category label.

    - If the column has 20 non-null values → Single-row figure (20 stimuli).
    - If the column has 40 non-null values → Two-row figure (40 stimuli).
    - Uses the 'colorblind' palette for clear distinctions.

    Raises an error if NaN values are found in the column or if the number of labels exceeds the colormap capacity.
    """

    # Drop NaN values and check dataset size
    df_valid = df.dropna(subset=[level_col])
    n_valid = len(df_valid)

    if n_valid not in {20, 40}:
        raise ValueError(f"Column '{level_col}' has {n_valid} valid entries (expected 20 or 40).")

    # Identify unique labels in this column and sort them
    unique_labels = sorted(df_valid[level_col].unique())
    n_labels = len(unique_labels)

    # Create a color palette using seaborn (n_labels distinct colors)
    palette = sns.color_palette("tab20", n_labels)

    # Maximum number of colors available in the palette
    max_colors = len(palette)  # Adjust as needed based on the colormap used
    if n_labels > max_colors:
        raise ValueError(f"Too many unique labels ({n_labels}) for the available colormap ({max_colors} max). Reduce categories or choose a different colormap.")

    # Create a color palette using seaborn (n_labels distinct colors)
    palette = sns.color_palette("tab20", n_labels)

    # Assign colors to each label
    label_to_color = {lbl: palette[i] for i, lbl in enumerate(unique_labels)}

    # Determine figure layout
    n_rows = 1 if n_valid == 20 else 2
    fig, axes = plt.subplots(nrows=n_rows, ncols=N_COLS,
                             figsize=(THUMB_WIDTH, THUMB_HEIGHT if n_rows == 2 else THUMB_HEIGHT//2))
    axes = axes.flatten()

    # Plot each stimulus
    for i, (idx, row) in enumerate(df_valid.iterrows()):

        if n_valid == 20 and i > 19:
            continue

        ax = axes[i]
        ax.set_axis_off()

        # Load the image
        img_path = os.path.join(stim_dir, row['filename'])
        if not os.path.isfile(img_path):
            import logging
            logging.basicConfig(level=logging.INFO)
            logging.warning("Image not found: %s", img_path)
            continue
        img = mpimg.imread(img_path)
        ax.imshow(img)

        # Determine the label and color
        label_value = row[level_col]
        color = label_to_color[label_value]

        # Draw a rectangle (thick border) around the image
        rect = Rectangle(
            (0, 0), 1, 1,  # left, bottom, width, height in Axes coords
            fill=False,
            transform=ax.transAxes,  # so it covers the full Axes
            linewidth=BORDER_LW,
            edgecolor=color
        )
        ax.add_patch(rect)

    # Title and save. Let's make the title bigger, split the "_" and capitalize
    plt.suptitle(f"{level_col.replace('_', ' ').title()}", fontsize=50)
    plt.tight_layout()
    plt.savefig(outpath, dpi=300, bbox_inches='tight')
    plt.show()
    plt.close(fig)

# -------------------------------------------------------------------
# 5) Make the "all stimuli" figure
# -------------------------------------------------------------------

# all_stimuli_outfile = os.path.join(OUTPUT_DIR, "all_stimuli_overview.png")
# plot_all_stimuli(df, STIM_DIR, all_stimuli_outfile)
# logging.info("Saved overview of all stimuli to: %s", all_stimuli_outfile)

# # -------------------------------------------------------------------
# # 6) For each numeric column (except 'filename'), create a color-coded figure
# # -------------------------------------------------------------------

# for col in level_columns:
#     outname = f"plot_{col}.png"
#     outpath = os.path.join(OUTPUT_DIR, outname)
#     logging.info("Creating figure for level '%s'...", col)
#     plot_level(df, STIM_DIR, col, outpath)


tagged_outfile = os.path.join(OUTPUT_DIR, "stimuli_with_tags.png")
plot_all_stimuli_with_dynamic_borders(df, STIM_DIR, tagged_outfile)
logging.info("Saved tagged stimuli overview to: %s", tagged_outfile)

logging.info("All figures have been generated in: %s", OUTPUT_DIR)
