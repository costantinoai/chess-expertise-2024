#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 16 18:21:16 2024

@author: costantino_ai
"""
import pandas as pd
import os, glob
import mat73
import scipy
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import warnings
from sklearn.manifold import MDS
from matplotlib.patches import Rectangle

plt.rcParams.update({'font.size': 23})
warnings.filterwarnings("ignore", category=FutureWarning)  # Mutes future warnings
warnings.filterwarnings("ignore", category=UserWarning)  # Mutes custom user warnings

OUT_ROOT = "/home/eik-tb/OneDrive_andreaivan.costantino@kuleuven.be/Presentations/2024-08-06_CCN-poster/poster/material"

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

def plot_mds(dissimilarity_matrix, title="MDS"):
    mds = MDS(n_components=2, dissimilarity="precomputed", random_state=42)
    results = mds.fit(dissimilarity_matrix)
    coords = results.embedding_

    strategy_colors, strategy_alpha = determine_color_and_alpha(STRATEGIES)

    # Plotting
    plt.figure(figsize=(12, 10))
    for (x, y, color, alpha) in zip(coords[:, 0], coords[:, 1], strategy_colors, strategy_alpha):
        plt.scatter(x, y, color=color, marker='o', alpha=alpha,s=120)

    plt.title(title, fontsize=30, pad=20)
    # plt.xlabel('Dimension 1')
    # plt.ylabel('Dimension 2')
    plt.xticks([])
    plt.yticks([])

    fname = os.path.join(OUT_ROOT, title)
    plt.savefig(fname, bbox_inches='tight', pad_inches=0.1)
    plt.show()

# Define normalize_response function adapted for pairwise comparisons
def create_pairwise_data(df):
    # This variable stores the current button mapping. 51 and 52 are the button codes
    # 1 indicates that the CURRENT board is preferred over the previous one
    # -1 indicates that the PREVIOUS board is preferred over the current one
    button_mapping = {
        1: {
            1: 51,
            -1: 52,
            0: 0,
        },
        2: {-1: 51, 1: 52, 0: 0},
    }
    # This function creates a pairwise DataFrame where each row represents a comparison
    comparisons = []
    for i in range(len(df) - 1):
        if int(df.iloc[i]["response"]) != 0:  # Only compare when there's a preference
            _bm = button_mapping[int(df.iloc[i + 1]["button_mapping"])]


            preferred = (
                int(df.iloc[i]["stim_id"])
                if int(df.iloc[i]["response"])
                == _bm[1]
                else int(df.iloc[i-1]["stim_id"])
            )

            non_preferred = (
                int(df.iloc[i]["stim_id"])
                if int(df.iloc[i]["response"])
                == _bm[-1]
                else int(df.iloc[i-1]["stim_id"])
            )

            comparisons.append({"better": preferred, "worse": non_preferred, "sub_id":df.iloc[i]["sub_id"]})

    return pd.DataFrame(comparisons)

def plot_rdm(rdm, title="RDM", colormap="viridis"):
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
        cbar_kws={"label": "Preference Discrepancy Score"},
        ax=ax
    )

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

    fname = os.path.join(OUT_ROOT, title)
    plt.savefig(fname, bbox_inches='tight', pad_inches=0.1)

    plt.show()

def rdm_sym(sub_df, expertise, plot=False):
    """
    Constructs a dissimilarity matrix from pairwise comparison data, visualizes it, and returns the matrix.

    Args:
        sub_df (DataFrame): DataFrame containing the pairwise comparison data with 'better' and 'worse' columns.
        sub (str): Identifier for the subject, used for plot titling.
        expertise (str): Expertise level of the subject, used for plot titling.

    Returns:
        ndarray: A symmetric dissimilarity matrix where each element i, j represents the dissimilarity between stimuli i and j.

    The function performs the following operations:
    - Extracts pairwise comparisons and counts how often each stimulus is preferred over another.
    - Constructs a symmetric matrix where the difference between reciprocal preferences is taken as a measure of dissimilarity.
    - Visualizes this matrix as a heatmap.

    Dissimilarity Interpretation:
    - The matrix entry (i, j) represents how dissimilar stimulus i is from stimulus j based on how frequently they are chosen over each other.
    - A higher score indicates a greater preference discrepancy between two stimuli, suggesting less similarity in how they are perceived.
    """

    # Create pairwise comparison data
    all_comparisons = create_pairwise_data(sub_df)

    # Count occurrences of each stimulus being preferred over another
    comparison_counts = all_comparisons.groupby(['better', 'worse']).size().unstack(fill_value=0)

    # Calculate the dissimilarity matrix by subtracting the transpose of the counts from the counts themselves
    # and taking the absolute value. This reflects the preference discrepancy between each pair of stimuli.
    diss_matrix = np.nan_to_num(comparison_counts.sub(comparison_counts.T).abs())

    if plot:
        title = f"Behavioural RDM - {expertise}"
        plot_rdm(diss_matrix, title, colormap="viridis")

    return diss_matrix

def dsm_directional_preferences(df_, expertise, plot=False):
    """
    Constructs and visualizes a directional preference matrix from pairwise comparison data,
    indicating the direction of preferences between stimuli.

    Args:
        df_ (DataFrame): DataFrame containing pairwise comparison data with 'better' and 'worse' columns.
        sub (str): Identifier for the subject, used for plot titling.
        expertise (str): Expertise level of the subject, used for plot titling.

    Returns:
        ndarray: A directional preference matrix where each element (i, j) represents how much more stimulus i is preferred over j compared to j over i.

    The function performs the following operations:
    - Extracts pairwise comparisons and counts how often each stimulus is preferred over another.
    - Constructs a directional preference matrix by subtracting the transpose of the counts from the counts without taking the absolute value.
    - Visualizes this matrix as a heatmap, where positive values indicate a preference for the row stimulus over the column stimulus and negative values indicate the opposite.

    Interpretation:
    - A positive score in matrix entry (i, j) indicates that stimulus i is preferred more often over stimulus j.
    - A negative score in matrix entry (i, j) indicates that stimulus j is preferred more often over stimulus i.
    - This matrix is not symmetric and focuses on the directionality of preferences rather than just the magnitude of differences.
    """

    # Create pairwise comparison data
    all_comparisons = create_pairwise_data(df_)

    # Count occurrences of each stimulus being preferred over another
    comparison_counts = all_comparisons.groupby(['better', 'worse']).size().unstack(fill_value=0)

    # Calculate the directional preference matrix by subtracting the transpose of the counts from the counts
    directional_preference_matrix = np.nan_to_num(comparison_counts.sub(comparison_counts.T))

    if plot:
        title = f'Stimulus Preference - {expertise}'
        plot_rdm(directional_preference_matrix, title, colormap="coolwarm")

    return directional_preference_matrix

def analyze_preferences(df_, sub, expertise, plot=False):
    """
    Analyze preferences using a Bradley-Terry model and visualize the frequency of choices.

    Args:
        df_ (DataFrame): DataFrame containing pairwise comparison data with 'better' and 'worse' columns.
        sub (str): Subject identifier to personalize the output and plot titles.
        expertise (str): Expertise level of the subject to include in the plot titles.

    Returns:
        Series: Probabilities of each stimulus being chosen over others, excluding the baseline (constant term).

    The function performs the following steps:
    - Create pairwise comparison data from the DataFrame.
    - Plot the choice frequency of each stimulus.
    - Prepare the data for logistic regression, where 'better' is treated as the dependent variable and 'worse' as the independent variable.
    - Fit a multinomial logistic regression (Bradley-Terry model) to the data.
    - Extract and convert the logistic regression logits (excluding the intercept) to probabilities.
    - Interpretation of these probabilities provides insight into the preference strength relative to a baseline which is implicitly defined by the logistic regression model.
    """

    # Create pairwise data, ensuring it has the correct structure for analysis
    all_comparisons = create_pairwise_data(df_)

    if plot:

        # Plot choice frequency using an external function to visualize how often each stimulus was chosen
        plot_choice_frequency(all_comparisons, sub, expertise)

    return

def plot_choice_frequency(data, subject, expertise):
    """
    Generates a bar plot for the frequency of each choice made.

    Args:
        data (DataFrame): DataFrame containing 'better' column with stimulus IDs.
        subject (str): Identifier for the subject to be included in the plot title.
        expertise (str): The expertise level of the subject for contextualizing the plot.
    """
    freq = data['better'].value_counts().sort_index()
    plt.figure(figsize=(12, 11))
    strategy_colors, strategy_alpha = determine_color_and_alpha(STRATEGIES)
    bar_plot = sns.barplot(x=freq.index, y=freq.values, palette=strategy_colors)
    # Adjust alpha for each bar
    for bar, alpha in zip(bar_plot.patches, strategy_alpha):
        bar.set_alpha(alpha)

    title = f'Pairwise Stimulus Choice Frequency - {expertise}'
    plt.title(title, fontsize=30, pad=20)
    plt.xlabel('Stimulus')
    plt.ylabel('Pairwise Choice Frequency')
    # Setting the y-ticks to minimum and maximum values on the plot
    plt.xticks([])
    plt.tick_params(labelsize=15)

    fname = os.path.join(OUT_ROOT, title)
    plt.savefig(fname, bbox_inches='tight', pad_inches=0.1)

    plt.show()


# !!! In these plots there are a few interesting things to note:
#   1. Frequencies:
#       - Experts prefer more often checkmate boards to non-checkmate one, and prefer easy boards over others
#       - Novices show similar plofiles between check vs. non-check, suggesting that they rely on visual properties in the decision!
#       - It is possible that some experts did not understand the task and pressed the wrong button.
#            See plots where the reds are up and greens down (7, 8, 16)
#   2. RDMs and MDS:
#       - When pooling all subjects, we see clear checkmate clusters + more. At the average
#           level, we also see some clustering and particularly some other similar things
#           to brain RDMs and valueHead in a0.
#       TODO: re-check the brain and a0 RDMs and compare to behaviour. It may explain some patterns we saw previously.
#   3. Directional DSM:
#       - This is very similar to the RDM, but it is not symmetrical as it focuses on the direction.
#           These plots clearly show that checkmate boards are preferred over non-check!


from config import BIDS_PATH
bids_root = str(BIDS_PATH)
participants_xlsx_path = "/home/eik-tb/OneDrive_andreaivan.costantino@kuleuven.be/Projects/Expertise/chess_files/chess_project_files/participants.xlsx"
participants_sourcedata_root = "/media/costantino_ai/eiK-backup2/chess/sourcedata"

# Load the participants excel file
participants_df = pd.read_excel(participants_xlsx_path)

# Filter rows where Expert column is not NaN
filtered_df = participants_df.dropna(subset=["Expert"])

# Create list of tuples
participants = [
    ("sub-{0:02}".format(sub_id), bool(expert))
    for sub_id, expert in zip(
        filtered_df["sub_id"].astype(int).astype(str).str.zfill(2), filtered_df["Expert"]
    )
]

# Sum all boolean values in the 'Expert' column
total_experts = int(filtered_df["Expert"].sum())
total_non_experts = len(filtered_df) - int(filtered_df["Expert"].sum())
import logging
logging.basicConfig(level=logging.INFO)
logging.info("Experts: %s | Non-experts: %s", total_experts, total_non_experts)

# Define the column names
columns = [
    "sub_id",  # Column 1: Replicate subject number across all trials.
    "run",  # Column 2: Replicate run number across all trials.
    "run_trial_n",  # Column 3: Sequential trial numbers.
    "stim_id",  # Column 4: Randomized stimulus indices, ensuring varied presentation.
    "stim_onset_real",  # Column 5: Placeholder for actual stimulus onset times.
    "response",  # Column 6: Placeholder for response data.
    "stim_onset_expected",  # Column 7: Calculate ideal stimulus onset times based on predefined durations and offsets.
    "button_mapping",  # Column 8: Apply button mapping uniformly across trials.
]

# Aggregate dataframes
all_comparisons = []
novices_list_rdm = []
experts_list_rdm = []
novices_list_distdsm = []
experts_list_distdsm = []

long_df = pd.DataFrame([], columns=columns)

novices_df = pd.DataFrame([], columns=columns)
experts_df = pd.DataFrame([], columns=columns)

for sub, exp in participants:
    logging.info("%s", sub)
    sub_dir = os.path.join(participants_sourcedata_root, sub, "bh")
    mat_files = sorted(glob.glob(os.path.join(sub_dir, "*.mat")))
    # assert len(mat_files) != 0
    if len(mat_files) == 0:
        continue

    # Create main dataframe
    sub_df = pd.DataFrame([], columns=columns)

    for mat_file in mat_files:
        try:
            ts, sub_id, run, task = os.path.basename(mat_file).split("_")
        except:
            sub_id, run, task = os.path.basename(mat_file).split("_")

        task = task.split(".")[0]

        if task != "exp":
            continue

        # Load the .mat file
        try:
            mat = mat73.loadmat(mat_file)
        except:
            mat = scipy.io.loadmat(mat_file)

        # Extract relevant info
        # stims = tuple([stim[0] for stim in mat["imList"]["name"]])
        trials = mat["trialList"]

        assert len(trials) != 0

        # Make sub temporary df
        run_df = pd.DataFrame(trials, columns=columns)

        if run_df["response"].sum() <= 0:
            warnings.warn(
                f"No non-zero responses found for participant {sub}, run {run}."
            )
            continue  # We skip if the participant did not respond
        else:
            # Append data to main dataframe
            sub_df = pd.concat([sub_df, run_df], ignore_index=True)

    # Check if the sum of responses is zero or not
    if sub_df["response"].sum() <= 0:
        # Raise a warning indicating no non-zero responses were found for this participant
        warnings.warn(f"No non-zero responses found for participant {sub}")
        continue

    expertise = "Expert" if exp else "Novice"

    # Calculate probabilities of chosing stimuli, and plot frequencies and probs
    analyze_preferences(sub_df, sub, expertise, plot=False)

    # Build DSM
    sub_rdm = rdm_sym(sub_df, expertise, plot=False)
    sub_directional_dsm = dsm_directional_preferences(sub_df, expertise, plot=False)

    if exp:
        experts_list_rdm.append(sub_rdm)
        experts_list_distdsm.append(sub_directional_dsm)
        experts_df = pd.concat([experts_df, sub_df], ignore_index=True)

    else:
        novices_list_rdm.append(sub_rdm)
        novices_list_distdsm.append(sub_directional_dsm)
        novices_df = pd.concat([novices_df, sub_df], ignore_index=True)



# Apply the analysis for each group
for i, df_ in enumerate([experts_df, novices_df]):
    expertise = "Experts" if i == 0 else "Novices"

    analyze_preferences(df_, 'All Subjects', expertise, plot=True)

    dissimilarity_matrix = rdm_sym(df_, expertise, plot=True)
    plot_mds(pd.DataFrame(dissimilarity_matrix, columns = list(range(40))), title = f'Behavioural MDS - {expertise}')

    sub_directional_dsm = dsm_directional_preferences(df_, expertise, plot=True)
