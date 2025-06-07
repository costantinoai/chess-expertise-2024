#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar  9 19:27:53 2024

@author: costantino_ai
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from scipy.spatial.distance import euclidean, correlation, squareform
from scipy.cluster.hierarchy import linkage, dendrogram

from modules import logging, sns, PALETTE, DATASET_CSV_PATH, DNN_RESPONSES_CSV_PATH
from modules.chess import get_all_moves_with_int_eval, check_accuracy
from modules.helpers import (
    create_output_directory,
    save_script_to_file,
    OutputLogger,
    create_run_id,
    calculate_mean_and_ci
    )

def plot_average_accuracy_by_stim_id(df, output_path):
    """
    Creates two subplots (bar plots) of average accuracy (y) by stim_id (x),
    separated by check_board == True (top) and check_board == False (bottom).

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing columns 'stim_id', 'acc', 'expert', and 'check_board'.
    output_path : str
        Output directory to save the resulting plot PDF.
    """
    plt.figure(figsize=(20, 16))

    # Plot 1: Checkmate boards (check_board == True)
    plt.subplot(2, 1, 1)
    sns.barplot(
        x='stim_id',
        y='acc',
        hue='expert',
        data=df[df['check_board'] == True],
        palette={True : PALETTE['Super-experts'], False: PALETTE['Non-experts']},
        errorbar=('ci', 95),
        capsize=.1,
        hue_order=[True, False]
    )
    plt.title('Average Accuracy by Stimulus ID (Checkmate Boards)', fontsize=20, fontweight='bold')
    plt.xlabel('Stimulus ID', fontsize=16, fontweight='bold')
    plt.ylabel('Average Accuracy', fontsize=16, fontweight='bold')
    plt.xticks(fontsize=12, rotation=45)
    plt.yticks(fontsize=12)
    plt.legend(title='Expert Status', fontsize=14, title_fontsize='15')
    plt.tight_layout()

    # Plot 2: Non-checkmate boards (check_board == False)
    plt.subplot(2, 1, 2)
    sns.barplot(
        x='stim_id',
        y='acc',
        hue='expert',
        data=df[df['check_board'] == False],
        palette=[PALETTE['Super-experts'], PALETTE['Non-experts']],
        errorbar=('ci', 95),
        capsize=.1
    )
    plt.title('Average Accuracy by Stimulus ID (Non-Checkmate Boards)', fontsize=20, fontweight='bold')
    plt.xlabel('Stimulus ID', fontsize=16, fontweight='bold')
    plt.ylabel('Average Accuracy', fontsize=16, fontweight='bold')
    plt.xticks(fontsize=12, rotation=45)
    plt.yticks(fontsize=12)
    plt.legend(title='Expert Status', fontsize=14, title_fontsize='15')

    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust if needed
    save_path = os.path.join(output_path, 'average_accuracy_check_vs_noncheck_barplot.pdf')
    plt.savefig(save_path, format='pdf', dpi=300, bbox_inches='tight')
    plt.show()
    logging.info(f"Saved bar plot to {save_path}")

def plot_violin_accuracy_by_stim_id(df, output_path):
    """
    Creates two subplots (violin plots) of accuracy (y) by stim_id (x),
    separated by check_board == True (top) and check_board == False (bottom).

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing columns 'stim_id', 'acc', 'expert', and 'check_board'.
    output_path : str
        Output directory to save the resulting plot PDF.
    """
    fig, axes = plt.subplots(2, 1, figsize=(15, 12))

    titles = [
        'Average Accuracy by Stimulus ID (Checkmate Boards)',
        'Average Accuracy by Stimulus ID (Non-Checkmate Boards)'
    ]

    # Iterate over [True, False] for check_board
    for i, check_status in enumerate([True, False]):
        ax = axes[i]
        data_filtered = df[df['check_board'] == check_status]

        sns.violinplot(
            ax=ax,
            x='stim_id',
            y='acc',
            hue='expert',
            data=data_filtered,
            split=True,
            inner='quart',
            palette='coolwarm'
        )

        ax.set_title(titles[i], fontsize=20, fontweight='bold')
        ax.set_xlabel('Stimulus ID', fontsize=16, fontweight='bold')
        ax.set_ylabel('Average Accuracy', fontsize=16, fontweight='bold')
        ax.tick_params(axis='x', labelsize=12, rotation=45)
        ax.tick_params(axis='y', labelsize=12)
        ax.legend(title='Expert Status', fontsize=14, title_fontsize='15')

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    save_path = os.path.join(output_path, 'average_accuracy_check_vs_noncheck_violin.pdf')
    plt.savefig(save_path, format='pdf', dpi=300, bbox_inches='tight')
    plt.show()
    logging.info(f"Saved violin plot to {save_path}")

def plot_average_accuracy_by_sub_id(df, output_path):
    """
    Creates two subplots (bar plots) for average accuracy grouped by subject ID:
      1) Checkmate boards
      2) Non-checkmate boards

    Each subplot includes separate bars for experts vs. non-experts, and
    also includes horizontal lines + confidence intervals to show overall means.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with columns 'sub_id_zfilled', 'acc', 'expert', 'check_board'.
    output_path : str
        Output directory to save the resulting plot PDF.
    """
    # Separate DataFrame subsets
    experts_checkmate = df[(df['check_board'] == True) & (df['expert'] == True)]
    non_experts_checkmate = df[(df['check_board'] == True) & (df['expert'] == False)]
    experts_non_checkmate = df[(df['check_board'] == False) & (df['expert'] == True)]
    non_experts_non_checkmate = df[(df['check_board'] == False) & (df['expert'] == False)]

    # Calculate mean and CI for each subset
    mean_experts_checkmate, ci_lower_experts_checkmate, ci_upper_experts_checkmate = calculate_mean_and_ci(experts_checkmate['acc'])
    mean_non_experts_checkmate, ci_lower_non_experts_checkmate, ci_upper_non_experts_checkmate = calculate_mean_and_ci(non_experts_checkmate['acc'])
    mean_experts_non_checkmate, ci_lower_experts_non_checkmate, ci_upper_experts_non_checkmate = calculate_mean_and_ci(experts_non_checkmate['acc'])
    mean_non_experts_non_checkmate, ci_lower_non_experts_non_checkmate, ci_upper_non_experts_non_checkmate = calculate_mean_and_ci(non_experts_non_checkmate['acc'])

    fig, axs = plt.subplots(2, 1, figsize=(20, 12))

    # --- Plot for Checkmate Boards ---
    sns.barplot(
        ax=axs[0],
        x='sub_id_zfilled',
        y='acc',
        data=experts_checkmate,
        color=PALETTE['Super-experts'],
        errorbar=('ci', 95),
        label='Experts'
    )
    sns.barplot(
        ax=axs[0],
        x='sub_id_zfilled',
        y='acc',
        data=non_experts_checkmate,
        color=PALETTE['Non-experts'],
        errorbar=('ci', 95),
        label='Non-experts'
    )

    # Horizontal lines + CIs for checkmate boards
    axs[0].axhline(mean_experts_checkmate, color=PALETTE['Super-experts'], ls='--', lw=2)
    axs[0].fill_between(
        axs[0].get_xlim(),
        ci_lower_experts_checkmate,
        ci_upper_experts_checkmate,
        color=PALETTE['Super-experts'],
        alpha=0.2
    )

    axs[0].axhline(mean_non_experts_checkmate, color=PALETTE['Non-experts'], ls='--', lw=2)
    axs[0].fill_between(
        axs[0].get_xlim(),
        ci_lower_non_experts_checkmate,
        ci_upper_non_experts_checkmate,
        color=PALETTE['Non-experts'],
        alpha=0.2
    )

    axs[0].set_title('Average Accuracy for Checkmate Boards by Subject ID')
    axs[0].set_xlabel('Subject ID')
    axs[0].set_ylabel('Average Accuracy')
    axs[0].tick_params(axis='x', rotation=45)
    axs[0].legend()

    # --- Plot for Non-Checkmate Boards ---
    sns.barplot(
        ax=axs[1],
        x='sub_id_zfilled',
        y='acc',
        data=experts_non_checkmate,
        color=PALETTE['Super-experts'],
        errorbar=('ci', 95),
        label='Experts'
    )
    sns.barplot(
        ax=axs[1],
        x='sub_id_zfilled',
        y='acc',
        data=non_experts_non_checkmate,
        color=PALETTE['Non-experts'],
        errorbar=('ci', 95),
        label='Non-experts'
    )

    # Horizontal lines + CIs for non-checkmate boards
    axs[1].axhline(mean_experts_non_checkmate, color=PALETTE['Super-experts'], ls='--', lw=2)
    axs[1].fill_between(
        axs[1].get_xlim(),
        ci_lower_experts_non_checkmate,
        ci_upper_experts_non_checkmate,
        color=PALETTE['Super-experts'],
        alpha=0.2
    )

    axs[1].axhline(mean_non_experts_non_checkmate, color=PALETTE['Non-experts'], ls='--', lw=2)
    axs[1].fill_between(
        axs[1].get_xlim(),
        ci_lower_non_experts_non_checkmate,
        ci_upper_non_experts_non_checkmate,
        color=PALETTE['Non-experts'],
        alpha=0.2
    )

    axs[1].set_title('Average Accuracy for Non-Checkmate Boards by Subject ID')
    axs[1].set_xlabel('Subject ID')
    axs[1].set_ylabel('Average Accuracy')
    axs[1].tick_params(axis='x', rotation=45)
    axs[1].legend()

    plt.tight_layout()
    save_path = os.path.join(output_path, 'average_accuracy_by_sub_id.pdf')
    plt.savefig(save_path, format='pdf', dpi=300, bbox_inches='tight')
    plt.show()
    logging.info(f"Saved average accuracy by sub-id plot to {save_path}")

def plot_correlation_acc_elo(df, output_path):
    """
    Calculate and plot the correlation between average accuracy (acc) and
    average ELO (sub_elo) for experts on checkmate boards only.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with columns 'sub_id', 'acc', 'sub_elo', 'expert', 'check_board'.
    output_path : str
        Output directory to save the resulting plot PDF.
    """
    # Filter for experts and checkmate boards
    experts_data = df[(df['expert'] == True) & (df['check_board'] == True)]

    # Compute average accuracy and ELO per subject
    experts_avg = experts_data.groupby('sub_id').agg({'acc': 'mean', 'sub_elo': 'mean'}).reset_index()

    # Pearson correlation
    correlation_coef, p_value = stats.pearsonr(experts_avg['sub_elo'], experts_avg['acc'])

    # Plot
    plt.figure(figsize=(10, 6))
    sns.regplot(
        x='sub_elo',
        y='acc',
        data=experts_avg,
        ci=95,
        scatter_kws={'s': 50, 'alpha': 0.5}
    )

    plt.title('Correlation of Average Accuracy vs. ELO for Experts (Checkmate Boards Only)', fontsize=16)
    plt.xlabel('ELO', fontsize=14)
    plt.ylabel('Average Accuracy', fontsize=14)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()

    # Annotate with correlation results
    plt.annotate(
        f'Pearson r: {correlation_coef:.3f}\nP-value: {p_value:.3f}',
        xy=(0.05, 0.85),
        xycoords='axes fraction',
        fontsize=12,
        bbox=dict(boxstyle="round,pad=0.3", facecolor='wheat')
    )

    save_path = os.path.join(output_path, 'correlation_acc_ELO.pdf')
    plt.savefig(save_path, format='pdf', dpi=300, bbox_inches='tight')
    plt.show()
    logging.info(f"Saved correlation plot to {save_path}")

def compute_distance_matrix(vectors, distance_metric='euclidean'):
    """
    Compute the pairwise distance matrix for given vectors using the specified distance metric.

    Parameters:
    - vectors (dict): Dictionary where keys are group/model names and values are their feature vectors.
    - distance_metric (str): Distance metric to use ('euclidean' or 'correlation').

    Returns:
    - pd.DataFrame: Distance matrix as a pandas DataFrame.
    """
    if distance_metric not in ['euclidean', 'correlation']:
        raise ValueError("distance_metric must be either 'euclidean' or 'correlation'")

    distance_function = euclidean if distance_metric == 'euclidean' else correlation

    distances = {
        name: {other_name: distance_function(vector, vectors[other_name])
               for other_name in vectors}
        for name, vector in vectors.items()
    }

    return pd.DataFrame(distances)

def plot_spearman_correlation(vectors, output_path):
    """
    Compute and plot the Spearman correlation matrix as a heatmap.

    Parameters:
    - vectors (dict): Dictionary of feature vectors for models and human groups.
    - output_path (str): Directory to save the plot.
    """
    spearman_coefficients = {
        name: {
            other_name: stats.spearmanr(vector, vectors[other_name]).correlation
            for other_name in vectors
        }
        for name, vector in vectors.items()
    }

    correlation_matrix = pd.DataFrame(spearman_coefficients)
    logging.info('Calculated Spearman correlation matrix.')

    # Plot heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0.0, linewidths=.5, fmt=".2f")
    plt.title('Spearman Correlation Matrix of Accuracies')
    plt.tight_layout()

    # Save and show plot
    plot_filename = os.path.join(output_path, 'spearman-correlation-matrix.pdf')
    plt.savefig(plot_filename, format='pdf', dpi=300)
    logging.info(f'Saved Spearman Correlation Coefficient Heatmap at {plot_filename}')
    plt.show()

def plot_human_model_similarity(vectors, output_path, distance_metric='euclidean'):
    """
    Plot similarity between human groups and models based on the selected distance metric.

    Parameters:
    - vectors (dict): Dictionary of feature vectors for models and human groups.
    - output_path (str): Directory to save the plot.
    - distance_metric (str): Distance metric to use ('euclidean' or 'correlation').
    """
    # Compute distance matrix
    distance_matrix = compute_distance_matrix(vectors, distance_metric)

    # Extract human groups and models
    human_groups = [name for name in vectors if "expert" in name.lower() or "non-expert" in name.lower()]
    model_names = [name for name in vectors if name not in human_groups]

    # Convert to long format for seaborn
    distance_long_df = (
        distance_matrix.loc[human_groups, model_names]
        .reset_index()
        .melt(id_vars=['index'], var_name='Model', value_name='Distance')
    )
    distance_long_df.rename(columns={'index': 'Group'}, inplace=True)

    # Plot grouped bar chart
    plt.figure(figsize=(8, 6))
    sns.barplot(x='Group', y='Distance', hue='Model', data=distance_long_df, dodge=True)
    plt.title('Behavioural Alignment', fontweight='bold')
    plt.xlabel('Groups')
    plt.ylabel(f'{distance_metric.capitalize()} Distance')
    plt.legend(title='Model', loc='upper right')
    plt.tight_layout()

    # Save and show plot
    plot_filename = os.path.join(output_path, f'{distance_metric}-distance-humans-models.pdf')
    plt.savefig(plot_filename, format='pdf', dpi=300)
    logging.info(f'Saved {distance_metric.capitalize()} Distance plot at {plot_filename}')
    plt.show()

def plot_dendrogram(vectors, output_path, distance_metric='euclidean'):
    """
    Plot a hierarchical clustering dendrogram based on pairwise distances.

    Parameters:
    - vectors (dict): Dictionary of feature vectors for models and human groups.
    - output_path (str): Directory to save the plot.
    - distance_metric (str): Distance metric to use ('euclidean' or 'correlation').
    """
    distance_matrix = compute_distance_matrix(vectors, distance_metric)

    # Convert to condensed form for hierarchical clustering
    condensed_distances = squareform(distance_matrix)
    linkage_matrix = linkage(condensed_distances, method='ward')

    # Plot dendrogram
    plt.figure(figsize=(12, 3))
    dendrogram(linkage_matrix, labels=distance_matrix.columns)
    plt.title('Hierarchical Clustering Dendrogram', fontweight='bold')
    plt.xticks(rotation=0)
    plt.yticks([])  # Remove y-axis ticks
    plt.tick_params(axis='x', which='major', labelsize=16)
    plt.tight_layout()

    # Save and show plot
    plot_filename = os.path.join(output_path, f'{distance_metric}-distance-dendrogram.pdf')
    plt.savefig(plot_filename, format='pdf', dpi=300)
    logging.info(f'Saved {distance_metric.capitalize()} Distance Dendrogram plot at {plot_filename}')
    plt.show()

def load_and_sort_data(dataset_path, dnn_responses_path, humans_response_path):
    """
    Load datasets into pandas DataFrames and sort them by 'stim_id'.

    Parameters:
    - dataset_path (str): Path to the dataset CSV file.
    - dnn_responses_path (str): Path to the DNN responses CSV file.
    - humans_response_path (str): Path to the human responses CSV file.

    Returns:
    - tuple: Sorted pandas DataFrames (dataset_df, net_df, humans_df).
    """
    dataset_df = pd.read_csv(dataset_path).sort_values(by='stim_id')
    net_df = pd.read_csv(dnn_responses_path).sort_values(by='stim_id')
    humans_df = pd.read_csv(humans_response_path).sort_values(by='stim_id')

    # Zero-pad 'sub_id' for consistent formatting
    humans_df['sub_id_zfilled'] = humans_df['sub_id'].apply(lambda x: f"{x:02d}")

    logging.info('Datasets loaded and sorted by stim_id.')
    return dataset_df, net_df, humans_df

def extract_human_vectors(humans_df):
    """
    Extract accuracy vectors for experts and non-experts from human responses.

    Parameters:
    - humans_df (pd.DataFrame): DataFrame containing human response data.

    Returns:
    - dict: Dictionary containing accuracy vectors for 'Experts' and 'Non-experts'.
    """
    # Filter only checkmate boards
    humans_df_checkmate = humans_df[humans_df['check_board']]
    logging.debug('Filtered checkmate boards for human responses.')

    # Compute mean accuracy per stimulus for experts and non-experts
    experts_vector = (humans_df_checkmate[humans_df_checkmate['expert']]
                      .groupby('stim_id')['acc']
                      .mean()
                      .sort_index()
                      .values)

    non_experts_vector = (humans_df_checkmate[~humans_df_checkmate['expert']]
                          .groupby('stim_id')['acc']
                          .mean()
                          .sort_index()
                          .values)

    logging.debug('Grouped human responses by expertise level.')
    return {
        'Non-experts': np.array(non_experts_vector, dtype=float),
        'Experts': np.array(experts_vector, dtype=float),
    }

def extract_model_vectors(dataset_df, net_df):
    """
    Compute accuracy vectors for Lc0 and AlphaZero models.

    Parameters:
    - dataset_df (pd.DataFrame): DataFrame containing ground truth moves.
    - net_df (pd.DataFrame): DataFrame containing model responses.

    Returns:
    - dict: Dictionary containing accuracy vectors for 'Lc0' and 'AlphaZero'.
    """
    # Extract model predictions
    lc0_moves = net_df['LC0']
    a0_moves = net_df['AlphaZero']

    # Compute correct moves using Stockfish evaluation
    correct_moves = dataset_df.apply(
        lambda row: get_all_moves_with_int_eval(row['stockfish_top_5'], row['stockfish_eval']), axis=1)

    logging.debug('Extracted LC0 and AlphaZero moves.')

    # Compute accuracy vectors for each model
    lc0_vector = [resp == moves[0] for resp, moves in zip(lc0_moves, correct_moves) if moves][:20]
    a0_vector = [check_accuracy(resp, moves) for resp, moves in zip(a0_moves, correct_moves)][:20]

    logging.info('Calculated accuracy vectors for models.')
    return {
        'Lc0': np.array(lc0_vector, dtype=float),
        'AlphaZero': np.array(a0_vector, dtype=float),
    }

################

humans_response_path = './results/20250213-193046_familiarization-task-group/fam_task_results.csv'
out_root = 'results'

# Create results dir and save script
output_path = os.path.join(out_root, f'{create_run_id()}_familiarization-task-plots')
create_output_directory(output_path)
save_script_to_file(output_path)
logging.info("Output folder created and script file saved")

out_text_file = os.path.join(output_path, 'bh_data_exploration.log')

with OutputLogger(True, out_text_file):

    # Load the datasets into pandas DataFrames and sort them by 'stim_id'
    dataset_df, net_df, humans_df = load_and_sort_data(DATASET_CSV_PATH, DNN_RESPONSES_CSV_PATH, humans_response_path)

    # Get human and models accuracy vectors, ordered by stimulus id
    humans_vectors = extract_human_vectors(humans_df)
    models_vectors = extract_model_vectors(dataset_df, net_df)
    all_vectors = {**models_vectors, **humans_vectors}

    # Create each plot
    plot_average_accuracy_by_stim_id(humans_df, output_path)

    plot_violin_accuracy_by_stim_id(humans_df, output_path)

    plot_average_accuracy_by_sub_id(humans_df, output_path)

    plot_correlation_acc_elo(humans_df, output_path)

    # Compute and plot Spearman correlation matrix
    plot_spearman_correlation(all_vectors, output_path)

    # Compute and plot human-model similarity (Euclidean distance)
    plot_human_model_similarity(all_vectors, output_path, distance_metric='euclidean')

    # Generate and plot dendrogram (Euclidean distance)
    plot_dendrogram(all_vectors, output_path, distance_metric='euclidean')
