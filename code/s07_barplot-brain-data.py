#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 16 17:22:42 2024

@author: costantino_ai
"""
import os
import re
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def plot_curves(df, title, labels, save_to=None, confidence_intervals=None):
    """
    Plots curves from a DataFrame and adds specified confidence interval shading,
    with an adjusted dark grid style suitable for publications.

    Parameters:
    - df (pd.DataFrame): The DataFrame containing the data to be plotted.
    - title (str): The title of the plot.
    - save_to (str, optional): The path to save the plot. If None, the plot is not saved.
    - confidence_intervals (pd.DataFrame, optional): DataFrame containing confidence intervals.
            Each row corresponds to a curve, and each column contains the confidence interval values.
    Returns:
    - None: Displays the plot and optionally saves it.
    """
    plt.figure(figsize=(15, 7))  # Increased width for a more elongated plot to match previous style
    sns.set(style="darkgrid", font_scale=1.75)  # Set dark grid style and increase font size to match previous style
    
    df = df.T  # Transpose to plot columns as separate lines

    # Plot lines with or without confidence intervals
    for i, curve in enumerate(df.columns):
        sns.lineplot(x=df.index, y=df[curve] * 100, label=labels[i], linewidth=2.5)  # Increased line thickness for consistency
        if confidence_intervals is not None and curve in confidence_intervals.columns:
            plt.fill_between(df.index, confidence_intervals[curve]['low'], confidence_intervals[curve]['high'], alpha=0.3)
    
    plt.xticks(rotation=30, ha="right", fontsize=14)  # Standardize font size for x-axis ticks
    plt.xlabel("Region")  # Add x-axis label
    plt.ylabel("Accuracy - Chance %")  # Add y-axis label
    plt.tick_params(axis='y', labelsize=10)  # Adjust y-axis tick labels to a smaller size for consistency

    # Place the legend inside the plot, adjust font size for consistency
    plt.legend(loc='upper right', fontsize='small')

    plt.title(title, fontsize=20, fontweight='bold')  # Make title bigger and bold

    # Adjust layout to ensure all elements are included in the figure
    plt.tight_layout()

    if save_to is not None:
        # Sanitize the title to create a valid filename
        filename = re.sub(r"[^a-zA-Z0-9\-]", "-", title) + ".pdf"

        # Determine the path to save the plot
        if os.path.isdir(save_to):
            save_path = os.path.join(save_to, filename.lower())
        elif os.path.isfile(save_to):
            print("Warning: 'save_to' is a file. Using its directory and renaming the file.")
            save_path = os.path.join(os.path.dirname(save_to), filename)
        else:
            # If save_to does not exist, assume it's a directory and attempt to create it
            os.makedirs(save_to, exist_ok=True)
            save_path = os.path.join(save_to, filename)

        # Save the plot
        plt.savefig(save_path, format="pdf", dpi=300, bbox_inches="tight")

    # Show the plot
    plt.show()

def plot_bars(df, title, labels, confidence_intervals, significance, save_to=None):
    """
    Plots bar graphs from a DataFrame with error bars and significance markers,
    formatted to align with specified publication-quality standards.

    Parameters:
    - df (pd.DataFrame): DataFrame containing the data to be plotted.
    - title (str): The title of the plot.
    - labels (list): List of labels for each dataset.
    - confidence_intervals (pd.DataFrame): DataFrame containing confidence interval values.
    - significance (pd.DataFrame): DataFrame containing p-values to mark significance on the bars.
    - save_to (str, optional): Directory path to save the plot as a PDF. If None, the plot is not saved.

    Each row in `df`, `confidence_intervals`, and `significance` should correspond to a different group or category,
    with columns representing different conditions or experiments.

    Returns:
    - None: Displays the plot and optionally saves it if a path is provided.
    """
    num_groups = len(df.columns)  # Determine the number of groups to plot
    bar_width = 0.5  # Set the width of each bar
    group_spacing = 2  # Factor to define spacing between groups of bars
    
    # Calculate positions for each group of bars
    positions = [i * group_spacing for i in range(num_groups)]
    
    # Set up the figure
    plt.figure(figsize=(15, 7))  # Maintain consistent size with previous plots
    sns.set(style="darkgrid", font_scale=1.75)  # Set visual style for the plot
    
    # Plot each group of bars
    for i, (index, row) in enumerate(df.iterrows()):
        mean_values = row.values * 100  # Scale values to percentage
        errors = [confidence_intervals.loc[index, 'low'] * 100, confidence_intervals.loc[index, 'high'] * 100]
        
        # Create bars with error bars, set edgecolor to 'none' or to match the bar color
        bars = plt.bar([p + bar_width * i for p in positions], mean_values, bar_width,
                       yerr=[mean_values - errors[0], errors[1] - mean_values],
                       label=labels[i], capsize=3, alpha=0.75, edgecolor='none')

        # Add significance markers
        for bar, p_value in zip(bars, significance.loc[index]):
            if p_value < 0.05:  # Mark significant bars
                plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 2.5, '*',
                         ha='center', va='bottom', color='red', fontsize=15)

    plt.axhline(0, color='red', linestyle='dotted', linewidth=1.5)  # Reference line at y=0
    plt.xlabel("Region", fontsize=14)
    plt.ylabel("Accuracy - Chance (%)", fontsize=14)
    plt.xticks([p + bar_width * (len(df) / 2 - 0.5) for p in positions], df.columns, rotation=30, ha='right', fontsize=14)
    plt.yticks(fontsize=10)  # Adjust y-axis tick labels to a smaller size for consistency
    
    # Clean y-tick labels
    xtick_labels = [tick.get_text()[2:].replace('_', ' ') for tick in plt.gca().get_xticklabels()]
    plt.gca().set_xticklabels(xtick_labels)
    
    plt.legend(loc='upper right', fontsize='small')
    plt.title(title, fontsize=20, fontweight='bold')  # Make title bigger and bold
    plt.tight_layout()

    # Save the plot if a path is provided
    if save_to:
        filename = re.sub(r"[^a-zA-Z0-9\-]", "-", title) + ".pdf"
        save_path = os.path.join(save_to, filename.lower())
        plt.savefig(save_path, format="pdf", dpi=300, bbox_inches="tight")

    # Display the plot
    plt.show()
    
def invert_name(name):
    """
    Converts a given column name to match the corresponding region name in coord_df.
    Assumes that column names are in the format 'L_V1_ROI' and need to be converted to 'V1_L'.
    
    Parameters:
        name (str): The column name in the format 'L_V1_ROI'.

    Returns:
        str: Converted region name in the format 'V1_L'.
    """
    parts = name.split('_')
    if len(parts) < 3:
        return name  # No inversion possible if the format does not include enough parts
    return '_'.join(parts[1:-1]) + '_' + parts[0]  # Rearrange to 'V1_L'
    
# Data Loading
coord_df = pd.read_csv('/home/eik-tb/Desktop/test/HCP-MMP1_UniqueRegionList.csv')
z_cog_mapping = coord_df.set_index('cortex')['Cortex_ID'].to_dict()  # Create mapping of cortex to Cortex_ID

# Sorting logic for hemisphere specific data
sorted_z_cog_list = sorted(z_cog_mapping.items(), key=lambda x: x[1])  # Sort cortex by Cortex_ID
# excluded_roi_ids = list(range(6, 15))  # Define ROIs to exclude
excluded_roi_ids = []  # Define ROIs to exclude

# Excluding specific ROIs and preparing columns for plotting
sorted_columns_rh = [f"R_{col}" for col, idx in sorted_z_cog_list if int(idx) not in excluded_roi_ids]
sorted_columns_lh = [f"L_{col}" for col, idx in sorted_z_cog_list if int(idx) not in excluded_roi_ids]

for mvpa_type in ["rsa_glm_hpc", "svm_hpc"]:
    for exp in ['experts', 'non-experts']:
        out_dir = f'/data/projects/chess/data/BIDS/derivatives/mvpa/{mvpa_type}/roi_groups/{exp}-fdr'
        
        # Load dataframes
        expert_df_flag = "expert" if exp == "experts" else "nonexpert"
        df = pd.read_csv(os.path.join(out_dir, f'{expert_df_flag}_averages_df.csv'), index_col=0)
        df_ci_low = pd.read_csv(os.path.join(out_dir, f'{expert_df_flag}_ci_lower_df.csv'), index_col=0)
        df_ci_up = pd.read_csv(os.path.join(out_dir, f'{expert_df_flag}_ci_upper_df.csv'), index_col=0)
        df_p_values = pd.read_csv(os.path.join(out_dir, f'{expert_df_flag}_p_values_df.csv'), index_col=0)
        
        # Combine CI data into a single DataFrame
        df_ci = pd.concat([df_ci_low.loc[:, sorted_columns_lh + sorted_columns_rh], df_ci_up.loc[:, sorted_columns_lh + sorted_columns_rh]], axis=1, keys=['low', 'high'])
        
        # Define the desired order of plots (assuming these are known and consistent)
        plot_labels = ["Visual", "Strategy", "Checkmate"]  # Desired order
        desired_order = {"Checkmate": 0, "Visual": 1, "Strategy": 2}  # Map labels to their current index

        # Reorder DataFrames according to 'plot_labels'
        df = df.reindex([desired_order[label] for label in plot_labels])
        df_ci = df_ci.reindex([desired_order[label] for label in plot_labels])
        df_p_values = df_p_values.reindex([desired_order[label] for label in plot_labels])

        # Plotting for each hemisphere with bar plots
        plot_curves(df[sorted_columns_lh], 'Left Hemisphere Analysis', plot_labels, confidence_intervals=df_ci.loc[:, pd.IndexSlice[:, sorted_columns_lh]], save_to=out_dir)
        plot_curves(df[sorted_columns_rh], 'Right Hemisphere Analysis', plot_labels, confidence_intervals=df_ci.loc[:, pd.IndexSlice[:, sorted_columns_rh]], save_to=out_dir)

        # Plotting for each hemisphere
        plot_bars(df[sorted_columns_lh], 'Left Hemisphere Analysis', plot_labels, df_ci.loc[:, pd.IndexSlice[:, sorted_columns_lh]], df_p_values[sorted_columns_lh], save_to=out_dir)
        plot_bars(df[sorted_columns_rh], 'Right Hemisphere Analysis', plot_labels, df_ci.loc[:, pd.IndexSlice[:, sorted_columns_rh]], df_p_values[sorted_columns_rh], save_to=out_dir)
