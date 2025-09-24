#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 16 20:50:12 2024

@author: costantino_ai
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 16 17:22:42 2024

@author: costantino_ai
"""
import os
import logging
import re
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

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
    plt.figure(figsize=(12, 6))  # Increased width for a more elongated plot
    sns.set(style="darkgrid", font_scale=1.2)  # Set dark grid style and increase font size
    
    df = df.T  # Transpose to plot columns as separate lines

    # Plot lines with or without confidence intervals
    for i, curve in enumerate(df.columns):
        sns.lineplot(x=df.index, y=df[curve] * 100, label=labels[i])
        if confidence_intervals is not None and curve in confidence_intervals.columns:
            plt.fill_between(df.index, confidence_intervals[curve]['low'], confidence_intervals[curve]['high'], alpha=0.3)
    
    plt.xticks(rotation=30, ha="right")
    plt.xlabel("Region")  # Add x-axis label, customize as necessary
    plt.ylabel("Accuracy - Chance %")  # Add y-axis label, customize as necessary
    
    # Place the legend inside the plot, reduce its size
    plt.legend(loc='upper right', fontsize='small')

    plt.title(title)

    # Adjust layout to ensure all elements are included in the figure
    plt.tight_layout()

    if save_to is not None:
        # Sanitize the title to create a valid filename
        filename = re.sub(r"[^a-zA-Z0-9\-]", "-", title) + ".pdf"

        # Check if save_to is a directory or a file
        if os.path.isdir(save_to):
            save_path = os.path.join(save_to, filename.lower())
        elif os.path.isfile(save_to):
            logging.warning("'save_to' is a file. Using its directory and renaming the file.")
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
    num_groups = len(df.columns)  # Number of groups
    bar_width = 0.5  # Width of each bar
    group_spacing = 2  # Factor to increase the spacing between groups
    positions = [i * group_spacing for i in range(num_groups)]  # Adjusted positions for the bars
    
    plt.figure(figsize=(14 + group_spacing, 7))  # Adjust figure size if needed
    sns.set(style="darkgrid", font_scale=1.3)  # Use dark grid style for better contrast and scaling fonts for readability
    
    for i, (index, row) in enumerate(df.iterrows()):
        mean_values = row.values * 100  # Convert proportions to percentages
        errors = [confidence_intervals.loc[index, 'low'] * 100, confidence_intervals.loc[index, 'high'] * 100]
        
        bars = plt.bar([p + bar_width * i for p in positions], mean_values, bar_width,
                       yerr=[mean_values - errors[0], errors[1] - mean_values],
                       label=labels[i], capsize=3, alpha=0.75)

        # Mark significant bars with an asterisk
        for bar, p_value in zip(bars, significance.loc[index]):
            if p_value < 0.05:
                plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 2.5, '*', 
                         ha='center', va='bottom', color='red', fontsize=15)

    plt.axhline(0, color='red', linestyle='dotted', linewidth=1.5)  # Dotted line at y=0 for reference
    plt.xlabel("Region")
    plt.ylabel("Accuracy - Chance (%)")
    plt.xticks([p + bar_width * (len(df) / 2 - 0.5) for p in positions], df.columns, rotation=30, ha='right')
    plt.legend(loc='upper right', fontsize='small')
    plt.title(title)
    plt.tight_layout()

    if save_to:
        filename = re.sub(r"[^a-zA-Z0-9\-]", "-", title) + ".pdf"
        save_path = os.path.join(save_to, filename.lower())
        plt.savefig(save_path, format="pdf", dpi=300, bbox_inches="tight")

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
    for exp in ['experts','non-experts']:
        out_dir = f'/data/projects/chess/data/BIDS/derivatives/mvpa/{mvpa_type}/roi_groups/{exp}-fdr'
        # Load dataframes
        expert_df_flag = "expert" if exp == "experts" else"nonexpert"
        df = pd.read_csv(os.path.join(out_dir, f'{expert_df_flag}_averages_df.csv'), index_col=0)
        df_ci_low = pd.read_csv(os.path.join(out_dir, f'{expert_df_flag}_ci_lower_df.csv'), index_col=0)
        df_ci_up = pd.read_csv(os.path.join(out_dir, f'{expert_df_flag}_ci_upper_df.csv'), index_col=0)
        df_p_values = pd.read_csv(os.path.join(out_dir, f'{expert_df_flag}_p_values_df.csv'), index_col=0)
        
        # Combine CI data into a single DataFrame
        df_ci = pd.concat([df_ci_low.loc[:, sorted_columns_lh + sorted_columns_rh], df_ci_up.loc[:, sorted_columns_lh + sorted_columns_rh]], axis=1, keys=['low', 'high'])
        
        # Plotting
        plot_labels = ["Checkmate", "Pixel similarity", "Strategy"]  # Define labels for plots
        
        # Plotting for each hemisphere with bar plots
        plot_curves(df[sorted_columns_lh], 'Left Hemisphere Analysis', plot_labels, confidence_intervals=df_ci.loc[:, pd.IndexSlice[:, sorted_columns_lh]], save_to=None)
        plot_curves(df[sorted_columns_rh], 'Right Hemisphere Analysis', plot_labels, confidence_intervals=df_ci.loc[:, pd.IndexSlice[:, sorted_columns_rh]], save_to=None)
        
        # Plotting for each hemisphere
        plot_bars(df[sorted_columns_lh], 'Left Hemisphere Analysis', plot_labels, df_ci.loc[:, pd.IndexSlice[:, sorted_columns_lh]], df_p_values[sorted_columns_lh], save_to=out_dir)
        plot_bars(df[sorted_columns_rh], 'Right Hemisphere Analysis', plot_labels, df_ci.loc[:, pd.IndexSlice[:, sorted_columns_rh]], df_p_values[sorted_columns_rh], save_to=out_dir)
    
