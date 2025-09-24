#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 16 13:35:14 2024

@author: costantino_ai
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 11 17:35:11 2024

@author: costantino_ai
"""

import os
import glob
import pandas as pd
import numpy as np
import nibabel as nib
from surfer import Brain
from scipy import stats
import re
import logging
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def load_tsv_files(root_path, sub_list):
    """
    Searches for all TSV files within subfolders of a specified root path and imports them.

    Parameters
    ----------
    root_path : str
        The root directory to search for TSV files.

    Returns
    -------
    list of pandas.DataFrame
        A list of dataframes imported from each found TSV file.
    """
    dfs = [
        pd.read_csv(
            glob.glob(os.path.join(root_path, f"sub-{sub}", "*.tsv"))[0], sep="\t"
        )
        for sub in sub_list
    ]
    return dfs


def calculate_stats(dfs, chance_levels, multiple_comp_corr='fdr'):
    """
    Extends the original function to include an option for multiple comparisons correction and
    calculates effect sizes for each comparison.

    Parameters
    ----------
    dfs : list of pandas.DataFrame
        A list of dataframes for which to calculate statistics.
    chance_levels : list of float
        The chance levels to use for the t-test comparison, applied per row.
    multiple_comp_corr : str, optional
        The type of multiple comparisons correction to apply. Can be None for no correction,
        or 'fdr' for False Discovery Rate correction. Defaults to 'fdr'.

    Returns
    -------
    tuple
        A tuple of five DataFrames: averages, CI lower bound, CI upper bound, (adjusted) p-values,
        and effect sizes.
    """
    concatenated_df = pd.concat(dfs, axis=0)
    # Adjust the dataframe for chance level
    # Subtract chance level from each value based on its group (index)
    for idx, chance_level in enumerate(chance_levels):
        concatenated_df.loc[idx] = concatenated_df.loc[idx].subtract(chance_level)

    # Calculate mean, standard deviation, and count
    stats_df = concatenated_df.groupby(level=0).agg(['mean', 'std', 'count'])

    # Initializing result dataframes
    effect_sizes_df = pd.DataFrame(index=stats_df.index, columns=stats_df.columns.levels[0])
    t_values_df = effect_sizes_df.copy()
    p_values_df = effect_sizes_df.copy()
    ci_lower_df = effect_sizes_df.copy()
    ci_upper_df = effect_sizes_df.copy()

    for idx in stats_df.index:
        n = stats_df.loc[idx, (slice(None), 'count')].iloc[0]  # Assuming equal count for all ROIs
        for roi in stats_df.columns.levels[0]:
            mean = stats_df.loc[idx, (roi, 'mean')]
            std = stats_df.loc[idx, (roi, 'std')]
            se = std / np.sqrt(n)
            ci_multiplier = stats.norm.ppf(0.975)  # Z value for 95% CI

            # Calculate CI
            ci_lower_df.loc[idx, roi] = mean - ci_multiplier * se
            ci_upper_df.loc[idx, roi] = mean + ci_multiplier * se

            # Calculate effect size (Cohen's d)
            effect_sizes_df.loc[idx, roi] = mean / std if std != 0 else 0

            # Perform t-test against the chance level
            # Perform t-test against the chance level (two tailed)
            # t_stat, p_value = stats.ttest_1samp(concatenated_df.loc[idx, roi], popmean=0.0, nan_policy='omit')

            # Calculate the one-tailed t-test on a single sample
            t_stat, p_value_one_tailed = stats.ttest_1samp(concatenated_df.loc[idx, roi], popmean=0.0, nan_policy='omit', alternative='greater')

            t_values_df.loc[idx, roi] = t_stat
            p_values_df.loc[idx, roi] = p_value_one_tailed

    # Apply multiple comparisons correction if specified
    if multiple_comp_corr == 'fdr':
        for idx in stats_df.index:
            from statsmodels.stats.multitest import multipletests
            p_values_flat = p_values_df.loc[idx].values.flatten()
            _, pvals_corrected, _, _ = multipletests(p_values_flat.astype(np.float64), method='fdr_bh', alpha=0.01)
            p_values_df.loc[idx] = pvals_corrected

    return stats_df.xs('mean', axis=1, level=1), ci_lower_df, ci_upper_df, p_values_df

def reverse_roi_name_to_annotation_label(roi_name, lh_labels_list, rh_labels_list):
    """
    Given a volume label, finds the corresponding label name in the sorted list for the
    appropriate hemisphere, then returns the 0-based index of this label in the original,
    unsorted list for that hemisphere.

    Parameters
    ----------
    volume_label : int
        The label number in the volume.
    lh_labels_list : list
        Original list of labels in the left hemisphere.
    rh_labels_list : list
        Original list of labels in the right hemisphere.

    Returns
    -------
    str
        The hemisphere ('lh', 'rh', or 'subcortical') the label belongs to.
    int
        The 0-based index of the label in the original, unsorted list for the specified hemisphere.
    """

    hemisphere_prefix = roi_name[0]

    if hemisphere_prefix == "L":
        hemisphere = "lh"
        original_list = [str(label) for label in np.array(lh_labels_list).astype(str)]
    elif hemisphere_prefix == "R":
        hemisphere = "rh"
        original_list = [str(label) for label in np.array(rh_labels_list).astype(str)]
    else:
        raise ValueError("Invalid hemisphere prefix in volume label.")

    # Get the list of rois for the selected group
    selected_rois = [f"{hemisphere_prefix}_{roi}_ROI" for roi in ROI_MAPPING[roi_name[2:]]]

    # Find the index of this label in the original list (adjusting for 0-based index)
    original_indices = [original_list.index(selected_roi) for selected_roi in selected_rois if selected_roi in original_list]

    return hemisphere, original_indices


def color_significant_parcels_on_flat(freesurfer_path, lh_annotation, rh_annotation, averages_df, p_values_df, contrasts, out_path, significance_level=0.05):
    """
    Create brain images with significant parcels colored on a flattened surface using FreeSurfer annotation files.
    It first calculates a global color range and then creates brain figures in a grid plot with shared color scales.

    Parameters
    ----------
    freesurfer_path : str
        Path to the FreeSurfer subjects directory.
    lh_annotation : str
        Annotation file for the left hemisphere.
    rh_annotation : str
        Annotation file for the right hemisphere.
    averages_df : pandas.DataFrame
        DataFrame containing average values for coloring.
    p_values_df : pandas.DataFrame
        DataFrame containing p-values for each parcel.
    contrasts : list
        List of contrast names used for titles in plots.
    out_path : str
        Output path to save the brain images.
    significance_level : float, optional
        The threshold below which a parcel is considered significant, by default 0.05.
    """

    # Load annotation files
    lh_labels, lh_ctab, lh_names = nib.freesurfer.read_annot(lh_annotation)
    rh_labels, rh_ctab, rh_names = nib.freesurfer.read_annot(rh_annotation)

    # Prepare to collect min and max values across all plots for consistent color scaling
    global_min = float('inf')
    global_max = float('-inf')

    # Storage for the overlay data to apply later
    overlay_data = []

    # Preparing the overlay data for each contrast
    for index, row in averages_df.iterrows():
        overlay_data_lh = np.full(lh_labels.shape, 0.0)
        overlay_data_rh = np.full(rh_labels.shape, 0.0)

        p_values_list = p_values_df.loc[index]  # p-values for this index

        for roi, p_value in zip(averages_df.columns, p_values_list):
            value = row[roi]
            hemi, label_indices = reverse_roi_name_to_annotation_label(roi, lh_names, rh_names)

            if hemi == "lh":
                vertices_lh = np.where(np.isin(lh_labels, list(set(label_indices))))[0]
                overlay_data_lh[vertices_lh] = value if p_value < significance_level else 0.0
            elif hemi == "rh":
                vertices_rh = np.where(np.isin(rh_labels, list(set(label_indices))))[0]
                overlay_data_rh[vertices_rh] = value if p_value < significance_level else 0.0

        # Calculate min and max values across all plots
        current_max = max(np.max(overlay_data_lh), np.max(overlay_data_rh))
        if current_max > global_max:
            global_max = current_max

        current_min = min(np.min(overlay_data_lh), np.min(overlay_data_rh))
        if current_min < global_min:
            global_min = current_min

        overlay_data.append((overlay_data_lh, overlay_data_rh))

    # Generate and save brain images with consistent color scaling across plots
    # Initialize figure for the grid plot
    fig, axs = plt.subplots(nrows=3, ncols=len(contrasts)*2, figsize=(4*len(contrasts)*2, 12), facecolor='black')  # Adjust figure size as needed
    fig.subplots_adjust(right=0.85)  # Adjust subplot to fit colorbar
    # Create the truncated version of the 'seismic' colormap, focusing on the red to white part
    truncated_seismic = plt.cm.colors.ListedColormap(plt.cm.seismic(np.linspace(0.5, 1, 256)))

    # Iterate through contrasts and hemispheres
    for index, contrast in enumerate(contrasts):
        for hemi_index, hemi in enumerate(["lh", "rh"]):
            for view_index, view in enumerate(["lateral", "medial", "ventral"]):
                # Create and configure brain
                brain = Brain("fsaverage", hemi, "inflated", title="", subjects_dir=freesurfer_path, views=view)

                overlay_data_hemi = overlay_data[index][0] if hemi == 'lh' else overlay_data[index][1]
                brain.add_data(
                    overlay_data_hemi,
                    min=0.0,
                    max=global_max,
                    colormap=truncated_seismic,
                    hemi=hemi,
                    colorbar=False  # Colorbar will be added later globally
                )

                # Save and load image to place on grid
                img_path = os.path.join(out_path, f"brain_{hemi}_{view}_{contrast}.png")
                brain.save_image(img_path)
                brain.close()

                img = mpimg.imread(img_path)
                os.remove(img_path)
                ax = axs[view_index, index * 2 + hemi_index]
                ax.imshow(img)
                ax.axis('off')  # Hide axes for clean look
                if view_index == 0:  # Set contrast title only on the first row
                    ax.set_title(f"{contrast.upper()}", color='white', fontsize=12)


    # Create colorbar in the last axis space
    cax = fig.add_axes([0.87, 0.15, 0.015, 0.7])  # Make the colorbar thinner
    norm = plt.Normalize(0.0, global_max)
    sm = plt.cm.ScalarMappable(cmap=truncated_seismic, norm=norm)
    sm.set_array([])

    # Create the colorbar with specified ticks for min, max, and avg
    cb = fig.colorbar(sm, cax=cax, orientation='vertical')
    cb.set_ticks([0.0, global_max])
    cb.set_ticklabels(["0 %", f"{global_max*100:.2f} %"])
    cb.ax.tick_params(labelsize=12, colors='white')  # Adjust tick font size and color

    # Set the colorbar's label
    cb.set_label('Accuracy - Chance %', color='white', fontsize=14)

    # Save the final plot
    plt.savefig(os.path.join(out_path, "combined_brain_grid.png"))
    plt.show()
    return os.path.join(out_path, "combined_brain_grid.png")

def group_rois_in_dfs(dfs_list):
    """
    Groups Regions of Interest (ROIs) in each DataFrame within a list according to
    predefined mappings, averaging the values of smaller ROIs into their respective larger
    ROI categories for both left and right hemispheres.

    Each DataFrame in the input list is expected to contain columns named with the pattern
    "{hemi}_{ROI}_ROI", where "{hemi}" is either 'L' or 'R' (for left and right hemisphere,
    respectively), and "{ROI}" is the name of the smaller ROI.

    The function returns a list of new DataFrames, each corresponding to an input DataFrame
    but with the columns grouped and averaged according to the larger ROI mappings.

    Parameters
    ----------
    dfs_list : list of pandas.DataFrame
        List of DataFrames, each containing data for different ROIs across left and right hemispheres.

    Returns
    -------
    list of pandas.DataFrame
        List of new DataFrames with ROIs grouped and averaged into larger categories.

    Usage
    -----
    >>> dfs_grouped = group_rois_in_dfs([df1, df2])
    >>> dfs_grouped[0]  # Access the first grouped DataFrame
    """

    # List to store the grouped DataFrames
    dfs_grouped = []

    # Iterate over each DataFrame in the input list
    for df in dfs_list:
        # Create a new DataFrame to hold the grouped ROIs
        grouped_df = pd.DataFrame(index=df.index)

        # Process each hemisphere
        for hemi in ['L', 'R']:
            # Process each larger ROI group
            for large_roi, small_rois in ROI_MAPPING.items():
                # Initialize a list to store data for all smaller ROIs in the current group
                small_roi_data = []

                # Collect data for each smaller ROI in the group
                for small_roi in small_rois:
                    column_name = f"{hemi}_{small_roi}_ROI"
                    if column_name in df.columns:
                        # Add the data for the current smaller ROI to the list
                        small_roi_data.append(df[column_name])
                    else:
                        logging.warning(f"{column_name} not found in the dataframe!!")

                # If we have collected any data, average it and add to the grouped DataFrame
                if small_roi_data:
                    # Calculate the mean across the smaller ROIs for the current larger ROI group
                    averaged_data = np.mean(small_roi_data, axis=0)
                    # Assign the averaged data to a new column in the grouped DataFrame
                    grouped_df[f"{hemi}_{large_roi}"] = averaged_data

        # Add the processed DataFrame to the list of grouped DataFrames
        dfs_grouped.append(grouped_df)

    return dfs_grouped


# Lists of subject IDs for each group
expertSubjects = (
    "03",
    "04",
    "06",
    "07",
    "08",
    "09",
    "10",
    "11",
    "12",
    "13",
    "16",
    "20",
    "22",
    "23",
    "24",
    "29",
    "30",
    "33",
    "34",
    "36",
    #"38"
)
nonExpertSubjects = (
    "01",
    "02",
    "15",
    "17",
    "18",
    "19",
    "21",
    "25",
    "26",
    "27",
    "28",
    "32",
    "35",
    "37",
    "39",
    "40",
    "41",
    "42",
    "43",
    "44"
)

# Define contrasts
# contrasts = ("checkmate", "visual", "strategy")
# chance_levels = [0.5, 0.05, 0.1]

contrasts = ('VisualStimuli', 'Strategies', 'Side', 'Difficulty', 'Motif', 'First Piece', 'Last Piece')
chance_levels = [1/x for x in [20,5,2, 2,2, 4, 4, 4]]

# Example usage
root_path = "/data/projects/chess/data/BIDS/derivatives/mvpa/svm_hpc"
freesurfer_path = "/data/projects/chess/data/BIDS/derivatives/fastsurfer"
lh_annotation = "/data/projects/chess/data/BIDS/derivatives/fastsurfer/fsaverage/label/lh.HCPMMP1.annot"
rh_annotation = "/data/projects/chess/data/BIDS/derivatives/fastsurfer/fsaverage/label/rh.HCPMMP1.annot"

plots_paths = []

csv_path='/data/projects/chess/data/misc/HCP-MMP1_UniqueRegionList.csv'

ROI_MAPPING = get_roi_mapping(csv_path)

for multiple_comparions_corr in [True, False]:
    # Main code
    fdr_string = "fdr" if multiple_comparions_corr==True else "uncorrected"

    exp_path = os.path.join(root_path, "roi_lobes", f"experts-{fdr_string}")
    nonexp_path = os.path.join(root_path, "roi_lobes", f"non-experts-{fdr_string}")

    os.makedirs(exp_path, exist_ok=True)
    os.makedirs(nonexp_path, exist_ok=True)

    # Load TSV files
    expert_dfs = load_tsv_files(root_path, expertSubjects)
    nonexpert_dfs = load_tsv_files(root_path, nonExpertSubjects)

    # Group columns according to mapping
    experts_dfs_roigroup = group_rois_in_dfs(expert_dfs)
    nonexpert_dfs_roigroup = group_rois_in_dfs(nonexpert_dfs)

    # Average dfs across subjects and get stats
    expert_averages_df, expert_ci_lower_df, expert_ci_upper_df, expert_p_values_df = (
        calculate_stats(experts_dfs_roigroup, chance_levels=chance_levels, multiple_comp_corr=fdr_string)
    )
    (
        nonexpert_averages_df,
        nonexpert_ci_lower_df,
        nonexpert_ci_upper_df,
        nonexpert_p_values_df,
    ) = calculate_stats(nonexpert_dfs_roigroup, chance_levels=chance_levels, multiple_comp_corr=fdr_string)

    # Save stats
    save_dataframes(
        nonexp_path,
        nonexpert_averages_df=nonexpert_averages_df,
        nonexpert_ci_lower_df=nonexpert_ci_lower_df,
        nonexpert_ci_upper_df=nonexpert_ci_upper_df,
        nonexpert_p_values_df=nonexpert_p_values_df
    )
    save_dataframes(
        exp_path,
        expert_averages_df=expert_averages_df,
        expert_ci_lower_df=expert_ci_lower_df,
        expert_ci_upper_df=expert_ci_upper_df,
        expert_p_values_df=expert_p_values_df,
    )

    # Color parcels on flat surface for significant results
    plot_dir = color_significant_parcels_on_flat(
        freesurfer_path,
        lh_annotation,
        rh_annotation,
        expert_averages_df,
        expert_p_values_df,
        contrasts,
        exp_path,
    )
    plots_paths.append([multiple_comparions_corr, 'exp', plot_dir])


    plot_dir = color_significant_parcels_on_flat(
        freesurfer_path,
        lh_annotation,
        rh_annotation,
        nonexpert_averages_df,
        nonexpert_p_values_df,
        contrasts,
        nonexp_path,
    )
    plots_paths.append([multiple_comparions_corr, 'control', plot_dir])

# Assuming each image has the same dimensions
img_example = mpimg.imread(plots_paths[0][2])
img_height, img_width, _ = img_example.shape

# Calculate figure size based on image size and number of images
fig_width = 2 * img_width / 100  # 2 images per row, width in inches (100 px/inch is a guess, adjust as needed)
fig_height = 2 * img_height / 100  # 2 rows, height in inches

# Initialize figure for the grid plot without gaps between images
fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(fig_width, fig_height), facecolor='black', gridspec_kw={'wspace':0, 'hspace':0})

# Load and display each image
for i, (correction, title, img_path) in enumerate(plots_paths):
    img = mpimg.imread(img_path)
    ax = axs[i // 2, i % 2]
    ax.imshow(img)
    ax.axis('off')  # Hide axes for a clean look
    display_title = f"{title} FDR correction" if correction else f"{title} un-corrected"
    ax.set_title(display_title.upper(), color='white', fontsize=21)

# Show the plot
plt.tight_layout(pad=0)  # Adjust padding to ensure no space between subplots
plt.show()

# Save the figure to a file
fig.savefig(os.path.join(root_path, "roi_lobes", 'brain_images_grid.png'), dpi=300, bbox_inches='tight', pad_inches=0, facecolor=fig.get_facecolor())
