import os
import glob
import pandas as pd
import numpy as np
import nibabel as nib
from surfer import Brain
from scipy import stats
import re
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def save_dataframes(out_dir, **dataframes):
    """
    Saves multiple pandas DataFrames to CSV files in the specified directory. The filenames are based on the
    keys provided in the `dataframes` argument. It's expected that the keys represent informative names
    for the data contained within the DataFrames.

    Parameters
    ----------
    out_dir : str
        The directory where the CSV files will be saved.
    **dataframes : dict
        Keyword arguments where keys are descriptive names for the DataFrames and the values are the DataFrames
        themselves.

    Usage
    -----
    >>> save_dataframes('/path/to/output', expert_averages_df=expert_averages_df, expert_ci_lower_df=expert_ci_lower_df)
    """
    import os

    # Ensure the output directory exists
    os.makedirs(out_dir, exist_ok=True)

    # Iterate through each provided DataFrame
    for name, df in dataframes.items():
        # Generate a filename from the DataFrame name
        filename = f"{name}.csv"
        # Save the DataFrame to CSV
        df.to_csv(os.path.join(out_dir, filename))
        
def atoi(text):
    return int(text) if text.isdigit() else text


def natural_keys(text):
    """
    Apply a natural sort algorithm that is case-insensitive.
    """
    # Convert the text to lowercase to achieve case-insensitive sorting
    return [atoi(c) for c in re.split(r"(\d+)", text.lower())]


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
            t_stat, p_value = stats.ttest_1samp(concatenated_df.loc[idx, roi], popmean=0.0, nan_policy='omit')
            t_values_df.loc[idx, roi] = t_stat
            p_values_df.loc[idx, roi] = p_value

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

    # Find the index of this label in the original list (adjusting for 0-based index)
    original_index = original_list.index(roi_name)

    return hemisphere, original_index

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
            hemi, label_index = reverse_roi_name_to_annotation_label(roi, lh_names, rh_names)

            if hemi == "lh":
                vertices_lh = np.where(lh_labels == label_index)[0]
                overlay_data_lh[vertices_lh] = value if p_value < significance_level else 0.0
            elif hemi == "rh":
                vertices_rh = np.where(rh_labels == label_index)[0]
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
    
    data_limit = max(np.abs(global_max), np.abs(global_min))
    global_min = -data_limit if data_limit != 0.0 else -0.01
    global_max = data_limit if data_limit != 0.0 else 0.01
    
    # Iterate through contrasts and hemispheres
    for index, contrast in enumerate(contrasts):
        for hemi_index, hemi in enumerate(["lh", "rh"]):
            for view_index, view in enumerate(["lateral", "medial", "ventral"]):
                # Create and configure brain
                brain = Brain("fsaverage", hemi, "inflated", title="", subjects_dir=freesurfer_path, views=view)
                
                overlay_data_hemi = overlay_data[index][0] if hemi == 'lh' else overlay_data[index][1]
                brain.add_data(
                    overlay_data_hemi,
                    min=global_min,
                    max=global_max,
                    mid=0,
                    colormap="seismic",
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
    norm = plt.Normalize(global_min, global_max)
    sm = plt.cm.ScalarMappable(cmap="seismic", norm=norm)
    sm.set_array([])
    
    # Create the colorbar with specified ticks for min, max, and avg
    cb = fig.colorbar(sm, cax=cax, orientation='vertical')
    cb.set_ticks([global_min, 0, global_max])
    cb.set_ticklabels([f"{-global_max:.2f}", "0", f"{global_max:.2f}"])
    cb.ax.tick_params(labelsize=12, colors='white')  # Adjust tick font size and color
    
    # Set the colorbar's label
    cb.set_label('Regression Coefficient', color='white', fontsize=14)

    # Save the final plot
    plt.savefig(os.path.join(out_path, "combined_brain_grid.png"))
    plt.show()
    return os.path.join(out_path, "combined_brain_grid.png")

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
)

# Define contrasts
contrasts = ("checkmate", "visual", "strategy")

# Example usage
root_path = "/data/projects/chess/data/BIDS/derivatives/mvpa/rsa_glm_hpc"
freesurfer_path = "/data/projects/chess/data/BIDS/derivatives/fastsurfer"
lh_annotation = "/data/projects/chess/data/BIDS/derivatives/fastsurfer/fsaverage/label/lh.HCPMMP1.annot"
rh_annotation = "/data/projects/chess/data/BIDS/derivatives/fastsurfer/fsaverage/label/rh.HCPMMP1.annot"

plots_paths = []

for multiple_comparions_corr in [True, False]:
    # Main code
    fdr_string = "fdr" if multiple_comparions_corr==True else "uncorrected"
    
    exp_path = os.path.join(root_path, "single_rois", f"experts-{fdr_string}")
    nonexp_path = os.path.join(root_path, "single_rois", f"non-experts-{fdr_string}")
    
    os.makedirs(exp_path, exist_ok=True)
    os.makedirs(nonexp_path, exist_ok=True)
    
    # Load and average TSV files
    expert_dfs = load_tsv_files(root_path, expertSubjects)
    nonexpert_dfs = load_tsv_files(root_path, nonExpertSubjects)
    
    expert_averages_df, expert_ci_lower_df, expert_ci_upper_df, expert_p_values_df = (
        calculate_stats(expert_dfs, chance_levels=[0.0, 0.0, 0.0], multiple_comp_corr=fdr_string)
    )
    nonexpert_averages_df, nonexpert_ci_lower_df, nonexpert_ci_upper_df, nonexpert_p_values_df = (
        calculate_stats(nonexpert_dfs, chance_levels=[0.0, 0.0, 0.0], multiple_comp_corr=fdr_string)
    )
    
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
fig.savefig(os.path.join(root_path, "single_rois", 'brain_images_grid.png'), dpi=300, bbox_inches='tight', pad_inches=0, facecolor=fig.get_facecolor())
