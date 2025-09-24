#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Group-Level RSA Meta-Analysis Pipeline (Experts vs. Novices)
============================================================

This script performs a full second-level representational similarity analysis (RSA)
comparing expert and novice participants across multiple searchlight-derived correlation
maps. It is designed for clarity, reproducibility, and modularity, with logging and
visualizations throughout.

Workflow Summary:
-----------------
1. Locate and split subject-level RSA maps by group (experts vs. novices).
2. Apply Fisher r-to-z transformation to stabilize correlation values.
3. Construct the second-level GLM design matrix (with intercept and group contrast).
4. Fit a second-level GLM with optional spatial smoothing (default: 6 mm FWHM).
5. Compute and plot the contrast (Experts > Novices) as glass brain and surface plots.
6. Split the resulting z-map into positive and negative maps for directional analysis.
7. Correlate directional z-maps with meta-analytic Neurosynth term maps.
8. Generate visualizations and LaTeX/CSV summary tables of correlation results.

Author: costantino_ai (refactored)
Date: 2025-06-11
"""

import os  # Filesystem operations
import logging  # Structured logging
import numpy as np  # Numerical arrays
import pandas as pd  # DataFrame tables
import matplotlib.pyplot as plt  # Visualization
from matplotlib.colors import LinearSegmentedColormap  # Custom colormaps
import nibabel as nib  # NIfTI image I/O
from nilearn import plotting, image, surface, datasets
from nilearn.image import load_img, math_img
import pingouin as pg  # Bootstrap Pearson correlation
from statsmodels.stats.multitest import fdrcorrection  # FDR correction
from joblib import Parallel, delayed  # Parallel processing
from tqdm import tqdm  # Progress bars
from plotly.subplots import make_subplots  # Multi-view 3D cortical plots
import warnings

from modules.io_utils import load_term_maps
from modules.stats_utils import save_latex_correlation_tables, generate_latex_multicolumn_table
from modules.plot_utils import plot_correlations, plot_difference
from modules.run_utils import (
    create_run_id,
    create_output_directory,
    save_script_to_file,
    add_file_logger,
)

# Suppress specific qfac warning from nibabel
warnings.filterwarnings("ignore", message="pixdim\[0\] \(qfac\) should be 1 \(default\) or -1; setting qfac to 1")
warnings.filterwarnings(
    "ignore",
    message="vmin cannot be chosen when cmap is symmetric",
    module="nilearn.plotting.surf_plotting"
)

# Set up logging to INFO level
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------- GLOBAL PLOT STYLE ----------------------
# Base font size for glass brain plots and others
base_font_size = 22
plt.rcParams.update({
    "font.family": 'Ubuntu Condensed',
    "font.size": base_font_size,
    "axes.titlesize": base_font_size * 1.4,  # 36.4 ~ 36
    "axes.labelsize": base_font_size * 1.2,  # 31.2 ~ 31
    "xtick.labelsize": base_font_size,  # 26
    "ytick.labelsize": base_font_size,  # 26
    "legend.fontsize": base_font_size,  # 26
    "figure.figsize": (12, 9),  # wide figures
})


def get_brain_mask(ref):
    """
    Loads the ICBM152 2009 gray matter (GM) probabilistic mask, resamples it
    to the space of a reference image, and binarizes it.

    Voxels with GM probability > 0.5 are retained in the final binary mask.
    This is useful for masking analyses to gray-matter voxels only.

    Parameters
    ----------
    ref : Nifti1Image
        A NIfTI image whose shape, resolution, and affine will be used to
        resample the GM mask (e.g., a group average or subject-level map).

    Returns
    -------
    binary_mask : Nifti1Image
        A binarized 3D NIfTI image where voxels with GM > 0.5 are True (1.0),
        and all others are False (0.0), in the same space as `ref`.
    """

    # 1. Fetch the ICBM152 2009 gray matter mask (MNI152 space, 2mm resolution)
    mni = datasets.fetch_icbm152_2009()
    brain_mask = nib.load(mni['gm'])

    # 2. Resample the GM mask to match the shape and space of the reference image
    resampled_mask = image.resample_to_img(brain_mask, ref, interpolation='nearest')

    # 3. Threshold and binarize: keep voxels where GM probability > 0.5
    binary_mask = math_img('img > 0.5', img=resampled_mask)

    return binary_mask


def remove_useless_data(data: np.ndarray, brain_mask_flat: np.ndarray = None, mask_negative=True):
    """
    Remove unusable voxels (NaNs, infs, zero variance, or outside brain mask)
    from a stack of flattened image data maps. Optionally zero-out negative values.

    Parameters
    ----------
    data : np.ndarray, shape (n_maps, n_voxels)
        Array of maps, where each row is a different image, and each column a voxel.

    brain_mask_flat : np.ndarray of bool, optional
        1D array representing the brain mask. If provided, voxels outside the brain are removed.

    mask_negative : bool, default=True
        If True, set all negative values in the data array to 0 before processing.

    Returns
    -------
    data_clean : np.ndarray
        The input data with invalid voxels removed (still 2D).

    keep_mask : np.ndarray of bool
        Boolean mask (length n_voxels) indicating which voxels were kept.
    """
    if data.ndim != 2:
        raise ValueError("remove_useless_data expects a 2D array")

    n_voxels = data.shape[1]
    logger.info(f"Initial number of voxels: {n_voxels}")

    # --- Optional: Mask out negative values before any processing ---
    if mask_negative:
        logger.info("Masking negative values: setting all values < 0 to 0")
        data = np.where(data < 0, 0, data)

    # --- Step 1: Remove voxels with any non-finite value (NaN, inf, -inf) ---
    finite_mask = np.all(np.isfinite(data), axis=0)
    logger.info(f"Voxels with all finite values: {np.sum(finite_mask)} "
                f"({n_voxels - np.sum(finite_mask)} removed)")

    # --- Step 2: Remove voxels with near-zero variance (uninformative) ---
    variance = np.var(data, axis=0)
    var_thresh = 0
    low_variance_mask = variance < var_thresh
    logger.info(f"Voxels with variance >= {var_thresh}: {np.sum(~low_variance_mask)} "
                f"({np.sum(low_variance_mask)} removed)")

    keep_mask = finite_mask & (~low_variance_mask)

    # --- Step 3: Apply brain mask if provided ---
    if brain_mask_flat is not None:
        if brain_mask_flat.shape[0] != data.shape[1]:
            raise ValueError("brain_mask_flat must have shape (n_voxels,)")

        brain_mask_flat = brain_mask_flat.astype(bool)
        logger.info(f"Voxels in brain mask: {np.sum(brain_mask_flat)} "
                    f"({n_voxels - np.sum(brain_mask_flat)} excluded outside brain)")
        keep_mask &= brain_mask_flat

    logger.info(f"Final number of voxels retained: {np.sum(keep_mask)} "
                f"({n_voxels - np.sum(keep_mask)} total removed)")

    return data[:, keep_mask], keep_mask

def make_brain_cmap():
    """
    Create a custom LinearSegmentedColormap blending cool blues for negative
    and pinks for positive values, centered at zero.
    Returns
    -------
    LinearSegmentedColormap
    """
    # center color from RdPu colormap at zero
    center = plt.cm.RdPu(0)[:3]
    # negative tail: interpolate from blue-teal to center
    neg = np.linspace([0.0, 0.5, 0.7], center, 256)
    # positive tail: take full RdPu colormap
    pos = plt.cm.RdPu(np.linspace(0, 1, 256))[:, :3]
    # stack and create new colormap
    return LinearSegmentedColormap.from_list('custom_brain', np.vstack((neg, pos)))

def plot_map(arr, ref_img, title, outpath, thresh=None):
    """
    Render and save a glass brain plot from a 3D numpy array.

    Parameters
    ----------
    arr : np.ndarray
        3D image data array to visualize. Typically a thresholded or z-stat map.

    ref_img : nibabel.Nifti1Image
        Reference image providing affine transformation and spatial metadata.

    title : str
        Title text displayed at the top of the brain plot.

    outpath : str
        File path (PNG) where the resulting image will be saved.

    thresh : float, optional
        Value used to threshold the image before plotting.
        Voxels with absolute values below this will not be displayed.
        If None, no thresholding is applied (default = norm.isf(ALPHA)).

    Returns
    -------
    None
    """
    # Create a NIfTI image from the data array, using the reference image for spatial metadata
    img = image.new_img_like(ref_img, arr)

    # Display a glass brain plot showing multiple anatomical views (L, R, etc.)
    display = plotting.plot_glass_brain(
        img,
        display_mode='lyrz',      # multiple views: left, right, sagittal, axial
        colorbar=True,            # include colorbar
        cmap=BRAIN_CMAP,          # custom or predefined colormap
        symmetric_cbar=True,      # ensure colorbar is centered around zero
        plot_abs=False,           # preserve sign (important for positive/negative effects)
        threshold=thresh          # optional statistical threshold
    )

    # Add the figure title with enhanced formatting
    display.title(title, size=base_font_size * 1.4,
                  color='black', bgcolor='white', weight='bold')

    # Save the figure to disk, if path provided
    if outpath:
        display.savefig(outpath)

    # Show the plot in interactive mode (useful in notebooks or scripts)
    plt.show()

    # Close the figure and free resources
    display.close()

def save_and_open_plotly_figure(fig, title='surface_plot', outdir='.', png_out=None):
    """
    Save a Plotly figure to HTML and optionally to PNG, then open the HTML in a browser.

    Parameters
    ----------
    fig : plotly.graph_objects.Figure
        The Plotly figure to save and open.

    title : str
        Title used to generate output filenames (e.g., 'Visual Similarity').

    outdir : str
        Directory where output files will be saved. Will be created if it doesn't exist.

    png_out : str or None
        Optional path to save a PNG snapshot. If None, PNG will be saved as <title>_surface.png.

    Returns
    -------
    None
    """
    import os

    # Ensure output directory exists
    os.makedirs(outdir, exist_ok=True)

    # Sanitize title to create safe filenames
    basename = title.replace(' ', '_').replace('|', '').replace(':', '')
    html_path = os.path.join(outdir, f"{basename}.html")

    # Save HTML
    fig.write_html(html_path)
    logger.info(f"Figure saved to: {html_path}")
    logger.info(f"Figure saved to: {html_path}")

    # # Open HTML in browser
    # webbrowser.open_new_tab(f'file://{html_path}')

    # Determine PNG path if not explicitly provided
    if png_out is None:
        png_out = os.path.join(outdir, f"{basename}_surface.png")

    # Save PNG
    fig.write_image(png_out, scale=2)
    logger.info(f"PNG image saved to: {png_out}")
    logger.info(f"PNG image saved to: {png_out}")

def plot_surface_map(img, title='Surface Map', threshold=None, output_file=None):
    """
    Project a volumetric image onto the cortical surface and create a 3D interactive Plotly figure.

    Parameters
    ----------
    img : nibabel.Nifti1Image
        3D statistical brain image to project and visualize.

    title : str
        Title of the final figure.

    threshold : float or None
        Value to threshold the statistical map (values below are masked out).

    output_file : str or None
        Optional full path to save the resulting PNG image.
        The HTML version will be saved in the same directory, named from `title`.

    Returns
    -------
    None
    """
    fsaverage = datasets.fetch_surf_fsaverage()
    views = [
        ('medial', 'left'), ('lateral', 'left'),
        ('medial', 'right'), ('lateral', 'right')
    ]
    surface_types = ['inflated']

    mesh_dict = {
        'pial': {'left': fsaverage.pial_left, 'right': fsaverage.pial_right},
        'inflated': {'left': fsaverage.infl_left, 'right': fsaverage.infl_right},
    }
    sulc_dict = {'left': fsaverage.sulc_left, 'right': fsaverage.sulc_right}
    view_angles = {
        'lateral': dict(x=2, y=0, z=0.1),
        'medial': dict(x=-2, y=0, z=0.1),
        'dorsal': dict(x=0, y=0, z=2),
        'ventral': dict(x=0, y=0, z=-2),
        'posterior': dict(x=0, y=-2, z=0.1),
    }

    n_rows = len(surface_types)
    n_cols = len(views)

    logger.info(f"Creating surface map with title: {title}")
    logger.debug(f"Number of rows: {n_rows}, columns: {n_cols}")

    fig = make_subplots(
        rows=n_rows, cols=n_cols,
        specs=[[{"type": "scene"} for _ in range(n_cols)] for _ in range(n_rows)],
        horizontal_spacing=0.005,
        vertical_spacing=0.01,
        subplot_titles=["" for _ in range(n_rows * n_cols)]
    )

    for row, surf_type in enumerate(surface_types, start=1):
        for col, (view, hemi) in enumerate(views, start=1):
            logger.debug(f"Processing {surf_type} surface, {hemi} hemisphere, {view} view")

            mesh = mesh_dict[surf_type][hemi]
            texture = surface.vol_to_surf(img, mesh_dict['pial'][hemi])

            # Only show colorbar for last column
            show_cb = (col == n_cols)

            sub_fig_wrapper = plotting.plot_surf_stat_map(
                surf_mesh=mesh,
                stat_map=texture,
                hemi=hemi,
                view=view,
                bg_map=sulc_dict[hemi],
                colorbar=show_cb,
                threshold=threshold,
                cmap=BRAIN_CMAP,
                engine="plotly",
                title=None,
            )

            sub_fig = sub_fig_wrapper.figure

            for trace in sub_fig.data:
                if show_cb and hasattr(trace, 'colorbar'):
                    trace.colorbar.thickness = 20
                    trace.colorbar.len = 0.8
                    trace.colorbar.tickfont = dict(size=base_font_size, family="Ubuntu Condensed")
                    trace.colorbar.title = dict(text="z", font=dict(size=base_font_size, family="Ubuntu Condensed"), side="right")
                    trace.colorbar.tickvals = [np.nanmin(texture), 0, np.nanmax(texture)]
                    trace.colorbar.ticktext = [f"{np.nanmin(texture):.2f}", "0", f"{np.nanmax(texture):.2f}"]
                fig.add_trace(trace, row=row, col=col)

            fig.update_scenes(
                dict(
                    xaxis=dict(visible=False),
                    yaxis=dict(visible=False),
                    zaxis=dict(visible=False),
                    camera=dict(eye=view_angles[view]),
                    aspectmode='data'
                ),
                row=row, col=col
            )

    fig.update_layout(
        title=dict(
            text=title,
            font=dict(size=base_font_size * 2.5, family='Ubuntu Condensed'),
            y=0.92
        ),
        height=450 * n_rows,
        width=450 * n_cols,
        showlegend=False,
        margin=dict(t=60, l=0, r=0, b=0)
    )

    # --- Handle output directory and filenames ---
    outdir = os.path.dirname(output_file) if output_file else '.'
    os.makedirs(outdir, exist_ok=True)

    # Ensure the PNG file is named correctly
    safe_title = title.replace(' ', '_').replace('|', '').replace(':', '')
    png_out = output_file or os.path.join(outdir, f"{safe_title}_surface.png")

    # Save and open figure
    save_and_open_plotly_figure(
        fig,
        title=title,
        outdir=outdir,
        png_out=png_out
    )

def plot_surface_map_flat(img, title='Flat Surface Map', threshold=None, output_file=None):
    """
    Project a volumetric image onto the flat cortical surface and create a 3D Plotly figure.

    Parameters
    ----------
    img : nibabel.Nifti1Image
        3D statistical brain image to project and visualize.

    title : str
        Title of the final figure.

    threshold : float or None
        Value to threshold the statistical map (values below are masked out).

    output_file : str or None
        Optional full path to save the resulting PNG image.

    Returns
    -------
    None
    """
    from nilearn import datasets, surface, plotting
    import numpy as np
    import os
    from plotly.subplots import make_subplots

    fsaverage = datasets.fetch_surf_fsaverage('fsaverage6')
    hemis = ['left', 'right']

    mesh_dict = {
        'pial': {'left': fsaverage.pial_left, 'right': fsaverage.pial_right},
        'flat': {'left': fsaverage.flat_left, 'right': fsaverage.flat_right}
    }
    sulc_maps = {'left': fsaverage.sulc_left, 'right': fsaverage.sulc_right}

    logger.info(f"Creating flat surface map with title: {title}")

    fig = make_subplots(
        rows=1, cols=2,
        specs=[[{"type": "scene"}, {"type": "scene"}]],
        horizontal_spacing=0.01,
        subplot_titles=["Left Hemisphere", "Right Hemisphere"]
    )

    camera = dict(eye=dict(x=0, y=-1, z=2))  # anterior up, medial inward

    for i, hemi in enumerate(hemis):
        mesh = mesh_dict['flat'][hemi]
        texture = surface.vol_to_surf(img, mesh_dict['pial'][hemi])

        sub_fig = plotting.plot_surf_stat_map(
            surf_mesh=mesh,
            stat_map=texture,
            hemi=hemi,
            bg_map=sulc_maps[hemi],
            colorbar=False,
            threshold=threshold,
            cmap=BRAIN_CMAP,
            engine="plotly",
            title=None,
        ).figure

        for trace in sub_fig.data:
            if hasattr(trace, 'colorbar'):
                # Apply colorbar styling only once
                trace.colorbar = dict(
                    thickness=30,
                    len=0.9,
                    tickfont=dict(size=20, family="Ubuntu Condensed"),
                    title=dict(
                        text="z",
                        font=dict(size=28, family="Ubuntu Condensed"),
                        side="right"
                    ),
                    tickvals=[np.nanmin(texture), 0, np.nanmax(texture)],
                    ticktext=[f"{np.nanmin(texture):.2f}", "0", f"{np.nanmax(texture):.2f}"]
                )
            fig.add_trace(trace, row=1, col=i + 1)

        fig.update_scenes(
            dict(
                xaxis=dict(visible=False),
                yaxis=dict(visible=False),
                zaxis=dict(visible=False),
                camera=camera,
                aspectmode='data'
            ),
            row=1, col=i + 1
        )

    fig.update_layout(
        title=dict(
            text=title,
            font=dict(size=28, family='Ubuntu Condensed'),
            y=0.88,
            x=0.5,
            xanchor='center'
        ),
        height=500,
        width=850,
        showlegend=False,
        margin=dict(t=30, l=0, r=0, b=0)
    )

    # Adjust subplot (hemisphere) titles
    for annotation in fig['layout']['annotations']:
        annotation['font'] = dict(size=24, family='Ubuntu Condensed')
        annotation['y'] -= 0.15
        annotation['yanchor'] = 'top'

    # Output handling
    outdir = os.path.dirname(output_file) if output_file else '.'
    os.makedirs(outdir, exist_ok=True)
    safe_title = title.replace(' ', '_').replace('|', '').replace(':', '')
    png_out = output_file or os.path.join(outdir, f"{safe_title}_flat_surface.png")

    save_and_open_plotly_figure(
        fig,
        title=title,
        outdir=outdir,
        png_out=png_out
    )


def find_nifti_files(data_dir, pattern=None):
    """
    Recursively find all .nii.gz files in `data_dir` containing `pattern` in filename.

    Parameters
    ----------
    data_dir : str
        Root directory to search for NIfTI files.

    pattern : str or None
        Optional substring to match in file names. If None, all .nii.gz files are returned.

    Returns
    -------
    list of str
        Sorted list of full file paths matching the criteria.
    """
    files = []  # initialize list
    logger.info(f"Searching for .nii.gz files in: {data_dir}")
    if pattern:
        logger.info(f"Filtering files with pattern: '{pattern}'")

    # Walk directory tree
    for root, _, fnames in os.walk(data_dir):
        for f in fnames:
            # check extension and optional pattern
            if f.endswith('.nii.gz') and (pattern is None or pattern in f):
                full_path = os.path.join(root, f)
                logger.debug(f"Found file: {full_path}")
                files.append(full_path)

    logger.info(f"Total files found: {len(files)}")
    return sorted(files)



def split_by_group(files, expert_ids, novice_ids):
    """
    Split a list of file paths into expert and novice lists based on subject IDs.

    Returns
    -------
    exp_files, nov_files : lists of str
    """
    exp, nov = [], []  # initialize lists
    logger.info("Splitting %d files into expert and novice groups", len(files))

    # assign to expert group
    for sid in expert_ids:
        matched = [f for f in files if f"sub-{sid}" in os.path.basename(f)]
        exp += matched
        logger.debug("Expert sub-%s: %d files", sid, len(matched))

    # assign to novice group
    for sid in novice_ids:
        matched = [f for f in files if f"sub-{sid}" in os.path.basename(f)]
        nov += matched
        logger.debug("Novice sub-%s: %d files", sid, len(matched))

    logger.info("Assigned %d expert files, %d novice files", len(exp), len(nov))
    return exp, nov


def fisher_z_maps(imgs_list):
    """
    Transforms decoding accuracy or correlation maps into z-maps.

    Parameters
    ----------
    file_list : list of str
        List of NIfTI file paths.
    out_dir : str
        Directory to save transformed z-maps.
    n_trials : int
        Number of test observations per fold (default = 40 for LORO).

    Returns
    -------
    z_imgs : list of Nifti1Image
        Transformed z-maps.
    """
    z_imgs = []
    logger.info("Applying Fisher z-transform to %d images", len(imgs_list))

    for i, img in enumerate(imgs_list):
        z_img = math_img('np.arctanh(img)', img=img)
        z_imgs.append(z_img)
        logger.debug("Transformed image %d/%d", i + 1, len(imgs_list))

    logger.info("Fisher z-transformation complete")
    return z_imgs


def load_and_mask_imgs(file_list):
    """
    Load and Fisher-transform NIfTI images, masking out voxels outside the brain mask.

    Parameters
    ----------
    file_list : list of str
        Paths to subject NIfTI images.

    Returns
    -------
    masked_imgs : list of Nifti1Image
        Fisher z-transformed images with non-brain voxels set to NaN.
    brain_mask : Nifti1Image
        The binary brain mask used for masking.
    """
    logger.info("Loading and masking %d images", len(file_list))
    ref_img = load_img(file_list[0])
    brain_mask = get_brain_mask(ref_img)
    logger.debug("Loaded reference image for masking")

    masked_imgs = []
    for i, fname in enumerate(file_list):
        img = load_img(fname)
        masked_img = math_img("np.where(np.squeeze(mask), np.squeeze(img), np.nan)", img=img, mask=brain_mask)
        masked_imgs.append(masked_img)

    logger.info("All images masked and transformed")
    return masked_imgs, brain_mask


def build_design_matrix(n_exp, n_nov):
    """
    Create a pandas DataFrame with an intercept column and a group column
    coding experts as +1 and novices as -1.

    Returns
    -------
    design : pandas.DataFrame
    """
    logger.info("Building design matrix: %d experts, %d novices", n_exp, n_nov)
    intercept = np.ones(n_exp + n_nov)
    group = np.concatenate([np.ones(n_exp), -np.ones(n_nov)])
    design = pd.DataFrame({'intercept': intercept, 'group': group})
    logger.debug("Design matrix head:\n%s", design.head())
    return design

def bootstrap_corr(x, y, n_boot):
    """
    Bootstrap Pearson correlation using Pingouin.
    """
    logger.info("Bootstrapping correlation with %d iterations", n_boot)
    return pg.corr(x=x, y=y, method='pearson', bootstraps=n_boot,
                   confidence=0.95, method_ci='percentile', alternative='two-sided')


def extract_corr_results(result):
    """
    Extract r, CI low/high, and p-value from Pingouin result.
    """
    r = result['r'].iloc[0]
    p = result['p-val'].iloc[0]
    ci_lo, ci_hi = result['CI95%'].iloc[0]
    logger.debug("Extracted results: r=%.4f, CI=[%.4f, %.4f], p=%.4g", r, ci_lo, ci_hi, p)
    return r, ci_lo, ci_hi, p


def bootstrap_corr_diff(term_map, x, y, n_boot, rng, ci_alpha, n_jobs):
    """
    Compute a bootstrap-based confidence interval and p-value for the difference
    in Pearson correlations between two conditions (e.g., experts and novices)
    and a common reference map (e.g., a term-based meta-analytic map).

    Parameters
    ----------
    t : np.ndarray
        Reference map (e.g., Neurosynth term map), shape (n_voxels,)
    x : np.ndarray
        First condition (e.g., Experts z-map), shape (n_voxels,)
    y : np.ndarray
        Second condition (e.g., Novices z-map), shape (n_voxels,)
    n_boot : int
        Number of bootstrap samples.
    rng : np.random.Generator
        Numpy random number generator for reproducibility.
    ci_alpha : float
        Alpha level for confidence interval (e.g., 0.05 = 95% CI).
    n_jobs : int
        Number of parallel jobs. Use -1 to use all available cores.

    Returns
    -------
    mean_diff : float
        Mean bootstrap estimate of the difference in correlation (r_pos - r_neg).
    lo : float
        Lower bound of the (1 - ci_alpha) bootstrap confidence interval.
    hi : float
        Upper bound of the (1 - ci_alpha) bootstrap confidence interval.
    p_val : float
        Two-sided bootstrap p-value (null hypothesis: r_pos == r_neg).
    """
    logger.info("Bootstrapping correlation difference with %d iterations (n_jobs=%d)", n_boot, n_jobs)
    n = len(x)  # number of voxels

    def _boot(seed):
        """
        A single bootstrap iteration: resample voxels with replacement and compute
        the difference in correlation coefficients.
        """
        sub_rng = np.random.default_rng(seed)
        # Sample with replacement from voxel indices
        idx = sub_rng.integers(0, n, size=n)

        # Compute correlations on resampled data
        r_pos_b = np.corrcoef(term_map[idx], x[idx])[0, 1]
        r_neg_b = np.corrcoef(term_map[idx], y[idx])[0, 1]
        return r_pos_b - r_neg_b  # Difference in correlations

    # Generate independent seeds for reproducibility across bootstrap iterations
    seeds = rng.integers(0, 2**32 - 1, size=n_boot)

    # Run the bootstraps either sequentially or in parallel
    if n_jobs == 1:
        diffs = np.array([_boot(s) for s in seeds])
    else:
        diffs = np.array(Parallel(n_jobs=n_jobs)(delayed(_boot)(s) for s in tqdm(seeds, desc="Bootstrapping differences")))

    # Sort the bootstrap differences to compute percentile-based CI
    diffs.sort()

    # Compute lower and upper confidence interval bounds
    lo = np.percentile(diffs, 100 * ci_alpha / 2)
    hi = np.percentile(diffs, 100 * (1 - ci_alpha / 2))

    # Mean of bootstrap distribution
    mean_diff = np.mean(diffs)

    # Compute two-sided p-value: proportion of bootstrap diffs crossing zero
    tail_low = np.mean(diffs <= 0)
    tail_high = np.mean(diffs >= 0)
    p_val = 2 * min(tail_low, tail_high)

    logger.info("Bootstrap diff result: r_diff=%.4f, CI=[%.4f, %.4f], p=%.4g",
                mean_diff, lo, hi, p_val)
    logger.debug("Full bootstrap diff distribution stats: min=%.4f, max=%.4f, std=%.4f",
                 diffs.min(), diffs.max(), diffs.std())


    return mean_diff, lo, hi, p_val

def compute_all_zmap_correlations(z_pos, z_neg, term_maps, ref_img,
                                  n_boot=10000, fdr_alpha=0.05,
                                  ci_alpha=0.05, random_state=42,
                                  n_jobs=1):
    """
    Compute correlations between group RSA maps and Neurosynth term maps,
    including bootstrapped CIs, differences, and FDR correction.
    """
    logger.info("Starting correlation analysis with %d terms", len(term_maps))
    logger.info("Bootstrap iterations: %d, FDR alpha: %.3f, CI alpha: %.3f", n_boot, fdr_alpha, ci_alpha)

    records_pos = []
    records_neg = []
    records_diff = []
    rng = np.random.default_rng(random_state)

    flat_pos = z_pos.ravel()
    flat_neg = z_neg.ravel()
    logger.info("Flattened input z-maps.")

    # Binary brain mask
    logger.info("Extracting brain mask from reference image.")
    flat_mask = get_brain_mask(ref_img).get_fdata().ravel() > 0.25

    for i, (term, path) in enumerate(term_maps.items()):
        logger.info("Processing term %d/%d: '%s'", i + 1, len(term_maps), term)
        # Load and resample term map
        resampled_map = image.resample_to_img(image.load_img(path), ref_img,
                                              force_resample=True, copy_header=True)
        flat_term = resampled_map.get_fdata().ravel()

        # Stack maps
        stacked_data = np.vstack([flat_pos, flat_neg, flat_term])
        cleaned, kept_mask = remove_useless_data(stacked_data, flat_mask)
        x, y, this_term_map = cleaned
        logger.info("Cleaned data for '%s'. Kept %d voxels.", term, kept_mask.sum())

        # POSITIVE correlation
        res_pos = bootstrap_corr(this_term_map, x, n_boot)
        records_pos.append((term, *extract_corr_results(res_pos)))
        logger.debug("POS correlation for '%s': r=%.4f, p=%.4g", term, res_pos['r'].iloc[0], res_pos['p-val'].iloc[0])

        # NEGATIVE correlation
        res_neg = bootstrap_corr(this_term_map, y, n_boot)
        records_neg.append((term, *extract_corr_results(res_neg)))
        logger.debug("NEG correlation for '%s': r=%.4f, p=%.4g", term, res_neg['r'].iloc[0], res_neg['p-val'].iloc[0])

        # DIFFERENCE
        res_diff = bootstrap_corr_diff(this_term_map, x, y, n_boot, rng, ci_alpha, n_jobs)
        records_diff.append((term, res_pos['r'].iloc[0], res_neg['r'].iloc[0], *res_diff))
        logger.debug("DIFF correlation for '%s': r_diff=%.4f, p=%.4g", term, res_diff[0], res_diff[2])

    logger.info("Finished computing all raw correlations. Constructing DataFrames...")

    df_pos = pd.DataFrame(records_pos, columns=['term', 'r', 'CI_low', 'CI_high', 'p_raw'])
    df_neg = pd.DataFrame(records_neg, columns=['term', 'r', 'CI_low', 'CI_high', 'p_raw'])
    df_diff = pd.DataFrame(records_diff, columns=['term', 'r_pos', 'r_neg', 'r_diff', 'CI_low', 'CI_high', 'p_raw'])

    # FDR correction
    logger.info("Applying FDR correction (alpha=%.3f)", fdr_alpha)
    for name, df in zip(["POS", "NEG", "DIFF"], [df_pos, df_neg, df_diff]):
        rej, p_fdr = fdrcorrection(df['p_raw'], alpha=fdr_alpha)
        df['p_fdr'] = p_fdr
        df['sig'] = rej
        logger.info("%s correlations: %d significant terms after FDR", name, rej.sum())

    logger.info("All computations completed.")
    return df_pos, df_neg, df_diff

def correlate_participant_maps(participant_imgs, term_map, brain_mask, var_thresh=1e-6):
    """
    Compute correlation between each participant map and the term map.

    Parameters
    ----------
    participant_imgs : list of Nifti1Image
    term_map : Nifti1Image
    brain_mask : Nifti1Image
    var_thresh : float
        Minimum variance required to compute correlation.

    Returns
    -------
    correlations : np.ndarray of shape (n_subjects,)
    """
    # Flatten and mask term data
    flat_mask = brain_mask.get_fdata().ravel() > 0
    term_data_all = term_map.get_fdata().ravel()

    correlations = []
    for img in participant_imgs:
        subj_data_all = img.get_fdata().ravel()

        # Stack and clean: shape (2, n_voxels)
        stacked = np.vstack([subj_data_all, term_data_all])
        cleaned, kept = remove_useless_data(stacked, brain_mask_flat=flat_mask)

        if cleaned.shape[1] == 0:
            r = np.nan
        else:
            subj_data_clean, term_data_clean = cleaned
            if (
                np.std(subj_data_clean) < var_thresh or
                np.std(term_data_clean) < var_thresh or
                np.isnan(subj_data_clean).any() or
                np.isnan(term_data_clean).any()
            ):
                r = np.nan
            else:
                r = np.corrcoef(subj_data_clean, term_data_clean)[0, 1]
        correlations.append(r)

    return np.array(correlations)


def compute_subjectwise_correlations_and_tests(
    expert_imgs, novice_imgs, term_maps, brain_mask,
    fdr_alpha=0.05
):
    records_pos = []  # experts
    records_neg = []  # novices
    records_diff = []  # difference

    for term, path in term_maps.items():
        logger.info(f"Processing term: {term}")
        term_img = image.load_img(path)
        term_img = image.resample_to_img(term_img, brain_mask, interpolation='linear')

        # Correlations per subject
        r_exp = correlate_participant_maps(expert_imgs, term_img, brain_mask)
        r_nov = correlate_participant_maps(novice_imgs, term_img, brain_mask)

        # Expert one-sample t-test (H0: r = 0)
        test_exp = pg.ttest(r_exp, 0, alternative='two-sided')
        r_mean_exp, ci_exp = r_exp.mean(), test_exp['CI95%'].iloc[0]
        p_exp = test_exp['p-val'].iloc[0]
        records_pos.append((term, r_mean_exp, ci_exp[0], ci_exp[1], p_exp))

        # Novice one-sample t-test
        test_nov = pg.ttest(r_nov, 0, alternative='two-sided')
        r_mean_nov, ci_nov = r_nov.mean(), test_nov['CI95%'].iloc[0]
        p_nov = test_nov['p-val'].iloc[0]
        records_neg.append((term, r_mean_nov, ci_nov[0], ci_nov[1], p_nov))

        # Difference test (independent)
        test_diff = pg.ttest(r_exp, r_nov, alternative='two-sided', paired=False)
        r_diff = r_mean_exp - r_mean_nov
        ci_diff = test_diff['CI95%'].iloc[0]
        p_diff = test_diff['p-val'].iloc[0]
        records_diff.append((term, r_mean_exp, r_mean_nov, r_diff, ci_diff[0], ci_diff[1], p_diff))

    # Construct DataFrames
    df_pos = pd.DataFrame(records_pos, columns=['term', 'r', 'CI_low', 'CI_high', 'p_raw'])
    df_neg = pd.DataFrame(records_neg, columns=['term', 'r', 'CI_low', 'CI_high', 'p_raw'])
    df_diff = pd.DataFrame(records_diff, columns=['term', 'r_pos', 'r_neg', 'r_diff', 'CI_low', 'CI_high', 'p_raw'])

    # FDR correction
    for name, df in zip(["POS", "NEG", "DIFF"], [df_pos, df_neg, df_diff]):
        rej, p_fdr = fdrcorrection(df['p_raw'], alpha=fdr_alpha)
        df['p_fdr'] = p_fdr
        df['sig'] = rej
        logger.info("%s: %d significant terms after FDR", name, rej.sum())

    return df_pos, df_neg, df_diff


# --- Configuration ---
DATA_DIR         = '/data/projects/chess/data/BIDS/derivatives/rsa_searchlight'
RESULTS_ROOT     = 'results'
SMOOTHING_MM     = 6            # Group-level smoothing (FWHM)
MIN_VOXEL_VALUE  = 1e-5         # Threshold for plotting
TERM_DIR         = 'data/terms'  # Meta-analytic term maps directory
BRAIN_CMAP       = make_brain_cmap()

# Subject group definitions
EXPERT_SUBJECTS = ["03","04","06","07","08","09","10","11","12","13",
                   "16","20","22","23","24","29","30","33","34","36"]
NOVICE_SUBJECTS = ["01","02","15","17","18","19","21","25","26","27",
                   "28","32","35","37","39","40","41","42","43","44"]

# Define analysis patterns
PATTERNS = [
    'searchlight_check',
    'searchlight_strategy',
    'searchlight_visualSimilarity',
]
pattern_clean = {
    'searchlight_check': "Checkmate | RSA searchlight",
    'searchlight_strategy': "Strategy | RSA searchlight",
    'searchlight_visualSimilarity': "Visual Similarity | RSA searchlight",
}

# --- Output Setup ---
RESULTS_DIR = os.path.join(RESULTS_ROOT, f"{create_run_id()}_neurosynth-rsa-searchlight")
os.makedirs(RESULTS_DIR, exist_ok=True)

create_output_directory(RESULTS_DIR)
save_script_to_file(RESULTS_DIR)
out_text_file = os.path.join(RESULTS_DIR, 'console.log')
add_file_logger(out_text_file, level=logging.INFO)

# --- Begin Logging ---
logger.info("=== Neurosynth-RSA Analysis Started ===")
logger.info("Results directory: %s", RESULTS_DIR)

    # STEP 1: Load Neurosynth term maps
    logger.info("Loading meta-analytic term maps from: %s", TERM_DIR)
    term_maps = load_term_maps(TERM_DIR)

    all_pos = {}
    all_neg = {}
    all_diff = {}

    # i = 0
    # for term_key, term_path in term_maps.items():
    #     i += 1
    #     term_name = f"Term {i}: {os.path.basename(term_path).split('.')[0][2:].title()}"
    #     img = nib.load(term_path)

    #     # Save surface and glass brain plots
    #     plot_surface_map(
    #         img,
    #         threshold=MIN_VOXEL_VALUE,
    #         title=term_name,
    #         output_file=os.path.join(RESULTS_DIR, f'termmap_{term_name.replace(".nii.gz", "")}_surface.png')
    #     )
    #     plot_surface_map_flat(
    #         img,
    #         threshold=MIN_VOXEL_VALUE,
    #         title=term_name,
    #         output_file=os.path.join(RESULTS_DIR, f'termmap_{term_name.replace(".nii.gz", "")}_surface_flat.png')
    #     )

    #     plot_map(
    #         img.get_fdata(),
    #         ref_img=img,
    #         title=term_name,
    #         outpath=os.path.join(RESULTS_DIR, f'termmap_{term_name.replace(".nii.gz", "")}_glass.png'),
    #         thresh=MIN_VOXEL_VALUE
    #     )

    # STEP 2: Loop through each RSA pattern
    for PAT in PATTERNS:
        pattern_name = pattern_clean[PAT]
        logger.info(f"--- Processing Pattern: {pattern_name} ---")

        # STEP 2a: Locate and split subject maps
        r_files = find_nifti_files(DATA_DIR, pattern=PAT)
        exp_files, nov_files = split_by_group(r_files, EXPERT_SUBJECTS, NOVICE_SUBJECTS)

        # STEP 2b: Apply Fisher z-transform and mask
        z_exp, _ = load_and_mask_imgs(fisher_z_maps(exp_files))
        z_nov, _ = load_and_mask_imgs(fisher_z_maps(nov_files))
        z_all = z_exp + z_nov

        # # STEP 2c: Build design matrix and fit GLM
        # design = build_design_matrix(len(z_exp), len(z_nov))
        # slm = SecondLevelModel(smoothing_fwhm=SMOOTHING_MM, n_jobs=-1).fit(z_all, design_matrix=design)

        # # STEP 2d: Compute contrast (Experts > Novices)
        # con_img = slm.compute_contrast('group', output_type='z_score')
        # z_data = con_img.get_fdata()
        # con_img_path = os.path.join(RESULTS_DIR, f'{pattern_name}_zmap_experts_gt_novices.nii.gz')
        # con_img.to_filename(con_img_path)

        # # STEP 2e: Plot and save group-level map
        # plot_map(
        #     z_data, con_img,
        #     title=f'{pattern_name} | Experts>Novices',
        #     outpath=os.path.join(RESULTS_DIR, f'{pattern_name}_glassbrain_experts_gt_novices.png'),
        #     thresh=MIN_VOXEL_VALUE
        # )
        # plot_surface_map(
        #     con_img,
        #     title=f'{pattern_name} | Experts>Novices',
        #     threshold=MIN_VOXEL_VALUE,
        #     output_file=os.path.join(RESULTS_DIR, f'{pattern_name}_surface_experts_gt_novices.png')
        # )
        # plot_surface_map_flat(
        #     con_img,
        #     title=f'{pattern_name} | Experts>Novices',
        #     threshold=MIN_VOXEL_VALUE,
        #     output_file=os.path.join(RESULTS_DIR, f'{pattern_name}_surface_flat_experts_gt_novices.png')
        # )

        # # STEP 2f: Split contrast map into positive/negative
        # z_pos = np.where(z_data > 0, z_data, 0)
        # z_neg = np.where(z_data < 0, -z_data, 0)

        # STEP 2g: Correlate z-maps with meta-analytic term maps
        df_pos, df_neg, df_diff = compute_subjectwise_correlations_and_tests(
            expert_imgs=z_exp,  # list of masked NIfTI images
            novice_imgs=z_nov,
            term_maps=term_maps,
            brain_mask=get_brain_mask(z_exp[0]),  # same mask used in masking step
            fdr_alpha=0.05
        )

        # STEP 2h: Save correlation data
        key = pattern_clean[PAT].split()[0].lower()  # e.g., 'checkmate'

        all_pos[key] = df_pos
        all_neg[key] = df_neg
        all_diff[key] = df_diff.rename(columns={'r': 'r_diff'})

        df_pos.to_csv(os.path.join(RESULTS_DIR, f'{pattern_name}_term_corr_positive.csv'), index=False)
        df_neg.to_csv(os.path.join(RESULTS_DIR, f'{pattern_name}_term_corr_negative.csv'), index=False)
        df_diff.to_csv(os.path.join(RESULTS_DIR, f'{pattern_name}_term_corr_difference.csv'), index=False)
        save_latex_correlation_tables(df_pos, df_neg, df_diff, out_dir=RESULTS_DIR, run_id=pattern_name)

        # STEP 2i: Plot correlation strength and difference
        plot_correlations(
            df_pos, df_neg, df_diff,
            run_id=pattern_name,
            out_fig=os.path.join(RESULTS_DIR, f'{pattern_name}_term_correlations.png')
        )
        plot_difference(
            df_diff,
            run_id=pattern_name,
            out_fig=os.path.join(RESULTS_DIR, f'{pattern_name}_term_correlation_differences.png')
        )


    # Assume all_diff, all_pos, all_neg are dicts:
    # e.g., {'Checkmate': df1, 'Strategy': df2, 'Visual Similarity': df3}

    generate_latex_multicolumn_table(
        data_dict=all_diff,
        output_path=os.path.join(RESULTS_DIR, 'rsa_searchlight_diff.tex'),
        table_type='diff',
        caption='RSA searchlight results for expert–novice difference in correlation with term maps.',
        label='tab:rsa_searchlight_diff'
    )

    generate_latex_multicolumn_table(
        data_dict=all_pos,
        output_path=os.path.join(RESULTS_DIR, 'rsa_searchlight_pos.tex'),
        table_type='pos',
        caption='RSA searchlight results for positive z-maps (experts only).',
        label='tab:rsa_searchlight_expert_pos'
    )

    generate_latex_multicolumn_table(
        data_dict=all_neg,
        output_path=os.path.join(RESULTS_DIR, 'rsa_searchlight_neg.tex'),
        table_type='neg',
        caption='RSA searchlight results for negative z-maps (novices > experts).',
        label='tab:rsa_searchlight_expert_neg'
    )

    # Final message
    logger.info("=== Analysis Complete. Results saved to: %s ===", RESULTS_DIR)
