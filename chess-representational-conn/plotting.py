# -*- coding: utf-8 -*-
"""Plotting helpers for representational connectivity matrices."""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import OrderedDict
from modules import MANAGER
from nilearn import plotting
import matplotlib.patches as mpatches

cortex_info_map = OrderedDict({
    # Group 1: Early Visual
    1: {"color": "#a6cee3", "group": "Early Visual"},      # Primary_Visual
    2: {"color": "#a6cee3", "group": "Early Visual"},      # Early_Visual

    # Group 2: Intermediate Visual
    3: {"color": "#1f78b4", "group": "Intermediate Visual"},
    4: {"color": "#1f78b4", "group": "Intermediate Visual"},
    5: {"color": "#1f78b4", "group": "Intermediate Visual"},

    # Group 3: Sensorimotor
    6: {"color": "#b2df8a", "group": "Sensorimotor"},
    7: {"color": "#b2df8a", "group": "Sensorimotor"},
    8: {"color": "#b2df8a", "group": "Sensorimotor"},
    9: {"color": "#b2df8a", "group": "Sensorimotor"},

    # Group 4: Auditory
    10: {"color": "#33a02c", "group": "Auditory"},
    11: {"color": "#33a02c", "group": "Auditory"},
    12: {"color": "#33a02c", "group": "Auditory"},

    # Group 5: Temporal
    13: {"color": "#fb9a99", "group": "Temporal"},
    14: {"color": "#fb9a99", "group": "Temporal"},

    # Group 6: Posterior
    15: {"color": "#e31a1c", "group": "Posterior"},
    16: {"color": "#e31a1c", "group": "Posterior"},
    17: {"color": "#e31a1c", "group": "Posterior"},
    18: {"color": "#e31a1c", "group": "Posterior"},

    # Group 7: Anterior
    19: {"color": "#fdbf6f", "group": "Anterior"},
    20: {"color": "#fdbf6f", "group": "Anterior"},
    21: {"color": "#fdbf6f", "group": "Anterior"},
    22: {"color": "#fdbf6f", "group": "Anterior"},
})


def plot_connectivity_matrix(matrix: np.ndarray,
                              title: str,
                              out_path: str,
                              mask: np.ndarray | None = None,
                              vmin: float = -1.0,
                              vmax: float = 1.0,
                              cmap: str = "coolwarm"):
    """Plot a square connectivity matrix with ROI labels."""

    # Get ROI info (filtered to left hemisphere)
    rois_left = [r for r in MANAGER.rois if "L" in r.hemisphere]

    # Sort by cortex_id (as int)
    sorted_rois = sorted(rois_left, key=lambda r: int(r.cortex_id))

    # Get cortex names in that sorted order, and remove duplicates while preserving order
    seen = set()
    roi_names_ordered = []
    for r in sorted_rois:
        if r.cortex not in seen:
            roi_names_ordered.append(r.cortex)
            seen.add(r.cortex)

    if mask is not None:
        plot_mat = np.where(mask, matrix, np.nan)
    else:
        plot_mat = matrix

    plot_pretty_matrix(
        plot_mat,
        title,
        MANAGER,
        cortex_info_map,
        roi_names_ordered,
        reorder=False,
        figsize=(20, 18),
        vmax=None,
        vmin=None,
        legend_loc="upper left",
        legend_bbox=(1.2, 1),
        tri="full",
        cmap="RdBu_r",
        draw_cortex_boundaries=True,
        filename=None
    )

def compute_vmin_vmax(matrix, vmin=None, vmax=None):
    """
    Computes appropriate vmin and vmax values for visualizing a matrix,
    excluding the diagonal and upper triangle.

    This function determines the color scale limits (`vmin` and `vmax`) based on
    the provided matrix. The behavior is as follows:

    - If both `vmin` and `vmax` are None:
      - The function sets them symmetrically around zero, using the largest absolute
        value from the **lower triangle** of the matrix (excluding the diagonal).

    - If only `vmin` is None:
      - It is set to the minimum value from the lower triangle (excluding diagonal).

    - If only `vmax` is None:
      - It is set to the maximum value from the lower triangle (excluding diagonal).

    Parameters
    ----------
    matrix : np.ndarray
        A 2D square NumPy array (e.g., correlation matrices, similarity matrices).

    vmin : float, optional
        The minimum value for the color scale. If None, it is computed based on the lower triangle.

    vmax : float, optional
        The maximum value for the color scale. If None, it is computed based on the lower triangle.

    Returns
    -------
    vmin : float
        The computed or provided minimum value for the color scale.

    vmax : float
        The computed or provided maximum value for the color scale.

    Example
    -------
    >>> matrix = np.array([[1.0, 0.2, -0.3],
                           [0.5, 1.0, -0.1],
                           [-0.8, 0.3, 1.0]])
    >>> compute_vmin_vmax(matrix)
    (-0.8, 0.5)

    >>> compute_vmin_vmax(matrix, vmax=1.0)
    (-0.8, 1.0)

    >>> compute_vmin_vmax(matrix, vmin=-1.0, vmax=1.0)
    (-1.0, 1.0)
    """
    # Extract lower triangular part, excluding the diagonal
    lower_triangle_values = matrix[np.tril_indices_from(matrix, k=-1)]

    if lower_triangle_values.size == 0:
        raise ValueError(
            "Matrix does not have a lower triangle to compute vmin and vmax."
        )

    if vmin is None and vmax is None:
        abs_max = np.max(
            np.abs(lower_triangle_values)
        )  # Largest absolute value in lower triangle
        vmin, vmax = -abs_max, abs_max  # Symmetric around zero
    else:
        if vmin is None:
            vmin = np.min(lower_triangle_values)  # Min from lower triangle
        if vmax is None:
            vmax = np.max(lower_triangle_values)  # Max from lower triangle

    return vmin, vmax


def plot_pretty_matrix(
    subject_mean_corr,
    title,
    manager,
    cortex_info_map,
    labels,
    reorder=False,
    figsize=(20, 18),
    vmax=None,
    vmin=None,
    legend_loc="upper left",
    legend_bbox=(1.2, 1),
    tri="full",
    cmap="RdBu_r",
    draw_cortex_boundaries=True,
    filename=None
):
    """
    Plot a subject-level average correlation matrix (using Nilearn's plot_matrix) with:
      - Tick labels colored according to cortex color
      - A legend showing each group's color
      - (Optionally) draw black boundary lines after certain rows/columns if
        reorder=False and draw_cortex_boundaries=True.

    Parameters
    ----------
    subject_mean_corr : 2D array
        Subject-level average correlation matrix.
    subj_id : str or int
        Subject identifier, used in the plot title.
    manager : object
        Your custom manager that can filter cortex information by name.
        Must implement a method like: manager.get_by_filter(cortex=...) -> list of objects
        each having '.cortex_color'.
    cortex_info_map : dict or OrderedDict
        Maps region_id -> {'color': <hex_color>, 'group': <group_name>}.
    labels : list of str
        Labels to place on x and y ticks.
    reorder : bool, optional
        Whether to pass `reorder=True` to nilearn's plot_matrix. (Default=False)
    figsize : tuple, optional
        Figure size (width, height).
    vmax, vmin : float, optional
        Max and min values for color range in the matrix plot.
    legend_loc : str, optional
        Legend location code (e.g. 'upper left', 'best').
    legend_bbox : tuple, optional
        (x, y) anchor point for legend outside the main Axes.
    tri : str, optional
        'lower' (default) or 'full' for the triangular part to display.
    cmap : str, optional
        Colormap name (default='RdBu_r').
    draw_cortex_boundaries : bool, optional
        If True and reorder=False, draw black grid lines on the heatmap
        at certain boundaries to separate cortical groups.

    Returns
    -------
    fig : matplotlib Figure
        The figure containing the correlation matrix and legend.
    """

    # -- 1) Compute vmin, vmax if None --
    vmin, vmax = compute_vmin_vmax(subject_mean_corr, vmin, vmax)

    # -- 2) Define the fixed order (same as your snippet) --
    new_order = [
        0,  1,  2,  3,  4,
        5,  6,  7,  8,
        9,  10, 11,
        12, 13,
        14, 15, 16, 17,
        18, 19, 20, 21,
    ]

    # -- 3) Reorder the correlation matrix --
    subject_mean_corr_reordered = subject_mean_corr[np.ix_(new_order, new_order)]
    labels_reordered = [labels[i] for i in new_order]

    # -- 4) Plot the matrix --
    fig = plotting.plot_matrix(
        subject_mean_corr_reordered,
        labels=labels_reordered,
        figure=figsize,
        vmax=vmax,
        vmin=vmin,
        title=f"{title} - Connectivity Matrix",
        reorder=reorder,  # This tells nilearn whether to do its own reorder internally.
        tri=tri,
        cmap=cmap,
    )

    # plot_matrix returns a matplotlib AxesImage object, but we can get the Axes via:
    ax_matrix = fig.axes  # In nilearn <=0.9, this is typically a single Axes

    # -- 5) Color tick labels by cortex color --
    for label_obj in ax_matrix.get_xticklabels():
        cortex_name = label_obj.get_text()
        try:
            # Retrieve color for this cortex from the manager
            this_color = manager.get_by_filter(cortex=cortex_name)[0][0].cortex_color
        except IndexError:
            this_color = "black"
        label_obj.set_color(this_color)

    for label_obj in ax_matrix.get_yticklabels():
        cortex_name = label_obj.get_text()
        try:
            this_color = manager.get_by_filter(cortex=cortex_name)[0][0].cortex_color
        except IndexError:
            this_color = "black"
        label_obj.set_color(this_color)

    # -- 6) Draw cortex boundaries if requested --
    if not reorder and draw_cortex_boundaries:
        # Boundaries to draw after these region-IDs in 'new_order'
        boundary_regions = [1, 4, 17, 13, 11, 8]

        # Because 'new_order' is the final arrangement, find each region's index in new_order
        # and draw a line at (index + 0.5) for both x and y.
        for region_id in boundary_regions:
            try:
                i = new_order.index(region_id)
                # Draw horizontal and vertical lines between i and i+1
                ax_matrix.axhline(i + 0.5, color="black", linewidth=1)
                ax_matrix.axvline(i + 0.5, color="black", linewidth=1)
            except ValueError:
                # region_id wasn't found in new_order, ignore
                pass

    # -- 7) Build and add a legend: one entry per group --
    group_to_color = {}
    for region_id, info in cortex_info_map.items():
        group = info["group"]
        color = info["color"]
        # only keep the first color we see for each group
        if group not in group_to_color:
            group_to_color[group] = color

    legend_handles = [
        mpatches.Patch(color=color, label=group)
        for group, color in group_to_color.items()
    ]
    ax_matrix.legend(
        handles=legend_handles,
        bbox_to_anchor=legend_bbox,
        loc=legend_loc,
        borderaxespad=0.0,
    )

    if filename is not None:
        plt.savefig(filename, dpi=300, bbox_inches="tight")

    plt.show()
    return fig
