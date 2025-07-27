#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 16 10:52:13 2025

@author: costantino_ai
"""

from nilearn.datasets import (
    load_fsaverage,load_fsaverage_data
)
from nilearn.surface import SurfaceImage
from nilearn.plotting import plot_surf_roi, show
import matplotlib.colors as mcolors

def load_freesurfer_lut(file_path):
    """Load a FreeSurfer-style LUT file and create a Matplotlib colormap."""
    lut_data = []
    label_to_color = {}

    with open(file_path, "r") as f:
        for line in f:
            if line.strip() and not line.startswith("#"):  # Ignore comments and empty lines
                parts = line.split()
                if len(parts) >= 6:  # Expected format: index, label, R, G, B, A
                    _ = int(parts[0])
                    label = parts[1]
                    r, g, b, a = map(int, parts[2:6])
                    color = (r / 255.0, g / 255.0, b / 255.0, a / 255.0)
                    lut_data.append(color)
                    label_to_color[label] = color

    cmap = mcolors.ListedColormap(lut_data)
    return cmap, label_to_color

fsaverage = load_fsaverage("fsaverage")

# fsaverage = load_fsaverage_data(
#     mesh="fsaverage",
#     mesh_type="flat",
#     data_type="sulcal"
#     )

# fsaverage_mesh = fsaverage.mesh
# fsaverage_data = fsaverage.data.parts

data = {
    "left": "/data/projects/chess/data/BIDS/derivatives/fastsurfer/fsaverage/label/lh.HCPMMP1.annot",
    "right": "/data/projects/chess/data/BIDS/derivatives/fastsurfer/fsaverage/label/rh.HCPMMP1.annot",
}

import pandas as pd
import matplotlib.colors as mcolors

# Load region info TSV
roi_df = pd.read_csv(
    "/home/eik-tb/OneDrive_andreaivan.costantino@kuleuven.be/GitHub/chess-expertise-2024/chess-rois/results/glasser_regions_bilateral/region_info.tsv",
    sep='\t'
).sort_values("region_id")



# Convert HEX colors to RGBA tuples
rgba_colors = [mcolors.to_rgba(c) for c in roi_df["color"]]

# Create ListedColormap using ROI-defined colors
glasser_cmap_custom = mcolors.ListedColormap(rgba_colors)

# Use the same for left and right, or make hemisphere-specific ones if desired
cmaps = {
    "left": glasser_cmap_custom,
    "right": glasser_cmap_custom,
}

glasser_atlas = SurfaceImage(
    mesh=fsaverage["pial"],
    data=data,
)
from nilearn.plotting import plot_surf_contours

# for view in ["lateral", "posterior", "ventral"]:
fig = plot_surf_roi(
    # surf_mesh=glasser_atlas,
    roi_map=glasser_atlas,
    hemi="left",
    view="lateral",
    # bg_map=fsaverage["sulcal"],
    bg_on_data=True,
    darkness=0.5,
    cmap=cmaps["left"],
    engine="plotly"
    # title=f"Destrieux parcellation on inflated surface\n{view} view",
)
fig.show()

# for view in ["lateral", "posterior", "ventral"]:
fig = plot_surf_roi(
    # surf_mesh=glasser_atlas,
    roi_map=glasser_atlas,
    hemi="right",
    view="lateral",
    # bg_map=fsaverage["sulcal"],
    bg_on_data=True,
    darkness=0.5,
    cmap=cmaps["right"],
    engine="plotly"
    # title=f"Destrieux parcellation on inflated surface\n{view} view",
)
fig.show()
