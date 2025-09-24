#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul  5 23:17:01 2025
@author: costantino_ai
"""

# --- Dependencies ---
from nilearn import plotting, image, masking
from nilearn.datasets import fetch_atlas_harvard_oxford
from templateflow.api import get
import numpy as np
import matplotlib.pyplot as plt
from nilearn.image import resample_to_img

# --- Load gray matter probability map from TemplateFlow ---
# ICBM 2009c nonlinear asymmetrical gray matter template
gm_prob_img = get(
    "MNI152NLin2009cAsym",
    label="GM",
    suffix="probseg",
    resolution=2
)

print("GM probseg file loaded:", gm_prob_img)

# --- Threshold the gray matter map at > 0.5 ---
gm_mask_img = image.math_img("img > 0.5", img=gm_prob_img)

# --- Load subcortical atlas (Harvard-Oxford) ---
subcortical_atlas = fetch_atlas_harvard_oxford('sub-maxprob-thr50-2mm')
subcortical_img = subcortical_atlas['maps']
labels = subcortical_atlas['labels']

# --- Resample subcortical atlas to match GM image space ---
subcortical_img_resampled = resample_to_img(subcortical_img, gm_mask_img, interpolation='nearest')

# --- Compute overlap (numerical check) ---
overlap_results = {}
for i, label in enumerate(labels[1:], start=1):  # skip background
    region_mask = image.math_img("img == %d" % i, img=subcortical_img_resampled)
    intersection = image.math_img("a * b", a=gm_mask_img, b=region_mask)  # use multiplication for binary AND

    # Check if intersection is non-empty
    data = intersection.get_fdata()
    overlap_voxels = int(np.sum(data > 0))

    overlap_results[label] = overlap_voxels

# --- Print table of overlap ---
print("\nSubcortical Regions Overlapping with GM Mask (voxels > 0):")
for region, count in overlap_results.items():
    print(f"  {region}: {count} voxels")

# --- Visualize gray matter mask ---
plotting.plot_glass_brain(
    gm_mask_img, colorbar=True, display_mode='lyrz', title="Gray Matter Mask > 0.5"
)

# --- Visualize subcortical atlas ---
plotting.plot_roi(
    subcortical_img_resampled, title="Harvard-Oxford Subcortical Atlas", cmap='Paired'
)

plt.show()

# --- Compute and plot only intersection (GM ∩ Subcortical) ---
intersection_mask = image.math_img("a * (b > 0)", a=gm_mask_img, b=subcortical_img_resampled)

# Plot intersection
plotting.plot_roi(
    intersection_mask,
    title="Voxels in Both Gray Matter and Subcortical Atlas",
    cmap='spring',
    display_mode='ortho',
    colorbar=False
)
plt.show()
