#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ROI utilities and stats for GLM summaries.

Contains region label helpers, CI/ttest wrappers, CoM labeling, and LaTeX export.
"""
from __future__ import annotations

import os
import logging
from glob import glob
from typing import Sequence

import numpy as np
from nilearn.image import coord_transform
from scipy.ndimage import center_of_mass
from scipy.stats import ttest_ind


from rois.meta import get_region_label as _get_region_label


def get_region_label(region_number: int) -> str:
    """Return human-readable label for a Glasser region (bilateral set)."""
    return _get_region_label(int(region_number), set_name="glasser_regions_bilateral")


def stars(p: float) -> str:
    if p < 0.001:
        return "***"
    elif p < 0.01:
        return "**"
    elif p < 0.05:
        return "*"
    else:
        return ""


def compute_confidence_intervals(X1: np.ndarray, X2: np.ndarray):
    """Compute mean difference, t-values, p-values, and 95% CI for each ROI."""
    means = X1.mean(0) - X2.mean(0)
    t_results = ttest_ind(X1, X2, axis=0)
    cis = t_results.confidence_interval()
    return means, t_results.statistic, t_results.pvalue, cis


def get_label_at_com(
    roi_val: int,
    glasser_data: np.ndarray,
    affine_glasser: np.ndarray,
    atlas_data_ho: np.ndarray,
    affine_ho: np.ndarray,
    atlas_labels_ho: Sequence[str],
) -> str:
    """Label ROI by Harvard-Oxford region at its center-of-mass (left hemisphere)."""
    try:
        roi_mask = glasser_data == roi_val
        if not np.any(roi_mask):
            return "No ROI voxels"

        coords = np.argwhere(roi_mask)
        mni_coords = np.array([coord_transform(x, y, z, affine_glasser) for x, y, z in coords])
        left_coords = coords[mni_coords[:, 0] < 0]
        if len(left_coords) == 0:
            return "No LH voxels"

        lh_mask = np.zeros_like(glasser_data, dtype=bool)
        lh_mask[tuple(left_coords.T)] = True
        com_voxel = center_of_mass(lh_mask)
        if np.any(np.isnan(com_voxel)):
            return "Invalid CoM"

        com_voxel = tuple(int(round(c)) for c in com_voxel)
        mni_com = coord_transform(*com_voxel, affine_glasser)
        ijk_ho = np.round(np.linalg.inv(affine_ho) @ np.array([*mni_com, 1])).astype(int)
        i, j, k, _ = ijk_ho
        if (
            i < 0 or j < 0 or k < 0 or
            i >= atlas_data_ho.shape[0] or j >= atlas_data_ho.shape[1] or k >= atlas_data_ho.shape[2]
        ):
            return "Out of bounds"
        ho_val = int(atlas_data_ho[i, j, k])
        if ho_val == 0:
            return "Unknown (0)"
        return atlas_labels_ho[ho_val]
    except Exception as e:
        logging.getLogger(__name__).warning("Failed CoM label for ROI %s: %s", roi_val, e)
        return "Error"


def paths_for(subject_ids: Sequence[str], mode: str, data_dir: str, fname: str) -> list[str]:
    """Build list of file paths for each subject and mode ('rsa' or 'univ')."""
    paths: list[str] = []
    for subject_id in subject_ids:
        subject_folder = os.path.join(data_dir, f"sub-{subject_id}")
        if mode == "rsa":
            filename = f"sub-{subject_id}_searchlight_{fname}"
        else:
            filename = f"exp/{fname}"
        full_pattern = os.path.join(subject_folder, filename)
        matched_files = glob(full_pattern)
        if not matched_files:
            raise FileNotFoundError(f"No file found for pattern: {full_pattern}")
        paths.append(matched_files[0])
    return paths


## Moved LaTeX export to glm/modules/glm_utils.py to keep one source of truth.
