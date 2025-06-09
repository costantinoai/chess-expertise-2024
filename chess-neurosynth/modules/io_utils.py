#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Utility functions for loading Neuroimaging data used in the pipeline."""

import os
from nilearn import image
from modules.config import logger

def load_term_maps(map_dir):
    """Return a dictionary mapping each term name to its NIfTI file.

    The filenames are converted to lowercase terms with underscores replaced by
    spaces so that ``navigation.nii.gz`` becomes ``"navigation"``.
    """
    logger.info(f"Loading term maps from: {map_dir}")
    maps = {}
    for fname in sorted(os.listdir(map_dir)):
        if fname.endswith(('.nii', '.nii.gz')):
            term = os.path.splitext(fname)[0].replace('_', ' ').lower()
            maps[term] = os.path.join(map_dir, fname)
    return maps

def load_nifti(path):
    """Load an image file using nilearn and return the NIfTI image object."""
    return image.load_img(path)
