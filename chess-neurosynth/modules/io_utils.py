#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun  7 22:18:21 2025

@author: costantino_ai
"""

import os
from nilearn import image
from modules.config import logger

def load_term_maps(map_dir):
    """
    Scan a directory for .nii/.nii.gz files, return dict term→filepath.
    """
    logger.info(f"Loading term maps from: {map_dir}")
    maps = {}
    for fname in sorted(os.listdir(map_dir)):
        if fname.endswith(('.nii', '.nii.gz')):
            term = os.path.splitext(fname)[0].replace('_', ' ').lower()
            maps[term] = os.path.join(map_dir, fname)
    return maps

def load_nifti(path):
    """Shortcut for nilearn.load_img."""
    return image.load_img(path)
