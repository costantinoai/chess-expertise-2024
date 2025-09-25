#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Shared NIfTI/Nilearn helpers for IO and masking.

Centralizes small, reusable utilities for finding images, Fisher-Z transform,
applying masks, and building simple design matrices.
"""
from __future__ import annotations

import os
from typing import Callable, Iterable, List, Tuple

import numpy as np
import pandas as pd
from nilearn.image import load_img, math_img


def find_nifti_files(data_dir: str, pattern: str | None = None) -> List[str]:
    files: list[str] = []
    for root, _, fnames in os.walk(data_dir):
        for f in fnames:
            if f.endswith((".nii", ".nii.gz")) and (pattern is None or pattern in f):
                files.append(os.path.join(root, f))
    return sorted(files)


def fisher_z_maps(file_list: Iterable[str]) -> list:
    return [math_img('np.arctanh(img)', img=load_img(f)) for f in file_list]


def build_design_matrix(n_exp: int, n_nov: int) -> pd.DataFrame:
    intercept = np.ones(n_exp + n_nov)
    group = np.concatenate([np.ones(n_exp), -np.ones(n_nov)])
    return pd.DataFrame({"intercept": intercept, "group": group})


def load_and_mask_imgs(
    imgs_list: list,
    mask_img=None,
    mask_func: Callable[[object], object] | None = None,
) -> Tuple[list, object | None]:
    """Apply a brain mask to a list of NIfTI images.

    If `mask_img` is provided, uses it directly. If `mask_func` is provided,
    it will be called with the first image to obtain a mask. If neither is
    provided, returns the loaded images without masking and mask=None.
    """
    if not imgs_list:
        return [], None
    ref_img = imgs_list[0]
    if not hasattr(ref_img, "get_fdata"):
        ref_img = load_img(ref_img)
    if mask_img is None and mask_func is not None:
        mask_img = mask_func(ref_img)
    if mask_img is None:
        # No masking requested; load and return
        loaded = [load_img(i) if not hasattr(i, "get_fdata") else i for i in imgs_list]
        return loaded, None
    masked = [
        math_img("np.where(np.squeeze(mask), np.squeeze(img), np.nan)", img=load_img(i), mask=mask_img)
        for i in imgs_list
    ]
    return masked, mask_img

