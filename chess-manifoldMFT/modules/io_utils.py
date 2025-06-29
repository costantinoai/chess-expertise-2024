#!/usr/bin/env python3
"""Input/output helpers for the manifold analysis module."""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Dict, List, Tuple

import nibabel as nib
import numpy as np
import scipy.io as sio

from . import logger, BASE_GLM_PATH, ATLAS_FILE


def get_spm_betas(subject_id: str) -> Dict[str, List[Dict[str, str]]]:
    """Return mapping of condition name to a list of beta file info."""
    spm_mat = os.path.join(BASE_GLM_PATH, f"sub-{subject_id}", "exp", "SPM.mat")
    if not os.path.isfile(spm_mat):
        raise FileNotFoundError(f"SPM.mat not found for sub-{subject_id}: {spm_mat}")

    spm = sio.loadmat(spm_mat, struct_as_record=False, squeeze_me=True)["SPM"]
    beta_info = spm.Vbeta
    regressor_names = spm.xX.name
    spm_dir = getattr(spm, "swd", os.path.dirname(spm_mat))

    cond_info: Dict[str, List[Dict[str, str]]] = {}
    pattern = r"Sn\((\d+)\)\s+(.*?)\*bf\(1\)"
    for idx, full_name in enumerate(regressor_names):
        m = re.match(pattern, full_name)
        if not m:
            continue
        run = int(m.group(1))
        cond = m.group(2)
        beta_fname = (
            beta_info[idx].fname
            if hasattr(beta_info[idx], "fname")
            else beta_info[idx].filename
        )
        beta_path = os.path.join(spm_dir, beta_fname)
        cond_info.setdefault(cond, []).append(
            {
                "run": run,
                "regressor_name": full_name,
                "beta_path": beta_path,
            }
        )
    return cond_info


def load_all_betas(subject_id: str) -> Tuple[np.ndarray, List[str]]:
    """Load all beta images for a subject into a 4D array."""
    info = get_spm_betas(subject_id)
    beta_paths = [entry["beta_path"] for entries in info.values() for entry in entries]
    beta_imgs = [nib.load(p) for p in beta_paths]
    beta_data = [img.get_fdata(dtype=np.float32) for img in beta_imgs]
    stacked = np.stack(beta_data, axis=0)
    return stacked, beta_paths


def load_atlas() -> Tuple[np.ndarray, np.ndarray]:
    """Load atlas image and return data and list of unique ROI labels."""
    atlas_img = nib.load(ATLAS_FILE)
    data = atlas_img.get_fdata().astype(int)
    labels = np.unique(data)
    labels = labels[labels != 0]
    return data, labels
