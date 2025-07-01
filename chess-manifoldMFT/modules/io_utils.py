#!/usr/bin/env python3
"""Input/output helpers for the manifold analysis module."""


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
    spm_dir = os.path.join(BASE_GLM_PATH, f"sub-{subject_id}", "exp")
    spm_mat = os.path.join(spm_dir, "SPM.mat")
    if not os.path.isfile(spm_mat):
        raise FileNotFoundError(f"SPM.mat not found for sub-{subject_id}: {spm_mat}")

    spm = sio.loadmat(spm_mat, struct_as_record=False, squeeze_me=True)["SPM"]
    beta_info = spm.Vbeta
    regressor_names = spm.xX.name

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


def load_all_betas(subject_id: str, avg_runs: bool) -> Tuple[Dict[str, np.ndarray], List[str]]:
    """Load all beta images for a subject without averaging across runs.

    Parameters
    ----------
    subject_id : str
        Subject identifier.

    Returns
    -------
    betas : dict
        Mapping from condition name to an array of shape ``(n_runs, X, Y, Z)``.
    labels : list of str
        Sorted list of condition names.
    """

    info = get_spm_betas(subject_id)

    # Sort conditions alphabetically for reproducibility
    conditions = sorted(info.keys())
    betas: Dict[str, np.ndarray] = {}
    for cond in conditions:
        entries = sorted(info[cond], key=lambda x: x["run"])
        paths = [entry["beta_path"] for entry in entries]
        imgs = [nib.load(p) for p in paths]
        data = [img.get_fdata(dtype=np.float32) for img in imgs]
        if avg_runs:
            betas[cond] = np.nanmean(np.stack(data, axis=0), axis=0)
            total_voxels = np.prod(betas[cond].shape)
            nans = np.isnan(betas[cond]).sum()
        else:
            betas[cond] = data
            total_voxels = np.prod(betas[cond][0].shape)
            nans = np.isnan(betas[cond][0]).sum()

        # Print th number of total voxels and the number of nans

        logger.info(
            f"Condition '{cond}' has {total_voxels} total voxels, "
            f"{nans} of which are NaN."
        )

    return betas, conditions


def load_atlas() -> Tuple[np.ndarray, np.ndarray]:
    """Load atlas image and return data and list of unique ROI labels."""
    atlas_img = nib.load(ATLAS_FILE)
    data = atlas_img.get_fdata().astype(int)
    labels = np.unique(data)
    labels = labels[labels != 0]
    return data, labels
