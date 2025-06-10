#!/usr/bin/env python3
"""Group-level TFCE analysis using PyMVPA.

This script recursively searches ``root_dir`` for NIfTI files containing
``filter_str`` in their filename. Subject identifiers are extracted from the
path (``sub-XX``), allowing separation into experts and novices based on the
lists defined in ``modules/__init__.py``. For each group a one-sample TFCE test
against chance is computed, and a two-sample test compares experts against
novices. Unthresholded and thresholded z-maps are saved under
``results/mvpa-second-level`` while preserving the original directory
structure.

Example
-------
>>> root_dir = '/path/to/searchlight/results'
>>> filter_str = 'checkmate'
>>> run_group_tfce(root_dir, filter_str)

Requirements
------------
PyMVPA must be installed and on the Python path.
"""

from __future__ import annotations

import glob
import logging
import os
import re
from pathlib import Path
from typing import Iterable, List

import nibabel as nib
import numpy as np
import mvpa2.suite as mvpa  # type: ignore

from modules import EXPERT_SUBJECTS, NONEXPERT_SUBJECTS


_LOG = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def infer_chance_from_name(fname: str) -> float:
    """Infer chance level from filename.

    If ``decoding`` is in the filename, the number of classes is inferred from
    keywords ``visual`` (20), ``check`` (2) or ``strategy`` (5). Otherwise zero
    is returned.
    """
    name = fname.lower()
    if "decoding" in name:
        if "visual" in name:
            classes = 20
        elif "check" in name:
            classes = 2
        elif "strategy" in name:
            classes = 5
        else:
            raise ValueError(f"Cannot determine number of classes from {fname}")
        return 1.0 / classes
    return 0.0


def collect_nifti_files(root_dir: str, filter_str: str) -> List[str]:
    pattern = os.path.join(root_dir, "**", f"*{filter_str}*.nii*")
    files = sorted(glob.glob(pattern, recursive=True))
    if not files:
        raise RuntimeError(f"No NIfTI files matching '{filter_str}' under {root_dir}")
    return files


def build_group_dataset(paths: Iterable[str]) -> mvpa.Dataset:
    """Load and stack fMRI datasets for a group."""
    datasets = []
    for idx, path in enumerate(paths):
        ds = mvpa.fmri_dataset(path)
        ds.sa["targets"] = 1
        ds.sa["chunks"] = idx
        datasets.append(ds)
    if not datasets:
        raise ValueError("No datasets to stack")
    return mvpa.vstack(datasets)


def compute_tfce(ds: mvpa.Dataset, h0_mean: float | None, niter: int) -> mvpa.Dataset:
    """Run TFCE-based cluster statistics using PyMVPA."""
    nh = mvpa.ClusterNN(ds)
    kwargs = dict(niter=niter)
    if h0_mean is not None:
        kwargs["h0_mean"] = h0_mean
    return mvpa.montecarlo_cluster_stat(ds, nh, **kwargs)


def save_zmap(ds: mvpa.Dataset, reference: str, out_path: Path, threshold: bool = False) -> None:
    """Save dataset as NIfTI image, optionally thresholded at |z|>1.96."""
    img = mvpa.map2nifti(ds, ref=reference)
    data = img.get_fdata()
    if threshold:
        data[np.abs(data) < 1.96] = 0
        img = nib.Nifti1Image(data, img.affine, img.header)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.to_filename(str(out_path))


# ---------------------------------------------------------------------------
# Main analysis pipeline
# ---------------------------------------------------------------------------


def run_group_tfce(root_dir: str, filter_str: str, niter: int = 1000) -> None:
    files = collect_nifti_files(root_dir, filter_str)
    h0_mean = infer_chance_from_name(Path(files[0]).name)

    expert_files, novice_files = [], []
    for fpath in files:
        m = re.search(r"sub-(\d+)", fpath)
        if not m:
            _LOG.warning("Could not parse subject ID from %s", fpath)
            continue
        sid = m.group(1)
        if sid in EXPERT_SUBJECTS:
            expert_files.append(fpath)
        elif sid in NONEXPERT_SUBJECTS:
            novice_files.append(fpath)
        else:
            _LOG.warning("Subject %s not recognised", sid)

    expert_ds = build_group_dataset(expert_files)
    novice_ds = build_group_dataset(novice_files)

    expert_stat = compute_tfce(expert_ds, h0_mean, niter)
    novice_stat = compute_tfce(novice_ds, h0_mean, niter)

    diff_ds = mvpa.vstack([expert_ds, novice_ds])
    diff_ds.sa["targets"] = [1] * len(expert_files) + [2] * len(novice_files)
    diff_ds.sa["chunks"] = list(range(len(expert_files) + len(novice_files)))
    diff_stat = compute_tfce(diff_ds, None, niter)

    res_root = Path("results/mvpa-second-level")
    template = Path(files[0]).relative_to(root_dir)
    stem = template.stem.replace(".nii", "")

    save_zmap(expert_stat, expert_files[0], res_root / "experts" / template.parent / f"{stem}_exp_tfce_z.nii.gz")
    save_zmap(novice_stat, novice_files[0], res_root / "novices" / template.parent / f"{stem}_nov_tfce_z.nii.gz")
    save_zmap(diff_stat, expert_files[0], res_root / "experts_vs_novices" / template.parent / f"{stem}_exp_vs_nov_tfce_z.nii.gz")

    save_zmap(expert_stat, expert_files[0], res_root / "experts" / template.parent / f"{stem}_exp_tfce_z_thr_p05.nii.gz", threshold=True)
    save_zmap(novice_stat, novice_files[0], res_root / "novices" / template.parent / f"{stem}_nov_tfce_z_thr_p05.nii.gz", threshold=True)
    save_zmap(diff_stat, expert_files[0], res_root / "experts_vs_novices" / template.parent / f"{stem}_exp_vs_nov_tfce_z_thr_p05.nii.gz", threshold=True)


if __name__ == "__main__":
    ROOT_DIR = "/path/to/searchlight/results"
    FILTER_STR = "checkmate"
    run_group_tfce(ROOT_DIR, FILTER_STR)
