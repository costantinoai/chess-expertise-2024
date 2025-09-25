from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Mapping, Tuple

import numpy as np
import pandas as pd
import nibabel as nib
import scipy.io as sio
from scipy.stats import ttest_ind
from sklearn.decomposition import PCA
from common.stats_utils import fdr_correction
## joblib used by root runner

## Logging and run orchestration live in the root runner
from config import GLM_BASE_PATH, ATLAS_CORTICES, EXPERTS, NONEXPERTS
from meta import ROI_NAME_MAP, apply_plot_style


apply_plot_style(28)


@dataclass(frozen=True)
class Config:
    base_path: Path = Path(str(GLM_BASE_PATH))
    spm_filename: str = "SPM.mat"
    atlas_file: Path = Path(str(ATLAS_CORTICES))
    expert_subjects: Tuple[str, ...] = tuple(EXPERTS)
    nonexpert_subjects: Tuple[str, ...] = tuple(NONEXPERTS)
    alpha_fdr: float = 0.05
    use_parallel: bool = True
    n_jobs: int = -1
    roi_name_map: Mapping[int, str] = field(default_factory=lambda: dict(ROI_NAME_MAP))


logger = logging.getLogger(__name__)


def _assert_file_exists(path: Path, label: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"{label} not found: {path}")


def _spm_dir_from_swd(spm_obj, fallback: Path) -> Path:
    swd = getattr(spm_obj, "swd", None)
    return Path(swd) if swd else fallback


def get_spm_condition_betas(subject_id: str, cfg: Config) -> Dict[str, nib.Nifti1Image]:
    spm_mat_path = cfg.base_path / f"sub-{subject_id}" / "exp" / cfg.spm_filename
    _assert_file_exists(spm_mat_path, "SPM.mat")
    mat = sio.loadmat(spm_mat_path.as_posix(), struct_as_record=False, squeeze_me=True)
    SPM = mat["SPM"]
    betas = SPM.Vbeta
    names = SPM.xX.name
    pattern = r"Sn\(\d+\)\s+(.*?)\*bf\(1\)"
    cond_indices: Dict[str, list[int]] = {}
    for idx, name in enumerate(names):
        m = re.match(pattern, name)
        if m:
            cond = m.group(1)
            cond_indices.setdefault(cond, []).append(idx)
    swd = _spm_dir_from_swd(SPM, spm_mat_path.parent)
    averaged: Dict[str, nib.Nifti1Image] = {}
    for cond, indices in cond_indices.items():
        sum_data = None
        affine = header = None
        for i in indices:
            img = nib.load((swd / betas[i].fname).as_posix())
            data = img.get_fdata()
            if sum_data is None:
                sum_data = np.zeros_like(data)
                affine, header = img.affine, img.header
            sum_data += data
        averaged[cond] = nib.Nifti1Image(sum_data / len(indices), affine, header)
    return averaged


def load_roi_voxel_data(subject_id: str, atlas_data: np.ndarray, unique_rois: np.ndarray, cfg: Config) -> Dict[int, np.ndarray]:
    betas = get_spm_condition_betas(subject_id, cfg)
    conditions = sorted(betas.keys())
    output = {roi: np.zeros((len(conditions), (atlas_data == roi).sum()), dtype=np.float32) for roi in unique_rois}
    for i, cond in enumerate(conditions):
        data = betas[cond].get_fdata()
        for roi in unique_rois:
            output[roi][i, :] = data[atlas_data == roi]
    return output


def compute_pr(roi_data: np.ndarray) -> Tuple[float, int]:
    valid = ~np.isnan(roi_data).all(axis=0)
    cleaned = roi_data[:, valid]
    if cleaned.shape[1] < 2:
        return np.nan, 0
    var = PCA().fit(cleaned).explained_variance_
    return (var.sum() ** 2) / np.sum(var ** 2), cleaned.shape[1]


def process_subject(subject_id: str, atlas_data: np.ndarray, unique_rois: np.ndarray, cfg: Config) -> Tuple[np.ndarray, np.ndarray]:
    roi_data = load_roi_voxel_data(subject_id, atlas_data, unique_rois, cfg)
    pr, vox = [], []
    for roi in unique_rois:
        p, v = compute_pr(roi_data[roi])
        pr.append(p)
        vox.append(v)
    return np.array(pr), np.array(vox)


def per_roi_welch_and_fdr(expert_pr: np.ndarray, novice_pr: np.ndarray, roi_labels: np.ndarray, alpha: float) -> pd.DataFrame:
    rows = []
    for j, roi in enumerate(roi_labels):
        x = expert_pr[:, j]
        y = novice_pr[:, j]
        mask = np.isfinite(x) & np.isfinite(y)
        t, p = ttest_ind(x[mask], y[mask], equal_var=False)
        rows.append({"ROI": int(roi), "t": float(t), "p": float(p), "delta_mean": np.nanmean(x - y)})
    df = pd.DataFrame(rows)
    df["p_fdr"] = fdr_correction(df["p"].values, alpha=alpha, method="fdr_bh")[1]
    return df


def consolidate_results(expert_pr: np.ndarray, novice_pr: np.ndarray, roi_labels: np.ndarray, stats_df: pd.DataFrame, roi_name_map: Mapping[int, str]) -> pd.DataFrame:
    df = pd.DataFrame({
        "ROI": roi_labels.astype(int),
        "ROI_Label": [roi_name_map.get(int(r), str(int(r))) for r in roi_labels],
        "PR_Expert": np.nanmean(expert_pr, axis=0),
        "PR_NonExpert": np.nanmean(novice_pr, axis=0),
    })
    df["delta_mean"] = df["PR_Expert"] - df["PR_NonExpert"]
    df = df.merge(stats_df[["ROI", "p", "p_fdr", "delta_mean"]], on="ROI", suffixes=("", "_stat"))
    return df


## No main() here: orchestration lives in run_pr_participation_ratio.py
