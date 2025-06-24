"""Representational connectivity analysis package."""

from .rc_analysis import (
    load_spm_betas,
    extract_roi_data,
    compute_rdm,
    compute_subject_matrix,
    fisher_z,
    group_statistics,
    group_difference,
    ATLAS_FILE,
)
from .plotting import plot_connectivity_matrix
from .run_rc import run_representational_connectivity

__all__ = [
    "load_spm_betas",
    "extract_roi_data",
    "compute_rdm",
    "compute_subject_matrix",
    "fisher_z",
    "group_statistics",
    "group_difference",
    "ATLAS_FILE",
    "plot_connectivity_matrix",
    "run_representational_connectivity",
]
