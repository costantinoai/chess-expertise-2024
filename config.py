"""
Central configuration for paths and subjects.
Values can be overridden via environment variables to adapt to different machines.

Env vars (optional):
- CHESS_DATA_ROOT: base data directory (default: /data/projects/chess/data)
- CHESS_GLM_PATH: override GLM base path
- CHESS_MVPA_RESULTS: override MVPA derivatives path
"""
from __future__ import annotations

import os
from pathlib import Path


DATA_ROOT = Path(os.environ.get("CHESS_DATA_ROOT", "/data/projects/chess/data")).resolve()
BIDS_PATH = DATA_ROOT / "BIDS"
DERIVATIVES_PATH = BIDS_PATH / "derivatives"
SOURCEDATA_PATH = BIDS_PATH / "sourcedata"

# GLM base path (can be overridden)
GLM_BASE_PATH = Path(
    os.environ.get(
        "CHESS_GLM_PATH",
        DERIVATIVES_PATH
        / "fmriprep-SPM_smoothed-NO_GS-FD-HMP_brainmasked/MNI/fmriprep-SPM-MNI/GLM",
    )
).resolve()

# MVPA results root (generic parent folder for MVPA derivatives)
MVPA_RESULTS_ROOT = Path(
    os.environ.get("CHESS_MVPA_RESULTS", DERIVATIVES_PATH / "mvpa")
).resolve()

MISC_PATH = DATA_ROOT / "misc"
TEMPLATES_PATH = MISC_PATH / "templates"

# Common atlases
ATLAS_CORTICES = (
    TEMPLATES_PATH
    / "tpl-MNI152NLin2009cAsym_res-02_atlas-Glasser2016_desc-cortices_bilateral_resampled.nii"
)
ATLAS_BILATERAL = (
    TEMPLATES_PATH
    / "tpl-MNI152NLin2009cAsym_res-02_atlas-Glasser2016_desc-bilateral_resampled.nii"
)

# ROI lookup resources (fallback to repo paths if DATA_ROOT lacks misc)
HCP_ROI_CSV = Path(
    os.environ.get(
        "CHESS_HCP_ROI_CSV",
        (Path("rois/data/HCP-MMP1_UniqueRegionList.csv")).as_posix(),
    )
)
HCP_LH_LUT = Path(
    os.environ.get(
        "CHESS_HCP_LH_LUT",
        (MISC_PATH / "lh_HCPMMP1_color_table.txt").as_posix(),
    )
)
HCP_RH_LUT = Path(
    os.environ.get(
        "CHESS_HCP_RH_LUT",
        (MISC_PATH / "rh_HCPMMP1_color_table.txt").as_posix(),
    )
)

# Behavioural inputs (override via env on different machines)
PARTICIPANTS_XLSX = Path(
    os.environ.get(
        "CHESS_PARTICIPANTS_XLSX",
        Path("data/participants.xlsx").as_posix(),
    )
).resolve()

CATEGORIES_XLSX = Path(
    os.environ.get(
        "CHESS_CATEGORIES_XLSX",
        Path("behavioural/data/categories.xlsx").as_posix(),
    )
).resolve()

DNN_RESPONSES_CSV = Path(
    os.environ.get(
        "CHESS_DNN_RESPONSES_CSV",
        Path("behavioural/data/correct_responses_human_and_net.csv").as_posix(),
    )
).resolve()

# Subject lists
EXPERTS = (
    "03",
    "04",
    "06",
    "07",
    "08",
    "09",
    "10",
    "11",
    "12",
    "13",
    "16",
    "20",
    "22",
    "23",
    "24",
    "29",
    "30",
    "33",
    "34",
    "36",
)

NONEXPERTS = (
    "01",
    "02",
    "15",
    "17",
    "18",
    "19",
    "21",
    "25",
    "26",
    "27",
    "28",
    "32",
    "35",
    "37",
    "39",
    "40",
    "41",
    "42",
    "43",
    "44",
)
