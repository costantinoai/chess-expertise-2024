#!/usr/bin/env python3
"""Common settings for manifold analysis."""

from __future__ import annotations

import logging
from pathlib import Path

logging.basicConfig(
    level=logging.INFO, format="[%(levelname)s %(asctime)s] %(message)s"
)
logger = logging.getLogger(__name__)

# Paths
BASE_GLM_PATH = "/data/projects/chess/data/BIDS/derivatives/fmriprep-SPM_smoothed-NO_GS-FD-HMP_brainmasked/MNI/fmriprep-SPM-MNI/GLM"
ATLAS_FILE = "/data/projects/chess/data/misc/templates/tpl-MNI152NLin2009cAsym_res-02_atlas-Glasser2016_desc-bilateral_resampled.nii"

# Subjects
EXPERTS = [
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
]
NONEXPERTS = [
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
]
