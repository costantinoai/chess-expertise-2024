#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 18 14:41:16 2025

@author: costantino_ai
"""
import os
import logging
import matplotlib.pyplot as plt
from pathlib import Path
import warnings
import nibabel as nib
from common.logging_utils import setup_logging
from config import (
    DATA_ROOT,
    BIDS_PATH as _BIDS_PATH,
    DERIVATIVES_PATH as _DERIVATIVES_PATH,
    MVPA_RESULTS_ROOT as _MVPA_RESULTS_ROOT,
)

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
setup_logging()

# Global settings for publication-quality plots
BASE_FONT_SIZE = 30  # Base font size for scaling (change this to scale all fonts)

plt.rcParams["figure.figsize"] = (14, 10)  # Slightly larger figure size
plt.rcParams["figure.dpi"] = 300  # High DPI for publication-ready images
plt.rcParams["font.size"] = BASE_FONT_SIZE  # Base font size for text
plt.rcParams["axes.titlesize"] = BASE_FONT_SIZE * 1.2  # Title font size (larger for emphasis)
plt.rcParams["axes.labelsize"] = BASE_FONT_SIZE  # Axis label font size
plt.rcParams["xtick.labelsize"] = BASE_FONT_SIZE * 0.9  # X-axis tick font size
plt.rcParams["ytick.labelsize"] = BASE_FONT_SIZE * 0.9  # Y-axis tick font size
plt.rcParams["legend.fontsize"] = BASE_FONT_SIZE * 0.7  # Legend font size (slightly smaller)
plt.rcParams["legend.title_fontsize"] = BASE_FONT_SIZE * 0.7  # Legend title font size
plt.rcParams["legend.frameon"] = False  # Make legend borderless
plt.rcParams["legend.loc"] = "upper right"  # Legend in the top-right corner
plt.rcParams["savefig.bbox"] = "tight"  # Ensure plots save tightly cropped
plt.rcParams["savefig.pad_inches"] = 0.1  # Add small padding around saved plots
plt.rcParams["savefig.format"] = "png"  # Default save format

BASE_DATA_PATH = Path(DATA_ROOT)
BIDS_PATH = str(_BIDS_PATH)
DERIVATIVES_PATH = str(_DERIVATIVES_PATH)
MVPA_ROOT_PATH = str(_MVPA_RESULTS_ROOT)

FS_PATH = os.path.join(DERIVATIVES_PATH, "fastsurfer")
LH_ANNOT =  os.path.join(FS_PATH, "fsaverage", "label", "lh.HCPMMP1.annot")
RH_ANNOT =  os.path.join(FS_PATH, "fsaverage", "label", "rh.HCPMMP1.annot")

# Deprecated LUT/CSV constants removed. Use `rois/` metadata + NIfTI inputs instead.

P_ALPHA = .05
FDR_ALPHA = .05
MULTI_CORRECTION = "fdr_bh"

# 3) Load the annotation files
HCPMMP1_LH_LABELS, lh_ctab, HCPMMP1_LH_NAMES = nib.freesurfer.read_annot(str(LH_ANNOT))
HCPMMP1_RH_LABELS, rh_ctab, HCPMMP1_RH_NAMES = nib.freesurfer.read_annot(str(RH_ANNOT))

HCPMMP1_LH_NAMES_STR = [x.decode().upper() for x in HCPMMP1_LH_NAMES]
HCPMMP1_RH_NAMES_STR = [x.decode().upper() for x in HCPMMP1_RH_NAMES]

# Subject Lists
EXPERT_SUBJECTS = (
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

NONEXPERT_SUBJECTS = (
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

# Define the groups and corresponding colors for the custom legend
CORTICES_GROUPS_CMAPS = [
    ("Early Visual", "#a6cee3"),
    ("Intermediate Visual", "#1f78b4"),
    ("Sensorimotor", "#b2df8a"),
    ("Auditory", "#33a02c"),
    ("Temporal", "#fb9a99"),
    ("Posterior", "#e31a1c"),
    ("Anterior", "#fdbf6f")
]

# Contrasts + Chance Levels in dictionary form
# BILALIC: are regressors on the half dataset (only checkmate, 20 stimuli)
# OLD: our categories on all the stimuli (40 stim)
CONTRAST_MAP = {
    "visualStimuli": {"chance": 1 / 20},    # OLD: 20 pairs of stimuli
    "checkmate": {"chance": 1 / 2},         # OLD: 40 stim, check vs no-check
    "categories": {"chance": 1 / 10},       # OLD: 40 stim, Felipe strategies
    "stimuli": {"chance": 1 / 40},          # OLD: all our stimuli, one label per stim
    "categories_half": {"chance": 1 / 5},   # BILALIC: these are our old "categories" but for 20 stim
    "side": {"chance": 1 / 2},              # BILALIC: ???
    "difficulty": {"chance": 1 / 2},        # BILALIC --> may be redundant
    "motif": {"chance": 1 / 4},             # BILALIC
    "first_piece": {"chance": 1 / 4},       # BILALIC
    "check_n": {"chance": 1 / 3},           # BILALIC: 1,3,4
    "checkmate_piece": {"chance": 1 / 4},   # BILALIC
    "stimuli_half": {"chance": 1 / 20},     # BILALIC: Used for the checkmate stim only on bilalic cat
    "total_pieces": {"chance": 1 / 13},
    "legal_moves": {"chance": 1 / 19},
}

CORTICES_NAMES = {
    "L_anterior_cingulate_and_medial_prefrontal_cortex": "Anterior Cingulate and Medial Prefrontal",
    "L_auditory_association_cortex": "Auditory Association",
    "L_dorsal_stream_visual_cortex": "Dorsal Stream Visual",
    "L_dorsolateral_prefrontal_cortex": "Dorsolateral Prefrontal",
    "L_early_auditory_cortex": "Early Auditory",
    "L_early_visual_cortex": "Early Visual",
    "L_inferior_frontal_cortex": "Inferior Frontal",
    "L_inferior_parietal_cortex": "Inferior Parietal",
    "L_insular_and_frontal_opercular_cortex": "Insular and Frontal Opercular",
    "L_lateral_temporal_cortex": "Lateral Temporal",
    "L_medial_temporal_cortex": "Medial Temporal",
    "L_mt__complex_and_neighboring_visual_areas_cortex": "MT+ Complex and Neighboring Visual Areas",
    "L_orbital_and_polar_frontal_cortex": "Orbital and Polar Frontal",
    "L_paracentral_lobular_and_mid_cingulate_cortex": "Paracentral Lobular and Mid Cingulate",
    "L_posterior_cingulate_cortex": "Posterior Cingulate",
    "L_premotor_cortex": "Premotor",
    "L_primary_visual_cortex": "Primary Visual",
    "L_somatosensory_and_motor_cortex": "Somatosensory and Motor",
    "L_superior_parietal_cortex": "Superior Parietal",
    "L_temporo_parieto_occipital_junction_cortex": "Temporo-Parieto-Occipital Junction",
    "L_ventral_stream_visual_cortex": "Ventral Stream Visual",
    "L_posterior_opercular_cortex": "Posterior Opercular"
}

from rois.meta import get_roi_info
_roi_info = get_roi_info("glasser_regions_bilateral")
# Build a 1-indexed list so that REGIONS_LABELS[region_id] -> (id, name)
_max_id = max(_roi_info.id_to_name) if _roi_info.id_to_name else 0
_tmp = [None] * (_max_id + 1)
for idx, name in _roi_info.id_to_name.items():
    _tmp[int(idx)] = (int(idx), name)
REGIONS_LABELS = tuple(_tmp)

# ROIManager was deprecated. Use rois/meta + rois/io for ROI metadata and atlas I/O.
