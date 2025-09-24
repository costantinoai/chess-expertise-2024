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
from logging_utils import setup_logging

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

BASE_DATA_PATH = Path("/data/projects/chess/data")
BIDS_PATH = os.path.join(BASE_DATA_PATH, "BIDS")
DERIVATIVES_PATH = os.path.join(BIDS_PATH, "derivatives")
MVPA_ROOT_PATH = os.path.join(DERIVATIVES_PATH, "/data/projects/chess/data/BIDS/derivatives/mvpa/20250206-214037_mvpa_bilalic_categories")

FS_PATH = os.path.join(DERIVATIVES_PATH, "fastsurfer")
LH_ANNOT =  os.path.join(FS_PATH, "fsaverage", "label", "lh.HCPMMP1.annot")
RH_ANNOT =  os.path.join(FS_PATH, "fsaverage", "label", "rh.HCPMMP1.annot")

ROIS_CSV = os.path.join(BASE_DATA_PATH, "misc/HCP-MMP1_UniqueRegionList.csv")
LEFT_LUT = os.path.join(BASE_DATA_PATH, "misc/lh_HCPMMP1_color_table.txt")
RIGHT_LUT = os.path.join(BASE_DATA_PATH, "misc/rh_HCPMMP1_color_table.txt")

P_ALPHA = .05
FDR_ALPHA = .05
MULTI_CORRECTION = "fdr_bh"

# 3) Load the annotation files
HCPMMP1_LH_LABELS, lh_ctab, HCPMMP1_LH_NAMES = nib.freesurfer.read_annot(str(LH_ANNOT))
HCPMMP1_RH_LABELS, rh_ctab, rh_names = nib.freesurfer.read_annot(str(RH_ANNOT))

HCPMMP1_LH_NAMES_STR = [x.decode().upper() for x in HCPMMP1_LH_NAMES]

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

REGIONS_LABELS = (
    (1, "Primary Visual Cortex"),
    (2, "Medial Superior Temporal Area"),
    (3, "Sixth Visual Area"),
    (4, "Second Visual Area"),
    (5, "Third Visual Area"),
    (6, "Fourth Visual Area"),
    (7, "Eighth Visual Area"),
    (8, "Primary Motor Cortex"),
    (9, "Primary Sensory Cortex"),
    (10, "Frontal Eye Fields"),
    (11, "Premotor Eye Field"),
    (12, "Area 55b"),
    (13, "Area V3A"),
    (14, "RetroSplenial Complex"),
    (15, "Parieto-Occipital Sulcus Area 2"),
    (16, "Seventh Visual Area"),
    (17, "IntraParietal Sulcus Area 1"),
    (18, "Fusiform Face Complex"),
    (19, "Area V3B"),
    (20, "Area Lateral Occipital 1"),
    (21, "Area Lateral Occipital 2"),
    (22, "Posterior InferoTemporal complex"),
    (23, "Middle Temporal Area"),
    (24, "Primary Auditory Cortex"),
    (25, "PeriSylvian Language Area"),
    (26, "Superior Frontal Language Area"),
    (27, "PreCuneus Visual Area"),
    (28, "Superior Temporal Visual Area"),
    (29, "Medial Area 7P"),
    (30, "Area 7m"),
    (31, "Parieto-Occipital Sulcus Area 1"),
    (32, "Area 23d"),
    (33, "Area ventral 23 a+b"),
    (34, "Area dorsal 23 a+b"),
    (35, "Area 31p ventral"),
    (36, "Area 5m"),
    (37, "Area 5m ventral"),
    (38, "Area 23c"),
    (39, "Area 5L"),
    (40, "Dorsal Area 24d"),
    (41, "Ventral Area 24d"),
    (42, "Lateral Area 7A"),
    (43, "Supplementary and Cingulate Eye Field"),
    (44, "Area 6m anterior"),
    (45, "Medial Area 7A"),
    (46, "Lateral Area 7P"),
    (47, "Area 7PC"),
    (48, "Area Lateral IntraParietal ventral"),
    (49, "Ventral IntraParietal Complex"),
    (50, "Medial IntraParietal Area"),
    (51, "Area 1"),
    (52, "Area 2"),
    (53, "Area 3a"),
    (54, "Dorsal area 6"),
    (55, "Area 6mp"),
    (56, "Ventral Area 6"),
    (57, "Area Posterior 24 prime"),
    (58, "Area 33 prime"),
    (59, "Anterior 24 prime"),
    (60, "Area p32 prime"),
    (61, "Area a24"),
    (62, "Area dorsal 32"),
    (63, "Area 8BM"),
    (64, "Area p32"),
    (65, "Area 10r"),
    (66, "Area 47m"),
    (67, "Area 8Av"),
    (68, "Area 8Ad"),
    (69, "Area 9 Middle"),
    (70, "Area 8B Lateral"),
    (71, "Area 9 Posterior"),
    (72, "Area 10d"),
    (73, "Area 8C"),
    (74, "Area 44"),
    (75, "Area 45"),
    (76, "Area 47l (47 lateral)"),
    (77, "Area anterior 47r"),
    (78, "Rostral Area 6"),
    (79, "Area IFJa"),
    (80, "Area IFJp"),
    (81, "Area IFSp"),
    (82, "Area IFSa"),
    (83, "Area posterior 9-46v"),
    (84, "Area 46"),
    (85, "Area anterior 9-46v"),
    (86, "Area 9-46d"),
    (87, "Area 9 anterior"),
    (88, "Area 10v"),
    (89, "Area anterior 10p"),
    (90, "Polar 10p"),
    (91, "Area 11l"),
    (92, "Area 13l"),
    (93, "Orbital Frontal Complex"),
    (94, "Area 47s"),
    (95, "Area Lateral IntraParietal dorsal"),
    (96, "Area 6 anterior"),
    (97, "Inferior 6-8 Transitional Area"),
    (98, "Superior 6-8 Transitional Area"),
    (99, "Area 43"),
    (100, "Area OP4/PV"),
    (101, "Area OP1/SII"),
    (102, "Area OP2-3/VS"),
    (103, "Area 52"),
    (104, "RetroInsular Cortex"),
    (105, "Area PFcm"),
    (106, "Posterior Insular Area 2"),
    (107, "Area TA2"),
    (108, "Frontal OPercular Area 4"),
    (109, "Middle Insular Area"),
    (110, "Pirform Cortex"),
    (111, "Anterior Ventral Insular Area"),
    (112, "Anterior Agranular Insula Complex"),
    (113, "Frontal OPercular Area 1"),
    (114, "Frontal OPercular Area 3"),
    (115, "Frontal OPercular Area 2"),
    (116, "Area PFt"),
    (117, "Anterior IntraParietal Area"),
    (118, "Entorhinal Cortex"),
    (119, "PreSubiculum"),
    (120, "Hippocampus"),
    (121, "ProStriate Area"),
    (122, "Perirhinal Ectorhinal Cortex"),
    (123, "Area STGa"),
    (124, "ParaBelt Complex"),
    (125, "Auditory 5 Complex"),
    (126, "ParaHippocampal Area 1"),
    (127, "ParaHippocampal Area 3"),
    (128, "Area STSd anterior"),
    (129, "Area STSd posterior"),
    (130, "Area STSv posterior"),
    (131, "Area TG dorsal"),
    (132, "Area TE1 anterior"),
    (133, "Area TE1 posterior"),
    (134, "Area TE2 anterior"),
    (135, "Area TF"),
    (136, "Area TE2 posterior"),
    (137, "Area PHT"),
    (138, "Area PH"),
    (139, "Area TemporoParietoOccipital Junction 1"),
    (140, "Area TemporoParietoOccipital Junction 2"),
    (141, "Area TemporoParietoOccipital Junction 3"),
    (142, "Dorsal Transitional Visual Area"),
    (143, "Area PGp"),
    (144, "Area IntraParietal 2"),
    (145, "Area IntraParietal 1"),
    (146, "Area IntraParietal 0"),
    (147, "Area PF opercular"),
    (148, "Area PF Complex"),
    (149, "Area PFm Complex"),
    (150, "Area PGi"),
    (151, "Area PGs"),
    (152, "Area V6A"),
    (153, "VentroMedial Visual Area 1"),
    (154, "VentroMedial Visual Area 3"),
    (155, "ParaHippocampal Area 2"),
    (156, "Area V4t"),
    (157, "Area FST"),
    (158, "Area V3CD"),
    (159, "Area Lateral Occipital 3"),
    (160, "VentroMedial Visual Area 2"),
    (161, "Area 31pd"),
    (162, "Area 31a"),
    (163, "Ventral Visual Complex"),
    (164, "Area 25"),
    (165, "Area s32"),
    (166, "posterior OFC Complex"),
    (167, "Area Posterior Insular 1"),
    (168, "Insular Granular Complex"),
    (169, "Area Frontal Opercular 5"),
    (170, "Area posterior 10p"),
    (171, "Area posterior 47r"),
    (172, "Area TG Ventral"),
    (173, "Medial Belt Complex"),
    (174, "Lateral Belt Complex"),
    (175, "Auditory 4 Complex"),
    (176, "Area STSv anterior"),
    (177, "Area TE1 Middle"),
    (178, "Para-Insular Area"),
    (179, "Area anterior 32 prime"),
    (180, "Area posterior 24"),
)

from modules.roi_manager import ROIManager

# 2) Initialize ROIManager
MANAGER = ROIManager(
    csv_path=ROIS_CSV,
    left_color_table_path=LEFT_LUT,
    right_color_table_path=RIGHT_LUT,
)
