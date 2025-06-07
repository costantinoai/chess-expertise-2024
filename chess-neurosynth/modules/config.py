#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun  7 22:17:34 2025

@author: costantino_ai
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
import logging

logger = logging.getLogger(__name__)

# Colors
COL_POS = '#006400'
COL_NEG = '#8B0000'
PALETTE = [COL_POS, COL_NEG]

# Plot styles
sns.set_style('white')
sns.set_palette(sns.color_palette(PALETTE))
plt.rcParams['font.family'] = 'Ubuntu Condensed'

# Custom brain colormap
def make_brain_cmap():
    center = plt.cm.RdPu(0)[:3]
    neg = np.linspace([0.0, 0.5, 0.7], center, 256)
    pos = plt.cm.RdPu(np.linspace(0, 1, 256))[:, :3]
    return LinearSegmentedColormap.from_list('custom_brain', np.vstack((neg, pos)))

BRAIN_CMAP = make_brain_cmap()

# Title font settings
TITLE_FONT = {
    'fontfamily': 'Ubuntu Condensed',
    'fontsize': 18,
    'fontweight': 'bold',
    'color': 'black',
    'backgroundcolor': 'white'
}

# Fixed term order
TERM_ORDER = [
    'working memory', 'navigation', 'memory',
    'language', 'object recognition', 'face recognition', 'early visual'
]

# Mapping for run‐ID levels
LEVELS_MAPS = {
    "check>nocheck": "Checkmate > Non-Checkmate",
    "all>rest": "All boards > Baseline",
    "exp": "Experts",
    "nov": "Novices",
    "exp>nov": "Experts > Novices",
    "exp>nonexp": "Experts > Novices",
    "nonexp": "Novices",
}
