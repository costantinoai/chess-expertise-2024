#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Configuration and plotting utilities for the Chess-Neurosynth project.

This module centralises all colour choices, matplotlib settings and other
constants so that plots across the different scripts share the same look and
feel.  Importing it also exposes a ``logger`` object used throughout the
package.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
import logging
import logging
from pathlib import Path

def setup_logging(log_file: str | None = None, level: int = logging.INFO) -> logging.Logger:
    """Configure and return a logger with a consistent format."""
    logger = logging.getLogger()
    if not logger.handlers:
        fmt = "[%(levelname)s %(asctime)s] %(message)s"
        logger.setLevel(level)
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(fmt))
        logger.addHandler(handler)
        if log_file:
            Path(log_file).parent.mkdir(parents=True, exist_ok=True)
            fh = logging.FileHandler(log_file)
            fh.setFormatter(logging.Formatter(fmt))
            logger.addHandler(fh)
    return logger
    
setup_logging()
logger = logging.getLogger(__name__)

# Colors
COL_POS = '#4c924c'
COL_NEG = '#ad4c4c'
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
    'fontsize': 22,
    'fontweight': 'bold',
    'color': 'black',
    'backgroundcolor': 'white'
}

# Fixed term order
TERM_ORDER = [
    'working memory', 'navigation', 'memory retrieval',
    'language network', 'object recognition', 'face recognition', 'early visual'
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
