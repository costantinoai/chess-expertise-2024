#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun  7 22:17:34 2025

@author: costantino_ai
"""

import logging
from common import plot_style
from common.plot_style import (
    COL_POS, COL_NEG, PALETTE,
    BRAIN_CMAP, TITLE_FONT,
)

logger = logging.getLogger(__name__)

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
