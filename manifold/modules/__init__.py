#!/usr/bin/env python3
"""Common settings for manifold analysis."""

from __future__ import annotations

import logging

logging.basicConfig(
    level=logging.INFO, format="[%(levelname)s %(asctime)s] %(message)s"
)
logger = logging.getLogger(__name__)

# Paths (import from central config; override via env vars if needed)
from config import GLM_BASE_PATH, ATLAS_CORTICES, EXPERTS as _EXPERTS, NONEXPERTS as _NONEXPERTS
BASE_GLM_PATH = str(GLM_BASE_PATH)
ATLAS_FILE = str(ATLAS_CORTICES)

# Subjects
EXPERTS = list(_EXPERTS)
NONEXPERTS = list(_NONEXPERTS)
