#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Behavioural stats helpers delegating to shared utilities.

Thin wrappers used by behavioural analyses to keep a single source of truth
for statistical routines (CIs, FDR, correlations).
"""
from __future__ import annotations

import numpy as np
from typing import Tuple

from common.stats_utils import (
    mean_ci_t,
    fdr_correction,
    pearson_corr_bootstrap,
    corr_diff_bootstrap,
)

__all__ = [
    "mean_ci_t",
    "fdr_correction",
    "pearson_corr_bootstrap",
    "corr_diff_bootstrap",
]

