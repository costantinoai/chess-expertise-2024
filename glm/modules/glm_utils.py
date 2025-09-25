#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""GLM analysis utilities (stats + exports).

These helpers consolidate common GLM post-processing steps so entry scripts can
stay thin: confidence intervals, Welch tests, FDR correction, and LaTeX export.
"""
from __future__ import annotations

from typing import Sequence, Tuple, Iterable
import os
import pandas as pd
import numpy as np

from common.stats_utils import (
    mean_ci_t,
    independent_ttest,
    fdr_correction,
)


def welch_two_sample(x: Sequence[float] | np.ndarray, y: Sequence[float] | np.ndarray) -> dict:
    """Welch's unequal-variance t-test with CI on mean difference."""
    return independent_ttest(np.asarray(x), np.asarray(y), equal_var=False)


def ci_on_means(values: Sequence[float] | np.ndarray, confidence: float = 0.95) -> Tuple[float, float, float]:
    """Mean and t-based CI wrapper (two-sided)."""
    return mean_ci_t(np.asarray(values), confidence=confidence)


def fdr_bh(p_values: Iterable[float], alpha: float = 0.05) -> Tuple[np.ndarray, np.ndarray]:
    """Benjamini–Hochberg FDR correction.

    Returns (reject_mask, corrected_pvalues).
    """
    return fdr_correction(np.asarray(list(p_values)), alpha=alpha, method="fdr_bh")


def export_diff_stats_to_latex(df: pd.DataFrame, out_path: str,
                               value_cols: Tuple[str, ...] = ("ROI", "delta_mean", "t", "p", "p_fdr"),
                               caption: str | None = None,
                               label: str | None = None) -> str:
    """Save a compact LaTeX table of per-ROI group differences.

    Parameters
    ----------
    df : DataFrame
        At least the columns listed in value_cols must be present.
    out_path : str
        Destination .tex path.
    value_cols : tuple
        Column order to export.
    caption, label : str | None
        Optional LaTeX caption and label.

    Returns
    -------
    str : absolute path to saved .tex file
    """
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    cols = [c for c in value_cols if c in df.columns]
    table = df[cols].copy()
    table = table.sort_values(by=cols[1] if len(cols) > 1 else cols[0], ascending=False)

    latex = table.to_latex(index=False, float_format=lambda x: f"{x:.3f}")
    if caption:
        latex = latex.replace("\\begin{tabular}", f"\\caption{{{caption}}}\n\\begin{tabular}")
    if label:
        latex = latex.replace("\\begin{tabular}", f"\\label{{{label}}}\n\\begin{tabular}")

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(latex)
    return os.path.abspath(out_path)

