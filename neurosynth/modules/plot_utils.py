#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Neurosynth plotting helpers (delegated to shared utilities).

This module exposes thin wrappers that delegate plotting to the shared
implementations under `common/` to keep a single source of truth for
styles and behavior.
"""

import os
from nilearn import image
from modules.config import (
    PALETTE,
    COL_POS,
    COL_NEG,
)


def plot_map(arr, ref_img, title, outpath, thresh=1e-5):
    """Delegate to shared brain plotting utility."""
    from common.brain_plotting import plot_map as _plot_map
    return _plot_map(arr, ref_img, title, outpath, thresh)


def plot_term_maps(term_maps, out_dir):
    """
    Plot and save brain maps for each term in term_maps.

    Parameters
    ----------
    term_maps : dict
        Mapping from term name to NIfTI file path.
    out_dir : str
        Directory to save term map PNGs.
    """
    os.makedirs(out_dir, exist_ok=True)
    for term, path in term_maps.items():
        data_img = image.load_img(path)
        data = data_img.get_fdata()
        filename = f"term_{term.replace(' ', '_')}.png"
        plot_map(
            data,
            data_img,
            f"Term: {term.title()}",
            os.path.join(out_dir, filename),
        )


def plot_correlations(df_pos, df_neg, df_diff, out_fig=None, out_csv=None, run_id=None):
    """
    Paired bar plot (POS vs NEG), delegated to shared plotting.
    """
    from common.stats_plotting import plot_correlations as _plot
    return _plot(
        df_pos,
        df_neg,
        df_diff,
        out_fig=out_fig,
        out_csv=out_csv,
        run_id=run_id,
        col_pos=COL_POS,
        col_neg=COL_NEG,
        palette=PALETTE,
    )


def plot_difference(diff_df, out_fig=None, run_id=None, col_pos=COL_POS, col_neg=COL_NEG):
    """
    Plot bar chart of correlation differences (pos - neg), delegated to shared plotting.
    """
    from common.stats_plotting import plot_difference as _plot
    return _plot(
        diff_df,
        out_fig=out_fig,
        run_id=run_id,
        col_pos=col_pos,
        col_neg=col_neg,
    )

