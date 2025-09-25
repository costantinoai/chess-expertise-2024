#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Utilities for univariate Neurosynth analyses (plotting and stats wrappers)."""
from __future__ import annotations

import os
import logging
import numpy as np
import pingouin as pg
from plotly.subplots import make_subplots
from nilearn import plotting, surface, datasets
from common.stats_utils import corr_diff_bootstrap, pearson_corr_bootstrap


logger = logging.getLogger(__name__)


def save_and_open_plotly_figure(fig, title='surface_plot', outdir='.', png_out=None):
    """Save Plotly figure to HTML and PNG.

    HTML is '<outdir>/<title>.html' and PNG is '<outdir>/<title>_surface.png' unless
    `png_out` is specified.
    """
    os.makedirs(outdir, exist_ok=True)
    basename = title.replace(' ', '_').replace('|', '').replace(':', '')
    html_path = os.path.join(outdir, f"{basename}.html")
    fig.write_html(html_path)
    logger.info("Figure saved to: %s", html_path)
    if png_out is None:
        png_out = os.path.join(outdir, f"{basename}_surface.png")
    fig.write_image(png_out)
    logger.info("PNG image saved to: %s", png_out)
    

def plot_surface_map(img, title='Surface Map', threshold=None, output_file=None, cmap=None):
    """Project volumetric image onto inflated cortical surfaces and build Plotly figure."""
    fsaverage = datasets.fetch_surf_fsaverage()
    views = [('medial', 'left'), ('lateral', 'left'), ('medial', 'right'), ('lateral', 'right')]
    surface_types = ['inflated']

    mesh_dict = {
        'pial': {'left': fsaverage.pial_left, 'right': fsaverage.pial_right},
        'inflated': {'left': fsaverage.infl_left, 'right': fsaverage.infl_right},
    }
    sulc_dict = {'left': fsaverage.sulc_left, 'right': fsaverage.sulc_right}
    view_angles = {
        'lateral': dict(x=2, y=0, z=0.1),
        'medial': dict(x=-2, y=0, z=0.1),
    }

    n_rows = len(surface_types)
    n_cols = len(views)
    fig = make_subplots(
        rows=n_rows, cols=n_cols,
        specs=[[{"type": "scene"} for _ in range(n_cols)] for _ in range(n_rows)],
        horizontal_spacing=0.005,
        vertical_spacing=0.01,
        subplot_titles=["" for _ in range(n_rows * n_cols)]
    )

    for row, surf_type in enumerate(surface_types, start=1):
        for col, (view, hemi) in enumerate(views, start=1):
            mesh = mesh_dict[surf_type][hemi]
            texture = surface.vol_to_surf(img, mesh_dict['pial'][hemi])
            show_cb = (col == n_cols)
            sub_fig_wrapper = plotting.plot_surf_stat_map(
                surf_mesh=mesh,
                stat_map=texture,
                hemi=hemi,
                view=view,
                bg_map=sulc_dict[hemi],
                colorbar=show_cb,
                threshold=threshold,
                cmap=cmap,
                engine="plotly",
                title=None,
            )
            sub_fig = sub_fig_wrapper.figure
            for trace in sub_fig.data:
                if show_cb and hasattr(trace, 'colorbar'):
                    trace.colorbar.thickness = 20
                    trace.colorbar.len = 0.8
            for trace in sub_fig.data:
                fig.add_trace(trace, row=row, col=col)
            fig.update_scenes(
                dict(
                    xaxis=dict(visible=False), yaxis=dict(visible=False), zaxis=dict(visible=False),
                    camera=dict(eye=view_angles[view]), aspectmode='data'
                ), row=row, col=col
            )

    return fig


def build_design_matrix(n_exp: int, n_nov: int):
    """Return DataFrame with intercept and group (+1 experts, -1 novices)."""
    import pandas as pd
    intercept = np.ones(n_exp + n_nov)
    group = np.concatenate([np.ones(n_exp), -np.ones(n_nov)])
    return pd.DataFrame({'intercept': intercept, 'group': group})


def bootstrap_corr(x, y, n_boot):
    """Bootstrap Pearson correlation (delegates to common.stats_utils)."""
    # Return a Pingouin-like small adapter dict to keep callers working
    res = pearson_corr_bootstrap(np.asarray(x), np.asarray(y), n_boot=n_boot, ci=0.95)
    class _Obj:
        def __init__(self, d):
            import pandas as pd
            self._df = pd.DataFrame({
                'r': [d['r']],
                'p-val': [d['p']],
                'CI95%': [(d['ci_low'], d['ci_high'])],
            })
        def __getitem__(self, k):
            return self._df[k]
    return _Obj(res)


def extract_corr_results(result):
    """Extract r, CI low/high, and p-value from Pingouin result DataFrame."""
    r = result['r'].iloc[0]
    p = result['p-val'].iloc[0]
    ci_lo, ci_hi = result['CI95%'].iloc[0]
    return float(r), float(ci_lo), float(ci_hi), float(p)


def bootstrap_corr_diff(term_map, x, y, n_boot, rng, ci_alpha, n_jobs):
    """Proxy to common.stats_utils.corr_diff_bootstrap for convenience."""
    return corr_diff_bootstrap(term_map, x, y, n_boot=n_boot, ci_alpha=ci_alpha, n_jobs=n_jobs, rng=rng)


def plot_surface_map_flat(img, title='Flat Surface Map', threshold=None, output_file=None, cmap=None):
    """Project volumetric image onto flat cortical surfaces and build Plotly figure."""
    fsaverage = datasets.fetch_surf_fsaverage('fsaverage6')
    hemis = ['left', 'right']
    mesh_dict = {
        'pial': {'left': fsaverage.pial_left, 'right': fsaverage.pial_right},
        'flat': {'left': fsaverage.flat_left, 'right': fsaverage.flat_right}
    }
    sulc_maps = {'left': fsaverage.sulc_left, 'right': fsaverage.sulc_right}

    fig = make_subplots(rows=1, cols=2, specs=[[{"type": "scene"}, {"type": "scene"}]],
                        horizontal_spacing=0.01, subplot_titles=["Left Hemisphere", "Right Hemisphere"])
    for i, hemi in enumerate(hemis):
        mesh = mesh_dict['flat'][hemi]
        texture = surface.vol_to_surf(img, mesh_dict['pial'][hemi])
        sub_fig = plotting.plot_surf_stat_map(
            surf_mesh=mesh,
            stat_map=texture,
            hemi=hemi,
            bg_map=sulc_maps[hemi],
            colorbar=False,
            threshold=threshold,
            cmap=cmap,
            engine="plotly",
            title=None,
        ).figure
        for trace in sub_fig.data:
            fig.add_trace(trace, row=1, col=i + 1)
        fig.update_scenes(
            dict(xaxis=dict(visible=False), yaxis=dict(visible=False), zaxis=dict(visible=False), aspectmode='data'),
            row=1, col=i + 1
        )
    return fig


def find_nifti_files(data_dir: str, pattern: str | None = None) -> list[str]:
    """Recursively list .nii.gz under `data_dir`, optionally filtering by substring `pattern`."""
    out: list[str] = []
    for root, _, files in os.walk(data_dir):
        for f in files:
            if f.endswith('.nii.gz') and (pattern is None or pattern in f):
                out.append(os.path.join(root, f))
    return sorted(out)


def split_by_group(files: list[str], expert_ids: list[str], novice_ids: list[str]) -> tuple[list[str], list[str]]:
    """Split file paths into expert/novice by matching 'sub-<ID>' in filename."""
    exp, nov = [], []
    for sid in expert_ids:
        exp += [f for f in files if f"sub-{sid}" in os.path.basename(f)]
    for sid in novice_ids:
        nov += [f for f in files if f"sub-{sid}" in os.path.basename(f)]
    return exp, nov


def load_and_mask_imgs(file_list: list[str]):
    """Load NIfTI images and mask them with ICBM GM brain mask; returns (masked_imgs, brain_mask)."""
    from nilearn.image import load_img, math_img
    from modules.stats_utils import get_brain_mask
    ref_img = load_img(file_list[0])
    brain_mask = get_brain_mask(ref_img)
    masked_imgs = [math_img("np.where(np.squeeze(mask), np.squeeze(img), np.nan)", img=load_img(f), mask=brain_mask)
                   for f in file_list]
    return masked_imgs, brain_mask


def t_to_two_tailed_z(t_map: np.ndarray, dof: int) -> np.ndarray:
    """Convert a t-map to a signed, two-tailed z-map (retain sign)."""
    from scipy.stats import t, norm
    t_abs = np.abs(t_map)
    p_two_tailed = 2 * t.sf(t_abs, df=dof)
    z_abs = norm.isf(p_two_tailed / 2)
    z_map = np.sign(t_map) * z_abs
    z_map[t_abs == 0] = 0.0
    return z_map


def split_zmap_by_sign(z_map: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (z_pos, z_neg) where z_neg is negative values kept negative."""
    z_pos = np.where(z_map > 0, z_map, 0.0)
    z_neg = np.where(z_map < 0, z_map, 0.0)
    return z_pos, z_neg
