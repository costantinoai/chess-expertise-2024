#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Plotting helpers for GLM summaries.

Includes surface overlay via Glasser annot and glass brain utility.
"""
from __future__ import annotations

import os
import logging
from typing import Dict

import numpy as np
import pandas as pd
from nilearn import plotting, image
from common.brain_plotting import save_and_open_plotly_figure, plot_glass_brain_from_df


## save_and_open_plotly_figure now imported from common.brain_plotting


def plot_sig_rois_on_glasser_surface(
    sig_df: pd.DataFrame,
    glasser_annot_files: Dict[str, str],
    fsaverage,
    title: str = "Flat Glasser Surface Map",
    threshold: float = 0,
    cmap="cold_hot",
    out_dir: str | None = None,
):
    """Plot T-values for ROIs on flat Glasser surface for LH/RH using Plotly engine.

    Expects columns ROI_idx and T_Diff in `sig_df`.
    """
    from nilearn import plotting as _plot
    from plotly.subplots import make_subplots
    import nibabel.freesurfer as fs

    views = ["lateral", "medial"]
    hemispheres = ["left", "right"]
    mesh_dict = {
        "inflated": {"left": fsaverage.infl_left, "right": fsaverage.infl_right},
        "flat": {"left": fsaverage.flat_left, "right": fsaverage.flat_right},
    }
    sulc_maps = {"left": fsaverage.sulc_left, "right": fsaverage.sulc_right}
    camera = dict(eye=dict(x=1.5, y=1.2, z=0.5))

    fig = make_subplots(
        rows=1,
        cols=len(views),
        specs=[[{"type": "scene"} for _ in views]],
        subplot_titles=[view.capitalize() for view in views],
        horizontal_spacing=0.02,
    )

    for i, view in enumerate(views):
        hemi = hemispheres[i % 2]
        annot_path = glasser_annot_files["left" if hemi == "left" else "right"]
        labels, ctab, names = fs.read_annot(annot_path)
        labels = labels.astype(int)
        texture = np.zeros_like(labels, dtype=float)

        for _, row in sig_df.iterrows():
            label_idx = int(row["ROI_idx"]) - 1
            t_val = float(row["T_Diff"])
            if 0 <= label_idx < len(names):
                if abs(t_val) >= threshold:
                    texture[labels == label_idx] = t_val

        sub_fig = _plot.plot_surf_stat_map(
            surf_mesh=mesh_dict["flat"][hemi],
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
            if hasattr(trace, "colorbar"):
                trace.colorbar = dict(
                    thickness=30,
                    len=0.9,
                    tickfont=dict(size=20, family="Ubuntu Condensed"),
                    title=dict(text="T", font=dict(size=28, family="Ubuntu Condensed"), side="right"),
                )
            fig.add_trace(trace, row=1, col=i + 1)

        fig.update_scenes(
            dict(xaxis=dict(visible=False), yaxis=dict(visible=False), zaxis=dict(visible=False), camera=camera, aspectmode="data"),
            row=1,
            col=i + 1,
        )

    fig.update_layout(
        title=dict(text=title, font=dict(size=28, family="Ubuntu Condensed"), y=0.88, x=0.5, xanchor="center"),
        height=500,
        width=850,
        showlegend=False,
        margin=dict(t=30, l=0, r=0, b=0),
    )

    for annotation in fig["layout"].get("annotations", []):
        annotation["font"] = dict(size=24, family="Ubuntu Condensed")
        annotation["y"] -= 0.15
        annotation["yanchor"] = "top"

    if out_dir is not None:
        save_and_open_plotly_figure(fig, title=title, outdir=out_dir, png_out=None)


## plot_glass_brain_from_df now imported from common.brain_plotting
