#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Shared brain plotting utilities (glass brain, surface maps)."""
from __future__ import annotations

import os
from typing import Optional

import numpy as np
from nilearn import image, plotting, datasets, surface


def save_and_open_plotly_figure(fig, title: str = "surface_plot", outdir: str = ".", png_out: Optional[str] = None) -> None:
    os.makedirs(outdir, exist_ok=True)
    basename = title.replace(" ", "_").replace("|", "").replace(":", "").replace(">", "gt")
    html_path = os.path.join(outdir, f"{basename}.html")
    fig.write_html(html_path)
    if png_out is None:
        png_out = os.path.join(outdir, f"{basename}_surface.png")
    fig.write_image(png_out, scale=2)


def plot_surface_map_flat(img, title: str = "Flat Surface Map", threshold: float | None = None, output_file: Optional[str] = None, cmap=None):
    fsaverage = datasets.fetch_surf_fsaverage('fsaverage6')
    hemis = ['left', 'right']
    mesh_dict = {
        'pial': {'left': fsaverage.pial_left, 'right': fsaverage.pial_right},
        'flat': {'left': fsaverage.flat_left, 'right': fsaverage.flat_right}
    }
    sulc_maps = {'left': fsaverage.sulc_left, 'right': fsaverage.sulc_right}

    from plotly.subplots import make_subplots
    fig = make_subplots(rows=1, cols=2, specs=[[{"type": "scene"}, {"type": "scene"}]], horizontal_spacing=0.01, subplot_titles=["Left Hemisphere", "Right Hemisphere"])
    for i, hemi in enumerate(hemis):
        mesh = mesh_dict['flat'][hemi]
        texture = surface.vol_to_surf(img, mesh_dict['pial'][hemi])
        sub_fig = plotting.plot_surf_stat_map(surf_mesh=mesh, stat_map=texture, hemi=hemi, bg_map=sulc_maps[hemi], colorbar=False, threshold=threshold, cmap=cmap, engine="plotly", title=None).figure
        for trace in sub_fig.data:
            fig.add_trace(trace, row=1, col=i + 1)
        fig.update_scenes(dict(xaxis=dict(visible=False), yaxis=dict(visible=False), zaxis=dict(visible=False), aspectmode='data'), row=1, col=i + 1)

    outdir = os.path.dirname(output_file) if output_file else '.'
    os.makedirs(outdir, exist_ok=True)
    safe_title = title.replace(' ', '_').replace('|', '').replace(':', '')
    png_out = output_file or os.path.join(outdir, f"{safe_title}_flat_surface.png")
    save_and_open_plotly_figure(fig, title=title, outdir=outdir, png_out=png_out)


def plot_glass_brain_from_df(df_input, atlas_img, title: str, cmap="cold_hot", base_font_size: int = 22, out_dir: Optional[str] = None):
    atlas_arr = atlas_img.get_fdata()
    diff_map = np.zeros_like(atlas_arr)
    for _, row in df_input.iterrows():
        roi_label = row["ROI_idx"]
        t_value = row["T_Diff"]
        diff_map[atlas_arr == roi_label] = t_value
    sig_diff_img = image.new_img_like(atlas_img, diff_map)
    safe_title = title.replace(" ", "_").replace("|", "").replace(":", "").replace(">", "gt")
    display = plotting.plot_glass_brain(sig_diff_img, title=title, cmap=cmap, colorbar=True, symmetric_cbar=True, plot_abs=False)
    display.title(title, size=base_font_size * 1.4, color="black", bgcolor="white", weight="bold")
    if out_dir is not None:
        out_png = os.path.join(out_dir, f"{safe_title}_glass.png")
        display.savefig(out_png)

