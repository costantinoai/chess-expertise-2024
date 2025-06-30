#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
compute_mds_carousels.py

Compute Procrustes‑aligned 2D/3D MDS embeddings for each layer of
two models (trained/untrained) and generate interactive Plotly carousels
with attribute‑based coloring.

Special behavior:
  - For 'Checkmate vs. Non‑checkmate' and 'Visual similarity', show all 40 points.
  - For all other attributes, hide the last 20 points (only first 20 meaningful).
  - Uses the 'Set2' categorical colormap for better visual separation.
"""

import os
import pickle
import logging
import numpy as np
import pandas as pd
from sklearn.manifold import MDS
from scipy.spatial import procrustes
import plotly.graph_objects as go
import plotly.io as pio
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from datetime import datetime


def create_run_id() -> str:
    """Timestamp‑based unique ID."""
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def setup_logging():
    """Configure INFO‑level logging."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S"
    )


def load_metadata(path: str, skip: set, mapping: dict):
    """
    Load Excel metadata, forward‑fill, sort, and extract attributes.
    Returns df_meta, attrs_dict, labels.
    """
    logging.info(f"Loading metadata from '{path}'")
    df = pd.read_excel(path).ffill()
    if "stim_id" in df.columns:
        df = df.sort_values("stim_id").reset_index(drop=True)
        logging.info("Sorted by 'stim_id'")
    attrs = {}
    for orig, human in mapping.items():
        if orig in skip: continue
        if orig not in df.columns:
            logging.warning(f"Missing '{orig}' → skip")
            continue
        attrs[human] = df[orig].tolist()
    labels = list(attrs.keys())
    logging.info(f"Extracted {len(labels)} attributes")
    return df, attrs, labels


def build_color_map(attrs: dict, cmap_name: str="Set2"):
    """
    Map each attribute’s categories to hex colors via matplotlib.
    Returns {label: [color_per_point]}.
    """
    import seaborn as sns
    logging.info(f"Building color map with '{cmap_name}'")
    try:
        cmap = cm.get_cmap(cmap_name)
    except:
        cmap = sns.color_palette(cmap_name, as_cmap=True)
    cmap_map = {}
    for lab, vals in attrs.items():
        codes = pd.Categorical(vals).codes
        n = int(codes.max())+1
        palette = [mcolors.to_hex(cmap(i/(n-1))) for i in range(n)] if n>1 else [mcolors.to_hex(cmap(0))]
        cmap_map[lab] = [palette[c] for c in codes]
        logging.info(f"  {lab}: {n} categories")
    return cmap_map


def load_activations(act_dir: str, tags: list):
    """
    Load pickled activations_model-{tag}_seed-0.pkl for each tag.
    """
    acts = {}
    for tag in tags:
        path = os.path.join(act_dir, f"activations_model-{tag}_seed-0.pkl")
        logging.info(f"Loading activations for '{tag}'")
        with open(path, "rb") as f:
            acts[tag] = pickle.load(f)
    return acts


def compute_aligned_mds(acts, df_meta, dims=2):
    """
    Compute MDS(dims) per layer → Procrustes-align to first layer.
    Returns aligned[tag][layer]=(N,dims).
    """
    mds = MDS(n_components=dims, dissimilarity="euclidean", random_state=0)
    aligned = {}
    stim_ids = df_meta["stim_id"].tolist() if "stim_id" in df_meta.columns else list(df_meta.index)
    for tag, layers in acts.items():
        logging.info(f"Computing {dims}D MDS for '{tag}'")
        raw = {}
        for lyr, stim_dict in layers.items():
            X = np.vstack([stim_dict[s]["activation"].ravel() for s in stim_ids])
            raw[lyr] = mds.fit_transform(X)
        keys = list(raw.keys())
        ref = raw[keys[0]]
        aligned[tag] = {}
        for lyr in keys:
            _, A, _ = procrustes(ref, raw[lyr])
            aligned[tag][lyr] = A
        logging.info(f"  Aligned {len(keys)} layers")
    return aligned


def generate_carousel(aligned, attrs, labels, color_map, out_dir, dims=2):
    """
    Build interactive carousel per model:
      - dims=2 → Scatter; dims=3 → Scatter3d.
      - Dropdown recolors & masks as required.
    """
    SHOW_ALL = {"Checkmate vs. Non‑checkmate", "Visual similarity"}

    for tag, layers in aligned.items():
        logging.info(f"Building {dims}D carousel for '{tag}'")
        layer_names = list(layers.keys())
        n_lyr = len(layer_names)
        N = layers[layer_names[0]].shape[0]

        # ── NEW: compute global min / max across *all* layers ────────────────
        all_xyz = np.vstack(list(layers.values()))         # shape (n_layers*N, dims)
        lo = all_xyz.min(axis=0)
        hi = all_xyz.max(axis=0)
        pad = 0.05 * (hi - lo)                                # 5 % tolerance
        x_rng = [lo[0]-pad[0], hi[0]+pad[0]]
        y_rng = [lo[1]-pad[1], hi[1]+pad[1]]
        if dims == 3:
            z_rng = [lo[2]-pad[2], hi[2]+pad[2]]
        # --------------------------------------------------------------------

        full_op = [1]*N
        mask_op = [1]*20 + [0]*(N-20)

        # build one trace per layer
        traces = []
        for i, lyr in enumerate(layer_names):
            coords = layers[lyr]
            mk = dict(size=16,
                      color=color_map[labels[0]],
                      opacity=1,
                      line=dict(width=0.5, color="white"))
            common = dict(
                mode="markers",
                marker=mk,
                customdata=np.stack([attrs[l] for l in labels], axis=-1),
                hovertemplate="<br>".join(f"{l}: %{{customdata[{k}]}}" for k,l in enumerate(labels)) + "<extra></extra>",
                name=lyr,
                visible=(i==0)
            )
            if dims==2:
                trace = go.Scatter(x=coords[:,0], y=coords[:,1], **common)
            else:
                trace = go.Scatter3d(x=coords[:,0], y=coords[:,1], z=coords[:,2], **common)
            traces.append(trace)

        # slider steps
        steps = [dict(
            method="update",
            args=[{"visible": [j==i for j in range(n_lyr)]},
                  {"title": f"{tag} MDS ({dims}D) — {lyr}"}],
            label=lyr
        ) for i, lyr in enumerate(layer_names)]
        slider = dict(active=0, currentvalue={"prefix":"Layer: "}, pad={"t":50}, steps=steps)

        # dropdown to recolor & mask points
        buttons = []
        for lab in labels:
            # copy original per‑point colors
            cols = color_map[lab].copy()
            # if attribute is NOT one of the two “show‑all”, make last 20 fully transparent
            if lab not in {"Checkmate vs. Non‑checkmate", "Visual similarity"}:
                cols = cols[:20] + ["rgba(0,0,0,0)"] * (N - 20)  # hide last 20

            # restyle: apply same color list to every trace selected by indices
            buttons.append(dict(
                label=lab,
                method="restyle",
                args=[{"marker.color": [cols]}, list(range(n_lyr))]
            ))

        # assemble
        fig = go.Figure(data=traces)
        base = dict(paper_bgcolor="black",
                    font_color="white",
                    margin=dict(l=60,r=60,t=100,b=60),
                    sliders=[slider],
                    updatemenus=[dict(type="dropdown", x=0.02, y=1.15, buttons=buttons)])
        if dims==2:
            fig.update_layout(
                title=f"{tag} MDS (2D) — {layer_names[0]}",
                plot_bgcolor="black",
                xaxis=dict(title="Dim 1", gridcolor="gray", color="white", range=x_rng),
                yaxis=dict(title="Dim 2", gridcolor="gray", color="white", scaleanchor="x", range=y_rng),
                **base
            )
            fname = os.path.join(out_dir, f"{tag}_2d.html")
        else:
            fig.update_layout(
                title=f"{tag} MDS (3D) — {layer_names[0]}",
                scene=dict(
                    xaxis=dict(title="Dim 1", gridcolor="gray", color="white", backgroundcolor="black",range=x_rng),
                    yaxis=dict(title="Dim 2", gridcolor="gray", color="white", backgroundcolor="black",range=y_rng),
                    zaxis=dict(title="Dim 3", gridcolor="gray", color="white", backgroundcolor="black",range=z_rng),
                    bgcolor="black"
                ),
                **base
            )
            fname = os.path.join(out_dir, f"{tag}_3d.html")

        pio.write_html(fig, fname, auto_open=False, include_plotlyjs="cdn")
        logging.info(f"Wrote '{fname}'")


setup_logging()
run_id = create_run_id()
out_dir = f"results/{run_id}_mds_carousel"
os.makedirs(out_dir, exist_ok=True)

# config
EXCEL = "data/categories.xlsx"
ACT_DIR = "results/20250419-190918_extract-net-activations-alphavile_dataset-fmri"
SKIP = {"stim_id"}
MAPPING = {
    "check":          "Checkmate vs. Non‑checkmate",
    "motif":          "Motif category",
    "check-n":        "Moves to checkmate",
    "side":           "Side of king",
    "strategy":       "Strategic pattern",
    "visual":         "Visual similarity",
    "total_pieces":   "Total pieces",
    "legal_moves":    "Legal moves",
    "difficulty":     "Difficulty",
    "first_piece":    "First piece moved",
    "checkmate_piece":"Checkmate piece",
}
TAGS = ["trained","untrained"]

df_meta, attrs, labels = load_metadata(EXCEL, SKIP, MAPPING)
cmap_map = build_color_map(attrs, cmap_name="flare")
activations = load_activations(ACT_DIR, TAGS)
aligned2d = compute_aligned_mds(activations, df_meta, dims=2)
aligned3d = compute_aligned_mds(activations, df_meta, dims=3)
generate_carousel(aligned2d, attrs, labels, cmap_map, out_dir, dims=2)
generate_carousel(aligned3d, attrs, labels, cmap_map, out_dir, dims=3)

logging.info("Done.")
