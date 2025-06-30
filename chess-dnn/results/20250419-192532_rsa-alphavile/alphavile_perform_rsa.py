#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 19 19:09:55 2025
@author: costantino_ai

Load saved activations and perform correlation‑based RSA (including ΔRSA plots).
This version:
- Builds only the required theoretical RDMs:
    • Full 40×40 for: check, visual, strategy
    • Truncated 20×20 for all other regressors (by slicing first 20×20)
- Executes one‑regressor RSA per layer, automatically choosing the matching neural RDM.
- Computes FDR‑corrected ΔRSA p‑values (trained vs. untrained).
- Plots each regressor separately, with a wider default figsize.
"""

import os
import pickle

import numpy as np
import pandas as pd
from natsort import natsorted

import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from scipy.spatial.distance import pdist, squareform
from scipy.stats import spearmanr, t, ttest_ind, t as t_dist

from sklearn.model_selection import KFold
from statsmodels.stats.multitest import multipletests

# utils for saving outputs
import sys
sys.path.insert(0, "/home/eik-tb/OneDrive_andreaivan.costantino@kuleuven.be/GitHub/AlphazeroChess")
from modules.utils.helper_funcs import create_run_id, save_script_to_file, create_output_directory

# ─── Configuration ──────────────────────────────────────────────────────────────

# Which regressors get full 40×40 vs. truncated 20×20 RDMs:
FULL_REGS  = {"check", "visual", "strategy"}
SKIP_REGS  = {"stim_id"}  # regressors to ignore entirely
# All other regressors will use only the first 20×20 of their full RDM.

# Human‑readable names
regressor_mapping = {
    "check":            "Checkmate vs. Non‑checkmate",
    "stim_id":          "All stimuli (pairwise)",
    "motif":            "Motif category",
    "check-n":          "Moves to checkmate",
    "side":             "Side of king",
    "strategy":         "Strategic pattern",
    "visual":           "Visual similarity",
    "total_pieces":     "Total pieces",
    "legal_moves":      "Legal moves",
    "difficulty":       "Difficulty",
    "first_piece":      "First piece moved",
    "checkmate_piece":  "Checkmate piece",
}

# Paths: update ACTIVATIONS_DIR to match your extraction output
EXCEL_PATH      = "data/categories.xlsx"
ACTIVATIONS_DIR = "results/20250419-190918_extract-net-activations-alphavile_dataset-fmri"
OUTPUT_DIR      = f"results/{create_run_id()}_rsa-alphavile"
create_output_directory(OUTPUT_DIR)
save_script_to_file(OUTPUT_DIR)

# Plot colors
COLORS = {
    "trained":   "#3d5a80",
    "untrained": "#e07a5f",
    "pos":       "#3d5a80",
    "neg":       "#e07a5f",
    "ns":        "#d3d3d3"
}

# ─── Utility functions ─────────────────────────────────────────────────────────

def extract_lower_triangular(mat: np.ndarray) -> np.ndarray:
    """Return 1D array of the lower‑triangle (k=–1 excludes diagonal)."""
    return mat[np.tril_indices_from(mat, k=-1)]

def compute_theoretical_rdms(df: pd.DataFrame, regs: list) -> dict:
    """
    Build theoretical RDMs per regressor:
      - Full 40×40 for regs in FULL_REGS
      - Truncated 20×20 for regs not in FULL_REGS ∪ SKIP_REGS
    Returns mapping { reg_name: RDM matrix }.
    """
    rdms = {}
    for col in regs:
        if col in SKIP_REGS:
            continue  # skip entirely
        vals = df[col].values
        # handle missing
        valid = ~pd.isna(vals)
        vals = vals[valid]
        # numeric vs. categorical
        if col in {"check-n", "total_pieces", "legal_moves"}:
            # numeric: absolute difference
            numeric = vals.astype(float)
            full = np.abs(numeric[:,None] - numeric[None,:])
        else:
            # categorical: 0 if same, 1 if different
            codes = pd.Categorical(vals).codes
            full = (codes[:,None] != codes[None,:]).astype(float)

        if col in FULL_REGS:
            rdms[col] = full  # keep full 40×40
        else:
            # truncate to 20 stimuli
            rdms[col] = full[:20, :20]
    return rdms

def run_cv_spearman(X: np.ndarray,
                    y: np.ndarray,
                    n_splits: int = 5,
                    random_state: int = 42,
                    return_folds: bool = False):
    """
    Cross‑validated Spearman RSA for a single regressor:
    - X: [n_pairs, 1] model RDM vector
    - y: [n_pairs] neural RDM vector
    Returns mean r, 95% CI lower, CI upper, p‑value, and optional folds.
    """
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    fold_rs = []
    for train_idx, _ in kf.split(X):
        r, _ = spearmanr(X[train_idx, 0], y[train_idx])
        fold_rs.append(r)
    fold_rs = np.array(fold_rs)  # shape [n_folds]

    mean_r = fold_rs.mean()
    sem_r  = fold_rs.std(ddof=1) / np.sqrt(len(fold_rs))
    df     = len(fold_rs) - 1
    tcrit  = t_dist.ppf(0.975, df=df)
    lower  = mean_r - tcrit * sem_r
    upper  = mean_r + tcrit * sem_r
    # test vs zero
    _, pval = ttest_ind(fold_rs, np.zeros_like(fold_rs))

    if return_folds:
        return mean_r, lower, upper, pval, fold_rs
    return mean_r, lower, upper, pval

def compute_neural_rdms_per_layer(act_dict: dict) -> tuple:
    """
    Given activations per layer (dict[layer] = {stim_id:{"activation":...}}):
    - Returns:
      neural_full: dict[layer] → lower_tri(full 40×40 RDM)
      neural_top20: dict[layer] → lower_tri(top 20×20 RDM)
    """
    neural_full, neural_top20 = {}, {}
    for layer, entries in act_dict.items():
        # stack all 40 stimuli
        X = np.vstack([v["activation"].ravel() for v in entries.values()])
        # full cosine RDM
        full = squareform(pdist(X, metric="cosine"))
        neural_full[layer] = extract_lower_triangular(full)
        # truncated 20×20
        top20 = full[:20, :20]
        neural_top20[layer] = extract_lower_triangular(top20)
    return neural_full, neural_top20

# ─── Main execution ─────────────────────────────────────────────────────────────

def main():
    # 1) Load metadata and build theoretical RDMs
    df_meta = pd.read_excel(EXCEL_PATH)
    if df_meta.shape[0] != 40:
        raise ValueError(f"Expected 40 stimuli, found {df_meta.shape[0]}")
    regressors = list(regressor_mapping.keys())
    theo_rdms   = compute_theoretical_rdms(df_meta, regressors)

    # 2) Load activations for both models
    activations = {}
    for tag in ("trained", "untrained"):
        path = os.path.join(ACTIVATIONS_DIR, f"activations_model-{tag}_seed-0.pkl")
        with open(path, "rb") as f:
            activations[tag] = pickle.load(f)

    # 3) Precompute neural RDMs per layer
    neural_rdms = {}
    for tag in ("trained", "untrained"):
        nf, nt = compute_neural_rdms_per_layer(activations[tag])
        neural_rdms[tag] = {"full": nf, "top20": nt}

    # 4) Run RSA regressor‑by‑regressor, layer‑by‑layer
    rsa_mean, rsa_ci, rsa_p    = {}, {}, {}
    rsa_folds                   = {"trained": {}, "untrained": {}}

    for tag in ("trained", "untrained"):
        rsa_mean[tag], rsa_ci[tag], rsa_p[tag] = {}, {}, {}
        for reg, rdm in theo_rdms.items():
            # choose which neural RDM to match: full vs top20
            if reg in FULL_REGS:
                key = "full"
            else:
                key = "top20"
            # prepare model vector (shape [n_pairs,1])
            model_vec = extract_lower_triangular(rdm)[:, None]
            # run layer‑wise RSA
            m_list, lo_list, hi_list, p_list, folds_list = [], [], [], [], []
            for layer in sorted(neural_rdms[tag][key].keys(), key=lambda x: x):
                y_vec = neural_rdms[tag][key][layer]
                m, lo, hi, pval, folds = run_cv_spearman(model_vec, y_vec, return_folds=True)
                m_list.append(m)
                lo_list.append(lo)
                hi_list.append(hi)
                p_list.append(pval)
                folds_list.append(folds)
            # aggregate into arrays/DataFrames
            layers = sorted(neural_rdms[tag][key].keys(), key=lambda x: x)
            rsa_mean[tag][reg]  = pd.Series(m_list, index=layers)
            rsa_ci[tag][reg]    = pd.Series((np.array(hi_list) - np.array(lo_list)) / 2, index=layers)
            rsa_p[tag][reg]     = pd.Series(p_list, index=layers)
            rsa_folds[tag][reg] = dict(zip(layers, folds_list))

    # 5) Compute ΔRSA p‑values (trained vs untrained) with FDR correction
    diff_pvals = {}
    for reg in theo_rdms:
        tr_f, un_f = rsa_folds["trained"][reg], rsa_folds["untrained"][reg]
        raw_p, layers = [], []
        for layer, arr_tr in tr_f.items():
            arr_un = un_f.get(layer)
            if arr_un is None:
                continue
            _, p = ttest_ind(arr_tr, arr_un, equal_var=False)
            raw_p.append(p)
            layers.append(layer)
        # FDR correct across layers for this regressor
        _, p_corr, _, _ = multipletests(raw_p, alpha=0.05, method="fdr_bh")
        diff_pvals[reg] = pd.Series(p_corr, index=layers)

    # 6) Plot Spearman RSA trajectories
    for reg in theo_rdms:
        # pick correct layer set
        key = "full" if reg in FULL_REGS else "top20"
        layers = sorted(neural_rdms["trained"][key].keys(), key=lambda x: x)
        plt.figure(figsize=(20, 8))
        for tag in ("trained", "untrained"):
            mean_s = rsa_mean[tag][reg].reindex(layers)
            ci_s   = rsa_ci[tag][reg].reindex(layers)
            plt.plot(layers, mean_s, label=tag.capitalize(),
                     color=COLORS[tag], marker='o', linewidth=2)
            plt.fill_between(layers,
                             mean_s - ci_s,
                             mean_s + ci_s,
                             color=COLORS[tag], alpha=0.3)
        # title & labels
        title = regressor_mapping.get(reg, reg)
        plt.title(
            f"Spearman RSA: {title}{' (Checkmate stimuli)' if reg not in FULL_REGS else ''}",
            pad=16
        )
        plt.xlabel("Layer"); plt.ylabel("Spearman r ± 95% CI")
        plt.xticks(rotation=45, ha='right')
        plt.axhline(0, linestyle='--', color='gray', linewidth=1)
        plt.grid(axis='y', linestyle='--', alpha=0.5)
        plt.legend(loc='best')
        plt.ylim(-0.2, 0.6); plt.tight_layout()
        fname = f"{reg.replace('/','-')}_rsa_corr.png"
        plt.savefig(os.path.join(OUTPUT_DIR, fname), dpi=300)
        plt.show()

    # 7) Plot ΔRSA barplots
    for reg, pseries in diff_pvals.items():
        key = "full" if reg in FULL_REGS else "top20"
        layers = sorted(neural_rdms["trained"][key].keys(), key=lambda x: x)
        tr_mean = rsa_mean["trained"][reg].reindex(layers)
        un_mean = rsa_mean["untrained"][reg].reindex(layers)
        delta   = tr_mean.values - un_mean.values
        pvals   = pseries.reindex(layers).fillna(1.0).values

        colors = [
            COLORS["ns"] if p >= 0.05 else (COLORS["pos"] if d > 0 else COLORS["neg"])
            for d, p in zip(delta, pvals)
        ]

        fig, ax = plt.subplots(figsize=(20, 8))
        x = np.arange(len(layers))
        ax.bar(x, delta, color=colors, edgecolor='black', linewidth=0.5)
        ax.axhline(0, linestyle='--', color='gray', linewidth=1)
        ax.set_xticks(x); ax.set_xticklabels(layers, rotation=45, ha='right')
        title = regressor_mapping.get(reg, reg)
        ax.set_title(
            f"ΔRSA (Trained − Untrained): {title}{' (Checkmate stimuli)' if reg not in FULL_REGS else ''}",
            pad=16
        )

        ax.set_xlabel("Layer"); ax.set_ylabel("ΔSpearman r")

        legend_items = [
            Patch(facecolor=COLORS["pos"], label="Trained > Untrained (p < .05)"),
            Patch(facecolor=COLORS["neg"], label="Untrained > Trained (p < .05)"),
            Patch(facecolor=COLORS["ns"],  label="Non‑significant")
        ]
        ax.legend(handles=legend_items, loc='upper center', bbox_to_anchor=(0.5, 1.12), ncol=3, frameon=False)
        sns.despine(fig=fig); plt.ylim(-0.25, 0.6); plt.tight_layout()
        fname = f"{reg.replace('/','-')}_rsa_diff.png"
        fig.savefig(os.path.join(OUTPUT_DIR, fname), dpi=300)
        plt.show()

if __name__ == "__main__":
    main()
