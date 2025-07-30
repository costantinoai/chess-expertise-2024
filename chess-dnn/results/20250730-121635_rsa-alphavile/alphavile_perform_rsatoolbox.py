#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 29 16:19:01 2025

@author: costantino_ai
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RSA Pipeline for AlphazeroChess Activations
Author: costantino_ai
Date: 2025-04-19
"""

# ─── Imports ────────────────────────────────────────────────────────────────────
import os
import sys
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from matplotlib.patches import Patch
from scipy.spatial.distance import pdist, squareform
from scipy.stats import ttest_rel
from statsmodels.stats.multitest import fdrcorrection

import rsatoolbox
from rsatoolbox.rdm.rdms import RDMs, concat
from rsatoolbox.data.dataset import Dataset
from rsatoolbox.model import ModelFixed
from rsatoolbox.inference.evaluate import eval_bootstrap_pattern
from rsatoolbox.vis.rdm_plot import show_rdm
from rsatoolbox.vis.model_plot import plot_model_comparison

from sklearn.feature_selection import VarianceThreshold
# Base font size for glass brain plots and others
base_font_size = 22
plt.rcParams.update({
    "font.family": 'Ubuntu Condensed',
    "font.size": base_font_size,
    "axes.titlesize": base_font_size * 1.4,  # 36.4 ~ 36
    "axes.labelsize": base_font_size * 1.2,  # 31.2 ~ 31
    "xtick.labelsize": base_font_size,  # 26
    "ytick.labelsize": base_font_size,  # 26
    "legend.fontsize": base_font_size,  # 26
    "figure.figsize": (13, 7),  # wide figures
})
# ─── Local Utilities ────────────────────────────────────────────────────────────
sys.path.insert(0, "/home/eik-tb/OneDrive_andreaivan.costantino@kuleuven.be/GitHub/AlphazeroChess")
from modules.utils.helper_funcs import create_run_id, save_script_to_file, create_output_directory

# ─── Configuration ──────────────────────────────────────────────────────────────
FULL_REGS = {"check", "visual"}
SKIP_REGS = {"stim_id", "side", "difficulty", "first_piece", "checkmate_piece"}

regressor_mapping = {
    "check": "Checkmate", "stim_id": "All stimuli (pairwise)", "motif": "Motif category",
    "check-n": "Moves to checkmate", "side": "Side of king", "strategy": "Strategic pattern",
    "visual": "Visual similarity", "total_pieces": "Total pieces", "legal_moves": "Legal moves",
    "difficulty": "Difficulty", "first_piece": "First piece moved", "checkmate_piece": "Checkmate piece"
}

EXCEL_PATH = "data/categories.xlsx"
ACTIVATIONS_DIR = "results/20250419-190918_extract-net-activations-alphavile_dataset-fmri"
OUTPUT_DIR = f"results/{create_run_id()}_rsa-alphavile"
create_output_directory(OUTPUT_DIR)
save_script_to_file(OUTPUT_DIR)

COLORS = {"trained": "#3d5a80", "untrained": "#e07a5f", "pos": "#3d5a80", "neg": "#e07a5f", "ns": "#d3d3d3"}


def is_numeric(col):
    return col in {"check-n", "total_pieces", "legal_moves"}


def compute_rdm_from_column(values, method='default'):
    """Generate dissimilarity matrix from numeric or categorical values."""
    codes = values.astype(float).reshape(-1, 1) if method == "numeric" else pd.Categorical(values).codes.reshape(-1, 1)
    dist = pdist(codes, metric='cityblock' if method == "numeric" else 'hamming')
    return squareform(dist * len(codes) if method == "categorical" else dist)


def make_model_rdm(name, matrix):
    """Create an RSA toolbox RDM object."""
    vec = matrix[np.triu_indices_from(matrix, k=1)]
    return ModelFixed(name, RDMs(dissimilarities=np.expand_dims(vec, 0), dissimilarity_measure="custom", rdm_descriptors={"name": [name]}))


def compute_theoretical_rdms(df):
    df = df.set_index("stim_id").rename(columns={"filename": "trial_type"})
    full_df = df[["check", "strategy", "visual", "trial_type"]]
    half_df = df.loc[df.index <= 19, ["strategy", "check-n", "total_pieces", "legal_moves", "motif", "trial_type"]]

    def compute_rdms(dataframe):
        models = []
        for col in dataframe.columns.drop("trial_type"):
            values = dataframe[col].values
            method = "numeric" if is_numeric(col) else "categorical"
            dist_matrix = compute_rdm_from_column(values, method)
            model = make_model_rdm(regressor_mapping[col], dist_matrix)
            models.append(model)
        return models

    return {"full": compute_rdms(full_df), "half": compute_rdms(half_df)}


def compute_neural_rdms_per_layer(act_dict):
    full_rdms, half_rdms = [], []

    for layer, stim_acts in act_dict.items():
        X = np.vstack([v["activation"].ravel() for v in stim_acts.values()])
        X = VarianceThreshold(1e-5).fit_transform(X)

        data_full = Dataset(X, descriptors={"layer": layer})
        data_half = Dataset(X[:20], descriptors={"layer": layer})

        full_rdms.append(rsatoolbox.rdm.calc_rdm(data_full, method="correlation", remove_mean=True))
        half_rdms.append(rsatoolbox.rdm.calc_rdm(data_half, method="correlation", remove_mean=True))
    return {"full": concat(full_rdms), "half": concat(half_rdms)}


def run_rsa(models, rdms, tag, key):
    results = {"mean": [], "lo": [], "hi": [], "p": [], "boot": []}

    for idx, rdm in enumerate(rdms):
        # If I were to sample a new set of conditions, how much could this model's score vary?
        # --> these CIs reflects stimulus-level variability
        res = eval_bootstrap_pattern(models=models, data=rdm, method="rho-a", N=10000, boot_noise_ceil=True)
        results["mean"].append(res.get_means())
        results["lo"].append(res.get_ci(0.95)[0])
        results["hi"].append(res.get_ci(0.95)[1])
        results["p"].append(res.test_zero())
        results["boot"].append(res.evaluations)

        # fig, _, _ = plot_model_comparison(res, sort=True)
        # plt.title(f"{tag.upper()} | {key} | Layer {idx}")
        # plt.show()

    return results

def compare_trained_untrained(bt, bu):
    return [ttest_rel(bt[:, i], bu[:, i])[1] for i in range(bt.shape[1])]

def plot_rsa_scores(rsa_mean, rsa_ci_lo, rsa_ci_hi, key, model_names):
    for model_idx, model_name in enumerate(model_names):
        layers = sorted(rsa_mean["trained"][key].index)
        # plt.figure(figsize=(20, 8))
        plt.figure()

        for tag in ("trained", "untrained"):
            mean = np.array([rsa_mean[tag][key][l][model_idx] for l in layers])
            lo = np.array([rsa_ci_lo[tag][key][l][model_idx] for l in layers])
            hi = np.array([rsa_ci_hi[tag][key][l][model_idx] for l in layers])
            ci = (hi - lo) / 2

            plt.plot(layers, mean, label=tag.capitalize(), color=COLORS[tag], marker="o", linewidth=2)
            plt.fill_between(layers, mean - ci, mean + ci, color=COLORS[tag], alpha=0.3)

        title = regressor_mapping.get(model_name, model_name)
        subtitle = " (Checkmate stimuli)" if key == "half" else ""
        plt.title(f"Spearman RSA: {title}{subtitle}")
        plt.axhline(0, linestyle='--', color='gray')
        plt.xlabel("Layer"); plt.ylabel("Spearman r ± 95% CI")
        plt.xticks(rotation=45)
        plt.ylim(-1, 1)
        plt.legend(); plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, f"{model_name}_rsa_corr.png"), dpi=300)
        plt.show()

def plot_delta_rsa(
    rsa_mean, diff_pvals, model_name, key, output_dir, regressor_mapping=None, full_regs=None, colors=None
):
    """
    Plot ΔRSA (trained − untrained) per layer for a given model (regressor).

    Parameters:
        rsa_mean         : dict – RSA means for "trained" and "untrained" models
        diff_pvals       : dict – FDR-corrected p-values for each regressor and layer
        model_name       : str  – Name of the regressor to plot
        key              : str  – Either "full" or "half", identifying the RDM set
        output_dir       : str  – Directory to save plots
        regressor_mapping: dict – Optional: human-readable label mapping
        full_regs        : set  – Optional: full RDMs set to control subtitles
        colors           : dict – Optional: dictionary of colors for 'pos', 'neg', 'ns'
    """
    if regressor_mapping is None:
        regressor_mapping = {}
    if full_regs is None:
        full_regs = set()
    if colors is None:
        colors = {"pos": "#3d5a80", "neg": "#e07a5f", "ns": "#d3d3d3"}

    layers = sorted(rsa_mean["trained"][key].index)
    model_idx = rsa_mean["model_names"][key].index(model_name)

    # Extract RSA scores
    tr_mean = rsa_mean["trained"][key].apply(lambda x: x[model_idx])
    un_mean = rsa_mean["untrained"][key].apply(lambda x: x[model_idx])
    delta   = tr_mean.values - un_mean.values
    pvals   = diff_pvals[model_name].reindex(layers).fillna(1.0).values

    # Determine bar colors
    bar_colors = [
        colors["ns"] if p >= 0.05 else (colors["pos"] if d > 0 else colors["neg"])
        for d, p in zip(delta, pvals)
    ]

    # Plot
    # fig, ax = plt.subplots(figsize=(20, 8))
    fig, ax = plt.subplots(figsize=(20, 8))
    x = np.arange(len(layers))
    ax.bar(x, delta, color=bar_colors, edgecolor='black', linewidth=0.5)
    ax.axhline(0, linestyle='--', color='gray', linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels(layers, rotation=45, ha='right')

    # Labels and title
    title = regressor_mapping.get(model_name, model_name)
    subtitle = " (Checkmate stimuli)" if model_name not in full_regs else ""
    ax.set_title(f"ΔRSA (Trained − Untrained): {title}{subtitle}", pad=45)
    ax.set_xlabel("Layer")
    ax.set_ylabel("ΔSpearman r")

    # Legend
    legend_items = [
        Patch(facecolor=colors["pos"], label="Trained > Untrained (p < .05)"),
        Patch(facecolor=colors["neg"], label="Untrained > Trained (p < .05)"),
        Patch(facecolor=colors["ns"], label="Non‑significant")
    ]
    ax.legend(handles=legend_items, loc='upper center', bbox_to_anchor=(0.5, 1.12), ncol=3, frameon=False)

    sns.despine()
    plt.ylim(-0.1, 0.2)
    plt.tight_layout()

    fname = f"{model_name.replace('/', '-')}_rsa_diff.png"
    fig.savefig(os.path.join(output_dir, fname), dpi=300)
    plt.show()


# ─── 1. Load Metadata ───────────────────────────────────────────────────────────

# Load metadata describing each of the 40 chess stimuli
df_meta = pd.read_excel(EXCEL_PATH)

# Safety check to ensure metadata contains all expected stimuli
assert df_meta.shape[0] == 40, f"Expected 40 stimuli, got {df_meta.shape[0]}"

# Compute the theoretical RDMs based on known stimulus features
theoretical_rdms = compute_theoretical_rdms(df_meta)

# ─── 2. Load Model Activations ─────────────────────────────────────────────────

# Dictionary to hold activations for both trained and untrained models
activations = {}
for tag in ("trained", "untrained"):
    fname = f"activations_model-{tag}_seed-0.pkl"
    with open(os.path.join(ACTIVATIONS_DIR, fname), "rb") as f:
        activations[tag] = pickle.load(f)

# ─── 3. RSA Containers Setup ───────────────────────────────────────────────────

# Create empty containers for RSA results
rsa_mean, rsa_ci_lo, rsa_ci_hi, rsa_p, rsa_boot = {}, {}, {}, {}, {}

# Define a fixed mapping between result keys and the container variables above
rsa_containers = {
    "mean": rsa_mean,
    "lo": rsa_ci_lo,
    "hi": rsa_ci_hi,
    "p": rsa_p,
    "boot": rsa_boot
}

# ─── 4. RSA Evaluation ─────────────────────────────────────────────────────────

# Run RSA analysis separately for each model type (trained vs untrained)
for tag in ("trained", "untrained"):
    # Initialize nested dicts: e.g., rsa_mean["trained"] = {}
    for container in rsa_containers.values():
        container[tag] = {}

    # Compute neural RDMs from the activations of this model
    rdms = compute_neural_rdms_per_layer(activations[tag])

    # Evaluate RSA for each type of theoretical RDM set: full or half (checkmate)
    for key, models in theoretical_rdms.items():
        # Run RSA for each layer against all theoretical models (regressors)
        res = run_rsa(models=models, rdms=rdms[key], tag=tag, key=key)

        # Store the results into the appropriate containers
        for k, container in rsa_containers.items():
            container[tag][key] = pd.Series(res[k], index=range(len(res[k])))

        # Save regressor names for bookkeeping (shared across models)
        rsa_mean.setdefault("model_names", {})[key] = [m.name for m in models]

# ─── 5. Plot RSA Scores Per Layer ──────────────────────────────────────────────

# Generate line plots (±95% CI) of Spearman RSA across network layers
for key, model_names in rsa_mean["model_names"].items():
    plot_rsa_scores(rsa_mean, rsa_ci_lo, rsa_ci_hi, key, model_names)

# ─── 6. Statistical Comparison (Trained vs Untrained) ──────────────────────────

from scipy.stats import permutation_test

# This will hold the FDR-corrected p-values for each regressor and layer
diff_pvals = {}

# Loop over each theoretical model group ("full" or "half")
for key, model_names in rsa_mean["model_names"].items():
    for model_idx, model_name in enumerate(model_names):
        layers = sorted(rsa_boot["trained"][key].index)
        pvals = []

        for l in layers:
            boots_tr = rsa_boot["trained"][key][l][:, model_idx]
            boots_un = rsa_boot["untrained"][key][l][:, model_idx]

            # Paired permutation test on bootstrapped samples
            # The test of difference uses paired differences, **not** group CIs
            # Why overlapping CIs ≠ no significance
            # Imagine the two groups always move up and down together across bootstraps — but one is consistently higher than the other.
            # In this case:
            #     Each group’s CI is wide (they wiggle up and down), and they overlap.
            #     But their paired difference is stable → small variance → significant.
            # It’s not about how wide the bars are — it’s about how tight the differences are.
            # so below we are comparing paired scores on the same sampled patterns

            res = permutation_test(
                (boots_tr, boots_un),
                statistic=lambda x, y, axis: np.mean(x - y, axis=axis),
                permutation_type='samples',
                vectorized=True,
                n_resamples=10000,
                alternative='two-sided'
            )

            pvals.append(res.pvalue)

            # plt.hist(boots_tr - boots_un, bins=50)
            # plt.axvline(np.mean(boots_tr - boots_un), color='red', linestyle='--')
            # plt.title("Bootstrap distribution of Δ(RSA Trained − Untrained)")

        # Apply FDR correction across layers for this model
        _, pvals_fdr = fdrcorrection(pvals, alpha=0.05)
        diff_pvals[model_name] = pd.Series(pvals_fdr, index=layers)


# ─── 7. Plot ΔRSA Bar Charts ───────────────────────────────────────────────────

# For each regressor, plot the layerwise difference between trained and untrained
for key, model_names in rsa_mean["model_names"].items():
    for model_name in model_names:
        plot_delta_rsa(
            rsa_mean=rsa_mean,
            diff_pvals=diff_pvals,
            model_name=model_name,
            key=key,
            output_dir=OUTPUT_DIR,
            regressor_mapping=regressor_mapping,
            full_regs=FULL_REGS,
            colors=COLORS
        )
