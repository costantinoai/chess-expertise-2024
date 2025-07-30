#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script for representational manifold analysis on activations from the Alphavile DNN,
across multiple ROIs (e.g., network layers), saved from prior model inference.

Main functionality includes:
  - Loading activation data (pickled per ROI)
  - Reshaping and cleaning data for analysis
  - Computing manifold geometry and dimensionality metrics
  - Aggregating results and plotting them for visualization

This is intended for network interpretability, not human fMRI data.
"""

# ---------------------- IMPORTS ---------------------- #

import os  # for directory operations
import shutil  # to copy script for provenance
import logging  # for logging progress and issues
from datetime import datetime  # for timestamping outputs
import pickle  # to load activations from file

import numpy as np  # numerical operations
import pandas as pd  # dataframe operations
import matplotlib.pyplot as plt  # low-level plotting
import seaborn as sns  # high-level plotting aesthetics
from joblib import Parallel, delayed  # for parallel processing

# manifold-analysis core functions
from mftma.manifold_analysis_correlation import manifold_analysis_corr  # correlation-based manifold metric
from mftma.alldata_dimension_analysis import alldata_dimension_analysis  # dimensionality metrics

# ---------------------- CONFIGURATION ---------------------- #
base_font_size = 22
plt.rcParams.update({
    "font.family": 'Ubuntu Condensed',
    "font.size": base_font_size,
    "axes.titlesize": base_font_size * 1.4,  # 36.4 ~ 36
    "axes.labelsize": base_font_size * 1.2,  # 31.2 ~ 31
    "xtick.labelsize": base_font_size,  # 26
    "ytick.labelsize": base_font_size,  # 26
    "legend.fontsize": base_font_size,  # 26
    "figure.figsize": (12, 9),  # wide figures
})

OUTPUT_ROOT = "results"  # root output directory for all results
PROJECTION_DIM = 5000  # feature count threshold to apply dimensionality reduction
ALPHA_FDR = 0.05  # unused here, placeholder for future statistical corrections
N_JOBS = -2  # -1 means use all cores in parallel mode
ACTIVATION_PATH = "/home/eik-tb/OneDrive_andreaivan.costantino@kuleuven.be/GitHub/chess-expertise-2024/chess-dnn/results/20250419-190918_extract-net-activations-alphavile_dataset-fmri/activations_model-untrained_seed-0.pkl"
USE_PARALLEL = True  # toggle for parallel ROI processing

# ---------------------- LOGGING ---------------------- #

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)  # logger instance for consistent use

# ---------------------- UTILITIES ---------------------- #

def create_run_id():
    """
    Generate a timestamp string for uniquely identifying output folders.
    Returns:
        str: Timestamp in YYYYMMDD-HHMMSS format.
    """
    now = datetime.now()
    return now.strftime("%Y%m%d-%H%M%S")

def save_script_to_file(script_path: str, out_directory: str):
    """
    Copy this script to the results folder to ensure full provenance.
    Args:
        script_path (str): Path to this script (typically __file__)
        out_directory (str): Output directory where script copy is stored
    """
    os.makedirs(out_directory, exist_ok=True)  # create directory if it doesn't exist
    dest_path = os.path.join(out_directory, os.path.basename(script_path))
    shutil.copy(script_path, dest_path)  # copy script file
    logger.info("Copied script to '%s' for provenance", dest_path)

# ---------------------- MANIFOLD ANALYSIS ---------------------- #

def is_valid_layer(layer_name: str) -> bool:
    """
    Check if the layer name is a valid entry (not a malformed placeholder).
    """
    return "value" not in layer_name


def extract_and_flatten_activations(layer_data: dict) -> dict:
    """
    Extract and flatten activations for each stimulus in the layer.
    Returns:
        dict: {stim_id: flattened_activation}
    """
    activations = {}
    for stim_entry in layer_data.values():
        stim_id = int(stim_entry["stim_id"])
        flat_act = stim_entry["activation"].reshape(-1)
        activations[stim_id] = flat_act
    return activations


def clean_and_stack_activations(activations: dict) -> np.ndarray:
    """
    Stack activation vectors into a matrix and remove invalid entries.
    Returns:
        np.ndarray: Cleaned activation matrix (n_stimuli x n_features)
    """
    sorted_ids = sorted(activations.keys())
    matrix = np.stack([activations[i] for i in sorted_ids], axis=0)

    valid_rows = ~np.all(np.isnan(matrix), axis=1) & ~np.all(matrix == 0, axis=1)
    return matrix[valid_rows, :]


def reshape_for_analysis(data_matrix: np.ndarray) -> list:
    """
    Reshape data matrix into a list of column vectors (1 per sample).
    Returns:
        list of np.ndarray
    """
    return [data_matrix[i].reshape(-1, 1) for i in range(data_matrix.shape[0])]


def apply_random_projection(X: list, projection_dim: int) -> list:
    """
    Apply Gaussian random projection if feature dimensionality is high.
    """
    original_dim = X[0].shape[0]
    if original_dim <= projection_dim:
        return X

    logger.info("Applying random projection to reduce dimensionality from %d to %d", original_dim, projection_dim)
    M = np.random.randn(projection_dim, original_dim)
    M /= np.linalg.norm(M, axis=1, keepdims=True)
    return [M @ x for x in X]


def interleave_stimuli(X: list) -> list:
    """
    Combine corresponding pairs of first 20 and next 20 stimuli (assumes n=40).
    """
    if len(X) < 40:
        return []

    return [np.concatenate([X[i], X[i + 20]], axis=1) for i in range(20)]


def compute_manifold_metrics(X_grouped: list) -> dict:
    """
    Run manifold and dimensionality analyses and return metrics.
    Returns:
        dict of metrics
    """
    alpha, r, d, rho0, _ = manifold_analysis_corr(X_grouped, 0, 300, n_reps=1)

    X_ungrouped = [np.expand_dims(m[:, 0], 1) for m in X_grouped] + [np.expand_dims(m[:, 1], 1) for m in X_grouped]
    D_pr, D_ev, D_feat = alldata_dimension_analysis(X_ungrouped, perc=0.9)
    return {
        "alpha_M": 1.0 / np.mean(1.0 / alpha),
        "R_M": np.mean(r),
        "D_M": np.mean(d),
        "rho_center": rho0,
        "D_participation_ratio": D_pr,
        "D_explained_variance": D_ev,
        "D_feature": D_feat,
    }


def process_single_layer(model_name, layer_name, layer_data):
    """
    Orchestrates the manifold analysis for a single layer (ROI).
    Returns:
        tuple: (layer_name, metrics_dict or None)
    """
    if not is_valid_layer(layer_name):
        return layer_name, None

    logger.info("[%s] Starting manifold analysis for layer: %s", model_name, layer_name)

    try:
        # Step 1: Extract activations and clean
        activations = extract_and_flatten_activations(layer_data)
        data_matrix = clean_and_stack_activations(activations)

        if data_matrix.size == 0:
            logger.warning("[%s] Skipping %s: no valid activations", model_name, layer_name)
            return layer_name, None

        # Step 2: Reshape and optionally reduce dimensionality
        X = reshape_for_analysis(data_matrix)
        X = apply_random_projection(X, PROJECTION_DIM)

        # Step 3: Interleave stimuli and run analysis
        X_grouped = interleave_stimuli(X)
        if not X_grouped:
            logger.warning("[%s] Skipping %s: insufficient stimuli (<40)", model_name, layer_name)
            return layer_name, None

        metrics = compute_manifold_metrics(X_grouped)
        return layer_name, metrics

    except Exception as e:
        logger.error("[%s] Error processing %s: %s", model_name, layer_name, str(e))
        return layer_name, None

def plot_layer_metrics_grid(df, metrics, run_id, out_dir, n_cols=4, subplot_width=5, subplot_height=4):
    """
    Plot a grid of line plots for each metric across layers.

    Args:
        df (pd.DataFrame): DataFrame containing 'layer' and metric columns.
        metrics (list of str): Names of metrics to plot.
        run_id (str): Run identifier for naming the output file.
        out_dir (str): Directory to save the plot.
        n_cols (int): Number of columns in subplot grid.
        subplot_width (int): Width of each subplot in inches.
        subplot_height (int): Height of each subplot in inches.
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np
    import os

    # Compute grid size
    n_metrics = len(metrics)
    n_rows = int(np.ceil(n_metrics / n_cols))

    # Create subplots with adjusted width
    sns.set_style("white")
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(n_cols * subplot_width, n_rows * subplot_height),
        squeeze=False
    )
    axes = axes.flatten()

    # Define x-tick positions
    x_vals = np.arange(len(df))

    # Create each subplot
    for ax, metric in zip(axes, metrics):
        sns.lineplot(data=df, x="layer", y=metric, marker="o", ax=ax)
        ax.set_title(f"{metric} across layers")
        ax.set_xlabel("Layer")
        ax.set_ylabel(metric)
        ax.tick_params(axis="x")

        # Show only every 5th tick
        xticks = x_vals[::5]
        ax.set_xticks(xticks)
        ax.set_xticklabels(xticks, rotation=45, ha='right')

        sns.despine(ax=ax)

    # Remove unused axes
    for ax in axes[n_metrics:]:
        fig.delaxes(ax)

    # Adjust layout and save
    fig.tight_layout()
    plot_path = os.path.join(out_dir, f"{run_id}_plots.png")
    fig.savefig(plot_path)
    logger.info("Saved plots to '%s'", plot_path)

# ---------------------- EXECUTION PIPELINE ---------------------- #

# Create output directory for this run
run_id = f"{create_run_id()}_alphavile_manifold"
out_dir = os.path.join(OUTPUT_ROOT, run_id)
os.makedirs(out_dir, exist_ok=True)
logger.info("Output directory created: '%s'", out_dir)

# Save this script to output folder
save_script_to_file(__file__, out_dir)

# Load activation dictionary from pickle
with open(ACTIVATION_PATH, 'rb') as f:
    activations = pickle.load(f)  # dictionary: layer_name -> {stim_id -> activation}

# Dictionary to collect results
model_results = {}

# Choose processing mode: parallel or serial
if USE_PARALLEL:
    logger.info("Running in parallel mode")
    results = Parallel(n_jobs=N_JOBS, backend="loky")(
        delayed(process_single_layer)("alphavile", layer, data)
        for layer, data in activations.items()
    )
else:
    logger.info("Running in serial mode")
    results = [
        process_single_layer("alphavile", layer, data)
        for layer, data in activations.items()
    ]

# Store results in flat dictionary
for layer, metrics in results:
    if metrics is not None:
        model_results[layer] = metrics

logger.info("Completed manifold analysis on %d layers", len(model_results))

# Convert results to DataFrame
records = [
    {"layer": layer, **metrics} for layer, metrics in model_results.items()
]
df = pd.DataFrame(records)

# Save to CSV
csv_path = os.path.join(out_dir, f"{run_id}_metrics.csv")
df.to_csv(csv_path, index=False)
logger.info("Saved metrics to '%s'", csv_path)

# ---------------------- PLOTTING ---------------------- #

logger.info("Generating summary plots")
sns.set(style="whitegrid")

# Metrics to plot
plot_metrics = [
    "alpha_M", "R_M", "D_M", "rho_center",
    "D_participation_ratio", "D_explained_variance", "D_feature"
]

plot_layer_metrics_grid(
    df=df,
    metrics=plot_metrics,
    run_id=run_id,
    out_dir=out_dir,
    n_cols=4,
    subplot_width=5  # slightly wider than default
)
