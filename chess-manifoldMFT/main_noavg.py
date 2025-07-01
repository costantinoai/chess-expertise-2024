#!/usr/bin/env python3
"""Run ROI-wise manifold analysis comparing experts and novices."""

import os
from typing import Tuple

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from modules import logger, EXPERTS, NONEXPERTS
from modules.io_utils import load_all_betas, load_atlas
from modules.analysis_utils import compute_manifold, fdr_ttest
from modules.plot_utils import (
    plot_group_comparison,
    plot_group_heatmap,
)

# from sklearn.preprocessing import StandardScaler

AVG_RUNS = True
PARALLEL=True

def process_subject(
    subject: str,
    atlas_data: np.ndarray,
    rois: np.ndarray,
    out_dir: str,
    roi_n_jobs: int = 1
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute manifold metrics for each ROI for a single subject."""
    logger.info(f"Processing subject: {subject}")
    betas_dict, labels = load_all_betas(subject, avg_runs=AVG_RUNS)
    logger.info(f"Loaded betas for subject {subject}: conditions={labels}")

    # ROI metrics arrays
    def roi_metrics(roi: int):
        logger.info(f"Subject {subject}: Processing ROI {roi}")
        mask = atlas_data == roi
        manifolds = []

        for cond in labels:
            logger.info(f"Subject {subject}, ROI {roi}: Processing condition {cond}")
            # extract runs × voxels within this ROI
            if type(betas_dict[cond]) == list:
                condition_vectors = np.stack([beta_img[mask] for beta_img in betas_dict[cond]], axis=0)  # shape = (n_runs, n_voxels)
            else:
                # add a dimension for runs
                condition_vectors = betas_dict[cond][mask][np.newaxis, :]  #

            # identify columns (voxels) to keep: no NaNs and variance > 0
            has_nan = np.isnan(condition_vectors).any(axis=0)
            valid_cols = ~has_nan

            logger.info(f"Subject {subject}, ROI {roi}, Condition {cond}: roi_runs shape {condition_vectors.shape} --> valid {np.sum(valid_cols)}")

            # filter out bad voxels
            filtered = condition_vectors[:, valid_cols]   # now (n_runs, n_valid_voxels)

            # if type(betas_dict[cond]) == list:
            #     # scale: observations are rows, features are columns
            #     scaler = StandardScaler()
            #     filtered = scaler.fit_transform(filtered)  # (n_runs, n_valid_voxels)

            # compute_manifold expects features × observations
            manifolds.append(filtered.T)

        logger.info(f"Subject {subject}, ROI {roi}: Computing manifold metrics")
        manifold_results = compute_manifold(manifolds=manifolds)

        logger.info(f"Subject {subject}, ROI {roi}: manifold analysis completed")
        return manifold_results

    if roi_n_jobs == 1:
        logger.info(f"Subject {subject}: Processing ROIs sequentially")
        results = {
            roi: roi_metrics(roi)
            for roi in rois
        }
    else:
        logger.info(f"Subject {subject}: Processing ROIs in parallel (n_jobs={roi_n_jobs})")
        parallel_rois_results = Parallel(n_jobs=roi_n_jobs)(
            delayed(roi_metrics)(roi) for roi in rois
        )
        results = dict(zip(rois, parallel_rois_results))

    logger.info(f"Subject {subject}: Finished processing all ROIs")
    return results


def run_group(
    subs,
    atlas_data,
    rois,
    out_dir,
    subj_n_jobs: int = -1,
    roi_n_jobs: int = 1
):
    """Process a group of subjects using joblib.Parallel if ``subj_n_jobs`` > 1."""
    logger.info(f"Running group analysis for {len(subs)} subjects")
    def _run(s):
        return process_subject(s, atlas_data, rois, out_dir, roi_n_jobs)

    if subj_n_jobs == 1:
        logger.info("Processing subjects sequentially")
        results = {
            s: process_subject(s, atlas_data, rois, out_dir, roi_n_jobs)
            for s in subs
        }
    else:
        logger.info(f"Processing subjects in parallel (n_jobs={subj_n_jobs})")
        parallel_subs_results = Parallel(n_jobs=subj_n_jobs)(
            delayed(process_subject)(s, atlas_data, rois, out_dir, roi_n_jobs)
            for s in subs
        )
        results = dict(zip(subs, parallel_subs_results))

    logger.info("Finished group processing")
    return results

logger.info("Loading atlas")
atlas_data, rois = load_atlas()
rois = [1]
out_dir = "manifold_results"
os.makedirs(out_dir, exist_ok=True)
logger.info(f"Output directory: {out_dir}")

# fully parallel across ROIs and subjects by default:
logger.info("Processing experts group")
exp_dict = run_group(
    EXPERTS, atlas_data, rois, out_dir,
    subj_n_jobs=-1 if PARALLEL else 1, roi_n_jobs=1
)
logger.info("Processing non-experts group")
nov_dict = run_group(
    NONEXPERTS, atlas_data, rois, out_dir,
    subj_n_jobs=-1 if PARALLEL else 1, roi_n_jobs=1
)

def flatten_metrics(data_dict, label):
    return [
        {"subject": subj, "roi": roi, **metrics, "expertise": label}
        for subj, roi_dict in data_dict.items()
        for roi, metrics in roi_dict.items()
    ]

# Flatten and merge
full_df = pd.DataFrame(flatten_metrics(exp_dict, "Expert") + flatten_metrics(nov_dict, "Novice"))

# # run FDR‐corrected t-tests on the metric arrays:
# logger.info("Running FDR-corrected t-tests for capacity")
# stats_c = fdr_ttest(exp_c, non_c, rois)
# logger.info("Running FDR-corrected t-tests for radius")
# stats_r = fdr_ttest(exp_r, non_r, rois)
# logger.info("Running FDR-corrected t-tests for dimension")
# stats_d = fdr_ttest(exp_d, non_d, rois)

# # build and save group means & stats
# logger.info("Building group means DataFrame")
# mean_df = pd.DataFrame({
#     "ROI": rois,
#     "capacity_expert": np.nanmean(exp_c, axis=0),
#     "capacity_nonexpert": np.nanmean(non_c, axis=0),
#     "radius_expert": np.nanmean(exp_r, axis=0),
#     "radius_nonexpert": np.nanmean(non_r, axis=0),
#     "dimension_expert": np.nanmean(exp_d, axis=0),
#     "dimension_nonexpert": np.nanmean(non_d, axis=0),
# })
# mean_path = os.path.join(out_dir, "group_means.csv")
# logger.info(f"Saving group means to {mean_path}")
# mean_df.to_csv(mean_path, index=False)
# stats_c_path = os.path.join(out_dir, "stats_capacity.csv")
# stats_r_path = os.path.join(out_dir, "stats_radius.csv")
# stats_d_path = os.path.join(out_dir, "stats_dimension.csv")
# logger.info(f"Saving stats to {stats_c_path}, {stats_r_path}, {stats_d_path}")
# stats_c.to_csv(stats_c_path, index=False)
# stats_r.to_csv(stats_r_path, index=False)
# stats_d.to_csv(stats_d_path, index=False)

# # plotting
# for metric in ("capacity", "radius", "dimension"):
#     logger.info(f"Plotting group comparison for {metric}")
#     df_long = mean_df.melt(
#         id_vars="ROI",
#         value_vars=[f"{metric}_expert", f"{metric}_nonexpert"],
#         var_name="group",
#         value_name=metric,
#     )
#     df_long["group"] = df_long["group"].str.contains("expert").map({True: "expert", False: "novice"})
#     plot_group_comparison(df_long, out_dir, metric)

#     logger.info(f"Plotting group heatmap for {metric}")
#     plot_group_heatmap(
#         mean_df[[f"{metric}_expert", f"{metric}_nonexpert"]].values,
#         rois, metric, out_dir
#     )

# logger.info("Analysis complete. Results in %s", out_dir)


import seaborn as sns
import matplotlib.pyplot as plt

# 8) Plotting: line plots for all manifold metrics by ROI, with hue for expertise (True/False)
logger.info("Generating line plots for all manifold metrics by expertise")
sns.set(style="whitegrid")

# List all metrics to plot
metrics_to_plot = [
    "alpha_M",
    "R_M",
    "D_M",
    "rho_center",
    "D_participation_ratio",
    "D_explained_variance",
    "D_feature",
]

# Determine subplot grid size (4 columns, rows as needed)
n_metrics = len(metrics_to_plot)
n_cols = 4
n_rows = int(np.ceil(n_metrics / n_cols))

# Create figure and axes
fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 4, n_rows * 4), sharey=False)
axes = axes.flatten()  # flatten to 1D for easy iteration

# Loop over each metric and corresponding axis
for ax, metric in zip(axes, metrics_to_plot):
    # Draw lineplot: ROI on x-axis, metric on y-axis, hue by expertise
    sns.lineplot(
        data=full_df,
        x="roi",
        y=metric,
        hue="expertise",         # separate lines for expert (True) vs non-expert (False)
        ci=95,             # 95% confidence interval around the mean
        marker="o",        # circle markers at each ROI
        err_style="band",  # shaded error bands
        ax=ax
    )
    # Title and axis labels
    ax.set_title(f"{metric} by ROI and Expertise")
    ax.set_xlabel("ROI")
    ax.set_ylabel(metric)
    # Rotate x-tick labels for readability
    ax.tick_params(axis="x", rotation=90)
    # Move legend inside subplot
    ax.legend(title="Expertise", loc="upper right", fontsize="small", title_fontsize="small")

# Remove any unused subplots if metrics < n_rows*n_cols
for unused_ax in axes[n_metrics:]:
    fig.delaxes(unused_ax)

# Adjust layout and save
fig.tight_layout()
# lineplots_path = os.path.join(out_dir, f"{run_id}_manifold_lineplots.png")
# fig.savefig(lineplots_path)
# logger.info("Saved line plots to '%s'", lineplots_path)
plt.show()
