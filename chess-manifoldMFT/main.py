#!/usr/bin/env python3
"""Run ROI-wise manifold analysis comparing experts and novices."""

from __future__ import annotations

import os
from typing import Tuple

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from modules import logger, EXPERTS, NONEXPERTS
from modules.io_utils import load_all_betas, load_atlas
from modules.analysis_utils import compute_manifold, fdr_ttest
from modules.plot_utils import (
    plot_subject_roi,
    plot_group_comparison,
    plot_group_heatmap,
)


def process_subject(
    subject: str,
    atlas_data: np.ndarray,
    rois: np.ndarray,
    out_dir: str,
    roi_n_jobs: int = -1
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute manifold metrics for each ROI for a single subject."""
    betas_dict, labels = load_all_betas(subject)

    # ROI metrics arrays
    capacities = np.zeros(len(rois))
    radii      = np.zeros(len(rois))
    dims       = np.zeros(len(rois))

    def roi_metrics(roi: int):
        mask = atlas_data == roi
        manifolds = []
        for cond in labels:
            roi_runs = betas_dict[cond][:, mask]
            # standard scale each run within the ROI
            mean = roi_runs.mean(axis=1, keepdims=True)
            std = roi_runs.std(axis=1, keepdims=True)
            std[std == 0] = 1.0
            scaled = (roi_runs - mean) / std
            manifolds.append(scaled.T)
        c, r, d = compute_manifold(manifolds=manifolds)
        plot_subject_roi(d, r, subject, roi, os.path.join(out_dir, "subject"))
        return c, r, d

    if roi_n_jobs == 1:
        results = [roi_metrics(roi) for roi in rois]
    else:
        results = Parallel(n_jobs=roi_n_jobs)(
            delayed(roi_metrics)(roi) for roi in rois
        )

    for i, (c, r, d) in enumerate(results):
        capacities[i] = c
        radii[i]      = r
        dims[i]       = d

    return capacities, radii, dims


def run_group(
    subs,
    atlas_data,
    rois,
    out_dir,
    subj_n_jobs: int = 1,
    roi_n_jobs: int = -1
):
    """Process a group of subjects using joblib.Parallel if ``subj_n_jobs`` > 1."""
    def _run(s):
        return process_subject(s, atlas_data, rois, out_dir, roi_n_jobs)

    if subj_n_jobs == 1:
        results = [_run(s) for s in subs]
    else:
        results = Parallel(n_jobs=subj_n_jobs)(
            delayed(_run)(s) for s in subs
        )

    caps, radii, dims = zip(*results)
    return np.stack(caps), np.stack(radii), np.stack(dims)


def main() -> None:
    atlas_data, rois = load_atlas()
    out_dir = "manifold_results"
    os.makedirs(out_dir, exist_ok=True)

    # fully parallel across ROIs and subjects by default:
    exp_c, exp_r, exp_d = run_group(
        EXPERTS, atlas_data, rois, out_dir,
        subj_n_jobs=-1, roi_n_jobs=-1
    )
    non_c, non_r, non_d = run_group(
        NONEXPERTS, atlas_data, rois, out_dir,
        subj_n_jobs=-1, roi_n_jobs=-1
    )

    # run FDR‐corrected t-tests on the metric arrays:
    stats_c = fdr_ttest(exp_c, non_c, rois)
    stats_r = fdr_ttest(exp_r, non_r, rois)
    stats_d = fdr_ttest(exp_d, non_d, rois)

    # build and save group means & stats
    mean_df = pd.DataFrame({
        "ROI": rois,
        "capacity_expert": np.nanmean(exp_c, axis=0),
        "capacity_nonexpert": np.nanmean(non_c, axis=0),
        "radius_expert": np.nanmean(exp_r, axis=0),
        "radius_nonexpert": np.nanmean(non_r, axis=0),
        "dimension_expert": np.nanmean(exp_d, axis=0),
        "dimension_nonexpert": np.nanmean(non_d, axis=0),
    })
    mean_df.to_csv(os.path.join(out_dir, "group_means.csv"), index=False)
    stats_c.to_csv(os.path.join(out_dir, "stats_capacity.csv"), index=False)
    stats_r.to_csv(os.path.join(out_dir, "stats_radius.csv"), index=False)
    stats_d.to_csv(os.path.join(out_dir, "stats_dimension.csv"), index=False)

    # plotting
    for metric in ("capacity", "radius", "dimension"):
        df_long = mean_df.melt(
            id_vars="ROI",
            value_vars=[f"{metric}_expert", f"{metric}_nonexpert"],
            var_name="group",
            value_name=metric,
        )
        df_long["group"] = df_long["group"].str.contains("expert").map({True: "expert", False: "novice"})
        plot_group_comparison(df_long, out_dir, metric)

        plot_group_heatmap(
            mean_df[[f"{metric}_expert", f"{metric}_nonexpert"]].values,
            rois, metric, out_dir
        )

    logger.info("Analysis complete. Results in %s", out_dir)


if __name__ == "__main__":
    main()
