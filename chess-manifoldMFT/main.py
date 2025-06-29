#!/usr/bin/env python3
"""Run ROI-wise manifold analysis comparing experts and novices."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from modules import (
    logger,
    EXPERTS,
    NONEXPERTS,
    BASE_GLM_PATH,
    ATLAS_FILE,
)
from modules.io_utils import load_all_betas, load_atlas
from modules.analysis_utils import compute_manifold, fdr_ttest
from modules.plot_utils import plot_subject_roi, plot_group_comparison


def process_subject(
    subject: str, atlas_data: np.ndarray, rois: np.ndarray, out_dir: str
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute radius and dimension for each ROI for a single subject."""
    betas, _ = load_all_betas(subject)
    n_cond = betas.shape[0]
    labels_a = np.array([0] * (n_cond // 2) + [1] * (n_cond - n_cond // 2))
    labels_b = np.concatenate([[i, i] for i in range(n_cond // 2)])
    radii = np.zeros((len(rois), 2))
    dims = np.zeros((len(rois), 2))
    for i, roi in enumerate(rois):
        mask = atlas_data == roi
        roi_vox = betas[:, mask]
        r_a, d_a = compute_manifold(roi_vox, labels_a)
        r_b, d_b = compute_manifold(roi_vox, labels_b)
        radii[i] = [r_a, r_b]
        dims[i] = [d_a, d_b]
        plot_subject_roi(d_a, r_a, subject, roi, os.path.join(out_dir, "subject"))
    return radii, dims


def main() -> None:
    atlas_data, rois = load_atlas()
    out_dir = "manifold_results"
    os.makedirs(out_dir, exist_ok=True)

    def run_group(subs, label):
        res_r = []
        res_d = []
        for s in subs:
            r, d = process_subject(s, atlas_data, rois, out_dir)
            res_r.append(r)
            res_d.append(d)
        return np.stack(res_r), np.stack(res_d)

    exp_r, exp_d = run_group(EXPERTS, "expert")
    non_r, non_d = run_group(NONEXPERTS, "nonexpert")

    stats_r = fdr_ttest(exp_r[:, :, 0], non_r[:, :, 0], rois)
    stats_d = fdr_ttest(exp_d[:, :, 0], non_d[:, :, 0], rois)

    mean_df = pd.DataFrame(
        {
            "ROI": rois,
            "radius_expert": np.nanmean(exp_r[:, :, 0], axis=0),
            "radius_nonexpert": np.nanmean(non_r[:, :, 0], axis=0),
            "dimension_expert": np.nanmean(exp_d[:, :, 0], axis=0),
            "dimension_nonexpert": np.nanmean(non_d[:, :, 0], axis=0),
        }
    )
    mean_df.to_csv(os.path.join(out_dir, "group_means.csv"), index=False)
    stats_r.to_csv(os.path.join(out_dir, "stats_radius.csv"), index=False)
    stats_d.to_csv(os.path.join(out_dir, "stats_dimension.csv"), index=False)

    df_long_r = mean_df.melt(
        id_vars="ROI",
        value_vars=["radius_expert", "radius_nonexpert"],
        var_name="group",
        value_name="radius",
    )
    df_long_r["group"] = (
        df_long_r["group"].str.contains("expert").map({True: "expert", False: "novice"})
    )
    plot_group_comparison(df_long_r, out_dir, "radius")

    df_long_d = mean_df.melt(
        id_vars="ROI",
        value_vars=["dimension_expert", "dimension_nonexpert"],
        var_name="group",
        value_name="dimension",
    )
    df_long_d["group"] = (
        df_long_d["group"].str.contains("expert").map({True: "expert", False: "novice"})
    )
    plot_group_comparison(df_long_d, out_dir, "dimension")

    logger.info("Analysis complete. Results in %s", out_dir)


if __name__ == "__main__":
    main()
