#!/usr/bin/env python3
"""Run MVPA fsaverage surface plots from repo root.

Builds LH overlays from significant ROI stats and saves flat surface plots
for selected contrasts/regressors.
"""

import os
import glob
import math
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt

from common.common_utils import create_run_id, save_script_to_file
from common.logging_utils import setup_logging
from common.surface_plotting import plot_left_fsaverage_grid
from config import MVPA_RESULTS_ROOT
from mvpa.modules.surf_helpers import build_significance_overlay as build_overlay
from mvpa.modules.text_utils import format_contrast
from mvpa.modules import HCPMMP1_LH_LABELS, HCPMMP1_RH_LABELS


RUN_ID = create_run_id()

def _discover_runs(root: str) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    cortex_runs = glob.glob(os.path.join(str(root), "*glasser_cortex*")) + glob.glob(os.path.join(str(root), "*glasser_cortices*"))
    for d in cortex_runs:
        if os.path.isdir(d):
            pairs.append((d, os.path.join("rois", "sets", "glasser_cortex_bilateral")))
    return pairs


def _generate_for_pair(analysis_results_path: str, roi_annotation_path: str, analyses: list[str], contrasts: list[str], regressors_to_keep: list[str] | None = None, cmap_used: str = "RdPu") -> None:
    regressors_to_keep = regressors_to_keep or []
    for analysis in analyses:
        pkl_files = glob.glob(os.path.join(analysis_results_path, analysis, "*group/*.pkl"))
        if not pkl_files:
            continue
        analysis_results_pickle = pkl_files[0]
        group_path = os.path.dirname(analysis_results_pickle)

        roi_ts = glob.glob(os.path.join(roi_annotation_path, "*.tsv"))
        if not roi_ts:
            continue
        measure_string = "Decoding Accuracy" if analysis == "svm" else "Coefficient"

        with open(analysis_results_pickle, "rb") as f:
            analysis_results = pickle.load(f)

        for contrast in contrasts:
            contrast_dir = os.path.join(group_path, contrast)
            os.makedirs(contrast_dir, exist_ok=True)
            out_dir = os.path.join(contrast_dir, f"{create_run_id()}_surfplots")
            os.makedirs(out_dir, exist_ok=False)
            formatted_contrast = format_contrast(contrast)

            contrast_results_dict = {
                regressor: rois
                for comp_key, reg_dict in analysis_results.items()
                if comp_key == contrast
                for regressor, rois in reg_dict.items()
            }
            all_regressors = list(contrast_results_dict.keys())
            regressors = [r for r in all_regressors if (not regressors_to_keep or r in regressors_to_keep)]

            mins, maxs = [], []
            for regressor in regressors:
                df = pd.DataFrame.from_dict(contrast_results_dict[regressor], orient='index')
                df = df[df["fdr_reject"] == True]
                min_val = (df["mean_diff"]).min()
                max_val = (df["mean_diff"]).max()
                vmin = 0 if np.isnan(min_val) or np.isinf(min_val) else math.floor(min_val * 1000) / 1000
                vmax = 0 if np.isnan(max_val) or np.isinf(max_val) else math.ceil(max_val * 1000) / 1000
                mins.append(vmin); maxs.append(vmax)
            vmin = float(np.min(mins)) if mins else 0.0
            vmax = float(np.max(maxs)) if maxs else 0.0

            for regressor in regressors:
                single_regressor_data = contrast_results_dict[regressor]
                title = f"{formatted_contrast} | {measure_string} difference | {regressor.replace('_',' ').capitalize()}"
                overlay_src = pd.DataFrame.from_dict(single_regressor_data, orient='index')
                overlay_src = overlay_src[overlay_src["fdr_reject"] == True].rename(columns={"mean_diff": "mean-chance"})
                overlay_lh, overlay_rh = build_overlay(overlay_src, lh_labels=HCPMMP1_LH_LABELS, rh_labels=HCPMMP1_RH_LABELS)
                _fig, _ = plot_left_fsaverage_grid(
                    overlay_lh=overlay_lh,
                    label_map_lh=HCPMMP1_LH_LABELS,
                    title=title,
                    out_path=os.path.join(out_dir, title + "_surfplot.png"),
                    color_range=(vmin, vmax),
                    cmap_positive=cmap_used,
                    cmap_negative=cmap_used,
                    add_colorbar=True,
                    add_contours=True,
                )
                plt.close(_fig)


if __name__ == "__main__":
    OUT_ROOT = os.path.join("results", f"{RUN_ID}_mvpa_surfplots")
    os.makedirs(OUT_ROOT, exist_ok=True)
    setup_logging(); setup_logging(log_file=os.path.join(OUT_ROOT, "mvpa_surfplots.log"))
    save_script_to_file(OUT_ROOT)

    pairs = _discover_runs(str(MVPA_RESULTS_ROOT))
    analyses = ["rsa_corr"]
    contrasts = ["experts_vs_nonexperts"]
    for res_dir, roi_dir in pairs:
        _generate_for_pair(res_dir, roi_dir, analyses, contrasts)
