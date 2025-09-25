#!/usr/bin/env python3
"""Run MVPA group barplots from repo root.

Discovers MVPA run folders under `config.MVPA_RESULTS_ROOT`, pairs them with
ROI TSVs under `rois/sets/*`, loads group pickles and writes barplots per
contrast and regressor.
"""

import os
import glob
import math
import numpy as np
import pandas as pd
import pickle
import seaborn as sns
import matplotlib.pyplot as plt

from common.common_utils import create_run_id, save_script_to_file
from common.logging_utils import setup_logging
from config import MVPA_RESULTS_ROOT
from mvpa.modules.text_utils import format_contrast


RUN_ID = create_run_id()

def _discover_runs(root: str) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    cortex_runs = glob.glob(os.path.join(str(root), "*glasser_cortex*")) + glob.glob(os.path.join(str(root), "*glasser_cortices*"))
    region_runs = glob.glob(os.path.join(str(root), "*glasser_regions*"))
    for d in cortex_runs:
        if os.path.isdir(d):
            pairs.append((d, os.path.join("rois", "sets", "glasser_cortex_bilateral")))
    for d in region_runs:
        if os.path.isdir(d):
            pairs.append((d, os.path.join("rois", "sets", "glasser_regions_bilateral")))
    return pairs


def _generate_for_pair(analysis_results_path: str, roi_annotation_path: str, analyses: list[str], contrasts: list[str], regressors_to_keep: list[str] | None = None) -> None:
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
        roi_df = pd.read_csv(roi_ts[0], sep='\t')

        measure_string = "Decoding Accuracy" if analysis == "svm" else "Coefficient"
        with open(analysis_results_pickle, "rb") as f:
            analysis_results = pickle.load(f)

        for contrast in contrasts:
            contrast_dir = os.path.join(group_path, contrast)
            os.makedirs(contrast_dir, exist_ok=True)
            out_dir = os.path.join(contrast_dir, f"{create_run_id()}_bplots")
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
                temp_df = pd.DataFrame.from_dict(contrast_results_dict[regressor], orient='index')
                temp_df["ci_low"] = temp_df["CI95"].apply(lambda x: x[0])
                temp_df["ci_high"] = temp_df["CI95"].apply(lambda x: x[1])
                temp_df["ci_low"] = np.where(np.isinf(temp_df["ci_low"]), temp_df["mean_diff"], temp_df["ci_low"])  # noqa: E501
                temp_df["ci_high"] = np.where(np.isinf(temp_df["ci_high"]), temp_df["mean_diff"], temp_df["ci_high"])  # noqa: E501
                min_val = (temp_df["mean_diff"] - (temp_df["mean_diff"] - temp_df["ci_low"]) - 0.05).min()
                max_val = (temp_df["mean_diff"] + (temp_df["ci_high"] - temp_df["mean_diff"]) + 0.05).max()
                y_min = 0 if (np.isnan(min_val) or np.isinf(min_val)) else math.floor(min_val * 1000) / 1000
                y_max = 0 if (np.isnan(max_val) or np.isinf(max_val)) else math.ceil(max_val * 1000) / 1000
                mins.append(y_min); maxs.append(y_max)
            y_min = np.min(mins) if mins else 0.0
            y_max = np.max(maxs) if maxs else 0.0

            for regressor in regressors:
                single_df = pd.DataFrame.from_dict(contrast_results_dict[regressor], orient='index').reset_index().rename(columns={"index": "roi"})
                single_df["ci_low"], single_df["ci_high"] = single_df["CI95"].apply(lambda x: x[0]), single_df["CI95"].apply(lambda x: x[1])
                roi_color_map = dict(zip(roi_df["region_name"], roi_df["color"]))
                roi_order_map = dict(zip(roi_df["region_name"], roi_df["order"] if ("order" in roi_df and not roi_df["order"].isna().all()) else roi_df["region_id"]))  # noqa: E501
                single_df["color"] = single_df["roi"].map(roi_color_map)
                single_df["order"] = single_df["roi"].map(roi_order_map)
                single_df["roi"] = single_df["roi"].str.replace("_", " ")
                single_df = single_df.sort_values("order")
                ordered_rois = list(single_df["roi"].values)
                single_df["roi"] = pd.Categorical(single_df["roi"], categories=ordered_rois, ordered=True)
                single_df.sort_values("roi", inplace=True)

                title = f"{formatted_contrast} | {measure_string} difference | {regressor.replace('_',' ').capitalize()}"
                plt.figure(figsize=(12, 6))
                ax = sns.barplot(data=single_df, x="roi", y="mean_diff", color="gray")
                err_low = single_df["mean_diff"] - single_df["ci_low"]
                err_high = single_df["ci_high"] - single_df["mean_diff"]
                ax.errorbar(x=np.arange(len(single_df)), y=single_df["mean_diff"], yerr=[err_low, err_high], fmt='none', ecolor='k', capsize=3)  # noqa: E501
                ax.set_title(title)
                ax.set_xlabel("ROI"); ax.set_ylabel(measure_string); ax.set_ylim(y_min, y_max)
                plt.xticks(rotation=90)
                out_png = os.path.join(out_dir, f"{regressor}_barplot.png")
                plt.tight_layout(); plt.savefig(out_png, dpi=300); plt.close()


if __name__ == "__main__":
    OUT_ROOT = os.path.join("results", f"{RUN_ID}_mvpa_barplots")
    os.makedirs(OUT_ROOT, exist_ok=True)
    setup_logging(); setup_logging(log_file=os.path.join(OUT_ROOT, "mvpa_barplots.log"))
    save_script_to_file(OUT_ROOT)

    pairs = _discover_runs(str(MVPA_RESULTS_ROOT))
    if not pairs:
        raise SystemExit("No MVPA run folders found under MVPA_RESULTS_ROOT")
    analyses = ["svm", "rsa_corr"]
    contrasts = ["experts_vs_nonexperts", "nonexperts_vs_chance", "experts_vs_chance"]
    for res_dir, roi_dir in pairs:
        _generate_for_pair(res_dir, roi_dir, analyses, contrasts)

