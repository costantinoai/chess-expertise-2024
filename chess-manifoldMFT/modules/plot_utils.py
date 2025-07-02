#!/usr/bin/env python3
"""Plotting helpers for manifold analysis."""


import os
from typing import Iterable

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

from . import logger

sns.set_style("whitegrid")

def plot_all_group_manifold_plots(full_df, out_dir, metrics_to_plot, rois, roi_name_map, exp_subs, nov_subs):
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    from modules.analysis_utils import build_metric_array, fdr_ttest
    CUSTOM_COLORS = [
        "#c6dbef", "#c6dbef", "#2171b5", "#2171b5", "#2171b5", "#a1d99b", "#a1d99b", "#a1d99b", "#a1d99b",
        "#00441b", "#00441b", "#00441b", "#fbb4b9", "#fbb4b9", "#cb181d", "#cb181d", "#cb181d", "#cb181d",
        "#fec44f", "#fec44f", "#fec44f", "#fec44f"
    ]
    # --- Plot 1: Barplot per ROI, experts vs non-experts ---
    for metric in metrics_to_plot:
        exp_arr = build_metric_array(full_df, exp_subs, rois, metric)
        nov_arr = build_metric_array(full_df, nov_subs, rois, metric)
        stats_df = fdr_ttest(exp_arr, nov_arr, rois, alpha=0.05)
        mean_exp = np.nanmean(exp_arr, axis=0)
        mean_nov = np.nanmean(nov_arr, axis=0)
        group_means_df = pd.DataFrame({
            "ROI_Label": rois,
            "ExpertMean": mean_exp,
            "NonExpertMean": mean_nov
        })
        merged = pd.merge(stats_df, group_means_df, on="ROI_Label")
        merged["ROI_Name"] = merged["ROI_Label"].map(roi_name_map)
        fig, ax = plt.subplots(figsize=(14, 6))
        x = np.arange(len(rois))
        bar_width = 0.35
        palette = {roi: CUSTOM_COLORS[i % len(CUSTOM_COLORS)] for i, roi in enumerate(rois)}
        ax.bar(x - bar_width/2, mean_exp, bar_width, color=[palette[r] for r in rois], label="Expert", edgecolor='black')
        ax.bar(x + bar_width/2, mean_nov, bar_width, color=[palette[r] for r in rois], label="Novice", edgecolor='black', hatch="//")
        for i in range(len(rois)):
            ax.errorbar(x[i] - bar_width/2, mean_exp[i], yerr=[[mean_exp[i]-merged["ci95_low"][i]], [merged["ci95_high"][i]-mean_exp[i]]],
                        fmt='none', ecolor='black', capsize=4)
            ax.errorbar(x[i] + bar_width/2, mean_nov[i], yerr=[[mean_nov[i]-merged["ci95_low"][i]], [merged["ci95_high"][i]-mean_nov[i]]],
                        fmt='none', ecolor='black', capsize=4)
        for i, row in merged.iterrows():
            if row["significant_fdr"]:
                y = max(mean_exp[i], mean_nov[i]) + 0.1 * abs(mean_exp[i])
                ax.plot([x[i] - bar_width/2, x[i] + bar_width/2], [y, y], color="black", linewidth=1.2)
                ax.text(x[i], y + 0.02 * abs(y), "*", ha="center", va="bottom", fontsize=18)
        ax.set_xticks(x)
        ax.set_xticklabels([roi_name_map[r] for r in rois], rotation=30, ha="right")
        ax.set_ylabel(metric)
        ax.set_title(f"{metric}: Experts vs Non-Experts by ROI (FDR p<0.05)")
        ax.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"{metric}_barplot_grouped.png"), dpi=300)
        plt.close()
    # --- Plot 2: Subplots of differences with CI95 and asterisks ---
    fig, axes = plt.subplots(len(metrics_to_plot), 1, figsize=(14, 4*len(metrics_to_plot)))
    if len(metrics_to_plot) == 1:
        axes = [axes]
    for idx, metric in enumerate(metrics_to_plot):
        exp_arr = build_metric_array(full_df, exp_subs, rois, metric)
        nov_arr = build_metric_array(full_df, nov_subs, rois, metric)
        stats_df = fdr_ttest(exp_arr, nov_arr, rois, alpha=0.05)
        diff = stats_df["mean_diff"]
        ci_low = stats_df["ci95_low"]
        ci_high = stats_df["ci95_high"]
        ax = axes[idx]
        x = np.arange(len(rois))
        ax.bar(x, diff, color=[CUSTOM_COLORS[i % len(CUSTOM_COLORS)] for i in range(len(rois))])
        ax.errorbar(x, diff, yerr=[diff - ci_low, ci_high - diff], fmt='none', ecolor='black', capsize=4)
        for i, row in stats_df.iterrows():
            if row["significant_fdr"]:
                y = ci_high[i] + 0.05 * abs(ci_high[i])
                ax.text(x[i], y, "*", ha="center", va="bottom", fontsize=18)
        ax.set_xticks(x)
        ax.set_xticklabels([roi_name_map[r] for r in rois], rotation=30, ha="right")
        ax.set_ylabel(f"{metric} (Expert - Novice)")
        ax.set_title(f"{metric}: Group Difference by ROI (FDR p<0.05)")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "all_metrics_differences.png"), dpi=300)
    plt.close()
 
