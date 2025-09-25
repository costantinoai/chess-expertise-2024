#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Helper functions to visualise the results of the correlation analysis.

The functions in this module produce brain overlays (glass brain images),
per-term correlation bar plots and difference plots with bootstrap confidence
intervals and significance annotations.  All plotting options are consistent
with the styles defined in :mod:`modules.config`.
"""

import os
import matplotlib.pyplot as plt
import seaborn as sns
from nilearn import image
from nilearn.plotting import plot_glass_brain
from matplotlib.ticker import MultipleLocator
import pandas as pd
import numpy as np
from modules.config import (
    BRAIN_CMAP,
    TERM_ORDER,
    logger,
    PALETTE,
    LEVELS_MAPS,
    COL_POS,
    COL_NEG,
)

YLIM = [-0.15, 0.3]

# --- Base font size ---
base_font_size = 26

# --- Global matplotlib settings using relative multipliers ---
plt.rcParams.update(
    {
        "font.size": base_font_size,
        "axes.titlesize": base_font_size * 1.4,  # 28
        "axes.labelsize": base_font_size * 1.2,  # 24
        "xtick.labelsize": base_font_size,  # 20
        "ytick.labelsize": base_font_size,  # 20
        "legend.fontsize": base_font_size,  # 20
        "figure.figsize": (12, 9),
    }
)


def plot_map(arr, ref_img, title, outpath, thresh=1e-5):
    """
    Render and save a glass brain plot of a 3D array.

    Parameters
    ----------
    arr : np.ndarray
        3D image data array to plot.
    ref_img : nibabel.Nifti1Image
        Reference image for affine and header.
    title : str
        Title text for the plot (brain levels are extracted if possible).
    outpath : str
        Filepath to save the output PNG.
    """
    # Attempt to extract level labels from title if "_" delimited
    try:
        parts = title.split("_")
        level1 = LEVELS_MAPS[parts[1]]
        level2 = LEVELS_MAPS[parts[2].split(":")[0]]
        if len(parts) > 3:
            level3 = LEVELS_MAPS.get(
                parts[3].split(":")[0], parts[3]
            )  # fallback to raw if not mapped
            title = f"{level1} | {level2} | {level3}"
        else:
            title = f"{level1} | {level2}"
    except Exception:
        pass  # Keep original title if parsing fails
    # Create image and display
    img = image.new_img_like(ref_img, arr)
    disp = plot_glass_brain(
        img,
        display_mode="lyrz",
        colorbar=True,
        cmap=BRAIN_CMAP,
        symmetric_cbar=True,
        plot_abs=False,
        threshold=thresh,
    )

    # Add custom title to the correct axis
    disp.title(title, size=28, color="black", bgcolor="white", weight="bold")

    # --- Set colorbar ticks to min, 0, max ---
    # data_max = np.nanmax(arr)
    # data_min = np.nanmin(arr)
    # lim = np.max([np.abs(data_max), np.abs(data_min)])
    # try:
    #     cbar_ax = disp._cbar.ax  # colorbar is usually the last axis in the figure
    #     cbar_ax.set_yticks([-lim, 0, lim])
    #     cbar_ax.set_yticklabels([f"{-lim:.2f}", "0", f"{lim:.2f}"])
    # except:
    #     pass

    # Save using the correct figure
    disp.savefig(outpath.replace(">", "-gt-"), dpi=300)
    plt.show()
    plt.close()
    logger.info(f"Saved brain map: {outpath}")


def plot_term_maps(term_maps, out_dir):
    """
    Plot and save brain maps for each term in term_maps.

    Parameters
    ----------
    term_maps : dict
        Mapping from term name to NIfTI file path.
    out_dir : str
        Directory to save term map PNGs.
    """
    os.makedirs(out_dir, exist_ok=True)
    for term, path in term_maps.items():
        # Load image data and call plot_map
        data = image.load_img(path).get_fdata()
        filename = f"term_{term.replace(' ', '_')}.png"
        plot_map(
            data,
            image.load_img(path),
            f"Term: {term.title()}",
            os.path.join(out_dir, filename),
        )


def plot_correlations(df_pos, df_neg, df_diff, out_fig=None, out_csv=None, run_id=None):
    """
    Paired bar plot (POS vs NEG).
    - Draw POS/NEG error bars and stars only if those cols exist.
    - Draw DIFF significance only if df_diff has ['p_fdr','sig'].
    """
    # Detect availability
    has_pos_ci  = {'CI_low', 'CI_high'}.issubset(df_pos.columns)
    has_neg_ci  = {'CI_low', 'CI_high'}.issubset(df_neg.columns)
    has_pos_sig = {'p_fdr'}.issubset(df_pos.columns)
    has_neg_sig = {'p_fdr'}.issubset(df_neg.columns)
    has_diff_sig = {'p_fdr', 'sig'}.issubset(df_diff.columns)
    has_diff_ci  = {'CI_low', 'CI_high'}.issubset(df_diff.columns)

    terms = TERM_ORDER
    n = len(terms)

    r_pos = list(df_pos['r'])
    r_neg = list(df_neg['r'])

    pos_ci_low  = list(df_pos['CI_low'])  if has_pos_ci else [np.nan] * n
    pos_ci_high = list(df_pos['CI_high']) if has_pos_ci else [np.nan] * n
    neg_ci_low  = list(df_neg['CI_low'])  if has_neg_ci else [np.nan] * n
    neg_ci_high = list(df_neg['CI_high']) if has_neg_ci else [np.nan] * n

    plot_df = pd.DataFrame({
        "Term": terms * 2,
        "Correlation": r_pos + r_neg,
        "CI_low": pos_ci_low + neg_ci_low,
        "CI_high": pos_ci_high + neg_ci_high,
        "Group": ["Positive z-map"] * n + ["Negative z-map"] * n,
    })
    plot_df["Term"] = plot_df["Term"].str.title()

    fig, ax = plt.subplots(figsize=(14, 10))
    sns.barplot(
        data=plot_df,
        x="Term", y="Correlation",
        hue="Group",
        palette=PALETTE, dodge=True,
        errorbar=None, ax=ax,
    )

    patches = ax.patches
    data_range = np.abs(YLIM[0]) + np.abs(YLIM[1])
    offset = data_range / 100 if data_range > 0 else 0.005

    group_heights = []
    for idx, row in plot_df.iterrows():
        patch = patches[idx]
        x_c = patch.get_x() + patch.get_width() / 2
        r   = row["Correlation"]
        lo  = row["CI_low"]
        hi  = row["CI_high"]

        is_pos_group = (idx < n)
        group_has_ci  = has_pos_ci if is_pos_group else has_neg_ci
        group_has_sig = has_pos_sig if is_pos_group else has_neg_sig

        # Error bars (only if present)
        if group_has_ci and np.isfinite(lo) and np.isfinite(hi):
            err_low  = abs(r - lo)
            err_high = abs(hi - r)
            ax.errorbar(
                x_c, r,
                yerr=[[err_low], [err_high]],
                fmt="none", ecolor="k",
                capsize=4, capthick=1.5,
            )
        else:
            err_high = 0.0

        # Stars for POS/NEG only if p_fdr present
        if group_has_sig:
            p = (df_pos if is_pos_group else df_neg)["p_fdr"].iloc[idx % n]
            if p < 0.05:
                star = "***" if p < 0.001 else ("**" if p < 0.01 else "*")
                # Place using CI if available; otherwise offset from bar
                if group_has_ci and np.isfinite(hi) and r >= 0:
                    y_star, va = hi + offset, "bottom"
                elif group_has_ci and np.isfinite(lo) and r < 0:
                    y_star, va = lo - offset, "top"
                else:
                    y_star, va = (r + offset*6, "bottom") if r >= 0 else (r - offset*6, "top")
                ax.text(x_c, y_star, star, ha="center", va=va, color="gray")
                group_heights.append(y_star)
            else:
                group_heights.append(r + err_high)
        else:
            group_heights.append(r + err_high)

    # DIFF significance only if df_diff has it
    if has_diff_sig:
        for i, sig in enumerate(df_diff["sig"]):
            if sig:
                bar1, bar2 = patches[i], patches[i + n]
                x1 = bar1.get_x() + bar1.get_width() / 2
                x2 = bar2.get_x() + bar2.get_width() / 2

                max_indiv_star_y = max(group_heights[i], group_heights[i + n])
                y_line = max_indiv_star_y + offset * 8

                ax.plot([x1, x2], [y_line, y_line], "k-", linewidth=1.5)

                p = df_diff.loc[i, "p_fdr"]
                star = "***" if p < 0.001 else ("**" if p < 0.01 else "*")
                ax.text((x1 + x2) / 2, y_line + offset, star, ha="center", va="bottom", color="black")

    # Styling
    ax.set_xlabel("Terms", fontweight="bold")
    ax.set_ylabel("Correlation (z)", fontweight="bold")
    ax.set_title(f"{run_id} Correlations", pad=20)
    ax.set_xticklabels(plot_df["Term"].unique(), rotation=30, ha="right")
    ax.legend(loc="upper right", bbox_to_anchor=(1, 1), frameon=False)
    sns.despine(ax=ax, top=True, right=True, left=False, bottom=False)
    ax.spines["left"].set_linewidth(1.0)
    ax.spines["bottom"].set_linewidth(1.0)
    ax.yaxis.set_major_locator(MultipleLocator(0.05))
    ax.set_ylim(YLIM)

    # Annotated CSV
    if has_pos_sig and has_neg_sig:
        plot_df["p_fdr_indiv"] = list(df_pos["p_fdr"]) + list(df_neg["p_fdr"])
    if has_diff_sig:
        plot_df["p_fdr_diff"] = list(df_diff["p_fdr"]) * 2

    # Save
    plt.tight_layout(rect=[0, 0, 0.85, 1])
    if out_fig is not None:
        if out_csv is not None:
            plot_df.to_csv(out_csv, index=False)
        plt.savefig(out_fig.replace(">", "-gt-"), dpi=300)
    plt.show()
    return plot_df


def plot_difference(diff_df, out_fig=None, run_id=None, col_pos=COL_POS, col_neg=COL_NEG):
    """
    Plot bar chart of correlation differences (pos - neg).
    - Draw error bars and significance only if the relevant columns exist.
    Expected columns (if available):
      - 'term', 'r_diff'
      - optionally: 'CI_low', 'CI_high', 'p_fdr', 'sig'
    """
    import matplotlib.patches as mpatches

    # Detect availability
    has_ci  = {'CI_low', 'CI_high'}.issubset(diff_df.columns)
    has_sig = {'p_fdr', 'sig'}.issubset(diff_df.columns)

    # Terms (drop the first two chars, then title-case, as you had)
    terms_raw = diff_df["term"].tolist()
    terms = [t[2:] for t in terms_raw]
    labels = [t.title() for t in terms]

    r_diff = diff_df["r_diff"].values
    ci_low = diff_df["CI_low"].values if has_ci else None
    ci_high = diff_df["CI_high"].values if has_ci else None
    p_vals = diff_df["p_fdr"].values if has_sig else None
    sigs   = diff_df["sig"].values   if has_sig else None

    fig, ax = plt.subplots()

    # Bars
    for i, v in enumerate(r_diff):
        ax.bar(i, v, color=col_pos if v >= 0 else col_neg)

    # Error bars (only if CI present)
    offset = 0.005
    annotation_levels = []
    if has_ci:
        for i, v in enumerate(r_diff):
            lo, hi = ci_low[i], ci_high[i]
            yerr_low = max(0, v - lo)
            yerr_high = max(0, hi - v)
            ax.errorbar(
                i, v,
                yerr=[[yerr_low], [yerr_high]],
                fmt="none",
                ecolor="k",
                elinewidth=1.5,
                capsize=5,
                capthick=1.5,
            )

    # Stars (only if p_fdr/sig present)
    if has_sig:
        for i, v in enumerate(r_diff):
            if sigs[i]:
                p = p_vals[i]
                star = "***" if p < 0.001 else ("**" if p < 0.01 else "*")
                if has_ci:
                    lo, hi = ci_low[i], ci_high[i]
                    if v >= 0:
                        y_star, va = hi + offset, "bottom"
                    else:
                        y_star, va = lo - offset, "top"
                else:
                    # No CI available; place star just above/below the bar
                    y_star, va = (v + offset*6, "bottom") if v >= 0 else (v - offset*6, "top")
                ax.text(i, y_star, star, ha="center", va=va, color="black")
                annotation_levels.append(y_star)

    # Y-lims
    ymin, ymax = ax.get_ylim()
    if annotation_levels:
        ymax = max(ymax, max(annotation_levels) + offset)
    ax.set_ylim([min(-0.2, ymin), max(0.35, ymax)])

    # Legend
    expert_patch = mpatches.Patch(color=col_pos, label="Experts > Novices")
    novice_patch = mpatches.Patch(color=col_neg, label="Novices > Experts")
    ax.legend(handles=[expert_patch, novice_patch], loc="upper right", bbox_to_anchor=(1, 1), frameon=False)

    # Styling
    ax.set_xlabel("Terms", fontweight="bold")
    ax.set_ylabel("ΔCorrelation (z)", fontweight="bold")
    ax.set_title(f"{run_id}", pad=20)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=30, ha="right")
    sns.despine(ax=ax, top=True, right=True, left=False, bottom=False)
    ax.spines["left"].set_linewidth(1.0)
    ax.spines["bottom"].set_linewidth(1.0)
    ax.yaxis.set_major_locator(MultipleLocator(0.1))

    # Save + show
    plt.tight_layout()
    if out_fig is not None:
        plt.savefig(out_fig.replace(">", "-gt-"), dpi=300)
    plt.show()
