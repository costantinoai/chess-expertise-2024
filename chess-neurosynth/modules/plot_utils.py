#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plot_utils.py

Plotting utilities for z-map and term correlation analysis.
Includes functions to produce glass brain maps, term maps,
correlation bar plots, and difference bar plots with confidence
intervals and significance annotations.
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
    BRAIN_CMAP, TERM_ORDER,
    logger, PALETTE, LEVELS_MAPS,
    COL_POS, COL_NEG
)

# --- Base font size ---
base_font_size = 26

# --- Global matplotlib settings using relative multipliers ---
plt.rcParams.update({
    'font.size': base_font_size,
    'axes.titlesize': base_font_size * 1.4,   # 28
    'axes.labelsize': base_font_size * 1.2,   # 24
    'xtick.labelsize': base_font_size,        # 20
    'ytick.labelsize': base_font_size,        # 20
    'legend.fontsize': base_font_size,        # 20
    'figure.figsize': (12, 9)
})



def plot_map(arr, ref_img, title, outpath):
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
        parts = title.split('_')
        level1 = LEVELS_MAPS[parts[1]]
        level2 = LEVELS_MAPS[parts[2].split(":")[0]]
        title = f"{level1} | {level2}"
    except Exception:
        pass  # Keep original title if parsing fails

    # Create image and display
    img = image.new_img_like(ref_img, arr)
    disp = plot_glass_brain(
        img,
        display_mode='lyrz',
        colorbar=True,
        cmap=BRAIN_CMAP,
        symmetric_cbar=True,
        plot_abs=False
    )

    # Customize title appearance
    fig = plt.gcf()
    fig.axes[0].set_title(title, color='black', backgroundcolor='white')

    # --- Set colorbar ticks to min, 0, max ---
    data_max = np.nanmax(arr)
    data_min = np.nanmin(arr)
    lim = np.max([np.abs(data_max), np.abs(data_min)])
    cbar_ax = fig.axes[-1]  # colorbar is usually the last axis in the figure
    cbar_ax.set_yticks([-lim, 0, lim])
    cbar_ax.set_yticklabels([f"{-lim:.2f}", "0", f"{lim:.2f}"])

    # Save and close
    disp.savefig(outpath, dpi=300)
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
            os.path.join(out_dir, filename)
        )


def plot_correlations(df_pos, df_neg, df_diff, out_fig, out_csv, run_id):
    """
    Create a paired bar plot for positive vs negative map correlations,
    annotate bootstrap CIs and significance stars (indiv. and diff.).

    Parameters
    ----------
    df_pos : pd.DataFrame
        Positive z-map correlations with columns ['term','r','CI_low','CI_high','p_fdr','sig']
    df_neg : pd.DataFrame
        Negative z-map correlations (same columns as df_pos)
    df_diff : pd.DataFrame
        Difference stats with columns ['term','r_diff','CI_low','CI_high','p_fdr','sig']
    out_fig : str
        Filepath to save the correlation bar plot.
    out_csv : str
        Filepath to save the annotated CSV table.
    run_id : str
        Title identifier for the figure.

    Returns
    -------
    pd.DataFrame
        The combined DataFrame used for plotting (with additional p_fdr columns).
    """
    # Build DataFrame for plotting both groups
    terms = TERM_ORDER
    n = len(terms)
    plot_df = pd.DataFrame({
        'Term': terms * 2,
        'Correlation': list(df_pos['r']) + list(df_neg['r']),
        'CI_low': list(df_pos['CI_low']) + list(df_neg['CI_low']),
        'CI_high': list(df_pos['CI_high']) + list(df_neg['CI_high']),
        'Group': ['Positive z-map'] * n + ['Negative z-map'] * n
    })
    plot_df['Term'] = plot_df['Term'].str.title()

    # Create bar plot
    fig, ax = plt.subplots()
    sns.barplot(
        data=plot_df,
        x='Term', y='Correlation', hue='Group',
        palette=PALETTE,
        dodge=True, errorbar=None,
        ax=ax
    )

    # Add error bars and individual significance
    patches = ax.patches
    offset = 0.001
    for idx, row in plot_df.iterrows():
        patch = patches[idx]
        x_c = patch.get_x() + patch.get_width() / 2
        r, lo, hi = row['Correlation'], row['CI_low'], row['CI_high']
        err_low = abs(r - lo)
        err_high = abs(hi - r)
        ax.errorbar(
            x_c, r,
            yerr=[[err_low], [err_high]],
            fmt='none', ecolor='k', capsize=4, capthick=1.5
        )
        p = (df_pos if idx < n else df_neg)['p_fdr'].iloc[idx % n]
        if p < 0.05:
            star = '***' if p < 0.001 else ('**' if p < 0.01 else '*')
            ax.text(
                x_c, hi + offset, star,
                ha='center', va='bottom', color='black'
            )

    # # Annotate group differences
    # for i, sig in enumerate(df_diff['sig']):
    #     if sig:
    #         bar1, bar2 = patches[i], patches[i + n]
    #         x1 = bar1.get_x() + bar1.get_width() / 2
    #         x2 = bar2.get_x() + bar2.get_width() / 2

    #         # Difference CI margin
    #         diff_ci = max(
    #             df_diff.loc[i, 'CI_high'] - df_diff.loc[i, 'r_diff'],
    #             df_diff.loc[i, 'r_diff'] - df_diff.loc[i, 'CI_low']
    #         )
    #         # Baseline y position (top of the tallest bar)
    #         y = max(bar1.get_height(), bar2.get_height()) + diff_ci

    #         # Draw horizontal line for group comparison
    #         ax.plot([x1, x2], [y + .01, y + .01], 'k-', linewidth=1.5)

    #         # Get significance level
    #         p = df_diff.loc[i, 'p_fdr']
    #         star = '***' if p < 0.001 else ('**' if p < 0.01 else '*')

    #         # Plot asterisk above the line
    #         ax.text(
    #             (x1 + x2) / 2, y + .02, star,
    #             ha='center', va='bottom', color='black'
    #         )

    # Final styling
    ax.set_xlabel('Terms', fontweight='bold')
    ax.set_ylabel('Correlation (z)', fontweight='bold')
    ax.set_title(f"{run_id} Correlations", pad=20)
    ax.set_xticklabels(plot_df['Term'].unique(), rotation=30, ha='right')
    ax.legend(frameon=False)
    sns.despine(ax=ax, top=True, right=True, left=False, bottom=False)
    ax.spines['left'].set_linewidth(1.0)
    ax.spines['bottom'].set_linewidth(1.0)
    ax.yaxis.set_major_locator(MultipleLocator(0.05))

    # Save figure and CSV
    plt.tight_layout()
    plt.savefig(out_fig, dpi=300)
    plt.show()
    plot_df['p_fdr_indiv'] = list(df_pos['p_fdr']) + list(df_neg['p_fdr'])
    plot_df['p_fdr_diff'] = list(df_diff['p_fdr']) * 2
    plot_df.to_csv(out_csv, index=False)

    return plot_df


def plot_difference(diff_df, out_fig, run_id, col_pos=COL_POS, col_neg=COL_NEG):
    """
    Plot bar chart of correlation differences (pos - neg) with CIs and sig.

    Parameters
    ----------
    diff_df : pd.DataFrame
        Must contain ['term','r_diff','CI_low','CI_high','p_fdr','sig']
    out_fig : str
        Filepath for saving the figure.
    run_id : str
        Title text for the plot.
    col_pos : str
        Hex color for positive bars.
    col_neg : str
        Hex color for negative bars.
    """
    import matplotlib.patches as mpatches

    # Extract terms and drop first two chars as requested
    terms = [t[2:] for t in diff_df['term'].tolist()]
    r_diff = diff_df['r_diff'].values
    ci_low = diff_df['CI_low'].values
    ci_high = diff_df['CI_high'].values
    p_vals = diff_df['p_fdr'].values

    fig, ax = plt.subplots()

    # Draw bars
    for i, v in enumerate(r_diff):
        ax.bar(i, v, color=col_pos if v >= 0 else col_neg, edgecolor='k')

    # Error bars and significance stars
    offset = 0.005
    annotation_levels = []
    for i, (v, lo, hi, p) in enumerate(zip(r_diff, ci_low, ci_high, p_vals)):
        yerr_low = max(0, v - lo)
        yerr_high = max(0, hi - v)
        ax.errorbar(
            i, v,
            yerr=[[yerr_low], [yerr_high]],
            fmt='none', ecolor='k', elinewidth=1.5,
            capsize=4, capthick=1.5
        )
        if p < 0.05:
            star = '***' if p < 0.001 else ('**' if p < 0.01 else '*')
            if v >= 0:
                y_star, va = hi + offset, 'bottom'
            else:
                y_star, va = lo - offset, 'top'
            ax.text(i, y_star, star, ha='center', va=va, color='black')
            annotation_levels.append(y_star)

    # Adjust y-limits to accommodate stars
    ymin, ymax = ax.get_ylim()
    ymin = min(ymin, -0.1)
    if annotation_levels:
        ymax = max(ymax, max(annotation_levels) + offset)
    ax.set_ylim([ymin, ymax])

    # Add legend for direction of effects
    expert_patch = mpatches.Patch(color=col_pos, label='Experts > Novices')
    novice_patch = mpatches.Patch(color=col_neg, label='Novices > Experts')
    ax.legend(handles=[expert_patch, novice_patch], loc='upper right', frameon=False)

    # Final styling
    ax.set_xlabel('Terms', fontweight='bold')
    ax.set_ylabel('ΔCorrelation (z)', fontweight='bold')
    ax.set_title(f"{run_id}", pad=20)
    ax.set_xticks(range(len(terms)))
    ax.set_xticklabels([t.title() for t in terms], rotation=30, ha='right')
    sns.despine(ax=ax, top=True, right=True, left=False, bottom=False)
    ax.spines['left'].set_linewidth(1.0)
    ax.spines['bottom'].set_linewidth(1.0)
    ax.yaxis.set_major_locator(MultipleLocator(0.1))

    # Save and show
    plt.tight_layout()
    plt.savefig(out_fig, dpi=300)
    plt.show()
