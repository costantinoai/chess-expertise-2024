#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun  7 22:18:45 2025

@author: costantino_ai
"""

import os
import matplotlib.pyplot as plt
import seaborn as sns
from nilearn import image
from nilearn.plotting import plot_glass_brain
from matplotlib.ticker import MultipleLocator
import pandas as pd
from modules.config import BRAIN_CMAP, TITLE_FONT, TERM_ORDER, COL_POS, COL_NEG, logger, PALETTE


def plot_map(arr, ref_img, title, outpath):
    disp = plot_glass_brain(
        image.new_img_like(ref_img, arr),
        display_mode='lyrz',
        colorbar=True,
        cmap=BRAIN_CMAP,
        symmetric_cbar=True,
        title=title,
        plot_abs=False
    )
    disp.savefig(outpath, dpi=300)
    plt.show()
    plt.close()
    logger.info(f"Saved brain map: {outpath}")

def plot_term_maps(term_maps, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    for term, path in term_maps.items():
        data = image.load_img(path).get_fdata()
        plot_map(data, image.load_img(path), f"Term: {term.title()}",
                 os.path.join(out_dir, f"term_{term.replace(' ', '_')}.png"))

from modules.stats_utils import compute_difference_stats


def plot_correlations(z_pos, z_neg, term_maps, ref_img,
                      out_fig, out_csv, run_id,
                      n_boot=10000, fdr_alpha=0.05):
    """
    Plot paired correlations for positive vs negative z-maps, annotate
    both individual significance (per map) and difference significance.

    Parameters
    ----------
    z_pos, z_neg : ndarray
        3D z-maps for positive and negative contrasts.
    term_maps : dict
        Mapping term -> nifti filepath.
    ref_img : nibabel image
        Reference for resampling.
    out_fig : str
        Output PNG path.
    out_csv : str
        Output CSV path.
    run_id : str
        Title identifier.
    n_boot : int
        Bootstraps for difference test.
    fdr_alpha : float
        FDR alpha-level.
    """
    # Compute individual correlation DataFrames
    from .stats_utils import compute_zmap_correlations
    df_pos = compute_zmap_correlations(z_pos, term_maps, ref_img)
    df_neg = compute_zmap_correlations(z_neg, term_maps, ref_img)

    # Compute difference statistics
    diff_df = compute_difference_stats(z_pos, z_neg, term_maps, ref_img,
                                       n_boot=n_boot, fdr_alpha=fdr_alpha)

    # Build combined DataFrame for plotting
    terms = TERM_ORDER
    n = len(terms)
    plot_df = pd.DataFrame({
        'Term': terms * 2,
        'Correlation': list(df_pos['r']) + list(df_neg['r']),
        'CI_low': list(df_pos['CI_low']) + list(df_neg['CI_low']),
        'CI_high': list(df_pos['CI_high']) + list(df_neg['CI_high']),
        'Group': ['Positive z-map']*n + ['Negative z-map']*n
    })
    plot_df['Term'] = plot_df['Term'].str.title()

    # Figure setup
    bar_w = 1.5
    fig_w = max(6, bar_w * n)
    fontsize = min(24, max(18, fig_w * 0.4))
    fig, ax = plt.subplots(figsize=(fig_w, 8))

    sns.barplot(data=plot_df, x='Term', y='Correlation', hue='Group',
                palette=PALETTE, dodge=True, errorbar=None, ax=ax)

    # Add error bars and individual significance
    patches = ax.patches
    offset = 0.02
    for idx, row in plot_df.iterrows():
        patch = patches[idx]
        x_c = patch.get_x() + patch.get_width()/2
        r = row['Correlation']
        lo, hi = row['CI_low'], row['CI_high']
        eb_low, eb_high = r - lo, hi - r
        ax.errorbar(x_c, r, yerr=[[eb_low],[eb_high]], fmt='none',
                    ecolor='k', capsize=4, capthick=1.5)
        # star for individual map
        p = (df_pos if idx < n else df_neg)['p_fdr'].iloc[idx % n]
        if p < 0.05:
            star = '***' if p<0.001 else ('**' if p<0.01 else '*')
            ax.text(x_c, hi+offset, star,
                    ha='center', va='bottom', fontsize=fontsize, color='gray')

    # Add difference significance stars
    for i, sig in enumerate(diff_df['sig']):
        if sig:
            # coordinates for two bars
            bar1 = patches[i]
            bar2 = patches[i+n]
            x1 = bar1.get_x()+bar1.get_width()/2
            x2 = bar2.get_x()+bar2.get_width()/2
            y1 = bar1.get_height() + max(diff_df.loc[i,'CI_high']-diff_df.loc[i,'r_diff'], diff_df.loc[i,'r_diff']-diff_df.loc[i,'CI_low']) + 2*offset
            y2 = bar2.get_height() + max(diff_df.loc[i,'CI_high']-diff_df.loc[i,'r_diff'], diff_df.loc[i,'r_diff']-diff_df.loc[i,'CI_low']) + 2*offset
            y = max(y1,y2)
            ax.plot([x1, x2],[y,y],'k-',linewidth=1.5)
            p = diff_df.loc[i,'p_fdr']
            star = '***' if p<0.001 else ('**' if p<0.01 else '*')
            ax.text((x1+x2)/2, y+offset, star, ha='center', va='bottom', fontsize=fontsize, color='black')

    # Final formatting
    ax.set_xlabel('Terms', fontfamily='Ubuntu Condensed', fontsize=fontsize+2, fontweight='bold')
    ax.set_ylabel('Correlation (z)', fontfamily='Ubuntu Condensed', fontsize=fontsize+2, fontweight='bold')
    ax.set_title(f"{run_id} Correlations", pad=20, **TITLE_FONT)
    ax.set_xticklabels(plot_df['Term'].unique(), rotation=30, ha='right', fontsize=fontsize)
    ax.legend(frameon=False, fontsize=fontsize)
    sns.despine(ax=ax, top=True, right=True, left=False, bottom=False)
    ax.spines['left'].set_linewidth(1.0)
    ax.spines['bottom'].set_linewidth(1.0)
    ax.yaxis.set_major_locator(MultipleLocator(0.05))

    plt.tight_layout()
    plt.savefig(out_fig, dpi=300)
    plt.close(fig)

    # Save CSV
    plot_df['p_fdr_indiv'] = list(df_pos['p_fdr']) + list(df_neg['p_fdr'])
    plot_df['p_fdr_diff'] = list(diff_df['p_fdr'])*2
    plot_df.to_csv(out_csv, index=False)
    return plot_df


def plot_difference(diff_df: pd.DataFrame,
                    out_fig: str,
                    run_id: str,
                    col_pos: str,
                    col_neg: str) -> None:
    """
    Plot the difference DataFrame from compute_difference_stats.

    Parameters
    ----------
    diff_df : pandas.DataFrame
        Must contain ['Term','r_diff','CI_low','CI_high','p_fdr','sig']
    out_fig : str
        Filepath for saving the figure.
    run_id : str
        Title text.
    col_pos, col_neg : str
        Colors for positive/negative bars.
    title_font : dict
        Font properties for the plot title.
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    from matplotlib.ticker import MultipleLocator

    terms = diff_df['Term'].tolist()
    r_diff = diff_df['r_diff'].values
    ci_low = diff_df['CI_low'].values
    ci_high = diff_df['CI_high'].values
    p_vals = diff_df['p_fdr'].values

    n_terms = len(terms)
    bar_width = 1.2
    fig_width = max(6, bar_width * n_terms)
    fontsize = min(24, max(18, fig_width * 0.4))

    fig, ax = plt.subplots(figsize=(fig_width, 6))
    for i, v in enumerate(r_diff):
        ax.bar(i, v, color=col_pos if v >= 0 else col_neg, edgecolor='k')

    offset = 0.02
    annotation_levels = []
    for i, (v, lo, hi, p) in enumerate(zip(r_diff, ci_low, ci_high, p_vals)):
        yerr_low = max(0, v - lo)
        yerr_high = max(0, hi - v)
        ax.errorbar(i, v, yerr=[[yerr_low], [yerr_high]], fmt='none',
                    ecolor='k', elinewidth=1.5, capsize=4, capthick=1.5)

        star = None
        if p < 0.001:
            star = '***'
        elif p < 0.01:
            star = '**'
        elif p < 0.05:
            star = '*'

        if star:
            if v >= 0:
                y_star = hi + offset
                va = 'bottom'
            else:
                y_star = lo - offset
                va = 'top'
            ax.text(i, y_star, star, ha='center', va=va,
                    fontsize=fontsize, color='black')
            annotation_levels.append(y_star)

    ymin, ymax = ax.get_ylim()
    ymin = min(ymin, -0.05)
    if annotation_levels:
        ymax = max(ymax, max(annotation_levels) + offset)
    ax.set_ylim([ymin, ymax])

    ax.set_xlabel('Terms', fontfamily='Ubuntu Condensed',
                  fontsize=fontsize+2, fontweight='bold')
    ax.set_ylabel('ΔCorrelation (z)', fontfamily='Ubuntu Condensed',
                  fontsize=fontsize+2, fontweight='bold')
    ax.set_title(f"{run_id}", pad=20, **title_font)

    ax.set_xticks(range(n_terms))
    ax.set_xticklabels([t.title() for t in terms], rotation=30,
                       ha='right', fontsize=fontsize)

    sns.despine(ax=ax, top=True, right=True, left=False, bottom=False)
    ax.spines['left'].set_linewidth(1.0)
    ax.spines['bottom'].set_linewidth(1.0)
    ax.yaxis.set_major_locator(MultipleLocator(0.1))

    plt.tight_layout()
    plt.savefig(out_fig, dpi=300)
    plt.show()
