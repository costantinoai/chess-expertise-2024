#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Shared statistical plotting utilities (bars for correlations/differences)."""
from __future__ import annotations

from typing import Optional

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def plot_correlations(
    df_pos: pd.DataFrame,
    df_neg: pd.DataFrame,
    df_diff: pd.DataFrame,
    out_fig: Optional[str] = None,
    out_csv: Optional[str] = None,
    run_id: Optional[str] = None,
    col_pos: str = '#4c924c',
    col_neg: str = '#ad4c4c',
    palette: Optional[list[str]] = None,
):
    """Paired bar plot (POS vs NEG), with optional CI and FDR stars if present."""
    sns.set_style('white')
    if palette is None:
        palette = [col_pos, col_neg]

    # Merge POS and NEG into long df
    df_pos2 = df_pos.copy(); df_pos2['sign'] = 'pos'
    df_neg2 = df_neg.copy(); df_neg2['sign'] = 'neg'
    plot_df = pd.concat([df_pos2, df_neg2], ignore_index=True)
    plot_df = plot_df.rename(columns={'term': 'Term'})
    plot_df['Term'] = plot_df['Term'].str[2:].str.title() if plot_df['Term'].str.startswith(('p_', 'n_')).any() else plot_df['Term'].str.title()

    # Add CI and significance if available from df_diff
    has_ci = {'CI_low', 'CI_high'}.issubset(df_diff.columns)
    has_sig = {'p_fdr', 'sig'}.issubset(df_diff.columns)
    if has_ci:
        ci_map = df_diff.set_index('term')[['CI_low', 'CI_high']].to_dict('index')
        plot_df['CI_low'] = plot_df['Term'].str.lower().map(lambda t: ci_map.get(t, {}).get('CI_low', np.nan))
        plot_df['CI_high'] = plot_df['Term'].str.lower().map(lambda t: ci_map.get(t, {}).get('CI_high', np.nan))

    # Plot
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.barplot(data=plot_df, x='Term', y='r', hue='sign', palette=palette, ci=None, ax=ax)

    # Error bars for CI
    if has_ci:
        for i, row in plot_df.iterrows():
            if pd.notna(row.get('CI_low')) and pd.notna(row.get('CI_high')):
                ax.errorbar(i//2 + (0 if row['sign']=='pos' else 0.2), row['r'],
                            yerr=[[max(0, row['r']-row['CI_low'])], [max(0, row['CI_high']-row['r'])]],
                            fmt='none', ecolor='k', elinewidth=1.2, capsize=3)

    ax.set_ylabel('Correlation (z)')
    ax.set_title(f"{run_id} Correlations" if run_id else "Term Correlations")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha='right')
    sns.despine(ax=ax)

    if out_csv is not None:
        plot_df.to_csv(out_csv, index=False)
    if out_fig is not None:
        plt.tight_layout(rect=[0, 0, 0.9, 1])
        plt.savefig(out_fig.replace('>', '-gt-'), dpi=300)
    plt.show()
    return plot_df


def plot_difference(
    diff_df: pd.DataFrame,
    out_fig: Optional[str] = None,
    run_id: Optional[str] = None,
    col_pos: str = '#4c924c',
    col_neg: str = '#ad4c4c',
):
    """Plot bar chart of correlation differences (pos - neg). Supports CI and FDR stars if provided."""
    sns.set_style('white')
    has_ci = {'CI_low', 'CI_high'}.issubset(diff_df.columns)
    has_sig = {'p_fdr', 'sig'}.issubset(diff_df.columns)

    terms_raw = diff_df['term'].tolist()
    terms = [t[2:] if t.startswith(('p_', 'n_')) else t for t in terms_raw]
    labels = [t.title() for t in terms]
    r_diff = diff_df['r_diff'].values if 'r_diff' in diff_df.columns else diff_df['r_diff_mean'].values
    ci_low = diff_df['CI_low'].values if has_ci else None
    ci_high = diff_df['CI_high'].values if has_ci else None
    p_vals = diff_df['p_fdr'].values if has_sig else None
    sigs = diff_df['sig'].values if has_sig else None

    fig, ax = plt.subplots(figsize=(12, 6))
    for i, v in enumerate(r_diff):
        ax.bar(i, v, color=col_pos if v >= 0 else col_neg)
        if has_ci:
            lo, hi = ci_low[i], ci_high[i]
            yerr_low = max(0, v - lo)
            yerr_high = max(0, hi - v)
            ax.errorbar(i, v, yerr=[[yerr_low], [yerr_high]], fmt='none', ecolor='k', elinewidth=1.2, capsize=3)
        if has_sig and sigs[i]:
            p = p_vals[i]
            star = '***' if p < 0.001 else ('**' if p < 0.01 else '*')
            y_star, va = (v + 0.01, 'bottom') if v >= 0 else (v - 0.01, 'top')
            ax.text(i, y_star, star, ha='center', va=va, color='black')

    ax.set_xlabel('Terms')
    ax.set_ylabel('ΔCorrelation (z)')
    ax.set_title(f"{run_id} Differences" if run_id else "Correlation Differences")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=30, ha='right')
    sns.despine(ax=ax)
    plt.tight_layout()
    if out_fig is not None:
        plt.savefig(out_fig.replace('>', '-gt-'), dpi=300)
    plt.show()
    return diff_df

