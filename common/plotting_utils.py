#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Shared plotting utilities (barplots, etc.)."""
from __future__ import annotations

import os
from typing import Dict, List, Any

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def plot_mvpa_barplot(
    data: Dict[str, pd.DataFrame],
    x_hue: str,
    y_col: str,
    x_groups: str | None = None,
    chance_level: float = 0.0,
    title: str = "",
    hue_order: List[str] | None = None,
    x_groups_order: List[str] | None = None,
    out_dir: str | None = None,
    plot_points: bool = False,
    use_corrected_p: bool = True,
    vmin: float | None = None,
    vmax: float | None = None,
):
    """Generic barplot for MVPA group results with optional significance stars.

    Expects `data` mapping each hue (e.g., regressor) to a DataFrame with index as groups (ROIs)
    and columns including y_col and p-values (either `p_corrected` or `p_uncorrected`).
    """

    def _long_df(d: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        rows: List[dict[str, Any]] = []
        for hue_value, df in d.items():
            for idx, row in df.iterrows():
                rows.append({"group": idx, "hue": hue_value, y_col: row.get(y_col, np.nan), "p": row.get("p_corrected" if use_corrected_p else "p_uncorrected", np.nan)})
        return pd.DataFrame(rows)

    long_df = _long_df(data)
    if x_groups_order is not None:
        long_df["group"] = pd.Categorical(long_df["group"], categories=x_groups_order, ordered=True)
    if hue_order is not None:
        long_df["hue"] = pd.Categorical(long_df["hue"], categories=hue_order, ordered=True)

    plt.figure(figsize=(12, 6))
    ax = sns.barplot(data=long_df, x="group", y=y_col, hue="hue", ci=None)
    if plot_points:
        sns.stripplot(data=long_df, x="group", y=y_col, hue="hue", dodge=True, color="k", alpha=0.4)
    ax.axhline(chance_level, color="gray", linestyle="--", linewidth=1)

    # Add star annotations
    for i, (x_group, sub) in enumerate(long_df.groupby("group")):
        for j, (hval, sub2) in enumerate(sub.groupby("hue")):
            y = sub2[y_col].iloc[0]
            p = sub2["p"].iloc[0]
            star = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else ""))
            if star:
                ax.text(i + (j - 0.5) * 0.2, y + 0.01, star, ha="center", va="bottom", color="black")

    ax.set_title(title)
    ax.set_xlabel(x_groups or "")
    ax.set_ylabel(y_col)
    if vmin is not None or vmax is not None:
        lo = vmin if vmin is not None else ax.get_ylim()[0]
        hi = vmax if vmax is not None else ax.get_ylim()[1]
        ax.set_ylim(lo, hi)
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f"{title.replace(' ', '_').replace('|','')}.png")
        plt.savefig(out_path, dpi=300)
    return long_df

