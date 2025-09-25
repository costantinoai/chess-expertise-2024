#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Shared behavioural utilities (strategy color/alpha mapping)."""
from __future__ import annotations

from typing import Sequence, Tuple, List


def compute_strategy_colors_alphas(
    strategies: Sequence[int] | Sequence[str],
    col_green: str = "#006400",
    col_red: str = "#8B0000",
) -> Tuple[List[str], List[float]]:
    """Assign colors and alphas per strategy block.

    First 5 unique blocks → green ramp of alphas (0.2..1.0), next 5 → red ramp.
    Returns (colors, alphas) lists of length == len(strategies).
    """
    current = None
    colors: List[str] = []
    alphas: List[float] = []
    color_idx = 0
    for strat in strategies:
        if strat != current:
            if color_idx < 5:
                color = col_green
                alpha = (color_idx + 1) / 5.0
            else:
                color = col_red
                alpha = (color_idx + 1 - 5) / 5.0
            current = strat
            color_idx += 1
        colors.append(color)
        alphas.append(alpha)
    return colors, alphas

