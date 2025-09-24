#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Central statistical utilities reused across analyses.

Includes t-tests with confidence intervals, FDR correction wrappers,
correlation with bootstrap confidence intervals, and correlation-difference
bootstrap. Designed to keep analysis scripts thin and consistent.
"""
from __future__ import annotations

import logging
from typing import Tuple

import numpy as np
import pingouin as pg
from scipy import stats
from statsmodels.stats.multitest import multipletests, fdrcorrection


def one_sample_ttest(values: np.ndarray, mu: float = 0.0, alternative: str = "two-sided") -> dict:
    """One-sample t-test against mean `mu` with 95% CI.

    Returns dict: mean, sem, t, p, dof, ci_low, ci_high, n
    """
    arr = np.asarray(values)
    res = stats.ttest_1samp(arr, mu, nan_policy="omit", alternative=alternative)
    mean_val = np.nanmean(arr)
    sem = stats.sem(arr, nan_policy="omit")
    n = int(np.sum(np.isfinite(arr)))
    ci_low, ci_high = (np.nan, np.nan)
    try:
        ci = res.confidence_interval(); ci_low, ci_high = float(ci.low), float(ci.high)
    except Exception:
        pass
    return {
        "mean": float(mean_val),
        "sem": float(sem) if np.isfinite(sem) else np.nan,
        "t": float(res.statistic),
        "p": float(res.pvalue),
        "dof": float(getattr(res, "df", n - 1)),
        "ci_low": ci_low,
        "ci_high": ci_high,
        "n": n,
    }


def independent_ttest(x: np.ndarray, y: np.ndarray, equal_var: bool = False, alternative: str = "two-sided") -> dict:
    """Independent-samples t-test (Welch by default) with 95% CI on mean difference.

    Returns dict: delta_mean, t, p, dof, ci_low, ci_high, n_x, n_y
    """
    x = np.asarray(x); y = np.asarray(y)
    res = stats.ttest_ind(x, y, equal_var=equal_var, nan_policy="omit", alternative=alternative)
    delta_mean = float(np.nanmean(x) - np.nanmean(y))
    n_x = int(np.sum(np.isfinite(x))); n_y = int(np.sum(np.isfinite(y)))
    ci_low, ci_high = (np.nan, np.nan)
    try:
        ci = res.confidence_interval(); ci_low, ci_high = float(ci.low), float(ci.high)
    except Exception:
        pass
    return {
        "delta_mean": delta_mean,
        "t": float(res.statistic),
        "p": float(res.pvalue),
        "dof": float(getattr(res, "df", n_x + n_y - 2)),
        "ci_low": ci_low,
        "ci_high": ci_high,
        "n_x": n_x,
        "n_y": n_y,
    }


def fdr_correction(p_values: np.ndarray, alpha: float = 0.05, method: str = "fdr_bh") -> Tuple[np.ndarray, np.ndarray]:
    """Multiple-comparison correction; returns (reject_mask, corrected_pvalues)."""
    p = np.asarray(p_values)
    if method == "fdr_bh":
        reject, p_corr = multipletests(p, alpha=alpha, method="fdr_bh")[0:2]
    else:
        reject, p_corr = multipletests(p, alpha=alpha, method=method)[0:2]
    return reject, p_corr


def pearson_corr_bootstrap(x: np.ndarray, y: np.ndarray, n_boot: int = 10000, ci: float = 0.95) -> dict:
    """Pearson correlation with bootstrap CI via Pingouin.

    Returns dict: r, p, ci_low, ci_high
    """
    res = pg.corr(x=x, y=y, method='pearson', bootstraps=n_boot,
                  confidence=ci, method_ci='percentile', alternative='two-sided')
    r = float(res['r'].iloc[0]); p = float(res['p-val'].iloc[0])
    ci_low, ci_high = res['CI95%'].iloc[0]
    return {"r": r, "p": p, "ci_low": float(ci_low), "ci_high": float(ci_high)}


def corr_diff_bootstrap(term_map: np.ndarray, x: np.ndarray, y: np.ndarray, n_boot: int = 10000,
                        ci_alpha: float = 0.05, n_jobs: int = 1, rng: np.random.Generator | None = None) -> dict:
    """Bootstrap CI and p-value for difference in Pearson correlations r(term,x) - r(term,y).

    Returns dict: r_diff_mean, ci_low, ci_high, p
    """
    logger = logging.getLogger(__name__)
    n = len(x)
    if rng is None:
        rng = np.random.default_rng()

    def _boot(seed):
        sub_rng = np.random.default_rng(seed)
        idx = sub_rng.integers(0, n, size=n)
        r_pos_b = np.corrcoef(term_map[idx], x[idx])[0, 1]
        r_neg_b = np.corrcoef(term_map[idx], y[idx])[0, 1]
        return r_pos_b - r_neg_b

    seeds = rng.integers(0, 2**32 - 1, size=n_boot)
    if n_jobs == 1:
        diffs = np.array([_boot(s) for s in seeds])
    else:
        from joblib import Parallel, delayed
        diffs = np.array(Parallel(n_jobs=n_jobs)(delayed(_boot)(s) for s in seeds))

    diffs.sort()
    lo = float(np.percentile(diffs, 100 * ci_alpha / 2))
    hi = float(np.percentile(diffs, 100 * (1 - ci_alpha / 2)))
    mean_diff = float(np.mean(diffs))
    tail_low = float(np.mean(diffs <= 0)); tail_high = float(np.mean(diffs >= 0))
    p_val = float(2 * min(tail_low, tail_high))
    logger.debug("Bootstrap corr diff: mean=%.4f, CI=[%.4f, %.4f], p=%.4g", mean_diff, lo, hi, p_val)
    return {"r_diff_mean": mean_diff, "ci_low": lo, "ci_high": hi, "p": p_val}


def spearman_corr(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    """Spearman correlation r and p-value."""
    r, p = stats.spearmanr(x, y)
    return float(r), float(p)


def mean_ci_t(data: np.ndarray, confidence: float = 0.95) -> Tuple[float, float, float]:
    """Mean and t-based confidence interval (two-sided)."""
    data = np.asarray(data)
    mean = float(np.mean(data))
    ci_low, ci_high = stats.t.interval(confidence=confidence, df=len(data)-1, loc=mean, scale=stats.sem(data))
    return mean, float(ci_low), float(ci_high)

