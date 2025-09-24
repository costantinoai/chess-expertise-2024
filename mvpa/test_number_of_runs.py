#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Do Experts and Non-Experts differ in the number of fMRI runs?

This script:
  1) Loads SPM.mat per subject and infers the number of runs.
  2) Summarizes per-group run counts (Experts vs Non-Experts).
  3) Runs Levene's test for equality of variances (diagnostic).
  4) Chooses the appropriate independent-samples t-test:
       - Student's t-test if Levene p >= 0.05 (equal variances assumed).
       - Welch's t-test if Levene p < 0.05 (unequal variances).
  5) Reports the correct 95% CI for the mean difference for the chosen test.
  6) Reports effect sizes:
       - Equal variances: Cohen's d (pooled SD) + Hedges' g.
       - Unequal variances: Cohen's d_av (using average SD) + Hedges' g_av.
  7) Optionally runs Mann–Whitney U as a robustness check (non-parametric).
  8) Saves:
       - results/run_count_group_test/run_counts_per_subject.csv
       - results/run_count_group_test/run_count_group_test_report.txt
       - results/run_count_group_test/run_count_group_test_table.tex
"""

# ---------- Standard library imports ----------
import os  # filesystem paths and directory creation
import re  # regular expressions to parse SPM fields
import math  # sqrt and other math utilities
import logging  # status messages for reproducibility

# ---------- Third-party imports ----------
import numpy as np  # array handling and NaN-safe ops
import pandas as pd  # tabular data + CSV I/O
import scipy.io as sio  # loading SPM.mat (MATLAB format)
from scipy.stats import ttest_ind, mannwhitneyu, levene  # stats tests

# ---------- Configuration (edit to your project) ----------
from config import GLM_BASE_PATH
BASE_PATH = str(GLM_BASE_PATH)  # root of subject folders
SPM_FILENAME = "SPM.mat"  # SPM file name inside each subject's folder

# List of subject IDs for Experts (as strings without "sub-" prefix)
EXPERT_SUBJECTS = [
    "03","04","06","07","08","09","10","11","12","13","16","20","22","23","24","29","30","33","34","36"
]

# List of subject IDs for Non-Experts (as strings without "sub-" prefix)
NONEXPERT_SUBJECTS = [
    "01","02","15","17","18","19","21","25","26","27","28","32","35","37","39","40","41","42","43","44"
]

# Output directory for all artifacts created by this script
OUTPUT_DIR = "results/run_count_group_test"

# Toggle: also compute Mann–Whitney U as a robustness check
RUN_ROBUSTNESS_MANNWHITNEY = True

# ---------- Logging setup ----------
logger = logging.getLogger(__name__)  # create a logger scoped to this module
logging.basicConfig(                  # configure basic logging format and level
    level=logging.INFO,
    format="%(levelname)s: %(message)s"
)

# ---------- Helper functions ----------

def count_runs_from_spm(spm_path: str) -> int:
    """
    Infer the number of runs for a subject from SPM.mat.

    Preferred method:
      - If SPM.Sess exists -> number of sessions = number of runs.
    Fallback:
      - Parse SPM.xX.name entries for patterns like 'Sn(1)', 'Sn(2)', etc.

    Parameters
    ----------
    spm_path : str
        Absolute path to the subject's SPM.mat file.

    Returns
    -------
    int
        Number of runs inferred from the SPM design.

    Raises
    ------
    FileNotFoundError
        If SPM.mat does not exist at the provided path.
    ValueError
        If the number of runs cannot be inferred from the loaded structure.
    """
    if not os.path.isfile(spm_path):  # ensure the file exists before loading
        raise FileNotFoundError(f"SPM.mat not found: {spm_path}")

    spm = sio.loadmat(spm_path, struct_as_record=False, squeeze_me=True)["SPM"]  # load MATLAB struct

    # Preferred: use the sessions field if present
    if hasattr(spm, "Sess"):  # SPM.Sess contains one struct per run
        sess = spm.Sess
        if isinstance(sess, (list, np.ndarray)):  # multiple sessions → list/array
            return int(len(sess))  # number of sessions equals number of runs
        return 1  # if single session struct, assume one run

    # Fallback: parse design matrix regressor names (SPM.xX.name)
    if hasattr(spm, "xX") and hasattr(spm.xX, "name"):  # ensure names exist
        names = spm.xX.name  # could be list or numpy array of strings
        run_ids = set()  # collect unique run indices found
        pat = re.compile(r"Sn\((\d+)\)")  # pattern like 'Sn(3)' for run 3
        for nm in np.atleast_1d(names):  # normalize to array-like
            if isinstance(nm, str):  # only parse strings
                m = pat.search(nm)  # search for Sn(#)
                if m:
                    run_ids.add(int(m.group(1)))  # store run index
        if run_ids:
            return int(len(run_ids))  # number of unique run indices

    # If neither method worked, we cannot infer the number of runs
    raise ValueError(f"Could not infer number of runs from: {spm_path}")


def summarize(arr: np.ndarray) -> dict:
    """
    Compute basic NaN-safe descriptive stats for a 1D array.

    Parameters
    ----------
    arr : np.ndarray
        Vector of values (may contain NaNs).

    Returns
    -------
    dict
        Keys: n, mean, sd, median, min, max.
    """
    arr = np.asarray(arr, dtype=float)  # ensure numeric array
    arr = arr[~np.isnan(arr)]  # drop NaNs
    if arr.size == 0:  # handle empty arrays gracefully
        return dict(n=0, mean=np.nan, sd=np.nan, median=np.nan, min=np.nan, max=np.nan)
    return dict(
        n=int(arr.size),  # sample size
        mean=float(np.mean(arr)),  # arithmetic mean
        sd=float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0,  # sample SD
        median=float(np.median(arr)),  # median
        min=float(np.min(arr)),  # minimum
        max=float(np.max(arr)),  # maximum
    )


def cohens_d_pooled_equal_var(x: np.ndarray, y: np.ndarray) -> float:
    """
    Cohen's d using pooled SD (appropriate when variances are approximately equal).

    Parameters
    ----------
    x, y : np.ndarray
        Two independent samples.

    Returns
    -------
    float
        Cohen's d (pooled SD). Returns NaN if not enough data.
    """
    x = np.asarray(x, dtype=float)  # cast to float
    y = np.asarray(y, dtype=float)  # cast to float
    x = x[~np.isnan(x)]  # drop NaNs in x
    y = y[~np.isnan(y)]  # drop NaNs in y
    nx, ny = len(x), len(y)  # sample sizes
    if nx < 2 or ny < 2:  # need at least 2 per group for SD
        return np.nan
    sx2 = np.var(x, ddof=1)  # sample variance of x
    sy2 = np.var(y, ddof=1)  # sample variance of y
    sp2 = ((nx - 1) * sx2 + (ny - 1) * sy2) / (nx + ny - 2)  # pooled variance
    sp = math.sqrt(sp2)  # pooled SD
    if sp == 0:  # avoid division by zero
        return 0.0
    return float((np.mean(x) - np.mean(y)) / sp)  # standardized mean difference


def cohens_d_average_sd_unequal_var(x: np.ndarray, y: np.ndarray) -> float:
    """
    Cohen's d using the average SD (often denoted d_av), recommended when variances differ.

    Parameters
    ----------
    x, y : np.ndarray
        Two independent samples.

    Returns
    -------
    float
        d_av = (mean_x - mean_y) / sqrt( (sd_x^2 + sd_y^2) / 2 )
    """
    x = np.asarray(x, dtype=float)  # cast to float
    y = np.asarray(y, dtype=float)  # cast to float
    x = x[~np.isnan(x)]  # drop NaNs
    y = y[~np.isnan(y)]  # drop NaNs
    if len(x) < 2 or len(y) < 2:  # need at least 2 per group for SD
        return np.nan
    sx2 = np.var(x, ddof=1)  # sample variance of x
    sy2 = np.var(y, ddof=1)  # sample variance of y
    s_av = math.sqrt((sx2 + sy2) / 2.0)  # average SD
    if s_av == 0:  # protect against zero SD
        return 0.0
    return float((np.mean(x) - np.mean(y)) / s_av)  # standardized diff using average SD


def hedges_g_from_d(d: float, nx: int, ny: int) -> float:
    """
    Hedges' g bias correction applied to any Cohen's d-like estimator.

    Parameters
    ----------
    d : float
        Cohen's d (either pooled or average-SD variant).
    nx, ny : int
        Sample sizes of the two groups.

    Returns
    -------
    float
        Hedges' g (bias-corrected standardized mean difference).
    """
    if np.isnan(d):  # pass through NaN if d is undefined
        return np.nan
    # Small-sample correction factor J based on total df for equal-variance case.
    # As an approximation we use J with df = nx + ny - 2 also when using d_av.
    df = nx + ny - 2  # degrees of freedom used by pooled-SD t
    if df <= 0:  # guard against degenerate sizes
        return np.nan
    J = 1.0 - (3.0 / (4.0 * df - 1.0))  # Hedges' correction
    return float(J * d)  # unbiased effect size


def save_run_count_report_txt(
    s_exp: dict,
    s_nov: dict,
    test_name: str,
    t_stat: float,
    df_used: float,
    p_value: float,
    ci_low: float,
    ci_high: float,
    d_label: str,
    d_value: float,
    g_value: float,
    lev_stat: float,
    lev_p: float,
    mw_stat: float | None,
    mw_p: float | None,
    output_dir: str,
    filename: str = "run_count_group_test_report.txt",
) -> str:
    """
    Save a plain-text summary report of the analysis.

    Returns
    -------
    str
        Path to the saved report file.
    """
    report_path = os.path.join(output_dir, filename)  # target file path
    with open(report_path, "w") as f:  # open for writing
        f.write(f"Run count comparison ({test_name})\n")  # header with test used
        f.write("=" * 60 + "\n")  # underline
        f.write(  # experts line with summary stats
            f"Experts     (n={s_exp['n']}): mean={s_exp['mean']:.2f}, sd={s_exp['sd']:.2f}, "
            f"median={s_exp['median']:.2f}, range=[{s_exp['min']:.0f}, {s_exp['max']:.0f}]\n"
        )
        f.write(  # non-experts line with summary stats
            f"Non-Experts (n={s_nov['n']}): mean={s_nov['mean']:.2f}, sd={s_nov['sd']:.2f}, "
            f"median={s_nov['median']:.2f}, range=[{s_nov['min']:.0f}, {s_nov['max']:.0f}]\n"
        )
        f.write("-" * 60 + "\n")  # separator
        f.write("Levene's test for equal variances (diagnostic):\n")  # section title
        f.write(f"  W = {lev_stat:.3f}, p = {lev_p:.4f}\n")  # Levene results
        f.write(f"{test_name}:\n")  # chosen test name
        f.write(f"  t = {t_stat:.3f}, df = {df_used:.3f}, p = {p_value:.4f}\n")  # t, df, p
        f.write(  # 95% CI for the mean difference
            f"  95% CI of mean difference (Experts - Non-Experts): [{ci_low:.2f}, {ci_high:.2f}]\n"
        )
        f.write(  # effect sizes
            f"  {d_label} = {d_value:.3f}  (Hedges' g = {g_value:.3f})\n"
        )
        if mw_stat is not None and mw_p is not None:  # optionally include MWU
            f.write("-" * 60 + "\n")
            f.write("Mann–Whitney U (robustness check):\n")
            f.write(f"  U = {mw_stat:.1f}, p = {mw_p:.4f}\n")
    logger.info(f"Saved report: {report_path}")  # log path
    return report_path  # return for convenience


def save_run_count_latex_table(
    s_exp: dict,
    s_nov: dict,
    lev_stat: float,
    lev_p: float,
    test_name: str,
    t_stat: float,
    df_used: float,
    p_value: float,
    ci_low: float,
    ci_high: float,
    d_label: str,
    d_value: float,
    g_value: float,
    mw_stat: float | None,
    mw_p: float | None,
    output_dir: str,
    filename: str = "run_count_group_test_table.tex",
) -> str:
    """
    Save a multi-column LaTeX table summarizing the analysis.

    Returns
    -------
    str
        Path to the saved LaTeX file.
    """
    tex_path = os.path.join(output_dir, filename)  # output path

    # Build the LaTeX lines; we use booktabs-style rules and a compact layout
    lines = [
        "\\begin{table}[p]",
        "\\centering",
        "\\resizebox{0.9\\linewidth}{!}{%",
        "\\begin{tabular}{lcc}",
        "\\toprule",
        " & Experts & Non-Experts \\\\",
        "\\midrule",
        f"$n$ & {s_exp['n']} & {s_nov['n']} \\\\",
        f"Mean (SD) & {s_exp['mean']:.2f} ({s_exp['sd']:.2f}) & {s_nov['mean']:.2f} ({s_nov['sd']:.2f}) \\\\",
        f"Median & {s_exp['median']:.2f} & {s_nov['median']:.2f} \\\\",
        f"Range & [{s_exp['min']:.0f}, {s_exp['max']:.0f}] & [{s_nov['min']:.0f}, {s_nov['max']:.0f}] \\\\",
        "\\midrule",
        f"Levene’s test (var.~equality) & \\multicolumn{{2}}{{c}}{{W = {lev_stat:.3f}, $p$ = {lev_p:.4f}}} \\\\",
        f"{test_name} & \\multicolumn{{2}}{{c}}{{$t$ = {t_stat:.3f}, df = {df_used:.3f}, $p$ = {p_value:.4f}}} \\\\",
        f"$95\\%$ CI (ΔMean) & \\multicolumn{{2}}{{c}}{{[{ci_low:.2f}, {ci_high:.2f}]}} \\\\",
        f"Effect size & \\multicolumn{{2}}{{c}}{{{d_label} = {d_value:.3f}, Hedges’ $g$ = {g_value:.3f}}} \\\\",
    ]
    # Optionally add Mann–Whitney U line if requested/computed
    if mw_stat is not None and mw_p is not None:
        lines.append(f"Mann--Whitney $U$ & \\multicolumn{{2}}{{c}}{{$U$ = {mw_stat:.1f}, $p$ = {mw_p:.4f}}} \\\\")
    # Close out the LaTeX table environment
    lines += [
        "\\bottomrule",
        "\\end{tabular}",
        "}",
        "\\caption{Comparison of number of runs per subject between Experts and Non-Experts. "
        "Descriptives are shown for each group. Levene’s test informs whether equal variances "
        "can be assumed; the analysis then uses Student’s $t$ (equal variances) or Welch’s $t$ "
        "(unequal variances) accordingly. We report the corresponding 95\\% CI of the mean "
        "difference (Experts $-$ Non-Experts) and an appropriate standardized effect size.}",
        "\\label{tab:run_count_group_test}",
        "\\end{table}",
    ]

    latex_str = "\n".join(lines)  # join the lines into a single string
    with open(tex_path, "w") as f:  # write to file
        f.write(latex_str)
    logger.info(f"Saved LaTeX table: {tex_path}")  # log the save path
    return tex_path  # return for convenience


# ---------- Main analysis pipeline ----------

def main() -> None:
    """Orchestrate data loading, testing, and reporting."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)  # ensure output folder exists

    # Collect per-subject run counts for both groups
    records = []  # accumulate dicts: {'subject': id, 'group': 'Expert'/'NonExpert', 'runs': int}

    # Iterate over Experts and read their SPM.mat to count runs
    for sid in EXPERT_SUBJECTS:  # loop through expert subject IDs
        spm_path = os.path.join(BASE_PATH, f"sub-{sid}", "exp", SPM_FILENAME)  # path to SPM.mat
        try:
            n_runs = count_runs_from_spm(spm_path)  # infer number of runs
            records.append(dict(subject=sid, group="Expert", runs=n_runs))  # store result
        except Exception as e:  # robust to missing/corrupt files
            logger.warning(f"Expert {sid}: {e}")  # log the issue
            records.append(dict(subject=sid, group="Expert", runs=np.nan))  # keep row with NaN

    # Iterate over Non-Experts and read their SPM.mat to count runs
    for sid in NONEXPERT_SUBJECTS:  # loop through non-expert subject IDs
        spm_path = os.path.join(BASE_PATH, f"sub-{sid}", "exp", SPM_FILENAME)  # path to SPM.mat
        try:
            n_runs = count_runs_from_spm(spm_path)  # infer number of runs
            records.append(dict(subject=sid, group="NonExpert", runs=n_runs))  # store result
        except Exception as e:  # robust to missing/corrupt files
            logger.warning(f"NonExpert {sid}: {e}")  # log the issue
            records.append(dict(subject=sid, group="NonExpert", runs=np.nan))  # keep row with NaN

    # Build a DataFrame with all subjects and their run counts
    df = pd.DataFrame(records)  # convert records to table
    csv_path = os.path.join(OUTPUT_DIR, "run_counts_per_subject.csv")  # CSV output path
    df.to_csv(csv_path, index=False)  # save subject-level counts
    logger.info(f"Saved per-subject run counts: {csv_path}")  # log save location

    # Prepare clean arrays for analysis by dropping NaNs per group
    exp = df.loc[df.group == "Expert", "runs"].to_numpy(float)  # expert run counts
    nov = df.loc[df.group == "NonExpert", "runs"].to_numpy(float)  # non-expert run counts
    exp = exp[~np.isnan(exp)]  # drop NaNs in expert vector
    nov = nov[~np.isnan(nov)]  # drop NaNs in non-expert vector

    # Compute simple group descriptives for reporting
    s_exp = summarize(exp)  # dict with n/mean/sd/median/min/max for experts
    s_nov = summarize(nov)  # dict with n/mean/sd/median/min/max for non-experts

    # Run Levene's test to diagnose equality of variances
    lev = levene(exp, nov, center="mean")  # returns statistic and p-value
    lev_stat = float(lev.statistic)  # extract W statistic as float
    lev_p = float(lev.pvalue)  # extract p-value as float

    # Decide which t-test to use based on Levene's p-value
    if lev_p < 0.05:  # variances likely unequal → use Welch's t-test
        equal_var = False  # flag for scipy ttest_ind
        test_name = "Welch's t-test (unequal variances)"
    else:  # variances not significantly different → use Student's t-test
        equal_var = True  # flag for scipy ttest_ind
        test_name = "Student's t-test (equal variances)"

    # Run the chosen independent-samples t-test
    tt = ttest_ind(exp, nov, equal_var=equal_var)  # SciPy test result
    t_stat = float(tt.statistic)  # t statistic
    p_value = float(tt.pvalue)  # two-sided p-value
    df_used = float(tt.df)  # test-specific degrees of freedom (Welch df or pooled df)

    # Compute the 95% CI of the mean difference from the test result
    try:
        ci = tt.confidence_interval(confidence_level=0.95)  # SciPy 1.9+
        ci_low, ci_high = float(ci.low), float(ci.high)  # lower/upper bounds
    except Exception:  # older SciPy: gracefully degrade
        ci_low, ci_high = np.nan, np.nan  # indicate unavailable CI

    # Compute effect sizes appropriate to the variance assumption
    if equal_var:  # equal variances → pooled SD Cohen's d
        d = cohens_d_pooled_equal_var(exp, nov)  # pooled-SD standardized difference
        d_label = "Cohen's $d$ (pooled SD)"  # label for table/report
    else:  # unequal variances → average SD Cohen's d (d_av)
        d = cohens_d_average_sd_unequal_var(exp, nov)  # average-SD standardized difference
        d_label = "Cohen's $d_{av}$ (avg. SD)"  # label for table/report

    # Convert d to Hedges' g with small-sample bias correction
    g = hedges_g_from_d(d, len(exp), len(nov))  # bias-corrected effect size

    # Optionally run a non-parametric robustness check (two-sided)
    if RUN_ROBUSTNESS_MANNWHITNEY:  # check toggle
        try:
            mw = mannwhitneyu(exp, nov, alternative="two-sided")  # MWU test
            mw_stat, mw_p = float(mw.statistic), float(mw.pvalue)  # extract values
        except Exception as e:  # handle edge cases
            logger.warning(f"Mann–Whitney failed: {e}")  # log the error
            mw_stat, mw_p = np.nan, np.nan  # set NaNs on failure
    else:
        mw_stat, mw_p = None, None  # skip entirely (not shown in outputs)

    import logging
    logging.basicConfig(level=logging.INFO)
    # Log a concise summary to the console for quick inspection
    logging.info("\n" + "=" * 72)
    logging.info("Run count comparison (%s)", test_name)
    logging.info("=" * 72)
    logging.info(  # Experts descriptives
        f"Experts     (n={s_exp['n']}): mean={s_exp['mean']:.2f}, sd={s_exp['sd']:.2f}, "
        f"median={s_exp['median']:.2f}, range=[{s_exp['min']:.0f}, {s_exp['max']:.0f}]"
    )
    logging.info(  # Non-Experts descriptives
        f"Non-Experts (n={s_nov['n']}): mean={s_nov['mean']:.2f}, sd={s_nov['sd']:.2f}, "
        f"median={s_nov['median']:.2f}, range=[{s_nov['min']:.0f}, {s_nov['max']:.0f}]"
    )
    logging.info("-" * 72)
    logging.info("Levene's test for equal variances (diagnostic):")  # heading
    logging.info(f"  W = {lev_stat:.3f}, p = {lev_p:.4f}")  # Levene results
    logging.info("-" * 72)
    logging.info(f"{test_name}:")  # which t-test was used
    logging.info(f"  t = {t_stat:.3f}, df = {df_used:.3f}, p = {p_value:.4f}")  # t, df, p
    logging.info(  # CI for mean difference
        f"  95% CI of mean difference (Experts - Non-Experts): [{ci_low:.2f}, {ci_high:.2f}]"
    )
    logging.info(f"  {d_label} = {d:.3f}  (Hedges' g = {g:.3f})")  # effect sizes
    if mw_stat is not None and mw_p is not None:  # optional MWU
        logging.info("-" * 72)
        logging.info("Mann–Whitney U (robustness check):")
        logging.info(f"  U = {mw_stat:.1f}, p = {mw_p:.4f}")
    logging.info("=" * 72 + "\n")

    # Save a plain-text report with the same information
    save_run_count_report_txt(
        s_exp=s_exp,  # experts descriptives
        s_nov=s_nov,  # non-experts descriptives
        test_name=test_name,  # test label
        t_stat=t_stat,  # t-statistic
        df_used=df_used,  # degrees of freedom
        p_value=p_value,  # p-value
        ci_low=ci_low,  # CI lower bound
        ci_high=ci_high,  # CI upper bound
        d_label=d_label,  # effect size label
        d_value=d,  # Cohen's d (pooled or d_av)
        g_value=g,  # Hedges' g
        lev_stat=lev_stat,  # Levene W
        lev_p=lev_p,  # Levene p
        mw_stat=mw_stat,  # MWU statistic (or None)
        mw_p=mw_p,  # MWU p-value (or None)
        output_dir=OUTPUT_DIR,  # where to save
        filename="run_count_group_test_report.txt",  # file name
    )

    # Save a formatted LaTeX table for manuscript/supplement
    save_run_count_latex_table(
        s_exp=s_exp,  # experts descriptives
        s_nov=s_nov,  # non-experts descriptives
        lev_stat=lev_stat,  # Levene W
        lev_p=lev_p,  # Levene p
        test_name=test_name,  # test label
        t_stat=t_stat,  # t-statistic
        df_used=df_used,  # degrees of freedom
        p_value=p_value,  # p-value
        ci_low=ci_low,  # CI lower bound
        ci_high=ci_high,  # CI upper bound
        d_label=d_label,  # effect size label
        d_value=d,  # Cohen's d (pooled) or d_av
        g_value=g,  # Hedges' g
        mw_stat=mw_stat,  # MWU statistic (or None)
        mw_p=mw_p,  # MWU p-value (or None)
        output_dir=OUTPUT_DIR,  # where to save
        filename="run_count_group_test_table.tex",  # file name
    )


# ---------- Entrypoint ----------

if __name__ == "__main__":  # run only when called as a script
    main()  # execute the pipeline
