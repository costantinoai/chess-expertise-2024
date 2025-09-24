#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 18 16:02:16 2025

@author: costantino_ai
"""

import os, pickle
import pandas as pd
import numpy as np
from scipy import stats
from statsmodels.stats.multitest import multipletests
from common.stats_utils import fdr_correction
from common.logging_utils import setup_logging
from modules import (
    logging,
    MVPA_ROOT_PATH,
    EXPERT_SUBJECTS,
    NONEXPERT_SUBJECTS,
    CONTRAST_MAP,
    MULTI_CORRECTION,
    MANAGER,
    REGIONS_LABELS
)

from modules.helpers import load_all_data, create_run_id, save_script_to_file
from pingouin import ttest  # pip install pingouin

def perform_ttest(
    values: np.ndarray, chance_level: np.float64, alternative="greater"
) -> dict:
    """
    Perform a one-sample t-test on a distribution of (accuracy-chance) values,
    i.e., we test if the mean is significantly different from 0.

    Parameters
    ----------
    values : np.ndarray
        1D array of accuracy minus chance values for multiple subjects.

    Returns
    -------
    result : dict
        Contains:
            "mean"       : float, average of the values
            "sem"        : float, standard error of the mean
            "t_stat"     : float, t-statistic from the t-test
            "p_value"    : float, p-value from the t-test
            "ci_low"     : float, lower bound of the 95% confidence interval
            "ci_high"    : float, upper bound of the 95% confidence interval
            "effect_size": float, Cohen's d
    """


    values = np.array(values)-chance_level

    # Perform the t-test
    ttest_result = stats.ttest_1samp(
        values, 0.0, nan_policy="omit", alternative=alternative
    )

    mean_val = np.nanmean(values)
    sem = stats.sem(values, nan_policy="omit")
    n = len(values[~np.isnan(values)])  # Count valid samples

    # Calculate the 95% confidence interval (parametric)
    if n > 1:

        ci_low, ci_high = ttest_result.confidence_interval()

    else:
        ci_low, ci_high = (np.nan, np.nan)

    # Return results
    result = {
        "mean-chance": mean_val,
        "sem": sem,
        "t_stat": ttest_result.statistic,
        "p_uncorrected": ttest_result.pvalue,
        "dof": ttest_result.df,
        "ci_low": ci_low,
        "ci_high": ci_high,
        # "effect_size": effect_size,
        "n": n,
        "chance": chance_level
    }

    return result


def run_ttest_analysis(
    levels=["region", "cortex", "lobe"],
    analysis_subdirs=["svm", "rsa_corr", "rsa_glm"],
    mvpa_root_path=MVPA_ROOT_PATH,
    out_root=None,
    twotailed=True
):
    """
    Runs a one-sample t-test analysis against chance level for each combination of:
    - analysis sub-directory (analysis_subdirs)
    - expertise level (expertise_values)
    - MVPA level (levels)
    - target regressor (found in df_all["target"].unique())
    - hemisphere (left, right)

    For each combination, this function:
        1. Loads and merges all relevant subject data into a single pandas DataFrame (df_all).
        2. Slices this DataFrame by each unique target regressor.
        3. Further slices by expertise level.
        4. Separates the data by hemisphere to perform a one-sample t-test against chance.
           NOTE: Although the function was originally written to handle both hemispheres,
                 the current data only includes the left hemisphere columns (which
                 actually reflect bilateral data). We therefore now process only
                 the 'l_' columns and skip any loop over 'r_'.
        5. Concatenates left and right hemisphere statistics (hemis_df), writes them to CSV,
           and returns all of these results in a dictionary.

    The final dictionary has the following nested structure:
        results_dict[analysis_subdir][level][target_regressor][expertise] = hemis_df

    Where hemis_df is a pandas DataFrame containing the following columns:
        'mean', 'sem', 't_stat', 'p_value', 'ci_low', 'ci_high',
        'effect_size', 'n', 'p_fdr', 'reject_fdr'
    with one row per ROI.

    Parameters
    ----------
    levels : list of str
        A list of "levels" in the MVPA or RSA analysis (e.g., 'roi-level', 'searchlight-level', etc.).
    expertise_values : list of str
        A list of expertise groups (e.g., ['expert', 'nonexpert']).
    analysis_subdirs : list of str
        A list of analysis sub-directories, each potentially containing different analyses
        (e.g., 'rsa_regression', 'svm_accuracy', etc.).

    Returns
    -------
    dict
        A nested dictionary containing hemispheric t-test results for each combination of
        (analysis_subdir, level, target_regressor, expertise). Each entry is a DataFrame
        of statistics for all ROIs in both hemispheres.

    Notes
    -----
    - This function uses global variables, including:
        - MVPA_ROOT_PATH: The root path for MVPA results.
        - EXPERT_SUBJECTS, NONEXPERT_SUBJECTS: Lists of subject IDs belonging to expert and non-expert groups.
        - CONTRAST_MAP: A dictionary mapping each target regressor to its chance level, e.g.:
            CONTRAST_MAP[target_regressor.capitalize()]['chance'] -> float
        - The function create_run_id() to generate a timestamped run ID string.
        - The function load_all_data() to load and merge subject data into a DataFrame.
        - The function perform_ttest() to execute the t-tests.
        - The function multipletests() for FDR correction.
    - The final plot (outside this function) should group by expertise, then by regressor, and compare hemispheres.
    - The large docstrings in this code are intentionally verbose to clarify the entire workflow.
    """

    if out_root is None:
        out_root = os.path.join(
            os.path.dirname(mvpa_root_path), f"{create_run_id()}_mvpa-group-stats"
        )

    os.makedirs(out_root, exist_ok=True)
    out_text_file = os.path.join(out_root, 'mvpa_logs_ttest.log')
    # Save a copy of the calling script for provenance
    save_script_to_file(out_root)
    setup_logging(log_file=out_text_file)

        # Initialize a nested dictionary to store all results
        results_dict = {}

        for analysis_subdir in analysis_subdirs:
            logging.info(f"Starting analysis for sub-directory: {analysis_subdir}")

            # Construct the path to the analysis sub-directory
            analysis_root = os.path.join(mvpa_root_path, analysis_subdir)
            logging.debug(f"Analysis root path: {analysis_root}")

            for level in levels: # cortex, region
                logging.info(
                    f"Processing level: {level} in analysis sub-directory: {analysis_subdir}"
                )

                # Create the level-specific folder where we store group results.
                # For instance: {analysis_root}/group/{level}/
                level_outdir = os.path.join(out_root, "group", level)
                os.makedirs(level_outdir, exist_ok=True)
                logging.debug(f"Output directory created (if not existing): {level_outdir}")

                # Load & Merge All Subject Data into one long DataFrame.
                # This DataFrame will contain data for all subjects and for all ROIs in a specific MVPA "level".
                # Example shape: (n_subjects, number_of_ROIs + meta_columns)
                df_all = load_all_data(
                    root_path=analysis_root,
                    expert_subjects=EXPERT_SUBJECTS,
                    nonexpert_subjects=NONEXPERT_SUBJECTS,
                    level=level,
                )
                logging.info(
                    f"Loaded data for level '{level}'. DataFrame shape: {df_all.shape}"
                )
                logging.debug(f"DataFrame columns: {df_all.columns.tolist()}")

                # Save the big merged DataFrame for all subjects.
                # This is useful for record-keeping or potential later analysis.
                df_filename = f"all-subjects_analysis-{analysis_subdir}_level-{level}.csv"
                df_all.to_csv(os.path.join(level_outdir, df_filename), index=False)
                logging.info(f"Saved merged DataFrame to {df_filename}")

                # We slice the DataFrame by each unique "target" regressor.
                # Each regressor might represent a particular condition in an RSA or SVM analysis.
                regressor_names = tuple(df_all["target"].unique())
                logging.debug(f"Unique target regressors found: {regressor_names}")

                # Initialize the sub-structure in the results dictionary for this analysis_subdir and level
                if analysis_subdir not in results_dict:
                    results_dict[analysis_subdir] = {}
                if level not in results_dict[analysis_subdir]:
                    results_dict[analysis_subdir][level] = {}

                for target_regressor in regressor_names:
                    logging.info(f"Analyzing target regressor: {target_regressor}")

                    # Slice the dataframe where the "target" column == current target_regressor
                    df_regressor = df_all[df_all["target"] == target_regressor]
                    logging.debug(
                        f"Sliced DataFrame for regressor '{target_regressor}'. Shape: {df_regressor.shape}"
                    )

                    # Prepare the nested structure for the current target_regressor
                    if target_regressor not in results_dict[analysis_subdir][level]:
                        results_dict[analysis_subdir][level][target_regressor] = {}

                    # We now look at each expertise level (e.g., 'expert', 'nonexpert')
                    for expertise in [True, False]:
                        logging.info(f"Analyzing expertise group: {expertise}")

                        # We slice to get participants matching the selected level of expertise
                        df_expertise = df_regressor[
                            df_regressor["expert"] == expertise
                        ].copy()

                        # The analysis should be done at the hemisphere level,
                        # but now we only have data in the "left" columns (l_*)—which
                        # effectively contain bilateral mask results.

                        # In typical naming, columns that start with 'l_' are left hemisphere,
                        # and 'r_' are right hemisphere. We handle them separately.
                        # However, we now only have 'l_' columns (which are effectively bilateral).
                        hemisphere = "L"  # we select only 'l' because results are already bilateral
                        logging.info(f"Processing hemisphere: {hemisphere}")

                        # Retrieve columns belonging to the given hemisphere.
                        # This step ensures we only pick ROI-like columns (e.g., l_roiX).
                        selected_roi_columns = [
                            col
                            for col in df_expertise.columns
                            if col.startswith(f"{hemisphere}_")
                        ]

                        # Instead of building an intermediate DataFrame with meta+ROI columns,
                        # we go straight for the ROI columns we need for analysis:
                        df_final = df_expertise[selected_roi_columns]
                        logging.debug(
                            f"Final DataFrame (only ROI columns) shape: {df_final.shape}"
                        )

                        # We can now run the analysis:
                        #    1) Loop over each ROI column (representing one ROI).
                        #    2) Extract the array of subject-level values.
                        #    3) Determine the chance level based on the regressor / analysis_subdir.
                        #    4) Perform a one-sample t-test against that chance level.
                        #    5) Accumulate the results in a dictionary.
                        results = {}

                        for selected_roi in selected_roi_columns:
                            # Get the group array (subjects measurements for this ROI)
                            group_vector = df_final[selected_roi].values

                            # Retrieve the chance level from the global CONTRAST_MAP
                            # e.g., CONTRAST_MAP[target_regressor.capitalize()]['chance']
                            chance_level = (
                                CONTRAST_MAP[target_regressor]["chance"]
                                if "svm" in analysis_subdir.lower()
                                else 0.0
                            )  # For RSA/corr, chance level is 0.0

                            # Perform a one-sample t-test against 0.0
                            # If it's an RSA-based analysis, we do "two-sided".
                            # If it's an SVM-based analysis, typically "greater".
                            stats_dict = perform_ttest(
                                group_vector,
                                chance_level=chance_level,
                                # alternative=(
                                #     "two-sided"
                                #     if "rsa" in analysis_subdir.lower()
                                #     else "greater"
                                # ),
                                alternative="two-sided" if ("rsa" in analysis_subdir.lower()) or (twotailed==True) else "greater"
                            )
                            results[selected_roi] = stats_dict

                            logging.debug(f"Performed t-tests for {selected_roi}: {stats_dict}")

                        # Convert the results dictionary into a DataFrame
                        stats_df = pd.DataFrame.from_dict(results, orient="index")

                        # Check for NaNs in "p_value"
                        if stats_df["p_uncorrected"].isna().any():
                            logging.error("NaN values detected in p_value column! Please check the data processing steps.")

                        # Optional: Drop rows with NaN p-values if needed
                        # stats_df = stats_df.dropna(subset=["p_value"])

                        # Apply FDR correction across all ROIs in this (left) hemisphere
                        uncorrected_p_values = stats_df["p_uncorrected"]
                        reject, corrected_p_values = fdr_correction(
                            uncorrected_p_values,
                            method=MULTI_CORRECTION,
                        )
                        logging.info(f"Applied FDR correction for hemisphere '{hemisphere}'.")
                        stats_df["p_corrected"] = corrected_p_values

                        # Assign regressor name to a new column so it's documented in the final DataFrame
                        stats_df["target"] = target_regressor

                        # Write the full hemis_df (only one hemisphere now) to a CSV file
                        df_fname = (
                            f"stats_regressor-{target_regressor}_"
                            f"level-{level}_"
                            f"expert-{expertise}.csv"
                        )
                        stats_df.T.to_csv(os.path.join(level_outdir, df_fname))
                        logging.info(f"Saved hemispheric stats DataFrame to {df_fname}")

                        # Save the current script in output dir for future reference
                        save_script_to_file(level_outdir)

                        # Finally, store hemis_df in the dictionary for easy programmatic retrieval
                        results_dict[analysis_subdir][level][target_regressor][expertise] = stats_df
                        logging.debug(
                            f"Stored hemispheric stats in results_dict under keys: "
                            f"analysis_subdir='{analysis_subdir}', level='{level}', "
                            f"target_regressor='{target_regressor}', expertise='{expertise}'."
                        )

                        # TODO: produce and save latex table
                        latex_table = generate_latex_apa_table(stats_df, caption="Statistical Results", label="tab:stats_results")

                        # Save LaTeX table as .txt
                        latex_txt_path = os.path.join(level_outdir, f"{df_fname}_latex.txt")
                        with open(latex_txt_path, "w") as f:
                            f.write(latex_table if latex_table is not None else "")
                        logging.info(f"Saved LaTeX table to {latex_txt_path}")

                    # After saving these DFs, they can be used later for plotting.
                    # A question arises about colorbar normalization in final plots:
                    #   - Do we want the same maximum for left and right hemisphere?
                    #   - Do we also want to unify across regressors or expertise levels?
                    # The approach typically depends on whether the focus is on comparing
                    # hemispheres, regressors, or expertise levels. Normalizing colorbars
                    # across certain dimensions can highlight relevant differences but may
                    # also obscure smaller signals.

        # Return the nested dictionary containing all t-test results
        ttest_results_path = os.path.join(out_root, f"{create_run_id()}_results_dict.pkl")
        with open(ttest_results_path, "wb") as f:
            pickle.dump(results_dict, f)

        logging.info(
            f"Completed all analyses. Returning results path '{ttest_results_path}'."
        )
    return ttest_results_path

def generate_latex_apa_table(
    df,
    alpha=0.05,
    use_corrected_p=True,
    caption="APA-Style T-Test Results",
    label="tab:apa_ttest",
):
    """
    Generate a LaTeX table summarizing significant statistical results from `df`,
    formatted in an APA-like style.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with columns at least including:
            ['mean-chance', 'sem', 't_stat', 'dof', 'p_uncorrected', 'p_corrected',
             'ci_low', 'ci_high', 'effect_size'].
        Rows are indexed by ROI (or contain an ROI label).
    alpha : float, optional
        Significance level (default=0.05).
    use_corrected_p : bool, optional
        If True, uses `p_corrected`; otherwise uses `p_uncorrected` (default=False).
    caption : str, optional
        Caption for the LaTeX table.
    label : str, optional
        Label for referencing the table in LaTeX.

    Returns
    -------
    None
        Prints the LaTeX table to stdout.
    """
    # Select which p-value column to use
    p_col = "p_corrected" if use_corrected_p else "p_uncorrected"

    # Filter to only significant rows
    df_sig = df[df[p_col] < alpha].copy()
    if df_sig.empty:
        logging.info(f"No significant results found at alpha={alpha}.")
        return ""

    # Build table header
    table_header = """
    \\begin{table}[htbp]
    \\centering
    \\caption{%s}
    \\label{%s}
    \\begin{tabular}{lccccccc}
    \\hline
    ROI & Mean Diff & SEM & 95\\%% CI & $t$ & df & $d$ & $p$ \\\\
    \\hline
    """ % (caption, label)

    table_rows = []
    for roi, row in df_sig.iterrows():
        # Determine significance (stars) based on chosen alpha
        p_value = row[p_col]
        if p_value < 0.001:
            significance = r"$^{***}$"
        elif p_value < 0.01:
            significance = r"$^{**}$"
        elif p_value < 0.05:
            significance = r"$^{*}$"
        else:
            significance = ""

        ci_str = f"[{row['ci_low']:.3f}, {row['ci_high']:.3f}]"

        roi_name_pretty = REGIONS_LABELS[MANAGER.get_by_filter(hemisphere='L', region=roi[2:].replace('_ROI', ''))[0][0].region_id][1] if "_ROI" in roi else roi

        row_str = (
            f"{roi_name_pretty} & "
            f"{row['mean-chance']:.4f}{significance} & "  # Mean difference from chance
            f"{row['sem']:.4f} & "
            f"{ci_str} & "
            f"{row['t_stat']:.3f} & "
            f"{row['dof']:.0f} & "  # Degrees of freedom as integer
            f"{row['effect_size']:.3f} & "
            f"{p_value:.4g} \\\\"
        )
        table_rows.append(row_str)

    # Build table footer
    table_footer = """
    \\hline
    \\multicolumn{8}{l}{\\textsuperscript{*}$p<0.05$, \\textsuperscript{**}$p<0.01$, \\textsuperscript{***}$p<0.001$} \\\\
    \\end{tabular}
    \\end{table}
    """

    # Combine and log
    full_table = table_header + "\n".join(table_rows) + table_footer
    logging.info("\n%s", full_table)
    return full_table

def generate_results_paragraph(stats_results):
    """
    Generate a textual summary of the statistical results.

    Parameters:
    - stats_results (list): List of dictionaries containing:
        - 'group': The group/category (e.g., ROI).
        - 'hue': The hue/label (e.g., Face, Vehicle).
        - 'mean': Mean accuracy.
        - 'sem': Standard error of the mean.
        - 't_stat': t-statistic.
        - 'p_value': p-value.

    # Example Input:
    stats_results = [
        {'group': 'FFA', 'hue': 'Face', 'mean': 0.4958, 'sem': 0.0406, 't_stat': 6.056, 'p_value': 1.774e-6, 'error_95': 0.0796},
        {'group': 'FFA', 'hue': 'Vehicle', 'mean': 0.4958, 'sem': 0.0419, 't_stat': 5.866, 'p_value': 2.801e-6, 'error_95': 0.0821},
        {'group': 'Foveal 0.5\degree', 'hue': 'Face', 'mean': 0.2708, 'sem': 0.0415, 't_stat': 0.502, 'p_value': 0.31, 'error_95': 0.0813}
    ]

    Returns:
    - str: A formatted paragraph summarizing the results.
    """
    paragraph = "Statistical analysis revealed the following results:\n\n"
    for res in stats_results:
        significance = "NOT statistically significant"
        if res["p_value"] < 0.05:
            significance = "statistically SIGNIFICANT"

        paragraph += (
            f"For group '{res['group']}', the mean accuracy was "
            f"{res['mean']:.2f} (SEM = {res['sem']:.2f}). A one-sample t-test against chance level "
            f"yielded t({len(stats_results) - 1}) = {res['t_stat']:.2f}, p = {res['p_value']:.3g}, "
            f"indicating that the result was {significance}.\n\n"
        )
    logging.info("\n%s", paragraph)
    return paragraph
