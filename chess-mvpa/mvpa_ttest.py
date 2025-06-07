import os
import glob
from natsort import natsorted
import pandas as pd
import pingouin as pg
import numpy as np
import warnings
import pickle
from collections import defaultdict
from modules import EXPERT_SUBJECTS
from modules.helpers import create_run_id
import logging

# ------------------------------------------------------------------------------
# 2) HELPER FUNCTION: ORGANIZE DATAFRAME BY TARGET & EXPERTISE
# ------------------------------------------------------------------------------
def organize_data_by_target_and_expertise(df: pd.DataFrame) -> dict:
    """
    Organizes a flat dataframe into a nested dict:
        target -> expert_status -> DataFrame

    Specifically:
      - Groups rows by 'target'
      - Within each 'target', splits data by 'expert' (True/False)
      - Drops metadata columns ('subject', 'expert', 'target') so remaining columns are only ROIs

    Returns:
        data_dict[target][expert_status] = DataFrame of ROI columns
    """
    data_dict = defaultdict(dict)

    # Group by the 'target' column (e.g., "checkmate", "categories", etc.)
    for target, df_target in df.groupby("target"):

        # Within each target group, further group by 'expert' status (True/False)
        for expert_status, df_group in df_target.groupby("expert"):

            # Remove the columns 'subject', 'expert', 'target' to keep only ROI columns
            data_dict[target][expert_status] = df_group.drop(columns=["subject", "expert", "target"])

    return data_dict

# ------------------------------------------------------------------------------
# 3) HELPER FUNCTION: LOAD AND CLEAN A SINGLE SUBJECT
# ------------------------------------------------------------------------------
def load_mvpa_subject(filepath: str) -> pd.DataFrame:
    """
    Loads a single subject's MVPA .tsv file and appends 'subject' + 'expert' metadata.
    Additionally:
      - Warns if some (but not all) ROI values are NaN in a given row.
      - Drops rows where ALL ROI values are NaN.

    Parameters:
        filepath (str): Full path to the subject's TSV file.

    Returns:
        pd.DataFrame: Cleaned DataFrame with columns:
            ['subject', 'expert', 'target', ROI_1, ..., ROI_n]
    """
    # Load the TSV file
    df = pd.read_csv(filepath, sep='\t')

    # Extract subject ID from the parent folder name:
    #   e.g., "/.../sub-12/mvpa_cv.tsv" → subject_folder="sub-12" → subject_id="12"
    subject_folder = os.path.basename(os.path.dirname(filepath))  # e.g., "sub-12"
    subject_id = subject_folder.split('-')[1]                    # e.g., "12"

    # Insert 'subject' (string ID) and 'expert' (True/False) columns at the front
    df.insert(0, 'subject', subject_id)
    df.insert(1, 'expert', subject_id in EXPERT_SUBJECTS)

    # Identify all ROI columns by excluding known metadata columns
    roi_columns = df.columns.difference(['subject', 'expert', 'target'])

    # Check if a row has some but not all ROI columns as NaN
    some_nan_mask = df[roi_columns].isna().any(axis=1)

    # Warn if partial NaNs are present in any row
    if some_nan_mask.any():
        warnings.warn(
            f"{some_nan_mask.sum()} row(s) in {filepath} contain some NaN ROI values.",
            RuntimeWarning
        )

    return df

def run_stats_on_df(df: pd.DataFrame, method: str) -> dict:
    """
    Runs expert vs. nonexpert stats for all regressors in the given DataFrame.

    Steps:
      1. Group data by 'target' → expert status → ROI columns.
      2. For each 'target', separate experts vs. nonexperts.
      3. Decide chance level (usually 0.0, or from CONTRAST_MAP if method == 'svm').
      4. Subtract chance from each ROI value (demeaning).
      5. Perform:
         a) One-tailed t-test (experts vs. chance)
         b) One-tailed t-test (nonexperts vs. chance)
         c) Two-tailed t-test (experts vs. nonexperts)
      6. Collect p-values, apply FDR correction within each target across all ROIs.
      7. Return a nested results dict keyed by:
         results[comparison][target][roi] = stats

    Parameters:
        df (pd.DataFrame): DataFrame with columns:
            ['subject', 'expert', 'target', ROI_1, ..., ROI_n]
        method (str): Name of the MVPA method (e.g., 'svm', 'rsa_corr'),
                      used here to set the chance level for SVM.

    Returns:
        dict: Nested structure of the form:
              {
                comparison_key : {
                  target : {
                    roi : {
                      "mean_diff": ...,
                      "t": ...,
                      "p_uncorrected": ...,
                      "p_fdr": ...,
                      ...
                    }
                  }
                }
              }
    """
    logger.info("Entering run_stats_on_df...")

    results = {}  # Will store final stats

    # 1. Build a nested dict: target -> { True: df_experts, False: df_nonexperts }
    logger.debug("Organizing data by target and expertise...")
    data_dict = organize_data_by_target_and_expertise(df)

    # Iterate over each target (e.g., "checkmate", "categories", "stimuli", etc.)
    for target, expertise_dict in data_dict.items():
        if target not in CONTRAST_MAP.keys():
            continue
        logger.info(f"Processing target '{target}'...")

        # Get DataFrame of expert rows
        experts_df = expertise_dict.get(True, pd.DataFrame()).drop(columns=["subject"], errors="ignore")
        # Get DataFrame of nonexpert rows
        nonexperts_df = expertise_dict.get(False, pd.DataFrame()).drop(columns=["subject"], errors="ignore")

        # Assert same shape
        assert experts_df.shape == nonexperts_df.shape, (
            f"Shape mismatch in target '{target}': "
            f"experts shape = {experts_df.shape}, nonexperts shape = {nonexperts_df.shape}"
        )

        # If either group is missing data, skip
        if experts_df.empty or nonexperts_df.empty:
            logger.warning(f"Missing expert/nonexpert data for target '{target}'. Skipping.")
            warnings.warn(f"Missing expert/nonexpert data for target '{target}'. Skipping.")
            continue

        # 2. Decide chance level:
        #    If method includes 'svm', read from CONTRAST_MAP[target]["chance"],
        #    otherwise default to 0.0 (useful for RSA/correlation).
        chance_level = (
            CONTRAST_MAP[target]["chance"] if "svm" in method.lower() else 0.0
        )
        logger.debug(f"Chance level for target '{target}': {chance_level}")

        # 3. Subtract the chance level from each ROI's values
        logger.debug(f"Demeaning experts and nonexperts data by chance level {chance_level}...")
        experts_demeaned = experts_df - chance_level
        nonexperts_demeaned = nonexperts_df - chance_level

        # Prepare lists to store p-values for each comparison across ROIs
        pvals_exp = []      # experts vs. chance
        pvals_nonexp = []   # nonexperts vs. chance
        pvals_diff = []     # experts vs. nonexperts

        # Temporary structure to hold raw stats per ROI
        results_per_roi = {}

        # For each ROI column, run the t-tests and store results
        logger.debug("Running t-tests for each ROI...")
        for roi in experts_demeaned.columns:
            logger.debug(f"ROI: {roi}")

            # Extract numeric arrays (dropping any NaNs)
            x_exp = experts_demeaned[roi].dropna().values
            x_nonexp = nonexperts_demeaned[roi].dropna().values

            logger.debug(f"Shapes -> Experts: {x_exp.shape}, Nonexperts: {x_nonexp.shape}")

            # a) One-tailed test: experts vs. chance
            ttest_exp = pg.ttest(x_exp, 0, alternative="two-sided", correction=False)

            # b) One-tailed test: nonexperts vs. chance
            ttest_nonexp = pg.ttest(x_nonexp, 0, alternative="two-sided", correction=False)

            # c) Two-tailed test: experts vs. nonexperts
            ttest_diff = pg.ttest(x_exp, x_nonexp, alternative="two-sided")

            # Save uncorrected p-values for FDR
            pvals_exp.append(ttest_exp["p-val"].iloc[0])
            pvals_nonexp.append(ttest_nonexp["p-val"].iloc[0])
            pvals_diff.append(ttest_diff["p-val"].iloc[0])

            # Collate raw stats for each of the three comparisons
            results_per_roi[roi] = {
                "experts_vs_chance": {
                    "mean_diff": np.mean(x_exp),
                    "std": np.std(x_exp, ddof=1),
                    "sem": np.std(x_exp, ddof=1) / np.sqrt(len(x_exp)),
                    "t": ttest_exp["T"].iloc[0],
                    "p_uncorrected": ttest_exp["p-val"].iloc[0],
                    "dof": ttest_exp["dof"].iloc[0],
                    "cohen_d": ttest_exp["cohen-d"].iloc[0],
                    "CI95": ttest_exp["CI95%"].iloc[0],
                    "BF10": ttest_exp["BF10"].iloc[0],
                    "chance": chance_level,
                },
                "nonexperts_vs_chance": {
                    "mean_diff": np.mean(x_nonexp),
                    "std": np.std(x_nonexp, ddof=1),
                    "sem": np.std(x_nonexp, ddof=1) / np.sqrt(len(x_nonexp)),
                    "t": ttest_nonexp["T"].iloc[0],
                    "p_uncorrected": ttest_nonexp["p-val"].iloc[0],
                    "dof": ttest_nonexp["dof"].iloc[0],
                    "cohen_d": ttest_nonexp["cohen-d"].iloc[0],
                    "CI95": ttest_nonexp["CI95%"].iloc[0],
                    "BF10": ttest_nonexp["BF10"].iloc[0],
                    "chance": chance_level,
                },
                "experts_vs_nonexperts": {
                    "mean_diff": np.mean(x_exp) - np.mean(x_nonexp),
                    "std": np.std(x_exp - x_nonexp, ddof=1),
                    "sem": np.std(x_exp - x_nonexp, ddof=1) / np.sqrt(len(x_nonexp)),
                    "t": ttest_diff["T"].iloc[0],
                    "p_uncorrected": ttest_diff["p-val"].iloc[0],
                    "dof": ttest_diff["dof"].iloc[0],
                    "cohen_d": ttest_diff["cohen-d"].iloc[0],
                    "CI95": ttest_diff["CI95%"].iloc[0],
                    "BF10": ttest_diff["BF10"].iloc[0],
                    "chance": chance_level,
                },
            }

        logger.debug("Applying FDR correction for all comparisons in this target...")
        # 4. FDR correction for each comparison within this target
        #    Because we collect one p-value per ROI for each comparison, we can correct them as a family
        reject_exp, pvals_exp_fdr = pg.multicomp(pvals_exp, method="fdr_bh")
        reject_nonexp, pvals_nonexp_fdr = pg.multicomp(pvals_nonexp, method="fdr_bh")
        reject_diff, pvals_diff_fdr = pg.multicomp(pvals_diff, method="fdr_bh")

        # 5. Compile final results into the 'results' dictionary
        #    comparisons = ["experts_vs_chance", "nonexperts_vs_chance", "experts_vs_nonexperts"]
        logger.debug("Compiling final results...")
        for comparison_key in ["experts_vs_chance", "nonexperts_vs_chance", "experts_vs_nonexperts"]:
            results.setdefault(comparison_key, {}).setdefault(target, {})

            # Go through each ROI in the same order they were processed
            for i, roi in enumerate(experts_demeaned.columns):
                p_fdr, fdr_reject = {
                    "experts_vs_chance": (pvals_exp_fdr[i], reject_exp[i]),
                    "nonexperts_vs_chance": (pvals_nonexp_fdr[i], reject_nonexp[i]),
                    "experts_vs_nonexperts": (pvals_diff_fdr[i], reject_diff[i]),
                }[comparison_key]

                # Merge final stats with p_fdr and fdr_reject
                results[comparison_key][target][roi] = {
                    **results_per_roi[roi][comparison_key],
                    "p_fdr": p_fdr,
                    "fdr_reject": fdr_reject,
                }

    logger.info("Exiting run_stats_on_df.")
    return results

# Contrasts + Chance Levels in dictionary form
CONTRAST_MAP = {
    "categories": {"chance": 1 / 10},       # OLD: 40 stim, Felipe strategies
    "categories_half": {"chance": 1 / 5},   # BILALIC: these are our old "categories" but for 20 stim
    "checkmate": {"chance": 1 / 2},         # OLD: 40 stim, check vs no-check
    "check_n_half": {"chance": 1 / 3},           # BILALIC: 1,3,4
    "checkmate_piece_half": {"chance": 1 / 4},   # BILALIC
    "difficulty_half": {"chance": 1 / 2},        # BILALIC --> may be redundant
    "first_piece_half": {"chance": 1 / 4},       # BILALIC
    "legal_moves": {"chance": 1 / 19},
    # "legal_moves_half": {"chance": 1/ 15},
    "motif_half": {"chance": 1 / 4},             # BILALIC
    "side_half": {"chance": 1 / 2},              # BILALIC: ???
    # "stimuli": {"chance": 1 / 40},          # OLD: all our stimuli, one label per stim
    # "stimuli_half": {"chance": 1 / 20},     # BILALIC: Used for the checkmate stim only on bilalic cat
    "total_pieces": {"chance": 1 / 13},
    # "total_pieces_half": {"chance": 1 / 12},
    "visualStimuli": {"chance": 1 / 20},    # OLD: 20 pairs of stimuli
}


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
pg.options["round.column.CI95%"] = 6

# ------------------------------------------------------------------------------
# 5) MAIN SCRIPT
# ------------------------------------------------------------------------------
mvpa_roots = [
    # "/data/projects/chess/data/BIDS/derivatives/mvpa/20250402-191833_glasser_cortices_bilateral",
    # "/data/projects/chess/data/BIDS/derivatives/mvpa/20250402-230003_glasser_regions_bilateral",
    # "/data/projects/chess/data/BIDS/derivatives/mvpa/20250402-191243_bilalic_sphere_rois",
    "/data/projects/chess/data/BIDS/derivatives/mvpa-lda/20250425-183118_glasser_regions_bilateral",
    "/data/projects/chess/data/BIDS/derivatives/mvpa-lda/20250425-175450_bilalic_sphere_rois"
    ]
mvpa_subdirs = ["svm", "rsa_corr"]
# mvpa_subdirs = ["rsa_corr_trained", "rsa_corr_untrained"]

for mvpa_root in mvpa_roots:
    # Loop over each MVPA subdirectory (analysis method)
    for mvpa_subdir in mvpa_subdirs:

        # Build the path: e.g., /data/.../20250402-191243_bilalic_sphere_rois/svm
        mvpa_dir = os.path.join(mvpa_root, mvpa_subdir)

        # Collect all TSV files under subfolders (e.g., sub-01/mvpa_cv.tsv, sub-02/mvpa_cv.tsv, etc.)
        subject_files = natsorted(glob.glob(os.path.join(mvpa_dir, "*/*.tsv")))

        # Read, clean, and concatenate all subject TSV files for this method
        df = pd.concat(
            [load_mvpa_subject(f) for f in subject_files],
            ignore_index=True
        )

        # Remove the stimuli matrices for RSA, since they would be meaningless
        # (0 in the diagonal and 1 everywhere else). Still useful for SVM
        if "rsa" in mvpa_subdir:
            # If 'layer' column exists (from RSA-DNN outputs), rename it to 'target'
            if "layer" in df.columns:
                df = df.rename(columns={"layer": "target"})

                # Drop irrelevant regressors
                df = df[~df["target"].isin(["stimuli", "stimuli_half"])]


        # Run the specialized analysis on this single method
        # 'method' argument helps pick the correct chance level for SVM
        analysis_results = run_stats_on_df(df, method=mvpa_subdir)

        # Here, 'analysis_results' is your final nested dictionary for that method.
        # You can print, save, or further process these stats as needed.
        print(f"Analysis complete for method: {mvpa_subdir}")

        results_out = os.path.join(mvpa_dir, f"{create_run_id()}_group")
        os.makedirs(results_out)

        pkl_fname = os.path.join(results_out, "ttest_group_results.pkl")

        with open(pkl_fname, "wb") as f:
            pickle.dump(analysis_results, f)

        print(f"Results saved in: {results_out}")


# ----------------------------------------------------------------------------
# Final structure of `analysis_results`:

# analysis_results = {
#     <comparison>: {
#         <regressor_name>: {
#             <roi>: {
#                 'mean', 'std', 'sem', 't', 'p_uncorrected', 'p_fdr', 'fdr_reject', ...
#             },
#             ...
#         },
#         ...
#     }
# }

# Where each "comparison" is one of:
#   'experts_vs_chance', 'nonexperts_vs_chance', or 'experts_vs_nonexperts'.
# ----------------------------------------------------------------------------
