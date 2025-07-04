#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 18 16:03:34 2025

@author: costantino_ai
"""
import inspect, os, shutil, random, sys
import numpy as np
from datetime import datetime
from pathlib import Path
import pandas as pd

from modules import logging, MANAGER, REGIONS_LABELS

class OutputLogger:
    """
    A context manager that logs output to a file and the console (stdout).

    Attributes:
      log (bool): Whether or not to log output to a file. If True, messages
          will be written to file_path.
      file_path (str): The file path where messages will be written if log is
          True.
      original_stdout (object): The original stdout stream, which will be
          restored when the context is exited.
      log_file (file object): The file object used to write output to the
          file_path.

    Methods:
      __enter__: Called when the context is entered. If log is True, opens the
          log file and redirects stdout to self.
      __exit__: Called when the context is exited. If log is True, restores
          stdout to its original state and closes the log file.
      write: Writes a message to stdout and the log file (if log is True).
      flush: Flushes the stdout and the log file (if log is True).
    """

    def __init__(self, log: bool, file_path: str):
        """
        Initializes the OutputLogger object.

        Args:
         log (bool): Whether or not to log output to a file. If True, messages
             will be written to file_path.
           file_path (str): The file path where messages will be written if log
               is True.
        """
        self.log = log
        self.file_path = file_path
        self.original_stdout = sys.stdout
        self.original_stderr = sys.stderr

    def __enter__(self):
        """
        Called when the context is entered. If log is True, opens the log file
        and redirects stdout to self.

        Returns:
          self: The OutputLogger object.
        """
        if self.log:
            self.log_file = open(
                self.file_path, "w"
            )  # open the log file for writing
            sys.stdout = self  # redirect stdout to this OutputLogger object
            sys.stderr = self
            # Reconfigure logging to use the new stderr
            for handler in logging.root.handlers:
                handler.stream = sys.stderr
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Called when the context is exited. If log is True, restores stdout to
        its original state and closes the log file.

        Args:
          exc_type (type): The exception type, if any.
          exc_val (Exception): The exception value, if any.
          exc_tb (traceback): The traceback object, if any.
        """
        if self.log:
            # Restore logging to use the original stderr
            for handler in logging.root.handlers:
                handler.stream = self.original_stderr
            sys.stdout = self.original_stdout
            sys.stderr = self.original_stderr
            self.log_file.close()

    def write(self, message: str):
        """
        Writes a message to stdout and the log file (if log is True).

        Args:
          message (str): The message to write.
        """
        self.original_stdout.write(message)  # write the message to the console
        if self.log and not self.log_file.closed:  # check if the file is not closed
            self.log_file.write(message)  # write the message to the log file

    def flush(self):
        """
        Flushes the stdout and the log file (if log is True).
        """
        self.original_stdout.flush()  # flush the console
        if self.log and not self.log_file.closed:  # check if the file is not closed
            self.log_file.flush()  # flush the log file

def set_rnd_seed(seed=42):
    # Set random seeds for reproducibility
    np.random.seed(seed)
    random.seed(seed)

def create_run_id() -> str:
    """
    Create a unique run ID based on the current timestamp.

    Returns:
        str: A string representing the current date and time in the format "YYYYMMDD-HHMMSS".
    """
    now = datetime.now()
    return now.strftime("%Y%m%d-%H%M%S")

def save_script_to_file(output_directory):
    """
    Save the calling script to a specified output directory.

    This function obtains the filename of the script that directly calls this function
    (i.e., the "caller frame") and copies that script to a target directory, providing
    reproducibility by capturing the exact code used in the analysis.

    Parameters
    ----------
    output_directory : str
        Path to the directory where the script file will be copied.

    Returns
    -------
    None
    """
    caller_frame = inspect.stack()[1]  # Stack frame of the caller
    script_file = caller_frame.filename
    script_file_out = os.path.join(output_directory, os.path.basename(script_file))
    shutil.copy(script_file, script_file_out)

def load_subject_data(
    root_path: Path, subject_list: list, is_expert: bool, level: str
) -> pd.DataFrame:
    """
    Load decoding accuracies from TSV files for a set of subjects, producing
    a DataFrame in wide format where each row = (subject, regressor), and
    columns = ROI decoding accuracies. Additionally includes columns for:
    'subject', 'expert', 'regressor'.

    Parameters
    ----------
    root_path : Path
        Path to the directory containing 'sub-XX' folders or CSV/TSV files.
    subject_list : list
        List of subject IDs (strings).
    is_expert : bool
        Whether these subjects are experts or not (populates the 'expert' column).
    contrasts_list : list of str
        The list of regressor names in the expected row order from each TSV.

    Returns
    -------
    df_out : pd.DataFrame
        DataFrame with columns:
          [ 'subject', 'expert', 'regressor', <ROI_1>, <ROI_2>, ..., <ROI_n> ].

        The DataFrame has (#subjects * #contrasts) rows.
    """
    all_rows = []

    for sub in subject_list:

        # Each subject is assumed to have exactly one TSV file, e.g., sub-03/*.tsv
        sub_path = Path(root_path, f"sub-{sub}")

        tsv_files = [file for file in list(sub_path.glob("*.tsv")) if level in file.name]
        if len(tsv_files) != 1:
            logging.warning(
                f"No or too many (>1) TSV found for sub-{sub}, level: {level} in {root_path}"
            )
            continue

        # Read the first and possibly only TSV
        tsv_file = tsv_files[0]

        # Load the TSV file into a dataframe
        df_tsv = pd.read_csv(tsv_file, sep="\t")

        # Check if the column "regressor" is present and rename it to "target"
        if "regressor" in df_tsv.columns:
            df_tsv.rename(columns={"regressor": "target"}, inplace=True)

        # We'll build row-dicts: each row merges 'subject', 'expert', 'regressor' with the ROI columns
        contrasts_list = tuple(df_tsv["target"].values)

        for i, regressor in enumerate(contrasts_list):

            row_data = df_tsv.iloc[i].to_dict()
            # Build the final row dict
            row_out = {
                "subject": sub,
                "expert": is_expert,
                "target": regressor,
            }

            # Merge with the ROI columns
            row_out.update(row_data)  # e.g., {'L_V1_ROI': 0.85, 'R_V1_ROI': 0.8, ...}
            all_rows.append(row_out)

    # Construct a single DataFrame for these subjects
    df_out = pd.DataFrame(all_rows)
    return df_out


def load_all_data(
    root_path: Path, expert_subjects: list, nonexpert_subjects: list, level: str
) -> pd.DataFrame:
    """
    Load data for expert & non-expert subjects, concatenate into one DataFrame.

    Returns
    -------
    df_all : pd.DataFrame
        A DataFrame containing all subjects (expert + non-expert), with columns:
        ['subject', 'expert', 'regressor', <ROI_1>, <ROI_2>, ..., <ROI_n>].
    """
    df_expert = load_subject_data(
        root_path=root_path, subject_list=expert_subjects, is_expert=True, level=level
    )

    df_nonexp = load_subject_data(
        root_path=root_path, subject_list=nonexpert_subjects, is_expert=False, level=level
    )

    df_all = pd.concat([df_expert, df_nonexp], axis=0).reset_index(drop=True)
    return df_all

def filter_significant_ROIs(sliced_dict, use_corrected_p_values, P_ALPHA, FDR_ALPHA):
    """
    Updates the 'mean' column in the dataframes within the given dictionary based on thresholding rules
    and removes rows where the significance condition is not met.

    Parameters:
        sliced_dict (dict): A dictionary where keys map to nested dictionaries containing dataframes.
        use_fdr (bool): A flag indicating whether to use FDR ('p_fdr') or p-value ('p_value') for thresholding.
        P_ALPHA (float): The alpha threshold for p-values.
        FDR_ALPHA (float): The alpha threshold for FDR values.

    Returns:
        dict: The updated dictionary with rows removed where the significance condition is not met.
    """
    current_max = 0.0
    max_y = 0.0
    has_negative = False

    significant_dict = {}

    for category, bool_dict in sliced_dict.items():
        significant_dict[category] = {}

        for expertise_bool, stats_df in bool_dict.items():
            # Choose appropriate thresholding column
            alpha = FDR_ALPHA if use_corrected_p_values else P_ALPHA
            p_col = "p_corrected" if use_corrected_p_values else "p_uncorrected"

            # Check if the required p-value column exists in the dataframe
            if p_col not in stats_df.columns:
                logging.warning(
                    f"Column '{p_col}' not found for category '{category}', expertise '{expertise_bool}'. "
                    "Skipping significance filtering for this group."
                )
                significant_dict[category][expertise_bool] = stats_df
                continue

            # Filter rows based on the significance condition
            significant_df = stats_df[stats_df[p_col] < alpha].copy()

            # If we found any significant rows, check for negatives and update maxima
            if not significant_df.empty:
                # Check if there's any negative mean-chance
                if (significant_df["mean-chance"] < 0).any():
                    has_negative = True

                # Update current_max with the maximum absolute value of 'mean-chance'
                local_max = np.nanmax(np.abs(significant_df["mean-chance"].values))
                if local_max > current_max:
                    current_max = local_max

                # Update max_y with the maximum absolute value of 'ci_high'
                local_max_y = np.nanmax(np.abs(significant_df["ci_high"].values))
                if local_max_y > max_y:
                    max_y = local_max_y


                sig_rois = significant_df.index

                if "cortex" in sig_rois[0]:
                    sig_rois_prettier = [sig_rois.values]
                else:
                    sig_rois_ids = [MANAGER.get_by_filter(hemisphere="l", region_name=roi)[0][0].region_id for roi in sig_rois]
                    sig_rois_prettier = [REGIONS_LABELS[roi_id-1] for roi_id in sig_rois_ids] if sig_rois_ids != [] else None
                    assert len(sig_rois) == len(sig_rois_ids)

                logging.info(
                    f"Significant ROIs detected | "
                    f"Regressor: {category}, "
                    f"Expertise: {expertise_bool}, "
                    f"ROIs: {sig_rois_prettier} "
                )

            # Update the dictionary with only the significant rows
            significant_dict[category][expertise_bool] = significant_df


    return significant_dict, current_max, has_negative, max_y
