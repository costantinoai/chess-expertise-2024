#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 12 17:10:14 2025

@author: costantino_ai
"""
import os
import sys
import inspect
import shutil
from datetime import datetime
from scipy import stats
import numpy as np
import pandas as pd

from modules import logging

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

def create_run_id():
    now = datetime.now()
    return now.strftime("%Y%m%d-%H%M%S")

def create_output_directory(directory_path):
    """
    Creates an output directory at the specified path.

    Parameters:
    - directory_path (str): The path where the output directory will be created.

    The function attempts to create a directory at the given path.
    It logs the process, indicating whether the directory creation was successful or if any error occurred.
    If the directory already exists, it will not be created again, and this will also be logged.
    """
    try:
        # Log the attempt to create the output directory
        logging.debug(f"Attempting to create output directory at: {directory_path}")

        # Check if directory already exists to avoid overwriting
        if not os.path.exists(directory_path):
            # Create the directory
            os.makedirs(directory_path)
            # Log the successful creation
            logging.info("Output directory created successfully.")
        else:
            # Log if the directory already exists
            logging.info("Output directory already exists.")
    except Exception as e:
        # Log any errors encountered during the directory creation
        logging.error(f"An error occurred while creating the output directory: {e}", exc_info=True)

def save_script_to_file(output_directory):
    """
    Saves the script file that is calling this function to the specified output directory.

    Parameters:
    - output_directory (str): The directory where the script file will be saved.

    This function automatically detects the script file that is executing this function
    and creates a copy of it in the output directory.
    It logs the process, indicating whether the saving was successful or if any error occurred.
    """
    try:
        # Get the frame of the caller to this function
        caller_frame = inspect.stack()[1]
        # Get the file name of the script that called this function
        script_file = caller_frame.filename

        # Construct the output file path
        script_file_out = os.path.join(output_directory, os.path.basename(script_file))

        # Log the attempt to save the script file
        logging.debug(f"Attempting to save the script file to: {script_file_out}")

        # Copy the script file to the output directory
        shutil.copy(script_file, script_file_out)

        # Log the successful save
        logging.info("Script file saved successfully.")
    except Exception as e:
        # Log any errors encountered during the saving process
        logging.error(f"An error occurred while saving the script file: {e}", exc_info=True)

# Function to calculate mean and 95% CI for a given dataset
def calculate_mean_and_ci(data):
    mean = np.mean(data)
    ci_lower, ci_upper = stats.t.interval(
        confidence=0.95, df=len(data)-1, loc=mean, scale=stats.sem(data))
    return mean, ci_lower, ci_upper  # mean and 95% confidence interval

def load_and_sort_data(dataset_path, net_responses_path, humans_response_path):
    """
    Load the datasets into Pandas DataFrames and sort them by 'stim_id'.

    Parameters:
        dataset_path (str): Path to the dataset CSV.
        net_responses_path (str): Path to the neural network responses CSV.
        humans_response_path (str): Path to the human responses CSV.

    Returns:
        tuple: Sorted DataFrames (dataset_df, net_df, humans_df).
    """
    dataset_df = pd.read_csv(dataset_path).sort_values(by='stim_id')
    net_df = pd.read_csv(net_responses_path).sort_values(by='stim_id')
    humans_df = pd.read_csv(humans_response_path).sort_values(by='stim_id')

    logging.info('Datasets loaded and sorted by stim_id.')
    return dataset_df, net_df, humans_df
