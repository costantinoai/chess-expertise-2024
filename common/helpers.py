import os
import sys
import inspect
import shutil
from datetime import datetime
from scipy import stats
import numpy as np
import logging

class OutputLogger:
    """Context manager to duplicate stdout/stderr to a log file."""
    def __init__(self, log: bool, file_path: str):
        self.log = log
        self.file_path = file_path
        self.original_stdout = sys.stdout
        self.original_stderr = sys.stderr

    def __enter__(self):
        if self.log:
            self.log_file = open(self.file_path, "w")
            sys.stdout = self
            sys.stderr = self
            for handler in logging.root.handlers:
                handler.stream = sys.stderr
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.log:
            for handler in logging.root.handlers:
                handler.stream = self.original_stderr
            sys.stdout = self.original_stdout
            sys.stderr = self.original_stderr
            self.log_file.close()

    def write(self, message: str):
        self.original_stdout.write(message)
        if self.log and not self.log_file.closed:
            self.log_file.write(message)

    def flush(self):
        self.original_stdout.flush()
        if self.log and not self.log_file.closed:
            self.log_file.flush()

def create_run_id() -> str:
    """Return a timestamp string for unique run directories."""
    return datetime.now().strftime("%Y%m%d-%H%M%S")

def create_output_directory(directory_path: str) -> None:
    """Create directory if it doesn't exist."""
    try:
        logging.debug(f"Attempting to create output directory at: {directory_path}")
        if not os.path.exists(directory_path):
            os.makedirs(directory_path)
            logging.info("Output directory created successfully.")
        else:
            logging.info("Output directory already exists.")
    except Exception as e:
        logging.error(f"An error occurred while creating the output directory: {e}", exc_info=True)

def save_script_to_file(output_directory: str) -> None:
    """Copy the calling script to output directory for reproducibility."""
    try:
        caller_frame = inspect.stack()[1]
        script_file = caller_frame.filename
        script_file_out = os.path.join(output_directory, os.path.basename(script_file))
        logging.debug(f"Attempting to save the script file to: {script_file_out}")
        shutil.copy(script_file, script_file_out)
        logging.info("Script file saved successfully.")
    except Exception as e:
        logging.error(f"An error occurred while saving the script file: {e}", exc_info=True)

def calculate_mean_and_ci(data):
    """Return mean and 95% confidence interval for an array-like."""
    data = np.asarray(data)
    mean = np.mean(data)
    ci_low, ci_high = stats.t.interval(0.95, len(data)-1, loc=mean, scale=stats.sem(data))
    return mean, ci_low, ci_high
