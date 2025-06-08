#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 12 17:10:14 2025

@author: costantino_ai
"""
import os
import pandas as pd

from modules import logging
from common.helpers import (
    OutputLogger,
    create_run_id,
    create_output_directory,
    save_script_to_file,
    calculate_mean_and_ci,
)


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
        logging.error(
            f"An error occurred while creating the output directory: {e}",
            exc_info=True,
        )

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
