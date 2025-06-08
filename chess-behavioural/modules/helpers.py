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
