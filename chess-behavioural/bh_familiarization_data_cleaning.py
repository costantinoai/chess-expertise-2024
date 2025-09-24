#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  1 17:06:12 2024

@author: mualla
"""
import pandas as pd
import os
import logging

from modules import DATASET_CSV_PATH
from modules.chess import (
    convert_shorthand_to_long,
    get_all_moves_with_int_eval,
    check_accuracy
    )
from modules.helpers import (
    create_output_directory,
    save_script_to_file,
    create_run_id,
    )
from logging_utils import setup_logging


# Set parameters
data_path = 'data/sub'
out_root = './results'

# Create results dir and save script
out_dir = os.path.join(out_root, f'{create_run_id()}_familiarization-task-group')
create_output_directory(out_dir)
save_script_to_file(out_dir)
logging.info("Output folder created and script file saved")

out_text_file = os.path.join(out_dir, 'bh_data_cleaning.log')
setup_logging(log_file=out_text_file)

    # Verify the existence of paths
    if not os.path.exists(data_path):
        logging.warning(f'Data path {data_path} does not exist.')
        raise FileNotFoundError(f'Data path {data_path} does not exist.')

    if not os.path.exists(DATASET_CSV_PATH):
        logging.warning(f'Dataset path {DATASET_CSV_PATH} does not exist.')
        raise FileNotFoundError(f'Dataset path {DATASET_CSV_PATH} does not exist.')

    try:
        dataset_df = pd.read_csv(DATASET_CSV_PATH)
        logging.info('Dataset loaded successfully.')
    except Exception as e:
        logging.error(f'Error loading dataset: {e}')
        raise

    logging.debug('Listing CSV files in the data directory.')
    csv_files = sorted([f for f in os.listdir(data_path) if f.endswith('.csv')])

    all_participants_data = pd.DataFrame()

    # Define columns to be selected in the final DataFrame
    final_columns = [
        'sub_id', 'expert', 'sub_elo', 'board_name', 'check_board',
        'sub_resp', 'stim_id', 'recommended_moves_long', 'acc', 'fen', 'sub_resp_long',
    ]

    column_name_mapping = {
        'Subject': 'sub_id',
        'IsExpert': 'expert',
        'ELO': 'sub_elo',
        'Experiment_try': 'board_name',
        'Checkmate': 'check_board',
        'checksequence.text': 'sub_resp',
        'trials.thisIndex': 'stim_id',
        'correct': 'correct_resp',
    }

    # Iterate over each CSV file in the dataset
    for csv_file in csv_files:
        sub = csv_file.split('_')[0]
        logging.debug('.')
        logging.debug('.')
        logging.debug(f'Processing: {sub}')
        logging.debug('='*17)

        # Construct the full path to the CSV file
        file_path = os.path.join(data_path, csv_file)
        logging.debug(f'Loading participant data from {sub}.')

        # Load the participant data, apply basic cleaning and filter rows with necessary data
        participant_df = pd.read_csv(file_path)
        participant_df = (participant_df
                          # Strip whitespace from strings
                          .applymap(lambda x: x.strip() if isinstance(x, str) else x)
                          .dropna(subset=['Experiment_try']))  # Drop rows where 'Experiment_try' is missing

        # Make everything lowercase to avoid errors in matching stimuli across dataframes
        participant_df['Experiment_try'] = participant_df['Experiment_try'].str.lower()
        dataset_df['filename'] = dataset_df['filename'].str.lower()

        # Merge participant data with dataset to align board names and add correct moves
        logging.debug(f'Merging data for {csv_file}.')
        merged_df = pd.merge(dataset_df[['filename', 'fen', 'correct', 'stockfish_top_5', 'stockfish_eval']],
                             participant_df,
                             how='right',
                             left_on='filename',
                             right_on='Experiment_try').fillna('')  # Fill NaN values with empty strings

        # Rename columns according to the predefined mapping
        logging.debug(f'Renaming columns for {sub}.')
        merged_df.rename(columns=column_name_mapping, inplace=True)

        # Extract and set additional participant information from filename
        logging.debug(f'Extracting participant info for {sub}.')
        merged_df['sub_id'] = sub.split('-')[-1]  # Participant ID
        merged_df['sub_elo'] = int(csv_file.split('_')[2])  # ELO rating
        merged_df['expert'] = csv_file.split('_')[1].startswith('E')  # Expert status
        merged_df['check_board'] = merged_df['filename'].str.startswith('c')

        # Convert subject responses to long algebraic notation where applicable
        logging.debug(f'Converting moves to long notation for {sub}.')
        merged_df['sub_resp_long'] = merged_df.apply(lambda row: convert_shorthand_to_long(
            row['fen'], row['sub_resp'], row['stim_id']), axis=1)

        # Apply the revised function across the rows of the DataFrame to update the new column
        merged_df['recommended_moves'] = merged_df.apply(
            lambda row: get_all_moves_with_int_eval(row['stockfish_top_5'], row['stockfish_eval']), axis=1)
        merged_df['recommended_moves_long'] = merged_df.apply(lambda row: convert_shorthand_to_long(
            row['fen'], row['recommended_moves'], row['stim_id']), axis=1)

        # Calculate accuracy based on the long notation move matching the correct response
        logging.debug(f'Calculating accuracy for {sub}.')

        # Apply the function to each row to create the 'acc' column
        merged_df['acc'] = merged_df.apply(
            lambda row: check_accuracy(row['sub_resp_long'], row['recommended_moves_long']), axis=1)

        # Select only the relevant columns for the final DataFrame
        logging.debug(f'Selecting relevant columns for {sub}.')
        relevant_df = merged_df[final_columns]

        # Make sure we have all the stimuli (sanity check before concat)
        assert len(relevant_df) == len(dataset_df)

        # Append processed data to the main DataFrame
        logging.info(f'Appending processed data for {sub}.')
        all_participants_data = pd.concat(
            [all_participants_data, relevant_df], ignore_index=True)


    logging.info('Completed processing all files.')

    # Sort the DataFrame by 'sub_id' and 'stim_id'.
    all_participants_data.sort_values(by=['sub_id', 'stim_id'], inplace=True)

    # Specify the desired column order.
    desired_column_order = [
        'sub_id', 'expert', 'sub_elo', 'check_board', 'stim_id', 'fen', 'board_name',
        'sub_resp', 'sub_resp_long', 'recommended_moves_long', 'acc'
    ]

    # Rearrange the columns in the specified order.
    all_participants_data = all_participants_data[desired_column_order]

    # Save the combined data
    output_path = os.path.join(out_dir, 'fam_task_results.csv')
    all_participants_data.to_csv(output_path, index=False)
    logging.info(f'Saved combined data to {output_path}')
