#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 16 15:39:31 2024

@author: costantino_ai
"""

import subprocess
import logging
import os

def run_python_scripts(directory):
    # List all files in the directory
    files = os.listdir(directory)
    # Filter to include only Python files
    python_files = [file for file in files if file.endswith('.py') and ("plot_HCP_" in file)]

    # Execute each Python file
    for script in python_files:
        script_path = os.path.join(directory, script)
        logging.info("Running %s...", script_path)
        try:
            # Run the script
            result = subprocess.run(['python', script_path], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            # Print the output and errors (if any)
            logging.info("Output: %s", result.stdout.decode())
        except subprocess.CalledProcessError as e:
            logging.error("Error running %s: %s", script_path, e)

# Specify the directory containing the Python scripts
script_directory = '/home/eik-tb/OneDrive_andreaivan.costantino@kuleuven.be/GitHub/chess-expertise-2024/fMRI/utils/plot_HCPMM1_results'
run_python_scripts(script_directory)
