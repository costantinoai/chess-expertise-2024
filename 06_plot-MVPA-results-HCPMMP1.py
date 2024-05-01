#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 16 15:39:31 2024

@author: costantino_ai
"""

import subprocess
import os

def run_python_scripts(directory):
    # List all files in the directory
    files = os.listdir(directory)
    # Filter to include only Python files
    python_files = [file for file in files if file.endswith('.py') and ("plot_HCP_" in file)]

    # Execute each Python file
    for script in python_files:
        script_path = os.path.join(directory, script)
        print(f"Running {script_path}...")
        try:
            # Run the script
            result = subprocess.run(['python', script_path], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            # Print the output and errors (if any)
            print("Output:", result.stdout.decode())
        except subprocess.CalledProcessError as e:
            print(f"Error running {script_path}: {e}")

# Specify the directory containing the Python scripts
script_directory = '/home/eik-tb/OneDrive_andreaivan.costantino@kuleuven.be/GitHub/chess_expertise/fMRI_analysis/test'
run_python_scripts(script_directory)
