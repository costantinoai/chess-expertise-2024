#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun  2 13:38:55 2024

@author: costantino_ai
"""

import os
import json
import argparse
import math
import simplejson

def is_nan(value):
    """Check if a value is NaN."""
    return isinstance(value, float) and math.isnan(value)

def process_file(file_path, dry_run=False):
    """Process a single JSON file, replacing NaN values and optionally printing changes."""
    with open(file_path, 'r') as file:
        data = json.load(file)

    def replace_and_print(data, path=[]):
        if isinstance(data, dict):
            for k, v in data.items():
                if math.isnan(v) or str(v).lower() == 'nan':
                    old_value = 'NaN'
                    new_value = 'null'
                    print(f"{file_path}:{'.'.join(path + [k])}: {old_value} --> {new_value}")
                    data[k] = None
                else:
                    replace_and_print(v, path + [k])
        elif isinstance(data, list):
            for i, v in enumerate(data):
                if math.isnan(v) or str(v).lower() == 'nan':
                    old_value = 'NaN'
                    new_value = 'null'
                    print(f"{file_path}:{'.'.join(path + [str(i)])}: {old_value} --> {new_value}")
                    data[i] = None
                else:
                    replace_and_print(v, path + [str(i)])

    replace_and_print(data)

    if not dry_run:
        with open(file_path, 'w') as file:
            json.dump(data, file, indent=4)

def find_nan_json_files(folder):
    """Scan for JSON files in the directory that contain non-compliant float values."""
    problematic_files = []
    for root, dirs, files in os.walk(folder):
        for file in files:
            if file.endswith('.json'):
                file_path = os.path.join(root, file)
                # print(file_path)
                try:
                    with open(file_path, 'r') as f:
                        data = json.load(f)
                    # Attempt to encode with simplejson
                    simplejson.dumps(data)
                except ValueError as e:
                    if 'nan' in str(e):
                        problematic_files.append(file_path)
                        print(f"Found problematic file: {file_path}")
                except Exception as e:
                    print(f"Error processing file {file_path}: {e}")
    return problematic_files

def main(folder, dry_run):
    problematic_files = find_nan_json_files(folder)
    for file_path in problematic_files:
        process_file(file_path, dry_run=dry_run)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Replace NaN values in JSON files and find problematic files.")
    parser.add_argument("folder", type=str, nargs='?', default="path_to_folder", help="The folder to scan for JSON files.")
    parser.add_argument("--dry", action="store_true", help="Perform a dry run without making changes.")
    args = parser.parse_args()

    # Defaults set for specific use case in IDE or command line
    folder_path = args.folder if args.folder != "path_to_folder" else "/data/projects/chess/data/BIDS"
    dry_run_mode = True

    main(folder_path, dry_run_mode)
