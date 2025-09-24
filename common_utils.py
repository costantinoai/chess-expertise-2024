#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Common utilities shared across analyses.

This module centralizes helpers that were duplicated across analysis folders.
It aims to keep code DRY, readable, and easy to maintain. Functions are
documented for clarity for non-expert readers.
"""
from __future__ import annotations

import inspect
import os
import random
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import logging as _logging
import numpy as np


class OutputLogger:
    """
    Context manager to tee console output to a log file.

    What: Mirrors everything printed to stdout/stderr into a file while keeping
    it visible in the terminal.

    Why: Capturing the exact console log for each run improves transparency and
    reproducibility when sharing analysis outputs.
    """

    def __init__(self, log: bool, file_path: str | os.PathLike[str]):
        self.log = bool(log)
        self.file_path = str(file_path)
        self.original_stdout = sys.stdout
        self.original_stderr = sys.stderr
        self.log_file: Optional[object] = None

    def __enter__(self):
        if self.log:
            Path(self.file_path).parent.mkdir(parents=True, exist_ok=True)
            self.log_file = open(self.file_path, "w", encoding="utf-8")
            sys.stdout = self
            sys.stderr = self
            # Reconfigure stdlib logging to use (possibly) redirected stderr
            for handler in _logging.root.handlers:
                handler.stream = sys.stderr
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.log:
            for handler in _logging.root.handlers:
                handler.stream = self.original_stderr
            sys.stdout = self.original_stdout
            sys.stderr = self.original_stderr
            assert self.log_file is not None
            self.log_file.close()

    def write(self, message: str):
        self.original_stdout.write(message)
        if self.log and self.log_file and not self.log_file.closed:
            self.log_file.write(message)

    def flush(self):
        self.original_stdout.flush()
        if self.log and self.log_file and not self.log_file.closed:
            self.log_file.flush()


def set_rnd_seed(seed: int = 42) -> None:
    """Set random seeds (NumPy and Python) for reproducibility."""
    np.random.seed(int(seed))
    random.seed(int(seed))


def create_run_id() -> str:
    """Return a timestamp string YYYYMMDD-HHMMSS for naming outputs."""
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def create_output_directory(directory_path: str | os.PathLike[str]) -> None:
    """
    Create an output directory if it does not exist.

    Why: Ensures scripts have a valid destination for results without
    hard-coding paths or risking overwrites.
    """
    try:
        Path(directory_path).mkdir(parents=True, exist_ok=True)
        _logging.info("Output directory ready: %s", directory_path)
    except Exception:
        _logging.exception("Failed to create output directory: %s", directory_path)


def save_script_to_file(output_directory: str | os.PathLike[str]) -> None:
    """
    Copy the calling script to the output directory for provenance.

    What: Detects the script file that called this function and copies it to
    the specified folder.

    Why: Preserves the exact code used to generate artefacts, aiding
    reproducibility and peer review.
    """
    try:
        caller_frame = inspect.stack()[1]
        script_file = caller_frame.filename
        dest = Path(output_directory) / Path(script_file).name
        shutil.copy(script_file, dest)
        _logging.info("Saved a copy of the script to: %s", dest)
    except Exception:
        _logging.exception("Could not save calling script to output directory")

