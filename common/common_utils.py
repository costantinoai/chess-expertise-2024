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
from datetime import datetime
from pathlib import Path
from typing import Optional

import logging as _logging
import numpy as np


def add_file_logger(log_file: str | os.PathLike[str], level: int = _logging.INFO) -> None:
    """Attach a file handler to the root logger in addition to console output.

    Use with an existing console logger to write logs both to the IDE console
    and to a file. Safe to call multiple times; only one handler per path.
    """
    path = str(log_file)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    logger = _logging.getLogger()
    logger.setLevel(level)
    # Avoid duplicate handlers for the same path
    for h in logger.handlers:
        if isinstance(h, _logging.FileHandler) and getattr(h, 'baseFilename', None) == os.path.abspath(path):
            return
    fh = _logging.FileHandler(path, encoding="utf-8")
    fmt = _logging.Formatter("[%(levelname)s %(asctime)s] %(message)s")
    fh.setFormatter(fmt)
    fh.setLevel(level)
    logger.addHandler(fh)


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
    Path(directory_path).mkdir(parents=True, exist_ok=True)
    _logging.info("Output directory ready: %s", directory_path)


def save_script_to_file(output_directory: str | os.PathLike[str]) -> None:
    """
    Copy the calling script to the output directory for provenance.

    What: Detects the script file that called this function and copies it to
    the specified folder.

    Why: Preserves the exact code used to generate artefacts, aiding
    reproducibility and peer review.
    """
    caller_frame = inspect.stack()[1]
    script_file = caller_frame.filename
    dest = Path(output_directory) / Path(script_file).name
    shutil.copy(script_file, dest)
    _logging.info("Saved a copy of the script to: %s", dest)

