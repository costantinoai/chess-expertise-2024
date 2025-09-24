import os
import inspect
import shutil
from datetime import datetime
import logging
from pathlib import Path


def create_run_id() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def create_output_directory(path: str) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)
    logging.info("Output directory: %s", path)


def save_script_to_file(out_dir: str) -> None:
    caller_frame = inspect.stack()[1]
    script_file = caller_frame.filename
    dest = os.path.join(out_dir, os.path.basename(script_file))
    shutil.copy(script_file, dest)
    logging.info("Saved script copy to %s", dest)


def add_file_logger(log_file: str, level: int = logging.INFO) -> None:
    """Attach a file handler to root logger (console remains active)."""
    logger = logging.getLogger()
    logger.setLevel(level)
    for h in logger.handlers:
        if isinstance(h, logging.FileHandler) and getattr(h, 'baseFilename', None) == os.path.abspath(log_file):
            return
    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setFormatter(logging.Formatter("[%(levelname)s %(asctime)s] %(message)s"))
    fh.setLevel(level)
    logger.addHandler(fh)
