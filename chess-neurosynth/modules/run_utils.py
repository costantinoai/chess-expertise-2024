import os
import sys
import inspect
import shutil
from datetime import datetime
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
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def create_output_directory(path: str) -> None:
    os.makedirs(path, exist_ok=True)
    logging.info(f"Output directory: {path}")


def save_script_to_file(out_dir: str) -> None:
    caller_frame = inspect.stack()[1]
    script_file = caller_frame.filename
    dest = os.path.join(out_dir, os.path.basename(script_file))
    shutil.copy(script_file, dest)
    logging.info(f"Saved script copy to {dest}")
