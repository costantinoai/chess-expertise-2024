import logging
from pathlib import Path


def setup_logging(log_file: str | None = None, level: int = logging.INFO) -> logging.Logger:
    """Configure and return a logger with a consistent format.

    - Always logs to console.
    - Optionally also logs to a file if `log_file` is provided.
    - Safe to call multiple times; it won’t duplicate handlers.
    """
    logger = logging.getLogger()
    fmt = "[%(levelname)s %(asctime)s] %(message)s"
    logger.setLevel(level)

    # Add console handler if missing
    if not any(isinstance(h, logging.StreamHandler) for h in logger.handlers):
        ch = logging.StreamHandler()
        ch.setFormatter(logging.Formatter(fmt))
        logger.addHandler(ch)

    # Add file handler if requested and not already present
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        abs_path = str(Path(log_file).resolve())
        if not any(isinstance(h, logging.FileHandler) and getattr(h, 'baseFilename', None) == abs_path for h in logger.handlers):
            fh = logging.FileHandler(abs_path)
            fh.setFormatter(logging.Formatter(fmt))
            logger.addHandler(fh)

    return logger

