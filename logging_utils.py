import logging
from pathlib import Path

def setup_logging(log_file: str | None = None, level: int = logging.INFO) -> logging.Logger:
    """Configure and return a logger with a consistent format."""
    logger = logging.getLogger()
    if not logger.handlers:
        fmt = "[%(levelname)s %(asctime)s] %(message)s"
        logger.setLevel(level)
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(fmt))
        logger.addHandler(handler)
        if log_file:
            Path(log_file).parent.mkdir(parents=True, exist_ok=True)
            fh = logging.FileHandler(log_file)
            fh.setFormatter(logging.Formatter(fmt))
            logger.addHandler(fh)
    return logger
