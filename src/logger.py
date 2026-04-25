"""
src/logger.py
─────────────
Centralized logging setup.
Call get_logger(__name__) in any module for consistent log output.
"""

import logging
import os
from datetime import datetime


def get_logger(name: str, log_dir: str = "logs") -> logging.Logger:
    """
    Returns a configured logger that writes to both console and a log file.

    Args:
        name      : Module name (use __name__).
        log_dir   : Directory where log files are stored.

    Returns:
        logging.Logger instance.
    """
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, "phishing_detector.log")

    logger = logging.getLogger(name)
    if logger.handlers:          # Avoid duplicate handlers on re-import
        return logger

    logger.setLevel(logging.DEBUG)

    fmt = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler — INFO and above
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)

    # File handler — DEBUG and above
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)

    logger.addHandler(ch)
    logger.addHandler(fh)

    return logger
