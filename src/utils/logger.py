"""Logging utility module.

Sets up standard Python logging storing logs to file and console with metadata
such as timestamp, log level, module, function name, and line number (Slide 18).
"""

import logging
import sys
from pathlib import Path


def get_logger(name: str = "iris_project", log_file: str = "logs.log") -> logging.Logger:
    """Create and configure a logger instance with console and file handlers.

    Args:
        name: Logger name (usually __name__).
        log_file: Path to destination log file.

    Returns:
        logging.Logger: Configured logger.
    """
    logger = logging.getLogger(name)

    if not logger.handlers:
        logger.setLevel(logging.INFO)

        # Formatter with time, module, function, and line number as per Slide 18
        formatter = logging.Formatter(
            fmt="[%(asctime)s] [%(levelname)s] [%(name)s.%(funcName)s:%(lineno)d]: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )

        # File Handler
        project_root = Path(__file__).resolve().parent.parent.parent
        file_path = project_root / log_file
        file_handler = logging.FileHandler(file_path, mode="a", encoding="utf-8")
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        # Stream / Console Handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    return logger
