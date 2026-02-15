"""Structured logging helpers."""

from __future__ import annotations

import logging

LOG_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"


def setup_logging(level: str = "INFO") -> logging.Logger:
    """Configure and return the root logger."""
    numeric_level = getattr(logging, level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f"Invalid log level: {level}")

    root_logger = logging.getLogger()
    root_logger.setLevel(numeric_level)

    formatter = logging.Formatter(LOG_FORMAT)
    if not root_logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(formatter)
        root_logger.addHandler(handler)
    else:
        for handler in root_logger.handlers:
            handler.setFormatter(formatter)

    return root_logger


def get_logger(name: str) -> logging.Logger:
    """Return a named logger."""
    if not name.strip():
        raise ValueError("Logger name cannot be empty.")
    return logging.getLogger(name)

