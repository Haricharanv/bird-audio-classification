"""Logging utilities for formatted console output, file logging, and metric table display."""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any, List, Optional, Union


def get_logger(
    name: str = "bird_classification",
    log_file: Optional[Union[str, Path]] = None,
    level: int = logging.INFO,
) -> logging.Logger:
    """Create or retrieve a standard logger with unified formatting.
    
    Args:
        name: Logger name.
        log_file: Optional path to log file.
        level: Logging verbosity level.
        
    Returns:
        Configured logging.Logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Avoid duplicate handlers if logger was already created
    if not logger.handlers:
        formatter = logging.Formatter(
            fmt="[%(asctime)s] [%(levelname)s] [%(filename)s:%(lineno)d]: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        # Console Stream Handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        # Optional File Handler
        if log_file:
            log_path = Path(log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(str(log_path), mode="a", encoding="utf-8")
            file_handler.setLevel(level)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

    return logger


def format_epoch_table(headers: List[str], rows: List[List[Any]]) -> str:
    """Format headers and rows into a clean, aligned Markdown/terminal table.
    
    Args:
        headers: Column header strings.
        rows: Rows of values.
        
    Returns:
        Formatted ASCII/Markdown table string.
    """
    str_rows = []
    for r in rows:
        formatted_r = []
        for v in r:
            if isinstance(v, float):
                formatted_r.append(f"{v:.4f}")
            else:
                formatted_r.append(str(v))
        str_rows.append(formatted_r)

    # Determine max column widths
    col_widths = [len(h) for h in headers]
    for r in str_rows:
        for idx, val in enumerate(r):
            if idx < len(col_widths):
                col_widths[idx] = max(col_widths[idx], len(val))

    def make_row(items: List[str]) -> str:
        padded = [item.ljust(col_widths[i]) for i, item in enumerate(items)]
        return "| " + " | ".join(padded) + " |"

    separator = "|-" + "-|-".join(["-" * w for w in col_widths]) + "-|"
    header_line = make_row(headers)
    data_lines = [make_row(r) for r in str_rows]

    return "\n".join([header_line, separator] + data_lines)

