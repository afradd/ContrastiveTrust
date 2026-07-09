"""Metrics logging callback for ContrastiveTrust training.

This module provides :class:`MetricsLogger` to record training metrics
to a file (e.g., JSON Lines) and the console.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, Union

from src.training.callbacks import Callback

logger = logging.getLogger(__name__)


class MetricsLogger(Callback):
    """Logs training metrics to a file and/or the console.

    Parameters
    ----------
    log_dir : str or Path
        Directory to write log files to.
    filename : str, default="metrics.jsonl"
        Name of the JSON lines log file.
    log_to_console : bool, default=True
        If True, prints metrics to standard logging as well.
    """

    def __init__(
        self,
        log_dir: Union[str, Path],
        filename: str = "metrics.jsonl",
        log_to_console: bool = True,
    ) -> None:
        self.log_dir = Path(log_dir)
        self.filename = filename
        self.log_to_console = log_to_console

        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.log_file = self.log_dir / self.filename

        logger.info("MetricsLogger initialized | log_file=%s", self.log_file)

    def on_epoch_end(
        self, trainer: Any, epoch: int, metrics: Dict[str, float]
    ) -> None:
        """Write metrics to file and log to console."""
        record = {"epoch": epoch, **metrics}
        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

        if self.log_to_console:
            metrics_str = " | ".join(f"{k}: {v:.4f}" for k, v in metrics.items())
            logger.info("Epoch %03d Metrics | %s", epoch, metrics_str)
