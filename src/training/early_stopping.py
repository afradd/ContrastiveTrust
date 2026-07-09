"""Early stopping callback.

This module provides :class:`EarlyStopping` which monitors a given metric
and halts training if no improvement is observed for a specified number
of epochs.
"""

from __future__ import annotations

import logging
from typing import Any, Dict

from src.training.callbacks import Callback

logger = logging.getLogger(__name__)


class EarlyStopping(Callback):
    """Stops training when a monitored metric has stopped improving.

    Parameters
    ----------
    monitor : str, default="val_loss"
        The metric to monitor.
    patience : int, default=5
        Number of epochs with no improvement after which training will
        be stopped.
    mode : str, default="min"
        One of ``{"min", "max"}``. In ``"min"`` mode, training will stop when
        the metric stops decreasing.
    min_delta : float, default=0.0
        Minimum change in the monitored metric to qualify as an improvement.
    """

    def __init__(
        self,
        monitor: str = "val_loss",
        patience: int = 5,
        mode: str = "min",
        min_delta: float = 0.0,
    ) -> None:
        self.monitor = monitor
        self.patience = patience
        self.mode = mode.lower()
        self.min_delta = min_delta

        self.wait = 0
        self.stopped_epoch = 0

        if self.mode not in ("min", "max"):
            raise ValueError(f"mode must be 'min' or 'max', got {self.mode}")

        self.best_score = float("inf") if self.mode == "min" else -float("inf")

        logger.info(
            "EarlyStopping configured | monitor=%s | patience=%d | mode=%s",
            self.monitor,
            self.patience,
            self.mode,
        )

    def on_epoch_end(
        self, trainer: Any, epoch: int, metrics: Dict[str, float]
    ) -> None:
        """Check the metric and potentially halt training.

        Parameters
        ----------
        trainer : Trainer
            The training orchestrator.  If early stopping is triggered,
            ``trainer.should_stop`` is set to ``True``.
        epoch : int
            The current epoch number.
        metrics : dict[str, float]
            The metrics computed during the epoch.
        """
        current = metrics.get(self.monitor)
        if current is None:
            logger.warning(
                "EarlyStopping: metric '%s' not found in metrics. Available: %s",
                self.monitor,
                list(metrics.keys()),
            )
            return

        if self.mode == "min":
            improvement = (self.best_score - current) > self.min_delta
        else:
            improvement = (current - self.best_score) > self.min_delta

        if improvement:
            self.best_score = current
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.stopped_epoch = epoch
                trainer.should_stop = True
                logger.info(
                    "EarlyStopping: patience exhausted. Halting training at epoch %d.",
                    epoch,
                )
