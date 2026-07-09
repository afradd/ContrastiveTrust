"""Model checkpointing callback.

This module provides :class:`ModelCheckpoint` to save the model's state
dictionary to disk during training, typically when a metric improves.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Union

import torch

from src.training.callbacks import Callback

logger = logging.getLogger(__name__)


class ModelCheckpoint(Callback):
    """Saves the model after every epoch if the monitored metric improves.

    Parameters
    ----------
    filepath : str or Path
        Path where the checkpoint will be saved.
    monitor : str, default="val_loss"
        The metric to monitor.
    mode : str, default="min"
        One of ``{"min", "max"}``.
    save_best_only : bool, default=True
        If ``True``, the latest best model according to the quantity
        monitored will not be overwritten by worse models.
    """

    def __init__(
        self,
        filepath: Union[str, Path],
        monitor: str = "val_loss",
        mode: str = "min",
        save_best_only: bool = True,
    ) -> None:
        self.filepath = Path(filepath)
        self.monitor = monitor
        self.mode = mode.lower()
        self.save_best_only = save_best_only

        if self.mode not in ("min", "max"):
            raise ValueError(f"mode must be 'min' or 'max', got {self.mode}")

        self.best_score = float("inf") if self.mode == "min" else -float("inf")
        self.filepath.parent.mkdir(parents=True, exist_ok=True)

        logger.info(
            "ModelCheckpoint configured | filepath=%s | monitor=%s | mode=%s",
            self.filepath,
            self.monitor,
            self.mode,
        )

    def on_epoch_end(
        self, trainer: Any, epoch: int, metrics: Dict[str, float]
    ) -> None:
        """Potentially save the model depending on metric improvement."""
        current = metrics.get(self.monitor)
        if current is None:
            logger.warning(
                "ModelCheckpoint: metric '%s' not found. Skipping save.",
                self.monitor,
            )
            return

        if self.mode == "min":
            improvement = current < self.best_score
        else:
            improvement = current > self.best_score

        if improvement or not self.save_best_only:
            if improvement:
                logger.info(
                    "ModelCheckpoint: %s improved from %.4f to %.4f. "
                    "Saving model to %s",
                    self.monitor,
                    self.best_score,
                    current,
                    self.filepath,
                )
                self.best_score = current
            else:
                logger.info("ModelCheckpoint: saving model to %s", self.filepath)

            state = {
                "epoch": epoch,
                "encoder_state_dict": trainer.encoder.state_dict(),
                "projection_head_state_dict": trainer.projection_head.state_dict(),
                "optimizer_state_dict": trainer.optimizer.state_dict(),
                "metrics": metrics,
            }
            if trainer.scheduler is not None:
                state["scheduler_state_dict"] = trainer.scheduler.state_dict()

            torch.save(state, self.filepath)
