"""Callback base class for ContrastiveTrust training.

This module provides the :class:`Callback` interface for injecting custom
logic (e.g., logging, early stopping, checkpointing) into the training
loop without modifying the core :class:`Trainer`.
"""

from __future__ import annotations

from typing import Any, Dict


class Callback:
    """Base class for all training callbacks.

    Subclasses should override the appropriate hooks. The `trainer` argument
    passed to each hook is the instance of :class:`Trainer` managing the loop.
    """

    def on_train_begin(self, trainer: Any) -> None:
        """Called once at the very beginning of training."""
        pass

    def on_train_end(self, trainer: Any) -> None:
        """Called once at the very end of training."""
        pass

    def on_epoch_begin(self, trainer: Any, epoch: int) -> None:
        """Called at the beginning of each epoch.

        Parameters
        ----------
        trainer : Trainer
            The training orchestrator.
        epoch : int
            The current epoch number (1-indexed).
        """
        pass

    def on_epoch_end(
        self, trainer: Any, epoch: int, metrics: Dict[str, float]
    ) -> None:
        """Called at the end of each epoch.

        Parameters
        ----------
        trainer : Trainer
            The training orchestrator.
        epoch : int
            The current epoch number (1-indexed).
        metrics : dict[str, float]
            The aggregated metrics for the epoch (e.g., `train_loss`, `val_loss`).
        """
        pass

    def on_batch_begin(self, trainer: Any, batch: int) -> None:
        """Called at the beginning of each training batch."""
        pass

    def on_batch_end(
        self, trainer: Any, batch: int, logs: Dict[str, float]
    ) -> None:
        """Called at the end of each training batch."""
        pass
