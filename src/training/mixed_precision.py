"""Mixed precision training utilities.

This module provides the :class:`MixedPrecisionManager`, which abstracts
the boilerplate of `torch.amp` (Automatic Mixed Precision), including
the context manager for the forward pass, and the gradient scaling logic
for the backward pass and optimizer stepping.
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from typing import Generator

import torch
import torch.nn as nn
import torch.optim as optim

logger = logging.getLogger(__name__)


class MixedPrecisionManager:
    """Manages Automatic Mixed Precision (AMP) for training.

    Parameters
    ----------
    enabled : bool
        Whether to enable mixed precision.
    device : str
        The target device (e.g., ``"cuda"``, ``"cpu"``). AMP is
        typically only used on CUDA devices.
    """

    def __init__(self, enabled: bool = True, device: str = "cuda") -> None:
        self.enabled = enabled
        self.device = device

        if self.enabled and self.device == "cuda":
            self.scaler = torch.amp.GradScaler(device="cuda")
            logger.info("MixedPrecisionManager | AMP enabled (cuda)")
        else:
            self.scaler = None
            if self.enabled:
                logger.warning(
                    f"MixedPrecisionManager | AMP enabled but device is "
                    f"'{device}'. AMP requires cuda. Falling back to FP32."
                )
            else:
                logger.info("MixedPrecisionManager | AMP disabled")

    @contextmanager
    def autocast(self) -> Generator[None, None, None]:
        """Context manager for the mixed-precision forward pass.

        Yields
        ------
        None
        """
        if self.scaler is not None:
            with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                yield
        else:
            yield

    def scale_and_backward(self, loss: torch.Tensor) -> None:
        """Scale the loss and compute gradients.

        Parameters
        ----------
        loss : torch.Tensor
            The scalar loss tensor.
        """
        if self.scaler is not None:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()

    def step_and_update(self, optimizer: optim.Optimizer) -> None:
        """Step the optimiser and update the gradient scaler.

        Parameters
        ----------
        optimizer : torch.optim.Optimizer
            The optimiser to step.
        """
        if self.scaler is not None:
            self.scaler.step(optimizer)
            self.scaler.update()
        else:
            optimizer.step()

    def unscale_(self, optimizer: optim.Optimizer) -> None:
        """Unscale gradients before clipping.

        Parameters
        ----------
        optimizer : torch.optim.Optimizer
            The optimiser whose gradients should be unscaled.
        """
        if self.scaler is not None:
            self.scaler.unscale_(optimizer)
