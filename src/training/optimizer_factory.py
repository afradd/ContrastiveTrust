"""Optimiser factory for ContrastiveTrust.

This module provides a factory function for instantiating PyTorch
optimisers based on the :class:`OptimizerConfig`.
"""

from __future__ import annotations

import logging
from typing import Iterable

import torch
import torch.optim as optim

from src.training.config import OptimizerConfig

logger = logging.getLogger(__name__)


def create_optimizer(
    params: Iterable[torch.Tensor] | Iterable[dict],
    config: OptimizerConfig,
) -> optim.Optimizer:
    """Instantiate a PyTorch optimiser from an :class:`OptimizerConfig`.

    Parameters
    ----------
    params : iterable
        An iterable of :class:`torch.Tensor` or :class:`dict` containing
        the model parameters to optimise.
    config : OptimizerConfig
        The configuration detailing the optimiser name, learning rate,
        weight decay, and additional kwargs.

    Returns
    -------
    torch.optim.Optimizer
        The instantiated PyTorch optimiser.

    Raises
    ------
    TypeError
        If *config* is not an :class:`OptimizerConfig`.
    ValueError
        If the requested optimiser name is not supported.

    Examples
    --------
    >>> import torch
    >>> from src.training.config import OptimizerConfig
    >>> from src.training.optimizer_factory import create_optimizer
    >>> model = torch.nn.Linear(10, 2)
    >>> config = OptimizerConfig(name="AdamW", lr=1e-3, weight_decay=1e-4)
    >>> opt = create_optimizer(model.parameters(), config)
    >>> type(opt).__name__
    'AdamW'
    """
    if not isinstance(config, OptimizerConfig):
        raise TypeError(
            f"config must be an OptimizerConfig, got {type(config).__name__}"
        )

    name = config.name.lower().strip()
    kwargs = dict(config.kwargs)

    logger.debug(
        "Creating optimiser '%s' | lr=%.2e | weight_decay=%.2e | kwargs=%s",
        config.name,
        config.lr,
        config.weight_decay,
        kwargs,
    )

    if name == "adam":
        optimizer = optim.Adam(
            params,
            lr=config.lr,
            weight_decay=config.weight_decay,
            **kwargs,
        )
    elif name == "adamw":
        optimizer = optim.AdamW(
            params,
            lr=config.lr,
            weight_decay=config.weight_decay,
            **kwargs,
        )
    elif name == "sgd":
        optimizer = optim.SGD(
            params,
            lr=config.lr,
            weight_decay=config.weight_decay,
            **kwargs,
        )
    else:
        raise ValueError(
            f"Unsupported optimiser '{config.name}'. "
            f"Supported optimisers are: Adam, AdamW, SGD."
        )

    logger.info("Instantiated optimiser %s", type(optimizer).__name__)
    return optimizer
