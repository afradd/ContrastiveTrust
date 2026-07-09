"""Scheduler factory for ContrastiveTrust.

This module provides a factory function for instantiating PyTorch
learning rate schedulers based on the :class:`SchedulerConfig`.
"""

from __future__ import annotations

import logging

import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler

from src.training.config import SchedulerConfig

logger = logging.getLogger(__name__)


def create_scheduler(
    optimizer: optim.Optimizer,
    config: SchedulerConfig,
) -> lr_scheduler.LRScheduler | lr_scheduler.ReduceLROnPlateau:
    """Instantiate a PyTorch learning rate scheduler from a :class:`SchedulerConfig`.

    Parameters
    ----------
    optimizer : torch.optim.Optimizer
        The PyTorch optimiser to wrap.
    config : SchedulerConfig
        The configuration detailing the scheduler name and additional kwargs.

    Returns
    -------
    torch.optim.lr_scheduler.LRScheduler or ReduceLROnPlateau
        The instantiated PyTorch learning rate scheduler.

    Raises
    ------
    TypeError
        If *optimizer* is not a :class:`torch.optim.Optimizer` or if
        *config* is not a :class:`SchedulerConfig`.
    ValueError
        If the requested scheduler name is not supported.

    Examples
    --------
    >>> import torch
    >>> from src.training.config import OptimizerConfig, SchedulerConfig
    >>> from src.training.optimizer_factory import create_optimizer
    >>> from src.training.scheduler_factory import create_scheduler
    >>> model = torch.nn.Linear(10, 2)
    >>> opt = create_optimizer(model.parameters(), OptimizerConfig(name="Adam"))
    >>> cfg = SchedulerConfig(name="StepLR", kwargs={"step_size": 10})
    >>> sched = create_scheduler(opt, cfg)
    >>> type(sched).__name__
    'StepLR'
    """
    if not isinstance(optimizer, optim.Optimizer):
        raise TypeError(
            f"optimizer must be a torch.optim.Optimizer, "
            f"got {type(optimizer).__name__}"
        )
    if not isinstance(config, SchedulerConfig):
        raise TypeError(
            f"config must be a SchedulerConfig, got {type(config).__name__}"
        )

    name = config.name.lower().strip()
    kwargs = dict(config.kwargs)

    logger.debug(
        "Creating scheduler '%s' | kwargs=%s",
        config.name,
        kwargs,
    )

    if name == "cosineannealinglr":
        if "T_max" not in kwargs:
            raise ValueError("CosineAnnealingLR requires 'T_max' in kwargs")
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, **kwargs)
    elif name == "steplr":
        if "step_size" not in kwargs:
            raise ValueError("StepLR requires 'step_size' in kwargs")
        scheduler = lr_scheduler.StepLR(optimizer, **kwargs)
    elif name == "reducelronplateau":
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, **kwargs)
    elif name == "exponentiallr":
        if "gamma" not in kwargs:
            raise ValueError("ExponentialLR requires 'gamma' in kwargs")
        scheduler = lr_scheduler.ExponentialLR(optimizer, **kwargs)
    else:
        raise ValueError(
            f"Unsupported scheduler '{config.name}'. "
            f"Supported schedulers are: CosineAnnealingLR, StepLR, "
            f"ReduceLROnPlateau, ExponentialLR."
        )

    logger.info("Instantiated scheduler %s", type(scheduler).__name__)
    return scheduler
