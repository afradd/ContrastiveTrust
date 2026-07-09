"""Training configurations for ContrastiveTrust.

This module provides dataclasses that define the hyper-parameters
and settings for the full training pipeline, including the optimiser,
learning rate scheduler, and the top-level orchestrating configuration.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict

from src.data.view_generator import ContrastiveViewGeneratorConfig
from src.losses.contrastive_trust_loss import ContrastiveTrustLossConfig
from src.models.encoder import EncoderConfig


@dataclass(frozen=True)
class OptimizerConfig:
    """Hyper-parameters for the optimiser.

    Parameters
    ----------
    name : str
        Name of the optimiser (e.g., ``"Adam"``, ``"AdamW"``, ``"SGD"``).
    lr : float
        Learning rate. Must be strictly positive.
    weight_decay : float
        L2 penalty (weight decay). Must be non-negative.
    kwargs : dict[str, Any]
        Additional keyword arguments to pass to the optimiser constructor.
    """

    name: str = "AdamW"
    lr: float = 1e-3
    weight_decay: float = 1e-4
    kwargs: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate optimiser configuration."""
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError(f"name must be a non-empty string, got {self.name!r}")
        if not isinstance(self.lr, (int, float)) or self.lr <= 0.0:
            raise ValueError(f"lr must be strictly positive, got {self.lr}")
        if not isinstance(self.weight_decay, (int, float)) or self.weight_decay < 0.0:
            raise ValueError(f"weight_decay must be non-negative, got {self.weight_decay}")
        if not isinstance(self.kwargs, dict):
            raise TypeError(f"kwargs must be a dict, got {type(self.kwargs).__name__}")


@dataclass(frozen=True)
class SchedulerConfig:
    """Hyper-parameters for the learning rate scheduler.

    Parameters
    ----------
    name : str
        Name of the scheduler (e.g., ``"CosineAnnealingLR"``, ``"StepLR"``).
    kwargs : dict[str, Any]
        Additional keyword arguments to pass to the scheduler constructor.
    """

    name: str = "CosineAnnealingLR"
    kwargs: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate scheduler configuration."""
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError(f"name must be a non-empty string, got {self.name!r}")
        if not isinstance(self.kwargs, dict):
            raise TypeError(f"kwargs must be a dict, got {type(self.kwargs).__name__}")


@dataclass(frozen=True)
class TrainingConfig:
    """Top-level configuration for the ContrastiveTrust training pipeline.

    Composes configurations for the sub-modules (encoder, loss, data
    augmentation/views) alongside training hyper-parameters (optimiser,
    scheduler, batch size, epochs).

    Parameters
    ----------
    epochs : int
        Number of training epochs. Must be strictly positive.
    batch_size : int
        Training batch size. Must be strictly positive.
    device : str
        Target device (e.g., ``"cpu"``, ``"cuda"``, ``"mps"``).
    optimizer : OptimizerConfig
        Optimiser hyper-parameters.
    scheduler : SchedulerConfig
        Learning rate scheduler hyper-parameters.
    encoder : EncoderConfig
        Dual-stream encoder configuration.
    loss : ContrastiveTrustLossConfig
        Unified multi-objective loss configuration.
    view_generator : ContrastiveViewGeneratorConfig
        Contrastive data augmentation configuration.
    """

    epochs: int = 100
    batch_size: int = 256
    device: str = "cuda"
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    encoder: EncoderConfig = field(default_factory=EncoderConfig)
    loss: ContrastiveTrustLossConfig = field(default_factory=ContrastiveTrustLossConfig)
    view_generator: ContrastiveViewGeneratorConfig = field(
        default_factory=ContrastiveViewGeneratorConfig
    )

    def __post_init__(self) -> None:
        """Validate training configuration."""
        if not isinstance(self.epochs, int) or self.epochs <= 0:
            raise ValueError(f"epochs must be a strictly positive integer, got {self.epochs}")
        if not isinstance(self.batch_size, int) or self.batch_size <= 0:
            raise ValueError(f"batch_size must be a strictly positive integer, got {self.batch_size}")
        if not isinstance(self.device, str) or not self.device.strip():
            raise ValueError(f"device must be a non-empty string, got {self.device!r}")
        
        # Ensure sub-configs are of correct type
        if not isinstance(self.optimizer, OptimizerConfig):
            raise TypeError(f"optimizer must be an OptimizerConfig, got {type(self.optimizer).__name__}")
        if not isinstance(self.scheduler, SchedulerConfig):
            raise TypeError(f"scheduler must be a SchedulerConfig, got {type(self.scheduler).__name__}")
        if not isinstance(self.encoder, EncoderConfig):
            raise TypeError(f"encoder must be an EncoderConfig, got {type(self.encoder).__name__}")
        if not isinstance(self.loss, ContrastiveTrustLossConfig):
            raise TypeError(f"loss must be a ContrastiveTrustLossConfig, got {type(self.loss).__name__}")
        if not isinstance(self.view_generator, ContrastiveViewGeneratorConfig):
            raise TypeError(f"view_generator must be a ContrastiveViewGeneratorConfig, got {type(self.view_generator).__name__}")
