"""Training Foundation for ContrastiveTrust.

This package provides the core configuration and factory functions
required to instantiate optimisers and learning rate schedulers for
the ContrastiveTrust training pipeline.

Public API
----------
OptimizerConfig
    Hyper-parameters for the optimiser.
SchedulerConfig
    Hyper-parameters for the learning rate scheduler.
TrainingConfig
    Top-level configuration orchestrating the entire training pipeline.
create_optimizer
    Factory function to instantiate a PyTorch optimiser.
create_scheduler
    Factory function to instantiate a PyTorch learning rate scheduler.
MixedPrecisionManager
    Context manager and scaler for Automatic Mixed Precision (AMP).
Trainer
    Orchestrates the pre-training loop for the Dual-Stream Encoder.
Callback
    Base class for training callbacks.
EarlyStopping
    Callback to halt training when metrics stop improving.
ModelCheckpoint
    Callback to save model state dictionary to disk.
MetricsLogger
    Callback to log training metrics.
"""

from src.training.config import OptimizerConfig, SchedulerConfig, TrainingConfig
from src.training.optimizer_factory import create_optimizer
from src.training.scheduler_factory import create_scheduler
from src.training.mixed_precision import MixedPrecisionManager
from src.training.trainer import Trainer
from src.training.callbacks import Callback
from src.training.early_stopping import EarlyStopping
from src.training.checkpoint import ModelCheckpoint
from src.training.logger import MetricsLogger

__all__: list[str] = [
    "OptimizerConfig",
    "SchedulerConfig",
    "TrainingConfig",
    "create_optimizer",
    "create_scheduler",
    "MixedPrecisionManager",
    "Trainer",
    "Callback",
    "EarlyStopping",
    "ModelCheckpoint",
    "MetricsLogger",
]
