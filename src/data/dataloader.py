"""DataLoader factory functions for ICS contrastive learning.

This module provides convenience functions for creating PyTorch
:class:`~torch.utils.data.DataLoader` instances with sensible defaults
for training, validation, and test splits.

Each factory function wraps a :class:`ContrastiveDataset` and returns
a fully configured ``DataLoader`` ready for iteration.
"""

from __future__ import annotations

import logging
from typing import Optional, Union

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor
from torch.utils.data import DataLoader

from src.data.contrastive_dataset import ContrastiveDataset

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Defaults
# ------------------------------------------------------------------

_DEFAULT_BATCH_SIZE: int = 64
_DEFAULT_NUM_WORKERS: int = 0
_DEFAULT_PIN_MEMORY: bool = True
_DEFAULT_PERSISTENT_WORKERS: bool = False


# ------------------------------------------------------------------
# Factory helpers
# ------------------------------------------------------------------

def _build_dataloader(
    dataset: ContrastiveDataset,
    batch_size: int,
    shuffle: bool,
    drop_last: bool,
    num_workers: int,
    pin_memory: bool,
    persistent_workers: bool,
    split_name: str,
) -> DataLoader:
    """Build a DataLoader and log its configuration.

    Args:
        dataset: The :class:`ContrastiveDataset` to wrap.
        batch_size: Number of samples per batch.
        shuffle: Whether to shuffle samples each epoch.
        drop_last: Whether to drop the last incomplete batch.
        num_workers: Number of subprocesses for data loading.
        pin_memory: Whether to copy tensors into pinned memory.
        persistent_workers: Whether to keep workers alive across epochs.
        split_name: Human-readable name for logging (e.g. ``"train"``).

    Returns:
        A configured :class:`~torch.utils.data.DataLoader`.

    Raises:
        ValueError: If ``batch_size`` is not a positive integer.
        TypeError: If ``dataset`` is not a :class:`ContrastiveDataset`.
    """
    if not isinstance(dataset, ContrastiveDataset):
        raise TypeError(
            f"dataset must be a ContrastiveDataset, "
            f"got {type(dataset).__name__}"
        )

    if not isinstance(batch_size, int) or isinstance(batch_size, bool):
        raise TypeError(
            f"batch_size must be a positive integer, "
            f"got {type(batch_size).__name__}"
        )

    if batch_size <= 0:
        raise ValueError(
            f"batch_size must be a positive integer, got {batch_size}"
        )

    # persistent_workers requires num_workers > 0
    effective_persistent_workers = persistent_workers and num_workers > 0

    loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=effective_persistent_workers,
    )

    num_batches = len(loader)

    logger.info(
        "[%s] DataLoader created: batch_size=%d | shuffle=%s | "
        "drop_last=%s | num_workers=%d | pin_memory=%s | "
        "persistent_workers=%s | samples=%d | batches=%d",
        split_name,
        batch_size,
        shuffle,
        drop_last,
        num_workers,
        pin_memory,
        effective_persistent_workers,
        len(dataset),
        num_batches,
    )

    return loader


# ------------------------------------------------------------------
# Public factory functions
# ------------------------------------------------------------------

def create_train_dataloader(
    dataset: ContrastiveDataset,
    batch_size: int = _DEFAULT_BATCH_SIZE,
    shuffle: bool = True,
    drop_last: bool = True,
    num_workers: int = _DEFAULT_NUM_WORKERS,
    pin_memory: bool = _DEFAULT_PIN_MEMORY,
    persistent_workers: bool = _DEFAULT_PERSISTENT_WORKERS,
) -> DataLoader:
    """Create a DataLoader for the training split.

    Training loaders shuffle data by default and drop the last
    incomplete batch to ensure uniform batch sizes across gradient
    accumulation steps.

    Args:
        dataset: Training :class:`ContrastiveDataset`.
        batch_size: Number of samples per batch.
        shuffle: Whether to shuffle each epoch.  Defaults to ``True``.
        drop_last: Whether to drop the final incomplete batch.
            Defaults to ``True``.
        num_workers: Number of data-loading subprocesses.
        pin_memory: Whether to pin tensors in CUDA memory.
        persistent_workers: Whether to keep workers alive across epochs.

    Returns:
        A configured training :class:`~torch.utils.data.DataLoader`.
    """
    return _build_dataloader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        split_name="train",
    )


def create_validation_dataloader(
    dataset: ContrastiveDataset,
    batch_size: int = _DEFAULT_BATCH_SIZE,
    shuffle: bool = False,
    drop_last: bool = False,
    num_workers: int = _DEFAULT_NUM_WORKERS,
    pin_memory: bool = _DEFAULT_PIN_MEMORY,
    persistent_workers: bool = _DEFAULT_PERSISTENT_WORKERS,
) -> DataLoader:
    """Create a DataLoader for the validation split.

    Validation loaders do not shuffle and preserve every sample for
    faithful metric computation.

    Args:
        dataset: Validation :class:`ContrastiveDataset`.
        batch_size: Number of samples per batch.
        shuffle: Whether to shuffle.  Defaults to ``False``.
        drop_last: Whether to drop the final incomplete batch.
            Defaults to ``False``.
        num_workers: Number of data-loading subprocesses.
        pin_memory: Whether to pin tensors in CUDA memory.
        persistent_workers: Whether to keep workers alive across epochs.

    Returns:
        A configured validation :class:`~torch.utils.data.DataLoader`.
    """
    return _build_dataloader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        split_name="validation",
    )


def create_test_dataloader(
    dataset: ContrastiveDataset,
    batch_size: int = _DEFAULT_BATCH_SIZE,
    shuffle: bool = False,
    drop_last: bool = False,
    num_workers: int = _DEFAULT_NUM_WORKERS,
    pin_memory: bool = _DEFAULT_PIN_MEMORY,
    persistent_workers: bool = _DEFAULT_PERSISTENT_WORKERS,
) -> DataLoader:
    """Create a DataLoader for the test split.

    Test loaders mirror validation settings: no shuffle, no dropping,
    ensuring deterministic and complete evaluation.

    Args:
        dataset: Test :class:`ContrastiveDataset`.
        batch_size: Number of samples per batch.
        shuffle: Whether to shuffle.  Defaults to ``False``.
        drop_last: Whether to drop the final incomplete batch.
            Defaults to ``False``.
        num_workers: Number of data-loading subprocesses.
        pin_memory: Whether to pin tensors in CUDA memory.
        persistent_workers: Whether to keep workers alive across epochs.

    Returns:
        A configured test :class:`~torch.utils.data.DataLoader`.
    """
    return _build_dataloader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        split_name="test",
    )


__all__ = [
    "create_train_dataloader",
    "create_validation_dataloader",
    "create_test_dataloader",
]
