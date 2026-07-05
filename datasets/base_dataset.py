"""Base dataset interface for the repository.

This module provides an abstract :class:`BaseDataset` that defines the
minimal interface expected by downstream training and evaluation code.

Design goals:
- Keep the base class file-I/O agnostic so tests and CI can supply
  synthetic or in-memory data without touching the filesystem.
- Provide lazy-loading semantics; subclasses can defer expensive I/O
  until the dataset is first accessed.
- Surface clear, actionable errors and log useful diagnostics for
  production troubleshooting.
"""

from __future__ import annotations

import abc
import logging
from typing import Any, Callable, Optional, Tuple, Union

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor
from torch.utils.data import Dataset

# Module-level logger so callers can configure handlers/formatters.
logger = logging.getLogger(__name__)


ArrayLike = Union[NDArray[Any], Tensor]


class BaseDataset(Dataset, abc.ABC):
    """Abstract base dataset.

    Subclasses must implement :meth:`load_data` and :meth:`get_num_features`.

    The base class accepts optional pre-loaded ``data`` and ``labels``
    to support testing and in-memory workflows. It intentionally does
    not perform any file I/O or parsing (CSV/Excel) — file reading
    belongs in dataset-specific subclasses.
    """

    def __init__(
        self,
        data: Optional[ArrayLike] = None,
        labels: Optional[ArrayLike] = None,
        transform: Optional[Callable[[Tensor], Tensor]] = None,
    ) -> None:
        """Initialise the dataset.

        Args:
            data: Optional pre-loaded features (numpy array or tensor).
            labels: Optional pre-loaded labels aligned with ``data``.
            transform: Optional transform applied to each sample tensor.

        Raises:
            TypeError: If ``data`` or ``labels`` are of unsupported type.
            ValueError: If lengths of ``data`` and ``labels`` mismatch.
        """
        self._data: Optional[ArrayLike] = None
        self._labels: Optional[ArrayLike] = None
        self._transform = transform

        # Assign provided arrays with validation; do not read files here.
        if data is not None:
            self._validate_and_set_data(data)
        if labels is not None:
            self._validate_and_set_labels(labels)

        if (self._data is not None) and (self._labels is not None):
            if len(self._data) != len(self._labels):
                logger.error(
                    "Data/labels length mismatch: %s vs %s",
                    len(self._data),
                    len(self._labels),
                )
                raise ValueError("`data` and `labels` must have the same length")

    # ------------------------------------------------------------------
    # Abstract API that concrete datasets must implement
    # ------------------------------------------------------------------
    @abc.abstractmethod
    def load_data(self) -> None:
        """Populate ``self._data`` and ``self._labels``.

        Subclasses implement dataset-specific loading (e.g. reading
        binary files, memory-mapping large arrays, or constructing
        synthetic datasets). The method must set ``self._data`` and
        optionally ``self._labels``. It should raise informative
        exceptions on failure.
        """

    @abc.abstractmethod
    def get_num_features(self) -> int:
        """Return the number of features for a single sample.

        This is useful for constructing model input layers without
        materialising a sample.
        """

    # --------------------
    # PyTorch Dataset API
    # --------------------
    def __len__(self) -> int:
        """Return number of samples (triggers lazy load).

        Using lazy loading keeps construction cheap and allows test
        code to override :meth:`load_data` behaviour.
        """
        self._ensure_data_loaded()
        assert self._data is not None
        return len(self._data)

    def __getitem__(self, index: int) -> Tuple[Tensor, Optional[Tensor]]:
        """Return (sample_tensor, label_tensor_or_None) for ``index``.

        The stored representation (ndarray or tensor) is converted to a
        :class:`torch.Tensor` on access so callers always receive a
        tensor. Transforms are applied after conversion.
        """
        self._ensure_data_loaded()

        if self._data is None:
            logger.exception("Data requested before dataset was loaded")
            raise RuntimeError("Dataset data is not available")

        if not (0 <= index < len(self._data)):
            logger.error("Index %s out of range [0, %s)", index, len(self._data))
            raise IndexError("Index out of range")

        try:
            sample = self._data[index]
        except Exception as exc:  # defensive programming
            logger.exception("Failed to retrieve sample at index %s", index)
            raise RuntimeError("Failed to retrieve sample") from exc

        label: Optional[Any] = None
        if self._labels is not None:
            try:
                label = self._labels[index]
            except Exception as exc:  # defensive programming
                logger.exception("Failed to retrieve label at index %s", index)
                raise RuntimeError("Failed to retrieve label") from exc

        sample_tensor = self._to_tensor(sample)
        label_tensor = self._to_tensor(label) if label is not None else None

        if self._transform is not None:
            try:
                sample_tensor = self._transform(sample_tensor)
            except Exception:
                logger.exception("Transform function failed for index %s", index)
                raise

        return sample_tensor, label_tensor

    # --------------------
    # Internal utilities
    # --------------------
    def _ensure_data_loaded(self) -> None:
        """Call :meth:`load_data` lazily when required.

        This avoids performing I/O in constructors where tests may want
        to control timing or substitute alternative data sources.
        """
        if self._data is None:
            logger.debug("Data missing; invoking load_data()")
            try:
                self.load_data()
            except Exception as exc:
                logger.exception("load_data() raised an exception")
                raise RuntimeError("Failed to load dataset") from exc

            if self._data is None:
                logger.error("load_data() did not populate self._data")
                raise RuntimeError("Dataset.load_data must populate self._data")

    def _validate_and_set_data(self, data: ArrayLike) -> None:
        """Validate externally-provided ``data`` and assign to ``_data``.

        Accepts numpy arrays and torch tensors. Rejects empty arrays and
        other types to surface errors early.
        """
        if isinstance(data, (np.ndarray, torch.Tensor)):
            if len(data) == 0:
                logger.error("Provided data array is empty")
                raise ValueError("`data` must not be empty")
            self._data = data
            logger.debug("Assigned pre-loaded data of length %s", len(data))
            return

        logger.error("Unsupported data type: %s", type(data))
        raise TypeError("`data` must be numpy.ndarray or torch.Tensor")

    def _validate_and_set_labels(self, labels: ArrayLike) -> None:
        """Validate externally-provided ``labels`` and assign to ``_labels``.

        Mirrors :meth:`_validate_and_set_data` behaviour for labels.
        """
        if isinstance(labels, (np.ndarray, torch.Tensor)):
            if len(labels) == 0:
                logger.error("Provided labels array is empty")
                raise ValueError("`labels` must not be empty")
            self._labels = labels
            logger.debug("Assigned pre-loaded labels of length %s", len(labels))
            return

        logger.error("Unsupported labels type: %s", type(labels))
        raise TypeError("`labels` must be numpy.ndarray or torch.Tensor")

    @staticmethod
    def _to_tensor(x: Any) -> Tensor:
        """Convert an ndarray/scalar/tensor to :class:`torch.Tensor`.

        The method purposefully keeps conversions explicit and raises
        :class:`TypeError` for unsupported types so callers can handle
        errors deterministically.
        """
        if isinstance(x, Tensor):
            return x

        if isinstance(x, np.ndarray):
            try:
                return torch.from_numpy(x)
            except Exception as exc:
                logger.exception("Failed to convert numpy.ndarray to tensor")
                raise TypeError("Could not convert numpy.ndarray to torch.Tensor") from exc

        if np.isscalar(x):
            try:
                return torch.tensor(x)
            except Exception as exc:
                logger.exception("Failed to convert scalar to tensor")
                raise TypeError("Could not convert scalar to torch.Tensor") from exc

        logger.error("Unsupported type for tensor conversion: %s", type(x))
        raise TypeError("Unsupported type for tensor conversion")

    # --------------------
    # Convenience helpers
    # --------------------
    @property
    def data(self) -> Optional[ArrayLike]:
        """Return internal feature storage (may be ndarray or tensor)."""
        return self._data

    @property
    def labels(self) -> Optional[ArrayLike]:
        """Return internal label storage (may be ndarray or tensor)."""
        return self._labels

    @property
    def num_samples(self) -> int:
        """Return dataset size (triggers lazy load)."""
        return len(self)

    def unique_labels(self) -> Optional[Tuple[Any, ...]]:
        """Return unique labels if available, otherwise ``None``.

        This is a convenience for evaluation code. The method converts
        tensors to numpy arrays when required.
        """
        self._ensure_data_loaded()
        if self._labels is None:
            return None

        try:
            if isinstance(self._labels, np.ndarray):
                uniques = tuple(np.unique(self._labels).tolist())
            else:
                uniques = tuple(np.unique(self._labels.cpu().numpy()).tolist())

            logger.debug("Computed %s unique labels", len(uniques))
            return uniques
        except Exception:
            logger.exception("Failed to compute unique labels")
            raise RuntimeError("Failed to compute unique labels") from None