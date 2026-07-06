"""PyTorch Dataset for contrastive learning on ICS time-series windows.

The :class:`ContrastiveDataset` wraps sliding windows produced by
:class:`preprocessing.windowing.SlidingWindowGenerator` and returns
PyTorch tensors suitable for self-supervised contrastive learning.

Each sample is a dictionary containing the window tensor and optional
label tensor and timestamp, following the convention used by modern
self-supervised frameworks.
"""

from __future__ import annotations

import logging
from typing import Any, Optional, Union

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


class ContrastiveDataset(Dataset):
    """PyTorch Dataset for contrastive learning on multivariate ICS windows.

    The dataset accepts sliding windows of shape
    ``(num_windows, window_size, num_features)`` produced by the
    preprocessing pipeline and returns dictionary samples with:

    - ``"window"``: ``torch.float32`` tensor of shape
      ``(window_size, num_features)``
    - ``"label"``: ``torch.long`` tensor (scalar), if labels are provided
    - ``"timestamp"``: original timestamp value, if timestamps are provided

    Attributes:
        num_windows: Total number of sliding windows.
        window_size: Number of time steps per window.
        num_features: Number of sensor/actuator features per time step.
        has_labels: Whether the dataset includes anomaly labels.
        has_timestamps: Whether the dataset includes window timestamps.
    """

    def __init__(
        self,
        windows: Union[NDArray[np.floating], Tensor],
        labels: Optional[Union[NDArray[Any], Tensor]] = None,
        timestamps: Optional[NDArray[Any]] = None,
    ) -> None:
        """Initialise the contrastive dataset.

        Args:
            windows: Sliding window array of shape
                ``(num_windows, window_size, num_features)``.
            labels: Optional per-window anomaly labels aligned with
                ``windows``.  Will be converted to ``torch.long``.
            timestamps: Optional per-window timestamps aligned with
                ``windows``.  Preserved as-is (not converted to tensors).

        Raises:
            TypeError: If ``windows`` is not a numpy array or torch Tensor.
            ValueError: If ``windows`` is empty, not 3-dimensional, or if
                ``labels``/``timestamps`` lengths do not match the number
                of windows.
        """
        self._validate_windows(windows)

        if labels is not None:
            self._validate_labels(labels, expected_length=len(windows))

        if timestamps is not None:
            self._validate_timestamps(timestamps, expected_length=len(windows))

        # ── Convert and store tensors ──────────────────────────────────
        self._windows: Tensor = self._to_float32_tensor(windows)
        self._labels: Optional[Tensor] = (
            self._to_long_tensor(labels) if labels is not None else None
        )
        # Store timestamps as string representations so that PyTorch's
        # default_collate can handle them in batched DataLoaders.
        # numpy.datetime64 and datetime.datetime are not collatable.
        if timestamps is not None:
            ts_array = np.asarray(timestamps)
            self._timestamps_list: Optional[list[Any]] = [
                str(t) for t in ts_array
            ]
        else:
            self._timestamps_list = None

        # ── Cache shape metadata ───────────────────────────────────────
        self.num_windows: int = self._windows.shape[0]
        self.window_size: int = self._windows.shape[1]
        self.num_features: int = self._windows.shape[2]
        self.has_labels: bool = self._labels is not None
        self.has_timestamps: bool = self._timestamps_list is not None

        # ── Post-conversion shape guard ────────────────────────────────
        self._validate_tensor_shape(self._windows)

        logger.info(
            "ContrastiveDataset created: %d windows | "
            "window_size=%d | num_features=%d | dtype=%s | "
            "labels=%s | timestamps=%s",
            self.num_windows,
            self.window_size,
            self.num_features,
            self._windows.dtype,
            self.has_labels,
            self.has_timestamps,
        )

    # ------------------------------------------------------------------
    # PyTorch Dataset API
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        """Return the total number of windows in the dataset."""
        return self.num_windows

    def __getitem__(self, index: int) -> dict[str, Any]:
        """Return a single sample as a dictionary.

        Args:
            index: Zero-based sample index.

        Returns:
            A dictionary with keys ``"window"`` (always),
            ``"label"`` (if labels exist), and ``"timestamp"``
            (if timestamps exist).

        Raises:
            IndexError: If ``index`` is out of range.
        """
        if not 0 <= index < self.num_windows:
            raise IndexError(
                f"Index {index} is out of range for dataset "
                f"with {self.num_windows} samples"
            )

        sample: dict[str, Any] = {
            "window": self._windows[index],
        }

        if self._labels is not None:
            sample["label"] = self._labels[index]

        if self._timestamps_list is not None:
            sample["timestamp"] = self._timestamps_list[index]

        return sample

    # ------------------------------------------------------------------
    # Convenience properties
    # ------------------------------------------------------------------

    @property
    def windows(self) -> Tensor:
        """Return the full windows tensor."""
        return self._windows

    @property
    def labels(self) -> Optional[Tensor]:
        """Return the labels tensor, or ``None``."""
        return self._labels

    @property
    def timestamps(self) -> Optional[list[Any]]:
        """Return the timestamps list, or ``None``."""
        return self._timestamps_list

    @property
    def shape(self) -> tuple[int, int, int]:
        """Return the ``(num_windows, window_size, num_features)`` shape."""
        return (self.num_windows, self.window_size, self.num_features)

    # ------------------------------------------------------------------
    # Validation helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_windows(
        windows: Union[NDArray[np.floating], Tensor],
    ) -> None:
        """Validate raw window input before conversion.

        Args:
            windows: Window array or tensor to validate.

        Raises:
            TypeError: If ``windows`` is not a supported array type.
            ValueError: If ``windows`` is empty or not 3-dimensional.
        """
        if not isinstance(windows, (np.ndarray, Tensor)):
            raise TypeError(
                f"windows must be a numpy.ndarray or torch.Tensor, "
                f"got {type(windows).__name__}"
            )

        if windows.ndim != 3:
            raise ValueError(
                f"windows must be 3-dimensional "
                f"(num_windows, window_size, num_features), "
                f"got {windows.ndim}D with shape {tuple(windows.shape)}"
            )

        if len(windows) == 0:
            raise ValueError(
                "windows must contain at least one sample; got 0 windows"
            )

        if windows.shape[1] == 0:
            raise ValueError(
                f"window_size must be at least 1; got shape {tuple(windows.shape)}"
            )

        if windows.shape[2] == 0:
            raise ValueError(
                f"num_features must be at least 1; got shape {tuple(windows.shape)}"
            )

    @staticmethod
    def _validate_labels(
        labels: Union[NDArray[Any], Tensor],
        expected_length: int,
    ) -> None:
        """Validate label array dimensions and length.

        Args:
            labels: Label array or tensor.
            expected_length: Expected number of labels (must match windows).

        Raises:
            TypeError: If ``labels`` is not a supported array type.
            ValueError: If label length does not match ``expected_length``.
        """
        if not isinstance(labels, (np.ndarray, Tensor)):
            raise TypeError(
                f"labels must be a numpy.ndarray or torch.Tensor, "
                f"got {type(labels).__name__}"
            )

        if len(labels) != expected_length:
            raise ValueError(
                f"labels length ({len(labels)}) does not match "
                f"the number of windows ({expected_length})"
            )

    @staticmethod
    def _validate_timestamps(
        timestamps: NDArray[Any],
        expected_length: int,
    ) -> None:
        """Validate timestamp array length.

        Args:
            timestamps: Timestamp array.
            expected_length: Expected number of timestamps.

        Raises:
            TypeError: If ``timestamps`` is not array-like.
            ValueError: If length does not match ``expected_length``.
        """
        if not isinstance(timestamps, (np.ndarray, list, tuple)):
            raise TypeError(
                f"timestamps must be a numpy.ndarray, list, or tuple, "
                f"got {type(timestamps).__name__}"
            )

        if len(timestamps) != expected_length:
            raise ValueError(
                f"timestamps length ({len(timestamps)}) does not match "
                f"the number of windows ({expected_length})"
            )

    @staticmethod
    def _validate_tensor_shape(tensor: Tensor) -> None:
        """Validate the converted tensor has a valid shape.

        Args:
            tensor: Converted window tensor.

        Raises:
            ValueError: If the tensor contains NaN values.
        """
        if torch.isnan(tensor).any():
            nan_count = int(torch.isnan(tensor).sum().item())
            raise ValueError(
                f"Window tensor contains {nan_count} NaN value(s); "
                f"clean the data before creating the dataset"
            )

    # ------------------------------------------------------------------
    # Conversion helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _to_float32_tensor(
        array: Union[NDArray[np.floating], Tensor],
    ) -> Tensor:
        """Convert an array to a ``torch.float32`` tensor.

        Args:
            array: Input numpy array or torch tensor.

        Returns:
            A contiguous ``torch.float32`` tensor.
        """
        if isinstance(array, Tensor):
            return array.to(dtype=torch.float32).contiguous()

        return torch.from_numpy(
            np.ascontiguousarray(array, dtype=np.float32)
        )

    @staticmethod
    def _to_long_tensor(
        array: Union[NDArray[Any], Tensor],
    ) -> Tensor:
        """Convert a label array to a ``torch.long`` tensor.

        Args:
            array: Input numpy array or torch tensor.

        Returns:
            A ``torch.long`` tensor.
        """
        if isinstance(array, Tensor):
            return array.to(dtype=torch.long)

        return torch.from_numpy(
            np.asarray(array, dtype=np.int64)
        )


__all__ = ["ContrastiveDataset"]
