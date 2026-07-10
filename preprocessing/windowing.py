"""Reusable sliding-window utilities for industrial control system time-series.

The :class:`SlidingWindowGenerator` converts cleaned pandas ``DataFrame``
objects from src.data such as SWaT and HAI into overlapping window tensors
suitable for sequence modelling.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view


logger = logging.getLogger(__name__)

LabelAggregationMethod = Literal["last", "max", "majority"]

_DEFAULT_TIMESTAMP_CANDIDATES = (
	"timestamp",
	"t_stamp",
	"time",
	"datetime",
	"date_time",
)
_DEFAULT_LABEL_CANDIDATES = (
	"label",
	"labels",
	"class",
	"target",
	"attack",
)


@dataclass(slots=True)
class WindowBatch:
	"""Container for generated sliding windows.

	Attributes:
		windows: Window tensor with shape ``(num_windows, window_size, num_features)``.
		timestamps: End timestamp for each window, or ``None`` when not requested.
		labels: Aggregated label for each window, or ``None`` when not requested.
		metadata: Generation metadata describing columns and parameters used.
	"""

	windows: np.ndarray
	timestamps: np.ndarray | None
	labels: np.ndarray | None
	metadata: dict[str, Any]


class SlidingWindowGenerator:
	"""Generate overlapping sliding windows from ICS time-series data.

	The generator is dataset-agnostic and works with SWaT (``t_stamp``,
	optional label) and HAI (``timestamp``, ``label``) frames as long as
	the timestamp and optional label columns are supplied or detected.
	"""

	def __init__(
		self,
		window_size: int,
		stride: int = 1,
		drop_last: bool = False,
		padding: int = 0,
		return_timestamp: bool = True,
		return_labels: bool = False,
		label_method: LabelAggregationMethod = "last",
		timestamp_column: str | None = None,
		label_column: str | None = None,
	) -> None:
		"""Initialise the sliding-window generator.

		Args:
			window_size: Number of rows in each window.
			stride: Step size between consecutive window start indices.
			drop_last: When ``True``, drop the final window if the span
				between the first and last valid start index is not divisible
				by ``stride``.
			padding: Number of rows to edge-pad at the end of the series
				before windowing. Useful when ``window_size`` exceeds the
				available rows or when the final stride-aligned window would
				otherwise extend past the end of the data.
			return_timestamp: Include end timestamps in the result.
			return_labels: Include aggregated window labels in the result.
			label_method: Strategy for collapsing row labels into one label
				per window. One of ``last``, ``max``, or ``majority``.
			timestamp_column: Optional explicit timestamp column name.
			label_column: Optional explicit label column name.

		Raises:
			ValueError: If parameters are invalid or ``label_method`` is unknown.
			TypeError: If ``window_size`` or ``stride`` are not integers.
		"""
		self._validate_positive_int(window_size, name="window_size")
		self._validate_positive_int(stride, name="stride")
		if isinstance(padding, bool) or not isinstance(padding, int):
			raise TypeError(
				f"padding must be a non-negative integer, got {type(padding).__name__}"
			)
		if padding < 0:
			raise ValueError(f"padding must be zero or greater, got {padding}")

		label_methods = {"last", "max", "majority"}
		if label_method not in label_methods:
			raise ValueError(
				f"label_method must be one of {sorted(label_methods)}, got {label_method!r}"
			)

		self.window_size = window_size
		self.stride = stride
		self.drop_last = drop_last
		self.padding = padding
		self.return_timestamp = return_timestamp
		self.return_labels = return_labels
		self.label_method: LabelAggregationMethod = label_method
		self.timestamp_column = timestamp_column
		self.label_column = label_column

	def generate(self, dataframe: pd.DataFrame) -> WindowBatch:
		"""Generate sliding windows from ``dataframe``.

		Args:
			dataframe: Input time-series frame with a timestamp column and
				optional label column.

		Returns:
			A :class:`WindowBatch` containing window tensors and optional
			timestamps and labels.

		Raises:
			TypeError: If ``dataframe`` is not a pandas DataFrame.
			ValueError: If required columns are missing, the frame is empty,
				``window_size`` exceeds the number of rows, or feature columns
				cannot be converted to numeric values.
		"""
		if not isinstance(dataframe, pd.DataFrame):
			raise TypeError("dataframe must be a pandas DataFrame")

		if dataframe.empty:
			raise ValueError("dataframe must contain at least one row")

		working = dataframe.reset_index(drop=True)

		timestamp_column = self._resolve_column(
			working,
			explicit_name=self.timestamp_column,
			candidates=_DEFAULT_TIMESTAMP_CANDIDATES,
			required=True,
			column_role="timestamp",
		)
		label_column = self._resolve_column(
			working,
			explicit_name=self.label_column,
			candidates=_DEFAULT_LABEL_CANDIDATES,
			required=self.return_labels,
			column_role="label",
		)

		original_rows = len(working)
		if self.padding:
			working = self._pad_dataframe(
				working,
				timestamp_column=timestamp_column,
				label_column=label_column,
			)
			logger.info(
				"Applied end padding of %s row(s); effective length %s -> %s",
				self.padding,
				original_rows,
				len(working),
			)

		num_rows = len(working)
		self._validate_window_size(num_rows, original_rows=original_rows)

		feature_columns = [
			column
			for column in working.columns
			if column not in {timestamp_column, label_column}
		]
		if not feature_columns:
			raise ValueError(
				"dataframe must contain at least one feature column besides "
				"the timestamp and label columns"
			)

		feature_values = self._to_numeric_feature_matrix(working, feature_columns)
		timestamp_values = working[timestamp_column].to_numpy()
		label_values = (
			working[label_column].to_numpy()
			if label_column is not None
			else None
		)

		start_indices = self._compute_start_indices(num_rows)
		num_windows = len(start_indices)
		if num_windows == 0:
			raise ValueError(
				f"no windows could be generated for {num_rows} row(s) with "
				f"window_size={self.window_size}, stride={self.stride}, "
				f"drop_last={self.drop_last}"
			)

		windows = self._build_windows(feature_values, start_indices)
		end_indices = start_indices + self.window_size - 1

		timestamps = (
			timestamp_values[end_indices]
			if self.return_timestamp
			else None
		)
		labels = (
			self._aggregate_labels(label_values, start_indices)
			if self.return_labels and label_values is not None
			else None
		)

		metadata = {
			"input_rows": original_rows,
			"effective_rows": num_rows,
			"num_windows": num_windows,
			"window_size": self.window_size,
			"stride": self.stride,
			"drop_last": self.drop_last,
			"padding": self.padding,
			"padded_rows": self.padding,
			"label_method": self.label_method,
			"timestamp_column": timestamp_column,
			"label_column": label_column,
			"feature_columns": feature_columns,
			"num_features": len(feature_columns),
			"start_indices": start_indices.tolist(),
			"end_indices": end_indices.tolist(),
		}

		logger.info(
			"Generated %s window(s) with shape (%s, %s, %s) from %s row(s)",
			num_windows,
			num_windows,
			self.window_size,
			len(feature_columns),
			num_rows,
		)

		return WindowBatch(
			windows=windows,
			timestamps=timestamps,
			labels=labels,
			metadata=metadata,
		)

	def _validate_window_size(self, num_rows: int, original_rows: int) -> None:
		"""Ensure the configured window fits the available rows."""
		if self.window_size > num_rows:
			shortfall = self.window_size - num_rows
			padding_hint = (
				f" Increase padding to at least {shortfall} to edge-pad the series."
				if self.padding == 0
				else f" Current padding ({self.padding}) is insufficient; "
				f"increase it by at least {shortfall}."
			)
			raise ValueError(
				f"window_size ({self.window_size}) cannot exceed the effective "
				f"number of rows ({num_rows}; original rows={original_rows})."
				f"{padding_hint}"
			)

		if self.window_size == num_rows and self.stride > 1:
			logger.warning(
				"stride=%s is greater than 1 while window_size equals the "
				"number of rows (%s); only one window can be produced",
				self.stride,
				num_rows,
			)

	def _compute_start_indices(self, num_rows: int) -> np.ndarray:
		"""Return valid window start indices for the configured parameters."""
		max_start = num_rows - self.window_size
		start_indices = np.arange(0, max_start + 1, self.stride, dtype=np.int64)

		if self.drop_last and start_indices.size and max_start % self.stride != 0:
			start_indices = start_indices[:-1]
			logger.debug(
				"Dropped final window because (num_rows - window_size) %% stride != 0 "
				"(%s %% %s = %s)",
				max_start,
				self.stride,
				max_start % self.stride,
			)

		return start_indices

	def _build_windows(
		self,
		feature_values: np.ndarray,
		start_indices: np.ndarray,
	) -> np.ndarray:
		"""Build window tensors from feature values and start indices."""
		all_windows = sliding_window_view(
			feature_values,
			window_shape=self.window_size,
			axis=0,
		)
		# NumPy inserts the window axis after the remaining feature axes.
		windows = np.moveaxis(all_windows[start_indices], -1, 1)
		return np.asarray(windows, dtype=np.float64).copy()

	def _aggregate_labels(
		self,
		label_values: np.ndarray,
		start_indices: np.ndarray,
	) -> np.ndarray:
		"""Aggregate row-level labels into one label per window."""
		label_windows = sliding_window_view(
			label_values,
			window_shape=self.window_size,
		)
		selected_windows = np.asarray(label_windows[start_indices])

		if self.label_method == "last":
			return selected_windows[:, -1].copy()

		numeric_labels = pd.to_numeric(
			selected_windows.reshape(-1),
			errors="coerce",
		).reshape(selected_windows.shape)

		if self.label_method == "max":
			if np.isnan(numeric_labels).any():
				raise ValueError(
					"label_method 'max' requires numeric labels; found non-numeric values"
				)
			return numeric_labels.max(axis=1)

		if self.label_method == "majority":
			return np.apply_along_axis(_majority_value, 1, selected_windows)

		raise ValueError(f"Unsupported label_method: {self.label_method!r}")

	def _pad_dataframe(
		self,
		dataframe: pd.DataFrame,
		timestamp_column: str,
		label_column: str | None,
	) -> pd.DataFrame:
		"""Edge-pad rows at the end of the series."""
		if self.padding == 0:
			return dataframe

		padded = dataframe.copy(deep=True)
		last_row = padded.iloc[[-1]].copy()

		timestamp_series = pd.to_datetime(padded[timestamp_column], errors="coerce")
		if timestamp_series.notna().sum() >= 2:
			inferred_freq = pd.infer_freq(timestamp_series.dropna())
			if inferred_freq is not None:
				last_timestamp = timestamp_series.iloc[-1]
				padded_timestamps = pd.date_range(
					start=last_timestamp,
					periods=self.padding + 1,
					freq=inferred_freq,
				)[1:]
				pad_frame = pd.concat([last_row] * self.padding, ignore_index=True)
				pad_frame[timestamp_column] = padded_timestamps.to_numpy()
				padded = pd.concat([padded, pad_frame], ignore_index=True)
				return padded

		padded = pd.concat([padded] + [last_row] * self.padding, ignore_index=True)
		return padded

	@staticmethod
	def _validate_positive_int(value: int, name: str) -> None:
		"""Validate that ``value`` is a positive integer parameter."""
		if isinstance(value, bool) or not isinstance(value, int):
			raise TypeError(f"{name} must be a positive integer, got {type(value).__name__}")
		if value <= 0:
			raise ValueError(f"{name} must be greater than zero, got {value}")

	@staticmethod
	def _resolve_column(
		dataframe: pd.DataFrame,
		explicit_name: str | None,
		candidates: tuple[str, ...],
		required: bool,
		column_role: str,
	) -> str | None:
		"""Resolve a timestamp or label column from explicit input or candidates."""
		if explicit_name is not None:
			if explicit_name in dataframe.columns:
				return explicit_name

			normalized_explicit = explicit_name.strip().lower()
			for column in dataframe.columns:
				if str(column).strip().lower() == normalized_explicit:
					return column

			raise ValueError(
				f"{column_role} column {explicit_name!r} was not found in the DataFrame"
			)

		for candidate in candidates:
			normalized_candidate = candidate.strip().lower()
			for column in dataframe.columns:
				if str(column).strip().lower() == normalized_candidate:
					return column

		if required:
			raise ValueError(
				f"Could not detect a {column_role} column; available columns: "
				f"{list(dataframe.columns)}"
			)

		return None

	@staticmethod
	def _to_numeric_feature_matrix(
		dataframe: pd.DataFrame,
		feature_columns: list[str],
	) -> np.ndarray:
		"""Convert feature columns to a numeric numpy matrix."""
		feature_frame = dataframe.loc[:, feature_columns].apply(
			pd.to_numeric,
			errors="coerce",
		)
		missing_mask = feature_frame.isna()
		if missing_mask.any().any():
			invalid_columns = [
				column
				for column in feature_columns
				if missing_mask[column].any()
			]
			raise ValueError(
				"Feature columns contain non-numeric or missing values after conversion: "
				f"{invalid_columns}"
			)

		return feature_frame.to_numpy(dtype=np.float64, copy=True)


def _majority_value(window_labels: np.ndarray) -> Any:
	"""Return the most frequent label inside a single window."""
	labels, counts = np.unique(window_labels, return_counts=True)
	return labels[int(counts.argmax())]


__all__ = [
	"LabelAggregationMethod",
	"SlidingWindowGenerator",
	"WindowBatch",
]
