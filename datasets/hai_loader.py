"""HAI CSV loader utilities.

This module loads the HAI test CSV pair, merges labels into the feature
dataframe, parses timestamps, and exposes a structured dataclass for
downstream processing.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import pandas as pd


logger = logging.getLogger(__name__)


@dataclass(slots=True)
class HAIData:
	"""Container for a loaded HAI dataset.

	Attributes:
		dataframe: The merged dataframe containing features and labels.
		timestamp_column: Name of the timestamp column.
		label_column: Name of the label column.
		feature_columns: Automatically detected feature columns.
		metadata: Additional load-time metadata.
	"""

	dataframe: pd.DataFrame
	timestamp_column: str
	label_column: str
	feature_columns: list[str]
	metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class HAILoader:
	"""Load and validate HAI test CSV files.

	The loader focuses on safe CSV ingestion and structural validation.
	It does not normalize, window, or engineer features.
	"""

	data_path: Path | str
	label_path: Path | str
	timestamp_column_name: str = "timestamp"
	label_column_name: str = "label"

	def __post_init__(self) -> None:
		"""Normalize the provided file paths."""
		self.data_path = Path(self.data_path).expanduser().resolve()
		self.label_path = Path(self.label_path).expanduser().resolve()

	def load(self) -> HAIData:
		"""Load the HAI feature and label CSV files.

		Returns:
			A ``HAIData`` instance containing the merged dataframe,
			timestamp and label column names, feature columns, and
			metadata.

		Raises:
			FileNotFoundError: If either CSV file does not exist.
			ValueError: If the files do not have the same row count or
				required columns are missing.
			RuntimeError: If loading or parsing fails.
		"""
		logger.info("Loading HAI feature file from %s", self.data_path)
		data_frame = self._read_csv(self.data_path)

		logger.info("Loading HAI label file from %s", self.label_path)
		label_frame = self._read_csv(self.label_path)

		self._validate_row_counts(data_frame, label_frame)

		data_frame = self._strip_column_whitespace(data_frame)
		label_frame = self._strip_column_whitespace(label_frame)

		timestamp_column = self._detect_timestamp_column(data_frame)
		label_column = self._detect_label_column(label_frame)

		if timestamp_column is None:
			logger.error("Could not detect a timestamp column in %s", self.data_path)
			raise ValueError("Could not detect a timestamp column in the HAI data file")

		if label_column is None:
			logger.error("Could not detect a label column in %s", self.label_path)
			raise ValueError("Could not detect a label column in the HAI label file")

		logger.info("Parsing timestamp column %s", timestamp_column)
		data_frame = self._parse_timestamp_column(data_frame, timestamp_column)

		self._validate_timestamp_alignment(data_frame, label_frame, timestamp_column)

		merged_frame = data_frame.copy()
		merged_frame[self.label_column_name] = label_frame[label_column].to_numpy()

		feature_columns = self._detect_feature_columns(
			columns=merged_frame.columns,
			timestamp_column=timestamp_column,
			label_column=self.label_column_name,
		)

		metadata = self._build_metadata(
			data_frame=data_frame,
			label_frame=label_frame,
			merged_frame=merged_frame,
			timestamp_column=timestamp_column,
			label_source_column=label_column,
			feature_columns=feature_columns,
		)

		logger.info(
			"Loaded HAI dataset with %s rows, %s feature columns, and label column %r",
			len(merged_frame),
			len(feature_columns),
			self.label_column_name,
		)
		return HAIData(
			dataframe=merged_frame,
			timestamp_column=timestamp_column,
			label_column=self.label_column_name,
			feature_columns=feature_columns,
			metadata=metadata,
		)

	def _read_csv(self, csv_path: Path | str) -> pd.DataFrame:
		"""Read a CSV file with robust error handling."""
		csv_path = Path(csv_path).expanduser().resolve()
		self._ensure_file_exists(csv_path)
		try:
			dataframe = pd.read_csv(csv_path)
		except Exception as exc:  # pragma: no cover - defensive logging
			logger.exception("Failed to read CSV file %s", csv_path)
			raise RuntimeError(f"Failed to read CSV file {csv_path}") from exc

		logger.debug("Loaded %s with shape %s", csv_path, dataframe.shape)
		return dataframe

	def _ensure_file_exists(self, csv_path: Path) -> None:
		"""Validate that a path exists and is a file."""
		logger.debug("Validating CSV path %s", csv_path)
		if not csv_path.exists():
			logger.error("CSV file does not exist: %s", csv_path)
			raise FileNotFoundError(f"CSV file does not exist: {csv_path}")

		if not csv_path.is_file():
			logger.error("CSV path is not a file: %s", csv_path)
			raise FileNotFoundError(f"CSV path is not a file: {csv_path}")

	def _validate_row_counts(self, data_frame: pd.DataFrame, label_frame: pd.DataFrame) -> None:
		"""Ensure the feature and label files have identical row counts."""
		if len(data_frame) != len(label_frame):
			logger.error(
				"Row count mismatch between data and label files: %s vs %s",
				len(data_frame),
				len(label_frame),
			)
			raise ValueError("HAI data and label files must have identical row counts")

		logger.info("Verified matching row counts: %s", len(data_frame))

	def _strip_column_whitespace(self, dataframe: pd.DataFrame) -> pd.DataFrame:
		"""Strip surrounding whitespace from column names."""
		cleaned = dataframe.copy()
		cleaned_columns = [str(column).strip() for column in cleaned.columns]

		if len(set(cleaned_columns)) != len(cleaned_columns):
			duplicates = sorted(
				{
					column
					for column in cleaned_columns
					if cleaned_columns.count(column) > 1
				}
			)
			logger.error("Duplicate column names after stripping whitespace: %s", duplicates)
			raise ValueError(f"Duplicate column names after stripping whitespace: {duplicates}")

		cleaned.columns = cleaned_columns
		return cleaned

	def _detect_timestamp_column(self, dataframe: pd.DataFrame) -> str | None:
		"""Detect a timestamp column by name."""
		for column in dataframe.columns:
			if self._normalize_name(column) == self.timestamp_column_name.lower():
				logger.debug("Detected timestamp column %s", column)
				return column

		logger.debug("No timestamp column detected in %s", self.data_path)
		return None

	def _detect_label_column(self, dataframe: pd.DataFrame) -> str | None:
		"""Detect a label column by name."""
		for column in dataframe.columns:
			if self._normalize_name(column) == self.label_column_name.lower():
				logger.debug("Detected label column %s", column)
				return column

		logger.debug("No label column detected in %s", self.label_path)
		return None

	def _parse_timestamp_column(self, dataframe: pd.DataFrame, timestamp_column: str) -> pd.DataFrame:
		"""Parse the timestamp column into datetimes."""
		parsed = dataframe.copy()
		parsed[timestamp_column] = pd.to_datetime(parsed[timestamp_column], errors="coerce")

		null_count = int(parsed[timestamp_column].isna().sum())
		if null_count:
			logger.warning(
				"Timestamp parsing produced %s NaT value(s) in %s",
				null_count,
				timestamp_column,
			)

		return parsed

	def _validate_timestamp_alignment(
		self,
		data_frame: pd.DataFrame,
		label_frame: pd.DataFrame,
		timestamp_column: str,
	) -> None:
		"""Ensure timestamps line up across the two files when both expose timestamps."""
		data_timestamp_column = timestamp_column
		label_timestamp_column = self._detect_timestamp_column(label_frame)

		if label_timestamp_column is None:
			logger.info("Label file does not expose a timestamp column; skipping alignment check")
			return

		data_values = pd.to_datetime(data_frame[data_timestamp_column], errors="coerce")
		label_values = pd.to_datetime(label_frame[label_timestamp_column], errors="coerce")

		if not data_values.reset_index(drop=True).equals(label_values.reset_index(drop=True)):
			logger.error("Timestamp columns in data and label files do not match")
			raise ValueError("Timestamp columns in HAI files do not align")

		logger.info("Verified timestamp alignment between data and label files")

	def _detect_feature_columns(
		self,
		columns: Iterable[str],
		timestamp_column: str,
		label_column: str,
	) -> list[str]:
		"""Detect feature columns by excluding timestamp and label columns."""
		feature_columns = [
			column
			for column in columns
			if column not in {timestamp_column, label_column}
		]

		logger.debug("Detected %s feature column(s)", len(feature_columns))
		return feature_columns

	def _build_metadata(
		self,
		data_frame: pd.DataFrame,
		label_frame: pd.DataFrame,
		merged_frame: pd.DataFrame,
		timestamp_column: str,
		label_source_column: str,
		feature_columns: list[str],
	) -> dict[str, Any]:
		"""Build metadata for the loaded HAI dataset."""
		metadata = {
			"data_path": str(self.data_path),
			"label_path": str(self.label_path),
			"row_count": int(len(merged_frame)),
			"column_count": int(len(merged_frame.columns)),
			"timestamp_column": timestamp_column,
			"label_column": self.label_column_name,
			"label_source_column": label_source_column,
			"feature_columns": list(feature_columns),
			"data_columns": list(data_frame.columns),
			"label_columns": list(label_frame.columns),
			"dtypes": {column: str(dtype) for column, dtype in merged_frame.dtypes.items()},
		}
		logger.debug("Built HAI metadata")
		return metadata

	@staticmethod
	def _normalize_name(value: Any) -> str:
		"""Normalize a column name for comparison."""
		return str(value).strip().lower()


__all__ = ["HAIData", "HAILoader"]
