"""Reusable cleaning utilities for industrial control system datasets.

The :class:`DataCleaner` class performs conservative, logged cleaning on
incoming pandas ``DataFrame`` objects while preserving timestamp and label
columns when present.
"""

from __future__ import annotations

import logging
from typing import Any, Literal

import numpy as np
import pandas as pd


logger = logging.getLogger(__name__)

MissingValueStrategy = Literal[
	"forward_fill",
	"backward_fill",
	"mean",
	"median",
	"interpolate",
]


class DataCleaner:
	"""Clean industrial control system datasets in a reusable way.

	The cleaner is intentionally dataset-agnostic. It preserves timestamp and
	label columns when they can be detected, normalizes boolean and alarm-like
	values, converts numeric strings to numeric dtypes, handles missing values,
	and returns both the cleaned frame and a metadata dictionary describing the
	performed operations.
	"""

	_ACTIVE_TOKEN = "active"
	_INACTIVE_TOKEN = "inactive"
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

	def __init__(
		self,
		missing_value_strategy: MissingValueStrategy = "forward_fill",
		timestamp_column: str | None = None,
		label_column: str | None = None,
	) -> None:
		"""Initialise the cleaner.

		Args:
			missing_value_strategy: Strategy used to fill missing values in
				non-preserved columns.
			timestamp_column: Optional explicit timestamp column name.
			label_column: Optional explicit label column name.

		Raises:
			ValueError: If the requested missing-value strategy is unknown.
		"""
		strategies = {
			"forward_fill",
			"backward_fill",
			"mean",
			"median",
			"interpolate",
		}
		if missing_value_strategy not in strategies:
			raise ValueError(
				"missing_value_strategy must be one of "
				f"{sorted(strategies)}"
			)

		self.missing_value_strategy: MissingValueStrategy = missing_value_strategy
		self.timestamp_column = timestamp_column
		self.label_column = label_column

	def clean(self, dataframe: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
		"""Clean ``dataframe`` and return the result plus metadata.

		Args:
			dataframe: Input pandas DataFrame to clean.

		Returns:
			A tuple of ``(cleaned_dataframe, metadata)``.

		Raises:
			TypeError: If ``dataframe`` is not a pandas DataFrame.
		"""
		if not isinstance(dataframe, pd.DataFrame):
			raise TypeError("dataframe must be a pandas DataFrame")

		working = dataframe.copy(deep=True)

		timestamp_column = self._resolve_column(
			working,
			explicit_name=self.timestamp_column,
			candidates=self._DEFAULT_TIMESTAMP_CANDIDATES,
		)
		label_column = self._resolve_column(
			working,
			explicit_name=self.label_column,
			candidates=self._DEFAULT_LABEL_CANDIDATES,
		)

		preserved_columns = {
			column for column in (timestamp_column, label_column) if column is not None
		}

		operations: list[dict[str, Any]] = []
		metadata: dict[str, Any] = {
			"input_shape": tuple(working.shape),
			"timestamp_column": timestamp_column,
			"label_column": label_column,
			"missing_value_strategy": self.missing_value_strategy,
			"preserved_columns": sorted(preserved_columns),
			"operations": operations,
		}

		logger.info(
			"Starting cleaning for frame with shape %s using strategy %s",
			working.shape,
			self.missing_value_strategy,
		)

		working = self._normalize_empty_strings(working)

		converted_active_inactive = self._convert_active_inactive_columns(
			working,
			preserved_columns=preserved_columns,
		)
		if converted_active_inactive:
			self._log_operation(
				operations,
				"Converted Active/Inactive alarm columns to 1/0",
				columns=converted_active_inactive,
			)

		converted_boolean = self._convert_boolean_columns(
			working,
			preserved_columns=preserved_columns,
		)
		if converted_boolean:
			self._log_operation(
				operations,
				"Converted boolean columns to integer dtype",
				columns=converted_boolean,
			)

		converted_numeric_strings = self._convert_numeric_string_columns(
			working,
			preserved_columns=preserved_columns,
		)
		if converted_numeric_strings:
			self._log_operation(
				operations,
				"Converted numeric strings to numeric dtype",
				columns=converted_numeric_strings,
			)

		infinite_columns, infinite_count = self._detect_and_replace_infinite_values(
			working,
			preserved_columns=preserved_columns,
		)
		if infinite_count:
			self._log_operation(
				operations,
				"Detected and replaced infinite values with missing values",
				columns=infinite_columns,
				count=infinite_count,
			)

		empty_rows_removed = self._drop_completely_empty_rows(working)
		if empty_rows_removed:
			self._log_operation(
				operations,
				"Removed completely empty rows",
				rows_removed=empty_rows_removed,
			)

		duplicate_rows_removed = self._drop_duplicate_timestamps(
			working,
			timestamp_column=timestamp_column,
		)
		if duplicate_rows_removed:
			self._log_operation(
				operations,
				"Removed duplicate timestamp rows",
				rows_removed=duplicate_rows_removed,
				timestamp_column=timestamp_column,
			)

		missing_before = int(working.isna().sum().sum())
		missing_by_column_before = self._missing_values_by_column(working)
		if missing_before:
			self._fill_missing_values(
				working,
				preserved_columns=preserved_columns,
				strategy=self.missing_value_strategy,
			)
			self._log_operation(
				operations,
				"Applied missing-value handling",
				strategy=self.missing_value_strategy,
				missing_values_before=missing_before,
				missing_by_column_before=missing_by_column_before,
			)

		remaining_missing_by_column = self._missing_values_by_column(working)
		remaining_missing = int(sum(remaining_missing_by_column.values()))
		if remaining_missing:
			logger.warning(
				"Missing values remain after cleaning; applying directional fallback fill"
			)
			self._fallback_fill_remaining_missing(working, preserved_columns=preserved_columns)
			remaining_missing_by_column = self._missing_values_by_column(working)
			remaining_missing = int(sum(remaining_missing_by_column.values()))

		metadata.update(
			{
				"output_shape": tuple(working.shape),
				"rows_removed_empty": int(empty_rows_removed),
				"rows_removed_duplicate_timestamps": int(duplicate_rows_removed),
				"converted_active_inactive_columns": converted_active_inactive,
				"converted_boolean_columns": converted_boolean,
				"converted_numeric_string_columns": converted_numeric_strings,
				"infinite_value_columns": infinite_columns,
				"infinite_value_count": int(infinite_count),
				"missing_values_before": int(missing_before),
				"missing_values_remaining": int(remaining_missing),
				"missing_values_remaining_by_column": remaining_missing_by_column,
				"final_dtypes": {
					column: str(dtype)
					for column, dtype in working.dtypes.items()
				},
			}
		)

		logger.info(
			"Finished cleaning: %s -> %s rows, %s -> %s columns",
			metadata["input_shape"][0],
			metadata["output_shape"][0],
			metadata["input_shape"][1],
			metadata["output_shape"][1],
		)

		return working, metadata

	def _resolve_column(
		self,
		dataframe: pd.DataFrame,
		explicit_name: str | None,
		candidates: tuple[str, ...],
	) -> str | None:
		"""Resolve a preferred column name from explicit input or candidates."""
		if explicit_name is not None:
			if explicit_name in dataframe.columns:
				return explicit_name

			normalized_explicit = explicit_name.strip().lower()
			for column in dataframe.columns:
				if str(column).strip().lower() == normalized_explicit:
					return column

			raise ValueError(f"Column {explicit_name!r} was not found in the DataFrame")

		for candidate in candidates:
			normalized_candidate = candidate.strip().lower()
			for column in dataframe.columns:
				if str(column).strip().lower() == normalized_candidate:
					return column

		return None

	@staticmethod
	def _normalize_empty_strings(dataframe: pd.DataFrame) -> pd.DataFrame:
		"""Convert blank string values to missing values."""
		logger.info("Normalizing blank string values to missing values")
		return dataframe.replace(r"^\s*$", np.nan, regex=True)

	def _convert_active_inactive_columns(
		self,
		dataframe: pd.DataFrame,
		preserved_columns: set[str],
	) -> list[str]:
		"""Convert Active/Inactive string columns to binary integer columns."""
		converted_columns: list[str] = []
		token_map = {
			self._ACTIVE_TOKEN: 1,
			self._INACTIVE_TOKEN: 0,
		}

		for column in dataframe.columns:
			if column in preserved_columns:
				continue

			series = dataframe[column]
			if not (pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series)):
				continue

			normalized_values = {
				str(value).strip().lower()
				for value in series.dropna().unique().tolist()
			}
			if not normalized_values:
				continue

			if normalized_values.issubset(token_map.keys()):
				dataframe[column] = series.map(
					lambda value: token_map[str(value).strip().lower()]
					if pd.notna(value)
					else np.nan
				).astype("Int64")
				converted_columns.append(column)

		if converted_columns:
			logger.info(
				"Converted Active/Inactive values in columns: %s",
				converted_columns,
			)

		return converted_columns

	def _convert_boolean_columns(
		self,
		dataframe: pd.DataFrame,
		preserved_columns: set[str],
	) -> list[str]:
		"""Convert boolean dtype columns to integer dtype."""
		converted_columns: list[str] = []

		for column in dataframe.columns:
			if column in preserved_columns:
				continue

			series = dataframe[column]
			non_missing_values = series.dropna()
			if non_missing_values.empty:
				continue

			is_native_boolean = pd.api.types.is_bool_dtype(series)
			is_object_boolean = non_missing_values.map(
				lambda value: isinstance(value, (bool, np.bool_))
			).all()

			if not (is_native_boolean or is_object_boolean):
				continue

			dataframe[column] = series.map(
				lambda value: 1 if value is True else 0 if value is False else np.nan
			).astype("Int64")
			converted_columns.append(column)

		if converted_columns:
			logger.info("Converted boolean columns to integers: %s", converted_columns)

		return converted_columns

	def _convert_numeric_string_columns(
		self,
		dataframe: pd.DataFrame,
		preserved_columns: set[str],
	) -> list[str]:
		"""Convert object/string columns that contain numeric strings."""
		converted_columns: list[str] = []

		for column in dataframe.columns:
			if column in preserved_columns:
				continue

			series = dataframe[column]
			if not (pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series)):
				continue

			non_missing_mask = series.notna()
			non_missing_count = int(non_missing_mask.sum())
			if non_missing_count == 0:
				continue

			converted = pd.to_numeric(series, errors="coerce")
			if int(converted.notna().sum()) == non_missing_count:
				dataframe[column] = converted
				converted_columns.append(column)

		if converted_columns:
			logger.info("Converted numeric string columns: %s", converted_columns)

		return converted_columns

	def _detect_and_replace_infinite_values(
		self,
		dataframe: pd.DataFrame,
		preserved_columns: set[str],
	) -> tuple[list[str], int]:
		"""Detect infinite values in numeric columns and replace them with NaN."""
		columns_with_infinite_values: list[str] = []
		total_infinite_values = 0

		for column in dataframe.columns:
			if column in preserved_columns:
				continue

			series = dataframe[column]
			if not pd.api.types.is_numeric_dtype(series):
				continue

			numeric_values = pd.to_numeric(series, errors="coerce").to_numpy(
				dtype="float64",
				na_value=np.nan,
			)
			infinite_mask = np.isinf(numeric_values)
			infinite_count = int(infinite_mask.sum())
			if infinite_count == 0:
				continue

			dataframe.loc[infinite_mask, column] = np.nan
			columns_with_infinite_values.append(column)
			total_infinite_values += infinite_count

		if columns_with_infinite_values:
			logger.info(
				"Detected %s infinite value(s) across columns: %s",
				total_infinite_values,
				columns_with_infinite_values,
			)

		return columns_with_infinite_values, total_infinite_values

	def _drop_completely_empty_rows(self, dataframe: pd.DataFrame) -> int:
		"""Remove rows where every value is missing."""
		before_rows = len(dataframe)
		dataframe.dropna(how="all", inplace=True)
		removed_rows = before_rows - len(dataframe)

		if removed_rows:
			logger.info("Removed %s completely empty row(s)", removed_rows)
		else:
			logger.info("No completely empty rows were found")

		return removed_rows

	def _drop_duplicate_timestamps(
		self,
		dataframe: pd.DataFrame,
		timestamp_column: str | None,
	) -> int:
		"""Remove duplicate rows based on the timestamp column when present."""
		if timestamp_column is None or timestamp_column not in dataframe.columns:
			logger.info("Timestamp column not present; skipping duplicate timestamp removal")
			return 0

		timestamp_series = dataframe[timestamp_column]
		duplicate_mask = timestamp_series.notna() & timestamp_series.duplicated(keep="first")
		removed_rows = int(duplicate_mask.sum())
		if removed_rows:
			dataframe.drop(index=dataframe.index[duplicate_mask], inplace=True)
			logger.info(
				"Removed %s row(s) with duplicate timestamp values in %r",
				removed_rows,
				timestamp_column,
			)
		else:
			logger.info("No duplicate timestamps found in %r", timestamp_column)

		return removed_rows

	def _fill_missing_values(
		self,
		dataframe: pd.DataFrame,
		preserved_columns: set[str],
		strategy: MissingValueStrategy,
	) -> None:
		"""Fill missing values in non-preserved columns using the configured strategy."""
		feature_columns = [
			column for column in dataframe.columns if column not in preserved_columns
		]
		if not feature_columns:
			logger.info("No feature columns available for missing-value handling")
			return

		logger.info(
			"Applying %s missing-value strategy to %s feature column(s)",
			strategy,
			len(feature_columns),
		)

		if strategy == "forward_fill":
			for column in feature_columns:
				dataframe[column] = dataframe[column].ffill().bfill()
			return

		if strategy == "backward_fill":
			for column in feature_columns:
				dataframe[column] = dataframe[column].bfill().ffill()
			return

		numeric_columns = [
			column for column in feature_columns if pd.api.types.is_numeric_dtype(dataframe[column])
		]
		if numeric_columns:
			if strategy == "mean":
				for column in numeric_columns:
					mean_value = dataframe[column].mean()
					dataframe[column] = dataframe[column].fillna(mean_value)
			elif strategy == "median":
				for column in numeric_columns:
					median_value = dataframe[column].median()
					dataframe[column] = dataframe[column].fillna(median_value)
			elif strategy == "interpolate":
				dataframe.loc[:, numeric_columns] = (
					dataframe.loc[:, numeric_columns]
					.interpolate(method="linear", limit_direction="both")
				)

		remaining_missing = [
			column for column in feature_columns if dataframe[column].isna().any()
		]
		if remaining_missing:
			logger.info(
				"Falling back to directional fill for remaining missing values in columns: %s",
				remaining_missing,
			)
			for column in remaining_missing:
				dataframe[column] = dataframe[column].ffill().bfill()

	@staticmethod
	def _fallback_fill_remaining_missing(
		dataframe: pd.DataFrame,
		preserved_columns: set[str],
	) -> None:
		"""Apply a final directional fill to any remaining feature missing values."""
		feature_columns = [
			column for column in dataframe.columns if column not in preserved_columns
		]
		if not feature_columns:
			return

		for column in feature_columns:
			dataframe[column] = dataframe[column].ffill().bfill()

	@staticmethod
	def _missing_values_by_column(dataframe: pd.DataFrame) -> dict[str, int]:
		"""Return missing-value counts for each column."""
		return {
			str(column): int(count)
			for column, count in dataframe.isna().sum().items()
			if int(count) > 0
		}

	@staticmethod
	def _log_operation(
		operations: list[dict[str, Any]],
		message: str,
		**details: Any,
	) -> None:
		"""Record and log a cleaning operation."""
		entry: dict[str, Any] = {"message": message}
		if details:
			entry["details"] = details
		operations.append(entry)
		logger.info("%s%s", message, f" | {details}" if details else "")


__all__ = ["DataCleaner", "MissingValueStrategy"]