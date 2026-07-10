"""Tests for :mod:`preprocessing.cleaner`."""

from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from preprocessing.cleaner import DataCleaner


def _base_frame(rows: int = 5) -> pd.DataFrame:
	"""Build a minimal ICS-like frame for cleaning tests."""
	return pd.DataFrame(
		{
			"timestamp": pd.date_range("2024-01-01", periods=rows, freq="s"),
			"feature_a": np.arange(rows, dtype=float),
			"feature_b": np.arange(rows, dtype=float) * 2.0,
			"label": np.zeros(rows, dtype=int),
		}
	)


class InitialisationTests(unittest.TestCase):
	"""Verify constructor validation."""

	def test_default_strategy_is_forward_fill(self) -> None:
		cleaner = DataCleaner()
		self.assertEqual(cleaner.missing_value_strategy, "forward_fill")

	def test_invalid_strategy_raises(self) -> None:
		with self.assertRaisesRegex(ValueError, "missing_value_strategy"):
			DataCleaner(missing_value_strategy="mode")  # type: ignore[arg-type]


class InputValidationTests(unittest.TestCase):
	"""Verify input type and column validation."""

	def test_non_dataframe_raises_type_error(self) -> None:
		with self.assertRaisesRegex(TypeError, "pandas DataFrame"):
			DataCleaner().clean([[1, 2, 3]])  # type: ignore[arg-type]

	def test_missing_explicit_timestamp_column_raises(self) -> None:
		frame = _base_frame()
		with self.assertRaisesRegex(ValueError, "Column 'missing_ts'"):
			DataCleaner(timestamp_column="missing_ts").clean(frame)


class ColumnDetectionTests(unittest.TestCase):
	"""Verify timestamp and label column auto-detection."""

	def test_detects_timestamp_column(self) -> None:
		frame = _base_frame()
		_, metadata = DataCleaner().clean(frame)
		self.assertEqual(metadata["timestamp_column"], "timestamp")

	def test_detects_t_stamp_for_swat_style_frames(self) -> None:
		frame = _base_frame().rename(columns={"timestamp": "t_stamp"})
		_, metadata = DataCleaner().clean(frame)
		self.assertEqual(metadata["timestamp_column"], "t_stamp")

	def test_detects_label_column(self) -> None:
		frame = _base_frame()
		_, metadata = DataCleaner().clean(frame)
		self.assertEqual(metadata["label_column"], "label")

	def test_preserves_timestamp_and_label_columns(self) -> None:
		frame = _base_frame()
		frame.loc[0, "feature_a"] = np.nan
		cleaned, metadata = DataCleaner().clean(frame)
		self.assertIn("timestamp", metadata["preserved_columns"])
		self.assertIn("label", metadata["preserved_columns"])
		self.assertFalse(cleaned["timestamp"].isna().any())
		self.assertFalse(cleaned["label"].isna().any())


class EmptyStringTests(unittest.TestCase):
	"""Verify blank strings become missing values."""

	def test_blank_strings_are_normalized_to_nan(self) -> None:
		frame = _base_frame()
		frame["feature_a"] = frame["feature_a"].astype(object)
		frame.loc[1, "feature_a"] = "   "
		cleaned, metadata = DataCleaner().clean(frame)
		self.assertFalse(pd.isna(cleaned.loc[0, "feature_a"]))
		self.assertEqual(metadata["missing_values_before"], 1)


class ActiveInactiveTests(unittest.TestCase):
	"""Verify SWaT-style Active/Inactive alarm conversion."""

	def test_active_inactive_columns_become_binary(self) -> None:
		frame = _base_frame()
		frame["LS201.Alarm"] = ["Active", "Inactive", "Active", "Inactive", "Active"]
		cleaned, metadata = DataCleaner().clean(frame)
		self.assertEqual(metadata["converted_active_inactive_columns"], ["LS201.Alarm"])
		self.assertEqual(cleaned["LS201.Alarm"].tolist(), [1, 0, 1, 0, 1])


class BooleanConversionTests(unittest.TestCase):
	"""Verify boolean columns are converted to integers."""

	def test_boolean_column_becomes_int(self) -> None:
		frame = _base_frame()
		frame["flag"] = [True, False, True, False, True]
		cleaned, metadata = DataCleaner().clean(frame)
		self.assertIn("flag", metadata["converted_boolean_columns"])
		self.assertEqual(cleaned["flag"].tolist(), [1, 0, 1, 0, 1])


class NumericStringTests(unittest.TestCase):
	"""Verify numeric strings are converted to numeric dtype."""

	def test_numeric_strings_become_float(self) -> None:
		frame = _base_frame()
		frame["sensor"] = ["1.5", "2.5", "3.5", "4.5", "5.5"]
		cleaned, metadata = DataCleaner().clean(frame)
		self.assertIn("sensor", metadata["converted_numeric_string_columns"])
		self.assertTrue(pd.api.types.is_float_dtype(cleaned["sensor"]))


class InfiniteValueTests(unittest.TestCase):
	"""Verify infinite values are replaced and filled."""

	def test_infinite_values_are_replaced(self) -> None:
		frame = _base_frame()
		frame.loc[2, "feature_a"] = np.inf
		cleaned, metadata = DataCleaner().clean(frame)
		self.assertEqual(metadata["infinite_value_count"], 1)
		self.assertFalse(np.isinf(cleaned["feature_a"]).any())
		self.assertFalse(cleaned["feature_a"].isna().any())


class RowRemovalTests(unittest.TestCase):
	"""Verify empty and duplicate rows are removed."""

	def test_completely_empty_rows_are_removed(self) -> None:
		frame = _base_frame()
		frame.loc[5] = [pd.NaT, np.nan, np.nan, np.nan]
		cleaned, metadata = DataCleaner().clean(frame)
		self.assertEqual(metadata["rows_removed_empty"], 1)
		self.assertEqual(len(cleaned), 5)

	def test_duplicate_timestamps_are_removed(self) -> None:
		frame = _base_frame()
		frame.loc[4, "timestamp"] = frame.loc[3, "timestamp"]
		cleaned, metadata = DataCleaner().clean(frame)
		self.assertEqual(metadata["rows_removed_duplicate_timestamps"], 1)
		self.assertEqual(len(cleaned), 4)


class MissingValueStrategyTests(unittest.TestCase):
	"""Verify missing-value handling strategies."""

	def test_forward_fill_fills_feature_gaps(self) -> None:
		frame = _base_frame()
		frame.loc[2, "feature_a"] = np.nan
		cleaned, metadata = DataCleaner(missing_value_strategy="forward_fill").clean(frame)
		self.assertEqual(cleaned.loc[2, "feature_a"], 1.0)
		self.assertEqual(metadata["missing_values_remaining"], 0)

	def test_mean_fill_uses_column_mean(self) -> None:
		frame = _base_frame()
		frame.loc[2, "feature_a"] = np.nan
		cleaned, _ = DataCleaner(missing_value_strategy="mean").clean(frame)
		self.assertEqual(cleaned.loc[2, "feature_a"], frame["feature_a"].mean())

	def test_median_fill_uses_column_median(self) -> None:
		frame = _base_frame()
		frame.loc[2, "feature_a"] = np.nan
		cleaned, _ = DataCleaner(missing_value_strategy="median").clean(frame)
		self.assertEqual(cleaned.loc[2, "feature_a"], frame["feature_a"].median())

	def test_interpolate_fill_fills_numeric_gaps(self) -> None:
		frame = _base_frame(6)
		frame.loc[2, "feature_a"] = np.nan
		cleaned, metadata = DataCleaner(missing_value_strategy="interpolate").clean(frame)
		self.assertFalse(cleaned["feature_a"].isna().any())
		self.assertEqual(metadata["missing_values_remaining"], 0)

	def test_backward_fill_fills_feature_gaps(self) -> None:
		frame = _base_frame()
		frame.loc[2, "feature_a"] = np.nan
		cleaned, metadata = DataCleaner(missing_value_strategy="backward_fill").clean(frame)
		self.assertEqual(cleaned.loc[2, "feature_a"], 3.0)
		self.assertEqual(metadata["missing_values_remaining"], 0)


class MetadataTests(unittest.TestCase):
	"""Verify metadata describes the cleaning run."""

	def test_metadata_records_input_and_output_shapes(self) -> None:
		frame = _base_frame()
		cleaned, metadata = DataCleaner().clean(frame)
		self.assertEqual(metadata["input_shape"], frame.shape)
		self.assertEqual(metadata["output_shape"], cleaned.shape)
		self.assertIn("operations", metadata)
		self.assertIn("final_dtypes", metadata)


class IntegrationTests(unittest.TestCase):
	"""Smoke-test cleaner on real dataset slices when available."""

	@classmethod
	def setUpClass(cls) -> None:
		cls.repo_root = Path(__file__).resolve().parent.parent

	def test_hai_slice_cleans_without_remaining_missing_values(self) -> None:
		from pathlib import Path

		from src.data.hai_loader import HAILoader

		repo = Path(__file__).resolve().parent.parent
		data_path = repo / "data" / "raw" / "HAI" / "hai_test1.csv"
		label_path = repo / "data" / "raw" / "HAI" / "hai_test1_label.csv"
		if not data_path.exists():
			self.skipTest("HAI data files are not available")

		hai = HAILoader(data_path=data_path, label_path=label_path).load()
		cleaned, metadata = DataCleaner().clean(hai.dataframe.iloc[:200])
		self.assertEqual(metadata["timestamp_column"], "timestamp")
		self.assertEqual(metadata["label_column"], "label")
		self.assertEqual(metadata["missing_values_remaining"], 0)
		self.assertGreater(len(cleaned), 0)

	def test_swat_slice_converts_alarm_columns(self) -> None:
		from pathlib import Path

		from src.data.swat_loader import SWaTLoader

		repo = Path(__file__).resolve().parent.parent
		swat_path = repo / "data" / "raw" / "SWaT" / "SWaT_Dec2019.xlsx"
		if not swat_path.exists():
			self.skipTest("SWaT workbook is not available")

		swat = SWaTLoader(file_path=swat_path).load()
		cleaned, metadata = DataCleaner(timestamp_column=swat.timestamp_column).clean(
			swat.dataframe.iloc[:200]
		)
		self.assertEqual(metadata["timestamp_column"], "t_stamp")
		self.assertGreater(len(metadata["converted_active_inactive_columns"]), 0)
		self.assertEqual(metadata["missing_values_remaining"], 0)
		self.assertGreater(len(cleaned), 0)


if __name__ == "__main__":
	unittest.main()
