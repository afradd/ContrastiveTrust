"""Tests for :mod:`preprocessing.windowing`."""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from preprocessing.windowing import SlidingWindowGenerator


def _make_frame(rows: int = 100, with_label: bool = True) -> pd.DataFrame:
	"""Build a small synthetic ICS-like frame."""
	data = {
		"timestamp": pd.date_range("2024-01-01", periods=rows, freq="s"),
		"feature_a": np.arange(rows, dtype=float),
		"feature_b": np.arange(rows, dtype=float) * 2.0,
	}
	if with_label:
		labels = np.zeros(rows, dtype=int)
		labels[rows // 2 :] = 1
		data["label"] = labels
	return pd.DataFrame(data)


class WindowSizeTests(unittest.TestCase):
	"""Verify window_size controls the temporal length of each window."""

	def test_window_size_shapes_output(self) -> None:
		for window_size in (10, 30, 60):
			with self.subTest(window_size=window_size):
				frame = _make_frame(100, with_label=False)
				batch = SlidingWindowGenerator(
					window_size=window_size,
					stride=1,
					return_labels=False,
				).generate(frame)
				self.assertEqual(batch.windows.shape[1], window_size)
				expected = 100 - window_size + 1
				self.assertEqual(batch.windows.shape[0], expected)

	def test_window_size_larger_than_rows_raises(self) -> None:
		frame = _make_frame(10, with_label=False)
		with self.assertRaisesRegex(ValueError, "window_size"):
			SlidingWindowGenerator(window_size=20, stride=1).generate(frame)


class StrideTests(unittest.TestCase):
	"""Verify stride controls window overlap and count."""

	def test_stride_one_produces_maximum_overlap(self) -> None:
		frame = _make_frame(100, with_label=False)
		batch = SlidingWindowGenerator(window_size=60, stride=1).generate(frame)
		self.assertEqual(batch.windows.shape[0], 41)
		self.assertEqual(batch.metadata["start_indices"][:3], [0, 1, 2])

	def test_stride_reduces_window_count(self) -> None:
		frame = _make_frame(100, with_label=False)
		batch = SlidingWindowGenerator(window_size=60, stride=10).generate(frame)
		self.assertEqual(batch.windows.shape[0], 5)
		self.assertEqual(batch.metadata["start_indices"], [0, 10, 20, 30, 40])

	def test_stride_invalid_raises(self) -> None:
		with self.assertRaises(ValueError):
			SlidingWindowGenerator(window_size=10, stride=0)


class LabelModeTests(unittest.TestCase):
	"""Verify label aggregation modes."""

	def test_label_last_uses_final_row(self) -> None:
		frame = _make_frame(20)
		frame.loc[5:9, "label"] = 1
		batch = SlidingWindowGenerator(
			window_size=5,
			stride=5,
			return_labels=True,
			label_method="last",
		).generate(frame)
		# Window starting at 5 covers rows 5-9, last label is 1.
		self.assertEqual(int(batch.labels[1]), 1)
		# Window starting at 0 covers rows 0-4, last label is 0.
		self.assertEqual(int(batch.labels[0]), 0)

	def test_label_max_detects_attack_in_window(self) -> None:
		frame = _make_frame(20)
		frame.loc[7, "label"] = 1
		batch = SlidingWindowGenerator(
			window_size=5,
			stride=1,
			return_labels=True,
			label_method="max",
		).generate(frame)
		# Window 3 covers rows 3-7 and should flag the attack.
		self.assertEqual(int(batch.labels[3]), 1)
		self.assertEqual(int(batch.labels[0]), 0)

	def test_label_majority_picks_dominant_class(self) -> None:
		frame = _make_frame(10)
		frame["label"] = 0
		frame.loc[0:2, "label"] = 1
		batch = SlidingWindowGenerator(
			window_size=5,
			stride=5,
			return_labels=True,
			label_method="majority",
		).generate(frame)
		# Rows 0-4: three ones, two zeros -> majority is 1.
		self.assertEqual(int(batch.labels[0]), 1)
		# Rows 5-9: all zeros.
		self.assertEqual(int(batch.labels[1]), 0)

	def test_invalid_label_method_raises(self) -> None:
		with self.assertRaises(ValueError):
			SlidingWindowGenerator(window_size=5, label_method="mean")  # type: ignore[arg-type]


class DropLastTests(unittest.TestCase):
	"""Verify drop_last removes the trailing misaligned window."""

	def test_drop_last_false_keeps_all_valid_windows(self) -> None:
		frame = _make_frame(100, with_label=False)
		batch = SlidingWindowGenerator(
			window_size=60,
			stride=7,
			drop_last=False,
		).generate(frame)
		self.assertEqual(batch.windows.shape[0], 6)

	def test_drop_last_true_removes_misaligned_final_window(self) -> None:
		frame = _make_frame(100, with_label=False)
		batch = SlidingWindowGenerator(
			window_size=60,
			stride=7,
			drop_last=True,
		).generate(frame)
		self.assertEqual(batch.windows.shape[0], 5)
		self.assertEqual(batch.metadata["start_indices"][-1], 28)


class TimestampTrackingTests(unittest.TestCase):
	"""Verify end timestamps are tracked per window."""

	def test_timestamps_match_window_end_rows(self) -> None:
		frame = _make_frame(20, with_label=False)
		batch = SlidingWindowGenerator(
			window_size=5,
			stride=5,
			return_timestamp=True,
		).generate(frame)
		self.assertIsNotNone(batch.timestamps)
		expected = frame["timestamp"].iloc[[4, 9, 14, 19]].to_numpy()
		np.testing.assert_array_equal(batch.timestamps, expected)

	def test_return_timestamp_false_omits_timestamps(self) -> None:
		frame = _make_frame(20, with_label=False)
		batch = SlidingWindowGenerator(
			window_size=5,
			stride=5,
			return_timestamp=False,
		).generate(frame)
		self.assertIsNone(batch.timestamps)

	def test_metadata_records_start_and_end_indices(self) -> None:
		frame = _make_frame(20, with_label=False)
		batch = SlidingWindowGenerator(window_size=5, stride=5).generate(frame)
		self.assertEqual(batch.metadata["end_indices"], [4, 9, 14, 19])


class PaddingTests(unittest.TestCase):
	"""Verify optional end padding for short or partial sequences."""

	def test_padding_allows_window_size_larger_than_rows(self) -> None:
		frame = _make_frame(55, with_label=False)
		batch = SlidingWindowGenerator(
			window_size=60,
			stride=1,
			padding=5,
		).generate(frame)
		self.assertEqual(batch.windows.shape, (1, 60, 2))
		self.assertEqual(batch.metadata["padding"], 5)

	def test_padding_completes_final_partial_window(self) -> None:
		frame = _make_frame(100, with_label=False)
		batch = SlidingWindowGenerator(
			window_size=60,
			stride=50,
			padding=10,
		).generate(frame)
		# Without padding only one full window fits; padding adds a second.
		self.assertGreaterEqual(batch.windows.shape[0], 2)
		self.assertEqual(batch.metadata["padded_rows"], 10)

	def test_no_padding_preserves_existing_behavior(self) -> None:
		frame = _make_frame(100, with_label=False)
		batch = SlidingWindowGenerator(window_size=60, stride=1, padding=0).generate(frame)
		self.assertEqual(batch.windows.shape[0], 41)
		self.assertEqual(batch.metadata["padding"], 0)


if __name__ == "__main__":
	unittest.main()
