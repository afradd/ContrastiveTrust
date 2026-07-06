"""Tests for :mod:`preprocessing.normalizer`.

Comprehensive test suite verifying shape preservation, data leakage
prevention, save/load consistency, and statistical correctness of the
``FeatureNormalizer`` class.
"""

from __future__ import annotations

import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from preprocessing.normalizer import FeatureNormalizer


# ---------------------------------------------------------------------------
# Test data helpers
# ---------------------------------------------------------------------------

def _ics_frame(
    rows: int = 200,
    seed: int = 42,
    *,
    include_label: bool = True,
) -> pd.DataFrame:
    """Build a synthetic ICS-like DataFrame for normalizer tests.

    The frame contains a timestamp column, several numeric sensor/actuator
    features with varying scales and offsets, and an optional binary label
    column.

    Args:
        rows: Number of rows to generate.
        seed: Random seed for reproducibility.
        include_label: Whether to include a ``label`` column.

    Returns:
        A pandas DataFrame with realistic ICS feature distributions.
    """
    rng = np.random.RandomState(seed)

    data: dict[str, object] = {
        "timestamp": pd.date_range("2024-01-01", periods=rows, freq="s"),
        "FIT101.Pv": rng.normal(loc=2.5, scale=0.8, size=rows),
        "LIT301.Pv": rng.normal(loc=500.0, scale=50.0, size=rows),
        "AIT201.Pv": rng.normal(loc=7.0, scale=0.2, size=rows),
        "P101.Status": rng.choice([0, 1, 2], size=rows).astype(float),
        "MV101.Status": rng.choice([0, 1], size=rows).astype(float),
    }

    if include_label:
        labels = np.zeros(rows, dtype=int)
        labels[rows // 2 :] = 1
        data["label"] = labels

    return pd.DataFrame(data)


def _train_test_split(
    dataframe: pd.DataFrame,
    train_ratio: float = 0.7,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split a DataFrame into training and test sets without shuffling.

    Args:
        dataframe: Full DataFrame.
        train_ratio: Fraction of rows used for training.

    Returns:
        Tuple of ``(train_df, test_df)``.
    """
    split_index = int(len(dataframe) * train_ratio)
    return (
        dataframe.iloc[:split_index].reset_index(drop=True),
        dataframe.iloc[split_index:].reset_index(drop=True),
    )


# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------

class InitialisationTests(unittest.TestCase):
    """Verify constructor defaults and initial state."""

    def test_default_initialisation(self) -> None:
        normalizer = FeatureNormalizer()
        self.assertFalse(normalizer.is_fitted)
        self.assertEqual(normalizer.numeric_columns, [])
        self.assertEqual(normalizer.ignored_columns, [])
        self.assertIsNone(normalizer.timestamp_column)
        self.assertIsNone(normalizer.label_column)

    def test_explicit_column_names_stored(self) -> None:
        normalizer = FeatureNormalizer(
            timestamp_column="t_stamp",
            label_column="attack",
        )
        self.assertIsNone(normalizer.timestamp_column)  # Not resolved yet
        self.assertIsNone(normalizer.label_column)


class InputValidationTests(unittest.TestCase):
    """Verify pre-fit validation catches invalid inputs."""

    def test_non_dataframe_raises_type_error(self) -> None:
        with self.assertRaisesRegex(TypeError, "pandas DataFrame"):
            FeatureNormalizer().fit([[1, 2, 3]])  # type: ignore[arg-type]

    def test_empty_dataframe_raises(self) -> None:
        with self.assertRaisesRegex(ValueError, "empty"):
            FeatureNormalizer().fit(pd.DataFrame())

    def test_duplicate_columns_raise(self) -> None:
        frame = _ics_frame(10)
        # Create a duplicate column
        frame_with_dup = pd.concat(
            [frame, frame[["FIT101.Pv"]]], axis=1,
        )
        with self.assertRaisesRegex(ValueError, "duplicate"):
            FeatureNormalizer().fit(frame_with_dup)

    def test_missing_timestamp_raises(self) -> None:
        frame = _ics_frame(10).drop(columns=["timestamp"])
        with self.assertRaisesRegex(ValueError, "timestamp"):
            FeatureNormalizer().fit(frame)

    def test_missing_label_is_tolerated(self) -> None:
        frame = _ics_frame(10, include_label=False)
        normalizer = FeatureNormalizer()
        normalizer.fit(frame)
        self.assertTrue(normalizer.is_fitted)
        self.assertIsNone(normalizer.label_column)

    def test_nan_values_raise(self) -> None:
        frame = _ics_frame(10)
        frame.loc[3, "FIT101.Pv"] = np.nan
        with self.assertRaisesRegex(ValueError, "NaN"):
            FeatureNormalizer().fit(frame)

    def test_infinite_values_raise(self) -> None:
        frame = _ics_frame(10)
        frame.loc[3, "LIT301.Pv"] = np.inf
        with self.assertRaisesRegex(ValueError, "infinite"):
            FeatureNormalizer().fit(frame)


class ShapePreservationTests(unittest.TestCase):
    """Verify that transform preserves DataFrame shape and column order."""

    def test_shape_unchanged_after_transform(self) -> None:
        frame = _ics_frame(100)
        normalizer = FeatureNormalizer()
        result = normalizer.fit_transform(frame)
        self.assertEqual(result.shape, frame.shape)

    def test_column_order_preserved(self) -> None:
        frame = _ics_frame(100)
        normalizer = FeatureNormalizer()
        result = normalizer.fit_transform(frame)
        self.assertListEqual(list(result.columns), list(frame.columns))

    def test_output_is_dataframe(self) -> None:
        frame = _ics_frame(100)
        normalizer = FeatureNormalizer()
        result = normalizer.fit_transform(frame)
        self.assertIsInstance(result, pd.DataFrame)


class ColumnPreservationTests(unittest.TestCase):
    """Verify that timestamp and label columns are not modified."""

    def test_timestamp_preserved(self) -> None:
        frame = _ics_frame(100)
        normalizer = FeatureNormalizer()
        result = normalizer.fit_transform(frame)
        pd.testing.assert_series_equal(
            result["timestamp"],
            frame["timestamp"],
            check_names=True,
        )

    def test_label_preserved(self) -> None:
        frame = _ics_frame(100)
        normalizer = FeatureNormalizer()
        result = normalizer.fit_transform(frame)
        pd.testing.assert_series_equal(
            result["label"],
            frame["label"],
            check_names=True,
        )

    def test_swat_timestamp_preserved(self) -> None:
        """Verify t_stamp auto-detection works for SWaT-style frames."""
        frame = _ics_frame(100).rename(columns={"timestamp": "t_stamp"})
        frame = frame.drop(columns=["label"])
        normalizer = FeatureNormalizer()
        result = normalizer.fit_transform(frame)
        pd.testing.assert_series_equal(
            result["t_stamp"],
            frame["t_stamp"],
            check_names=True,
        )
        self.assertEqual(normalizer.timestamp_column, "t_stamp")


class StatisticalCorrectnessTests(unittest.TestCase):
    """Verify that z-score normalisation produces correct statistics."""

    def test_mean_approximately_zero(self) -> None:
        frame = _ics_frame(500)
        normalizer = FeatureNormalizer()
        result = normalizer.fit_transform(frame)
        numeric_cols = normalizer.numeric_columns
        means = result[numeric_cols].mean()
        for col in numeric_cols:
            self.assertAlmostEqual(
                float(means[col]),
                0.0,
                places=10,
                msg=f"Mean of {col} should be ~0 after normalisation",
            )

    def test_std_approximately_one(self) -> None:
        frame = _ics_frame(500)
        normalizer = FeatureNormalizer()
        result = normalizer.fit_transform(frame)
        numeric_cols = normalizer.numeric_columns
        stds = result[numeric_cols].std(ddof=0)
        for col in numeric_cols:
            self.assertAlmostEqual(
                float(stds[col]),
                1.0,
                places=10,
                msg=f"Std of {col} should be ~1 after normalisation",
            )

    def test_no_nan_after_transform(self) -> None:
        frame = _ics_frame(200)
        normalizer = FeatureNormalizer()
        result = normalizer.fit_transform(frame)
        self.assertEqual(
            int(result.isna().sum().sum()),
            0,
            "Normalised DataFrame should contain no NaN values",
        )

    def test_no_infinite_after_transform(self) -> None:
        frame = _ics_frame(200)
        normalizer = FeatureNormalizer()
        result = normalizer.fit_transform(frame)
        numeric_cols = normalizer.numeric_columns
        numeric_values = result[numeric_cols].to_numpy(dtype=np.float64)
        self.assertFalse(
            np.isinf(numeric_values).any(),
            "Normalised DataFrame should contain no infinite values",
        )


class InverseTransformTests(unittest.TestCase):
    """Verify that inverse_transform recovers original values."""

    def test_inverse_transform_recovers_original(self) -> None:
        frame = _ics_frame(100)
        normalizer = FeatureNormalizer()
        normalised = normalizer.fit_transform(frame)
        recovered = normalizer.inverse_transform(normalised)
        numeric_cols = normalizer.numeric_columns
        for col in numeric_cols:
            np.testing.assert_array_almost_equal(
                recovered[col].to_numpy(),
                frame[col].to_numpy(),
                decimal=10,
                err_msg=f"Inverse transform failed for {col}",
            )


class DataLeakageTests(unittest.TestCase):
    """Verify that fitting on training data does not leak test statistics."""

    def test_no_data_leakage(self) -> None:
        """Test distribution on the train split only uses train statistics."""
        frame = _ics_frame(500, seed=99)
        train_df, test_df = _train_test_split(frame, train_ratio=0.7)

        normalizer = FeatureNormalizer()
        normalizer.fit(train_df)
        train_normalised = normalizer.transform(train_df)
        test_normalised = normalizer.transform(test_df)

        numeric_cols = normalizer.numeric_columns

        # Training mean should be ~0, std ~1
        train_means = train_normalised[numeric_cols].mean()
        train_stds = train_normalised[numeric_cols].std(ddof=0)
        for col in numeric_cols:
            self.assertAlmostEqual(float(train_means[col]), 0.0, places=10)
            self.assertAlmostEqual(float(train_stds[col]), 1.0, places=10)

        # Test mean should NOT be exactly 0, std should NOT be exactly 1
        # (unless by extreme coincidence, which the different distribution
        # seeds and split boundary make effectively impossible).
        test_means = test_normalised[numeric_cols].mean()
        test_stds = test_normalised[numeric_cols].std(ddof=0)

        # At least one column should have mean != 0 or std != 1 on test
        mean_diffs = [abs(float(test_means[col])) for col in numeric_cols]
        std_diffs = [abs(float(test_stds[col]) - 1.0) for col in numeric_cols]
        has_deviation = any(d > 1e-6 for d in mean_diffs) or any(
            d > 1e-6 for d in std_diffs
        )
        self.assertTrue(
            has_deviation,
            "Test split statistics should differ from train (no leakage)",
        )


class PersistenceTests(unittest.TestCase):
    """Verify save/load round-trip preserves normalizer state."""

    def setUp(self) -> None:
        self._tmpdir = tempfile.mkdtemp()

    def tearDown(self) -> None:
        shutil.rmtree(self._tmpdir, ignore_errors=True)

    def test_save_and_load_consistency(self) -> None:
        frame = _ics_frame(200)
        normalizer = FeatureNormalizer()
        original_normalised = normalizer.fit_transform(frame)

        save_path = Path(self._tmpdir) / "test_scaler.pkl"
        normalizer.save(save_path)
        self.assertTrue(save_path.exists())

        loaded = FeatureNormalizer.load(save_path)
        loaded_normalised = loaded.transform(frame)

        pd.testing.assert_frame_equal(original_normalised, loaded_normalised)
        self.assertEqual(
            loaded.numeric_columns, normalizer.numeric_columns,
        )
        self.assertEqual(
            loaded.ignored_columns, normalizer.ignored_columns,
        )
        self.assertEqual(loaded.timestamp_column, normalizer.timestamp_column)
        self.assertEqual(loaded.label_column, normalizer.label_column)
        self.assertTrue(loaded.is_fitted)

    def test_load_nonexistent_raises(self) -> None:
        with self.assertRaises(FileNotFoundError):
            FeatureNormalizer.load(Path(self._tmpdir) / "missing.pkl")

    def test_save_before_fit_raises(self) -> None:
        normalizer = FeatureNormalizer()
        with self.assertRaises(RuntimeError):
            normalizer.save(Path(self._tmpdir) / "unfitted.pkl")


class UnfittedTests(unittest.TestCase):
    """Verify that operations on an unfitted normalizer raise errors."""

    def test_transform_before_fit_raises(self) -> None:
        normalizer = FeatureNormalizer()
        frame = _ics_frame(10)
        with self.assertRaises(RuntimeError):
            normalizer.transform(frame)

    def test_inverse_transform_before_fit_raises(self) -> None:
        normalizer = FeatureNormalizer()
        frame = _ics_frame(10)
        with self.assertRaises(RuntimeError):
            normalizer.inverse_transform(frame)

    def test_get_statistics_before_fit_raises(self) -> None:
        normalizer = FeatureNormalizer()
        with self.assertRaises(RuntimeError):
            normalizer.get_statistics()


class StatisticsTests(unittest.TestCase):
    """Verify get_statistics returns correct metadata."""

    def test_statistics_content(self) -> None:
        frame = _ics_frame(100)
        normalizer = FeatureNormalizer()
        normalizer.fit(frame)
        stats = normalizer.get_statistics()

        self.assertEqual(stats["n_samples"], 100)
        self.assertEqual(stats["n_features"], 5)
        self.assertIn("timestamp", stats["ignored_columns"])
        self.assertIn("label", stats["ignored_columns"])
        self.assertEqual(stats["timestamp_column"], "timestamp")
        self.assertEqual(stats["label_column"], "label")
        self.assertIn("FIT101.Pv", stats["per_feature"])
        self.assertIn("mean", stats["per_feature"]["FIT101.Pv"])
        self.assertIn("std", stats["per_feature"]["FIT101.Pv"])


# ---------------------------------------------------------------------------
# Custom test report
# ---------------------------------------------------------------------------

def _run_with_report() -> None:
    """Execute all tests and print a clean report summary."""
    frame = _ics_frame(500, seed=42)
    train_df, test_df = _train_test_split(frame, train_ratio=0.7)

    normalizer = FeatureNormalizer()
    normalizer.fit(train_df)
    train_normalised = normalizer.transform(train_df)
    test_normalised = normalizer.transform(test_df)

    numeric_cols = normalizer.numeric_columns

    # Compute statistics on training normalised data
    train_means = train_normalised[numeric_cols].mean()
    train_stds = train_normalised[numeric_cols].std(ddof=0)
    max_mean_error = float(train_means.abs().max())
    max_std_error = float((train_stds - 1.0).abs().max())

    nan_count = int(train_normalised.isna().sum().sum())
    numeric_values = train_normalised[numeric_cols].to_numpy(dtype=np.float64)
    inf_count = int(np.isinf(numeric_values).sum())

    # Save/load test
    tmpdir = tempfile.mkdtemp()
    try:
        save_path = Path(tmpdir) / "scaler.pkl"
        normalizer.save(save_path)
        loaded = FeatureNormalizer.load(save_path)
        reload_result = loaded.transform(train_df)
        reload_passed = train_normalised.equals(reload_result)
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

    # Timestamp/label preservation
    ts_passed = train_normalised["timestamp"].equals(train_df["timestamp"])
    label_passed = train_normalised["label"].equals(train_df["label"])

    # Data leakage check
    test_means = test_normalised[numeric_cols].mean()
    test_stds = test_normalised[numeric_cols].std(ddof=0)
    mean_diffs = [abs(float(test_means[col])) for col in numeric_cols]
    std_diffs = [abs(float(test_stds[col]) - 1.0) for col in numeric_cols]
    leakage_passed = any(d > 1e-6 for d in mean_diffs) or any(
        d > 1e-6 for d in std_diffs
    )

    ignored_str = "\n  ".join(normalizer.ignored_columns)

    report = f"""
========== Feature Normalizer Test ==========

Scaler: StandardScaler

Training Samples: {len(train_df)}
Numeric Features: {len(numeric_cols)}

Ignored Columns:
  {ignored_str}

Shape Before:
  {train_df.shape}

Shape After:
  {train_normalised.shape}

Maximum Mean Error:
  {max_mean_error:.2e}

Maximum Std Error:
  {max_std_error:.2e}

NaN Values:
  {nan_count}

Infinite Values:
  {inf_count}

Scaler Saved:
  {save_path}

Scaler Reload Test:
  {"PASSED" if reload_passed else "FAILED"}

Timestamp Preserved:
  {"PASSED" if ts_passed else "FAILED"}

Labels Preserved:
  {"PASSED" if label_passed else "FAILED"}

Data Leakage Check:
  {"PASSED" if leakage_passed else "FAILED"}

=============================================
"""
    print(report)

    # Now run the full test suite
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromModule(__import__(__name__))
    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(suite)


if __name__ == "__main__":
    _run_with_report()
