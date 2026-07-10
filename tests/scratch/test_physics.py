"""Tests for :mod:`src.features.physics`.

Comprehensive test suite verifying shape preservation, deterministic
behaviour, NaN/Inf rejection, batch extraction, save/load configuration
round-tripping, feature-name consistency, and performance on moderate
batches for the ``PhysicsFeatureExtractor`` class.
"""

from __future__ import annotations

import shutil
import tempfile
import time
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.features.physics import PhysicsFeatureExtractor


# ---------------------------------------------------------------------------
# Test data helpers
# ---------------------------------------------------------------------------

def _swat_window(
    rows: int = 50,
    seed: int = 42,
    *,
    include_label: bool = True,
    include_timestamp: bool = True,
) -> pd.DataFrame:
    """Build a synthetic SWaT-style sliding window DataFrame.

    The frame contains representative column names for both SWaT and
    HAI datasets: FIT (flow), PIT (pressure), LIT (level), AIT (analytic),
    P (pump status), and MV (valve status).

    Parameters
    ----------
    rows : int
        Number of rows (time steps) in the window.
    seed : int
        Random seed for reproducibility.
    include_label : bool
        Whether to include a ``label`` column.
    include_timestamp : bool
        Whether to include a ``timestamp`` column.

    Returns
    -------
    pandas.DataFrame
    """
    rng = np.random.RandomState(seed)

    data: dict[str, object] = {}

    if include_timestamp:
        data["timestamp"] = pd.date_range("2024-01-01", periods=rows, freq="s")

    # Analog sensors
    data["FIT101.Pv"] = rng.normal(loc=2.5, scale=0.3, size=rows)
    data["FIT201.Pv"] = rng.normal(loc=3.0, scale=0.4, size=rows)
    data["PIT101.Pv"] = rng.normal(loc=100.0, scale=5.0, size=rows)
    data["PIT301.Pv"] = rng.normal(loc=120.0, scale=6.0, size=rows)
    data["LIT101.Pv"] = rng.normal(loc=500.0, scale=50.0, size=rows)
    data["AIT201.Pv"] = rng.normal(loc=7.0, scale=0.2, size=rows)

    # Actuators
    data["P101.Status"] = rng.choice([0, 1, 2], size=rows).astype(float)
    data["P201.Status"] = rng.choice([0, 1], size=rows).astype(float)
    data["MV101.Status"] = rng.choice([0, 1], size=rows).astype(float)
    data["MV201.Status"] = rng.choice([0, 1], size=rows).astype(float)

    if include_label:
        labels = np.zeros(rows, dtype=int)
        labels[rows // 2:] = 1
        data["label"] = labels

    return pd.DataFrame(data)


def _hai_window(
    rows: int = 50,
    seed: int = 42,
) -> pd.DataFrame:
    """Build a synthetic HAI-style sliding window DataFrame.

    HAI uses slightly different column naming conventions but the same
    sensor/actuator prefix patterns.

    Parameters
    ----------
    rows : int
        Number of rows.
    seed : int
        Random seed.

    Returns
    -------
    pandas.DataFrame
    """
    rng = np.random.RandomState(seed)

    data: dict[str, object] = {
        "timestamp": pd.date_range("2024-06-01", periods=rows, freq="s"),
        "FIT_001": rng.normal(loc=5.0, scale=1.0, size=rows),
        "FIT_002": rng.normal(loc=4.5, scale=0.8, size=rows),
        "PIT_001": rng.normal(loc=200.0, scale=10.0, size=rows),
        "LIT_001": rng.normal(loc=800.0, scale=20.0, size=rows),
        "AIT_001": rng.normal(loc=6.5, scale=0.3, size=rows),
        "P_001_Status": rng.choice([0, 1], size=rows).astype(float),
        "MV_001": rng.choice([0, 1], size=rows).astype(float),
        "attack": np.zeros(rows, dtype=int),
    }

    return pd.DataFrame(data)


# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------

class TestPhysicsFeatureExtractorInit(unittest.TestCase):
    """Tests for ``__init__`` parameter validation."""

    def test_valid_default_init(self) -> None:
        extractor = PhysicsFeatureExtractor()
        self.assertEqual(extractor.rolling_window, 5)
        self.assertFalse(extractor._is_configured)

    def test_custom_rolling_window(self) -> None:
        extractor = PhysicsFeatureExtractor(rolling_window=10)
        self.assertEqual(extractor.rolling_window, 10)

    def test_rolling_window_type_error_bool(self) -> None:
        with self.assertRaises(TypeError):
            PhysicsFeatureExtractor(rolling_window=True)

    def test_rolling_window_type_error_float(self) -> None:
        with self.assertRaises(TypeError):
            PhysicsFeatureExtractor(rolling_window=3.5)

    def test_rolling_window_too_small(self) -> None:
        with self.assertRaises(ValueError):
            PhysicsFeatureExtractor(rolling_window=1)

    def test_rolling_window_zero(self) -> None:
        with self.assertRaises(ValueError):
            PhysicsFeatureExtractor(rolling_window=0)

    def test_rolling_window_negative(self) -> None:
        with self.assertRaises(ValueError):
            PhysicsFeatureExtractor(rolling_window=-3)


class TestExtractShapeValidation(unittest.TestCase):
    """Tests for output shape and column count correctness."""

    def setUp(self) -> None:
        self.extractor = PhysicsFeatureExtractor(rolling_window=3)
        self.window = _swat_window(rows=30, seed=42)

    def test_output_rows_match_input(self) -> None:
        result = self.extractor.extract(self.window)
        self.assertEqual(len(result), len(self.window))

    def test_output_has_more_columns_than_input(self) -> None:
        result = self.extractor.extract(self.window)
        self.assertGreater(len(result.columns), len(self.window.columns))

    def test_original_columns_preserved(self) -> None:
        result = self.extractor.extract(self.window)
        for col in self.window.columns:
            self.assertIn(col, result.columns)

    def test_timestamp_preserved(self) -> None:
        result = self.extractor.extract(self.window)
        pd.testing.assert_series_equal(
            result["timestamp"],
            self.window["timestamp"].reset_index(drop=True),
        )

    def test_label_preserved(self) -> None:
        result = self.extractor.extract(self.window)
        pd.testing.assert_series_equal(
            result["label"],
            self.window["label"].reset_index(drop=True),
        )

    def test_no_timestamp_no_label(self) -> None:
        window = _swat_window(
            rows=30,
            seed=42,
            include_label=False,
            include_timestamp=False,
        )
        extractor = PhysicsFeatureExtractor(rolling_window=3)
        result = extractor.extract(window)
        self.assertEqual(len(result), 30)
        self.assertGreater(len(result.columns), len(window.columns))

    def test_expected_analog_feature_count(self) -> None:
        """Each analog sensor produces 6 engineered features."""
        result = self.extractor.extract(self.window)
        n_analog = len(self.extractor.analog_columns)
        n_fit_pairs = len(self.extractor.fit_columns) * (len(self.extractor.fit_columns) - 1) // 2
        n_pit_pairs = len(self.extractor.pit_columns) * (len(self.extractor.pit_columns) - 1) // 2
        n_pumps = len(self.extractor.pump_columns)
        n_valves = len(self.extractor.valve_columns)

        expected_engineered = (
            n_analog * 6
            + n_fit_pairs
            + n_pit_pairs
            + n_pumps
            + n_valves
        )
        actual_engineered = len(result.columns) - len(self.window.columns)
        self.assertEqual(actual_engineered, expected_engineered)


class TestDeterministicBehaviour(unittest.TestCase):
    """Verify that extraction produces identical results on repeated calls."""

    def test_deterministic_single_extract(self) -> None:
        window = _swat_window(rows=30, seed=42)

        extractor = PhysicsFeatureExtractor(rolling_window=3)
        result_1 = extractor.extract(window)
        result_2 = extractor.extract(window)

        pd.testing.assert_frame_equal(result_1, result_2)

    def test_deterministic_across_instances(self) -> None:
        window = _swat_window(rows=30, seed=42)

        ext_a = PhysicsFeatureExtractor(rolling_window=3)
        ext_b = PhysicsFeatureExtractor(rolling_window=3)

        result_a = ext_a.extract(window)
        result_b = ext_b.extract(window)

        pd.testing.assert_frame_equal(result_a, result_b)


class TestNoNaNOrInf(unittest.TestCase):
    """Verify that output contains no NaN or Inf values."""

    def test_no_nan_in_output(self) -> None:
        window = _swat_window(rows=30, seed=42)
        extractor = PhysicsFeatureExtractor(rolling_window=3)
        result = extractor.extract(window)

        numeric_result = result.select_dtypes(include=[np.number])
        self.assertFalse(
            numeric_result.isna().any().any(),
            "Output contains NaN values",
        )

    def test_no_inf_in_output(self) -> None:
        window = _swat_window(rows=30, seed=42)
        extractor = PhysicsFeatureExtractor(rolling_window=3)
        result = extractor.extract(window)

        numeric_result = result.select_dtypes(include=[np.number])
        values = numeric_result.to_numpy(dtype=np.float64)
        self.assertFalse(
            np.isinf(values).any(),
            "Output contains infinite values",
        )

    def test_rejects_nan_input(self) -> None:
        window = _swat_window(rows=30, seed=42)
        window.iloc[5, 1] = np.nan  # Inject NaN

        extractor = PhysicsFeatureExtractor(rolling_window=3)
        with self.assertRaises(ValueError, msg="NaN"):
            extractor.extract(window)

    def test_rejects_inf_input(self) -> None:
        window = _swat_window(rows=30, seed=42)
        window.iloc[5, 1] = np.inf  # Inject Inf

        extractor = PhysicsFeatureExtractor(rolling_window=3)
        with self.assertRaises(ValueError, msg="infinite"):
            extractor.extract(window)


class TestBatchExtraction(unittest.TestCase):
    """Tests for ``extract_batch()``."""

    def test_batch_from_list_of_dataframes(self) -> None:
        windows = [_swat_window(rows=20, seed=s) for s in range(5)]
        extractor = PhysicsFeatureExtractor(rolling_window=3)

        results = extractor.extract_batch(windows)

        self.assertEqual(len(results), 5)
        for r in results:
            self.assertEqual(len(r), 20)

    def test_batch_from_3d_numpy(self) -> None:
        rng = np.random.RandomState(42)
        batch = rng.randn(8, 20, 6)

        extractor = PhysicsFeatureExtractor(rolling_window=3)
        results = extractor.extract_batch(batch)

        self.assertEqual(len(results), 8)
        for r in results:
            self.assertEqual(len(r), 20)

    def test_batch_empty_raises(self) -> None:
        extractor = PhysicsFeatureExtractor(rolling_window=3)
        with self.assertRaises(ValueError):
            extractor.extract_batch([])

    def test_batch_invalid_type_raises(self) -> None:
        extractor = PhysicsFeatureExtractor(rolling_window=3)
        with self.assertRaises(TypeError):
            extractor.extract_batch("not_a_batch")

    def test_batch_2d_numpy_raises(self) -> None:
        extractor = PhysicsFeatureExtractor(rolling_window=3)
        with self.assertRaises(ValueError):
            extractor.extract_batch(np.zeros((10, 5)))

    def test_batch_deterministic(self) -> None:
        windows = [_swat_window(rows=20, seed=s) for s in range(3)]
        ext_a = PhysicsFeatureExtractor(rolling_window=3)
        ext_b = PhysicsFeatureExtractor(rolling_window=3)

        results_a = ext_a.extract_batch(windows)
        results_b = ext_b.extract_batch(windows)

        for ra, rb in zip(results_a, results_b):
            pd.testing.assert_frame_equal(ra, rb)


class TestSaveLoadConfig(unittest.TestCase):
    """Tests for ``save_config()`` and ``load_config()`` round-tripping."""

    def setUp(self) -> None:
        self.tmp_dir = Path(tempfile.mkdtemp())
        self.config_path = self.tmp_dir / "test_config.json"

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_save_load_round_trip(self) -> None:
        window = _swat_window(rows=30, seed=42)
        extractor = PhysicsFeatureExtractor(rolling_window=4)
        original_result = extractor.extract(window)

        saved_path = extractor.save_config(self.config_path)
        self.assertTrue(saved_path.exists())

        loaded = PhysicsFeatureExtractor.load_config(self.config_path)
        loaded_result = loaded.extract(window)

        pd.testing.assert_frame_equal(original_result, loaded_result)

    def test_save_before_configure_raises(self) -> None:
        extractor = PhysicsFeatureExtractor()
        with self.assertRaises(RuntimeError):
            extractor.save_config(self.config_path)

    def test_load_missing_file_raises(self) -> None:
        with self.assertRaises(FileNotFoundError):
            PhysicsFeatureExtractor.load_config(
                self.tmp_dir / "nonexistent.json"
            )

    def test_load_preserves_rolling_window(self) -> None:
        window = _swat_window(rows=30, seed=42)
        extractor = PhysicsFeatureExtractor(rolling_window=7)
        extractor.extract(window)
        extractor.save_config(self.config_path)

        loaded = PhysicsFeatureExtractor.load_config(self.config_path)
        self.assertEqual(loaded.rolling_window, 7)

    def test_load_preserves_column_metadata(self) -> None:
        window = _swat_window(rows=30, seed=42)
        extractor = PhysicsFeatureExtractor(rolling_window=3)
        extractor.extract(window)
        extractor.save_config(self.config_path)

        loaded = PhysicsFeatureExtractor.load_config(self.config_path)
        self.assertEqual(loaded.analog_columns, extractor.analog_columns)
        self.assertEqual(loaded.actuator_columns, extractor.actuator_columns)
        self.assertEqual(loaded.fit_columns, extractor.fit_columns)
        self.assertEqual(loaded.pit_columns, extractor.pit_columns)
        self.assertEqual(loaded.pump_columns, extractor.pump_columns)
        self.assertEqual(loaded.valve_columns, extractor.valve_columns)

    def test_load_invalid_json_raises(self) -> None:
        self.config_path.write_text("{}", encoding="utf-8")
        with self.assertRaises(ValueError):
            PhysicsFeatureExtractor.load_config(self.config_path)


class TestFeatureNameConsistency(unittest.TestCase):
    """Verify that ``get_feature_names()`` matches actual output columns."""

    def test_names_match_output_columns(self) -> None:
        window = _swat_window(rows=30, seed=42)
        extractor = PhysicsFeatureExtractor(rolling_window=3)
        result = extractor.extract(window)

        feature_names = extractor.get_feature_names()
        self.assertEqual(feature_names, list(result.columns))

    def test_get_feature_names_before_extract_raises(self) -> None:
        extractor = PhysicsFeatureExtractor()
        with self.assertRaises(RuntimeError):
            extractor.get_feature_names()

    def test_names_stable_across_calls(self) -> None:
        window = _swat_window(rows=30, seed=42)
        extractor = PhysicsFeatureExtractor(rolling_window=3)
        extractor.extract(window)

        names_1 = extractor.get_feature_names()
        names_2 = extractor.get_feature_names()
        self.assertEqual(names_1, names_2)

    def test_feature_names_after_load(self) -> None:
        tmp_dir = Path(tempfile.mkdtemp())
        try:
            config_path = tmp_dir / "cfg.json"
            window = _swat_window(rows=30, seed=42)

            extractor = PhysicsFeatureExtractor(rolling_window=3)
            result = extractor.extract(window)
            extractor.save_config(config_path)

            loaded = PhysicsFeatureExtractor.load_config(config_path)
            self.assertEqual(
                loaded.get_feature_names(),
                list(result.columns),
            )
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)


class TestColumnDetection(unittest.TestCase):
    """Tests for automatic column classification."""

    def test_swat_analog_detection(self) -> None:
        window = _swat_window(rows=20, seed=42)
        extractor = PhysicsFeatureExtractor(rolling_window=3)
        extractor.extract(window)

        self.assertIn("FIT101.Pv", extractor.analog_columns)
        self.assertIn("FIT201.Pv", extractor.analog_columns)
        self.assertIn("PIT101.Pv", extractor.analog_columns)
        self.assertIn("PIT301.Pv", extractor.analog_columns)
        self.assertIn("LIT101.Pv", extractor.analog_columns)
        self.assertIn("AIT201.Pv", extractor.analog_columns)

    def test_swat_actuator_detection(self) -> None:
        window = _swat_window(rows=20, seed=42)
        extractor = PhysicsFeatureExtractor(rolling_window=3)
        extractor.extract(window)

        self.assertIn("P101.Status", extractor.actuator_columns)
        self.assertIn("P201.Status", extractor.actuator_columns)
        self.assertIn("MV101.Status", extractor.actuator_columns)
        self.assertIn("MV201.Status", extractor.actuator_columns)

    def test_swat_fit_pit_pump_valve_subsets(self) -> None:
        window = _swat_window(rows=20, seed=42)
        extractor = PhysicsFeatureExtractor(rolling_window=3)
        extractor.extract(window)

        self.assertEqual(len(extractor.fit_columns), 2)
        self.assertEqual(len(extractor.pit_columns), 2)
        self.assertEqual(len(extractor.pump_columns), 2)
        self.assertEqual(len(extractor.valve_columns), 2)

    def test_hai_column_detection(self) -> None:
        window = _hai_window(rows=20, seed=42)
        extractor = PhysicsFeatureExtractor(rolling_window=3)
        extractor.extract(window)

        self.assertEqual(len(extractor.fit_columns), 2)
        self.assertGreaterEqual(len(extractor.pit_columns), 1)

    def test_pit_not_classified_as_pump(self) -> None:
        """PIT columns must not match the 'P' pump prefix."""
        window = _swat_window(rows=20, seed=42)
        extractor = PhysicsFeatureExtractor(rolling_window=3)
        extractor.extract(window)

        for col in extractor.pump_columns:
            self.assertFalse(
                col.upper().startswith("PIT"),
                f"PIT column {col!r} incorrectly classified as pump",
            )


class TestInputValidation(unittest.TestCase):
    """Tests for input validation and error handling."""

    def test_empty_dataframe_raises(self) -> None:
        extractor = PhysicsFeatureExtractor()
        with self.assertRaises(ValueError):
            extractor.extract(pd.DataFrame())

    def test_single_row_raises(self) -> None:
        """Rate of change needs at least 2 rows."""
        window = _swat_window(rows=1, seed=42)
        extractor = PhysicsFeatureExtractor()
        with self.assertRaises(ValueError):
            extractor.extract(window)

    def test_invalid_type_string_raises(self) -> None:
        extractor = PhysicsFeatureExtractor()
        with self.assertRaises(TypeError):
            extractor.extract("not_valid")

    def test_invalid_type_list_raises(self) -> None:
        extractor = PhysicsFeatureExtractor()
        with self.assertRaises(TypeError):
            extractor.extract([[1, 2], [3, 4]])

    def test_1d_numpy_raises(self) -> None:
        extractor = PhysicsFeatureExtractor()
        with self.assertRaises(ValueError):
            extractor.extract(np.array([1.0, 2.0, 3.0]))

    def test_3d_numpy_raises(self) -> None:
        extractor = PhysicsFeatureExtractor()
        with self.assertRaises(ValueError):
            extractor.extract(np.zeros((5, 3, 2)))


class TestNumpyArrayInput(unittest.TestCase):
    """Tests for numpy array inputs (generic column names)."""

    def test_numpy_2d_extraction(self) -> None:
        rng = np.random.RandomState(42)
        data = rng.randn(30, 6)

        extractor = PhysicsFeatureExtractor(rolling_window=3)
        result = extractor.extract(data)

        self.assertEqual(len(result), 30)
        # Generic columns won't match ICS prefixes, so no engineered
        # features beyond the originals. But the extractor should not crash.
        self.assertEqual(len(result.columns), 6)

    def test_numpy_with_ics_compatible_names(self) -> None:
        """Verify that numpy arrays with manually set ICS column names work."""
        rng = np.random.RandomState(42)
        data = rng.randn(30, 4)
        df = pd.DataFrame(data, columns=["FIT1", "FIT2", "PIT1", "MV1"])

        extractor = PhysicsFeatureExtractor(rolling_window=3)
        result = extractor.extract(df)

        self.assertGreater(len(result.columns), 4)


class TestPhysicsFeatureValues(unittest.TestCase):
    """Spot-check computed feature values for correctness."""

    def setUp(self) -> None:
        self.extractor = PhysicsFeatureExtractor(rolling_window=3)

    def test_rate_of_change_first_element_zero(self) -> None:
        values = np.array([1.0, 3.0, 6.0, 10.0, 15.0])
        roc = PhysicsFeatureExtractor._rate_of_change(values)
        self.assertEqual(roc[0], 0.0)

    def test_rate_of_change_values(self) -> None:
        values = np.array([1.0, 3.0, 6.0, 10.0, 15.0])
        roc = PhysicsFeatureExtractor._rate_of_change(values)
        np.testing.assert_array_equal(roc, [0.0, 2.0, 3.0, 4.0, 5.0])

    def test_delta_first_element_zero(self) -> None:
        values = np.array([5.0, 8.0, 3.0])
        delta = PhysicsFeatureExtractor._delta(values)
        self.assertEqual(delta[0], 0.0)

    def test_delta_values(self) -> None:
        values = np.array([5.0, 8.0, 3.0])
        delta = PhysicsFeatureExtractor._delta(values)
        np.testing.assert_array_almost_equal(delta, [0.0, 3.0, -2.0])

    def test_abs_delta_non_negative(self) -> None:
        values = np.array([5.0, 8.0, 3.0])
        abs_d = PhysicsFeatureExtractor._abs_delta(values)
        self.assertTrue(np.all(abs_d >= 0))

    def test_flow_balance(self) -> None:
        a = np.array([10.0, 20.0, 30.0])
        b = np.array([5.0, 15.0, 25.0])
        balance = PhysicsFeatureExtractor._flow_balance(a, b)
        np.testing.assert_array_equal(balance, [5.0, 5.0, 5.0])

    def test_pressure_difference(self) -> None:
        a = np.array([100.0, 200.0])
        b = np.array([90.0, 180.0])
        diff = PhysicsFeatureExtractor._pressure_difference(a, b)
        np.testing.assert_array_equal(diff, [10.0, 20.0])

    def test_pump_transition_no_change(self) -> None:
        values = np.array([1.0, 1.0, 1.0, 1.0])
        transitions = PhysicsFeatureExtractor._pump_transition(values)
        np.testing.assert_array_equal(transitions, [0.0, 0.0, 0.0, 0.0])

    def test_pump_transition_with_change(self) -> None:
        values = np.array([0.0, 0.0, 1.0, 1.0, 0.0])
        transitions = PhysicsFeatureExtractor._pump_transition(values)
        np.testing.assert_array_equal(transitions, [0.0, 0.0, 1.0, 0.0, 1.0])

    def test_valve_transition_with_change(self) -> None:
        values = np.array([0.0, 1.0, 1.0, 0.0])
        transitions = PhysicsFeatureExtractor._valve_transition(values)
        np.testing.assert_array_equal(transitions, [0.0, 1.0, 0.0, 1.0])


class TestCrossDatasetCompatibility(unittest.TestCase):
    """Verify that the extractor works with both SWaT and HAI data."""

    def test_swat_extraction(self) -> None:
        window = _swat_window(rows=30, seed=42)
        extractor = PhysicsFeatureExtractor(rolling_window=3)
        result = extractor.extract(window)
        self.assertGreater(len(result.columns), len(window.columns))

    def test_hai_extraction(self) -> None:
        window = _hai_window(rows=30, seed=42)
        extractor = PhysicsFeatureExtractor(rolling_window=3)
        result = extractor.extract(window)
        self.assertGreater(len(result.columns), len(window.columns))

    def test_hai_label_detected_as_attack(self) -> None:
        window = _hai_window(rows=30, seed=42)
        extractor = PhysicsFeatureExtractor(rolling_window=3)
        extractor.extract(window)
        self.assertEqual(extractor.label_column, "attack")


class TestPerformance(unittest.TestCase):
    """Performance test on a moderate batch to ensure reasonable speed."""

    def test_batch_performance(self) -> None:
        """Extract features from 200 windows of size 50 within 30 seconds."""
        windows = [_swat_window(rows=50, seed=s) for s in range(200)]
        extractor = PhysicsFeatureExtractor(rolling_window=5)

        start = time.perf_counter()
        results = extractor.extract_batch(windows)
        elapsed = time.perf_counter() - start

        self.assertEqual(len(results), 200)
        self.assertLess(
            elapsed,
            30.0,
            f"Batch extraction of 200 windows took {elapsed:.2f}s "
            f"(expected < 30s)",
        )


if __name__ == "__main__":
    unittest.main()
