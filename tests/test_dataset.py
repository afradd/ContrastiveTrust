"""Tests for :mod:`datasets.contrastive_dataset` and :mod:`datasets.dataloader`.

Comprehensive test suite verifying tensor conversion, shape correctness,
dtype enforcement, validation guards, DataLoader batching, and
device compatibility for the contrastive learning data pipeline.
"""

from __future__ import annotations

import unittest
from typing import Any

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader

from datasets.contrastive_dataset import ContrastiveDataset
from datasets.dataloader import (
    create_test_dataloader,
    create_train_dataloader,
    create_validation_dataloader,
)


# ---------------------------------------------------------------------------
# Test data helpers
# ---------------------------------------------------------------------------

_NUM_WINDOWS: int = 200
_WINDOW_SIZE: int = 60
_NUM_FEATURES: int = 86
_SEED: int = 42


def _synthetic_windows(
    num_windows: int = _NUM_WINDOWS,
    window_size: int = _WINDOW_SIZE,
    num_features: int = _NUM_FEATURES,
    seed: int = _SEED,
) -> np.ndarray:
    """Generate synthetic ICS-like sliding windows.

    Args:
        num_windows: Number of windows.
        window_size: Time steps per window.
        num_features: Features per time step.
        seed: Random seed for reproducibility.

    Returns:
        A ``float64`` numpy array of shape
        ``(num_windows, window_size, num_features)``.
    """
    rng = np.random.RandomState(seed)
    return rng.randn(num_windows, window_size, num_features).astype(np.float64)


def _synthetic_labels(
    num_windows: int = _NUM_WINDOWS,
    seed: int = _SEED,
) -> np.ndarray:
    """Generate synthetic binary anomaly labels.

    Args:
        num_windows: Number of labels.
        seed: Random seed.

    Returns:
        A ``int64`` numpy array of shape ``(num_windows,)``.
    """
    rng = np.random.RandomState(seed)
    return rng.choice([0, 1], size=num_windows).astype(np.int64)


def _synthetic_timestamps(
    num_windows: int = _NUM_WINDOWS,
) -> np.ndarray:
    """Generate synthetic timestamps as numpy datetime64 array.

    Args:
        num_windows: Number of timestamps.

    Returns:
        A numpy array of datetime64 values.
    """
    import pandas as pd

    return pd.date_range(
        "2024-01-01", periods=num_windows, freq="s"
    ).to_numpy()


# ---------------------------------------------------------------------------
# Dataset tests
# ---------------------------------------------------------------------------

class DatasetCreationTests(unittest.TestCase):
    """Verify dataset construction and metadata."""

    def test_creation_windows_only(self) -> None:
        """Dataset accepts windows without labels or timestamps."""
        windows = _synthetic_windows()
        ds = ContrastiveDataset(windows=windows)
        self.assertEqual(len(ds), _NUM_WINDOWS)
        self.assertFalse(ds.has_labels)
        self.assertFalse(ds.has_timestamps)

    def test_creation_with_labels(self) -> None:
        """Dataset accepts windows with labels."""
        windows = _synthetic_windows()
        labels = _synthetic_labels()
        ds = ContrastiveDataset(windows=windows, labels=labels)
        self.assertEqual(len(ds), _NUM_WINDOWS)
        self.assertTrue(ds.has_labels)

    def test_creation_with_timestamps(self) -> None:
        """Dataset accepts windows with timestamps."""
        windows = _synthetic_windows()
        timestamps = _synthetic_timestamps()
        ds = ContrastiveDataset(windows=windows, timestamps=timestamps)
        self.assertEqual(len(ds), _NUM_WINDOWS)
        self.assertTrue(ds.has_timestamps)

    def test_creation_full(self) -> None:
        """Dataset accepts windows, labels, and timestamps."""
        windows = _synthetic_windows()
        labels = _synthetic_labels()
        timestamps = _synthetic_timestamps()
        ds = ContrastiveDataset(
            windows=windows, labels=labels, timestamps=timestamps,
        )
        self.assertEqual(len(ds), _NUM_WINDOWS)
        self.assertTrue(ds.has_labels)
        self.assertTrue(ds.has_timestamps)

    def test_shape_metadata(self) -> None:
        """Shape attributes correctly reflect input dimensions."""
        ds = ContrastiveDataset(windows=_synthetic_windows())
        self.assertEqual(ds.num_windows, _NUM_WINDOWS)
        self.assertEqual(ds.window_size, _WINDOW_SIZE)
        self.assertEqual(ds.num_features, _NUM_FEATURES)
        self.assertEqual(ds.shape, (_NUM_WINDOWS, _WINDOW_SIZE, _NUM_FEATURES))


class DatasetLengthTests(unittest.TestCase):
    """Test 1: Dataset length matches input."""

    def test_length_matches_input(self) -> None:
        ds = ContrastiveDataset(windows=_synthetic_windows(num_windows=500))
        self.assertEqual(len(ds), 500)

    def test_length_single_window(self) -> None:
        ds = ContrastiveDataset(windows=_synthetic_windows(num_windows=1))
        self.assertEqual(len(ds), 1)


class SampleTensorShapeTests(unittest.TestCase):
    """Test 2: Sample tensor shape is correct."""

    def test_window_shape(self) -> None:
        ds = ContrastiveDataset(windows=_synthetic_windows())
        sample = ds[0]
        self.assertEqual(
            tuple(sample["window"].shape),
            (_WINDOW_SIZE, _NUM_FEATURES),
        )

    def test_window_shape_varied(self) -> None:
        ds = ContrastiveDataset(
            windows=_synthetic_windows(window_size=30, num_features=10),
        )
        sample = ds[0]
        self.assertEqual(tuple(sample["window"].shape), (30, 10))


class TensorDtypeTests(unittest.TestCase):
    """Test 3: Window tensor dtype is float32."""

    def test_window_dtype_float32(self) -> None:
        ds = ContrastiveDataset(windows=_synthetic_windows())
        sample = ds[0]
        self.assertEqual(sample["window"].dtype, torch.float32)

    def test_window_dtype_from_float64_input(self) -> None:
        """float64 numpy input is converted to float32."""
        windows = _synthetic_windows().astype(np.float64)
        ds = ContrastiveDataset(windows=windows)
        self.assertEqual(ds[0]["window"].dtype, torch.float32)


class LabelDtypeTests(unittest.TestCase):
    """Test 4: Label tensor dtype is long (int64)."""

    def test_label_dtype_long(self) -> None:
        ds = ContrastiveDataset(
            windows=_synthetic_windows(),
            labels=_synthetic_labels(),
        )
        sample = ds[0]
        self.assertEqual(sample["label"].dtype, torch.long)

    def test_label_dtype_from_float_input(self) -> None:
        """Float labels are converted to long."""
        labels = _synthetic_labels().astype(np.float32)
        ds = ContrastiveDataset(
            windows=_synthetic_windows(),
            labels=labels,
        )
        self.assertEqual(ds[0]["label"].dtype, torch.long)


class BatchShapeTests(unittest.TestCase):
    """Test 5: Batch shapes are correct through DataLoader."""

    def test_train_batch_shape(self) -> None:
        batch_size = 32
        ds = ContrastiveDataset(
            windows=_synthetic_windows(),
            labels=_synthetic_labels(),
        )
        loader = create_train_dataloader(ds, batch_size=batch_size)
        batch = next(iter(loader))
        self.assertEqual(
            tuple(batch["window"].shape),
            (batch_size, _WINDOW_SIZE, _NUM_FEATURES),
        )
        self.assertEqual(tuple(batch["label"].shape), (batch_size,))

    def test_validation_batch_preserves_all(self) -> None:
        """Validation loader does not drop last batch."""
        num_windows = 100
        batch_size = 32
        ds = ContrastiveDataset(
            windows=_synthetic_windows(num_windows=num_windows),
        )
        loader = create_validation_dataloader(ds, batch_size=batch_size)
        total_samples = sum(b["window"].shape[0] for b in loader)
        self.assertEqual(total_samples, num_windows)


class ShuffleConfigurationTests(unittest.TestCase):
    """Test 6: Shuffle configuration is correct."""

    def test_train_loader_shuffles(self) -> None:
        """Training loader has shuffle=True by default."""
        ds = ContrastiveDataset(windows=_synthetic_windows())
        loader = create_train_dataloader(ds)
        # DataLoader stores the sampler; a RandomSampler indicates shuffle
        from torch.utils.data import RandomSampler
        self.assertIsInstance(loader.sampler, RandomSampler)

    def test_validation_loader_no_shuffle(self) -> None:
        """Validation loader has shuffle=False by default."""
        ds = ContrastiveDataset(windows=_synthetic_windows())
        loader = create_validation_dataloader(ds)
        from torch.utils.data import SequentialSampler
        self.assertIsInstance(loader.sampler, SequentialSampler)

    def test_test_loader_no_shuffle(self) -> None:
        """Test loader has shuffle=False by default."""
        ds = ContrastiveDataset(windows=_synthetic_windows())
        loader = create_test_dataloader(ds)
        from torch.utils.data import SequentialSampler
        self.assertIsInstance(loader.sampler, SequentialSampler)


class NoNaNTests(unittest.TestCase):
    """Test 7: No NaN values in tensors."""

    def test_no_nan_in_windows(self) -> None:
        ds = ContrastiveDataset(windows=_synthetic_windows())
        self.assertFalse(torch.isnan(ds.windows).any())

    def test_nan_input_raises(self) -> None:
        """Dataset rejects windows containing NaN."""
        windows = _synthetic_windows()
        windows[0, 0, 0] = np.nan
        with self.assertRaisesRegex(ValueError, "NaN"):
            ContrastiveDataset(windows=windows)


class DeviceCompatibilityTests(unittest.TestCase):
    """Test 8: Device compatibility."""

    def test_cpu_tensors(self) -> None:
        ds = ContrastiveDataset(windows=_synthetic_windows())
        sample = ds[0]
        self.assertEqual(sample["window"].device, torch.device("cpu"))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA not available")
    def test_gpu_transfer(self) -> None:
        """Tensors can be moved to CUDA."""
        ds = ContrastiveDataset(windows=_synthetic_windows(num_windows=10))
        sample = ds[0]
        gpu_window = sample["window"].cuda()
        self.assertTrue(gpu_window.is_cuda)
        self.assertEqual(tuple(gpu_window.shape), (_WINDOW_SIZE, _NUM_FEATURES))

    def test_tensor_input(self) -> None:
        """Dataset accepts pre-built torch tensors."""
        windows_tensor = torch.randn(50, 30, 10, dtype=torch.float32)
        ds = ContrastiveDataset(windows=windows_tensor)
        self.assertEqual(len(ds), 50)
        self.assertEqual(ds[0]["window"].dtype, torch.float32)


class TimestampPreservationTests(unittest.TestCase):
    """Test 9: Timestamp preservation."""

    def test_timestamps_preserved(self) -> None:
        timestamps = _synthetic_timestamps()
        ds = ContrastiveDataset(
            windows=_synthetic_windows(),
            timestamps=timestamps,
        )
        # Timestamps are stored as strings for DataLoader collation
        for i in range(min(5, len(ds))):
            sample = ds[i]
            self.assertEqual(sample["timestamp"], str(timestamps[i]))

    def test_timestamps_none_when_absent(self) -> None:
        ds = ContrastiveDataset(windows=_synthetic_windows())
        sample = ds[0]
        self.assertNotIn("timestamp", sample)


class DictionaryOutputTests(unittest.TestCase):
    """Test 10: Correct dictionary output."""

    def test_output_is_dict(self) -> None:
        ds = ContrastiveDataset(windows=_synthetic_windows())
        sample = ds[0]
        self.assertIsInstance(sample, dict)

    def test_keys_without_labels(self) -> None:
        ds = ContrastiveDataset(windows=_synthetic_windows())
        sample = ds[0]
        self.assertIn("window", sample)
        self.assertNotIn("label", sample)

    def test_keys_with_labels(self) -> None:
        ds = ContrastiveDataset(
            windows=_synthetic_windows(),
            labels=_synthetic_labels(),
        )
        sample = ds[0]
        self.assertIn("window", sample)
        self.assertIn("label", sample)

    def test_keys_with_all(self) -> None:
        ds = ContrastiveDataset(
            windows=_synthetic_windows(),
            labels=_synthetic_labels(),
            timestamps=_synthetic_timestamps(),
        )
        sample = ds[0]
        self.assertIn("window", sample)
        self.assertIn("label", sample)
        self.assertIn("timestamp", sample)

    def test_window_value_is_tensor(self) -> None:
        ds = ContrastiveDataset(windows=_synthetic_windows())
        sample = ds[0]
        self.assertIsInstance(sample["window"], Tensor)

    def test_label_value_is_tensor(self) -> None:
        ds = ContrastiveDataset(
            windows=_synthetic_windows(),
            labels=_synthetic_labels(),
        )
        sample = ds[0]
        self.assertIsInstance(sample["label"], Tensor)


# ---------------------------------------------------------------------------
# Validation tests
# ---------------------------------------------------------------------------

class ValidationTests(unittest.TestCase):
    """Verify rejection of invalid inputs."""

    def test_empty_windows_raises(self) -> None:
        with self.assertRaisesRegex(ValueError, "at least one sample"):
            ContrastiveDataset(
                windows=np.empty((0, _WINDOW_SIZE, _NUM_FEATURES)),
            )

    def test_non_3d_windows_raises(self) -> None:
        with self.assertRaisesRegex(ValueError, "3-dimensional"):
            ContrastiveDataset(
                windows=np.random.randn(_NUM_WINDOWS, _NUM_FEATURES),
            )

    def test_4d_windows_raises(self) -> None:
        with self.assertRaisesRegex(ValueError, "3-dimensional"):
            ContrastiveDataset(
                windows=np.random.randn(10, 5, 3, 2),
            )

    def test_invalid_windows_type_raises(self) -> None:
        with self.assertRaises(TypeError):
            ContrastiveDataset(windows=[[1, 2, 3]])  # type: ignore[arg-type]

    def test_label_length_mismatch_raises(self) -> None:
        with self.assertRaisesRegex(ValueError, "does not match"):
            ContrastiveDataset(
                windows=_synthetic_windows(num_windows=100),
                labels=_synthetic_labels(num_windows=50),
            )

    def test_timestamp_length_mismatch_raises(self) -> None:
        with self.assertRaisesRegex(ValueError, "does not match"):
            ContrastiveDataset(
                windows=_synthetic_windows(num_windows=100),
                timestamps=_synthetic_timestamps(num_windows=50),
            )

    def test_index_out_of_range_raises(self) -> None:
        ds = ContrastiveDataset(windows=_synthetic_windows(num_windows=10))
        with self.assertRaises(IndexError):
            _ = ds[10]

    def test_negative_index_out_of_range_raises(self) -> None:
        ds = ContrastiveDataset(windows=_synthetic_windows(num_windows=10))
        with self.assertRaises(IndexError):
            _ = ds[-1]

    def test_zero_window_size_raises(self) -> None:
        with self.assertRaisesRegex(ValueError, "window_size"):
            ContrastiveDataset(
                windows=np.empty((10, 0, 5)),
            )

    def test_zero_features_raises(self) -> None:
        with self.assertRaisesRegex(ValueError, "num_features"):
            ContrastiveDataset(
                windows=np.empty((10, 5, 0)),
            )


# ---------------------------------------------------------------------------
# DataLoader factory tests
# ---------------------------------------------------------------------------

class DataLoaderFactoryTests(unittest.TestCase):
    """Verify DataLoader factory functions."""

    def test_invalid_dataset_type_raises(self) -> None:
        with self.assertRaises(TypeError):
            create_train_dataloader(dataset="not_a_dataset")  # type: ignore[arg-type]

    def test_invalid_batch_size_raises(self) -> None:
        ds = ContrastiveDataset(windows=_synthetic_windows())
        with self.assertRaises(ValueError):
            create_train_dataloader(ds, batch_size=0)

    def test_negative_batch_size_raises(self) -> None:
        ds = ContrastiveDataset(windows=_synthetic_windows())
        with self.assertRaises(ValueError):
            create_train_dataloader(ds, batch_size=-1)

    def test_bool_batch_size_raises(self) -> None:
        ds = ContrastiveDataset(windows=_synthetic_windows())
        with self.assertRaises(TypeError):
            create_train_dataloader(ds, batch_size=True)  # type: ignore[arg-type]

    def test_train_drop_last_default(self) -> None:
        """Training loader drops last batch by default."""
        num_windows = 100
        batch_size = 32
        ds = ContrastiveDataset(
            windows=_synthetic_windows(num_windows=num_windows),
        )
        loader = create_train_dataloader(ds, batch_size=batch_size)
        total_samples = sum(b["window"].shape[0] for b in loader)
        # 100 // 32 = 3 batches of 32 = 96 (drops 4)
        self.assertEqual(total_samples, 96)

    def test_test_preserves_all_samples(self) -> None:
        """Test loader does not drop any samples by default."""
        num_windows = 100
        batch_size = 32
        ds = ContrastiveDataset(
            windows=_synthetic_windows(num_windows=num_windows),
        )
        loader = create_test_dataloader(ds, batch_size=batch_size)
        total_samples = sum(b["window"].shape[0] for b in loader)
        self.assertEqual(total_samples, num_windows)

    def test_custom_batch_size(self) -> None:
        batch_size = 16
        ds = ContrastiveDataset(windows=_synthetic_windows())
        loader = create_train_dataloader(ds, batch_size=batch_size)
        batch = next(iter(loader))
        self.assertEqual(batch["window"].shape[0], batch_size)

    def test_returns_dataloader_type(self) -> None:
        ds = ContrastiveDataset(windows=_synthetic_windows())
        loader = create_train_dataloader(ds)
        self.assertIsInstance(loader, DataLoader)


# ---------------------------------------------------------------------------
# Custom test report
# ---------------------------------------------------------------------------

def _run_with_report() -> None:
    """Execute all tests and print a clean report summary."""
    windows = _synthetic_windows()
    labels = _synthetic_labels()
    timestamps = _synthetic_timestamps()

    # ── Dataset creation ───────────────────────────────────────────
    ds_full = ContrastiveDataset(
        windows=windows, labels=labels, timestamps=timestamps,
    )
    ds_windows_only = ContrastiveDataset(windows=windows)

    # ── Sample inspection ──────────────────────────────────────────
    sample_full = ds_full[0]
    sample_simple = ds_windows_only[0]

    # ── DataLoader creation ────────────────────────────────────────
    batch_size = 32
    train_loader = create_train_dataloader(ds_full, batch_size=batch_size)
    val_loader = create_validation_dataloader(ds_full, batch_size=batch_size)
    test_loader = create_test_dataloader(ds_full, batch_size=batch_size)

    train_batch = next(iter(train_loader))

    # ── Checks ─────────────────────────────────────────────────────
    checks: list[tuple[str, bool]] = [
        ("Dataset length", len(ds_full) == _NUM_WINDOWS),
        (
            "Sample tensor shape",
            tuple(sample_full["window"].shape) == (_WINDOW_SIZE, _NUM_FEATURES),
        ),
        ("Tensor dtype (float32)", sample_full["window"].dtype == torch.float32),
        ("Label dtype (long)", sample_full["label"].dtype == torch.long),
        (
            "Batch shape",
            tuple(train_batch["window"].shape)
            == (batch_size, _WINDOW_SIZE, _NUM_FEATURES),
        ),
        (
            "Train shuffle (RandomSampler)",
            type(train_loader.sampler).__name__ == "RandomSampler",
        ),
        ("No NaN values", not torch.isnan(ds_full.windows).any().item()),
        (
            "Device (CPU)",
            sample_full["window"].device == torch.device("cpu"),
        ),
        (
            "Timestamp preserved",
            sample_full["timestamp"] == str(timestamps[0]),
        ),
        (
            "Dict output (with labels)",
            set(sample_full.keys()) == {"window", "label", "timestamp"},
        ),
        (
            "Dict output (without labels)",
            set(sample_simple.keys()) == {"window"},
        ),
    ]

    max_name_len = max(len(name) for name, _ in checks)

    report_lines = [
        "",
        "=" * 60,
        "  Contrastive Dataset & DataLoader Test Report",
        "=" * 60,
        "",
        f"  Windows shape:      {tuple(windows.shape)}",
        f"  Window tensor dtype: {ds_full.windows.dtype}",
        f"  Label tensor dtype:  {ds_full.labels.dtype if ds_full.labels is not None else 'N/A'}",
        f"  Batch size:          {batch_size}",
        f"  Train batches:       {len(train_loader)}",
        f"  Val batches:         {len(val_loader)}",
        f"  Test batches:        {len(test_loader)}",
        "",
        "-" * 60,
        "  Verification Checks",
        "-" * 60,
        "",
    ]

    all_passed = True
    for name, passed in checks:
        status = "PASSED" if passed else "FAILED"
        if not passed:
            all_passed = False
        report_lines.append(
            f"  {name:<{max_name_len}}  {status}"
        )

    report_lines.extend([
        "",
        "-" * 60,
        f"  Overall: {'ALL PASSED' if all_passed else 'SOME FAILED'}",
        "=" * 60,
        "",
    ])

    print("\n".join(report_lines))

    # ── Run full unittest suite ────────────────────────────────────
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromModule(__import__(__name__))
    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(suite)


if __name__ == "__main__":
    _run_with_report()
