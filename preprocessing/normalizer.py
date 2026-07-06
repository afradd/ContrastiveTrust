"""Reusable feature normalisation utilities for industrial control system datasets.

The :class:`FeatureNormalizer` class applies z-score normalisation
(``sklearn.preprocessing.StandardScaler``) to numeric sensor and actuator
features while preserving timestamp and label columns untouched.

Data-leakage prevention is enforced by design: only training data is used to
fit the scaler; validation and test splits are transformed using the
previously fitted statistics.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


logger = logging.getLogger(__name__)

_DEFAULT_TIMESTAMP_CANDIDATES: tuple[str, ...] = (
    "timestamp",
    "t_stamp",
    "time",
    "datetime",
    "date_time",
)
_DEFAULT_LABEL_CANDIDATES: tuple[str, ...] = (
    "label",
    "labels",
    "class",
    "target",
    "attack",
)

_DEFAULT_SCALER_PATH: str = "artifacts/scaler.pkl"


class FeatureNormalizer:
    """Normalize ICS numeric features using z-score standardisation.

    The normalizer automatically detects numeric columns, excludes timestamp
    and label columns, and applies ``StandardScaler`` to the remaining
    features.  All methods accept and return ``pd.DataFrame`` objects so that
    column names and index are preserved throughout the pipeline.

    Attributes:
        timestamp_column: Resolved name of the timestamp column, or ``None``.
        label_column: Resolved name of the label column, or ``None``.
        numeric_columns: Ordered list of detected numeric feature columns.
        ignored_columns: Ordered list of columns excluded from normalisation.
        is_fitted: Whether the scaler has been fitted to training data.
    """

    def __init__(
        self,
        timestamp_column: str | None = None,
        label_column: str | None = None,
    ) -> None:
        """Initialise the feature normalizer.

        Args:
            timestamp_column: Optional explicit timestamp column name.
                When ``None``, auto-detection is performed against a list of
                common ICS timestamp column names.
            label_column: Optional explicit label column name.
                When ``None``, auto-detection is attempted; a missing label
                column is tolerated (e.g. SWaT normal-operation data).
        """
        self._explicit_timestamp_column: str | None = timestamp_column
        self._explicit_label_column: str | None = label_column

        self._scaler: StandardScaler = StandardScaler()
        self._numeric_columns: list[str] = []
        self._ignored_columns: list[str] = []
        self._original_column_order: list[str] = []
        self._is_fitted: bool = False

        self.timestamp_column: str | None = None
        self.label_column: str | None = None

    # ------------------------------------------------------------------
    # Public properties
    # ------------------------------------------------------------------

    @property
    def numeric_columns(self) -> list[str]:
        """Return the ordered list of numeric feature columns."""
        return list(self._numeric_columns)

    @property
    def ignored_columns(self) -> list[str]:
        """Return the ordered list of ignored (non-feature) columns."""
        return list(self._ignored_columns)

    @property
    def is_fitted(self) -> bool:
        """Return whether the scaler has been fitted."""
        return self._is_fitted

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def fit(self, dataframe: pd.DataFrame) -> FeatureNormalizer:
        """Fit the scaler using training data only.

        This method computes the per-feature mean and standard deviation from
        the supplied ``dataframe`` and stores them internally.  Only training
        data should be passed here to prevent data leakage.

        Args:
            dataframe: Training-split DataFrame containing timestamp, optional
                label, and numeric feature columns.

        Returns:
            ``self`` for method chaining.

        Raises:
            TypeError: If ``dataframe`` is not a ``pd.DataFrame``.
            ValueError: If validation fails (empty frame, duplicate columns,
                missing timestamp, NaN or infinite values in features, etc.).
        """
        self._validate_dataframe(dataframe)
        self._resolve_special_columns(dataframe)
        self._detect_numeric_columns(dataframe)

        feature_values: np.ndarray = (
            dataframe[self._numeric_columns]
            .to_numpy(dtype=np.float64, copy=True)
        )

        self._scaler.fit(feature_values)
        self._is_fitted = True
        self._original_column_order = list(dataframe.columns)

        logger.info(
            "Scaler fitted on %s samples with %s numeric features",
            len(dataframe),
            len(self._numeric_columns),
        )
        return self

    def transform(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """Transform a DataFrame using the previously fitted scaler.

        Timestamp and label columns are preserved unchanged.  Column order
        is identical to the input frame.

        Args:
            dataframe: DataFrame to normalise (train, validation, or test).

        Returns:
            A new ``pd.DataFrame`` with normalised numeric features.

        Raises:
            RuntimeError: If the scaler has not been fitted yet.
            ValueError: If the DataFrame does not contain the expected
                numeric columns.
        """
        self._ensure_fitted()
        self._validate_transform_columns(dataframe)

        shape_before = dataframe.shape
        logger.info("Shape before transform: %s", shape_before)

        result: pd.DataFrame = dataframe.copy(deep=True)

        feature_values: np.ndarray = (
            result[self._numeric_columns]
            .to_numpy(dtype=np.float64, copy=True)
        )
        normalised_values: np.ndarray = self._scaler.transform(feature_values)

        result[self._numeric_columns] = normalised_values

        shape_after = result.shape
        logger.info("Shape after transform: %s", shape_after)
        return result

    def fit_transform(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """Fit the scaler and transform the training DataFrame in one step.

        Equivalent to calling ``fit(dataframe)`` followed by
        ``transform(dataframe)``.

        Args:
            dataframe: Training-split DataFrame.

        Returns:
            A new normalised ``pd.DataFrame``.
        """
        self.fit(dataframe)
        return self.transform(dataframe)

    def inverse_transform(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """Reverse the normalisation to recover original-scale values.

        Args:
            dataframe: Previously normalised DataFrame.

        Returns:
            A new ``pd.DataFrame`` with numeric features in their
            original scale.

        Raises:
            RuntimeError: If the scaler has not been fitted yet.
            ValueError: If required numeric columns are missing.
        """
        self._ensure_fitted()
        self._validate_transform_columns(dataframe)

        result: pd.DataFrame = dataframe.copy(deep=True)

        normalised_values: np.ndarray = (
            result[self._numeric_columns]
            .to_numpy(dtype=np.float64, copy=True)
        )
        original_values: np.ndarray = self._scaler.inverse_transform(
            normalised_values,
        )

        result[self._numeric_columns] = original_values
        return result

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str | Path | None = None) -> Path:
        """Persist the fitted scaler to disk using joblib.

        The entire normalizer state (scaler, column metadata, fitted flag)
        is saved so that :meth:`load` can fully restore the object.

        Args:
            path: File path for the saved artifact.  Defaults to
                ``artifacts/scaler.pkl``.

        Returns:
            The resolved ``Path`` where the artifact was written.

        Raises:
            RuntimeError: If the scaler has not been fitted yet.
        """
        self._ensure_fitted()

        save_path = Path(path) if path is not None else Path(_DEFAULT_SCALER_PATH)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        state: dict[str, Any] = {
            "scaler": self._scaler,
            "numeric_columns": self._numeric_columns,
            "ignored_columns": self._ignored_columns,
            "original_column_order": self._original_column_order,
            "timestamp_column": self.timestamp_column,
            "label_column": self.label_column,
            "explicit_timestamp_column": self._explicit_timestamp_column,
            "explicit_label_column": self._explicit_label_column,
            "is_fitted": self._is_fitted,
        }

        joblib.dump(state, save_path)
        logger.info("Scaler saved to %s", save_path)
        return save_path

    @classmethod
    def load(cls, path: str | Path | None = None) -> FeatureNormalizer:
        """Load a previously saved normalizer from disk.

        Args:
            path: File path of the saved artifact.  Defaults to
                ``artifacts/scaler.pkl``.

        Returns:
            A fully restored :class:`FeatureNormalizer` instance.

        Raises:
            FileNotFoundError: If the artifact file does not exist.
            ValueError: If the loaded state is invalid or corrupt.
        """
        load_path = Path(path) if path is not None else Path(_DEFAULT_SCALER_PATH)

        if not load_path.exists():
            raise FileNotFoundError(
                f"Scaler artifact not found at {load_path}"
            )

        state: dict[str, Any] = joblib.load(load_path)

        required_keys = {
            "scaler",
            "numeric_columns",
            "ignored_columns",
            "original_column_order",
            "timestamp_column",
            "label_column",
            "is_fitted",
        }
        missing_keys = required_keys - set(state.keys())
        if missing_keys:
            raise ValueError(
                f"Loaded state is missing required keys: {sorted(missing_keys)}"
            )

        normalizer = cls(
            timestamp_column=state.get("explicit_timestamp_column"),
            label_column=state.get("explicit_label_column"),
        )
        normalizer._scaler = state["scaler"]
        normalizer._numeric_columns = state["numeric_columns"]
        normalizer._ignored_columns = state["ignored_columns"]
        normalizer._original_column_order = state["original_column_order"]
        normalizer.timestamp_column = state["timestamp_column"]
        normalizer.label_column = state["label_column"]
        normalizer._is_fitted = state["is_fitted"]

        logger.info(
            "Scaler loaded from %s (%s numeric features)",
            load_path,
            len(normalizer._numeric_columns),
        )
        return normalizer

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    def get_statistics(self) -> dict[str, Any]:
        """Return descriptive statistics about the fitted scaler.

        Returns:
            A dictionary containing per-feature means, standard deviations,
            the number of training samples, and column metadata.

        Raises:
            RuntimeError: If the scaler has not been fitted yet.
        """
        self._ensure_fitted()

        means: np.ndarray = self._scaler.mean_
        scales: np.ndarray = self._scaler.scale_
        n_samples: int = int(self._scaler.n_samples_seen_)

        per_feature: dict[str, dict[str, float]] = {}
        for idx, column in enumerate(self._numeric_columns):
            per_feature[column] = {
                "mean": float(means[idx]),
                "std": float(scales[idx]),
            }

        return {
            "n_samples": n_samples,
            "n_features": len(self._numeric_columns),
            "numeric_columns": list(self._numeric_columns),
            "ignored_columns": list(self._ignored_columns),
            "timestamp_column": self.timestamp_column,
            "label_column": self.label_column,
            "per_feature": per_feature,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _validate_dataframe(self, dataframe: pd.DataFrame) -> None:
        """Run pre-fit validation checks on the input DataFrame.

        Args:
            dataframe: DataFrame to validate.

        Raises:
            TypeError: If the input is not a ``pd.DataFrame``.
            ValueError: If validation fails.
        """
        if not isinstance(dataframe, pd.DataFrame):
            raise TypeError(
                f"Expected a pandas DataFrame, got {type(dataframe).__name__}"
            )

        if dataframe.empty:
            raise ValueError("DataFrame is empty; cannot fit a scaler on zero rows")

        # Duplicate column names
        duplicate_columns = [
            col for col in dataframe.columns
            if list(dataframe.columns).count(col) > 1
        ]
        if duplicate_columns:
            unique_duplicates = sorted(set(str(c) for c in duplicate_columns))
            raise ValueError(
                f"DataFrame contains duplicate column names: {unique_duplicates}"
            )

        # Timestamp presence
        resolved_timestamp = self._resolve_column(
            dataframe,
            explicit_name=self._explicit_timestamp_column,
            candidates=_DEFAULT_TIMESTAMP_CANDIDATES,
        )
        if resolved_timestamp is None:
            raise ValueError(
                "DataFrame does not contain a recognisable timestamp column. "
                f"Searched candidates: {_DEFAULT_TIMESTAMP_CANDIDATES}. "
                "Pass timestamp_column explicitly if the name differs."
            )

        # Resolve label (missing label is tolerated)
        resolved_label = self._resolve_column(
            dataframe,
            explicit_name=self._explicit_label_column,
            candidates=_DEFAULT_LABEL_CANDIDATES,
        )
        if resolved_label is None and self._explicit_label_column is not None:
            raise ValueError(
                f"Explicit label column {self._explicit_label_column!r} "
                "was not found in the DataFrame"
            )
        if resolved_label is None:
            logger.info(
                "No label column detected; proceeding without label preservation"
            )

        # Detect numeric columns for NaN/Inf checks
        excluded = {
            col for col in (resolved_timestamp, resolved_label) if col is not None
        }
        numeric_cols = [
            col for col in dataframe.columns
            if col not in excluded
            and pd.api.types.is_numeric_dtype(dataframe[col])
        ]

        if not numeric_cols:
            raise ValueError(
                "No numeric feature columns detected in the DataFrame after "
                "excluding timestamp and label columns"
            )

        # NaN values in numeric features
        nan_counts = dataframe[numeric_cols].isna().sum()
        total_nans = int(nan_counts.sum())
        if total_nans > 0:
            affected = [
                str(col) for col, cnt in nan_counts.items() if int(cnt) > 0
            ]
            raise ValueError(
                f"DataFrame contains {total_nans} NaN value(s) in numeric "
                f"feature columns: {affected}. Clean the data before fitting."
            )

        # Infinite values in numeric features
        numeric_values = dataframe[numeric_cols].to_numpy(
            dtype=np.float64,
            na_value=np.nan,
        )
        inf_mask = np.isinf(numeric_values)
        total_infs = int(inf_mask.sum())
        if total_infs > 0:
            inf_per_col = inf_mask.sum(axis=0)
            affected = [
                str(numeric_cols[i])
                for i in range(len(numeric_cols))
                if int(inf_per_col[i]) > 0
            ]
            raise ValueError(
                f"DataFrame contains {total_infs} infinite value(s) in numeric "
                f"feature columns: {affected}. Clean the data before fitting."
            )

    def _resolve_special_columns(self, dataframe: pd.DataFrame) -> None:
        """Resolve and store timestamp and label column names.

        Args:
            dataframe: DataFrame to resolve columns from.
        """
        self.timestamp_column = self._resolve_column(
            dataframe,
            explicit_name=self._explicit_timestamp_column,
            candidates=_DEFAULT_TIMESTAMP_CANDIDATES,
        )
        self.label_column = self._resolve_column(
            dataframe,
            explicit_name=self._explicit_label_column,
            candidates=_DEFAULT_LABEL_CANDIDATES,
        )

    def _detect_numeric_columns(self, dataframe: pd.DataFrame) -> None:
        """Identify numeric feature columns, excluding special columns.

        The detected columns are stored in ``self._numeric_columns`` in the
        order they appear in the DataFrame.

        Args:
            dataframe: DataFrame to inspect.
        """
        excluded: set[str] = {
            col for col in (self.timestamp_column, self.label_column)
            if col is not None
        }

        numeric_columns: list[str] = []
        ignored_columns: list[str] = []

        for column in dataframe.columns:
            if column in excluded:
                ignored_columns.append(column)
                continue
            if pd.api.types.is_numeric_dtype(dataframe[column]):
                numeric_columns.append(column)
            else:
                ignored_columns.append(column)

        self._numeric_columns = numeric_columns
        self._ignored_columns = ignored_columns

        logger.info(
            "Detected %s numeric feature column(s): %s",
            len(numeric_columns),
            numeric_columns,
        )
        logger.info(
            "Ignored %s column(s): %s",
            len(ignored_columns),
            ignored_columns,
        )

    def _validate_transform_columns(self, dataframe: pd.DataFrame) -> None:
        """Verify that the DataFrame contains all expected numeric columns.

        Args:
            dataframe: DataFrame to check.

        Raises:
            ValueError: If any expected numeric column is missing.
        """
        available_columns = set(dataframe.columns)
        missing_columns = [
            col for col in self._numeric_columns
            if col not in available_columns
        ]
        if missing_columns:
            raise ValueError(
                f"DataFrame is missing {len(missing_columns)} expected "
                f"numeric column(s): {missing_columns}"
            )

    def _ensure_fitted(self) -> None:
        """Raise ``RuntimeError`` if the scaler has not been fitted.

        Raises:
            RuntimeError: If :meth:`fit` or :meth:`load` has not been called.
        """
        if not self._is_fitted:
            raise RuntimeError(
                "FeatureNormalizer has not been fitted yet. "
                "Call fit() or load() before transform/save/get_statistics."
            )

    @staticmethod
    def _resolve_column(
        dataframe: pd.DataFrame,
        explicit_name: str | None,
        candidates: tuple[str, ...],
    ) -> str | None:
        """Resolve a column name from explicit input or candidate names.

        Args:
            dataframe: DataFrame to search.
            explicit_name: User-supplied column name, or ``None``.
            candidates: Ordered tuple of candidate column names for
                auto-detection.

        Returns:
            The resolved column name, or ``None`` if no match was found.
        """
        if explicit_name is not None:
            if explicit_name in dataframe.columns:
                return explicit_name

            normalised_explicit = explicit_name.strip().lower()
            for column in dataframe.columns:
                if str(column).strip().lower() == normalised_explicit:
                    return str(column)

            return None

        for candidate in candidates:
            normalised_candidate = candidate.strip().lower()
            for column in dataframe.columns:
                if str(column).strip().lower() == normalised_candidate:
                    return str(column)

        return None


__all__ = ["FeatureNormalizer"]
