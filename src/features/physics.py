"""Physics-aware feature engineering for industrial control system datasets.

The :class:`PhysicsFeatureExtractor` class computes domain-informed features
such as rate of change, rolling statistics, flow balances, pressure
differences, and actuator transition indicators from sliding windows of
ICS sensor and actuator data.

The extractor is dataset-agnostic and works with both SWaT and HAI column
naming conventions by auto-detecting analog sensor columns (FIT, PIT, LIT,
AIT) and actuator columns (MV, P, Status).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Column detection prefixes
# ---------------------------------------------------------------------------

_ANALOG_SENSOR_PREFIXES: tuple[str, ...] = ("FIT", "PIT", "LIT", "AIT")
_ACTUATOR_PREFIXES: tuple[str, ...] = ("MV", "P")
_ACTUATOR_SUFFIX: str = "Status"

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

_DEFAULT_CONFIG_PATH: str = "artifacts/physics_config.json"


class PhysicsFeatureExtractor:
    """Extract physics-aware features from ICS sliding windows.

    The extractor automatically detects analog sensor columns (prefixes
    ``FIT``, ``PIT``, ``LIT``, ``AIT``) and actuator columns (prefixes
    ``MV``, ``P``, or containing ``Status``) and computes the following
    engineered features:

    - **Rate of Change** (first-order difference) per analog sensor
    - **Delta** (difference from first value) per analog sensor
    - **Absolute Delta** per analog sensor
    - **Rolling Mean** per analog sensor
    - **Rolling Standard Deviation** per analog sensor
    - **Rolling Energy** (sum of squares) per analog sensor
    - **Flow Balance** (pair-wise differences between FIT sensors)
    - **Pressure Difference** (pair-wise differences between PIT sensors)
    - **Pump Transition Detection** (per pump actuator)
    - **Valve Transition Detection** (per valve actuator)

    Engineered features are concatenated with the original columns to
    produce an enriched output.

    Attributes
    ----------
    rolling_window : int
        Number of rows used for rolling statistics.
    timestamp_column : str or None
        Resolved name of the timestamp column (preserved, not engineered).
    label_column : str or None
        Resolved name of the label column (preserved, not engineered).
    analog_columns : list of str
        Detected analog sensor columns.
    actuator_columns : list of str
        Detected actuator columns.
    fit_columns : list of str
        Detected FIT (flow) sensor columns.
    pit_columns : list of str
        Detected PIT (pressure) sensor columns.
    pump_columns : list of str
        Detected pump actuator columns.
    valve_columns : list of str
        Detected MV (valve) actuator columns.
    """

    def __init__(
        self,
        rolling_window: int = 5,
        timestamp_column: str | None = None,
        label_column: str | None = None,
    ) -> None:
        """Initialise the physics feature extractor.

        Parameters
        ----------
        rolling_window : int
            Number of rows for rolling statistics.  Must be a positive
            integer ≥ 2.
        timestamp_column : str or None
            Optional explicit timestamp column name.
        label_column : str or None
            Optional explicit label column name.

        Raises
        ------
        TypeError
            If ``rolling_window`` is not an integer.
        ValueError
            If ``rolling_window`` is less than 2.
        """
        if isinstance(rolling_window, bool) or not isinstance(rolling_window, int):
            raise TypeError(
                f"rolling_window must be a positive integer, "
                f"got {type(rolling_window).__name__}"
            )
        if rolling_window < 2:
            raise ValueError(
                f"rolling_window must be at least 2, got {rolling_window}"
            )

        self.rolling_window: int = rolling_window

        self._explicit_timestamp_column: str | None = timestamp_column
        self._explicit_label_column: str | None = label_column

        # Resolved after first call to extract()
        self.timestamp_column: str | None = None
        self.label_column: str | None = None
        self.analog_columns: list[str] = []
        self.actuator_columns: list[str] = []
        self.fit_columns: list[str] = []
        self.pit_columns: list[str] = []
        self.pump_columns: list[str] = []
        self.valve_columns: list[str] = []

        self._feature_names: list[str] = []
        self._is_configured: bool = False

    # ------------------------------------------------------------------
    # Core public API
    # ------------------------------------------------------------------

    def extract(
        self,
        data: Union[pd.DataFrame, np.ndarray],
    ) -> pd.DataFrame:
        """Extract physics-aware features from a single window.

        Parameters
        ----------
        data : pandas.DataFrame or numpy.ndarray
            A 2-D array or DataFrame representing one sliding window with
            shape ``(window_length, num_columns)``.  If a numpy array is
            provided, generic column names are generated.

        Returns
        -------
        pandas.DataFrame
            The original columns concatenated with all engineered features.
            Timestamps and labels are preserved unchanged.

        Raises
        ------
        TypeError
            If ``data`` is not a DataFrame or numpy array.
        ValueError
            If ``data`` is empty, not 2-D, or contains NaN/Inf values in
            numeric feature columns.
        """
        dataframe = self._coerce_to_dataframe(data)
        self._validate_dataframe(dataframe)

        if not self._is_configured:
            self._configure(dataframe)

        return self._extract_single(dataframe)

    def extract_batch(
        self,
        batch: Union[list[pd.DataFrame], list[np.ndarray], np.ndarray],
    ) -> list[pd.DataFrame]:
        """Extract physics-aware features from a batch of windows.

        Parameters
        ----------
        batch : list of DataFrame, list of ndarray, or 3-D ndarray
            A collection of windows.  If a 3-D numpy array is supplied,
            its first axis is treated as the batch dimension.

        Returns
        -------
        list of pandas.DataFrame
            One enriched DataFrame per window.

        Raises
        ------
        TypeError
            If ``batch`` is not a list or 3-D numpy array.
        ValueError
            If ``batch`` is empty.
        """
        windows = self._coerce_batch(batch)

        if len(windows) == 0:
            raise ValueError("batch must contain at least one window")

        logger.info("Extracting physics features for batch of %d windows", len(windows))

        results: list[pd.DataFrame] = []
        for idx, window in enumerate(windows):
            result = self.extract(window)
            results.append(result)
            if (idx + 1) % 100 == 0:
                logger.debug("Processed %d / %d windows", idx + 1, len(windows))

        logger.info(
            "Batch extraction complete: %d windows, %d features each",
            len(results),
            len(results[0].columns) if results else 0,
        )
        return results

    def get_feature_names(self) -> list[str]:
        """Return the ordered list of output feature names.

        Returns
        -------
        list of str
            Column names of the DataFrame produced by :meth:`extract`,
            including both original and engineered feature names.

        Raises
        ------
        RuntimeError
            If :meth:`extract` has not been called yet.
        """
        if not self._is_configured:
            raise RuntimeError(
                "PhysicsFeatureExtractor has not been configured yet. "
                "Call extract() on at least one window first."
            )
        return list(self._feature_names)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save_config(self, path: str | Path | None = None) -> Path:
        """Save extractor configuration to a JSON file.

        Parameters
        ----------
        path : str or Path or None
            Destination path.  Defaults to ``artifacts/physics_config.json``.

        Returns
        -------
        pathlib.Path
            Resolved path where the configuration was written.

        Raises
        ------
        RuntimeError
            If the extractor has not been configured yet.
        """
        if not self._is_configured:
            raise RuntimeError(
                "PhysicsFeatureExtractor has not been configured yet. "
                "Call extract() before save_config()."
            )

        save_path = Path(path) if path is not None else Path(_DEFAULT_CONFIG_PATH)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        config: dict[str, Any] = {
            "rolling_window": self.rolling_window,
            "timestamp_column": self.timestamp_column,
            "label_column": self.label_column,
            "explicit_timestamp_column": self._explicit_timestamp_column,
            "explicit_label_column": self._explicit_label_column,
            "analog_columns": self.analog_columns,
            "actuator_columns": self.actuator_columns,
            "fit_columns": self.fit_columns,
            "pit_columns": self.pit_columns,
            "pump_columns": self.pump_columns,
            "valve_columns": self.valve_columns,
            "feature_names": self._feature_names,
        }

        with open(save_path, "w", encoding="utf-8") as fp:
            json.dump(config, fp, indent=2)

        logger.info("Physics feature configuration saved to %s", save_path)
        return save_path

    @classmethod
    def load_config(cls, path: str | Path | None = None) -> PhysicsFeatureExtractor:
        """Load extractor configuration from a JSON file.

        Parameters
        ----------
        path : str or Path or None
            Path to the configuration file.  Defaults to
            ``artifacts/physics_config.json``.

        Returns
        -------
        PhysicsFeatureExtractor
            A fully restored extractor instance.

        Raises
        ------
        FileNotFoundError
            If the configuration file does not exist.
        ValueError
            If the loaded configuration is invalid or corrupt.
        """
        load_path = Path(path) if path is not None else Path(_DEFAULT_CONFIG_PATH)

        if not load_path.exists():
            raise FileNotFoundError(
                f"Physics feature configuration not found at {load_path}"
            )

        with open(load_path, "r", encoding="utf-8") as fp:
            config: dict[str, Any] = json.load(fp)

        required_keys = {
            "rolling_window",
            "analog_columns",
            "actuator_columns",
            "fit_columns",
            "pit_columns",
            "pump_columns",
            "valve_columns",
            "feature_names",
        }
        missing_keys = required_keys - set(config.keys())
        if missing_keys:
            raise ValueError(
                f"Loaded configuration is missing required keys: "
                f"{sorted(missing_keys)}"
            )

        extractor = cls(
            rolling_window=config["rolling_window"],
            timestamp_column=config.get("explicit_timestamp_column"),
            label_column=config.get("explicit_label_column"),
        )

        extractor.timestamp_column = config.get("timestamp_column")
        extractor.label_column = config.get("label_column")
        extractor.analog_columns = config["analog_columns"]
        extractor.actuator_columns = config["actuator_columns"]
        extractor.fit_columns = config["fit_columns"]
        extractor.pit_columns = config["pit_columns"]
        extractor.pump_columns = config["pump_columns"]
        extractor.valve_columns = config["valve_columns"]
        extractor._feature_names = config["feature_names"]
        extractor._is_configured = True

        logger.info(
            "Physics feature configuration loaded from %s "
            "(%d feature columns)",
            load_path,
            len(extractor._feature_names),
        )
        return extractor

    # ------------------------------------------------------------------
    # Configuration (column detection)
    # ------------------------------------------------------------------

    def _configure(self, dataframe: pd.DataFrame) -> None:
        """Detect column roles and build the output feature name list.

        Parameters
        ----------
        dataframe : pandas.DataFrame
            Representative window used to detect columns.
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

        preserved = {
            col for col in (self.timestamp_column, self.label_column)
            if col is not None
        }

        numeric_columns = [
            col for col in dataframe.columns
            if col not in preserved
            and pd.api.types.is_numeric_dtype(dataframe[col])
        ]

        self.analog_columns = self._detect_analog_columns(numeric_columns)
        self.actuator_columns = self._detect_actuator_columns(numeric_columns)

        self.fit_columns = [
            col for col in self.analog_columns
            if self._column_has_prefix(col, ("FIT",))
        ]
        self.pit_columns = [
            col for col in self.analog_columns
            if self._column_has_prefix(col, ("PIT",))
        ]
        self.pump_columns = [
            col for col in self.actuator_columns
            if self._column_has_prefix(col, ("P",))
            and not self._column_has_prefix(col, ("PIT",))
        ]
        self.valve_columns = [
            col for col in self.actuator_columns
            if self._column_has_prefix(col, ("MV",))
        ]

        # Build the complete output feature name list
        self._feature_names = self._build_feature_names(dataframe)
        self._is_configured = True

        logger.info(
            "PhysicsFeatureExtractor configured: "
            "%d analog sensors, %d actuators, %d FIT, %d PIT, "
            "%d pumps, %d valves -> %d total output features",
            len(self.analog_columns),
            len(self.actuator_columns),
            len(self.fit_columns),
            len(self.pit_columns),
            len(self.pump_columns),
            len(self.valve_columns),
            len(self._feature_names),
        )

    # ------------------------------------------------------------------
    # Feature extraction (single window)
    # ------------------------------------------------------------------

    def _extract_single(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """Compute all physics features for a single window DataFrame.

        Parameters
        ----------
        dataframe : pandas.DataFrame
            Validated window DataFrame.

        Returns
        -------
        pandas.DataFrame
            Original columns plus engineered features.
        """
        result = dataframe.copy(deep=True)

        # Analog sensor features
        for col in self.analog_columns:
            values = dataframe[col].to_numpy(dtype=np.float64)
            result[f"{col}__roc"] = self._rate_of_change(values)
            result[f"{col}__delta"] = self._delta(values)
            result[f"{col}__abs_delta"] = self._abs_delta(values)
            result[f"{col}__rolling_mean"] = self._rolling_mean(values)
            result[f"{col}__rolling_std"] = self._rolling_std(values)
            result[f"{col}__rolling_energy"] = self._rolling_energy(values)

        # Flow balance (pair-wise FIT differences)
        for i in range(len(self.fit_columns)):
            for j in range(i + 1, len(self.fit_columns)):
                col_a = self.fit_columns[i]
                col_b = self.fit_columns[j]
                a_vals = dataframe[col_a].to_numpy(dtype=np.float64)
                b_vals = dataframe[col_b].to_numpy(dtype=np.float64)
                result[f"flow_balance__{col_a}__{col_b}"] = (
                    self._flow_balance(a_vals, b_vals)
                )

        # Pressure difference (pair-wise PIT differences)
        for i in range(len(self.pit_columns)):
            for j in range(i + 1, len(self.pit_columns)):
                col_a = self.pit_columns[i]
                col_b = self.pit_columns[j]
                a_vals = dataframe[col_a].to_numpy(dtype=np.float64)
                b_vals = dataframe[col_b].to_numpy(dtype=np.float64)
                result[f"pressure_diff__{col_a}__{col_b}"] = (
                    self._pressure_difference(a_vals, b_vals)
                )

        # Pump transition detection
        for col in self.pump_columns:
            values = dataframe[col].to_numpy(dtype=np.float64)
            result[f"{col}__pump_transition"] = (
                self._pump_transition(values)
            )

        # Valve transition detection
        for col in self.valve_columns:
            values = dataframe[col].to_numpy(dtype=np.float64)
            result[f"{col}__valve_transition"] = (
                self._valve_transition(values)
            )

        return result

    # ------------------------------------------------------------------
    # Physics feature computations (deterministic, pure functions)
    # ------------------------------------------------------------------

    @staticmethod
    def _rate_of_change(values: np.ndarray) -> np.ndarray:
        """Compute the first-order difference (rate of change).

        The first element is set to 0.0 to preserve alignment.

        Parameters
        ----------
        values : numpy.ndarray
            1-D array of sensor readings.

        Returns
        -------
        numpy.ndarray
            Rate of change array of the same length.
        """
        roc = np.diff(values, prepend=values[0])
        return roc

    @staticmethod
    def _delta(values: np.ndarray) -> np.ndarray:
        """Compute the difference from the first value in the window.

        Parameters
        ----------
        values : numpy.ndarray
            1-D array of sensor readings.

        Returns
        -------
        numpy.ndarray
            Delta array of the same length.
        """
        return values - values[0]

    @staticmethod
    def _abs_delta(values: np.ndarray) -> np.ndarray:
        """Compute the absolute difference from the first value.

        Parameters
        ----------
        values : numpy.ndarray
            1-D array of sensor readings.

        Returns
        -------
        numpy.ndarray
            Absolute delta array of the same length.
        """
        return np.abs(values - values[0])

    def _rolling_mean(self, values: np.ndarray) -> np.ndarray:
        """Compute the rolling mean using a causal (left-aligned) window.

        The first ``rolling_window - 1`` elements use a shrinking window
        (minimum 1 observation) to avoid NaN.

        Parameters
        ----------
        values : numpy.ndarray
            1-D array of sensor readings.

        Returns
        -------
        numpy.ndarray
            Rolling mean array of the same length.
        """
        series = pd.Series(values)
        return (
            series.rolling(window=self.rolling_window, min_periods=1)
            .mean()
            .to_numpy(dtype=np.float64)
        )

    def _rolling_std(self, values: np.ndarray) -> np.ndarray:
        """Compute the rolling standard deviation.

        Uses ``ddof=0`` for a population standard deviation and
        ``min_periods=1`` to avoid NaN at the boundaries.

        Parameters
        ----------
        values : numpy.ndarray
            1-D array of sensor readings.

        Returns
        -------
        numpy.ndarray
            Rolling standard deviation array of the same length.
        """
        series = pd.Series(values)
        return (
            series.rolling(window=self.rolling_window, min_periods=1)
            .std(ddof=0)
            .to_numpy(dtype=np.float64)
        )

    def _rolling_energy(self, values: np.ndarray) -> np.ndarray:
        """Compute rolling energy (sum of squared values) over the window.

        Parameters
        ----------
        values : numpy.ndarray
            1-D array of sensor readings.

        Returns
        -------
        numpy.ndarray
            Rolling energy array of the same length.
        """
        squared = pd.Series(values ** 2)
        return (
            squared.rolling(window=self.rolling_window, min_periods=1)
            .sum()
            .to_numpy(dtype=np.float64)
        )

    @staticmethod
    def _flow_balance(
        flow_a: np.ndarray,
        flow_b: np.ndarray,
    ) -> np.ndarray:
        """Compute the flow balance (difference) between two FIT sensors.

        Parameters
        ----------
        flow_a : numpy.ndarray
            Readings from the first flow sensor.
        flow_b : numpy.ndarray
            Readings from the second flow sensor.

        Returns
        -------
        numpy.ndarray
            Element-wise difference ``flow_a - flow_b``.
        """
        return flow_a - flow_b

    @staticmethod
    def _pressure_difference(
        pressure_a: np.ndarray,
        pressure_b: np.ndarray,
    ) -> np.ndarray:
        """Compute the pressure difference between two PIT sensors.

        Parameters
        ----------
        pressure_a : numpy.ndarray
            Readings from the first pressure sensor.
        pressure_b : numpy.ndarray
            Readings from the second pressure sensor.

        Returns
        -------
        numpy.ndarray
            Element-wise difference ``pressure_a - pressure_b``.
        """
        return pressure_a - pressure_b

    @staticmethod
    def _pump_transition(values: np.ndarray) -> np.ndarray:
        """Detect state transitions in a pump actuator.

        A transition is indicated by a non-zero first-order difference
        in the pump status signal.  The first element is always 0.

        Parameters
        ----------
        values : numpy.ndarray
            1-D pump status array (typically 0/1/2).

        Returns
        -------
        numpy.ndarray
            Binary transition indicator (0 or 1) of the same length.
        """
        diff = np.diff(values, prepend=values[0])
        return (diff != 0.0).astype(np.float64)

    @staticmethod
    def _valve_transition(values: np.ndarray) -> np.ndarray:
        """Detect state transitions in a valve actuator.

        A transition is indicated by a non-zero first-order difference
        in the valve status signal.  The first element is always 0.

        Parameters
        ----------
        values : numpy.ndarray
            1-D valve status array (typically 0/1).

        Returns
        -------
        numpy.ndarray
            Binary transition indicator (0 or 1) of the same length.
        """
        diff = np.diff(values, prepend=values[0])
        return (diff != 0.0).astype(np.float64)

    # ------------------------------------------------------------------
    # Feature name construction
    # ------------------------------------------------------------------

    def _build_feature_names(self, dataframe: pd.DataFrame) -> list[str]:
        """Build the ordered list of all output column names.

        Parameters
        ----------
        dataframe : pandas.DataFrame
            Representative window DataFrame.

        Returns
        -------
        list of str
            Complete output column names in order.
        """
        names: list[str] = list(dataframe.columns)

        # Per-analog-sensor engineered features
        for col in self.analog_columns:
            names.append(f"{col}__roc")
            names.append(f"{col}__delta")
            names.append(f"{col}__abs_delta")
            names.append(f"{col}__rolling_mean")
            names.append(f"{col}__rolling_std")
            names.append(f"{col}__rolling_energy")

        # Flow balance pairs
        for i in range(len(self.fit_columns)):
            for j in range(i + 1, len(self.fit_columns)):
                names.append(
                    f"flow_balance__{self.fit_columns[i]}__{self.fit_columns[j]}"
                )

        # Pressure difference pairs
        for i in range(len(self.pit_columns)):
            for j in range(i + 1, len(self.pit_columns)):
                names.append(
                    f"pressure_diff__{self.pit_columns[i]}__{self.pit_columns[j]}"
                )

        # Pump transitions
        for col in self.pump_columns:
            names.append(f"{col}__pump_transition")

        # Valve transitions
        for col in self.valve_columns:
            names.append(f"{col}__valve_transition")

        return names

    # ------------------------------------------------------------------
    # Column detection helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _detect_analog_columns(numeric_columns: list[str]) -> list[str]:
        """Identify analog sensor columns by prefix matching.

        Parameters
        ----------
        numeric_columns : list of str
            Candidate numeric column names.

        Returns
        -------
        list of str
            Columns whose name starts with an analog sensor prefix.
        """
        result = [
            col for col in numeric_columns
            if PhysicsFeatureExtractor._column_has_prefix(
                col, _ANALOG_SENSOR_PREFIXES,
            )
        ]
        logger.info("Detected %d analog sensor column(s): %s", len(result), result)
        return result

    @staticmethod
    def _detect_actuator_columns(numeric_columns: list[str]) -> list[str]:
        """Identify actuator columns by prefix or suffix matching.

        A column is classified as an actuator if its name starts with one
        of the actuator prefixes (``MV``, ``P``) **or** contains the
        ``Status`` suffix.  Columns that match analog sensor prefixes
        are excluded to avoid double-counting (e.g. ``PIT`` should not
        match the ``P`` prefix).

        Parameters
        ----------
        numeric_columns : list of str
            Candidate numeric column names.

        Returns
        -------
        list of str
            Detected actuator column names.
        """
        result: list[str] = []
        for col in numeric_columns:
            # Skip columns already classified as analog sensors
            if PhysicsFeatureExtractor._column_has_prefix(
                col, _ANALOG_SENSOR_PREFIXES,
            ):
                continue

            is_actuator = (
                PhysicsFeatureExtractor._column_has_prefix(col, _ACTUATOR_PREFIXES)
                or _ACTUATOR_SUFFIX.lower() in col.lower()
            )
            if is_actuator:
                result.append(col)

        logger.info("Detected %d actuator column(s): %s", len(result), result)
        return result

    @staticmethod
    def _column_has_prefix(column: str, prefixes: tuple[str, ...]) -> bool:
        """Check whether a column name starts with any of the given prefixes.

        The comparison is case-insensitive and checks the first token of
        the column name (split on common ICS separators: ``.``, ``_``,
        ``-``).

        Parameters
        ----------
        column : str
            Column name.
        prefixes : tuple of str
            Prefix strings to check.

        Returns
        -------
        bool
        """
        upper_col = column.upper()
        for prefix in prefixes:
            upper_prefix = prefix.upper()
            if upper_col.startswith(upper_prefix):
                return True
        return False

    # ------------------------------------------------------------------
    # Validation helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _coerce_to_dataframe(
        data: Union[pd.DataFrame, np.ndarray],
    ) -> pd.DataFrame:
        """Convert input data to a pandas DataFrame.

        Parameters
        ----------
        data : pandas.DataFrame or numpy.ndarray
            Input data.  Numpy arrays receive generic column names.

        Returns
        -------
        pandas.DataFrame

        Raises
        ------
        TypeError
            If ``data`` is not a DataFrame or numpy array.
        ValueError
            If the array is not 2-dimensional.
        """
        if isinstance(data, pd.DataFrame):
            return data.reset_index(drop=True)

        if isinstance(data, np.ndarray):
            if data.ndim != 2:
                raise ValueError(
                    f"numpy array must be 2-dimensional, "
                    f"got {data.ndim}D with shape {data.shape}"
                )
            columns = [f"feature_{i}" for i in range(data.shape[1])]
            return pd.DataFrame(data, columns=columns)

        raise TypeError(
            f"data must be a pandas DataFrame or numpy ndarray, "
            f"got {type(data).__name__}"
        )

    @staticmethod
    def _validate_dataframe(dataframe: pd.DataFrame) -> None:
        """Run validation checks on the input DataFrame.

        Parameters
        ----------
        dataframe : pandas.DataFrame
            DataFrame to validate.

        Raises
        ------
        ValueError
            If the DataFrame is empty, contains NaN or infinite values
            in numeric columns, or has zero numeric feature columns.
        """
        if dataframe.empty:
            raise ValueError(
                "Input data is empty; cannot extract physics features "
                "from zero rows"
            )

        if len(dataframe) < 2:
            raise ValueError(
                f"Input data must contain at least 2 rows for "
                f"rate-of-change computation, got {len(dataframe)}"
            )

        # Check numeric columns for NaN / Inf
        numeric_cols = dataframe.select_dtypes(include=[np.number]).columns.tolist()
        if not numeric_cols:
            raise ValueError(
                "Input data contains no numeric columns; physics "
                "features require numeric sensor/actuator columns"
            )

        numeric_values = dataframe[numeric_cols].to_numpy(
            dtype=np.float64, na_value=np.nan,
        )
        nan_count = int(np.isnan(numeric_values).sum())
        if nan_count > 0:
            raise ValueError(
                f"Input data contains {nan_count} NaN value(s) in "
                f"numeric columns. Clean the data before extraction."
            )

        inf_count = int(np.isinf(numeric_values).sum())
        if inf_count > 0:
            raise ValueError(
                f"Input data contains {inf_count} infinite value(s) in "
                f"numeric columns. Clean the data before extraction."
            )

    @staticmethod
    def _coerce_batch(
        batch: Union[list[pd.DataFrame], list[np.ndarray], np.ndarray],
    ) -> list[Union[pd.DataFrame, np.ndarray]]:
        """Normalise a batch input into a list of windows.

        Parameters
        ----------
        batch : list or 3-D numpy.ndarray
            Batch of windows.

        Returns
        -------
        list
            Individual windows as DataFrames or 2-D arrays.

        Raises
        ------
        TypeError
            If ``batch`` is not a list or numpy array.
        ValueError
            If a numpy array batch is not 3-dimensional.
        """
        if isinstance(batch, np.ndarray):
            if batch.ndim != 3:
                raise ValueError(
                    f"numpy batch must be 3-dimensional "
                    f"(batch, rows, columns), got {batch.ndim}D "
                    f"with shape {batch.shape}"
                )
            return [batch[i] for i in range(batch.shape[0])]

        if isinstance(batch, list):
            return batch

        raise TypeError(
            f"batch must be a list or numpy ndarray, "
            f"got {type(batch).__name__}"
        )

    @staticmethod
    def _resolve_column(
        dataframe: pd.DataFrame,
        explicit_name: str | None,
        candidates: tuple[str, ...],
    ) -> str | None:
        """Resolve a column name from explicit input or candidates.

        Parameters
        ----------
        dataframe : pandas.DataFrame
            DataFrame to search.
        explicit_name : str or None
            User-supplied column name, or ``None``.
        candidates : tuple of str
            Ordered candidates for auto-detection.

        Returns
        -------
        str or None
            The resolved column name, or ``None`` if not found.
        """
        if explicit_name is not None:
            if explicit_name in dataframe.columns:
                return explicit_name
            normalised = explicit_name.strip().lower()
            for column in dataframe.columns:
                if str(column).strip().lower() == normalised:
                    return str(column)
            return None

        for candidate in candidates:
            normalised_candidate = candidate.strip().lower()
            for column in dataframe.columns:
                if str(column).strip().lower() == normalised_candidate:
                    return str(column)

        return None


__all__ = ["PhysicsFeatureExtractor"]
