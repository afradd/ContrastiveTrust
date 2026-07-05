"""SWaT Dec2019 workbook loader.

This module loads the SWaT Excel workbook, detects the real header row,
categorizes the sensor/actuator/state/alarm columns, and returns a
structured dataclass for downstream analysis.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import pandas as pd


logger = logging.getLogger(__name__)


@dataclass(slots=True)
class SWaTData:
    """Container for a loaded SWaT worksheet.

    Attributes:
        dataframe: Loaded worksheet with the detected header row applied.
        timestamp_column: Timestamp column name, fixed to ``t_stamp``.
        label_column: Label column name, ``None`` for this dataset.
        sensor_columns: Sensor measurement columns.
        actuator_columns: Actuator/control columns.
        state_columns: Plant state columns.
        alarm_columns: Alarm indicator columns.
        metadata: Additional workbook and load metadata.
    """

    dataframe: pd.DataFrame
    timestamp_column: str
    label_column: str | None
    sensor_columns: list[str]
    actuator_columns: list[str]
    state_columns: list[str]
    alarm_columns: list[str]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class SWaTLoader:
    """Load and inspect the SWaT Excel workbook.

    The loader is intentionally workbook-centric. It does not normalize,
    window, split, or engineer features. It only loads the workbook,
    detects the correct header row, and organizes columns into the
    categories required by the repository.
    """

    file_path: Path | str
    sheet_name: str | int | None = None
    header_search_rows: int = 20

    sensor_prefixes: tuple[str, ...] = ("FIT", "LIT", "AIT", "PIT", "DPIT")
    actuator_prefixes: tuple[str, ...] = ("MV", "P", "UV")
    state_columns: tuple[str, ...] = (
        "P1_STATE",
        "P2_STATE",
        "P3_STATE",
        "P4_STATE",
        "P5_STATE",
        "P6_STATE",
    )

    def __post_init__(self) -> None:
        """Normalize the supplied workbook path."""
        self.file_path = Path(self.file_path).expanduser().resolve()

    def list_sheet_names(self) -> list[str]:
        """Return the worksheet names contained in the workbook."""
        workbook = self._open_workbook()
        sheet_names = [str(sheet_name) for sheet_name in workbook.sheet_names]
        logger.info("Detected %s worksheet(s) in %s", len(sheet_names), self.file_path)
        return sheet_names

    def load(self, sheet_name: str | int | None = None) -> SWaTData:
        """Load and classify a worksheet from the workbook.

        Args:
            sheet_name: Optional worksheet name or zero-based index. If
                omitted, ``self.sheet_name`` is used and the first sheet
                is chosen as a fallback.

        Returns:
            A ``SWaTData`` object containing the cleaned DataFrame,
            categorized column lists, and metadata.

        Raises:
            FileNotFoundError: If the workbook path does not exist.
            ValueError: If the worksheet cannot be resolved or the
                header row cannot be detected.
            RuntimeError: If workbook loading or parsing fails.
        """
        workbook = self._open_workbook()
        resolved_sheet_name = self._resolve_sheet_name(
            workbook=workbook,
            requested_sheet=sheet_name if sheet_name is not None else self.sheet_name,
        )

        logger.info(
            "Loading SWaT workbook %s from worksheet %r",
            self.file_path,
            resolved_sheet_name,
        )

        header_row = self._detect_header_row(resolved_sheet_name)
        dataframe = self._load_with_header(resolved_sheet_name, header_row)
        dataframe = self._strip_column_whitespace(dataframe)

        if "t_stamp" not in dataframe.columns:
            logger.error(
                "Detected header row %s but column 't_stamp' was not found",
                header_row,
            )
            raise ValueError("Workbook header row does not contain 't_stamp'")

        dataframe = self._convert_timestamp_column(dataframe)

        timestamp_column = "t_stamp"
        label_column = None
        sensor_columns = self._categorize_sensor_columns(dataframe.columns)
        actuator_columns = self._categorize_actuator_columns(dataframe.columns)
        state_columns = [column for column in dataframe.columns if column in self.state_columns]
        alarm_columns = [column for column in dataframe.columns if column.endswith(".Alarm")]

        categorized_columns = {
            timestamp_column,
            *sensor_columns,
            *actuator_columns,
            *state_columns,
            *alarm_columns,
        }
        metadata_columns = [
            column for column in dataframe.columns if column not in categorized_columns
        ]

        metadata = self._build_metadata(
            sheet_name=resolved_sheet_name,
            header_row=header_row,
            dataframe=dataframe,
            sensor_columns=sensor_columns,
            actuator_columns=actuator_columns,
            state_columns=state_columns,
            alarm_columns=alarm_columns,
            metadata_columns=metadata_columns,
            workbook=workbook,
        )

        logger.info(
            "Loaded SWaT worksheet %r: %s rows, %s columns, %s sensors, %s actuators, %s states, %s alarms",
            resolved_sheet_name,
            len(dataframe),
            len(dataframe.columns),
            len(sensor_columns),
            len(actuator_columns),
            len(state_columns),
            len(alarm_columns),
        )

        return SWaTData(
            dataframe=dataframe,
            timestamp_column=timestamp_column,
            label_column=label_column,
            sensor_columns=sensor_columns,
            actuator_columns=actuator_columns,
            state_columns=state_columns,
            alarm_columns=alarm_columns,
            metadata=metadata,
        )

    def _open_workbook(self) -> pd.ExcelFile:
        """Open the workbook after validating its path."""
        workbook_path = self._ensure_file_exists()
        logger.debug("Opening workbook at %s", workbook_path)
        try:
            return pd.ExcelFile(workbook_path)
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.exception("Failed to open workbook %s", workbook_path)
            raise RuntimeError(f"Failed to open workbook {workbook_path}") from exc

    def _ensure_file_exists(self) -> Path:
        """Validate that the workbook path exists and is a file."""
        workbook_path = Path(self.file_path)
        logger.debug("Validating workbook path %s", workbook_path)
        if not workbook_path.exists():
            logger.error("Workbook does not exist: %s", workbook_path)
            raise FileNotFoundError(f"Workbook does not exist: {workbook_path}")

        if not workbook_path.is_file():
            logger.error("Workbook path is not a file: %s", workbook_path)
            raise FileNotFoundError(f"Workbook path is not a file: {workbook_path}")

        return workbook_path

    def _resolve_sheet_name(
        self,
        workbook: pd.ExcelFile,
        requested_sheet: str | int | None,
    ) -> str:
        """Resolve a worksheet identifier to a concrete sheet name."""
        sheet_names = [str(sheet_name) for sheet_name in workbook.sheet_names]
        if not sheet_names:
            logger.error("Workbook %s contains no worksheets", self.file_path)
            raise ValueError(f"Workbook {self.file_path} contains no worksheets")

        if requested_sheet is None:
            logger.info("No worksheet requested; using the first available sheet")
            return sheet_names[0]

        if isinstance(requested_sheet, int):
            if requested_sheet < 0 or requested_sheet >= len(sheet_names):
                logger.error(
                    "Worksheet index %s is out of range for %s",
                    requested_sheet,
                    self.file_path,
                )
                raise ValueError(
                    f"Worksheet index {requested_sheet} is out of range for {self.file_path}"
                )

            return sheet_names[requested_sheet]

        if requested_sheet not in sheet_names:
            logger.error(
                "Worksheet %r is not present in %s",
                requested_sheet,
                self.file_path,
            )
            raise ValueError(
                f"Worksheet {requested_sheet!r} is not present in {self.file_path}. "
                f"Available worksheets: {sheet_names}"
            )

        return requested_sheet

    def _detect_header_row(self, sheet_name: str) -> int:
        """Find the row that contains ``t_stamp`` within the first rows.

        The inspected SWaT Dec2019 workbook places the column names in a
        non-default row. This method searches the first ``header_search_rows``
        rows without a header and returns the matching row index.
        """
        logger.info(
            "Searching the first %s rows of worksheet %r for header row",
            self.header_search_rows,
            sheet_name,
        )

        try:
            preview = pd.read_excel(
                self.file_path,
                sheet_name=sheet_name,
                header=None,
                nrows=self.header_search_rows,
            )
        except Exception as exc:
            logger.exception(
                "Failed to preview worksheet %r while searching for the header row",
                sheet_name,
            )
            raise RuntimeError(f"Failed to preview worksheet {sheet_name!r}") from exc

        for row_index, row in enumerate(preview.itertuples(index=False, name=None)):
            values = [self._normalize_text(value) for value in row]
            if "t_stamp" in values:
                logger.info("Detected header row %s in worksheet %r", row_index, sheet_name)
                return row_index

        logger.error(
            "Could not find 't_stamp' in the first %s rows of worksheet %r",
            self.header_search_rows,
            sheet_name,
        )
        raise ValueError(
            f"Could not detect the SWaT header row in the first {self.header_search_rows} rows"
        )

    def _load_with_header(self, sheet_name: str, header_row: int) -> pd.DataFrame:
        """Reload the workbook using the detected header row."""
        logger.info(
            "Loading worksheet %r with detected header row %s",
            sheet_name,
            header_row,
        )
        try:
            dataframe = pd.read_excel(
                self.file_path,
                sheet_name=sheet_name,
                header=header_row,
                engine="openpyxl",
            )
        except Exception as exc:
            logger.exception(
                "Failed to load worksheet %r with header row %s",
                sheet_name,
                header_row,
            )
            raise RuntimeError(
                f"Failed to load worksheet {sheet_name!r} with header row {header_row}"
            ) from exc

        logger.debug("Loaded worksheet %r with shape %s", sheet_name, dataframe.shape)
        return dataframe

    def _strip_column_whitespace(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """Strip whitespace from column names and reject duplicates."""
        logger.debug("Stripping whitespace from column names")
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
            logger.error("Duplicate columns detected after stripping whitespace: %s", duplicates)
            raise ValueError(f"Duplicate column names after stripping whitespace: {duplicates}")

        cleaned.columns = cleaned_columns
        return cleaned

    def _convert_timestamp_column(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """Convert ``t_stamp`` to pandas datetime values."""
        logger.debug("Converting t_stamp to datetime")
        converted = dataframe.copy()
        converted["t_stamp"] = pd.to_datetime(converted["t_stamp"], errors="coerce")

        null_count = int(converted["t_stamp"].isna().sum())
        if null_count:
            logger.warning(
                "t_stamp conversion produced %s NaT values; check workbook timestamp format",
                null_count,
            )
        else:
            logger.info("Converted t_stamp column to datetime successfully")

        return converted

    def _categorize_sensor_columns(self, columns: Iterable[str]) -> list[str]:
        """Categorize SWaT sensor columns.

        Sensors are the FIT, LIT, AIT, PIT, and DPIT columns ending in
        ``.Pv``.
        """
        sensor_columns: list[str] = []
        for column in columns:
            if column == "t_stamp":
                continue

            upper = column.upper()
            if not upper.endswith(".PV"):
                continue

            if any(upper.startswith(prefix) for prefix in self.sensor_prefixes):
                sensor_columns.append(column)

        logger.debug("Detected %s sensor column(s)", len(sensor_columns))
        return sensor_columns

    def _categorize_actuator_columns(self, columns: Iterable[str]) -> list[str]:
        """Categorize SWaT actuator columns.

        Actuators are the MV, P, and UV columns ending in ``.Status``.
        """
        actuator_columns: list[str] = []
        for column in columns:
            if column == "t_stamp":
                continue

            upper = column.upper()
            if not upper.endswith(".STATUS"):
                continue

            if any(upper.startswith(prefix) for prefix in self.actuator_prefixes):
                actuator_columns.append(column)

        logger.debug("Detected %s actuator column(s)", len(actuator_columns))
        return actuator_columns

    def _build_metadata(
        self,
        sheet_name: str,
        header_row: int,
        dataframe: pd.DataFrame,
        sensor_columns: list[str],
        actuator_columns: list[str],
        state_columns: list[str],
        alarm_columns: list[str],
        metadata_columns: list[str],
        workbook: pd.ExcelFile,
    ) -> dict[str, Any]:
        """Assemble metadata for the loaded workbook."""
        metadata = {
            "source_path": str(self.file_path),
            "sheet_name": sheet_name,
            "available_sheets": [str(item) for item in workbook.sheet_names],
            "header_row": int(header_row),
            "row_count": int(len(dataframe)),
            "column_count": int(len(dataframe.columns)),
            "columns": list(dataframe.columns),
            "dtypes": {column: str(dtype) for column, dtype in dataframe.dtypes.items()},
            "timestamp_column": "t_stamp",
            "label_column": None,
            "sensor_columns": list(sensor_columns),
            "actuator_columns": list(actuator_columns),
            "state_columns": list(state_columns),
            "alarm_columns": list(alarm_columns),
            "metadata_columns": list(metadata_columns),
        }
        logger.debug("Built metadata for worksheet %r", sheet_name)
        return metadata

    @staticmethod
    def _normalize_text(value: Any) -> str:
        """Normalize a cell value for header detection."""
        if pd.isna(value):
            return ""
        return str(value).strip().lower()


__all__ = ["SWaTData", "SWaTLoader"]