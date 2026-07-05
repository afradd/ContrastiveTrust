"""Inspect an Excel workbook and print structural diagnostics.

The script is intended for quick debugging of SWaT-style workbooks and
similar Excel files. It reports workbook metadata, previews the data,
and highlights rows that look mostly numeric or mostly textual.
"""

from __future__ import annotations

import argparse
import logging
import numbers
from pathlib import Path
from typing import Any, Iterable

import pandas as pd


LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
	"""Parse command-line arguments."""
	parser = argparse.ArgumentParser(
		description="Inspect an Excel workbook and print workbook diagnostics.",
	)
	parser.add_argument(
		"workbook",
		type=Path,
		help="Path to the Excel workbook to inspect.",
	)
	parser.add_argument(
		"--sheet",
		default=None,
		help="Worksheet name or zero-based index. Defaults to the first sheet.",
	)
	parser.add_argument(
		"--max-matches",
		type=int,
		default=10,
		help="Maximum number of mostly numeric or mostly string rows to print.",
	)
	parser.add_argument(
		"--numeric-threshold",
		type=float,
		default=0.6,
		help="Minimum numeric fraction for a row to count as mostly numeric.",
	)
	parser.add_argument(
		"--string-threshold",
		type=float,
		default=0.6,
		help="Minimum string fraction for a row to count as mostly string.",
	)
	parser.add_argument(
		"--preview-rows",
		type=int,
		default=20,
		help="Number of top rows to display.",
	)
	parser.add_argument(
		"--preview-columns",
		type=int,
		default=30,
		help="Number of leading column names to display.",
	)
	parser.add_argument(
		"--log-level",
		default="INFO",
		choices=("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"),
		help="Logging level for the script.",
	)
	return parser.parse_args()


def configure_logging(level_name: str) -> None:
	"""Configure console logging for the script."""
	logging.basicConfig(
		level=getattr(logging, level_name.upper(), logging.INFO),
		format="%(levelname)s: %(message)s",
	)


def validate_workbook_path(workbook_path: Path) -> None:
	"""Ensure that the workbook path exists and points to a file."""
	if not workbook_path.exists():
		raise FileNotFoundError(f"Workbook does not exist: {workbook_path}")

	if not workbook_path.is_file():
		raise FileNotFoundError(f"Workbook path is not a file: {workbook_path}")


def resolve_sheet_name(excel_file: pd.ExcelFile, sheet: str | None) -> str:
	"""Resolve a requested sheet name or index to an actual worksheet name."""
	sheet_names: list[str] = [str(sheet_name) for sheet_name in excel_file.sheet_names]
	if not sheet_names:
		raise ValueError("The workbook does not contain any worksheets.")

	if sheet is None:
		return sheet_names[0]

	try:
		sheet_index = int(sheet)
	except ValueError:
		if sheet not in sheet_names:
			raise ValueError(
				f"Worksheet {sheet!r} was not found. Available sheets: {sheet_names}"
			)
		return sheet

	if sheet_index < 0 or sheet_index >= len(sheet_names):
		raise ValueError(
			f"Worksheet index {sheet_index} is out of range. Available sheets: {sheet_names}"
		)

	return sheet_names[sheet_index]


def read_sheet(workbook_path: Path, sheet: str | None) -> tuple[pd.ExcelFile, str, pd.DataFrame]:
	"""Open the workbook and read one worksheet into a DataFrame."""
	excel_file = pd.ExcelFile(workbook_path)
	resolved_sheet = resolve_sheet_name(excel_file, sheet)
	dataframe = pd.read_excel(excel_file, sheet_name=resolved_sheet)
	return excel_file, resolved_sheet, dataframe


def preview_columns(columns: Iterable[object], limit: int) -> list[str]:
	"""Return the first N column names as strings."""
	return [str(column) for column in list(columns)[:limit]]


def is_text_value(value: object) -> bool:
	"""Return whether a value should be counted as text."""
	return isinstance(value, str) and value.strip() != ""


def is_numeric_value(value: object) -> bool:
	"""Return whether a value should be counted as numeric."""
	typed_value: Any = value
	return pd.notna(typed_value) and isinstance(typed_value, numbers.Number) and not isinstance(typed_value, bool)


def classify_row(row: pd.Series) -> tuple[float, float, int]:
	"""Compute numeric and textual fractions for a row.

	Returns:
		A tuple containing the numeric fraction, text fraction, and the
		number of non-missing values.
	"""
	values = [value for value in row.tolist() if pd.notna(value)]
	if not values:
		return 0.0, 0.0, 0

	numeric_count = sum(1 for value in values if is_numeric_value(value))
	text_count = sum(1 for value in values if is_text_value(value))
	total = len(values)
	return numeric_count / total, text_count / total, total


def find_rows_by_type(
	dataframe: pd.DataFrame,
	numeric_threshold: float,
	string_threshold: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
	"""Find rows that are mostly numeric or mostly text."""
	numeric_rows: list[Any] = []
	text_rows: list[Any] = []

	for index, row in dataframe.iterrows():
		numeric_fraction, text_fraction, non_null_count = classify_row(row)
		if non_null_count == 0:
			continue

		if numeric_fraction >= numeric_threshold:
			numeric_rows.append(index)

		if text_fraction >= string_threshold:
			text_rows.append(index)

	return dataframe.loc[numeric_rows], dataframe.loc[text_rows]


def print_section(title: str) -> None:
	"""Print a section heading."""
	print(f"\n{title}")
	print("-" * len(title))


def print_row_sample(label: str, rows: pd.DataFrame, max_matches: int) -> None:
	"""Print a capped preview of matching rows."""
	print(f"{label}: {len(rows)} row(s) matched")
	if rows.empty:
		return

	preview = rows.head(max_matches)
	with pd.option_context("display.max_columns", None, "display.width", 200):
		print(preview)
	if len(rows) > max_matches:
		print(f"... showing first {max_matches} match(es) only")


def main() -> int:
	"""Run the workbook inspection workflow."""
	args = parse_args()
	configure_logging(args.log_level)

	workbook_path = args.workbook.expanduser().resolve()
	LOGGER.info("Inspecting workbook: %s", workbook_path)
	validate_workbook_path(workbook_path)

	excel_file, sheet_name, dataframe = read_sheet(workbook_path, args.sheet)

	print_section("Workbook Sheets")
	print("Sheet names:", list(excel_file.sheet_names))
	print("Selected sheet:", sheet_name)

	print_section("Basic Shape")
	print("Rows:", len(dataframe))
	print("Columns:", len(dataframe.columns))

	print_section("First 20 Rows")
	with pd.option_context("display.max_columns", None, "display.width", 200):
		print(dataframe.head(args.preview_rows))

	print_section("First 30 Column Names")
	print(preview_columns(dataframe.columns, args.preview_columns))

	print_section("Data Types")
	print(dataframe.dtypes)

	numeric_rows, text_rows = find_rows_by_type(
		dataframe=dataframe,
		numeric_threshold=args.numeric_threshold,
		string_threshold=args.string_threshold,
	)

	print_section("Mostly Numeric Rows")
	print_row_sample("Mostly numeric", numeric_rows, args.max_matches)

	print_section("Mostly String Rows")
	print_row_sample("Mostly string", text_rows, args.max_matches)

	return 0


if __name__ == "__main__":
	raise SystemExit(main())
