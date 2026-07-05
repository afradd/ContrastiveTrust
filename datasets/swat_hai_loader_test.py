"""Run SWaT and HAI loaders side by side.

This script is a small debugging utility. It loads the SWaT Dec2019
workbook and the HAI test CSV pair, then prints a compact summary so
you can verify both loaders together.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from datasets.hai_loader import HAILoader
from datasets.swat_loader import SWaTLoader


logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Load and compare the SWaT and HAI datasets.",
    )
    parser.add_argument(
        "--swat",
        default=REPO_ROOT / "data" / "raw" / "SWaT" / "SWaT_Dec2019.xlsx",
        type=Path,
        help="Path to the SWaT Excel workbook.",
    )
    parser.add_argument(
        "--hai-data",
        default=REPO_ROOT / "data" / "raw" / "HAI" / "hai_test1.csv",
        type=Path,
        help="Path to the HAI feature CSV file.",
    )
    parser.add_argument(
        "--hai-label",
        default=REPO_ROOT / "data" / "raw" / "HAI" / "hai_test1_label.csv",
        type=Path,
        help="Path to the HAI label CSV file.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"),
        help="Logging level for the script.",
    )
    return parser.parse_args()


def configure_logging(level_name: str) -> None:
    """Configure console logging."""
    logging.basicConfig(
        level=getattr(logging, level_name.upper(), logging.INFO),
        format="%(levelname)s: %(message)s",
    )


def print_summary(title: str, shape: tuple[int, int], timestamp_column: str, label_column: str | None) -> None:
    """Print a compact dataset summary."""
    print(f"\n{title}")
    print("-" * len(title))
    print("Shape:", shape)
    print("Timestamp column:", timestamp_column)
    print("Label column:", label_column)


def main() -> int:
    """Load both datasets and print a short diagnostic summary."""
    args = parse_args()
    configure_logging(args.log_level)

    logger.info("Loading SWaT dataset from %s", args.swat)
    swat = SWaTLoader(file_path=args.swat).load()

    logger.info("Loading HAI dataset from %s and %s", args.hai_data, args.hai_label)
    hai = HAILoader(data_path=args.hai_data, label_path=args.hai_label).load()

    print_summary(
        "SWaT",
        swat.dataframe.shape,
        swat.timestamp_column,
        swat.label_column,
    )
    print("Sensors:", len(swat.sensor_columns))
    print("Actuators:", len(swat.actuator_columns))
    print("States:", len(swat.state_columns))
    print("Alarms:", len(swat.alarm_columns))
    print("Header row:", swat.metadata.get("header_row"))

    print_summary(
        "HAI",
        hai.dataframe.shape,
        hai.timestamp_column,
        hai.label_column,
    )
    print("Features:", len(hai.feature_columns))
    print("Rows:", hai.metadata.get("row_count"))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())