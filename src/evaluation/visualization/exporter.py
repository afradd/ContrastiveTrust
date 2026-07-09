"""Data export utility for the visualization framework."""

import csv
import json
import logging
import os
from typing import Any, Dict, List, Optional, Union

logger = logging.getLogger(__name__)


class DataExporter:
    """Exports structured evaluation data to tables and reports."""

    @staticmethod
    def export_csv(
        filepath: str, 
        data: List[Dict[str, Any]], 
        fieldnames: Optional[List[str]] = None
    ) -> None:
        """Export a list of dictionaries to a CSV file.

        Parameters
        ----------
        filepath : str
            Path to save the CSV.
        data : list of dict
            List of rows to write.
        fieldnames : list of str, optional
            Specific column names. If None, derived from the first dictionary.
        """
        if not data:
            logger.warning("No data provided for CSV export.")
            return

        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        
        if fieldnames is None:
            # Gather all possible keys just in case
            keys = set()
            for row in data:
                keys.update(row.keys())
            fieldnames = sorted(list(keys))

        try:
            with open(filepath, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(data)
            logger.info(f"Exported data to {filepath}")
        except Exception as e:
            logger.error(f"Failed to export CSV to {filepath}: {e}")

    @staticmethod
    def export_json(filepath: str, data: Union[Dict[str, Any], List[Any]]) -> None:
        """Export data to a JSON file.

        Parameters
        ----------
        filepath : str
            Path to save the JSON.
        data : dict or list
            Data to serialize.
        """
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        
        try:
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=4)
            logger.info(f"Exported JSON report to {filepath}")
        except Exception as e:
            logger.error(f"Failed to export JSON to {filepath}: {e}")
