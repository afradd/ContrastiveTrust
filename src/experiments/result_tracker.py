"""Result tracking and reporting.

This module provides the ResultTracker class to systematically save experiment
results, including metrics, execution time, and hardware information, to both
JSON and CSV formats.
"""

import csv
import json
import logging
import platform
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import torch

from src.experiments.artifact_manager import ArtifactManager

logger = logging.getLogger(__name__)


class ResultTracker:
    """Tracks and saves experiment results and metadata."""

    def __init__(self, artifact_manager: ArtifactManager) -> None:
        """Initialize the ResultTracker.

        Parameters
        ----------
        artifact_manager : ArtifactManager
            The artifact manager controlling the experiment directories.
        """
        self.artifact_manager = artifact_manager
        self.metrics_history: List[Dict[str, Any]] = []
        self.hardware_info = self._gather_hardware_info()

    def _gather_hardware_info(self) -> Dict[str, Any]:
        """Gather hardware and system information.

        Returns
        -------
        dict
            A dictionary containing hardware information.
        """
        info = {
            "platform": platform.platform(),
            "python_version": platform.python_version(),
            "cuda_available": torch.cuda.is_available(),
        }
        if info["cuda_available"]:
            info["cuda_device_name"] = torch.cuda.get_device_name(0)
            info["cuda_device_count"] = torch.cuda.device_count()
        return info

    def log_metrics(self, epoch: int, metrics: Dict[str, float]) -> None:
        """Log metrics for a specific epoch.

        Parameters
        ----------
        epoch : int
            The current epoch.
        metrics : dict
            Dictionary of metrics (e.g., {"loss": 0.5, "accuracy": 0.9}).
        """
        entry = {"epoch": epoch, **metrics}
        self.metrics_history.append(entry)

    def save_results(self, summary_metrics: Optional[Dict[str, Any]] = None) -> None:
        """Save all tracked results and metrics to disk.

        Saves history to CSV and JSON, and summary/hardware info to JSON.

        Parameters
        ----------
        summary_metrics : dict, optional
            Final summary metrics to save (e.g., best test accuracy).
        """
        metrics_dir = self.artifact_manager.get_dir("metrics")
        
        # Save metrics history to CSV
        if self.metrics_history:
            csv_path = metrics_dir / "history.csv"
            keys = self.metrics_history[0].keys()
            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=keys)
                writer.writeheader()
                writer.writerows(self.metrics_history)
                
            # Save metrics history to JSON
            json_path = metrics_dir / "history.json"
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(self.metrics_history, f, indent=4)

        # Save summary and hardware info
        summary_data = {
            "hardware_info": self.hardware_info,
            "summary_metrics": summary_metrics or {},
        }
        summary_path = metrics_dir / "summary.json"
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary_data, f, indent=4)

        logger.info(f"Results saved to {metrics_dir}")
