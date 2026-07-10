"""Reporting module for saving evaluation summaries in multiple formats."""

import csv
import json
import logging
import os
from datetime import datetime
from typing import Any, Dict, Optional

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

logger = logging.getLogger(__name__)


class ReportGenerator:
    """Generates evaluation reports in JSON, CSV, and PDF formats."""

    def __init__(self, metrics: Dict[str, Any], metadata: Optional[Dict[str, Any]] = None) -> None:
        """Initialize the ReportGenerator.

        Args:
            metrics: Dictionary of metric names to values.
            metadata: Optional dictionary of additional context (e.g. model_name, dataset).
        """
        self.metrics = metrics
        self.metadata = metadata or {}
        
        # Make sure metrics are cleanly serializable
        self._clean_metrics = {}
        for k, v in self.metrics.items():
            if isinstance(v, float):
                # Handle NaN and Inf for JSON/CSV
                import math
                if math.isnan(v) or math.isinf(v):
                    self._clean_metrics[k] = str(v)
                else:
                    self._clean_metrics[k] = v
            else:
                self._clean_metrics[k] = v

    def to_json(self, filepath: str) -> None:
        """Export report to JSON.

        Args:
            filepath: Destination file path.
        """
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        
        payload = {
            "timestamp": datetime.now().isoformat(),
            "metadata": self.metadata,
            "metrics": self._clean_metrics
        }
        
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=4)
        logger.info(f"Saved JSON report to {filepath}")

    def to_csv(self, filepath: str) -> None:
        """Export report to CSV. 
        Metadata is prepended as comments or flattened depending on format.
        Here we flatten them into a single row.

        Args:
            filepath: Destination file path.
        """
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        
        row_dict = {}
        # Prefix metadata keys
        for k, v in self.metadata.items():
            row_dict[f"meta_{k}"] = v
            
        row_dict.update(self._clean_metrics)
        
        file_exists = os.path.isfile(filepath)
        
        with open(filepath, "a" if file_exists else "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(row_dict.keys()))
            if not file_exists:
                writer.writeheader()
            writer.writerow(row_dict)
            
        logger.info(f"Saved CSV report to {filepath}")

    def to_pdf(self, filepath: str, title: str = "Evaluation Report") -> None:
        """Export report to a text-based PDF using Matplotlib.

        Args:
            filepath: Destination file path.
            title: Title of the PDF report.
        """
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        
        with PdfPages(filepath) as pdf:
            fig, ax = plt.subplots(figsize=(8.5, 11))
            ax.axis('off')
            
            y_pos = 0.95
            line_height = 0.03
            
            # Title
            ax.text(0.5, y_pos, title, fontsize=16, fontweight='bold', ha='center', va='top')
            y_pos -= line_height * 2
            
            # Timestamp
            timestamp_str = f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
            ax.text(0.1, y_pos, timestamp_str, fontsize=10, ha='left', va='top')
            y_pos -= line_height * 2
            
            # Metadata
            if self.metadata:
                ax.text(0.1, y_pos, "Metadata", fontsize=14, fontweight='bold', ha='left', va='top')
                y_pos -= line_height * 1.5
                for k, v in self.metadata.items():
                    ax.text(0.15, y_pos, f"{k}: {v}", fontsize=11, ha='left', va='top')
                    y_pos -= line_height
                y_pos -= line_height
            
            # Metrics
            ax.text(0.1, y_pos, "Metrics", fontsize=14, fontweight='bold', ha='left', va='top')
            y_pos -= line_height * 1.5
            
            # Format metrics nicely
            for k, v in self._clean_metrics.items():
                if isinstance(v, (int, float)) and not isinstance(v, bool):
                    val_str = f"{v:.4f}" if isinstance(v, float) else f"{v}"
                else:
                    val_str = str(v)
                    
                ax.text(0.15, y_pos, f"{k}:", fontsize=11, fontweight='bold', ha='left', va='top')
                ax.text(0.4, y_pos, val_str, fontsize=11, ha='left', va='top')
                y_pos -= line_height
                
                # Check page overflow
                if y_pos < 0.1:
                    pdf.savefig(fig)
                    plt.close(fig)
                    fig, ax = plt.subplots(figsize=(8.5, 11))
                    ax.axis('off')
                    y_pos = 0.95
                    
            pdf.savefig(fig)
            plt.close(fig)
            
        logger.info(f"Saved PDF report to {filepath}")

    def export_all(self, base_filepath: str, title: str = "Evaluation Report") -> None:
        """Export to JSON, CSV, and PDF formats using the same base name.
        
        Args:
            base_filepath: File path without extension.
            title: Title for the PDF report.
        """
        self.to_json(f"{base_filepath}.json")
        self.to_csv(f"{base_filepath}.csv")
        self.to_pdf(f"{base_filepath}.pdf", title=title)
