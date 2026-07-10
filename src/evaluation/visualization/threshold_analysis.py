"""Visualizer for threshold analysis and performance metrics."""

import logging
from typing import Any, Dict, List, Optional, Union
import numpy as np

from src.evaluation.visualization.base import BaseVisualizer
from src.evaluation.visualization.styles import style_manager
from src.evaluation.metrics import EvaluationMetrics

logger = logging.getLogger(__name__)


class ThresholdVisualizer(BaseVisualizer):
    """Plots metric variations across different threshold values."""

    def plot(
        self,
        y_true: Union[np.ndarray, list],
        y_score: Union[np.ndarray, list],
        num_thresholds: int = 100,
        metrics_to_plot: Optional[List[str]] = None,
        title: str = "Threshold Analysis",
        ax_params: Optional[dict] = None,
    ) -> "ThresholdVisualizer":
        """Generate the threshold analysis plot.

        Parameters
        ----------
        y_true : array-like
            Ground truth labels.
        y_score : array-like
            Predicted anomaly scores.
        num_thresholds : int
            Number of threshold points to evaluate between min and max score.
        metrics_to_plot : list of str, optional
            List of metrics to plot (e.g., ['f1_score', 'precision', 'recall']).
        title : str
            Plot title.
        ax_params : dict, optional
            Additional parameters to pass to ax.set().

        Returns
        -------
        ThresholdVisualizer
            Self reference.
        """
        self._setup_figure()
        
        y_true = np.asarray(y_true, dtype=int)
        y_score = np.asarray(y_score, dtype=float)
        
        if len(y_true) != len(y_score):
            raise ValueError("y_true and y_score must have the same length.")
            
        if metrics_to_plot is None:
            metrics_to_plot = ["f1_score", "precision", "recall"]
            
        min_score = float(np.min(y_score))
        max_score = float(np.max(y_score))
        
        thresholds = np.linspace(min_score, max_score, num_thresholds)
        
        eval_metrics = EvaluationMetrics()
        results: Dict[str, List[float]] = {m: [] for m in metrics_to_plot}
        
        for th in thresholds:
            y_pred = (y_score >= th).astype(int)
            # Evaluate at this threshold
            mets = eval_metrics.compute(y_true=y_true, y_score=y_score, y_pred=y_pred)
            for m in metrics_to_plot:
                results[m].append(mets.get(m, float('nan')))
                
        palette = style_manager.get_palette()
        if not palette:
            palette = ["blue", "red", "green", "orange", "purple"]

        if self.ax is not None:
            for idx, m in enumerate(metrics_to_plot):
                color = palette[idx % len(palette)]
                self.ax.plot(
                    thresholds, 
                    results[m], 
                    label=m.replace("_", " ").title(),
                    color=color,
                    linewidth=style_manager.get_line_width()
                )
                
            self.ax.set_title(title)
            self.ax.set_xlabel("Anomaly Threshold")
            self.ax.set_ylabel("Metric Value")
            self.ax.set_ylim(-0.05, 1.05)
            self.ax.legend()
            
            if ax_params:
                self.ax.set(**ax_params)
                
        return self
