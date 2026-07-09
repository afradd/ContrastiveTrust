"""Visualizer for Ablation Studies."""

from typing import Any, Dict, List, Optional
import numpy as np

from src.evaluation.visualization.base import BaseVisualizer
from src.evaluation.visualization.styles import style_manager


class AblationVisualizer(BaseVisualizer):
    """Plots ablation study results."""

    def plot(
        self,
        results: Dict[str, Dict[str, float]],
        metric_keys: List[str],
        title: str = "Ablation Performance Comparison",
        ylabel: str = "Score",
        ax_params: Optional[dict] = None,
    ) -> "AblationVisualizer":
        """Generate the ablation bar plot.

        Parameters
        ----------
        results : dict
            Dictionary mapping configuration names to a dictionary of metrics.
            e.g. {"Base": {"f1": 0.9, "auc": 0.95}, "No-Physics": {"f1": 0.8, "auc": 0.85}}
        metric_keys : list of str
            Which metrics to plot (e.g., ["f1", "auc"]).
        title : str
            Title of the plot.
        ylabel : str
            Y-axis label.
        ax_params : dict, optional
            Additional parameters to pass to ax.set().

        Returns
        -------
        AblationVisualizer
            Self reference.
        """
        self._setup_figure()
        
        if not results:
            raise ValueError("Results dictionary cannot be empty.")
        if not metric_keys:
            raise ValueError("metric_keys cannot be empty.")

        configs = list(results.keys())
        n_configs = len(configs)
        n_metrics = len(metric_keys)
        
        width = 0.8 / n_metrics
        x = np.arange(n_configs)

        palette = style_manager.get_palette()
        if not palette:
            palette = ["#0072B2", "#D55E00", "#009E73", "#CC79A7"]

        if self.ax is not None:
            for i, metric in enumerate(metric_keys):
                values = [results[config].get(metric, 0.0) for config in configs]
                offset = (i - n_metrics/2 + 0.5) * width
                
                color = palette[i % len(palette)]
                self.ax.bar(x + offset, values, width, label=metric, color=color)

            self.ax.set_title(title)
            self.ax.set_ylabel(ylabel)
            self.ax.set_xticks(x)
            self.ax.set_xticklabels(configs, rotation=45, ha="right")
            
            if ax_params:
                self.ax.set(**ax_params)
                
            self.ax.legend()

        return self
