"""Visualizer for Benchmarking (Latency, Throughput, Memory)."""

from typing import Any, Dict, List, Optional
import numpy as np

from src.evaluation.visualization.base import BaseVisualizer
from src.evaluation.visualization.styles import style_manager


class BenchmarkVisualizer(BaseVisualizer):
    """Plots benchmarking results."""

    def plot(
        self,
        results: Dict[str, float],
        title: str = "Benchmark Results",
        ylabel: str = "Value",
        log_scale: bool = False,
        ax_params: Optional[dict] = None,
    ) -> "BenchmarkVisualizer":
        """Generate the benchmark bar plot.

        Parameters
        ----------
        results : dict
            Dictionary mapping metric names to their scalar values.
            e.g., {"Latency (ms)": 12.5, "Throughput (fps)": 80.0, "Mem (MB)": 500}
        title : str
            Title of the plot.
        ylabel : str
            Y-axis label.
        log_scale : bool
            Whether to use log scale on Y-axis.
        ax_params : dict, optional
            Additional parameters to pass to ax.set().

        Returns
        -------
        BenchmarkVisualizer
            Self reference.
        """
        self._setup_figure()
        
        if not results:
            raise ValueError("Results dictionary cannot be empty.")

        metrics = list(results.keys())
        values = list(results.values())
        
        x = np.arange(len(metrics))

        palette = style_manager.get_palette()
        color = palette[0] if palette else "blue"

        if self.ax is not None:
            bars = self.ax.bar(x, values, color=color, alpha=0.8)

            self.ax.set_title(title)
            self.ax.set_ylabel(ylabel)
            self.ax.set_xticks(x)
            self.ax.set_xticklabels(metrics, rotation=45, ha="right")
            
            if log_scale:
                self.ax.set_yscale("log")
                
            # Add text labels on bars
            for bar in bars:
                yval = bar.get_height()
                # Place text above bar, handle log scale appropriately if needed (rough approx)
                offset = yval * 0.05 if not log_scale else 0
                self.ax.text(
                    bar.get_x() + bar.get_width()/2, 
                    yval + offset, 
                    f"{yval:.2f}", 
                    ha='center', 
                    va='bottom'
                )
            
            if ax_params:
                self.ax.set(**ax_params)

        return self
