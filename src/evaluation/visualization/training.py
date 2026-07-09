"""Visualizer for Training Metrics."""

from typing import Any, Optional, Union
import numpy as np

from src.evaluation.visualization.base import BaseVisualizer
from src.evaluation.visualization.styles import style_manager


class TrainingVisualizer(BaseVisualizer):
    """Plots training metrics such as loss and learning rate."""

    def plot(
        self,
        metrics: dict[str, Union[np.ndarray, list]],
        epochs: Optional[Union[np.ndarray, list]] = None,
        title: str = "Training Metrics",
        xlabel: str = "Epoch",
        ylabel: str = "Value",
        log_scale: bool = False,
        ax_params: Optional[dict] = None,
    ) -> "TrainingVisualizer":
        """Generate the training metrics plot.

        Parameters
        ----------
        metrics : dict
            Dictionary mapping metric names to arrays of values over epochs.
        epochs : array-like, optional
            X-axis values. If None, uses range(len(metrics)).
        title : str
            Title of the plot.
        xlabel : str
            X-axis label.
        ylabel : str
            Y-axis label.
        log_scale : bool
            Whether to use log scale for the Y-axis.
        ax_params : dict, optional
            Additional parameters to pass to ax.set().

        Returns
        -------
        TrainingVisualizer
            Self reference.
        """
        self._setup_figure()
        
        if not metrics:
            raise ValueError("Metrics dictionary cannot be empty.")
            
        first_key = list(metrics.keys())[0]
        n_points = len(metrics[first_key])
        
        if epochs is None:
            epochs = np.arange(1, n_points + 1)
        else:
            epochs = np.asarray(epochs)

        palette = style_manager.get_palette()
        if not palette:
            palette = ["blue", "orange", "green", "red", "purple"]

        if self.ax is not None:
            for i, (name, values) in enumerate(metrics.items()):
                color = palette[i % len(palette)]
                self.ax.plot(epochs, values, color=color, label=name)

            self.ax.set_title(title)
            self.ax.set_xlabel(xlabel)
            self.ax.set_ylabel(ylabel)
            
            if log_scale:
                self.ax.set_yscale("log")
                
            if ax_params:
                self.ax.set(**ax_params)
                
            self.ax.legend()

        return self
