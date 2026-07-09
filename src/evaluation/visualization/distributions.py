"""Visualizer for Distributions (Scores, Distances)."""

from typing import Any, Optional, Union
import numpy as np

from src.evaluation.visualization.base import BaseVisualizer
from src.evaluation.visualization.styles import style_manager


class DistributionVisualizer(BaseVisualizer):
    """Plots distributions of anomaly scores and distances."""

    def plot(
        self,
        scores: Union[np.ndarray, list],
        labels: Optional[Union[np.ndarray, list]] = None,
        threshold: Optional[float] = None,
        bins: int = 50,
        title: str = "Score Distribution",
        xlabel: str = "Anomaly Score",
        ylabel: str = "Density",
        class_names: Optional[list] = None,
        ax_params: Optional[dict] = None,
    ) -> "DistributionVisualizer":
        """Generate the distribution plot.

        Parameters
        ----------
        scores : array-like
            Anomaly scores or distances.
        labels : array-like, optional
            Binary labels for Normal vs Attack.
        threshold : float, optional
            Threshold value to plot as a vertical line.
        bins : int
            Number of histogram bins.
        title : str
            Title of the plot.
        xlabel : str
            X-axis label.
        ylabel : str
            Y-axis label.
        class_names : list of str, optional
            Names corresponding to labels (e.g., ["Normal", "Anomaly"]).
        ax_params : dict, optional
            Additional parameters to pass to ax.set().

        Returns
        -------
        DistributionVisualizer
            Self reference.
        """
        self._setup_figure()
        
        scores = np.asarray(scores)
        if len(scores) == 0:
            raise ValueError("Scores cannot be empty.")

        palette = style_manager.get_palette()
        if not palette:
            palette = ["blue", "red", "green", "orange"]

        if self.ax is not None:
            if labels is not None:
                labels = np.asarray(labels)
                unique_labels = np.unique(labels)
                
                if class_names is None:
                    class_names = [f"Class {int(l)}" for l in unique_labels]
                
                for idx, u_label in enumerate(unique_labels):
                    mask = (labels == u_label)
                    color = palette[idx % len(palette)]
                    name = class_names[idx] if idx < len(class_names) else f"Class {int(u_label)}"
                    
                    self.ax.hist(
                        scores[mask], 
                        bins=bins, 
                        density=True, 
                        alpha=0.6, 
                        color=color, 
                        label=name
                    )
            else:
                self.ax.hist(scores, bins=bins, density=True, alpha=0.6, color=palette[0], label="Scores")

            if threshold is not None:
                self.ax.axvline(x=threshold, color="black", linestyle="--", label=f"Threshold ({threshold:.2f})")

            self.ax.set_title(title)
            self.ax.set_xlabel(xlabel)
            self.ax.set_ylabel(ylabel)
            
            if ax_params:
                self.ax.set(**ax_params)
                
            self.ax.legend()

        return self
