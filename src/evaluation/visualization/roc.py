"""Visualizer for ROC Curves."""

from typing import Any, Optional, Tuple, Union
import numpy as np
from sklearn.metrics import roc_curve, auc

from src.evaluation.visualization.base import BaseVisualizer
from src.evaluation.visualization.styles import style_manager


class ROCVisualizer(BaseVisualizer):
    """Plots Receiver Operating Characteristic (ROC) curves."""

    def plot(
        self,
        y_true: Union[np.ndarray, list],
        y_score: Union[np.ndarray, list],
        label: str = "Model",
        title: str = "ROC Curve",
        plot_random: bool = True,
        ax_params: Optional[dict] = None,
    ) -> "ROCVisualizer":
        """Generate the ROC curve plot.

        Parameters
        ----------
        y_true : array-like
            Ground truth binary labels.
        y_score : array-like
            Predicted anomaly scores.
        label : str
            Label for the curve in the legend.
        title : str
            Title of the plot.
        plot_random : bool
            Whether to plot the random chance diagonal line.
        ax_params : dict, optional
            Additional parameters to pass to ax.set().

        Returns
        -------
        ROCVisualizer
            Self reference.
        """
        self._setup_figure()
        
        y_true = np.asarray(y_true)
        y_score = np.asarray(y_score)

        if len(y_true) == 0 or len(np.unique(y_true)) < 2:
            raise ValueError("y_true must contain at least one positive and one negative sample.")

        fpr, tpr, _ = roc_curve(y_true, y_score)
        roc_auc = auc(fpr, tpr)

        palette = style_manager.get_palette()
        color = palette[0] if palette else "blue"

        if self.ax is not None:
            self.ax.plot(fpr, tpr, color=color, label=f"{label} (AUC = {roc_auc:.4f})")
            
            if plot_random:
                self.ax.plot([0, 1], [0, 1], color="gray", linestyle="--", alpha=0.7)

            self.ax.set_xlabel("False Positive Rate")
            self.ax.set_ylabel("True Positive Rate")
            self.ax.set_title(title)
            self.ax.set_xlim([0.0, 1.0])
            self.ax.set_ylim([0.0, 1.05])
            
            if ax_params:
                self.ax.set(**ax_params)
                
            self.ax.legend(loc="lower right")

        return self
