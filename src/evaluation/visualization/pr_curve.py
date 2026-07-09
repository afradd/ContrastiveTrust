"""Visualizer for Precision-Recall Curves."""

from typing import Any, Optional, Tuple, Union
import numpy as np
from sklearn.metrics import precision_recall_curve, auc

from src.evaluation.visualization.base import BaseVisualizer
from src.evaluation.visualization.styles import style_manager


class PRCurveVisualizer(BaseVisualizer):
    """Plots Precision-Recall curves."""

    def plot(
        self,
        y_true: Union[np.ndarray, list],
        y_score: Union[np.ndarray, list],
        label: str = "Model",
        title: str = "Precision-Recall Curve",
        plot_baseline: bool = True,
        ax_params: Optional[dict] = None,
    ) -> "PRCurveVisualizer":
        """Generate the PR curve plot.

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
        plot_baseline : bool
            Whether to plot the random chance baseline (positive class prevalence).
        ax_params : dict, optional
            Additional parameters to pass to ax.set().

        Returns
        -------
        PRCurveVisualizer
            Self reference.
        """
        self._setup_figure()
        
        y_true = np.asarray(y_true)
        y_score = np.asarray(y_score)

        if len(y_true) == 0 or len(np.unique(y_true)) < 2:
            raise ValueError("y_true must contain at least one positive and one negative sample.")

        precision, recall, _ = precision_recall_curve(y_true, y_score)
        pr_auc = auc(recall, precision)

        palette = style_manager.get_palette()
        color = palette[0] if palette else "blue"

        if self.ax is not None:
            self.ax.plot(recall, precision, color=color, label=f"{label} (AUC = {pr_auc:.4f})")
            
            if plot_baseline:
                baseline = np.sum(y_true) / len(y_true)
                self.ax.plot([0, 1], [baseline, baseline], color="gray", linestyle="--", alpha=0.7, label=f"Baseline ({baseline:.2f})")

            self.ax.set_xlabel("Recall")
            self.ax.set_ylabel("Precision")
            self.ax.set_title(title)
            self.ax.set_xlim([0.0, 1.0])
            self.ax.set_ylim([0.0, 1.05])
            
            if ax_params:
                self.ax.set(**ax_params)
                
            self.ax.legend(loc="lower left")

        return self
