"""Visualizer for Confusion Matrix."""

from typing import Any, Optional, Union
import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

from src.evaluation.visualization.base import BaseVisualizer


class ConfusionMatrixVisualizer(BaseVisualizer):
    """Plots Confusion Matrix heatmaps."""

    def plot(
        self,
        y_true: Union[np.ndarray, list],
        y_pred: Union[np.ndarray, list],
        labels: Optional[list] = None,
        title: str = "Confusion Matrix",
        cmap: str = "Blues",
        normalize: bool = False,
        ax_params: Optional[dict] = None,
    ) -> "ConfusionMatrixVisualizer":
        """Generate the confusion matrix plot.

        Parameters
        ----------
        y_true : array-like
            Ground truth binary labels.
        y_pred : array-like
            Predicted binary labels.
        labels : list of str, optional
            Class names (default: ["Normal", "Anomaly"]).
        title : str
            Title of the plot.
        cmap : str
            Colormap name for the heatmap.
        normalize : bool
            Whether to normalize values over true labels (rows).
        ax_params : dict, optional
            Additional parameters to pass to ax.set().

        Returns
        -------
        ConfusionMatrixVisualizer
            Self reference.
        """
        self._setup_figure()
        
        if labels is None:
            labels = ["Normal", "Anomaly"]
            
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)

        if len(y_true) == 0:
            raise ValueError("y_true must not be empty.")

        cm = confusion_matrix(y_true, y_pred)
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        if self.ax is not None:
            im = self.ax.imshow(cm, interpolation='nearest', cmap=cmap)
            self.fig.colorbar(im, ax=self.ax)
            
            tick_marks = np.arange(len(labels))
            self.ax.set_xticks(tick_marks)
            self.ax.set_yticks(tick_marks)
            self.ax.set_xticklabels(labels)
            self.ax.set_yticklabels(labels)

            fmt = '.2f' if normalize else 'd'
            thresh = cm.max() / 2.
            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                    self.ax.text(j, i, format(cm[i, j], fmt),
                                 ha="center", va="center",
                                 color="white" if cm[i, j] > thresh else "black")

            self.ax.set_ylabel('True label')
            self.ax.set_xlabel('Predicted label')
            self.ax.set_title(title)
            
            if ax_params:
                self.ax.set(**ax_params)

        return self
