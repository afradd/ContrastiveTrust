"""Visualizer for Embeddings."""

import logging
from typing import Any, Optional, Union
import numpy as np

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from src.evaluation.visualization.base import BaseVisualizer
from src.evaluation.visualization.styles import style_manager

logger = logging.getLogger(__name__)


class EmbeddingVisualizer(BaseVisualizer):
    """Plots 2D projections of embeddings (t-SNE, UMAP, PCA)."""

    def _get_projector(self, method: str, n_components: int = 2) -> Any:
        method = method.lower()
        if method == "pca":
            return PCA(n_components=n_components)
        elif method == "tsne":
            return TSNE(n_components=n_components, init="pca", learning_rate="auto")
        elif method == "umap":
            try:
                import umap
                return umap.UMAP(n_components=n_components)
            except ImportError:
                logger.warning("umap-learn not installed. Falling back to PCA.")
                return PCA(n_components=n_components)
        else:
            raise ValueError(f"Unknown projection method: {method}")

    def plot(
        self,
        embeddings: Union[np.ndarray, list],
        labels: Optional[Union[np.ndarray, list]] = None,
        method: str = "tsne",
        title: str = "Embedding Projection",
        class_names: Optional[list] = None,
        ax_params: Optional[dict] = None,
    ) -> "EmbeddingVisualizer":
        """Generate the embedding projection plot.

        Parameters
        ----------
        embeddings : array-like
            High-dimensional embeddings.
        labels : array-like, optional
            Binary labels for Normal vs Attack.
        method : str
            Projection method ('pca', 'tsne', 'umap').
        title : str
            Title of the plot.
        class_names : list of str, optional
            Names corresponding to labels (e.g., ["Normal", "Anomaly"]).
        ax_params : dict, optional
            Additional parameters to pass to ax.set().

        Returns
        -------
        EmbeddingVisualizer
            Self reference.
        """
        self._setup_figure()
        
        embeddings = np.asarray(embeddings)
        if len(embeddings) == 0:
            raise ValueError("Embeddings cannot be empty.")
            
        projector = self._get_projector(method)
        projections = projector.fit_transform(embeddings)

        palette = style_manager.get_palette()
        if not palette:
            palette = ["blue", "red", "green", "orange", "purple"]

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
                    
                    self.ax.scatter(
                        projections[mask, 0], 
                        projections[mask, 1], 
                        color=color, 
                        label=name, 
                        alpha=0.6,
                        s=15
                    )
                self.ax.legend()
            else:
                self.ax.scatter(projections[:, 0], projections[:, 1], color=palette[0], alpha=0.6, s=15)

            self.ax.set_title(title)
            self.ax.set_xlabel(f"{method.upper()} Component 1")
            self.ax.set_ylabel(f"{method.upper()} Component 2")
            
            if ax_params:
                self.ax.set(**ax_params)

        return self
