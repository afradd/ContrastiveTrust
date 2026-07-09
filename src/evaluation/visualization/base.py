"""Base class for publication-quality visualizations."""

import logging
import os
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple, Union

import matplotlib.pyplot as plt

from src.evaluation.visualization.styles import style_manager

logger = logging.getLogger(__name__)


class BaseVisualizer(ABC):
    """Abstract base class for all publication-quality visualizers."""

    def __init__(self) -> None:
        """Initialize the base visualizer."""
        self.fig: Optional[plt.Figure] = None
        self.ax: Optional[Union[plt.Axes, Any]] = None

    def style(
        self,
        style_name: str = "ieee",
        figsize: Tuple[float, float] = (3.5, 2.5),
        dpi: int = 300,
        font_family: Optional[str] = None,
        font_size: Optional[int] = None,
        line_width: Optional[float] = None,
    ) -> "BaseVisualizer":
        """Apply a visualization style.

        Parameters
        ----------
        style_name : str
            The base style to apply ('ieee', 'science', etc.).
        figsize : tuple of float
            The dimensions of the figure (width, height) in inches.
        dpi : int
            The resolution of the output figure.
        font_family : str, optional
            Override the default font family.
        font_size : int, optional
            Override the default font size.
        line_width : float, optional
            Override the default line width.

        Returns
        -------
        BaseVisualizer
            Self reference for method chaining.
        """
        style_manager.apply_style(
            style=style_name,
            figsize=figsize,
            dpi=dpi,
            font_family=font_family,
            font_size=font_size,
            line_width=line_width,
        )
        return self

    def _setup_figure(
        self, 
        figsize: Optional[Tuple[float, float]] = None,
        create_axes: bool = True
    ) -> None:
        """Set up the figure and axes if they don't exist.

        Parameters
        ----------
        figsize : tuple of float, optional
            Figure size override.
        create_axes : bool
            Whether to create default axes.
        """
        if self.fig is None:
            if figsize is not None:
                self.fig = plt.figure(figsize=figsize)
            else:
                self.fig = plt.figure()
            
            if create_axes:
                self.ax = self.fig.add_subplot(111)

    @abstractmethod
    def plot(self, *args: Any, **kwargs: Any) -> "BaseVisualizer":
        """Generate the plot. Must be implemented by subclasses.

        Returns
        -------
        BaseVisualizer
            Self reference for method chaining.
        """
        pass

    def save(
        self,
        filepath: str,
        format: Optional[str] = None,
        dpi: Optional[int] = None,
        transparent: bool = False,
        bbox_inches: str = "tight",
        pad_inches: float = 0.05,
    ) -> None:
        """Save the generated figure to disk.

        Parameters
        ----------
        filepath : str
            The path where the figure should be saved.
        format : str, optional
            The image format (e.g., 'pdf', 'png', 'svg', 'eps').
            Inferred from the file extension if not provided.
        dpi : int, optional
            The resolution for raster formats (default relies on style settings).
        transparent : bool
            Whether to save with a transparent background.
        bbox_inches : str
            Bounding box configuration (default 'tight').
        pad_inches : float
            Padding around the figure if bbox_inches is 'tight'.
        """
        if self.fig is None:
            logger.warning("No figure exists to save. Call plot() first.")
            return

        # Ensure directory exists
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)

        if format is None:
            format = os.path.splitext(filepath)[1].lower().strip(".")
            if not format:
                format = "pdf"
                filepath += ".pdf"

        # Matplotlib doesn't support DPI for vector formats natively in the same way,
        # but passing it doesn't hurt. For High DPI requirements (300/600), this is key for PNG.
        save_kwargs: Dict[str, Any] = {
            "format": format,
            "transparent": transparent,
            "bbox_inches": bbox_inches,
            "pad_inches": pad_inches,
        }
        
        if dpi is not None:
            save_kwargs["dpi"] = dpi
        elif format == "png":
            save_kwargs["dpi"] = plt.rcParams.get("savefig.dpi", 300)

        try:
            self.fig.savefig(filepath, **save_kwargs)
            logger.info(f"Saved figure to {filepath}")
        except Exception as e:
            logger.error(f"Failed to save figure to {filepath}: {e}")

    def show(self) -> None:
        """Display the figure in an interactive window or notebook."""
        if self.fig is not None:
            plt.show()
        else:
            logger.warning("No figure exists to show. Call plot() first.")

    def close(self) -> None:
        """Close the figure to free up memory."""
        if self.fig is not None:
            plt.close(self.fig)
            self.fig = None
            self.ax = None
