"""Styling utilities for the publication visualization framework."""

import logging
from typing import Any, Dict, Optional, Tuple

import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)

# Fallback IEEE-like style for when scienceplots is not available
FALLBACK_STYLE = {
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif", "serif"],
    "font.size": 9,
    "axes.labelsize": 9,
    "axes.titlesize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "legend.title_fontsize": 8,
    "axes.linewidth": 0.5,
    "grid.linewidth": 0.5,
    "lines.linewidth": 1.0,
    "lines.markersize": 3,
    "patch.linewidth": 0.5,
    "xtick.major.width": 0.5,
    "ytick.major.width": 0.5,
    "xtick.minor.width": 0.5,
    "ytick.minor.width": 0.5,
    "xtick.major.size": 3,
    "ytick.major.size": 3,
    "xtick.minor.size": 1.5,
    "ytick.minor.size": 1.5,
    "axes.grid": True,
    "grid.alpha": 0.5,
    "grid.linestyle": "--",
    "legend.frameon": True,
    "legend.edgecolor": "inherit",
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.format": "pdf",
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
    # Standard color palette (tableau 10)
    "axes.prop_cycle": plt.cycler('color', [
        '#0072B2', '#D55E00', '#009E73', '#CC79A7',
        '#F0E442', '#56B4E9', '#E69F00', '#000000'
    ])
}


class VisualizationStyle:
    """Manages styling for publication-quality visualizations."""

    def __init__(self) -> None:
        """Initialize style manager."""
        self._current_context: Optional[Any] = None

    @staticmethod
    def _is_scienceplots_available() -> bool:
        """Check if scienceplots is installed."""
        try:
            import scienceplots  # noqa: F401
            return True
        except ImportError:
            return False

    def apply_style(
        self,
        style: str = "ieee",
        figsize: Tuple[float, float] = (3.5, 2.5),  # standard IEEE single column width
        dpi: int = 300,
        font_family: Optional[str] = None,
        font_size: Optional[int] = None,
        line_width: Optional[float] = None,
    ) -> None:
        """Apply a global visualization style.

        Parameters
        ----------
        style : str
            The name of the style to apply ('ieee', 'science', or 'default').
        figsize : tuple of float
            The figure size in inches (width, height).
        dpi : int
            The resolution of the figure.
        font_family : str, optional
            Override the default font family.
        font_size : int, optional
            Override the default font size.
        line_width : float, optional
            Override the default line width.
        """
        # Base setup
        if self._is_scienceplots_available():
            if style == "ieee":
                plt.style.use(['science', 'ieee'])
            elif style == "science":
                plt.style.use(['science'])
            else:
                plt.style.use('default')
        else:
            if style in ["ieee", "science"]:
                logger.info(
                    "scienceplots not found. Using fallback IEEE-compatible Matplotlib style."
                )
                plt.style.use("default")
                plt.rcParams.update(FALLBACK_STYLE)
            else:
                plt.style.use("default")

        # Apply specific overrides
        plt.rcParams["figure.figsize"] = figsize
        plt.rcParams["figure.dpi"] = dpi
        plt.rcParams["savefig.dpi"] = dpi

        if font_family is not None:
            plt.rcParams["font.family"] = font_family
        if font_size is not None:
            plt.rcParams["font.size"] = font_size
            plt.rcParams["axes.labelsize"] = font_size
            plt.rcParams["axes.titlesize"] = font_size
            plt.rcParams["legend.fontsize"] = max(font_size - 1, 6)
            plt.rcParams["xtick.labelsize"] = max(font_size - 1, 6)
            plt.rcParams["ytick.labelsize"] = max(font_size - 1, 6)
        if line_width is not None:
            plt.rcParams["lines.linewidth"] = line_width
            plt.rcParams["axes.linewidth"] = line_width / 2

    def context(self, **kwargs: Any) -> Any:
        """Get a matplotlib context manager for temporary styling.

        Parameters
        ----------
        kwargs : Any
            Any valid matplotlib rcParams key-value pairs.

        Returns
        -------
        context_manager
            A context manager for styling.
        """
        return plt.rc_context(kwargs)

    @staticmethod
    def get_palette() -> list[str]:
        """Get the current color palette."""
        return [c["color"] for c in plt.rcParams["axes.prop_cycle"]]


style_manager = VisualizationStyle()
