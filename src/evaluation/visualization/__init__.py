"""Publication Visualization Framework.

Provides a unified API for generating publication-quality figures,
metrics exports, and benchmark visualizations for IEEE-style reporting.
"""

from src.evaluation.visualization.base import BaseVisualizer
from src.evaluation.visualization.styles import VisualizationStyle, style_manager
from src.evaluation.visualization.roc import ROCVisualizer
from src.evaluation.visualization.pr_curve import PRCurveVisualizer
from src.evaluation.visualization.confusion import ConfusionMatrixVisualizer
from src.evaluation.visualization.embeddings import EmbeddingVisualizer
from src.evaluation.visualization.training import TrainingVisualizer
from src.evaluation.visualization.distributions import DistributionVisualizer
from src.evaluation.visualization.ablation import AblationVisualizer
from src.evaluation.visualization.benchmarking import BenchmarkVisualizer
from src.evaluation.visualization.exporter import DataExporter

__all__ = [
    "BaseVisualizer",
    "VisualizationStyle",
    "style_manager",
    "ROCVisualizer",
    "PRCurveVisualizer",
    "ConfusionMatrixVisualizer",
    "EmbeddingVisualizer",
    "TrainingVisualizer",
    "DistributionVisualizer",
    "AblationVisualizer",
    "BenchmarkVisualizer",
    "DataExporter",
]
