"""Automatic threshold estimation for zero-shot anomaly detection."""

from __future__ import annotations

import abc
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Type, Union

import torch

from .anomaly_scorer import AnomalyScorer

logger = logging.getLogger(__name__)


class ThresholdStrategy(abc.ABC):
    """Abstract base class for threshold estimation strategies."""

    @abc.abstractmethod
    def fit(self, scores: torch.Tensor) -> float:
        """Estimate the threshold given a distribution of normal scores.

        Args:
            scores: Tensor of shape (N,) containing anomaly scores.

        Returns:
            The estimated threshold.
        """
        pass

    def get_state(self) -> Dict[str, Any]:
        """Get the state of the strategy for serialization."""
        return {}

    def load_state(self, state: Dict[str, Any]) -> None:
        """Load the state of the strategy."""
        pass


class ManualThreshold(ThresholdStrategy):
    """Simply returns a predefined threshold."""

    def __init__(self, threshold: float = 0.5):
        self.threshold = float(threshold)

    def fit(self, scores: torch.Tensor) -> float:
        return self.threshold

    def get_state(self) -> Dict[str, Any]:
        return {"threshold": self.threshold}

    def load_state(self, state: Dict[str, Any]) -> None:
        self.threshold = float(state.get("threshold", 0.5))


class PercentileThreshold(ThresholdStrategy):
    """Sets threshold at a specific percentile of the scores."""

    def __init__(self, percentile: float = 95.0):
        if not (0.0 <= percentile <= 100.0):
            raise ValueError("Percentile must be between 0 and 100.")
        self.percentile = float(percentile)

    def fit(self, scores: torch.Tensor) -> float:
        if scores.numel() == 0:
            raise ValueError("Scores tensor cannot be empty.")
        # quantile expects values in [0, 1]
        q = self.percentile / 100.0
        # Compute quantile. Ensure input is float for quantile
        threshold = torch.quantile(scores.to(torch.float32), q).item()
        return float(threshold)

    def get_state(self) -> Dict[str, Any]:
        return {"percentile": self.percentile}

    def load_state(self, state: Dict[str, Any]) -> None:
        self.percentile = float(state.get("percentile", 95.0))


class MeanStdThreshold(ThresholdStrategy):
    """Sets threshold at Mean + k * Std."""

    def __init__(self, k: float = 3.0):
        self.k = float(k)

    def fit(self, scores: torch.Tensor) -> float:
        if scores.numel() == 0:
            raise ValueError("Scores tensor cannot be empty.")
        mean = scores.mean().item()
        # std needs at least 2 elements, else 0
        std = scores.std(unbiased=True).item() if scores.numel() > 1 else 0.0
        return float(mean + self.k * std)

    def get_state(self) -> Dict[str, Any]:
        return {"k": self.k}

    def load_state(self, state: Dict[str, Any]) -> None:
        self.k = float(state.get("k", 3.0))


class MedianMADThreshold(ThresholdStrategy):
    """Sets threshold at Median + k * MAD (robust to outliers)."""

    def __init__(self, k: float = 3.0):
        self.k = float(k)

    def fit(self, scores: torch.Tensor) -> float:
        if scores.numel() == 0:
            raise ValueError("Scores tensor cannot be empty.")
        median = torch.median(scores).item()
        abs_dev = torch.abs(scores - median)
        mad = torch.median(abs_dev).item()
        
        # 1.4826 is the scaling factor to approximate std deviation for normal distributions
        # (1 / 0.6745)
        approx_std = mad * 1.4826
        return float(median + self.k * approx_std)

    def get_state(self) -> Dict[str, Any]:
        return {"k": self.k}

    def load_state(self, state: Dict[str, Any]) -> None:
        self.k = float(state.get("k", 3.0))


class ThresholdEstimator:
    """Estimates and manages thresholds for anomaly detection."""

    _strategy_registry: Dict[str, Type[ThresholdStrategy]] = {
        "manual": ManualThreshold,
        "percentile": PercentileThreshold,
        "mean_std": MeanStdThreshold,
        "median_mad": MedianMADThreshold,
    }

    def __init__(
        self,
        strategy: Union[str, ThresholdStrategy],
        **strategy_kwargs: Any
    ):
        """Initialize the threshold estimator.

        Args:
            strategy: Strategy name (e.g., 'percentile') or ThresholdStrategy instance.
            **strategy_kwargs: Arguments passed to the strategy constructor.
        """
        if isinstance(strategy, str):
            self.strategy_name = strategy.lower()
            self.strategy = self._create_strategy(self.strategy_name, **strategy_kwargs)
        else:
            self.strategy_name = "custom"
            self.strategy = strategy
            
        self.threshold: Optional[float] = None

    @classmethod
    def _create_strategy(cls, name: str, **kwargs: Any) -> ThresholdStrategy:
        if name not in cls._strategy_registry:
            raise ValueError(
                f"Unknown threshold strategy '{name}'. "
                f"Available: {', '.join(cls.available_strategies())}"
            )
        return cls._strategy_registry[name](**kwargs)

    @classmethod
    def available_strategies(cls) -> List[str]:
        """List all available threshold strategies."""
        return list(cls._strategy_registry.keys())

    @classmethod
    def register_strategy(cls, name: str, strategy_cls: Type[ThresholdStrategy]) -> None:
        """Register a custom threshold strategy."""
        cls._strategy_registry[name.lower()] = strategy_cls

    def fit(
        self,
        scores: Optional[torch.Tensor] = None,
        scorer: Optional[AnomalyScorer] = None,
        val_queries: Optional[torch.Tensor] = None,
    ) -> float:
        """Estimate the threshold using the configured strategy.

        Args:
            scores: Pre-computed anomaly scores.
            scorer: AnomalyScorer to compute scores if `scores` is not provided.
            val_queries: Queries to score using `scorer`. If None, and `scorer` is
                provided, it may use the bank's internal references (if applicable).

        Returns:
            The estimated threshold.
            
        Raises:
            ValueError: If neither `scores` nor (`scorer` + `val_queries`) are provided.
        """
        if scores is not None:
            self.threshold = self.strategy.fit(scores)
        elif scorer is not None and val_queries is not None:
            scores = scorer.batch_score(val_queries)
            self.threshold = self.strategy.fit(scores)
        else:
            raise ValueError("Must provide either 'scores' or both 'scorer' and 'val_queries'.")
            
        logger.info(f"Threshold estimated at {self.threshold:.4f} using {self.strategy.__class__.__name__}")
        return self.threshold

    def predict_threshold(self) -> float:
        """Returns the estimated threshold.

        Raises:
            RuntimeError: If `fit` has not been called yet.
        """
        if self.threshold is None:
            raise RuntimeError("ThresholdEstimator must be fitted before predicting.")
        return self.threshold

    def get_state(self) -> Dict[str, Any]:
        """Get estimator state for serialization."""
        return {
            "strategy_name": self.strategy_name,
            "strategy_state": self.strategy.get_state(),
            "threshold": self.threshold,
        }

    def load_state(self, state: Dict[str, Any]) -> None:
        """Load estimator state."""
        self.strategy_name = state["strategy_name"]
        
        if self.strategy_name in self._strategy_registry:
            self.strategy = self._create_strategy(self.strategy_name)
            
        self.strategy.load_state(state["strategy_state"])
        self.threshold = state["threshold"]

    def save(self, path: Union[str, Path]) -> None:
        """Save the estimator state to a JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.get_state(), f, indent=2)

    def load(self, path: Union[str, Path]) -> None:
        """Load the estimator state from a JSON file."""
        with open(path, "r") as f:
            state = json.load(f)
        self.load_state(state)

    def summary(self) -> Dict[str, Any]:
        """Returns a summary of the threshold estimator's state."""
        return {
            "strategy": self.strategy.__class__.__name__,
            "strategy_kwargs": self.strategy.get_state(),
            "threshold": self.threshold,
            "fitted": self.threshold is not None,
        }
