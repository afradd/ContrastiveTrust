"""Anomaly Scorer for evaluating embeddings."""

from __future__ import annotations

import abc
import logging
from typing import Any, Dict, List, Optional, Type, Union

import torch

from .distance_metrics import DistanceMetric, DistanceMetricFactory
from .embedding_bank import EmbeddingBank

logger = logging.getLogger(__name__)


class ScoringStrategy(abc.ABC):
    """Abstract base class for scoring strategies."""

    @abc.abstractmethod
    def fit(self, reference_distances: torch.Tensor) -> None:
        """Fit the strategy to a distribution of reference distances.
        
        Args:
            reference_distances: Tensor of shape (N,) containing distances.
        """
        pass

    @abc.abstractmethod
    def score(self, distances: torch.Tensor) -> torch.Tensor:
        """Convert distances to anomaly scores.

        Args:
            distances: Tensor of shape (...,).

        Returns:
            Tensor of the same shape containing anomaly scores.
        """
        pass

    def get_state(self) -> Dict[str, Any]:
        """Get the state of the strategy for serialization."""
        return {}

    def load_state(self, state: Dict[str, Any]) -> None:
        """Load the state of the strategy."""
        pass


class RawDistanceStrategy(ScoringStrategy):
    """Uses raw distance as the anomaly score."""

    def fit(self, reference_distances: torch.Tensor) -> None:
        pass

    def score(self, distances: torch.Tensor) -> torch.Tensor:
        return distances.clone()


class MinMaxStrategy(ScoringStrategy):
    """Normalizes distances to [0, 1] based on observed min/max."""

    def __init__(self) -> None:
        self.min_val: Optional[torch.Tensor] = None
        self.max_val: Optional[torch.Tensor] = None

    def fit(self, reference_distances: torch.Tensor) -> None:
        if reference_distances.numel() == 0:
            raise ValueError("Reference distances cannot be empty.")
        self.min_val = reference_distances.min()
        self.max_val = reference_distances.max()

    def score(self, distances: torch.Tensor) -> torch.Tensor:
        if self.min_val is None or self.max_val is None:
            raise RuntimeError("MinMaxStrategy must be fitted before scoring.")
        
        diff = self.max_val - self.min_val
        if diff == 0:
            return torch.zeros_like(distances)
        
        return torch.clamp((distances - self.min_val) / diff, min=0.0, max=1.0)

    def get_state(self) -> Dict[str, Any]:
        return {
            "min_val": self.min_val.item() if self.min_val is not None else None,
            "max_val": self.max_val.item() if self.max_val is not None else None,
        }

    def load_state(self, state: Dict[str, Any]) -> None:
        self.min_val = torch.tensor(state["min_val"]) if state["min_val"] is not None else None
        self.max_val = torch.tensor(state["max_val"]) if state["max_val"] is not None else None


class RobustZScoreStrategy(ScoringStrategy):
    """Normalizes distances using Median and Median Absolute Deviation (MAD)."""

    def __init__(self) -> None:
        self.median: Optional[torch.Tensor] = None
        self.mad: Optional[torch.Tensor] = None

    def fit(self, reference_distances: torch.Tensor) -> None:
        if reference_distances.numel() == 0:
            raise ValueError("Reference distances cannot be empty.")
        self.median = torch.median(reference_distances)
        abs_deviation = torch.abs(reference_distances - self.median)
        self.mad = torch.median(abs_deviation)

    def score(self, distances: torch.Tensor) -> torch.Tensor:
        if self.median is None or self.mad is None:
            raise RuntimeError("RobustZScoreStrategy must be fitted before scoring.")
        
        if self.mad == 0:
            return distances - self.median
        
        # 0.6745 is the constant to make MAD consistent with standard deviation for normal distribution
        return (distances - self.median) / (self.mad / 0.6745)

    def get_state(self) -> Dict[str, Any]:
        return {
            "median": self.median.item() if self.median is not None else None,
            "mad": self.mad.item() if self.mad is not None else None,
        }

    def load_state(self, state: Dict[str, Any]) -> None:
        self.median = torch.tensor(state["median"]) if state["median"] is not None else None
        self.mad = torch.tensor(state["mad"]) if state["mad"] is not None else None


class PercentileStrategy(ScoringStrategy):
    """Scores based on the empirical cumulative distribution function (CDF)."""

    def __init__(self) -> None:
        self.reference_sorted: Optional[torch.Tensor] = None

    def fit(self, reference_distances: torch.Tensor) -> None:
        if reference_distances.numel() == 0:
            raise ValueError("Reference distances cannot be empty.")
        self.reference_sorted, _ = torch.sort(reference_distances)

    def score(self, distances: torch.Tensor) -> torch.Tensor:
        if self.reference_sorted is None:
            raise RuntimeError("PercentileStrategy must be fitted before scoring.")
        
        device = distances.device
        ref = self.reference_sorted.to(device)
        
        # searchsorted returns the index where elements should be inserted to maintain order
        indices = torch.searchsorted(ref, distances)
        
        percentiles = indices.to(distances.dtype) / ref.numel()
        return percentiles

    def get_state(self) -> Dict[str, Any]:
        return {
            "reference_sorted": self.reference_sorted.tolist() if self.reference_sorted is not None else None,
        }

    def load_state(self, state: Dict[str, Any]) -> None:
        self.reference_sorted = torch.tensor(state["reference_sorted"]) if state["reference_sorted"] is not None else None


class LogisticStrategy(ScoringStrategy):
    """Applies a logistic function to distances (maps to 0-1)."""

    def __init__(self, steepness: float = 1.0, midpoint: float = 0.0) -> None:
        self.steepness = steepness
        self.midpoint = midpoint

    def fit(self, reference_distances: torch.Tensor) -> None:
        if reference_distances.numel() > 0:
            self.midpoint = torch.median(reference_distances).item()

    def score(self, distances: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.steepness * (distances - self.midpoint))

    def get_state(self) -> Dict[str, Any]:
        return {
            "steepness": self.steepness,
            "midpoint": self.midpoint,
        }

    def load_state(self, state: Dict[str, Any]) -> None:
        self.steepness = state["steepness"]
        self.midpoint = state["midpoint"]


class AnomalyScorer:
    """Computes continuous anomaly scores for query embeddings."""

    _strategy_registry: Dict[str, Type[ScoringStrategy]] = {
        "raw": RawDistanceStrategy,
        "minmax": MinMaxStrategy,
        "robust_z": RobustZScoreStrategy,
        "percentile": PercentileStrategy,
        "logistic": LogisticStrategy,
    }

    def __init__(
        self,
        bank: EmbeddingBank,
        metric: Union[str, DistanceMetric],
        strategy: Union[str, ScoringStrategy],
        k: int = 1,
        **strategy_kwargs: Any
    ) -> None:
        """Initialize the anomaly scorer.

        Args:
            bank: The EmbeddingBank containing normal reference embeddings.
            metric: DistanceMetric instance or name (e.g., 'cosine').
            strategy: ScoringStrategy instance or name (e.g., 'robust_z').
            k: Number of nearest neighbors to consider (default: 1).
            **strategy_kwargs: Additional arguments for strategy instantiation.
        """
        self.bank = bank
        self.k = k

        if isinstance(metric, str):
            self.metric = DistanceMetricFactory.create(metric)
        else:
            self.metric = metric

        if isinstance(strategy, str):
            self.strategy = self._create_strategy(strategy, **strategy_kwargs)
        else:
            self.strategy = strategy

    @classmethod
    def _create_strategy(cls, name: str, **kwargs: Any) -> ScoringStrategy:
        name_lower = name.lower()
        if name_lower not in cls._strategy_registry:
            raise ValueError(
                f"Unknown scoring strategy '{name}'. "
                f"Available: {', '.join(cls.available_strategies())}"
            )
        return cls._strategy_registry[name_lower](**kwargs)

    @classmethod
    def available_strategies(cls) -> List[str]:
        return list(cls._strategy_registry.keys())

    @classmethod
    def register_strategy(cls, name: str, strategy_cls: Type[ScoringStrategy]) -> None:
        cls._strategy_registry[name.lower()] = strategy_cls

    def _get_k_distances(self, distances: torch.Tensor) -> torch.Tensor:
        """Reduces distances to k-NN distances."""
        if self.k == 1:
            return torch.min(distances, dim=-1).values
        else:
            if distances.ndim == 1:
                distances = distances.unsqueeze(0)
            
            k = min(self.k, distances.shape[-1])
            topk_dist, _ = torch.topk(distances, k, dim=-1, largest=False)
            mean_topk = topk_dist.mean(dim=-1)
            return mean_topk.squeeze(0) if mean_topk.numel() == 1 else mean_topk

    def fit(self, val_queries: Optional[torch.Tensor] = None) -> None:
        """Fit the scoring strategy using reference distances.

        Args:
            val_queries: Optional validation queries. If None, computes pairwise
                distances within the embedding bank itself.
        """
        if self.bank.embeddings is None:
            raise RuntimeError("Cannot fit scorer: EmbeddingBank is empty.")

        if val_queries is not None:
            distances = self.metric.batch_compute(val_queries, self.bank.embeddings)
            ref_distances = self._get_k_distances(distances)
        else:
            distances = self.metric.pairwise(self.bank.embeddings, self.bank.embeddings)
            distances.fill_diagonal_(float('inf'))
            
            if self.k == 1:
                ref_distances = torch.min(distances, dim=-1).values
            else:
                k = min(self.k, distances.shape[-1])
                topk_dist, _ = torch.topk(distances, k, dim=-1, largest=False)
                ref_distances = topk_dist.mean(dim=-1)

        self.strategy.fit(ref_distances)
        logger.info(f"Fitted anomaly scorer with strategy '{self.strategy.__class__.__name__}'.")

    def score(self, query: torch.Tensor) -> torch.Tensor:
        """Compute anomaly score for a single query."""
        if self.bank.embeddings is None:
            raise RuntimeError("EmbeddingBank is empty.")
            
        distances = self.metric.compute(query, self.bank.embeddings)
        reduced_dist = self._get_k_distances(distances)
        return self.strategy.score(reduced_dist)

    def batch_score(self, queries: torch.Tensor) -> torch.Tensor:
        """Compute anomaly scores for a batch of queries."""
        if self.bank.embeddings is None:
            raise RuntimeError("EmbeddingBank is empty.")
            
        distances = self.metric.batch_compute(queries, self.bank.embeddings)
        reduced_dist = self._get_k_distances(distances)
        return self.strategy.score(reduced_dist)

    def summary(self) -> Dict[str, Any]:
        """Returns a summary of the scorer's state."""
        return {
            "metric": self.metric.__class__.__name__,
            "strategy": self.strategy.__class__.__name__,
            "k": self.k,
            "strategy_state": self.strategy.get_state(),
        }
