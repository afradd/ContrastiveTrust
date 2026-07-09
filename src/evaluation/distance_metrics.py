"""Distance metrics for evaluating embeddings."""

from __future__ import annotations

import abc
import logging
from typing import Any, Dict, List, Optional, Type

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class DistanceMetric(abc.ABC):
    """Abstract base class for distance metrics."""

    @abc.abstractmethod
    def pairwise(self, embeddings1: torch.Tensor, embeddings2: torch.Tensor) -> torch.Tensor:
        """Compute pairwise distances between two sets of embeddings.

        Args:
            embeddings1: Tensor of shape (N, D).
            embeddings2: Tensor of shape (M, D).

        Returns:
            Tensor of shape (N, M) containing pairwise distances.
        """
        pass

    def compute(self, query: torch.Tensor, bank_embeddings: torch.Tensor) -> torch.Tensor:
        """Compute distances between a single query and a bank of embeddings.

        Args:
            query: Tensor of shape (D,) or (1, D).
            bank_embeddings: Tensor of shape (M, D).

        Returns:
            Tensor of shape (M,) containing distances.
        """
        if query.ndim == 1:
            query = query.unsqueeze(0)
        if query.shape[0] != 1:
            raise ValueError(f"Expected query to have batch size 1, got {query.shape[0]}")
            
        distances = self.pairwise(query, bank_embeddings)
        return distances.squeeze(0)

    def batch_compute(self, queries: torch.Tensor, bank_embeddings: torch.Tensor) -> torch.Tensor:
        """Compute distances between a batch of queries and a bank of embeddings.

        Args:
            queries: Tensor of shape (N, D).
            bank_embeddings: Tensor of shape (M, D).

        Returns:
            Tensor of shape (N, M) containing distances.
        """
        return self.pairwise(queries, bank_embeddings)


class CosineDistance(DistanceMetric):
    """Cosine distance metric (1 - cosine similarity)."""

    def __init__(self, eps: float = 1e-8):
        self.eps = eps

    def pairwise(self, embeddings1: torch.Tensor, embeddings2: torch.Tensor) -> torch.Tensor:
        e1_norm = F.normalize(embeddings1, p=2, dim=-1, eps=self.eps)
        e2_norm = F.normalize(embeddings2, p=2, dim=-1, eps=self.eps)
        
        sim = torch.mm(e1_norm, e2_norm.transpose(0, 1))
        # Distance is 1 - similarity. Clamp to [0, 2] to handle numerical issues.
        return torch.clamp(1.0 - sim, min=0.0, max=2.0)


class EuclideanDistance(DistanceMetric):
    """Euclidean (L2) distance metric."""

    def pairwise(self, embeddings1: torch.Tensor, embeddings2: torch.Tensor) -> torch.Tensor:
        # torch.cdist computes euclidean distance by default (p=2)
        return torch.cdist(embeddings1, embeddings2, p=2.0)


class SquaredEuclideanDistance(DistanceMetric):
    """Squared Euclidean distance metric."""

    def pairwise(self, embeddings1: torch.Tensor, embeddings2: torch.Tensor) -> torch.Tensor:
        # Expand norms
        e1_sq = (embeddings1 ** 2).sum(dim=1, keepdim=True)
        e2_sq = (embeddings2 ** 2).sum(dim=1).unsqueeze(0)
        
        # e1_sq (N, 1) + e2_sq (1, M) - 2 * e1 @ e2.T (N, M)
        dot_product = torch.mm(embeddings1, embeddings2.transpose(0, 1))
        
        dist_sq = e1_sq + e2_sq - 2.0 * dot_product
        # Clamp at zero to handle numerical instability where x**2 might be slightly negative
        return torch.clamp(dist_sq, min=0.0)


class ManhattanDistance(DistanceMetric):
    """Manhattan (L1) distance metric."""

    def pairwise(self, embeddings1: torch.Tensor, embeddings2: torch.Tensor) -> torch.Tensor:
        return torch.cdist(embeddings1, embeddings2, p=1.0)


class MahalanobisDistance(DistanceMetric):
    """Mahalanobis distance metric.
    
    Requires the inverse covariance matrix of the data distribution.
    """

    def __init__(self, cov_inv: Optional[torch.Tensor] = None):
        """Initialize the Mahalanobis distance metric.

        Args:
            cov_inv: Inverse covariance matrix of shape (D, D). If None, an identity
                matrix is used initially, effectively making it Squared Euclidean.
        """
        self.cov_inv = cov_inv

    def pairwise(self, embeddings1: torch.Tensor, embeddings2: torch.Tensor) -> torch.Tensor:
        N, D = embeddings1.shape
        M, D2 = embeddings2.shape
        
        if D != D2:
            raise ValueError(f"Feature dimensions must match. Got {D} and {D2}.")

        if self.cov_inv is None:
            logger.warning("cov_inv is None, using identity matrix (Squared Euclidean fallback).")
            cov_inv = torch.eye(D, dtype=embeddings1.dtype, device=embeddings1.device)
        else:
            cov_inv = self.cov_inv.to(dtype=embeddings1.dtype, device=embeddings1.device)

        if cov_inv.shape != (D, D):
            raise ValueError(f"Expected cov_inv of shape ({D}, {D}), got {cov_inv.shape}.")

        # x^T cov_inv x -> (N,)
        x_cov_x = (embeddings1 @ cov_inv * embeddings1).sum(dim=1, keepdim=True)
        # y^T cov_inv y -> (M,)
        y_cov_y = (embeddings2 @ cov_inv * embeddings2).sum(dim=1).unsqueeze(0)
        # 2 x^T cov_inv y -> (N, M)
        x_cov_y = 2.0 * torch.mm(embeddings1 @ cov_inv, embeddings2.transpose(0, 1))
        
        dist_sq = x_cov_x + y_cov_y - x_cov_y
        dist = torch.sqrt(torch.clamp(dist_sq, min=0.0))
        return dist


class DistanceMetricFactory:
    """Factory for creating distance metric instances."""

    _registry: Dict[str, Type[DistanceMetric]] = {
        "cosine": CosineDistance,
        "euclidean": EuclideanDistance,
        "l2": EuclideanDistance,
        "squared_euclidean": SquaredEuclideanDistance,
        "manhattan": ManhattanDistance,
        "l1": ManhattanDistance,
        "mahalanobis": MahalanobisDistance,
    }

    @classmethod
    def create(cls, name: str, **kwargs: Any) -> DistanceMetric:
        """Create a distance metric by name.

        Args:
            name: Name of the distance metric (e.g., 'cosine', 'euclidean').
            **kwargs: Additional arguments to pass to the metric constructor.

        Returns:
            An instance of a DistanceMetric subclass.

        Raises:
            ValueError: If the metric name is not registered.
        """
        name_lower = name.lower()
        if name_lower not in cls._registry:
            raise ValueError(
                f"Unknown distance metric '{name}'. "
                f"Available metrics: {', '.join(cls.available_metrics())}"
            )
        metric_cls = cls._registry[name_lower]
        return metric_cls(**kwargs)

    @classmethod
    def available_metrics(cls) -> List[str]:
        """List all available distance metrics.

        Returns:
            A list of registered metric names.
        """
        return list(cls._registry.keys())

    @classmethod
    def register(cls, name: str, metric_cls: Type[DistanceMetric]) -> None:
        """Register a custom distance metric.

        Args:
            name: Name of the metric.
            metric_cls: The DistanceMetric subclass.
        """
        cls._registry[name.lower()] = metric_cls
