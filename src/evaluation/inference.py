"""High-level inference API for ContrastiveTrust."""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F

from .post_processing import PostProcessor
from .zero_shot_detector import ZeroShotDetector

logger = logging.getLogger(__name__)


@dataclass
class AnomalyPrediction:
    """Standardized prediction object for a single instance."""
    is_anomaly: bool
    score: float
    threshold: float
    distance: float
    confidence: float
    metadata: Dict[str, Any]


@dataclass
class BatchAnomalyPrediction:
    """Standardized prediction object for a batch of instances."""
    is_anomaly: torch.Tensor
    scores: torch.Tensor
    threshold: float
    distances: torch.Tensor
    confidences: torch.Tensor
    metadata: List[Dict[str, Any]]


@dataclass
class Explanation:
    """Explanation detailing nearest neighbors in the embedding bank."""
    indices: List[int]
    distances: List[float]
    metadata: List[Dict[str, Any]]


class ContrastiveTrustInference:
    """High-level inference interface for trained ContrastiveTrust models."""

    def __init__(
        self,
        detector: ZeroShotDetector,
        post_processor: Optional[PostProcessor] = None,
    ) -> None:
        """Initialize the inference engine.

        Args:
            detector: An initialized (and typically fitted) ZeroShotDetector.
            post_processor: Optional PostProcessor for temporal smoothing/filtering.
        """
        self.detector = detector
        self.post_processor = post_processor

    @classmethod
    def load_model(
        cls,
        path: Union[str, Path],
        detector: ZeroShotDetector,
        post_processor: Optional[PostProcessor] = None,
    ) -> "ContrastiveTrustInference":
        """Load a saved model state into the provided detector and create an inference engine.

        Args:
            path: Path to the directory containing saved detector state.
            detector: Pre-instantiated ZeroShotDetector with matching architecture.
            post_processor: Optional PostProcessor to include in the inference engine.

        Returns:
            A ready-to-use ContrastiveTrustInference instance.
        """
        detector.load(path)
        logger.info(f"Loaded ContrastiveTrust model from {path}.")
        return cls(detector=detector, post_processor=post_processor)

    def load_embedding_bank(self, path: Union[str, Path]) -> None:
        """Replace or update the current embedding bank from disk.

        Args:
            path: Path to the saved embedding bank file (.pt).
        """
        self.detector.scorer.bank.load(path)
        logger.info(f"Loaded embedding bank from {path}.")

    def _compute_confidence(self, scores: torch.Tensor, threshold: float) -> torch.Tensor:
        """Compute confidence based on margin from the threshold."""
        # Simple heuristic: absolute distance from threshold relative to threshold
        margin = torch.abs(scores - threshold)
        confidence = torch.clamp(margin / (threshold + 1e-8), min=0.0, max=1.0)
        return confidence

    def score(
        self, window: torch.Tensor, physics_features: torch.Tensor
    ) -> torch.Tensor:
        """Return the continuous anomaly scores for a batch.

        Args:
            window: Tensor of shape (B, T, S).
            physics_features: Tensor of shape (B, P).

        Returns:
            Tensor of shape (B,) containing anomaly scores.
        """
        scores = self.detector.score(window, physics_features)
        
        # Apply post-processing if configured
        if self.post_processor is not None:
            # We don't have predictions yet, so we just pass dummy ones or zeros
            # since some strategies only affect scores (e.g. MovingAverage)
            dummy_preds = torch.zeros_like(scores)
            if scores.ndim == 1:
                scores, _ = self.post_processor.process(scores, dummy_preds)
            else:
                scores, _ = self.post_processor.batch_process(scores, dummy_preds)
            
        return scores

    def predict_batch(
        self, window: torch.Tensor, physics_features: torch.Tensor, metadata: Optional[List[Dict[str, Any]]] = None
    ) -> BatchAnomalyPrediction:
        """Perform inference on a batch of instances.

        Args:
            window: Tensor of shape (B, T, S).
            physics_features: Tensor of shape (B, P).
            metadata: Optional metadata for each instance in the batch.

        Returns:
            BatchAnomalyPrediction containing scores, predictions, distances, confidences.
        """
        if metadata is None:
            metadata = [{} for _ in range(window.shape[0])]
            
        # Extract embeddings and raw distances
        self.detector.encoder.eval()
        with torch.no_grad():
            w = window.to(self.detector.device)
            p = physics_features.to(self.detector.device)
            out = self.detector.encoder(w, p)
            embeddings = out["embedding"].to(self.detector.scorer.bank.device)
            
            # Raw distance calculation
            if self.detector.scorer.bank.embeddings is None:
                raise RuntimeError("Embedding bank is empty. Cannot compute distances.")
                
            raw_dists = self.detector.scorer.metric.batch_compute(
                embeddings, self.detector.scorer.bank.embeddings
            )
            distances = self.detector.scorer._get_k_distances(raw_dists)
            
            # Anomaly Scores
            scores = self.detector.scorer.strategy.score(distances).to(self.detector.device)
            
        threshold = self.detector.threshold_estimator.predict_threshold()
        predictions = (scores > threshold).to(scores.dtype)

        # Apply post-processing
        if self.post_processor is not None:
            if scores.ndim == 1:
                scores, predictions = self.post_processor.process(scores, predictions)
            else:
                scores, predictions = self.post_processor.batch_process(scores, predictions)
            
        is_anomaly = predictions > 0.5
        confidences = self._compute_confidence(scores, threshold)

        return BatchAnomalyPrediction(
            is_anomaly=is_anomaly,
            scores=scores,
            threshold=threshold,
            distances=distances.to(self.detector.device),
            confidences=confidences,
            metadata=metadata,
        )

    def predict(
        self, window: torch.Tensor, physics_features: torch.Tensor, metadata: Optional[Dict[str, Any]] = None
    ) -> AnomalyPrediction:
        """Perform inference on a single instance.

        Args:
            window: Tensor of shape (T, S) or (1, T, S).
            physics_features: Tensor of shape (P,) or (1, P).
            metadata: Optional metadata for the instance.

        Returns:
            AnomalyPrediction containing detailed context.
        """
        if window.ndim == 2:
            window = window.unsqueeze(0)
        if physics_features.ndim == 1:
            physics_features = physics_features.unsqueeze(0)
            
        if metadata is None:
            metadata = {}

        batch_result = self.predict_batch(window, physics_features, metadata=[metadata])
        
        return AnomalyPrediction(
            is_anomaly=bool(batch_result.is_anomaly[0].item()),
            score=float(batch_result.scores[0].item()),
            threshold=float(batch_result.threshold),
            distance=float(batch_result.distances[0].item()),
            confidence=float(batch_result.confidences[0].item()),
            metadata=batch_result.metadata[0],
        )

    def explain(self, window: torch.Tensor, physics_features: torch.Tensor, k: int = 5) -> Explanation:
        """Provide nearest-neighbor explanation for a single instance.

        Args:
            window: Tensor of shape (T, S) or (1, T, S).
            physics_features: Tensor of shape (P,) or (1, P).
            k: Number of nearest neighbors to retrieve.

        Returns:
            Explanation object containing indices, distances, and metadata of neighbors.
        """
        if window.ndim == 2:
            window = window.unsqueeze(0)
        if physics_features.ndim == 1:
            physics_features = physics_features.unsqueeze(0)
            
        if window.shape[0] != 1 or physics_features.shape[0] != 1:
            raise ValueError("explain() expects a single instance (batch size 1).")

        self.detector.encoder.eval()
        with torch.no_grad():
            w = window.to(self.detector.device)
            p = physics_features.to(self.detector.device)
            out = self.detector.encoder(w, p)
            embedding = out["embedding"].to(self.detector.scorer.bank.device)
            
            distances, indices = self.detector.scorer.bank.nearest_neighbors(
                embedding, k=k, metric=self.detector.scorer.metric.__class__.__name__.lower().replace('distance', '')
            )
            # The 'metric' arg to nearest_neighbors expects a string like "cosine" or "l2".
            # Handling Mahalanobis/etc. may not be supported by nearest_neighbors directly if it's custom,
            # but usually it's cosine or l2. If not, we fallback to computing it manually.
            
            try:
                # Attempt to use the bank's builtin method if it's standard
                metric_name = "cosine" if "cosine" in str(type(self.detector.scorer.metric)).lower() else "l2"
                distances, indices = self.detector.scorer.bank.nearest_neighbors(embedding, k=k, metric=metric_name)
                
            except ValueError:
                # Fallback to manual computation using the exact metric
                raw_dists = self.detector.scorer.metric.compute(embedding, self.detector.scorer.bank.embeddings)
                distances, indices = torch.topk(raw_dists, k, largest=False)
                # Reshape to (1, k) to match nearest_neighbors output
                distances = distances.unsqueeze(0)
                indices = indices.unsqueeze(0)

        # Retrieve metadata
        neighbor_indices = indices[0].tolist()
        neighbor_distances = distances[0].tolist()
        
        # bank.metadata is a dict of lists, we want a list of dicts
        neighbor_metadata = []
        for idx in neighbor_indices:
            meta = {}
            for key, values_list in self.detector.scorer.bank.metadata.items():
                meta[key] = values_list[idx]
            neighbor_metadata.append(meta)
            
        return Explanation(
            indices=neighbor_indices,
            distances=neighbor_distances,
            metadata=neighbor_metadata,
        )
