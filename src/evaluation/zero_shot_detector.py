"""Zero-shot anomaly detector orchestrating the full evaluation pipeline."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple, Union

import torch
import torch.nn as nn

try:
    from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

from .anomaly_scorer import AnomalyScorer
from .threshold import ThresholdEstimator

logger = logging.getLogger(__name__)


class ZeroShotDetector:
    """Orchestrates the zero-shot inference pipeline for ContrastiveTrust.

    Combines the DualStreamEncoder, EmbeddingBank, AnomalyScorer, and
    ThresholdEstimator into a single, cohesive workflow.
    """

    def __init__(
        self,
        encoder: nn.Module,
        scorer: AnomalyScorer,
        threshold_estimator: ThresholdEstimator,
        device: Union[str, torch.device] = "cpu",
    ) -> None:
        """Initialize the ZeroShotDetector.

        Args:
            encoder: The trained encoder module.
            scorer: The initialized AnomalyScorer (must contain an EmbeddingBank).
            threshold_estimator: The initialized ThresholdEstimator.
            device: Device to run inference on.
        """
        self.encoder = encoder
        self.scorer = scorer
        self.threshold_estimator = threshold_estimator
        
        if isinstance(device, str):
            self.device = torch.device(device)
        else:
            self.device = device

        self.encoder.to(self.device)

    def fit(
        self,
        normal_loader: Iterable[Tuple[torch.Tensor, torch.Tensor]],
        val_loader: Optional[Iterable[Tuple[torch.Tensor, torch.Tensor]]] = None,
    ) -> None:
        """Fit the detector on normal data and optionally validation data.

        Populates the embedding bank with normal data, fits the anomaly scorer,
        and estimates the threshold.

        Args:
            normal_loader: Iterable yielding (window, physics_features) of normal data.
            val_loader: Optional iterable yielding (window, physics_features). Used to
                fit the threshold estimator and scorer. If None, the threshold and scorer
                are fitted using pairwise distances within the normal data bank.
        """
        self.encoder.eval()
        
        # 1. Populate EmbeddingBank
        all_embeddings = []
        with torch.no_grad():
            for batch in normal_loader:
                if len(batch) >= 2:
                    window, physics = batch[0], batch[1]
                else:
                    raise ValueError("Loader must yield at least (window, physics_features).")
                
                window = window.to(self.device)
                physics = physics.to(self.device)
                
                out = self.encoder(window, physics)
                all_embeddings.append(out["embedding"].detach().cpu())
                
        if not all_embeddings:
            raise ValueError("normal_loader provided no data.")
            
        embeddings_tensor = torch.cat(all_embeddings, dim=0)
        self.scorer.bank.add(embeddings_tensor)
        logger.info(f"Populated EmbeddingBank with {len(self.scorer.bank)} samples.")
        
        # 2. Extract Validation Embeddings
        val_queries = None
        if val_loader is not None:
            val_embeddings = []
            with torch.no_grad():
                for batch in val_loader:
                    if len(batch) >= 2:
                        window, physics = batch[0], batch[1]
                    else:
                        raise ValueError("Loader must yield at least (window, physics_features).")
                        
                    window = window.to(self.device)
                    physics = physics.to(self.device)
                    out = self.encoder(window, physics)
                    val_embeddings.append(out["embedding"].detach().cpu())
                    
            if val_embeddings:
                val_queries = torch.cat(val_embeddings, dim=0).to(self.scorer.bank.device)

        # 3. Fit AnomalyScorer
        self.scorer.fit(val_queries)
        
        # 4. Fit ThresholdEstimator
        if val_queries is not None:
            self.threshold_estimator.fit(scorer=self.scorer, val_queries=val_queries)
        else:
            # Compute scores on the bank itself, excluding self-matches
            bank_emb = self.scorer.bank.embeddings
            distances = self.scorer.metric.pairwise(bank_emb, bank_emb)
            distances.fill_diagonal_(float('inf'))
            
            if self.scorer.k == 1:
                ref_distances = torch.min(distances, dim=-1).values
            else:
                k = min(self.scorer.k, distances.shape[-1])
                topk_dist, _ = torch.topk(distances, k, dim=-1, largest=False)
                ref_distances = topk_dist.mean(dim=-1)
                
            scores = self.scorer.strategy.score(ref_distances)
            self.threshold_estimator.fit(scores=scores)
            
        logger.info("ZeroShotDetector fit complete.")

    def score(self, window: torch.Tensor, physics_features: torch.Tensor) -> torch.Tensor:
        """Compute anomaly scores for a batch of inputs.

        Args:
            window: Tensor of shape (B, T, S).
            physics_features: Tensor of shape (B, P).

        Returns:
            Tensor of shape (B,) containing anomaly scores.
        """
        self.encoder.eval()
        with torch.no_grad():
            window = window.to(self.device)
            physics_features = physics_features.to(self.device)
            out = self.encoder(window, physics_features)
            embeddings = out["embedding"].to(self.scorer.bank.device)
            scores = self.scorer.batch_score(embeddings)
            return scores.to(self.device)

    def predict_batch(
        self, window: torch.Tensor, physics_features: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Predict anomalies for a batch of inputs.

        Args:
            window: Tensor of shape (B, T, S).
            physics_features: Tensor of shape (B, P).

        Returns:
            Tuple of (is_anomaly_tensor, scores_tensor).
        """
        scores = self.score(window, physics_features)
        threshold = self.threshold_estimator.predict_threshold()
        is_anomaly = scores > threshold
        return is_anomaly, scores

    def predict(
        self, window: torch.Tensor, physics_features: torch.Tensor
    ) -> Tuple[bool, float]:
        """Predict anomaly for a single input.

        Args:
            window: Tensor of shape (T, S) or (1, T, S).
            physics_features: Tensor of shape (P,) or (1, P).

        Returns:
            Tuple of (is_anomaly_bool, score_float).
        """
        if window.ndim == 2:
            window = window.unsqueeze(0)
        if physics_features.ndim == 1:
            physics_features = physics_features.unsqueeze(0)
            
        if window.shape[0] != 1 or physics_features.shape[0] != 1:
            raise ValueError("predict expects a single instance (batch size 1).")
            
        is_anomaly, scores = self.predict_batch(window, physics_features)
        return bool(is_anomaly.item()), float(scores.item())

    def evaluate(
        self, dataloader: Iterable[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]
    ) -> Dict[str, float]:
        """Evaluate the detector on a dataset with ground-truth labels.

        Args:
            dataloader: Iterable yielding (window, physics_features, labels).
                Labels should be 1 for anomaly, 0 for normal.

        Returns:
            Dictionary containing 'auroc', 'auprc', and 'f1' metrics.
            Returns 0.0 for metrics if sklearn is not installed.
        """
        if not SKLEARN_AVAILABLE:
            logger.warning("scikit-learn is not installed. Returning 0.0 for all metrics.")
            return {"auroc": 0.0, "auprc": 0.0, "f1": 0.0}
            
        all_scores = []
        all_labels = []
        
        for batch in dataloader:
            if len(batch) >= 3:
                window, physics, labels = batch[0], batch[1], batch[2]
            else:
                raise ValueError("Loader must yield at least (window, physics_features, labels).")
                
            scores = self.score(window, physics)
            all_scores.append(scores.cpu())
            all_labels.append(labels.cpu())
            
        if not all_scores:
            return {"auroc": 0.0, "auprc": 0.0, "f1": 0.0}
            
        scores_arr = torch.cat(all_scores, dim=0).numpy()
        labels_arr = torch.cat(all_labels, dim=0).numpy()
        
        threshold = self.threshold_estimator.predict_threshold()
        preds_arr = (scores_arr > threshold).astype(int)
        
        try:
            auroc = float(roc_auc_score(labels_arr, scores_arr))
        except ValueError:
            auroc = 0.0
            
        try:
            auprc = float(average_precision_score(labels_arr, scores_arr))
        except ValueError:
            auprc = 0.0
            
        try:
            f1 = float(f1_score(labels_arr, preds_arr))
        except ValueError:
            f1 = 0.0
            
        return {
            "auroc": auroc,
            "auprc": auprc,
            "f1": f1
        }

    def save(self, path: Union[str, Path]) -> None:
        """Save the detector's state to a directory.

        Args:
            path: Directory to save the state into.
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        # Save Encoder
        torch.save(self.encoder.state_dict(), path / "encoder.pt")
        
        # Save EmbeddingBank
        self.scorer.bank.save(path / "embedding_bank.pt")
        
        # Save AnomalyScorer state
        scorer_state = {
            "k": self.scorer.k,
            "strategy_state": self.scorer.strategy.get_state(),
        }
        with open(path / "scorer.json", "w") as f:
            json.dump(scorer_state, f, indent=2)
            
        # Save ThresholdEstimator
        self.threshold_estimator.save(path / "threshold.json")
        
        logger.info(f"ZeroShotDetector saved to {path}.")

    def load(self, path: Union[str, Path]) -> None:
        """Load the detector's state from a directory.

        Args:
            path: Directory containing the saved state.
        """
        path = Path(path)
        
        # Load Encoder
        encoder_path = path / "encoder.pt"
        if encoder_path.exists():
            self.encoder.load_state_dict(torch.load(encoder_path, map_location=self.device))
        else:
            logger.warning(f"Encoder state not found at {encoder_path}.")
            
        # Load EmbeddingBank
        bank_path = path / "embedding_bank.pt"
        if bank_path.exists():
            self.scorer.bank.load(bank_path)
        else:
            logger.warning(f"EmbeddingBank state not found at {bank_path}.")
            
        # Load AnomalyScorer state
        scorer_path = path / "scorer.json"
        if scorer_path.exists():
            with open(scorer_path, "r") as f:
                scorer_state = json.load(f)
            self.scorer.k = scorer_state.get("k", self.scorer.k)
            if "strategy_state" in scorer_state:
                self.scorer.strategy.load_state(scorer_state["strategy_state"])
        else:
            logger.warning(f"AnomalyScorer state not found at {scorer_path}.")
            
        # Load ThresholdEstimator
        threshold_path = path / "threshold.json"
        if threshold_path.exists():
            self.threshold_estimator.load(threshold_path)
        else:
            logger.warning(f"ThresholdEstimator state not found at {threshold_path}.")
            
        logger.info(f"ZeroShotDetector loaded from {path}.")
