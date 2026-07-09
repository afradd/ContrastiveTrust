"""Evaluation module for ContrastiveTrust."""

import json
import logging
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from src.evaluation.inference import ContrastiveTrustInference
from src.evaluation.metrics import EvaluationMetrics

logger = logging.getLogger(__name__)


def default_batch_unpacker(batch: Any) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[List[Dict[str, Any]]]]:
    """Default unpacker for DataLoader batches.

    Assumes batch is a dict with keys: 'window', 'physics_features', 'label'.
    Optionally 'metadata'. Or a tuple of (window, physics, label).

    Returns:
        Tuple of (window, physics_features, labels, metadata)
    """
    if isinstance(batch, dict):
        window = batch["window"]
        physics = batch["physics_features"]
        labels = batch.get("label", batch.get("labels"))
        metadata = batch.get("metadata")
        return window, physics, labels, metadata
    elif isinstance(batch, (tuple, list)):
        window = batch[0]
        physics = batch[1]
        labels = batch[2]
        metadata = batch[3] if len(batch) > 3 else None
        return window, physics, labels, metadata
    else:
        raise ValueError("Unsupported batch format. Please provide a custom batch_unpacker.")


class Evaluator:
    """Evaluates the ContrastiveTrust model on datasets."""

    def __init__(
        self,
        inference_engine: ContrastiveTrustInference,
        metrics: Optional[EvaluationMetrics] = None,
    ) -> None:
        """Initialize the evaluator.

        Args:
            inference_engine: The inference engine to use for predictions.
            metrics: Optional EvaluationMetrics instance. If None, a new one is created.
        """
        self.inference_engine = inference_engine
        self.metrics = metrics or EvaluationMetrics()
        self._latest_results: Dict[str, float] = {}

    def evaluate(
        self,
        window: torch.Tensor,
        physics_features: torch.Tensor,
        y_true: torch.Tensor,
        metadata: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, float]:
        """Evaluate a single batch.

        Args:
            window: Tensor of shape (B, T, S).
            physics_features: Tensor of shape (B, P).
            y_true: Tensor of shape (B,) containing ground truth labels.
            metadata: Optional list of metadata dictionaries.

        Returns:
            Dictionary containing metrics.
        """
        start_time = time.perf_counter()
        
        preds = self.inference_engine.predict_batch(window, physics_features, metadata)
        
        end_time = time.perf_counter()
        batch_latency = end_time - start_time
        batch_size = window.shape[0]
        inference_times = [batch_latency / batch_size] * batch_size

        y_true_np = y_true.detach().cpu().numpy().astype(int)
        y_score_np = preds.scores.detach().cpu().numpy()
        y_pred_np = preds.is_anomaly.detach().cpu().numpy().astype(int)

        results = self.metrics.compute(
            y_true=y_true_np,
            y_score=y_score_np,
            y_pred=y_pred_np,
            inference_times=inference_times,
        )
        
        self._latest_results = results
        return results

    def evaluate_loader(
        self,
        loader: DataLoader,
        batch_unpacker: Callable[..., Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[List[Dict[str, Any]]]]] = default_batch_unpacker,
    ) -> Dict[str, float]:
        """Evaluate over a PyTorch DataLoader.

        Args:
            loader: The DataLoader to evaluate on.
            batch_unpacker: Callable to extract (window, physics, label, metadata) from a batch.

        Returns:
            Dictionary containing aggregated metrics.
        """
        y_true_all = []
        y_score_all = []
        y_pred_all = []
        inf_times_all = []

        logger.info("Starting evaluation over DataLoader...")

        for batch_idx, batch in enumerate(loader):
            window, physics, labels, metadata = batch_unpacker(batch)

            start_time = time.perf_counter()
            preds = self.inference_engine.predict_batch(window, physics, metadata)
            end_time = time.perf_counter()

            batch_size = window.shape[0]
            batch_latency = end_time - start_time
            
            y_true_all.extend(labels.detach().cpu().numpy().astype(int).tolist())
            y_score_all.extend(preds.scores.detach().cpu().numpy().tolist())
            y_pred_all.extend(preds.is_anomaly.detach().cpu().numpy().astype(int).tolist())
            inf_times_all.extend([batch_latency / batch_size] * batch_size)

        logger.info("Finished evaluation. Computing metrics...")
        
        results = self.metrics.compute(
            y_true=y_true_all,
            y_score=y_score_all,
            y_pred=y_pred_all,
            inference_times=inf_times_all,
        )
        
        self._latest_results = results
        return results

    def evaluate_dataset(
        self,
        dataset: Dataset,
        batch_size: int = 32,
        num_workers: int = 0,
        batch_unpacker: Callable[..., Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[List[Dict[str, Any]]]]] = default_batch_unpacker,
    ) -> Dict[str, float]:
        """Evaluate over a PyTorch Dataset.

        Args:
            dataset: The PyTorch Dataset.
            batch_size: Batch size for DataLoader.
            num_workers: Number of workers for DataLoader.
            batch_unpacker: Callable to extract (window, physics, label, metadata) from a batch.

        Returns:
            Dictionary containing metrics.
        """
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            drop_last=False,
        )
        return self.evaluate_loader(loader, batch_unpacker=batch_unpacker)

    def save_results(self, path: Union[str, Path]) -> None:
        """Save the latest evaluation results to a JSON file.

        Args:
            path: Path to the output JSON file.
        """
        if not self._latest_results:
            logger.warning("No results to save.")
            return

        out_path = Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(self._latest_results, f, indent=4)
            
        logger.info(f"Saved evaluation results to {out_path}.")

    def summary(self) -> str:
        """Get a formatted summary of the latest metrics.

        Returns:
            String summary.
        """
        return self.metrics.summary()
