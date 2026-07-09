import logging
from typing import Any, Dict, List, Optional, Union

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    auc,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)

logger = logging.getLogger(__name__)


class EvaluationMetrics:
    """
    Computes comprehensive evaluation metrics for industrial anomaly detection.
    
    Supports metrics for classification, ranking, confusion matrix,
    detection rates, threshold-based assessment, and inference timing.
    """

    def __init__(self) -> None:
        """Initialize the EvaluationMetrics instance."""
        self._latest_metrics: Dict[str, float] = {}

    def _validate_inputs(
        self,
        y_true: np.ndarray,
        y_score: np.ndarray,
        y_pred: Optional[np.ndarray] = None,
        inference_times: Optional[np.ndarray] = None,
    ) -> None:
        """
        Validate input arrays for shape and content.
        
        Args:
            y_true: Ground truth binary labels.
            y_score: Predicted anomaly scores.
            y_pred: Predicted binary labels (optional).
            inference_times: Inference latency per sample (optional).
            
        Raises:
            ValueError: If inputs are empty, have mismatched shapes, or are not 1D.
        """
        if len(y_true) == 0:
            raise ValueError("Input arrays must not be empty.")

        if y_true.shape[0] != y_score.shape[0]:
            raise ValueError(
                f"Shape mismatch: y_true ({y_true.shape[0]}) and "
                f"y_score ({y_score.shape[0]}) must have the same length."
            )

        if y_pred is not None and y_true.shape[0] != y_pred.shape[0]:
            raise ValueError("Shape mismatch: y_pred must have the same length as y_true.")

        if y_true.ndim != 1 or y_score.ndim != 1:
            raise ValueError("Inputs must be 1D arrays.")
            
        if y_pred is not None and y_pred.ndim != 1:
            raise ValueError("y_pred must be a 1D array.")

    def compute(
        self,
        y_true: Union[np.ndarray, List[int]],
        y_score: Union[np.ndarray, List[float]],
        y_pred: Optional[Union[np.ndarray, List[int]]] = None,
        threshold: Optional[float] = None,
        inference_times: Optional[Union[np.ndarray, List[float]]] = None,
    ) -> Dict[str, float]:
        """
        Compute evaluation metrics based on true labels and predictions.
        
        Args:
            y_true: Ground truth binary labels (0 for normal, 1 for anomaly).
            y_score: Predicted anomaly scores (higher is more anomalous).
            y_pred: Predicted binary labels. If None, can be computed from threshold.
            threshold: Threshold to convert y_score into y_pred if y_pred is None.
            inference_times: List of inference latencies (seconds per sample).
            
        Returns:
            Dictionary containing all computed metrics.
        """
        y_true_np = np.asarray(y_true, dtype=int)
        y_score_np = np.asarray(y_score, dtype=float)

        y_pred_np = None
        if y_pred is not None:
            y_pred_np = np.asarray(y_pred, dtype=int)
        elif threshold is not None:
            y_pred_np = (y_score_np >= threshold).astype(int)

        inf_times_np = None
        if inference_times is not None:
            inf_times_np = np.asarray(inference_times, dtype=float)

        self._validate_inputs(y_true_np, y_score_np, y_pred_np, inf_times_np)

        metrics: Dict[str, float] = {}

        # 1. Ranking Metrics
        if len(np.unique(y_true_np)) < 2:
            logger.warning("Only one class present in y_true. Ranking metrics are undefined.")
            metrics["roc_auc"] = float("nan")
            metrics["average_precision"] = float("nan")
            metrics["pr_auc"] = float("nan")
        else:
            try:
                metrics["roc_auc"] = float(roc_auc_score(y_true_np, y_score_np))
                metrics["average_precision"] = float(average_precision_score(y_true_np, y_score_np))

                precision, recall, _ = precision_recall_curve(y_true_np, y_score_np)
                metrics["pr_auc"] = float(auc(recall, precision))
            except ValueError as e:
                logger.warning("Could not compute ranking metrics: %s", e)
                metrics["roc_auc"] = float("nan")
                metrics["average_precision"] = float("nan")
                metrics["pr_auc"] = float("nan")

        # 2. Classification, Confusion Matrix, Detection, and Threshold Metrics
        if y_pred_np is not None:
            metrics["accuracy"] = float(accuracy_score(y_true_np, y_pred_np))
            metrics["precision"] = float(precision_score(y_true_np, y_pred_np, zero_division=0))
            metrics["recall"] = float(recall_score(y_true_np, y_pred_np, zero_division=0))
            metrics["f1_score"] = float(f1_score(y_true_np, y_pred_np, zero_division=0))
            metrics["balanced_accuracy"] = float(balanced_accuracy_score(y_true_np, y_pred_np))

            try:
                tn, fp, fn, tp = confusion_matrix(y_true_np, y_pred_np, labels=[0, 1]).ravel()
            except ValueError:
                tn, fp, fn, tp = 0, 0, 0, 0

            metrics["tp"] = float(tp)
            metrics["tn"] = float(tn)
            metrics["fp"] = float(fp)
            metrics["fn"] = float(fn)

            specificity = tn / (tn + fp) if (tn + fp) > 0 else float("nan")
            sensitivity = metrics["recall"]
            far = fp / (fp + tn) if (fp + tn) > 0 else float("nan")
            miss_rate = fn / (fn + tp) if (fn + tp) > 0 else float("nan")

            metrics["specificity"] = float(specificity)
            metrics["sensitivity"] = float(sensitivity)
            metrics["false_alarm_rate"] = float(far)
            metrics["miss_rate"] = float(miss_rate)
            metrics["detection_rate"] = float(sensitivity)
        else:
            logger.info("y_pred and threshold not provided. Skipping classification metrics.")

        # 3. Timing Metrics
        if inf_times_np is not None and len(inf_times_np) > 0:
            avg_latency = float(np.mean(inf_times_np))
            metrics["avg_inference_latency"] = avg_latency
            metrics["throughput"] = 1.0 / avg_latency if avg_latency > 0 else float("nan")

        self._latest_metrics = metrics
        return metrics

    def compute_batch(self, batches: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        Compute metrics over a sequence of batches.
        
        Args:
            batches: List of dictionaries. Each dictionary must contain 'y_true'
                     and 'y_score'. Optionally, they can contain 'y_pred' and
                     'inference_times'.
                     
        Returns:
            Dictionary containing all computed metrics aggregated over batches.
        """
        if not batches:
            raise ValueError("Input batches list must not be empty.")

        y_true_all: List[int] = []
        y_score_all: List[float] = []
        y_pred_all: List[int] = []
        inf_times_all: List[float] = []

        has_pred = False
        has_inf = False

        for b in batches:
            y_true_all.extend(b["y_true"])
            y_score_all.extend(b["y_score"])
            
            if "y_pred" in b and b["y_pred"] is not None:
                has_pred = True
                y_pred_all.extend(b["y_pred"])
                
            if "inference_times" in b and b["inference_times"] is not None:
                has_inf = True
                inf_times_all.extend(b["inference_times"])

        y_pred_input = y_pred_all if has_pred else None
        inf_times_input = inf_times_all if has_inf else None

        return self.compute(
            y_true=y_true_all,
            y_score=y_score_all,
            y_pred=y_pred_input,
            inference_times=inf_times_input,
        )

    def summary(self) -> str:
        """
        Return a formatted string summarizing the latest computed metrics.
        
        Returns:
            A string containing the summary of the computed metrics.
        """
        if not self._latest_metrics:
            return "No metrics computed yet."

        lines = ["Evaluation Metrics Summary:", "-" * 30]
        for k, v in self._latest_metrics.items():
            if np.isnan(v):
                lines.append(f"{k:<25}: NaN")
            elif k in ["tp", "tn", "fp", "fn"]:
                lines.append(f"{k:<25}: {int(v)}")
            else:
                lines.append(f"{k:<25}: {v:.4f}")
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, float]:
        """
        Return the latest computed metrics as a dictionary.
        
        Returns:
            A copy of the latest computed metrics dictionary.
        """
        return self._latest_metrics.copy()
