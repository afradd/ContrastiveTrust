"""Score calibration for converting anomaly scores to probabilities."""

import logging
from typing import Dict, Literal, Optional, Union

import numpy as np

logger = logging.getLogger(__name__)

CalibrationMethod = Literal["min-max", "z-score", "logistic", "temperature"]


class ScoreCalibrator:
    """Calibrates raw anomaly scores into probabilities [0, 1]."""

    def __init__(self, method: CalibrationMethod = "min-max", **kwargs) -> None:
        """Initialize the ScoreCalibrator.

        Args:
            method: The calibration method to use. Options are:
                'min-max': Linear scaling to [0, 1].
                'z-score': Standard scaling followed by a sigmoid to map to [0, 1].
                'logistic': Platt scaling (logistic regression).
                'temperature': Temperature scaling.
            **kwargs: Additional parameters for specific methods (e.g., 'temperature').
        """
        self.method = method
        self.kwargs = kwargs
        self.is_fitted = False
        self._params: Dict[str, float] = {}

    def fit(self, scores: np.ndarray, labels: Optional[np.ndarray] = None) -> "ScoreCalibrator":
        """Fit the calibrator to the provided scores.

        Args:
            scores: Raw anomaly scores (1D array).
            labels: Ground truth labels (required for 'logistic').

        Returns:
            self
        """
        scores = np.asarray(scores, dtype=float)
        
        if len(scores) == 0:
            raise ValueError("Scores array cannot be empty.")
            
        if np.any(np.isnan(scores)) or np.any(np.isinf(scores)):
            scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)

        if self.method == "min-max":
            self._params["min"] = float(np.min(scores))
            self._params["max"] = float(np.max(scores))
            
        elif self.method == "z-score":
            self._params["mean"] = float(np.mean(scores))
            std = float(np.std(scores))
            self._params["std"] = std if std > 1e-8 else 1.0
            
        elif self.method == "logistic":
            if labels is None:
                raise ValueError("Labels are required for logistic calibration.")
            
            labels = np.asarray(labels, dtype=int)
            if len(labels) != len(scores):
                raise ValueError("Labels and scores must have the same length.")
                
            try:
                from sklearn.linear_model import LogisticRegression
                
                # Reshape for sklearn
                X = scores.reshape(-1, 1)
                clf = LogisticRegression(solver="lbfgs", C=1.0)
                clf.fit(X, labels)
                
                # If only one class is present, coef_ might behave differently or fit might complain
                if len(clf.classes_) < 2:
                    logger.warning("Only one class present during logistic calibration fit.")
                    self._params["coef"] = 1.0
                    self._params["intercept"] = 0.0
                else:
                    self._params["coef"] = float(clf.coef_[0][0])
                    self._params["intercept"] = float(clf.intercept_[0])
            except ImportError:
                logger.warning("scikit-learn not available. Falling back to dummy logistic params.")
                self._params["coef"] = 1.0
                self._params["intercept"] = 0.0
                
        elif self.method == "temperature":
            # Temperature scaling requires a validation set in practice, but often uses a fixed temp
            self._params["temperature"] = float(self.kwargs.get("temperature", 1.5))
            if self._params["temperature"] <= 0:
                raise ValueError("Temperature must be greater than 0.")
        else:
            raise ValueError(f"Unknown calibration method: {self.method}")

        self.is_fitted = True
        return self

    def transform(self, scores: np.ndarray) -> np.ndarray:
        """Apply calibration to raw scores.

        Args:
            scores: Raw anomaly scores (1D array).

        Returns:
            Calibrated probabilities [0, 1].
        """
        if not self.is_fitted:
            raise RuntimeError("Calibrator must be fitted before calling transform.")

        scores = np.asarray(scores, dtype=float)
        
        if len(scores) == 0:
            return np.array([])
            
        if np.any(np.isnan(scores)) or np.any(np.isinf(scores)):
            scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)

        if self.method == "min-max":
            s_min = self._params["min"]
            s_max = self._params["max"]
            if s_max - s_min < 1e-8:
                return np.zeros_like(scores)
            probs = (scores - s_min) / (s_max - s_min)
            return np.clip(probs, 0.0, 1.0)
            
        elif self.method == "z-score":
            mean = self._params["mean"]
            std = self._params["std"]
            z = (scores - mean) / std
            # apply sigmoid to get probabilities
            probs = 1.0 / (1.0 + np.exp(-z))
            return probs
            
        elif self.method == "logistic":
            coef = self._params["coef"]
            intercept = self._params["intercept"]
            z = scores * coef + intercept
            probs = 1.0 / (1.0 + np.exp(-z))
            return probs
            
        elif self.method == "temperature":
            temp = self._params["temperature"]
            # To map distance/score to probability using temperature:
            # simple logistic sigmoid with temp scaling:
            z = scores / temp
            probs = 1.0 / (1.0 + np.exp(-z))
            return probs
            
        return scores

    def fit_transform(self, scores: np.ndarray, labels: Optional[np.ndarray] = None) -> np.ndarray:
        """Fit and transform in one step."""
        return self.fit(scores, labels).transform(scores)
