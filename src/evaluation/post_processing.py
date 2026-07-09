"""Temporal post-processing strategies for anomaly detection."""

from __future__ import annotations

import abc
import logging
from typing import Any, Dict, List, Tuple, Type, Union

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class PostProcessingStrategy(abc.ABC):
    """Abstract base class for post-processing strategies."""

    @abc.abstractmethod
    def process(self, scores: torch.Tensor, predictions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply the post-processing strategy.

        Args:
            scores: Continuous anomaly scores, shape (T,) or (B, T).
            predictions: Binary predictions (0 or 1), shape (T,) or (B, T).

        Returns:
            Tuple of updated (scores, predictions).
        """
        pass

    def get_state(self) -> Dict[str, Any]:
        """Get the state of the strategy for serialization."""
        return {}

    def load_state(self, state: Dict[str, Any]) -> None:
        """Load the state of the strategy."""
        pass


class MovingAverageStrategy(PostProcessingStrategy):
    """Smooths scores using a causal moving average."""

    def __init__(self, window_size: int = 3) -> None:
        if window_size < 1:
            raise ValueError(f"window_size must be >= 1, got {window_size}")
        self.window_size = window_size

    def process(self, scores: torch.Tensor, predictions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.window_size == 1 or scores.numel() == 0:
            return scores, predictions

        is_1d = scores.ndim == 1
        x = scores.unsqueeze(0) if is_1d else scores
        
        # (B, 1, T)
        x = x.unsqueeze(1).float()
        
        # Causal padding (replicate the first element for the padded region)
        pad_size = self.window_size - 1
        if pad_size > 0:
            padded = F.pad(x, (pad_size, 0), mode='replicate')
        else:
            padded = x
            
        weight = torch.ones(1, 1, self.window_size, dtype=x.dtype, device=x.device) / self.window_size
        smoothed = F.conv1d(padded, weight)
        
        smoothed = smoothed.squeeze(1)
        if is_1d:
            smoothed = smoothed.squeeze(0)
            
        # Cast back to original dtype if needed, though scores are usually float
        return smoothed.to(scores.dtype), predictions

    def get_state(self) -> Dict[str, Any]:
        return {"window_size": self.window_size}

    def load_state(self, state: Dict[str, Any]) -> None:
        self.window_size = state.get("window_size", 3)


class EMAStrategy(PostProcessingStrategy):
    """Smooths scores using an Exponential Moving Average."""

    def __init__(self, alpha: float = 0.1) -> None:
        if not (0.0 < alpha <= 1.0):
            raise ValueError(f"alpha must be in (0, 1], got {alpha}")
        self.alpha = float(alpha)

    def process(self, scores: torch.Tensor, predictions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.alpha == 1.0 or scores.numel() == 0:
            return scores, predictions

        is_1d = scores.ndim == 1
        x = scores.unsqueeze(0) if is_1d else scores
        B, T = x.shape
        
        smoothed = x.clone().float()
        for t in range(1, T):
            smoothed[:, t] = self.alpha * x[:, t] + (1.0 - self.alpha) * smoothed[:, t - 1]
            
        if is_1d:
            smoothed = smoothed.squeeze(0)
            
        return smoothed.to(scores.dtype), predictions

    def get_state(self) -> Dict[str, Any]:
        return {"alpha": self.alpha}

    def load_state(self, state: Dict[str, Any]) -> None:
        self.alpha = float(state.get("alpha", 0.1))


class MajorityVotingStrategy(PostProcessingStrategy):
    """Applies a causal majority vote to the binary predictions."""

    def __init__(self, window_size: int = 3) -> None:
        if window_size < 1:
            raise ValueError(f"window_size must be >= 1, got {window_size}")
        self.window_size = window_size

    def process(self, scores: torch.Tensor, predictions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.window_size == 1 or predictions.numel() == 0:
            return scores, predictions

        is_1d = predictions.ndim == 1
        p = predictions.unsqueeze(0) if is_1d else predictions
        p = p.unsqueeze(1).float()
        
        pad_size = self.window_size - 1
        if pad_size > 0:
            padded = F.pad(p, (pad_size, 0), mode='replicate')
        else:
            padded = p
            
        weight = torch.ones(1, 1, self.window_size, dtype=p.dtype, device=p.device)
        sums = F.conv1d(padded, weight).squeeze(1)
        
        threshold = self.window_size / 2.0
        smoothed_preds = (sums > threshold).to(predictions.dtype)
        
        if is_1d:
            smoothed_preds = smoothed_preds.squeeze(0)
            
        return scores, smoothed_preds

    def get_state(self) -> Dict[str, Any]:
        return {"window_size": self.window_size}

    def load_state(self, state: Dict[str, Any]) -> None:
        self.window_size = state.get("window_size", 3)


class MinDurationStrategy(PostProcessingStrategy):
    """Removes anomaly predictions that do not last for a minimum duration."""

    def __init__(self, min_duration: int = 3) -> None:
        if min_duration < 1:
            raise ValueError(f"min_duration must be >= 1, got {min_duration}")
        self.min_duration = min_duration

    def process(self, scores: torch.Tensor, predictions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.min_duration == 1 or predictions.numel() == 0:
            return scores, predictions

        is_1d = predictions.ndim == 1
        p = predictions.unsqueeze(0) if is_1d else predictions
        B, T = p.shape
        
        # We need to find runs of 1s and check if their length is >= min_duration.
        # This is deterministic and easier to do with a small loop in PyTorch, 
        # or by shifting and masking.
        # Let's use a straightforward approach on CPU to avoid complex CUDA ops if they aren't needed.
        device = p.device
        dtype = p.dtype
        p_cpu = p.cpu().numpy()
        
        # Simple run-length analysis
        import numpy as np
        
        res = np.zeros_like(p_cpu)
        for b in range(B):
            seq = p_cpu[b]
            # Find boundaries of runs
            padded = np.pad(seq, (1, 1), mode='constant', constant_values=0)
            diffs = np.diff(padded)
            starts = np.where(diffs == 1)[0]
            ends = np.where(diffs == -1)[0]
            
            for start, end in zip(starts, ends):
                if end - start >= self.min_duration:
                    res[b, start:end] = 1
                    
        res_tensor = torch.tensor(res, dtype=dtype, device=device)
        
        if is_1d:
            res_tensor = res_tensor.squeeze(0)
            
        return scores, res_tensor

    def get_state(self) -> Dict[str, Any]:
        return {"min_duration": self.min_duration}

    def load_state(self, state: Dict[str, Any]) -> None:
        self.min_duration = state.get("min_duration", 3)


class PostProcessor:
    """Orchestrates multiple temporal post-processing strategies."""

    _registry: Dict[str, Type[PostProcessingStrategy]] = {
        "moving_average": MovingAverageStrategy,
        "ema": EMAStrategy,
        "majority_voting": MajorityVotingStrategy,
        "min_duration": MinDurationStrategy,
    }

    def __init__(self, strategies: Union[str, PostProcessingStrategy, List[Union[str, PostProcessingStrategy]]], **kwargs: Any) -> None:
        """Initialize the PostProcessor.

        Args:
            strategies: A strategy name, instance, or a list of them.
            **kwargs: Configuration passed to the strategy if a single name is provided.
        """
        if not isinstance(strategies, list):
            strategies = [strategies]

        self.strategies: List[PostProcessingStrategy] = []
        for strat in strategies:
            if isinstance(strat, str):
                self.strategies.append(self._create_strategy(strat, **kwargs))
            else:
                self.strategies.append(strat)

    @classmethod
    def _create_strategy(cls, name: str, **kwargs: Any) -> PostProcessingStrategy:
        name_lower = name.lower()
        if name_lower not in cls._registry:
            raise ValueError(f"Unknown strategy '{name}'. Available: {', '.join(cls.available_methods())}")
        return cls._registry[name_lower](**kwargs)

    @classmethod
    def available_methods(cls) -> List[str]:
        """List all available post-processing strategies."""
        return list(cls._registry.keys())

    @classmethod
    def register_method(cls, name: str, strategy_cls: Type[PostProcessingStrategy]) -> None:
        """Register a custom strategy."""
        cls._registry[name.lower()] = strategy_cls

    def process(self, scores: torch.Tensor, predictions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply all configured strategies sequentially to a sequence.

        Args:
            scores: Tensor of shape (T,).
            predictions: Tensor of shape (T,).

        Returns:
            Tuple of updated (scores, predictions).
        """
        if scores.ndim != 1 or predictions.ndim != 1:
            raise ValueError("process() expects 1D tensors. Use batch_process() for 2D.")
            
        for strat in self.strategies:
            scores, predictions = strat.process(scores, predictions)
        return scores, predictions

    def batch_process(self, scores: torch.Tensor, predictions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply all configured strategies sequentially to a batch of sequences.

        Args:
            scores: Tensor of shape (B, T).
            predictions: Tensor of shape (B, T).

        Returns:
            Tuple of updated (scores, predictions).
        """
        if scores.ndim != 2 or predictions.ndim != 2:
            raise ValueError("batch_process() expects 2D tensors of shape (B, T).")
            
        for strat in self.strategies:
            scores, predictions = strat.process(scores, predictions)
        return scores, predictions
