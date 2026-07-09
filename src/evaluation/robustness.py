"""Robustness evaluation for assessing degradation under noisy conditions."""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
from torch.utils.data import DataLoader

from src.evaluation.evaluator import Evaluator, default_batch_unpacker

logger = logging.getLogger(__name__)


class Perturbation(ABC):
    """Base class for all robustness perturbations."""

    @abstractmethod
    def __call__(self, window: torch.Tensor) -> torch.Tensor:
        """Apply the perturbation to a batch of windows.

        Args:
            window: Input tensor of shape (B, T, S).

        Returns:
            Perturbed tensor of the same shape.
        """
        pass

    @property
    def name(self) -> str:
        """Return the name of the perturbation."""
        return self.__class__.__name__


class GaussianNoisePerturbation(Perturbation):
    """Adds zero-mean Gaussian noise to the input window."""
    def __init__(self, std: float = 0.1):
        self.std = std

    def __call__(self, window: torch.Tensor) -> torch.Tensor:
        noise = torch.randn_like(window) * self.std
        return window + noise

    @property
    def name(self) -> str:
        return f"GaussianNoise(std={self.std})"


class MissingValuesPerturbation(Perturbation):
    """Randomly zeroes out individual elements with probability p."""
    def __init__(self, p: float = 0.1):
        self.p = p

    def __call__(self, window: torch.Tensor) -> torch.Tensor:
        mask = torch.rand_like(window) > self.p
        return window * mask

    @property
    def name(self) -> str:
        return f"MissingValues(p={self.p})"


class SensorDropoutPerturbation(Perturbation):
    """Randomly zeroes out entire channels (sensors) for a batch."""
    def __init__(self, p: float = 0.1):
        self.p = p

    def __call__(self, window: torch.Tensor) -> torch.Tensor:
        B, T, S = window.shape
        mask = torch.rand(B, 1, S, device=window.device) > self.p
        return window * mask

    @property
    def name(self) -> str:
        return f"SensorDropout(p={self.p})"


class RandomSpikesPerturbation(Perturbation):
    """Injects large spikes at random points."""
    def __init__(self, p: float = 0.05, multiplier: float = 5.0):
        self.p = p
        self.multiplier = multiplier

    def __call__(self, window: torch.Tensor) -> torch.Tensor:
        mask = torch.rand_like(window) < self.p
        std = window.std(dim=(0, 1), keepdim=True)
        # Avoid zero std by replacing zeros with ones before multiplying
        std = torch.where(std == 0, torch.ones_like(std), std)
        spikes = mask.to(window.dtype) * std * self.multiplier
        # Random sign for the spike (+ or -)
        signs = torch.randint(0, 2, size=spikes.shape, device=window.device).to(window.dtype) * 2 - 1
        return window + (spikes * signs)

    @property
    def name(self) -> str:
        return f"RandomSpikes(p={self.p}, mult={self.multiplier})"


class TimeShiftPerturbation(Perturbation):
    """Rolls the temporal axis by a specified offset."""
    def __init__(self, shift: int = 5):
        self.shift = shift

    def __call__(self, window: torch.Tensor) -> torch.Tensor:
        # Roll along the time axis (dim=1 for (B, T, S) tensor)
        return torch.roll(window, shifts=self.shift, dims=1)

    @property
    def name(self) -> str:
        return f"TimeShift(shift={self.shift})"


@dataclass
class RobustnessReport:
    """Stores the evaluation metrics for baseline and perturbed data."""
    perturbation_name: str
    baseline_metrics: Dict[str, float]
    perturbed_metrics: Dict[str, float]
    degradation: Dict[str, float]

    def to_dict(self) -> Dict[str, Any]:
        """Convert the report to a serializable dictionary."""
        return {
            "perturbation": self.perturbation_name,
            "baseline": self.baseline_metrics,
            "perturbed": self.perturbed_metrics,
            "degradation": self.degradation,
        }


class RobustnessEvaluator:
    """Evaluates the robustness of an inference engine against data perturbations."""

    def __init__(
        self,
        evaluator: Evaluator,
        perturbations: Optional[List[Perturbation]] = None,
    ):
        """Initialize the RobustnessEvaluator.

        Args:
            evaluator: The base Evaluator instance to generate predictions.
            perturbations: A list of Perturbations to test against.
        """
        self.evaluator = evaluator
        self.perturbations = perturbations or [
            GaussianNoisePerturbation(),
            MissingValuesPerturbation(),
            SensorDropoutPerturbation(),
            RandomSpikesPerturbation(),
            TimeShiftPerturbation(),
        ]
        self.reports: List[RobustnessReport] = []

    def evaluate(
        self,
        loader: DataLoader,
        batch_unpacker: Callable[..., Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[List[Dict[str, Any]]]]] = default_batch_unpacker,
    ) -> List[RobustnessReport]:
        """Run robustness evaluation across all registered perturbations.

        Args:
            loader: A PyTorch DataLoader providing the evaluation data.
            batch_unpacker: Function to extract (window, physics, labels, metadata) from batches.

        Returns:
            A list of RobustnessReport objects detailing performance degradation.
        """
        logger.info("Evaluating baseline performance.")
        baseline_metrics = self.evaluator.evaluate_loader(loader, batch_unpacker=batch_unpacker).copy()

        self.reports = []

        for p in self.perturbations:
            logger.info(f"Evaluating under perturbation: {p.name}")

            # Wrapper unpacker that intercepts and perturbs the window on-the-fly
            def perturbed_unpacker(batch: Any) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[List[Dict[str, Any]]]]:
                window, physics, labels, metadata = batch_unpacker(batch)
                
                # Clone to prevent modifying references to cached or shared tensors
                perturbed_window = p(window.clone())
                return perturbed_window, physics, labels, metadata

            perturbed_metrics = self.evaluator.evaluate_loader(loader, batch_unpacker=perturbed_unpacker).copy()

            degradation = {}
            for metric in ["f1_score", "roc_auc", "detection_rate"]:
                base = baseline_metrics.get(metric, float("nan"))
                pert = perturbed_metrics.get(metric, float("nan"))
                
                # degradation = Baseline - Perturbed (higher degradation is worse)
                if not torch.isnan(torch.tensor(base)) and not torch.isnan(torch.tensor(pert)):
                    degradation[metric] = base - pert
                else:
                    degradation[metric] = float("nan")

            report = RobustnessReport(
                perturbation_name=p.name,
                baseline_metrics=baseline_metrics,
                perturbed_metrics=perturbed_metrics,
                degradation=degradation,
            )
            self.reports.append(report)

        return self.reports
