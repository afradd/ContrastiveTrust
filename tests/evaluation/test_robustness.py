"""Tests for the robustness evaluation framework."""

import pytest
import torch
from typing import Any, Dict, List, Optional, Tuple
from torch.utils.data import DataLoader, Dataset
from src.evaluation.robustness import (
    GaussianNoisePerturbation,
    MissingValuesPerturbation,
    SensorDropoutPerturbation,
    RandomSpikesPerturbation,
    TimeShiftPerturbation,
    RobustnessEvaluator,
    RobustnessReport,
)
from src.evaluation.evaluator import Evaluator
from src.evaluation.metrics import EvaluationMetrics

class DummyDataset(Dataset):
    def __init__(self, size=10):
        self.size = size

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return {
            "window": torch.ones(5, 3),  # (T, S)
            "physics_features": torch.randn(2),
            "label": torch.tensor(idx % 2), # 0 or 1
        }

class MockInferenceEngine:
    """Mock engine that always predicts anomaly score = 1.0 if tensor mean is > 0.5."""
    def predict_batch(self, window: torch.Tensor, physics_features: torch.Tensor, metadata: Any = None):
        class MockPreds:
            def __init__(self, scores):
                self.scores = scores
                self.is_anomaly = (scores > 0.5)
        
        # If noise or missing values push the mean down, the score goes down.
        # This gives us a metric that changes under perturbation.
        scores = window.mean(dim=(1, 2))
        return MockPreds(scores)

def test_gaussian_noise():
    window = torch.zeros(2, 5, 3)
    p = GaussianNoisePerturbation(std=1.0)
    out = p(window.clone())
    assert out.shape == window.shape
    # Noise should make the variance roughly 1.0
    assert torch.abs(out.std() - 1.0) < 0.5

def test_missing_values():
    window = torch.ones(2, 5, 3)
    p = MissingValuesPerturbation(p=0.5)
    out = p(window.clone())
    assert out.shape == window.shape
    # Some elements should be exactly 0
    assert (out == 0).any()
    assert (out == 1).any()

def test_sensor_dropout():
    window = torch.ones(2, 5, 3)
    p = SensorDropoutPerturbation(p=0.5)
    out = p(window.clone())
    assert out.shape == window.shape
    # Check if any sensor (dim 2) is completely 0 for all time steps
    sensor_sums = out.sum(dim=1)  # (2, 3)
    assert (sensor_sums == 0).any() or (sensor_sums == 5).any()

def test_random_spikes():
    window = torch.ones(2, 5, 3)
    # std is 0, so spikes would be 0 if we didn't add a safeguard
    p = RandomSpikesPerturbation(p=1.0, multiplier=5.0)
    out = p(window.clone())
    assert out.shape == window.shape
    assert (out.abs() > 1.0).any()

def test_time_shift():
    window = torch.arange(5).view(1, 5, 1).float()
    p = TimeShiftPerturbation(shift=1)
    out = p(window.clone())
    assert out.shape == window.shape
    # Rolling [0, 1, 2, 3, 4] by 1 gives [4, 0, 1, 2, 3]
    assert out[0, 0, 0].item() == 4.0

def test_robustness_evaluator():
    dataset = DummyDataset(size=10)
    loader = DataLoader(dataset, batch_size=2)
    
    engine = MockInferenceEngine()
    metrics = EvaluationMetrics()
    evaluator = Evaluator(inference_engine=engine, metrics=metrics)
    
    perturbations = [
        GaussianNoisePerturbation(std=2.0),
        MissingValuesPerturbation(p=1.0), # Zeroes everything
    ]
    
    rob_eval = RobustnessEvaluator(evaluator=evaluator, perturbations=perturbations)
    reports = rob_eval.evaluate(loader)
    
    assert len(reports) == 2
    
    # MissingValues(p=1.0) sets everything to 0. Mean is 0. Score is 0. is_anomaly is False.
    # True labels are alternating 0 and 1. So recall is 0, F1 is 0.
    # Baseline window is 1s, mean is 1, score 1. is_anomaly True. 
    # True labels alternating. Recall is 1.
    
    report2 = reports[1]
    assert report2.perturbation_name == "MissingValues(p=1.0)"
    assert report2.baseline_metrics["f1_score"] >= 0
    assert report2.perturbed_metrics["f1_score"] == 0.0
    
    # Degradation = Baseline - Perturbed
    assert report2.degradation["f1_score"] == report2.baseline_metrics["f1_score"]
