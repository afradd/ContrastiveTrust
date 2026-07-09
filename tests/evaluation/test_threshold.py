"""Tests for the threshold estimator."""

import json
import tempfile
from pathlib import Path

import pytest
import torch

from src.evaluation.anomaly_scorer import AnomalyScorer
from src.evaluation.embedding_bank import EmbeddingBank
from src.evaluation.threshold import (
    ManualThreshold,
    PercentileThreshold,
    MeanStdThreshold,
    MedianMADThreshold,
    ThresholdEstimator,
    ThresholdStrategy,
)


def test_manual_threshold():
    strategy = ManualThreshold(threshold=0.8)
    scores = torch.tensor([0.1, 0.2, 0.3])
    threshold = strategy.fit(scores)
    assert threshold == 0.8
    assert strategy.get_state() == {"threshold": 0.8}


def test_percentile_threshold():
    strategy = PercentileThreshold(percentile=90.0)
    # [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    scores = torch.arange(1, 11, dtype=torch.float32)
    threshold = strategy.fit(scores)
    # 90th percentile of 1..10 is 9.1
    assert pytest.approx(threshold, 0.01) == 9.1

    with pytest.raises(ValueError):
        PercentileThreshold(percentile=150.0)


def test_mean_std_threshold():
    strategy = MeanStdThreshold(k=2.0)
    scores = torch.tensor([2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0])
    mean = scores.mean().item()
    std = scores.std(unbiased=True).item()
    threshold = strategy.fit(scores)
    assert pytest.approx(threshold) == mean + 2.0 * std


def test_median_mad_threshold():
    strategy = MedianMADThreshold(k=2.0)
    scores = torch.tensor([2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0])
    # PyTorch median for even lengths returns the lower median.
    # median = 4.0
    # abs dev = [2.0, 0.0, 0.0, 0.0, 1.0, 1.0, 3.0, 5.0]
    # median(abs dev) = 1.0
    # approx_std = 1.0 * 1.4826 = 1.4826
    # threshold = 4.0 + 2 * 1.4826 = 6.9652
    threshold = strategy.fit(scores)
    assert pytest.approx(threshold, 0.01) == 6.9652


def test_estimator_with_scores():
    estimator = ThresholdEstimator(strategy="percentile", percentile=50.0)
    scores = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
    threshold = estimator.fit(scores=scores)
    
    assert pytest.approx(threshold) == 3.0
    assert estimator.predict_threshold() == threshold
    assert estimator.summary()["fitted"] is True


def test_estimator_with_scorer():
    device = torch.device("cpu")
    bank = EmbeddingBank(embedding_dim=2, device=device)
    bank.build(torch.tensor([[0.0, 0.0], [1.0, 1.0]], device=device))
    
    scorer = AnomalyScorer(bank, metric="l2", strategy="raw", k=1)
    
    estimator = ThresholdEstimator(strategy="mean_std", k=1.0)
    val_queries = torch.tensor([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]], device=device)
    
    # scores will be [0.0, 0.0, sqrt(2)] -> 0.0, 0.0, 1.4142
    # mean = 0.4714, std = 0.8165
    # threshold = 0.4714 + 1.0 * 0.8165 = 1.2879
    
    threshold = estimator.fit(scorer=scorer, val_queries=val_queries)
    assert pytest.approx(threshold, 0.01) == 1.2879


def test_estimator_invalid_fit():
    estimator = ThresholdEstimator(strategy="manual")
    with pytest.raises(ValueError, match="Must provide either"):
        estimator.fit()


def test_estimator_unfitted():
    estimator = ThresholdEstimator(strategy="manual")
    with pytest.raises(RuntimeError, match="must be fitted"):
        estimator.predict_threshold()


def test_serialization():
    estimator = ThresholdEstimator(strategy="mean_std", k=2.5)
    scores = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
    estimator.fit(scores)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "threshold.json"
        estimator.save(path)
        
        loaded = ThresholdEstimator(strategy="manual") # initialize with something else
        loaded.load(path)
        
        assert loaded.strategy_name == "mean_std"
        assert loaded.strategy.k == 2.5
        assert loaded.predict_threshold() == estimator.predict_threshold()


def test_custom_strategy():
    class CustomThreshold(ThresholdStrategy):
        def fit(self, scores: torch.Tensor) -> float:
            return 42.0

    ThresholdEstimator.register_strategy("custom", CustomThreshold)
    
    estimator = ThresholdEstimator(strategy="custom")
    assert estimator.fit(torch.tensor([1.0])) == 42.0
