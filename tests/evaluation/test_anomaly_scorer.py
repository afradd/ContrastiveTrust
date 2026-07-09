"""Tests for the anomaly scorer."""

import pytest
import torch

from src.evaluation.embedding_bank import EmbeddingBank
from src.evaluation.distance_metrics import EuclideanDistance
from src.evaluation.anomaly_scorer import (
    AnomalyScorer,
    RawDistanceStrategy,
    MinMaxStrategy,
    RobustZScoreStrategy,
    PercentileStrategy,
    LogisticStrategy,
)


@pytest.fixture
def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def bank(device):
    bank = EmbeddingBank(embedding_dim=2, device=device)
    embeddings = torch.tensor([
        [0.0, 0.0],
        [1.0, 1.0],
        [2.0, 2.0],
        [3.0, 3.0]
    ], device=device)
    bank.build(embeddings)
    return bank


def test_raw_distance_strategy(device):
    strategy = RawDistanceStrategy()
    dist = torch.tensor([1.0, 2.0, 5.0], device=device)
    strategy.fit(dist)
    scores = strategy.score(dist)
    assert torch.allclose(scores, dist)


def test_minmax_strategy(device):
    strategy = MinMaxStrategy()
    ref = torch.tensor([1.0, 2.0, 5.0], device=device)
    strategy.fit(ref)
    
    test_dist = torch.tensor([1.0, 3.0, 5.0, 6.0, 0.0], device=device)
    scores = strategy.score(test_dist)
    
    # min=1.0, max=5.0, diff=4.0
    # 1.0 -> 0.0
    # 3.0 -> 2.0 / 4.0 = 0.5
    # 5.0 -> 1.0
    # 6.0 -> 1.0 (clamped)
    # 0.0 -> 0.0 (clamped)
    expected = torch.tensor([0.0, 0.5, 1.0, 1.0, 0.0], device=device)
    assert torch.allclose(scores, expected)


def test_minmax_zero_diff(device):
    strategy = MinMaxStrategy()
    ref = torch.tensor([2.0, 2.0], device=device)
    strategy.fit(ref)
    
    test_dist = torch.tensor([1.0, 2.0, 3.0], device=device)
    scores = strategy.score(test_dist)
    # diff is 0, should return 0
    expected = torch.tensor([0.0, 0.0, 0.0], device=device)
    assert torch.allclose(scores, expected)


def test_robust_zscore_strategy(device):
    strategy = RobustZScoreStrategy()
    # median = 3.0
    # abs deviations = [2.0, 1.0, 0.0, 1.0, 2.0]
    # mad = median of abs devs = 1.0
    ref = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0], device=device)
    strategy.fit(ref)
    
    test_dist = torch.tensor([3.0, 4.0], device=device)
    scores = strategy.score(test_dist)
    
    # z = (x - median) / (mad / 0.6745)
    # 3.0 -> 0.0
    # 4.0 -> 1.0 / (1.0 / 0.6745) = 0.6745
    expected = torch.tensor([0.0, 0.6745], device=device)
    assert torch.allclose(scores, expected)


def test_percentile_strategy(device):
    strategy = PercentileStrategy()
    ref = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0], device=device)
    strategy.fit(ref)
    
    test_dist = torch.tensor([0.0, 1.0, 3.5, 5.0, 6.0], device=device)
    scores = strategy.score(test_dist)
    
    # 0.0 is <= 0 elements -> 0.0
    # 1.0 searchsorted right=false gives index 0 (0 <= 1.0). 0/5 = 0.0
    # Wait, searchsorted default is side='left'.
    # If ref is [1, 2, 3, 4, 5]
    # 1.0 -> index 0 (0/5 = 0.0)
    # 3.5 -> index 3 (3/5 = 0.6)
    # 5.0 -> index 4 (4/5 = 0.8)
    # 6.0 -> index 5 (5/5 = 1.0)
    
    expected = torch.tensor([0.0, 0.0, 0.6, 0.8, 1.0], device=device)
    assert torch.allclose(scores, expected)


def test_logistic_strategy(device):
    strategy = LogisticStrategy(steepness=2.0)
    ref = torch.tensor([1.0, 2.0, 3.0], device=device)
    strategy.fit(ref)  # midpoint should become 2.0
    
    test_dist = torch.tensor([2.0, 3.0], device=device)
    scores = strategy.score(test_dist)
    
    # 2.0 -> sigmoid(0) = 0.5
    # 3.0 -> sigmoid(2.0 * 1.0) = sigmoid(2.0) ~ 0.8808
    expected = torch.tensor([0.5, torch.sigmoid(torch.tensor(2.0))], device=device)
    assert torch.allclose(scores, expected)


def test_anomaly_scorer_integration(bank, device):
    scorer = AnomalyScorer(bank, metric="l2", strategy="raw", k=1)
    
    # test query
    query = torch.tensor([[4.0, 4.0]], device=device)
    
    # distances to bank:
    # to [0,0] -> sqrt(32) = 5.6568
    # to [1,1] -> sqrt(18) = 4.2426
    # to [2,2] -> sqrt(8) = 2.8284
    # to [3,3] -> sqrt(2) = 1.4142
    # 1-NN min dist = 1.4142
    
    score = scorer.score(query)
    assert score.shape == ()
    assert torch.allclose(score, torch.tensor(1.4142, device=device), atol=1e-4)
    
    # Batch score
    queries = torch.tensor([
        [4.0, 4.0],
        [3.0, 3.0]
    ], device=device)
    scores = scorer.batch_score(queries)
    assert scores.shape == (2,)
    assert torch.allclose(scores, torch.tensor([1.4142, 0.0], device=device), atol=1e-4)


def test_anomaly_scorer_fit_self(bank, device):
    scorer = AnomalyScorer(bank, metric="l2", strategy="minmax", k=1)
    scorer.fit()
    
    summary = scorer.summary()
    assert summary["metric"] == "EuclideanDistance"
    assert summary["strategy"] == "MinMaxStrategy"
    assert summary["k"] == 1
    
    # Since self distance (ignoring self) for [0,0] is to [1,1] -> 1.4142
    # min_val should be 1.4142, max_val should be 1.4142 (because all points have nearest neighbor at distance sqrt(2))
    assert torch.allclose(torch.tensor(summary["strategy_state"]["min_val"]), torch.tensor(1.41421, device=device), atol=1e-4)


def test_anomaly_scorer_fit_val_queries(bank, device):
    scorer = AnomalyScorer(bank, metric="l2", strategy="minmax", k=1)
    
    val_queries = torch.tensor([
        [0.0, 0.0],
        [4.0, 4.0]
    ], device=device)
    
    scorer.fit(val_queries)
    
    # 1-NN for [0,0] is 0.0 (matches [0,0] in bank)
    # 1-NN for [4,4] is 1.4142 (matches [3,3] in bank)
    # min_val = 0.0, max_val = 1.4142
    
    summary = scorer.summary()
    assert summary["strategy_state"]["min_val"] == 0.0
    assert torch.allclose(torch.tensor(summary["strategy_state"]["max_val"]), torch.tensor(1.41421, device=device), atol=1e-4)


def test_anomaly_scorer_k_greater_than_1(bank, device):
    scorer = AnomalyScorer(bank, metric="l2", strategy="raw", k=2)
    query = torch.tensor([[4.0, 4.0]], device=device)
    
    # Distances to bank: 5.6568, 4.2426, 2.8284, 1.4142
    # 2-NN mean dist: (1.4142 + 2.8284) / 2 = 2.1213
    
    score = scorer.score(query)
    assert torch.allclose(score, torch.tensor(2.1213, device=device), atol=1e-4)


def test_scorer_empty_bank(device):
    empty_bank = EmbeddingBank(embedding_dim=2, device=device)
    scorer = AnomalyScorer(empty_bank, metric="l2", strategy="raw")
    
    with pytest.raises(RuntimeError, match="Cannot fit scorer"):
        scorer.fit()
        
    with pytest.raises(RuntimeError, match="EmbeddingBank is empty"):
        scorer.score(torch.tensor([[1.0, 1.0]], device=device))


def test_unfitted_strategy(device):
    strategy = MinMaxStrategy()
    with pytest.raises(RuntimeError, match="must be fitted"):
        strategy.score(torch.tensor([1.0], device=device))

    strategy = RobustZScoreStrategy()
    with pytest.raises(RuntimeError, match="must be fitted"):
        strategy.score(torch.tensor([1.0], device=device))

    strategy = PercentileStrategy()
    with pytest.raises(RuntimeError, match="must be fitted"):
        strategy.score(torch.tensor([1.0], device=device))
