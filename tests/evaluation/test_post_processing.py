"""Tests for the temporal post-processing module."""

import pytest
import torch

from src.evaluation.post_processing import (
    PostProcessor,
    MovingAverageStrategy,
    EMAStrategy,
    MajorityVotingStrategy,
    MinDurationStrategy,
)


def test_moving_average():
    """Test MovingAverageStrategy smooths correctly."""
    scores = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
    predictions = torch.zeros_like(scores)
    
    strategy = MovingAverageStrategy(window_size=3)
    smoothed, _ = strategy.process(scores, predictions)
    
    # Causal moving average with padding mode='replicate'.
    # Padded sequence: [1.0, 1.0, 1.0, 2.0, 3.0, 4.0, 5.0]
    # Window size 3 sums over index i-2, i-1, i.
    # At i=0: avg(1.0, 1.0, 1.0) = 1.0
    # At i=1: avg(1.0, 1.0, 2.0) = 4/3 ~ 1.333
    # At i=2: avg(1.0, 2.0, 3.0) = 2.0
    # At i=3: avg(2.0, 3.0, 4.0) = 3.0
    # At i=4: avg(3.0, 4.0, 5.0) = 4.0
    
    expected = torch.tensor([1.0, 4/3, 2.0, 3.0, 4.0])
    torch.testing.assert_close(smoothed, expected, rtol=1e-5, atol=1e-5)
    
    # Test batching
    batch_scores = scores.unsqueeze(0).repeat(2, 1)
    batch_preds = predictions.unsqueeze(0).repeat(2, 1)
    batch_smoothed, _ = strategy.process(batch_scores, batch_preds)
    assert batch_smoothed.shape == (2, 5)
    torch.testing.assert_close(batch_smoothed[0], expected, rtol=1e-5, atol=1e-5)


def test_ema():
    """Test EMAStrategy applies exponential decay correctly."""
    scores = torch.tensor([1.0, 2.0, 3.0])
    predictions = torch.zeros_like(scores)
    
    strategy = EMAStrategy(alpha=0.5)
    smoothed, _ = strategy.process(scores, predictions)
    
    # EMA: y_0 = x_0 = 1.0
    # y_1 = 0.5 * 2.0 + 0.5 * 1.0 = 1.5
    # y_2 = 0.5 * 3.0 + 0.5 * 1.5 = 2.25
    expected = torch.tensor([1.0, 1.5, 2.25])
    torch.testing.assert_close(smoothed, expected)
    

def test_majority_voting():
    """Test MajorityVotingStrategy removes isolated spikes."""
    predictions = torch.tensor([0, 1, 0, 1, 1, 1, 0])
    scores = torch.zeros_like(predictions)
    
    strategy = MajorityVotingStrategy(window_size=3)
    _, smoothed_preds = strategy.process(scores, predictions)
    
    # Padded sequence: [0, 0, 0, 1, 0, 1, 1, 1, 0]
    # At i=0: [0, 0, 0] -> sum=0 -> >1.5 False (0)
    # At i=1: [0, 0, 1] -> sum=1 -> False (0)
    # At i=2: [0, 1, 0] -> sum=1 -> False (0)
    # At i=3: [1, 0, 1] -> sum=2 -> True (1)
    # At i=4: [0, 1, 1] -> sum=2 -> True (1)
    # At i=5: [1, 1, 1] -> sum=3 -> True (1)
    # At i=6: [1, 1, 0] -> sum=2 -> True (1)
    
    expected = torch.tensor([0, 0, 0, 1, 1, 1, 1])
    torch.testing.assert_close(smoothed_preds, expected)


def test_min_duration():
    """Test MinDurationStrategy removes short bursts."""
    predictions = torch.tensor([0, 1, 1, 0, 1, 1, 1, 0, 1])
    scores = torch.zeros_like(predictions)
    
    strategy = MinDurationStrategy(min_duration=3)
    _, smoothed_preds = strategy.process(scores, predictions)
    
    # [1, 1] is length 2 < 3 -> 0
    # [1, 1, 1] is length 3 >= 3 -> keep
    # [1] is length 1 < 3 -> 0
    expected = torch.tensor([0, 0, 0, 0, 1, 1, 1, 0, 0])
    
    # Integer comparison
    assert (smoothed_preds == expected).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_cuda_compatibility():
    """Test strategies work on CUDA tensors."""
    scores = torch.tensor([1.0, 2.0, 3.0], device="cuda")
    predictions = torch.tensor([0, 1, 0], device="cuda")
    
    processor = PostProcessor(["moving_average", "min_duration"])
    smoothed_scores, smoothed_preds = processor.process(scores, predictions)
    
    assert smoothed_scores.device.type == "cuda"
    assert smoothed_preds.device.type == "cuda"


def test_post_processor_orchestration():
    """Test that PostProcessor applies strategies sequentially."""
    processor = PostProcessor([MovingAverageStrategy(window_size=3), MinDurationStrategy(min_duration=2)])
    
    scores = torch.tensor([1.0, 1.0, 4.0, 4.0, 1.0])
    predictions = torch.tensor([0, 1, 0, 1, 0])
    
    # Moving Average (window=3):
    # padded: [1, 1, 1, 1, 4, 4, 1]
    # [1,1,1]->1, [1,1,1]->1, [1,1,4]->2, [1,4,4]->3, [4,4,1]->3
    # smoothed scores = [1.0, 1.0, 2.0, 3.0, 3.0]
    
    # Min Duration (min=2):
    # predictions = [0, 1, 0, 1, 0]
    # Runs are length 1. Both removed.
    # smoothed preds = [0, 0, 0, 0, 0]
    
    new_scores, new_preds = processor.process(scores, predictions)
    expected_scores = torch.tensor([1.0, 1.0, 2.0, 3.0, 3.0])
    expected_preds = torch.tensor([0, 0, 0, 0, 0])
    
    torch.testing.assert_close(new_scores, expected_scores)
    assert (new_preds == expected_preds).all()
