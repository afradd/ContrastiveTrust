"""Tests for CrossValidator."""

import numpy as np
import pytest
from torch.utils.data import TensorDataset
import torch

from src.experiments.cross_validation import CrossValidator

@pytest.fixture
def dummy_dataset():
    """Create a dummy dataset."""
    x = torch.randn(100, 10)
    y = torch.randint(0, 2, (100,))
    return TensorDataset(x, y), y.numpy()

def test_cross_validator_kfold(dummy_dataset):
    """Test standard K-Fold splitting."""
    dataset, _ = dummy_dataset
    cv = CrossValidator(strategy="kfold", n_splits=5)
    
    splits = cv.split(dataset)
    
    assert len(splits) == 5
    for train, val in splits:
        assert len(train) == 80
        assert len(val) == 20

def test_cross_validator_stratified(dummy_dataset):
    """Test Stratified K-Fold splitting."""
    dataset, labels = dummy_dataset
    cv = CrossValidator(strategy="stratified_kfold", n_splits=5)
    
    splits = cv.split(dataset, labels=labels)
    
    assert len(splits) == 5
    for train, val in splits:
        # Check lengths are roughly equal (stratification might shift slightly based on dist)
        assert 19 <= len(val) <= 21

def test_cross_validator_aggregate_metrics():
    """Test metric aggregation."""
    cv = CrossValidator()
    
    metrics = [
        {"acc": 0.8, "loss": 0.5},
        {"acc": 0.9, "loss": 0.4},
        {"acc": 0.85, "loss": 0.45},
    ]
    
    aggregated = cv.aggregate_metrics(metrics)
    
    assert "acc" in aggregated
    assert "loss" in aggregated
    assert aggregated["acc"]["mean"] == pytest.approx(0.85)
    assert aggregated["loss"]["mean"] == pytest.approx(0.45)
