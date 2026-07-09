"""Tests for the Evaluator module."""

import json
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, Dataset

from src.evaluation.evaluator import Evaluator, default_batch_unpacker
from src.evaluation.inference import BatchAnomalyPrediction, ContrastiveTrustInference
from src.evaluation.metrics import EvaluationMetrics


class DummyDataset(Dataset):
    def __init__(self, size=10):
        self.size = size

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return {
            "window": torch.randn(5, 3),
            "physics_features": torch.randn(4),
            "label": torch.tensor(idx % 2),
            "metadata": {"idx": idx}
        }


@pytest.fixture
def mock_inference_engine():
    engine = MagicMock(spec=ContrastiveTrustInference)
    
    def mock_predict_batch(window, physics, metadata=None):
        batch_size = window.shape[0]
        # mock returns some dummy predictions
        return BatchAnomalyPrediction(
            is_anomaly=torch.randint(0, 2, (batch_size,)).bool(),
            scores=torch.rand(batch_size),
            threshold=0.5,
            distances=torch.rand(batch_size),
            confidences=torch.rand(batch_size),
            metadata=metadata or [{} for _ in range(batch_size)]
        )
        
    engine.predict_batch.side_effect = mock_predict_batch
    return engine


@pytest.fixture
def evaluator(mock_inference_engine):
    return Evaluator(inference_engine=mock_inference_engine)


def test_default_batch_unpacker_dict():
    batch = {
        "window": torch.zeros(2, 5, 3),
        "physics_features": torch.zeros(2, 4),
        "label": torch.tensor([0, 1])
    }
    w, p, l, m = default_batch_unpacker(batch)
    assert w.shape == (2, 5, 3)
    assert p.shape == (2, 4)
    assert (l == torch.tensor([0, 1])).all()
    assert m is None


def test_default_batch_unpacker_tuple():
    batch = (
        torch.zeros(2, 5, 3),
        torch.zeros(2, 4),
        torch.tensor([0, 1])
    )
    w, p, l, m = default_batch_unpacker(batch)
    assert w.shape == (2, 5, 3)
    assert p.shape == (2, 4)
    assert (l == torch.tensor([0, 1])).all()
    assert m is None


def test_default_batch_unpacker_invalid():
    with pytest.raises(ValueError):
        default_batch_unpacker("invalid batch format")


def test_evaluate_single_batch(evaluator):
    window = torch.randn(4, 5, 3)
    physics = torch.randn(4, 4)
    y_true = torch.tensor([0, 1, 0, 1])
    
    results = evaluator.evaluate(window, physics, y_true)
    
    assert isinstance(results, dict)
    assert "accuracy" in results
    assert "avg_inference_latency" in results
    
    # check that predict_batch was called
    evaluator.inference_engine.predict_batch.assert_called_once()


def test_evaluate_loader(evaluator):
    dataset = DummyDataset(size=12)
    loader = DataLoader(dataset, batch_size=4, shuffle=False)
    
    results = evaluator.evaluate_loader(loader)
    
    assert isinstance(results, dict)
    assert "accuracy" in results
    assert evaluator.inference_engine.predict_batch.call_count == 3


def test_evaluate_dataset(evaluator):
    dataset = DummyDataset(size=10)
    
    results = evaluator.evaluate_dataset(dataset, batch_size=5)
    
    assert isinstance(results, dict)
    assert "accuracy" in results
    assert evaluator.inference_engine.predict_batch.call_count == 2


def test_save_results(evaluator, tmp_path):
    dataset = DummyDataset(size=4)
    evaluator.evaluate_dataset(dataset, batch_size=4)
    
    out_file = tmp_path / "results.json"
    evaluator.save_results(out_file)
    
    assert out_file.exists()
    with open(out_file, "r") as f:
        data = json.load(f)
        
    assert isinstance(data, dict)
    assert "accuracy" in data


def test_save_results_no_data(evaluator, tmp_path, caplog):
    out_file = tmp_path / "results.json"
    evaluator.save_results(out_file)
    
    assert not out_file.exists()
    assert "No results to save" in caplog.text


def test_summary(evaluator):
    summary = evaluator.summary()
    assert "No metrics computed yet." in summary
    
    dataset = DummyDataset(size=4)
    evaluator.evaluate_dataset(dataset, batch_size=4)
    
    summary = evaluator.summary()
    assert "Evaluation Metrics Summary" in summary
    assert "accuracy" in summary
