"""Tests for the distance metrics."""

import pytest
import torch
from typing import Type

from src.evaluation.distance_metrics import (
    DistanceMetric,
    CosineDistance,
    EuclideanDistance,
    SquaredEuclideanDistance,
    ManhattanDistance,
    MahalanobisDistance,
    DistanceMetricFactory,
)


@pytest.fixture
def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def test_cosine_distance(device):
    metric = CosineDistance()
    
    # query: [1, 0]
    query = torch.tensor([[1.0, 0.0]], device=device)
    # bank: [[1, 0], [0, 1], [-1, 0]]
    bank = torch.tensor([
        [1.0, 0.0],
        [0.0, 1.0],
        [-1.0, 0.0]
    ], device=device)
    
    # Compute
    dist = metric.compute(query.squeeze(0), bank)
    assert dist.shape == (3,)
    assert torch.allclose(dist, torch.tensor([0.0, 1.0, 2.0], device=device), atol=1e-5)
    
    # Pairwise
    pw_dist = metric.pairwise(query, bank)
    assert pw_dist.shape == (1, 3)
    assert torch.allclose(pw_dist[0], torch.tensor([0.0, 1.0, 2.0], device=device), atol=1e-5)


def test_euclidean_distance(device):
    metric = EuclideanDistance()
    
    query = torch.tensor([[0.0, 0.0]], device=device)
    bank = torch.tensor([
        [3.0, 4.0],
        [1.0, 1.0],
        [0.0, 0.0]
    ], device=device)
    
    dist = metric.compute(query, bank)
    assert torch.allclose(dist, torch.tensor([5.0, 1.41421356, 0.0], device=device), atol=1e-5)


def test_squared_euclidean_distance(device):
    metric = SquaredEuclideanDistance()
    
    query = torch.tensor([[0.0, 0.0]], device=device)
    bank = torch.tensor([
        [3.0, 4.0],
        [1.0, 1.0],
        [0.0, 0.0]
    ], device=device)
    
    dist = metric.compute(query, bank)
    assert torch.allclose(dist, torch.tensor([25.0, 2.0, 0.0], device=device), atol=1e-5)
    
    # Check that identical vectors result in 0 and not negatives due to numerical issues
    identical_dist = metric.pairwise(bank, bank)
    assert (identical_dist >= 0.0).all()
    assert torch.allclose(torch.diag(identical_dist), torch.zeros(3, device=device))


def test_manhattan_distance(device):
    metric = ManhattanDistance()
    
    query = torch.tensor([[0.0, 0.0]], device=device)
    bank = torch.tensor([
        [3.0, -4.0],
        [1.0, 1.0],
        [0.0, 0.0]
    ], device=device)
    
    dist = metric.compute(query, bank)
    assert torch.allclose(dist, torch.tensor([7.0, 2.0, 0.0], device=device), atol=1e-5)


def test_mahalanobis_distance(device):
    # D = 2
    # Identity cov_inv should act like Euclidean (but Euclidean is sqrt)
    cov_inv = torch.eye(2, device=device)
    metric = MahalanobisDistance(cov_inv=cov_inv)
    
    query = torch.tensor([[0.0, 0.0]], device=device)
    bank = torch.tensor([
        [3.0, 4.0],
        [1.0, 1.0],
        [0.0, 0.0]
    ], device=device)
    
    dist = metric.compute(query, bank)
    # Should equal standard Euclidean distance
    assert torch.allclose(dist, torch.tensor([5.0, 1.41421356, 0.0], device=device), atol=1e-5)
    
    # Scale one dimension
    cov_inv_scaled = torch.tensor([[1.0, 0.0], [0.0, 4.0]], device=device)
    metric_scaled = MahalanobisDistance(cov_inv=cov_inv_scaled)
    # dist^2 = 1 * x^2 + 4 * y^2
    # For [3, 4], dist^2 = 9 + 4*16 = 73 => dist = sqrt(73) ~ 8.544
    dist_scaled = metric_scaled.compute(query, bank)
    assert torch.allclose(dist_scaled[0], torch.tensor(73.0, device=device).sqrt(), atol=1e-5)


def test_mahalanobis_distance_fallback(device):
    # Without providing cov_inv, it logs a warning and uses identity
    metric = MahalanobisDistance()
    query = torch.tensor([[0.0, 0.0]], device=device)
    bank = torch.tensor([[3.0, 4.0]], device=device)
    dist = metric.compute(query, bank)
    assert torch.allclose(dist, torch.tensor([5.0], device=device), atol=1e-5)


def test_batch_compute(device):
    metric = CosineDistance()
    
    queries = torch.tensor([
        [1.0, 0.0],
        [0.0, 1.0]
    ], device=device)
    bank = torch.tensor([
        [1.0, 0.0],
        [0.0, 1.0],
        [-1.0, 0.0]
    ], device=device)
    
    batch_dist = metric.batch_compute(queries, bank)
    assert batch_dist.shape == (2, 3)
    
    # Query 0: [1, 0] -> distances: 0, 1, 2
    assert torch.allclose(batch_dist[0], torch.tensor([0.0, 1.0, 2.0], device=device), atol=1e-5)
    
    # Query 1: [0, 1] -> distances: 1, 0, 1
    assert torch.allclose(batch_dist[1], torch.tensor([1.0, 0.0, 1.0], device=device), atol=1e-5)


def test_compute_1d_query(device):
    metric = EuclideanDistance()
    query = torch.tensor([0.0, 0.0], device=device)
    bank = torch.tensor([[3.0, 4.0]], device=device)
    
    dist = metric.compute(query, bank)
    assert dist.shape == (1,)
    assert torch.allclose(dist, torch.tensor([5.0], device=device))


def test_compute_invalid_query_batch(device):
    metric = EuclideanDistance()
    queries = torch.tensor([[0.0, 0.0], [1.0, 1.0]], device=device)
    bank = torch.tensor([[3.0, 4.0]], device=device)
    
    with pytest.raises(ValueError, match="Expected query to have batch size 1"):
        metric.compute(queries, bank)


def test_factory_create():
    metric1 = DistanceMetricFactory.create("cosine")
    assert isinstance(metric1, CosineDistance)
    
    metric2 = DistanceMetricFactory.create("L2")
    assert isinstance(metric2, EuclideanDistance)
    
    cov_inv = torch.eye(2)
    metric3 = DistanceMetricFactory.create("mahalanobis", cov_inv=cov_inv)
    assert isinstance(metric3, MahalanobisDistance)
    assert torch.allclose(metric3.cov_inv, cov_inv)


def test_factory_invalid():
    with pytest.raises(ValueError, match="Unknown distance metric"):
        DistanceMetricFactory.create("unknown_metric")


def test_factory_available_metrics():
    metrics = DistanceMetricFactory.available_metrics()
    assert "cosine" in metrics
    assert "euclidean" in metrics
    assert "mahalanobis" in metrics


def test_factory_register():
    class CustomMetric(DistanceMetric):
        def pairwise(self, embeddings1, embeddings2):
            return torch.zeros(embeddings1.shape[0], embeddings2.shape[0])
            
    DistanceMetricFactory.register("custom", CustomMetric)
    
    metric = DistanceMetricFactory.create("custom")
    assert isinstance(metric, CustomMetric)
    assert "custom" in DistanceMetricFactory.available_metrics()
