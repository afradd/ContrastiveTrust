"""Tests for the High-Level Inference API."""

import pytest
import torch
import tempfile
from pathlib import Path

from src.models.encoder import DualStreamEncoder, EncoderConfig
from src.evaluation.embedding_bank import EmbeddingBank
from src.evaluation.anomaly_scorer import AnomalyScorer
from src.evaluation.threshold import ThresholdEstimator
from src.evaluation.zero_shot_detector import ZeroShotDetector
from src.evaluation.post_processing import PostProcessor
from src.evaluation.inference import ContrastiveTrustInference


from src.models.temporal_encoder import TemporalEncoderConfig
from src.models.physics_encoder import PhysicsEncoderConfig
from src.models.fusion import FusionConfig

@pytest.fixture
def dummy_detector():
    """Create a dummy fitted ZeroShotDetector."""
    cfg = EncoderConfig(
        temporal=TemporalEncoderConfig(input_channels=3, embedding_dim=8, hidden_channels=[4], kernel_sizes=[3]),
        physics=PhysicsEncoderConfig(input_dim=2, embedding_dim=8, hidden_dims=[4]),
        fusion=FusionConfig(embedding_dim=8)
    )
    encoder = DualStreamEncoder(cfg)
    bank = EmbeddingBank(embedding_dim=8, max_size=100)
    scorer = AnomalyScorer(bank=bank, metric="cosine", strategy="minmax")
    threshold_est = ThresholdEstimator(strategy="percentile", percentile=95.0)
    
    detector = ZeroShotDetector(
        encoder=encoder,
        scorer=scorer,
        threshold_estimator=threshold_est
    )
    
    # Fake fit
    bank.add(torch.randn(10, 8))
    bank.metadata = {"id": [f"id_{i}" for i in range(10)]}
    
    # Provide scores so ThresholdEstimator doesn't crash on val_queries=None without fake scores
    # Actually, we fixed ZeroShotDetector to compute self-scores if val_loader=None
    # But detector.fit expects a dataloader. We can just manually fit components.
    scorer.fit(torch.randn(5, 8))
    threshold_est.fit(scores=torch.rand(10))
    
    return detector


@pytest.fixture
def inference_engine(dummy_detector):
    post_processor = PostProcessor(["moving_average"])
    return ContrastiveTrustInference(detector=dummy_detector, post_processor=post_processor)


def test_score(inference_engine):
    """Test the score method."""
    window = torch.randn(2, 5, 3)
    physics = torch.randn(2, 2)
    
    scores = inference_engine.score(window, physics)
    assert scores.shape == (2,)


def test_predict_batch(inference_engine):
    """Test batch prediction."""
    window = torch.randn(2, 5, 3)
    physics = torch.randn(2, 2)
    
    metadata = [{"source": "test1"}, {"source": "test2"}]
    prediction = inference_engine.predict_batch(window, physics, metadata=metadata)
    
    assert prediction.is_anomaly.shape == (2,)
    assert prediction.scores.shape == (2,)
    assert prediction.distances.shape == (2,)
    assert prediction.confidences.shape == (2,)
    assert len(prediction.metadata) == 2
    assert prediction.metadata[0]["source"] == "test1"
    
    # Confidences should be between 0 and 1
    assert (prediction.confidences >= 0).all()
    assert (prediction.confidences <= 1).all()


def test_predict_single(inference_engine):
    """Test single instance prediction."""
    window = torch.randn(5, 3)
    physics = torch.randn(2,)
    
    metadata = {"source": "single"}
    prediction = inference_engine.predict(window, physics, metadata=metadata)
    
    assert isinstance(prediction.is_anomaly, bool)
    assert isinstance(prediction.score, float)
    assert isinstance(prediction.distance, float)
    assert isinstance(prediction.confidence, float)
    assert prediction.metadata["source"] == "single"


def test_explain(inference_engine):
    """Test explanation generation."""
    window = torch.randn(5, 3)
    physics = torch.randn(2,)
    
    explanation = inference_engine.explain(window, physics, k=3)
    
    assert len(explanation.indices) == 3
    assert len(explanation.distances) == 3
    assert len(explanation.metadata) == 3
    
    # Metadata should contain the ID we injected
    assert "id" in explanation.metadata[0]


def test_save_load_model(inference_engine):
    """Test serialization of the inference API model state."""
    with tempfile.TemporaryDirectory() as tmpdir:
        inference_engine.detector.save(tmpdir)
        
        # Create a fresh detector structure
        cfg = EncoderConfig(
            temporal=TemporalEncoderConfig(input_channels=3, embedding_dim=8, hidden_channels=[4], kernel_sizes=[3]),
            physics=PhysicsEncoderConfig(input_dim=2, embedding_dim=8, hidden_dims=[4]),
            fusion=FusionConfig(embedding_dim=8)
        )
        encoder = DualStreamEncoder(cfg)
        bank = EmbeddingBank(embedding_dim=8, max_size=100)
        scorer = AnomalyScorer(bank=bank, metric="cosine", strategy="minmax")
        threshold_est = ThresholdEstimator(strategy="percentile", percentile=95.0)
        
        fresh_detector = ZeroShotDetector(encoder, scorer, threshold_est)
        
        # Load
        loaded_engine = ContrastiveTrustInference.load_model(
            path=tmpdir,
            detector=fresh_detector,
            post_processor=PostProcessor(["moving_average"])
        )
        
        assert loaded_engine.detector.threshold_estimator.threshold == inference_engine.detector.threshold_estimator.threshold
        assert torch.allclose(
            loaded_engine.detector.scorer.bank.embeddings,
            inference_engine.detector.scorer.bank.embeddings
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_cuda_compatibility(inference_engine):
    """Test CUDA compatibility for inference API."""
    inference_engine.detector.device = torch.device("cuda")
    inference_engine.detector.encoder.to("cuda")
    inference_engine.detector.scorer.bank.config.device = torch.device("cuda")
    if inference_engine.detector.scorer.bank.embeddings is not None:
        inference_engine.detector.scorer.bank.embeddings = inference_engine.detector.scorer.bank.embeddings.cuda()
    
    window = torch.randn(2, 5, 3).cuda()
    physics = torch.randn(2, 2).cuda()
    
    # Batch predict
    prediction = inference_engine.predict_batch(window, physics)
    assert prediction.scores.device.type == "cuda"
    assert prediction.distances.device.type == "cuda"
    
    # Score
    scores = inference_engine.score(window, physics)
    assert scores.device.type == "cuda"
