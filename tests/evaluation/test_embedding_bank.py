"""Tests for the EmbeddingBank class."""

import tempfile
from pathlib import Path

import pytest
import torch

from src.evaluation.embedding_bank import EmbeddingBank, EmbeddingBankConfig


def test_initialization_backward_compatibility():
    bank = EmbeddingBank(embedding_dim=128, max_size=1000)
    assert bank.embedding_dim == 128
    assert bank.max_size == 1000
    assert len(bank) == 0
    assert bank.metadata == {}


def test_initialization_config():
    config = EmbeddingBankConfig(embedding_dim=64, max_size=500, normalize=True)
    bank = EmbeddingBank(config=config)
    assert bank.embedding_dim == 64
    assert bank.max_size == 500
    assert bank.normalize is True


def test_build():
    bank = EmbeddingBank(embedding_dim=64)
    embeddings = torch.randn(10, 64)
    metadata = {"ids": list(range(10))}
    
    bank.build(embeddings, metadata)
    assert len(bank) == 10
    assert bank.embeddings.shape == (10, 64)
    assert len(bank.metadata["ids"]) == 10


def test_add():
    bank = EmbeddingBank(embedding_dim=64)
    emb1 = torch.randn(10, 64)
    meta1 = {"ids": list(range(10))}
    bank.build(emb1, meta1)
    
    emb2 = torch.randn(5, 64)
    meta2 = {"ids": list(range(10, 15))}
    bank.add(emb2, meta2)
    
    assert len(bank) == 15
    assert len(bank.metadata["ids"]) == 15
    assert bank.metadata["ids"][-1] == 14


def test_metadata_validation_length_mismatch():
    bank = EmbeddingBank(embedding_dim=64)
    emb = torch.randn(10, 64)
    # Metadata length is 5, embeddings is 10
    meta = {"ids": list(range(5))}
    with pytest.raises(ValueError, match="does not match number of embeddings"):
        bank.build(emb, meta)


def test_metadata_validation_type():
    bank = EmbeddingBank(embedding_dim=64)
    emb = torch.randn(10, 64)
    meta = {"ids": "not_a_list"}
    with pytest.raises(TypeError, match="must be a list"):
        bank.build(emb, meta)


def test_max_size():
    bank = EmbeddingBank(embedding_dim=32, max_size=10)
    emb = torch.randn(15, 32)
    meta = {"ids": list(range(15))}
    bank.build(emb, meta)
    
    assert len(bank) == 10
    assert bank.metadata["ids"] == list(range(5, 15))
    
    emb2 = torch.randn(5, 32)
    meta2 = {"ids": list(range(15, 20))}
    bank.add(emb2, meta2)
    
    assert len(bank) == 10
    assert bank.metadata["ids"] == list(range(10, 20))


def test_remove():
    bank = EmbeddingBank(embedding_dim=16)
    emb = torch.randn(5, 16)
    meta = {"ids": list(range(5))}
    bank.build(emb, meta)
    
    bank.remove([1, 3])
    
    assert len(bank) == 3
    assert bank.metadata["ids"] == [0, 2, 4]


def test_remove_all():
    bank = EmbeddingBank(embedding_dim=16)
    emb = torch.randn(3, 16)
    meta = {"ids": [0, 1, 2]}
    bank.build(emb, meta)
    
    bank.remove([0, 1, 2])
    assert len(bank) == 0
    assert bank.metadata == {}


def test_clear():
    bank = EmbeddingBank(embedding_dim=16)
    emb = torch.randn(5, 16)
    bank.build(emb)
    
    bank.clear()
    assert len(bank) == 0
    assert bank.metadata == {}


def test_save_load_backward_compatible():
    bank = EmbeddingBank(embedding_dim=16, normalize=True)
    emb = torch.randn(10, 16)
    meta = {"labels": list(range(10))}
    bank.build(emb, meta)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "bank.pt"
        bank.save(path)
        
        new_bank = EmbeddingBank(embedding_dim=16)
        new_bank.load(path)
        
        assert new_bank.normalize is True
        assert torch.allclose(new_bank.embeddings, bank.embeddings)
        assert new_bank.metadata == bank.metadata


def test_save_load_config():
    config = EmbeddingBankConfig(embedding_dim=16, max_size=100)
    bank = EmbeddingBank(config=config)
    emb = torch.randn(10, 16)
    bank.build(emb)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "bank_cfg.pt"
        bank.save(path)
        
        new_bank = EmbeddingBank(embedding_dim=1) # Will be overwritten by load
        new_bank.load(path)
        
        assert new_bank.embedding_dim == 16
        assert new_bank.max_size == 100


def test_validation_shape():
    bank = EmbeddingBank(embedding_dim=16)
    with pytest.raises(ValueError, match="Expected embedding dim"):
        bank.build(torch.randn(10, 32))
        
    with pytest.raises(ValueError, match="must be 2D"):
        bank.build(torch.randn(10, 16, 2))


def test_validation_dtype():
    bank = EmbeddingBank(embedding_dim=16, dtype=torch.float32)
    with pytest.raises(TypeError, match="Expected dtype"):
        bank.build(torch.randn(10, 16, dtype=torch.float64))


def test_validation_nan_inf():
    bank = EmbeddingBank(embedding_dim=16)
    emb = torch.randn(10, 16)
    emb[0, 0] = float('nan')
    with pytest.raises(ValueError, match="NaN or Inf"):
        bank.build(emb)
        
    emb = torch.randn(10, 16)
    emb[0, 0] = float('inf')
    with pytest.raises(ValueError, match="NaN or Inf"):
        bank.build(emb)


def test_normalize():
    bank = EmbeddingBank(embedding_dim=16, normalize=True)
    emb = torch.randn(10, 16)
    bank.build(emb)
    
    norms = torch.norm(bank.embeddings, p=2, dim=1)
    assert torch.allclose(norms, torch.ones_like(norms))


def test_summary():
    bank = EmbeddingBank(embedding_dim=16)
    summary = bank.summary()
    assert summary["embedding_dim"] == 16
    assert summary["current_size"] == 0
    assert "cached_stats" in summary


def test_magic_methods():
    bank = EmbeddingBank(embedding_dim=8)
    emb = torch.randn(10, 8)
    meta = {"val": list(range(10))}
    bank.build(emb, meta)
    
    assert len(bank) == 10
    
    # __getitem__ integer
    e, m = bank[2]
    assert e.shape == (8,)
    assert m["val"] == 2
    
    # __getitem__ slice
    e, m = bank[2:5]
    assert e.shape == (3, 8)
    assert m["val"] == [2, 3, 4]
    
    # __getitem__ list
    e, m = bank[[0, 8, 9]]
    assert e.shape == (3, 8)
    assert m["val"] == [0, 8, 9]
    
    # __iter__
    items = list(bank)
    assert len(items) == 10
    assert items[0][1]["val"] == 0
    assert items[9][1]["val"] == 9


def test_cached_statistics():
    bank = EmbeddingBank(embedding_dim=8)
    emb = torch.randn(10, 8)
    bank.build(emb)
    
    mean = bank.mean()
    std = bank.std()
    cov = bank.covariance()
    
    assert mean.shape == (8,)
    assert std.shape == (8,)
    assert cov.shape == (8, 8)
    
    # Check if cached
    assert "mean" in bank._stats_cache
    assert "std" in bank._stats_cache
    assert "covariance" in bank._stats_cache
    
    # Invalidation on add
    bank.add(torch.randn(5, 8))
    assert "mean" not in bank._stats_cache
    
    # Invalidation on remove
    bank.mean() # recreate
    assert "mean" in bank._stats_cache
    bank.remove([0])
    assert "mean" not in bank._stats_cache

    # Invalidation on clear
    bank.mean()
    bank.clear()
    assert "mean" not in bank._stats_cache


def test_nearest_neighbors():
    bank = EmbeddingBank(embedding_dim=4, normalize=True)
    # Create simple basis vectors
    emb = torch.eye(4)
    bank.build(emb)
    
    # Query is close to the first basis vector
    query = torch.tensor([[1.0, 0.1, 0.0, 0.0]])
    
    # Cosine
    dist, idx = bank.nearest_neighbors(query, k=2, metric="cosine")
    assert idx[0, 0] == 0
    
    # Euclidean
    dist, idx = bank.nearest_neighbors(query, k=2, metric="l2")
    assert idx[0, 0] == 0

    # Test single vector vs batch query
    query1d = torch.tensor([0.0, 1.0, 0.1, 0.0])
    dist, idx = bank.nearest_neighbors(query1d, k=1)
    assert idx[0, 0] == 1
