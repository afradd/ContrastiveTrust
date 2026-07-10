"""Tests for ArtifactManager."""

import pytest
from pathlib import Path
from src.experiments.artifact_manager import ArtifactManager

def test_artifact_manager_initialization(tmp_path):
    """Test directory creation."""
    manager = ArtifactManager(base_dir=tmp_path, experiment_name="test_exp", use_timestamp=False)
    
    assert manager.run_dir == tmp_path / "test_exp"
    assert manager.run_dir.exists()
    
    for d in ["checkpoints", "metrics", "figures", "reports", "configs", "embeddings"]:
        assert (manager.run_dir / d).exists()
        assert manager.get_dir(d) == manager.run_dir / d

def test_artifact_manager_timestamp(tmp_path):
    """Test timestamp appending."""
    manager = ArtifactManager(base_dir=tmp_path, experiment_name="test_exp", use_timestamp=True)
    assert manager.run_dir != tmp_path / "test_exp"
    assert str(manager.run_dir).startswith(str(tmp_path / "test_exp"))

def test_artifact_manager_get_path(tmp_path):
    """Test get_path helper."""
    manager = ArtifactManager(base_dir=tmp_path, experiment_name="test_exp", use_timestamp=False)
    filepath = manager.get_path("figures", "plot.png")
    
    assert filepath == tmp_path / "test_exp" / "figures" / "plot.png"

def test_artifact_manager_invalid_dir(tmp_path):
    """Test error on invalid directory request."""
    manager = ArtifactManager(base_dir=tmp_path, experiment_name="test")
    with pytest.raises(KeyError):
        manager.get_dir("invalid_dir")
