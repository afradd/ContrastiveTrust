"""Tests for ReproducibilityValidator."""

import json
import yaml
from pathlib import Path
from src.experiments.reproducibility import ReproducibilityValidator

def test_hash_file(tmp_path):
    """Test cryptographic hashing of files."""
    file_path = tmp_path / "test.txt"
    with open(file_path, "w") as f:
        f.write("test content")
        
    hash1 = ReproducibilityValidator.hash_file(file_path)
    hash2 = ReproducibilityValidator.hash_file(file_path)
    
    assert hash1 == hash2
    assert isinstance(hash1, str)
    assert len(hash1) == 64  # SHA256 length

def test_diff_configs(tmp_path):
    """Test configuration diffing."""
    conf1 = tmp_path / "config1.yaml"
    conf2 = tmp_path / "config2.yaml"
    
    with open(conf1, "w") as f:
        yaml.dump({"a": 1, "b": 2}, f)
        
    with open(conf2, "w") as f:
        yaml.dump({"a": 1, "b": 3, "c": 4}, f)
        
    diffs = ReproducibilityValidator.diff_configs(conf1, conf2)
    
    assert "a" not in diffs
    assert "b" in diffs
    assert diffs["b"]["config1"] == 2
    assert diffs["b"]["config2"] == 3
    assert "c" in diffs
    assert diffs["c"]["config1"] is None
    assert diffs["c"]["config2"] == 4

def test_verify_artifact_integrity(tmp_path):
    """Test artifact integrity verification."""
    file1 = tmp_path / "file1.txt"
    file2 = tmp_path / "file2.txt"
    
    with open(file1, "w") as f: f.write("1")
    with open(file2, "w") as f: f.write("2")
    
    hash1 = ReproducibilityValidator.hash_file(file1)
    
    # Intentionally provide wrong hash for file2
    expected = {
        "file1.txt": hash1,
        "file2.txt": "wrong_hash"
    }
    
    results = ReproducibilityValidator.verify_artifact_integrity(tmp_path, expected)
    
    assert len(results) == 2
    for rel_path, is_valid in results:
        if rel_path == "file1.txt":
            assert is_valid is True
        else:
            assert is_valid is False
