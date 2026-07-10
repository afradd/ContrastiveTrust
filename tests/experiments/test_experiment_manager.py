"""Tests for ExperimentManager."""

import json
from pathlib import Path
from src.experiments.experiment_manager import ExperimentManager

def test_experiment_manager_registration(tmp_path):
    """Test registering experiments."""
    registry_path = tmp_path / "registry.json"
    manager = ExperimentManager(registry_path=registry_path)
    
    manager.register_experiment("exp1", "path/to/config.yaml", "running")
    
    assert "exp1" in manager.registry
    assert manager.registry["exp1"]["status"] == "running"
    assert manager.registry["exp1"]["config_path"] == "path/to/config.yaml"
    
    # Verify it saved to disk
    with open(registry_path, "r") as f:
        data = json.load(f)
    assert "exp1" in data

def test_experiment_manager_update_status(tmp_path):
    """Test updating the status of an experiment."""
    manager = ExperimentManager(registry_path=tmp_path / "registry.json")
    manager.register_experiment("exp1", "path/to/config.yaml")
    
    manager.update_status("exp1", "completed")
    assert manager.registry["exp1"]["status"] == "completed"

def test_experiment_manager_comparison(tmp_path):
    """Test comparison of multiple experiments."""
    manager = ExperimentManager(registry_path=tmp_path / "registry.json")
    
    # Setup mock file structure for two experiments
    exp1_dir = tmp_path / "exp1"
    exp1_config = exp1_dir / "configs" / "config.yaml"
    exp1_summary = exp1_dir / "metrics" / "summary.json"
    exp1_summary.parent.mkdir(parents=True)
    exp1_config.parent.mkdir(parents=True)
    
    with open(exp1_summary, "w") as f:
        json.dump({"summary_metrics": {"accuracy": 0.9}}, f)
        
    exp2_dir = tmp_path / "exp2"
    exp2_config = exp2_dir / "configs" / "config.yaml"
    exp2_summary = exp2_dir / "metrics" / "summary.json"
    exp2_summary.parent.mkdir(parents=True)
    exp2_config.parent.mkdir(parents=True)
    
    with open(exp2_summary, "w") as f:
        json.dump({"summary_metrics": {"accuracy": 0.95}}, f)
        
    manager.register_experiment("exp1", str(exp1_config))
    manager.register_experiment("exp2", str(exp2_config))
    
    comparison = manager.compare_experiments(["exp1", "exp2"])
    
    assert comparison["exp1"]["accuracy"] == 0.9
    assert comparison["exp2"]["accuracy"] == 0.95
