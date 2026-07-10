"""Tests for ResultTracker."""

import json
import csv
from src.experiments.artifact_manager import ArtifactManager
from src.experiments.result_tracker import ResultTracker

def test_result_tracker_hardware_info(tmp_path):
    """Test hardware info collection."""
    manager = ArtifactManager(base_dir=tmp_path, experiment_name="test", use_timestamp=False)
    tracker = ResultTracker(manager)
    
    info = tracker.hardware_info
    assert "platform" in info
    assert "python_version" in info
    assert "cuda_available" in info

def test_result_tracker_log_and_save(tmp_path):
    """Test logging metrics and saving to CSV/JSON."""
    manager = ArtifactManager(base_dir=tmp_path, experiment_name="test", use_timestamp=False)
    tracker = ResultTracker(manager)
    
    tracker.log_metrics(1, {"loss": 0.5, "acc": 0.8})
    tracker.log_metrics(2, {"loss": 0.4, "acc": 0.9})
    
    tracker.save_results(summary_metrics={"best_acc": 0.9})
    
    metrics_dir = manager.get_dir("metrics")
    csv_path = metrics_dir / "history.csv"
    json_path = metrics_dir / "history.json"
    summary_path = metrics_dir / "summary.json"
    
    assert csv_path.exists()
    assert json_path.exists()
    assert summary_path.exists()
    
    # Check JSON history
    with open(json_path, "r") as f:
        history = json.load(f)
    assert len(history) == 2
    assert history[0]["epoch"] == 1
    assert history[0]["loss"] == 0.5
    
    # Check CSV history
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    assert len(rows) == 2
    assert rows[0]["epoch"] == "1"
    
    # Check Summary
    with open(summary_path, "r") as f:
        summary = json.load(f)
    assert "hardware_info" in summary
    assert summary["summary_metrics"]["best_acc"] == 0.9
