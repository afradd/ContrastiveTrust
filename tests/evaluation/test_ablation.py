"""Tests for the ablation framework."""

import os
import json
import csv
import pytest
from src.evaluation.ablation import AblationConfig, AblationStudy
from src.evaluation.evaluator import Evaluator
from src.evaluation.metrics import EvaluationMetrics

class MockEvaluator:
    def __init__(self, config):
        self.config = config
    
    def evaluate_loader(self, loader, batch_unpacker=None):
        # Return mock metrics based on config values
        f1 = 0.9 if self.config.use_physics_encoder else 0.7
        roc = 0.95 if self.config.distance_metric == "cosine" else 0.85
        return {
            "f1_score": f1,
            "roc_auc": roc,
            "pr_auc": 0.8,
            "detection_rate": 0.9,
        }

class MockErrorEvaluator:
    def __init__(self, config):
        self.config = config
    def evaluate_loader(self, loader, batch_unpacker=None):
        raise ValueError("Simulated error")

def test_generate_grid():
    base = AblationConfig(name="base")
    search_space = {
        "use_physics_encoder": [True, False],
        "distance_metric": ["cosine", "euclidean"]
    }
    configs = AblationStudy.generate_grid(base, search_space)
    
    assert len(configs) == 4
    names = [c.name for c in configs]
    assert "base_use_physics_encoder=True_distance_metric=cosine" in names
    
    # Check if configs actually have the properties set
    assert any(c.use_physics_encoder is False and c.distance_metric == "euclidean" for c in configs)

def test_ablation_run():
    c1 = AblationConfig(name="c1", use_physics_encoder=True, distance_metric="cosine")
    c2 = AblationConfig(name="c2", use_physics_encoder=False, distance_metric="euclidean")
    
    study = AblationStudy([c1, c2])
    
    def factory(config):
        return MockEvaluator(config)
    
    study.run(factory, loader=None)
    
    assert "c1" in study.results
    assert "c2" in study.results
    
    assert study.results["c1"]["metrics"]["f1_score"] == 0.9
    assert study.results["c2"]["metrics"]["f1_score"] == 0.7
    
    assert study.results["c1"]["metrics"]["roc_auc"] == 0.95
    assert study.results["c2"]["metrics"]["roc_auc"] == 0.85

def test_ablation_error_handling():
    c1 = AblationConfig(name="c1")
    study = AblationStudy([c1])
    def factory(config):
        return MockErrorEvaluator(config)
    study.run(factory, loader=None)
    
    assert "c1" in study.results
    assert "error" in study.results["c1"]
    assert "Simulated error" in study.results["c1"]["error"]

def test_ablation_exports(tmp_path):
    c1 = AblationConfig(name="c1", use_physics_encoder=True, distance_metric="cosine")
    c2 = AblationConfig(name="c2", use_physics_encoder=False, distance_metric="euclidean")
    
    study = AblationStudy([c1, c2])
    def factory(config):
        return MockEvaluator(config)
    study.run(factory, loader=None)
    
    json_path = tmp_path / "results.json"
    csv_path = tmp_path / "results.csv"
    
    study.export_results(str(json_path), str(csv_path))
    
    assert os.path.exists(json_path)
    assert os.path.exists(csv_path)
    
    with open(json_path, "r") as f:
        data = json.load(f)
        assert "c1" in data
        assert data["c1"]["metrics"]["f1_score"] == 0.9
        
    with open(csv_path, "r") as f:
        reader = csv.reader(f)
        rows = list(reader)
        assert len(rows) == 3 # Header + 2 data rows
        header = rows[0]
        assert "name" in header
        assert "f1_score" in header
        assert "error" in header
        
        c1_row = rows[1]
        # Check that config name is in row
        assert "c1" in c1_row
