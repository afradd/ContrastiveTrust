"""Tests for HyperparameterSearch."""

import pytest
from src.experiments.experiment_config import ExperimentConfig
from src.experiments.hyperparameter_search import GridSearch, RandomSearch

@pytest.fixture
def base_config():
    """Create a base configuration."""
    return ExperimentConfig(
        experiment_name="hp_search",
        dataset_name="dummy",
        dataset_paths={"train": "path"}
    )

def test_grid_search_combinations(base_config):
    """Test that GridSearch generates the correct number of combinations."""
    search_space = {
        "training_config.optimizer.lr": [0.01, 0.001],
        "training_config.batch_size": [32, 64]
    }
    
    grid = GridSearch(base_config, search_space)
    configs = grid.generate_configs()
    
    assert len(configs) == 4
    
    # Check that paths were updated correctly
    lrs = [c.training_config.optimizer.lr for c in configs]
    batch_sizes = [c.training_config.batch_size for c in configs]
    
    assert set(lrs) == {0.01, 0.001}
    assert set(batch_sizes) == {32, 64}

def test_random_search_generation(base_config):
    """Test that RandomSearch generates the requested number of trials."""
    search_space = {
        "training_config.optimizer.lr": [0.01, 0.001, 0.0001],
        "training_config.batch_size": [16, 32, 64, 128]
    }
    
    random_search = RandomSearch(base_config, search_space, n_trials=5, random_state=42)
    configs = random_search.generate_configs()
    
    assert len(configs) == 5
    
    for config in configs:
        assert config.training_config.optimizer.lr in [0.01, 0.001, 0.0001]
        assert config.training_config.batch_size in [16, 32, 64, 128]
