"""Tests for ExperimentConfig."""

import pytest
from src.experiments.experiment_config import ExperimentConfig
from src.training.config import TrainingConfig

def test_experiment_config_initialization():
    """Test successful initialization of ExperimentConfig."""
    config = ExperimentConfig(
        experiment_name="test_exp",
        dataset_name="dummy_dataset",
        dataset_paths={"train": "path/to/train"}
    )
    assert config.experiment_name == "test_exp"
    assert config.dataset_name == "dummy_dataset"
    assert config.random_seeds == [42]
    assert isinstance(config.training_config, TrainingConfig)

def test_experiment_config_validation():
    """Test validation errors for empty/invalid fields."""
    with pytest.raises(ValueError, match="experiment_name must not be empty"):
        config = ExperimentConfig("", "dummy_dataset", {"train": "path"})
        config.validate()

    with pytest.raises(ValueError, match="dataset_paths must be a non-empty dictionary"):
        config = ExperimentConfig("test", "dummy_dataset", {})
        config.validate()

def test_experiment_config_serialization(tmp_path):
    """Test saving and loading YAML."""
    config = ExperimentConfig(
        experiment_name="test_serialization",
        dataset_name="dummy",
        dataset_paths={"train": "path"}
    )
    
    yaml_file = tmp_path / "config.yaml"
    config.save_yaml(yaml_file)
    
    assert yaml_file.exists()
    
    loaded_config = ExperimentConfig.load_yaml(yaml_file)
    
    assert loaded_config.experiment_name == config.experiment_name
    assert loaded_config.dataset_name == config.dataset_name
    assert loaded_config.dataset_paths == config.dataset_paths
    # Ensure nested TrainingConfig loaded correctly (at least basics)
    assert loaded_config.training_config.epochs == config.training_config.epochs
