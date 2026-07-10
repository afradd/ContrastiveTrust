"""Tests for ExperimentRunner."""

import pytest
from unittest.mock import MagicMock

from src.experiments.experiment_config import ExperimentConfig
from src.experiments.experiment_runner import ExperimentRunner

def test_experiment_runner_initialization():
    """Test initializing the runner with mocks."""
    config = ExperimentConfig(
        experiment_name="test_runner",
        dataset_name="dummy",
        dataset_paths={"train": "path"}
    )
    
    trainer_mock = MagicMock()
    evaluator_mock = MagicMock()
    
    runner = ExperimentRunner(
        config=config,
        trainer=trainer_mock,
        evaluator=evaluator_mock,
        train_loader=MagicMock(),
        val_loader=MagicMock(),
        test_loader=MagicMock()
    )
    
    assert runner.config == config
    assert runner.trainer == trainer_mock
    assert runner.evaluator == evaluator_mock

def test_experiment_runner_train_eval_flow():
    """Test the orchestration flow of train and eval."""
    config = ExperimentConfig(
        experiment_name="test_flow",
        dataset_name="dummy",
        dataset_paths={"train": "path"}
    )
    
    trainer_mock = MagicMock()
    trainer_mock.fit.return_value = [{"epoch": 1, "loss": 0.5}, {"epoch": 2, "loss": 0.4}]
    
    evaluator_mock = MagicMock()
    evaluator_mock.evaluate_loader.return_value = {"accuracy": 0.95}
    
    runner = ExperimentRunner(
        config=config,
        trainer=trainer_mock,
        evaluator=evaluator_mock,
        train_loader=MagicMock(),
        val_loader=MagicMock(),
        test_loader=MagicMock()
    )
    
    runner.run()
    
    # Assert trainer was called
    trainer_mock.fit.assert_called_once()
    
    # Assert evaluator was called
    evaluator_mock.evaluate_loader.assert_called_once()
    evaluator_mock.save_results.assert_called_once()
    
    # Check results tracking
    assert len(runner.result_tracker.metrics_history) == 2
