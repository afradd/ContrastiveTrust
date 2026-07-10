"""Hyperparameter search strategies.

This module provides Grid Search and Random Search utilities for hyperparameter
tuning, designed to be extensible for future integration with Optuna.
"""

import itertools
import logging
import random
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional

from src.experiments.experiment_config import ExperimentConfig
from src.experiments.experiment_runner import ExperimentRunner

logger = logging.getLogger(__name__)


class HyperparameterSearch(ABC):
    """Base class for hyperparameter search algorithms."""

    def __init__(self, base_config: ExperimentConfig, search_space: Dict[str, Any]) -> None:
        """Initialize the search strategy.

        Parameters
        ----------
        base_config : ExperimentConfig
            The base configuration to modify for each trial.
        search_space : dict
            The defined search space for hyperparameters.
        """
        self.base_config = base_config
        self.search_space = search_space
        self.trials: List[Dict[str, Any]] = []

    @abstractmethod
    def generate_configs(self) -> List[ExperimentConfig]:
        """Generate a list of configurations to evaluate."""
        pass

    def run(self, runner_factory: Callable[[ExperimentConfig], ExperimentRunner]) -> List[Dict[str, Any]]:
        """Execute the hyperparameter search.

        Parameters
        ----------
        runner_factory : Callable
            A factory function that takes an ExperimentConfig and returns a
            configured ExperimentRunner.

        Returns
        -------
        list of dict
            The results of all trials, sorted by the primary metric.
        """
        configs = self.generate_configs()
        logger.info(f"Starting hyperparameter search with {len(configs)} trials.")

        for i, config in enumerate(configs, 1):
            logger.info(f"--- Running Trial {i}/{len(configs)} ---")
            logger.info(f"Config overrides: {config.experiment_name}")

            runner = runner_factory(config)
            
            # Run the experiment workflow
            runner.train()
            metrics = runner.evaluate()
            
            trial_result = {
                "trial": i,
                "experiment_name": config.experiment_name,
                "metrics": metrics,
            }
            self.trials.append(trial_result)

        logger.info("Hyperparameter search complete.")
        return self.trials


class GridSearch(HyperparameterSearch):
    """Exhaustive search over specified parameter values."""

    def generate_configs(self) -> List[ExperimentConfig]:
        """Generate all combinations in the search space.

        The search space should be a dictionary where keys are parameter paths
        (e.g., 'training_config.optimizer.lr') and values are lists of options.
        """
        keys = list(self.search_space.keys())
        values = list(self.search_space.values())
        
        # Ensure all values are lists
        for i, v in enumerate(values):
            if not isinstance(v, list):
                values[i] = [v]

        combinations = list(itertools.product(*values))
        configs = []

        for i, combination in enumerate(combinations, 1):
            config_dict = self.base_config.to_dict()
            name_suffix = []
            
            for key, val in zip(keys, combination):
                self._set_nested_value(config_dict, key.split('.'), val)
                # create a brief string for the experiment name
                short_key = key.split('.')[-1]
                name_suffix.append(f"{short_key}={val}")

            new_config = ExperimentConfig.from_dict(config_dict)
            new_config.experiment_name = f"{self.base_config.experiment_name}_trial{i}_" + "_".join(name_suffix)
            configs.append(new_config)

        return configs

    def _set_nested_value(self, d: Dict[str, Any], keys: List[str], value: Any) -> None:
        """Set a value in a nested dictionary recursively."""
        for key in keys[:-1]:
            d = d.setdefault(key, {})
        d[keys[-1]] = value


class RandomSearch(HyperparameterSearch):
    """Randomized search over specified parameter distributions."""

    def __init__(
        self,
        base_config: ExperimentConfig,
        search_space: Dict[str, Any],
        n_trials: int = 10,
        random_state: Optional[int] = 42,
    ) -> None:
        """Initialize RandomSearch.

        Parameters
        ----------
        n_trials : int
            Number of random combinations to try.
        random_state : int, optional
            Seed for random sampling.
        """
        super().__init__(base_config, search_space)
        self.n_trials = n_trials
        if random_state is not None:
            random.seed(random_state)

    def generate_configs(self) -> List[ExperimentConfig]:
        """Generate random configurations from the search space.

        Values in search space can be lists (to pick from) or callables
        (to sample from, e.g., lambda: random.uniform(0.1, 1.0)).
        """
        keys = list(self.search_space.keys())
        configs = []

        for i in range(1, self.n_trials + 1):
            config_dict = self.base_config.to_dict()
            name_suffix = []

            for key in keys:
                space_val = self.search_space[key]
                if callable(space_val):
                    val = space_val()
                elif isinstance(space_val, list):
                    val = random.choice(space_val)
                else:
                    val = space_val

                self._set_nested_value(config_dict, key.split('.'), val)
                short_key = key.split('.')[-1]
                # Format floats nicely for the name
                val_str = f"{val:.4f}" if isinstance(val, float) else str(val)
                name_suffix.append(f"{short_key}={val_str}")

            new_config = ExperimentConfig.from_dict(config_dict)
            new_config.experiment_name = f"{self.base_config.experiment_name}_trial{i}_" + "_".join(name_suffix)
            configs.append(new_config)

        return configs

    def _set_nested_value(self, d: Dict[str, Any], keys: List[str], value: Any) -> None:
        for key in keys[:-1]:
            d = d.setdefault(key, {})
        d[keys[-1]] = value
