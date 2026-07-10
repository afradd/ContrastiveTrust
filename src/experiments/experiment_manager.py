"""Experiment management.

This module provides the ExperimentManager to handle multiple experiments,
maintain an experiment registry, support pausing/resuming, and facilitate
experiment comparisons.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Union

from src.experiments.experiment_config import ExperimentConfig
from src.experiments.experiment_runner import ExperimentRunner

logger = logging.getLogger(__name__)


class ExperimentManager:
    """Manages a registry of experiments."""

    def __init__(self, registry_path: Union[str, Path] = "artifacts/registry.json") -> None:
        """Initialize the ExperimentManager.

        Parameters
        ----------
        registry_path : str or Path
            The file path to the JSON registry file.
        """
        self.registry_path = Path(registry_path)
        self.registry: Dict[str, Dict[str, str]] = {}
        self._load_registry()

    def _load_registry(self) -> None:
        """Load the experiment registry from disk."""
        if self.registry_path.exists():
            try:
                with open(self.registry_path, "r", encoding="utf-8") as f:
                    self.registry = json.load(f)
                logger.info(f"Loaded experiment registry from {self.registry_path}")
            except Exception as e:
                logger.error(f"Failed to load registry: {e}")
                self.registry = {}
        else:
            self.registry = {}

    def _save_registry(self) -> None:
        """Save the experiment registry to disk."""
        self.registry_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.registry_path, "w", encoding="utf-8") as f:
            json.dump(self.registry, f, indent=4)

    def register_experiment(self, experiment_name: str, config_path: str, status: str = "created") -> None:
        """Register an experiment in the registry.

        Parameters
        ----------
        experiment_name : str
            The name of the experiment.
        config_path : str
            The path to the saved configuration file.
        status : str, default="created"
            The current status of the experiment.
        """
        self.registry[experiment_name] = {
            "config_path": config_path,
            "status": status,
        }
        self._save_registry()
        logger.info(f"Registered experiment '{experiment_name}' with status '{status}'")

    def update_status(self, experiment_name: str, status: str) -> None:
        """Update the status of an experiment.

        Parameters
        ----------
        experiment_name : str
            The name of the experiment.
        status : str
            The new status (e.g., 'running', 'completed', 'failed').
        """
        if experiment_name in self.registry:
            self.registry[experiment_name]["status"] = status
            self._save_registry()
            logger.info(f"Updated experiment '{experiment_name}' status to '{status}'")
        else:
            logger.warning(f"Experiment '{experiment_name}' not found in registry.")

    def run_experiment(self, runner: ExperimentRunner) -> None:
        """Run an experiment and track its status in the registry.

        Parameters
        ----------
        runner : ExperimentRunner
            The configured runner for the experiment.
        """
        exp_name = runner.config.experiment_name
        config_path = str(runner.artifact_manager.get_path("configs", "experiment_config.yaml"))
        
        self.register_experiment(exp_name, config_path, status="running")
        
        try:
            runner.run()
            self.update_status(exp_name, "completed")
        except Exception as e:
            logger.error(f"Experiment '{exp_name}' failed: {e}")
            self.update_status(exp_name, "failed")
            raise

    def compare_experiments(self, experiment_names: List[str]) -> Dict[str, Dict[str, float]]:
        """Compare the summary metrics of multiple completed experiments.

        Parameters
        ----------
        experiment_names : list of str
            A list of experiment names to compare.

        Returns
        -------
        dict
            A dictionary mapping experiment names to their summary metrics.
        """
        comparison = {}
        for name in experiment_names:
            if name not in self.registry:
                logger.warning(f"Experiment '{name}' not found in registry. Skipping.")
                continue
                
            config_path = Path(self.registry[name]["config_path"])
            # Assuming the metrics summary is in the metrics directory parallel to configs
            summary_path = config_path.parent.parent / "metrics" / "summary.json"
            
            if summary_path.exists():
                try:
                    with open(summary_path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    comparison[name] = data.get("summary_metrics", {})
                except Exception as e:
                    logger.error(f"Failed to load summary for '{name}': {e}")
            else:
                logger.warning(f"Summary metrics not found for '{name}' at {summary_path}")
                
        return comparison

    def get_history(self) -> Dict[str, Dict[str, str]]:
        """Get the full history of registered experiments.

        Returns
        -------
        dict
            The experiment registry.
        """
        return self.registry
