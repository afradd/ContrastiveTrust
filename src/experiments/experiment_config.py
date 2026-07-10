"""Experiment configuration management.

This module provides the ExperimentConfig dataclass to encapsulate all
parameters required to run an experiment, ensuring reproducibility.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import yaml

from src.training.config import TrainingConfig


@dataclass
class ExperimentConfig:
    """Configuration for a ContrastiveTrust experiment.

    Parameters
    ----------
    experiment_name : str
        Name of the experiment.
    dataset_name : str
        Name of the dataset being used.
    dataset_paths : dict[str, str]
        Dictionary of dataset paths (e.g., {"train": "path/to/train"}).
    training_config : TrainingConfig
        Configuration for the training pipeline.
    evaluation_config : dict[str, Any]
        Configuration for evaluation (e.g., metrics to compute).
    inference_config : dict[str, Any]
        Configuration for inference.
    random_seeds : list[int]
        List of random seeds to run for reproducibility/averaging.
    output_dir : str
        Root directory for saving experiment artifacts.
    logging_config : dict[str, Any]
        Configuration for logging.
    hardware_config : dict[str, Any]
        Hardware settings like num_workers, pin_memory.
    """

    experiment_name: str
    dataset_name: str
    dataset_paths: Dict[str, str]
    training_config: TrainingConfig = field(default_factory=TrainingConfig)
    evaluation_config: Dict[str, Any] = field(default_factory=dict)
    inference_config: Dict[str, Any] = field(default_factory=dict)
    random_seeds: List[int] = field(default_factory=lambda: [42])
    output_dir: str = "artifacts/experiments"
    logging_config: Dict[str, Any] = field(default_factory=dict)
    hardware_config: Dict[str, Any] = field(
        default_factory=lambda: {"num_workers": 4, "pin_memory": True}
    )

    def validate(self) -> None:
        """Validate the configuration."""
        if not self.experiment_name:
            raise ValueError("experiment_name must not be empty.")
        if not self.dataset_name:
            raise ValueError("dataset_name must not be empty.")
        if not isinstance(self.dataset_paths, dict) or not self.dataset_paths:
            raise ValueError("dataset_paths must be a non-empty dictionary.")
        if not isinstance(self.random_seeds, list) or not self.random_seeds:
            raise ValueError("random_seeds must be a non-empty list.")
        if not isinstance(self.training_config, TrainingConfig):
            raise TypeError("training_config must be a TrainingConfig instance.")

    def to_dict(self) -> Dict[str, Any]:
        """Convert the configuration to a dictionary.

        Returns
        -------
        dict
            Dictionary representation of the configuration.
        """
        # asdict correctly handles nested dataclasses like TrainingConfig
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ExperimentConfig:
        """Create an ExperimentConfig from a dictionary.

        Parameters
        ----------
        data : dict
            Dictionary representation of the configuration.

        Returns
        -------
        ExperimentConfig
            The parsed configuration.
        """
        # Create a shallow copy to modify safely
        data_copy = data.copy()

        # Reconstruct TrainingConfig if it exists in the data and is a dict
        training_config_data = data_copy.get("training_config")
        if isinstance(training_config_data, dict):
            # This is a bit tricky as TrainingConfig itself contains nested dataclasses
            # (OptimizerConfig, SchedulerConfig, EncoderConfig, ContrastiveTrustLossConfig, ContrastiveViewGeneratorConfig)
            # Since Python 3.7+ dataclasses don't automatically recursively reconstruct from dicts
            # we need a custom deserialization or just trust that TrainingConfig's __init__ handles dicts
            # Unfortunately, standard dataclass __init__ doesn't cast dicts to sub-dataclasses.
            # For simplicity, if we need it robust, we should parse it. 
            # We'll implement a helper to reconstruct it if needed, or assume dacite/similar is used.
            # Assuming simple unpacking works if we don't have deeply nested custom types that fail.
            # We will handle nested dataclasses manually for TrainingConfig.
            from src.training.config import (
                OptimizerConfig, SchedulerConfig, TrainingConfig
            )
            from src.models.encoder import EncoderConfig
            from src.models.temporal_encoder import TemporalEncoderConfig
            from src.models.physics_encoder import PhysicsEncoderConfig
            from src.models.fusion import FusionConfig
            from src.losses.contrastive_trust_loss import ContrastiveTrustLossConfig
            from src.data.view_generator import ContrastiveViewGeneratorConfig
            
            tc_data = training_config_data.copy()
            if "optimizer" in tc_data and isinstance(tc_data["optimizer"], dict):
                tc_data["optimizer"] = OptimizerConfig(**tc_data["optimizer"])
            if "scheduler" in tc_data and isinstance(tc_data["scheduler"], dict):
                tc_data["scheduler"] = SchedulerConfig(**tc_data["scheduler"])
            if "encoder" in tc_data and isinstance(tc_data["encoder"], dict):
                enc_data = tc_data["encoder"].copy()
                if "temporal" in enc_data and isinstance(enc_data["temporal"], dict):
                    enc_data["temporal"] = TemporalEncoderConfig(**enc_data["temporal"])
                if "physics" in enc_data and isinstance(enc_data["physics"], dict):
                    enc_data["physics"] = PhysicsEncoderConfig(**enc_data["physics"])
                if "fusion" in enc_data and isinstance(enc_data["fusion"], dict):
                    enc_data["fusion"] = FusionConfig(**enc_data["fusion"])
                tc_data["encoder"] = EncoderConfig(**enc_data)
            if "loss" in tc_data and isinstance(tc_data["loss"], dict):
                tc_data["loss"] = ContrastiveTrustLossConfig(**tc_data["loss"])
            if "view_generator" in tc_data and isinstance(tc_data["view_generator"], dict):
                tc_data["view_generator"] = ContrastiveViewGeneratorConfig(**tc_data["view_generator"])
                
            data_copy["training_config"] = TrainingConfig(**tc_data)
            
        return cls(**data_copy)

    def save_yaml(self, path: Union[str, Path]) -> None:
        """Save the configuration to a YAML file.

        Parameters
        ----------
        path : str or Path
            The file path to save to.
        """
        out_path = Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            yaml.dump(self.to_dict(), f, sort_keys=False)

    @classmethod
    def load_yaml(cls, path: Union[str, Path]) -> ExperimentConfig:
        """Load a configuration from a YAML file.

        Parameters
        ----------
        path : str or Path
            The file path to load from.

        Returns
        -------
        ExperimentConfig
            The loaded configuration.
        """
        in_path = Path(path)
        with open(in_path, "r", encoding="utf-8") as f:
            data = yaml.load(f, Loader=yaml.UnsafeLoader)
        if data is None:
            raise ValueError(f"YAML file {in_path} is empty or invalid.")
        return cls.from_dict(data)
