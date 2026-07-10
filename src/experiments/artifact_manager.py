"""Artifact management for experiments.

This module provides the ArtifactManager to create and organize output
directories for experiments, ensuring files are saved systematically and
preventing accidental overwrites.
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Union

logger = logging.getLogger(__name__)


class ArtifactManager:
    """Manages directory structures and artifact paths for an experiment."""

    def __init__(
        self,
        base_dir: Union[str, Path] = "artifacts/experiments",
        experiment_name: str = "experiment",
        use_timestamp: bool = True,
    ) -> None:
        """Initialize the artifact manager.

        Parameters
        ----------
        base_dir : str or Path
            The root directory for all experiments.
        experiment_name : str
            The name of the current experiment.
        use_timestamp : bool
            Whether to append a timestamp to the experiment directory to prevent
            overwrites.
        """
        self.base_dir = Path(base_dir)
        self.experiment_name = experiment_name
        
        if use_timestamp:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.run_dir = self.base_dir / f"{experiment_name}_{timestamp}"
        else:
            self.run_dir = self.base_dir / experiment_name

        self._directories: Dict[str, Path] = {}
        self._initialize_structure()

    def _initialize_structure(self) -> None:
        """Create the directory structure for the experiment."""
        dirs_to_create = [
            "checkpoints",
            "metrics",
            "figures",
            "reports",
            "logs",
            "configs",
            "embeddings",
        ]

        self.run_dir.mkdir(parents=True, exist_ok=True)
        
        for dir_name in dirs_to_create:
            dir_path = self.run_dir / dir_name
            dir_path.mkdir(exist_ok=True)
            self._directories[dir_name] = dir_path

        logger.info(f"Artifact manager initialized at {self.run_dir}")

    def get_dir(self, name: str) -> Path:
        """Get the path to a specific artifact directory.

        Parameters
        ----------
        name : str
            Name of the directory (e.g., 'checkpoints', 'metrics').

        Returns
        -------
        Path
            The path to the requested directory.

        Raises
        ------
        KeyError
            If the directory name is not managed.
        """
        if name not in self._directories:
            raise KeyError(f"Directory '{name}' not managed by ArtifactManager.")
        return self._directories[name]

    def get_path(self, dir_name: str, filename: str) -> Path:
        """Get the full path for a file within a managed directory.

        Parameters
        ----------
        dir_name : str
            Name of the directory (e.g., 'figures').
        filename : str
            Name of the file.

        Returns
        -------
        Path
            The full path to the file.
        """
        return self.get_dir(dir_name) / filename
