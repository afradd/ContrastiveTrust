"""Seed management for deterministic execution.

This module provides utilities to set, validate, and report random seeds across
the Python standard library, NumPy, PyTorch, and CUDA to ensure reproducibility.
"""

import logging
import os
import random
from typing import Dict, Any

import numpy as np
import torch

logger = logging.getLogger(__name__)


class SeedManager:
    """Manages deterministic execution environments."""

    @staticmethod
    def set_seed(seed: int, deterministic_cudnn: bool = True) -> None:
        """Set all random seeds for reproducible results.

        Parameters
        ----------
        seed : int
            The random seed to use.
        deterministic_cudnn : bool, default=True
            Whether to force cuDNN to use deterministic algorithms. This can
            impact performance but is necessary for exact reproducibility.
        """
        logger.info(f"Setting global random seed to {seed}")

        # Python random
        random.seed(seed)
        os.environ["PYTHONHASHSEED"] = str(seed)

        # NumPy
        np.random.seed(seed)

        # PyTorch
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        # CUDA / cuDNN
        if deterministic_cudnn:
            logger.info("Enabling deterministic cuDNN algorithms")
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            
            # Use deterministic algorithms in PyTorch where possible
            if hasattr(torch, 'use_deterministic_algorithms'):
                try:
                    torch.use_deterministic_algorithms(True, warn_only=True)
                    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
                except Exception as e:
                    logger.warning(f"Failed to enable PyTorch deterministic algorithms: {e}")
        else:
            logger.warning("cuDNN deterministic mode disabled. Results may not be perfectly reproducible.")
            torch.backends.cudnn.deterministic = False
            torch.backends.cudnn.benchmark = True

    @staticmethod
    def get_rng_states() -> Dict[str, Any]:
        """Capture the current state of all random number generators.

        Returns
        -------
        dict
            A dictionary containing the RNG states.
        """
        states = {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch_cpu": torch.get_rng_state(),
        }
        if torch.cuda.is_available():
            states["torch_cuda"] = torch.cuda.get_rng_state_all()
            
        return states

    @staticmethod
    def set_rng_states(states: Dict[str, Any]) -> None:
        """Restore the state of all random number generators.

        Parameters
        ----------
        states : dict
            A dictionary containing the RNG states to restore.
        """
        if "python" in states:
            random.setstate(states["python"])
        if "numpy" in states:
            np.random.set_state(states["numpy"])
        if "torch_cpu" in states:
            torch.set_rng_state(states["torch_cpu"])
        if "torch_cuda" in states and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(states["torch_cuda"])

    @staticmethod
    def report_reproducibility_settings() -> Dict[str, Any]:
        """Report current reproducibility settings for logging.

        Returns
        -------
        dict
            A dictionary containing the current configuration.
        """
        settings = {
            "cudnn_deterministic": torch.backends.cudnn.deterministic,
            "cudnn_benchmark": torch.backends.cudnn.benchmark,
            "python_hash_seed": os.environ.get("PYTHONHASHSEED", "Not Set"),
        }
        
        if hasattr(torch, 'are_deterministic_algorithms_enabled'):
            settings["torch_deterministic_algorithms"] = torch.are_deterministic_algorithms_enabled()
            
        return settings
