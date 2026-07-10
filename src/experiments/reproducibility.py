"""Reproducibility verification.

This module provides the ReproducibilityValidator class to verify identical
seeds, configuration consistency, and file integrity across experiments.
"""

import hashlib
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple

import yaml

logger = logging.getLogger(__name__)


class ReproducibilityValidator:
    """Validates reproducibility of experiments."""

    @staticmethod
    def hash_file(filepath: Path, algorithm: str = "sha256") -> str:
        """Compute the cryptographic hash of a file.

        Parameters
        ----------
        filepath : Path
            Path to the file.
        algorithm : str, default="sha256"
            Hash algorithm to use.

        Returns
        -------
        str
            The hex digest of the file.
        """
        hash_func = getattr(hashlib, algorithm)()
        with open(filepath, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_func.update(chunk)
        return hash_func.hexdigest()

    @staticmethod
    def diff_configs(config1_path: Path, config2_path: Path) -> Dict[str, Any]:
        """Find differences between two configuration files.

        Parameters
        ----------
        config1_path : Path
            Path to the first YAML configuration.
        config2_path : Path
            Path to the second YAML configuration.

        Returns
        -------
        dict
            A dictionary containing the differences.
        """
        with open(config1_path, "r", encoding="utf-8") as f:
            config1 = yaml.safe_load(f)
        with open(config2_path, "r", encoding="utf-8") as f:
            config2 = yaml.safe_load(f)

        differences = {}
        for k in config1.keys() | config2.keys():
            v1 = config1.get(k)
            v2 = config2.get(k)
            if v1 != v2:
                differences[k] = {"config1": v1, "config2": v2}

        return differences

    @staticmethod
    def verify_artifact_integrity(
        directory: Path,
        expected_hashes: Dict[str, str],
        algorithm: str = "sha256"
    ) -> List[Tuple[str, bool]]:
        """Verify the integrity of files in a directory against expected hashes.

        Parameters
        ----------
        directory : Path
            The directory containing the artifacts.
        expected_hashes : dict
            A dictionary mapping relative file paths to expected hash strings.
        algorithm : str, default="sha256"
            The hashing algorithm used.

        Returns
        -------
        list of tuple
            A list of (filepath, is_valid) tuples.
        """
        results = []
        for rel_path, expected_hash in expected_hashes.items():
            filepath = directory / rel_path
            if filepath.exists():
                actual_hash = ReproducibilityValidator.hash_file(filepath, algorithm)
                is_valid = (actual_hash == expected_hash)
                results.append((rel_path, is_valid))
                if not is_valid:
                    logger.warning(
                        f"Integrity check failed for {rel_path}. "
                        f"Expected {expected_hash}, got {actual_hash}."
                    )
            else:
                logger.warning(f"File {rel_path} not found for integrity check.")
                results.append((rel_path, False))
        return results

    @staticmethod
    def generate_reproducibility_report(
        experiment_dir: Path,
        output_path: Path
    ) -> None:
        """Generate a reproducibility report for an experiment.

        Parameters
        ----------
        experiment_dir : Path
            The base directory of the experiment.
        output_path : Path
            The path to save the report.
        """
        logger.info(f"Generating reproducibility report for {experiment_dir}")
        report = {
            "experiment_dir": str(experiment_dir),
            "config_exists": (experiment_dir / "configs" / "experiment_config.yaml").exists(),
            "metrics_exists": (experiment_dir / "metrics" / "summary.json").exists(),
            "hashes": {}
        }
        
        # Hash important artifacts to allow future verification
        checkpoints_dir = experiment_dir / "checkpoints"
        if checkpoints_dir.exists():
            for ckpt in checkpoints_dir.glob("*.pt"):
                rel_path = ckpt.relative_to(experiment_dir)
                report["hashes"][str(rel_path)] = ReproducibilityValidator.hash_file(ckpt)

        # Hash the configuration itself
        config_file = experiment_dir / "configs" / "experiment_config.yaml"
        if config_file.exists():
            rel_path = config_file.relative_to(experiment_dir)
            report["hashes"][str(rel_path)] = ReproducibilityValidator.hash_file(config_file)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=4)
            
        logger.info(f"Reproducibility report saved to {output_path}")
