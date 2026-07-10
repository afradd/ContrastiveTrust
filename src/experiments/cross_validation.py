"""Cross-validation support for experiments.

This module provides the CrossValidator class to automate K-Fold, Stratified
K-Fold, and TimeSeriesSplit cross-validation strategies, aggregating metrics
across all folds.
"""

import logging
from typing import Any, Callable, Dict, List, Optional

import numpy as np
from sklearn.model_selection import KFold, StratifiedKFold, TimeSeriesSplit
from torch.utils.data import Dataset, Subset

from src.experiments.experiment_config import ExperimentConfig
from src.experiments.experiment_runner import ExperimentRunner

logger = logging.getLogger(__name__)


class CrossValidator:
    """Manages cross-validation splits and metric aggregation."""

    def __init__(
        self,
        strategy: str = "kfold",
        n_splits: int = 5,
        shuffle: bool = True,
        random_state: Optional[int] = 42,
    ) -> None:
        """Initialize the CrossValidator.

        Parameters
        ----------
        strategy : str, default="kfold"
            The CV strategy to use ('kfold', 'stratified_kfold', 'timeseries').
        n_splits : int, default=5
            Number of splits/folds.
        shuffle : bool, default=True
            Whether to shuffle data before splitting (not applicable for timeseries).
        random_state : int, optional
            Random state for reproducibility.
        """
        self.strategy = strategy.lower()
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state

        if self.strategy == "kfold":
            self.splitter = KFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
        elif self.strategy == "stratified_kfold":
            self.splitter = StratifiedKFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
        elif self.strategy == "timeseries":
            self.splitter = TimeSeriesSplit(n_splits=n_splits)
        else:
            raise ValueError(f"Unknown CV strategy: {strategy}")

    def split(self, dataset: Dataset, labels: Optional[np.ndarray] = None) -> List[tuple[Subset, Subset]]:
        """Generate train/val subsets for each fold.

        Parameters
        ----------
        dataset : Dataset
            The PyTorch dataset to split.
        labels : np.ndarray, optional
            The labels for stratification. Required if strategy is 'stratified_kfold'.

        Returns
        -------
        list of tuple
            A list of (train_subset, val_subset) tuples.
        """
        if self.strategy == "stratified_kfold" and labels is None:
            raise ValueError("Labels must be provided for StratifiedKFold.")

        indices = np.arange(len(dataset))
        splits = []

        for train_idx, val_idx in self.splitter.split(indices, y=labels):
            train_subset = Subset(dataset, train_idx)
            val_subset = Subset(dataset, val_idx)
            splits.append((train_subset, val_subset))

        return splits

    def run_cv(
        self,
        dataset: Dataset,
        config: ExperimentConfig,
        runner_factory: Callable[[ExperimentConfig, Dataset, Dataset], ExperimentRunner],
        labels: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """Execute cross-validation and aggregate results.

        Parameters
        ----------
        dataset : Dataset
            The complete dataset.
        config : ExperimentConfig
            The base configuration.
        runner_factory : Callable
            A factory function that takes (config, train_subset, val_subset) and
            returns a configured ExperimentRunner.
        labels : np.ndarray, optional
            Labels for stratification.

        Returns
        -------
        dict
            A dictionary of aggregated metrics across all folds.
        """
        splits = self.split(dataset, labels)
        all_metrics: List[Dict[str, float]] = []

        logger.info(f"Starting {self.n_splits}-fold cross-validation ({self.strategy})")

        for fold, (train_subset, val_subset) in enumerate(splits, 1):
            logger.info(f"--- Running Fold {fold}/{self.n_splits} ---")
            
            # Modify config for this fold
            fold_config = ExperimentConfig.from_dict(config.to_dict())
            fold_config.experiment_name = f"{config.experiment_name}_fold_{fold}"
            
            # Initialize runner
            runner = runner_factory(fold_config, train_subset, val_subset)
            
            # Run training and evaluation
            runner.train()
            fold_metrics = runner.evaluate()
            all_metrics.append(fold_metrics)
            
            logger.info(f"Fold {fold} complete. Metrics: {fold_metrics}")

        return self.aggregate_metrics(all_metrics)

    @staticmethod
    def aggregate_metrics(metrics_list: List[Dict[str, float]]) -> Dict[str, Dict[str, float]]:
        """Aggregate metrics across multiple folds.

        Parameters
        ----------
        metrics_list : list of dict
            List of metric dictionaries from each fold.

        Returns
        -------
        dict
            A dictionary containing the mean and standard deviation for each metric.
        """
        if not metrics_list:
            return {}

        keys = metrics_list[0].keys()
        aggregated = {}

        for k in keys:
            values = [m[k] for m in metrics_list if k in m]
            if values:
                aggregated[k] = {
                    "mean": float(np.mean(values)),
                    "std": float(np.std(values)),
                }

        logger.info("Cross-validation aggregation complete.")
        return aggregated
