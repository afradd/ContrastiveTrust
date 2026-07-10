"""Experiment orchestration.

This module provides the ExperimentRunner class to orchestrate the entire
lifecycle of an experiment: training, evaluation, benchmarking, and reporting.
"""

import logging
from typing import Any, Callable, Dict, Optional

from src.evaluation.benchmark import BenchmarkRunner
from src.evaluation.evaluator import Evaluator
from src.evaluation.reporting import ReportGenerator
from src.experiments.artifact_manager import ArtifactManager
from src.experiments.experiment_config import ExperimentConfig
from src.experiments.result_tracker import ResultTracker
from src.experiments.seed_manager import SeedManager
from src.training.trainer import Trainer

logger = logging.getLogger(__name__)


class ExperimentRunner:
    """Orchestrates the execution of a single experiment."""

    def __init__(
        self,
        config: ExperimentConfig,
        artifact_manager: Optional[ArtifactManager] = None,
        trainer: Optional[Trainer] = None,
        evaluator: Optional[Evaluator] = None,
        benchmark: Optional[BenchmarkRunner] = None,
        train_loader: Optional[Any] = None,
        val_loader: Optional[Any] = None,
        test_loader: Optional[Any] = None,
    ) -> None:
        """Initialize the ExperimentRunner.

        Parameters
        ----------
        config : ExperimentConfig
            The configuration for this experiment.
        artifact_manager : ArtifactManager, optional
            The artifact manager. If None, a new one is created based on config.
        trainer : Trainer, optional
            The configured trainer instance.
        evaluator : Evaluator, optional
            The configured evaluator instance.
        benchmark : BenchmarkRunner, optional
            The configured benchmark instance.
        train_loader : DataLoader, optional
            The training dataloader.
        val_loader : DataLoader, optional
            The validation dataloader.
        test_loader : DataLoader, optional
            The testing dataloader.
        """
        self.config = config
        self.artifact_manager = artifact_manager or ArtifactManager(
            base_dir=config.output_dir,
            experiment_name=config.experiment_name,
        )
        self.result_tracker = ResultTracker(self.artifact_manager)
        
        self.trainer = trainer
        self.evaluator = evaluator
        self.benchmark = benchmark
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        
        # Ensure base reproducibility settings
        SeedManager.set_seed(self.config.random_seeds[0])
        
        # Save config
        self.config.save_yaml(self.artifact_manager.get_path("configs", "experiment_config.yaml"))

    def train(self) -> None:
        """Run the training loop."""
        if self.trainer is None or self.train_loader is None or self.val_loader is None:
            logger.warning("Trainer or dataloaders not provided. Skipping training.")
            return

        logger.info(f"Starting training for experiment {self.config.experiment_name}")
        history = self.trainer.fit(
            train_loader=self.train_loader,
            val_loader=self.val_loader,
            epochs=self.config.training_config.epochs,
        )
        
        # Log metrics from history
        for epoch_data in history:
            epoch = epoch_data.pop("epoch")
            self.result_tracker.log_metrics(epoch, epoch_data)
            
        logger.info("Training complete.")

    def evaluate(self) -> Dict[str, float]:
        """Run the evaluation loop.

        Returns
        -------
        dict
            A dictionary containing the evaluation metrics.
        """
        if self.evaluator is None or self.test_loader is None:
            logger.warning("Evaluator or test dataloader not provided. Skipping evaluation.")
            return {}

        logger.info(f"Starting evaluation for experiment {self.config.experiment_name}")
        results = self.evaluator.evaluate_loader(self.test_loader)
        
        # Save evaluation results to artifacts
        eval_path = self.artifact_manager.get_path("metrics", "evaluation_results.json")
        self.evaluator.save_results(eval_path)
        
        logger.info("Evaluation complete.")
        return results

    def run_benchmark(self) -> Dict[str, Any]:
        """Run the benchmark suite.

        Returns
        -------
        dict
            Benchmark results.
        """
        if self.benchmark is None or self.test_loader is None:
            logger.warning("Benchmark or test dataloader not provided. Skipping benchmark.")
            return {}

        logger.info(f"Starting benchmark for experiment {self.config.experiment_name}")
        # Note: BenchmarkRunner run() takes dataloader and dataset_name, and doesn't save to file by default.
        # So we adapt it slightly.
        report = self.benchmark.run(self.test_loader, dataset_name=self.config.dataset_name)
        
        bench_path = self.artifact_manager.get_path("metrics", "benchmark_results.json")
        import json
        with open(bench_path, "w", encoding="utf-8") as f:
            json.dump(report.to_dict(), f, indent=4)
        
        logger.info("Benchmark complete.")
        return report.to_dict()

    def generate_report(self, results: Dict[str, float]) -> None:
        """Generate the experiment report.

        Parameters
        ----------
        results : dict
            The evaluation results to include in the report.
        """
        logger.info("Generating report...")
        report_path = self.artifact_manager.get_path("reports", "experiment_report.md")
        
        # Using a simplistic markdown generation for now, ideally hooks into ReportGenerator
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(f"# Experiment Report: {self.config.experiment_name}\n\n")
            f.write(f"## Configuration\nDataset: {self.config.dataset_name}\n\n")
            f.write("## Results\n")
            for k, v in results.items():
                f.write(f"- **{k}**: {v:.4f}\n")
                
        logger.info(f"Report saved to {report_path}")

    def run(self) -> None:
        """Execute the full experiment workflow."""
        logger.info(f"=== Starting Experiment: {self.config.experiment_name} ===")
        
        self.train()
        eval_results = self.evaluate()
        bench_results = self.run_benchmark()
        
        self.generate_report(eval_results)
        
        # Save all tracked results
        self.result_tracker.save_results(summary_metrics=eval_results)
        
        logger.info(f"=== Experiment {self.config.experiment_name} Completed ===")
