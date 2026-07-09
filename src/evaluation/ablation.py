"""Ablation evaluation framework for systematic component analysis."""

import csv
import json
import logging
from dataclasses import asdict, dataclass
from typing import Any, Callable, Dict, List
import itertools

from torch.utils.data import DataLoader

from src.evaluation.evaluator import Evaluator

logger = logging.getLogger(__name__)


@dataclass
class AblationConfig:
    """Configuration for a single ablation experiment.

    Attributes
    ----------
    name : str
        Name or identifier of this specific configuration.
    use_physics_encoder : bool
        Whether the physics encoder is enabled.
    use_fusion_module : bool
        Whether the fusion module is enabled.
    use_projection_head : bool
        Whether the projection head is used during training/evaluation.
    use_physics_consistency_loss : bool
        Whether the physics consistency loss was used during training.
    distance_metric : str
        The distance metric to use (e.g., 'cosine', 'euclidean').
    threshold_method : str
        The threshold method to use (e.g., 'otsu', 'static').
    """
    name: str
    use_physics_encoder: bool = True
    use_fusion_module: bool = True
    use_projection_head: bool = True
    use_physics_consistency_loss: bool = True
    distance_metric: str = "cosine"
    threshold_method: str = "otsu"


class AblationStudy:
    """Orchestrates systematic evaluation of model components.

    Automatically compares multiple `AblationConfig` setups by
    evaluating them over a dataset and generating reports.
    """

    def __init__(self, configs: List[AblationConfig]):
        """Initialize the AblationStudy.

        Parameters
        ----------
        configs : list of AblationConfig
            The specific experimental setups to evaluate.
        """
        self.configs = configs
        self.results: Dict[str, Dict[str, Any]] = {}

    @classmethod
    def generate_grid(
        cls, base_config: AblationConfig, search_space: Dict[str, List[Any]]
    ) -> List[AblationConfig]:
        """Generate a grid of ablation configurations.

        Parameters
        ----------
        base_config : AblationConfig
            The baseline configuration.
        search_space : dict
            A dictionary mapping attribute names to lists of possible values.
            For example: `{'use_physics_encoder': [True, False]}`.

        Returns
        -------
        list of AblationConfig
            All permutations of the search space applied to the base config.
        """
        keys = list(search_space.keys())
        values_lists = [search_space[k] for k in keys]
        
        permutations = list(itertools.product(*values_lists))
        
        configs = []
        for _, permutation in enumerate(permutations):
            overrides = dict(zip(keys, permutation))
            
            # Construct a descriptive name
            name_parts = []
            for k, v in overrides.items():
                name_parts.append(f"{k}={v}")
            new_name = base_config.name + "_" + "_".join(name_parts)
            
            # Create the new config by converting base to dict, updating, and re-instantiating
            base_dict = asdict(base_config)
            base_dict.update(overrides)
            base_dict["name"] = new_name
            
            configs.append(AblationConfig(**base_dict))
            
        return configs

    def run(
        self,
        evaluator_factory: Callable[[AblationConfig], Evaluator],
        loader: DataLoader,
    ) -> None:
        """Run the ablation study across all configured variations.

        Parameters
        ----------
        evaluator_factory : Callable[[AblationConfig], Evaluator]
            A user-provided factory function that accepts an `AblationConfig`
            and returns an instantiated, loaded `Evaluator` ready to process data.
        loader : DataLoader
            The evaluation dataset.
        """
        self.results = {}
        for idx, config in enumerate(self.configs):
            logger.info(
                f"Running ablation config {idx + 1}/{len(self.configs)}: {config.name}"
            )
            try:
                evaluator = evaluator_factory(config)
                metrics = evaluator.evaluate_loader(loader)
                
                self.results[config.name] = {
                    "config": asdict(config),
                    "metrics": metrics,
                }
            except Exception as e:
                logger.error(f"Failed to evaluate config {config.name}: {e}")
                self.results[config.name] = {
                    "config": asdict(config),
                    "error": str(e),
                }

    def summary(self) -> str:
        """Generate a formatted markdown table summarizing the results."""
        if not self.results:
            return "No results available. Call `run()` first."

        headers = [
            "Configuration Name",
            "F1-Score",
            "ROC-AUC",
            "PR-AUC",
            "Detection Rate",
        ]
        
        rows = []
        for name, data in self.results.items():
            if "error" in data:
                rows.append(f"| {name} | ERROR | ERROR | ERROR | ERROR |")
                continue
            
            m = data.get("metrics", {})
            f1 = m.get("f1_score", float('nan'))
            roc = m.get("roc_auc", float('nan'))
            pr = m.get("pr_auc", float('nan'))
            dr = m.get("detection_rate", float('nan'))
            
            rows.append(f"| {name} | {f1:.4f} | {roc:.4f} | {pr:.4f} | {dr:.4f} |")
            
        header_str = "| " + " | ".join(headers) + " |"
        sep_str = "| " + " | ".join(["---"] * len(headers)) + " |"
        
        return "\n".join([header_str, sep_str] + rows)

    def export_results(self, json_path: str, csv_path: str) -> None:
        """Export the ablation results to JSON and CSV formats.

        Parameters
        ----------
        json_path : str
            Path to write the JSON results.
        csv_path : str
            Path to write the CSV summary.
        """
        # 1. Export JSON
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(self.results, f, indent=4)
            
        # 2. Export CSV
        if not self.results:
            return
            
        # Dynamically extract all config keys and metric keys from the first successful result
        config_keys = list(asdict(AblationConfig(name="dummy")).keys())
        metric_keys = ["f1_score", "roc_auc", "pr_auc", "precision", "recall", "detection_rate"]
        
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            header = config_keys + metric_keys + ["error"]
            writer.writerow(header)
            
            for _, data in self.results.items():
                row = []
                c = data.get("config", {})
                for k in config_keys:
                    row.append(c.get(k, ""))
                
                m = data.get("metrics", {})
                for k in metric_keys:
                    row.append(m.get(k, ""))
                    
                row.append(data.get("error", ""))
                writer.writerow(row)
        
        logger.info(f"Exported ablation results to {json_path} and {csv_path}")
