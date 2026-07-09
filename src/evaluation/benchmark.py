"""Benchmarking utilities for model evaluation."""

import logging
import os
import time
import tracemalloc
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Union

import torch
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkReport:
    """Structured report containing dataset benchmarking metrics."""
    dataset_name: str
    total_samples: int
    batch_size: int
    total_time_sec: float
    throughput_fps: float
    latency_ms_per_batch: float
    latency_ms_per_sample: float
    cpu_memory_peak_mb: float
    gpu_memory_peak_mb: Optional[float]
    device: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert the report to a dictionary representation."""
        return {
            "dataset_name": self.dataset_name,
            "total_samples": self.total_samples,
            "batch_size": self.batch_size,
            "total_time_sec": self.total_time_sec,
            "throughput_fps": self.throughput_fps,
            "latency_ms_per_batch": self.latency_ms_per_batch,
            "latency_ms_per_sample": self.latency_ms_per_sample,
            "cpu_memory_peak_mb": self.cpu_memory_peak_mb,
            "gpu_memory_peak_mb": self.gpu_memory_peak_mb,
            "device": self.device,
        }


class BenchmarkRunner:
    """Runner for evaluating dataset and model performance metrics."""

    def __init__(self, inference_fn: Callable[[Any], Any], device: Union[str, torch.device] = "cpu") -> None:
        """Initialize the BenchmarkRunner.

        Args:
            inference_fn: A callable that accepts a single batch from the DataLoader
                and performs the forward pass/inference.
            device: The device on which the benchmarking is being performed.
        """
        self.inference_fn = inference_fn
        self.device = str(device)

    def _get_cpu_memory_mb(self) -> float:
        """Get CPU peak memory using psutil if available, fallback to tracemalloc."""
        try:
            import psutil
            process = psutil.Process(os.getpid())
            return process.memory_info().rss / (1024 * 1024)
        except ImportError:
            # Fallback to tracemalloc peak
            _, peak = tracemalloc.get_traced_memory()
            return peak / (1024 * 1024)

    def run(self, dataloader: DataLoader, dataset_name: str = "unknown") -> BenchmarkReport:
        """Run benchmark on the provided dataloader.

        Args:
            dataloader: PyTorch DataLoader containing the evaluation dataset.
            dataset_name: Human-readable name of the dataset for the report.

        Returns:
            A BenchmarkReport containing latency, throughput, and memory metrics.
        """
        logger.info(f"Starting benchmark for dataset: {dataset_name}")

        batch_size = dataloader.batch_size if dataloader.batch_size else 1
        
        # Setup memory tracking
        if not tracemalloc.is_tracing():
            tracemalloc.start()
        tracemalloc.reset_peak()
        
        if torch.cuda.is_available() and "cuda" in self.device:
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.empty_cache()

        total_samples = 0
        total_batches = 0

        start_time = time.perf_counter()

        for batch in dataloader:
            # Determine actual batch size (might be smaller at the end of epoch)
            if isinstance(batch, dict) and "window" in batch:
                actual_batch_size = len(batch["window"])
            elif isinstance(batch, (list, tuple)) and len(batch) > 0:
                actual_batch_size = len(batch[0])
            elif isinstance(batch, torch.Tensor):
                actual_batch_size = len(batch)
            else:
                actual_batch_size = batch_size
                
            self.inference_fn(batch)
            
            total_samples += actual_batch_size
            total_batches += 1

        if torch.cuda.is_available() and "cuda" in self.device:
            torch.cuda.synchronize()
            
        end_time = time.perf_counter()
        
        total_time_sec = end_time - start_time
        
        cpu_memory_peak_mb = self._get_cpu_memory_mb()
        
        gpu_memory_peak_mb = None
        if torch.cuda.is_available() and "cuda" in self.device:
            gpu_memory_peak_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)

        if total_samples == 0:
            logger.warning("No samples were processed during benchmarking.")
            return BenchmarkReport(
                dataset_name=dataset_name,
                total_samples=0,
                batch_size=batch_size,
                total_time_sec=total_time_sec,
                throughput_fps=0.0,
                latency_ms_per_batch=0.0,
                latency_ms_per_sample=0.0,
                cpu_memory_peak_mb=cpu_memory_peak_mb,
                gpu_memory_peak_mb=gpu_memory_peak_mb,
                device=self.device,
            )

        throughput_fps = total_samples / total_time_sec if total_time_sec > 0 else 0.0
        latency_ms_per_batch = (total_time_sec / total_batches) * 1000 if total_batches > 0 else 0.0
        latency_ms_per_sample = (total_time_sec / total_samples) * 1000 if total_samples > 0 else 0.0

        report = BenchmarkReport(
            dataset_name=dataset_name,
            total_samples=total_samples,
            batch_size=batch_size,
            total_time_sec=total_time_sec,
            throughput_fps=throughput_fps,
            latency_ms_per_batch=latency_ms_per_batch,
            latency_ms_per_sample=latency_ms_per_sample,
            cpu_memory_peak_mb=cpu_memory_peak_mb,
            gpu_memory_peak_mb=gpu_memory_peak_mb,
            device=self.device,
        )
        
        logger.info(f"Benchmark completed for {dataset_name}: {total_samples} samples in {total_time_sec:.4f}s")
        return report
