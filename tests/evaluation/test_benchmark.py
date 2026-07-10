"""Tests for the benchmark runner."""

import time
import pytest
import torch
from torch.utils.data import DataLoader, Dataset
from src.evaluation.benchmark import BenchmarkRunner, BenchmarkReport

class DummyDictDataset(Dataset):
    def __init__(self, size=20):
        self.size = size

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return {"window": torch.randn(5, 3), "physics_features": torch.randn(2)}

class DummyTensorDataset(Dataset):
    def __len__(self):
        return 10

    def __getitem__(self, idx):
        return torch.randn(5)


def test_benchmark_runner_dict_batch():
    """Test benchmark runner with a DataLoader that yields dictionaries."""
    dataset = DummyDictDataset(size=20)
    dataloader = DataLoader(dataset, batch_size=5)

    def dummy_inference(batch):
        # Simulate some processing time
        time.sleep(0.01)

    runner = BenchmarkRunner(inference_fn=dummy_inference, device="cpu")
    report = runner.run(dataloader, dataset_name="TestDict")

    assert report.dataset_name == "TestDict"
    assert report.total_samples == 20
    assert report.batch_size == 5
    assert report.total_time_sec > 0
    assert report.throughput_fps > 0
    assert report.latency_ms_per_batch > 0
    assert report.latency_ms_per_sample > 0
    
    # We should have captured some CPU memory footprint
    assert report.cpu_memory_peak_mb >= 0
    
    # Not using CUDA device here, should be None
    assert report.gpu_memory_peak_mb is None

    report_dict = report.to_dict()
    assert report_dict["dataset_name"] == "TestDict"
    assert report_dict["total_samples"] == 20


def test_benchmark_runner_tensor_batch():
    """Test benchmark runner with a DataLoader that yields raw Tensors."""
    dataloader = DataLoader(DummyTensorDataset(), batch_size=2)
    
    def dummy_inference(batch):
        pass

    runner = BenchmarkRunner(inference_fn=dummy_inference, device="cpu")
    report = runner.run(dataloader, dataset_name="TestTensor")

    assert report.dataset_name == "TestTensor"
    assert report.total_samples == 10
    assert report.batch_size == 2


def test_benchmark_runner_empty():
    """Test benchmark runner with an empty dataloader."""
    class EmptyDataset(Dataset):
        def __len__(self):
            return 0
        def __getitem__(self, idx):
            raise IndexError

    dataloader = DataLoader(EmptyDataset(), batch_size=2)
    runner = BenchmarkRunner(inference_fn=lambda x: None, device="cpu")
    report = runner.run(dataloader, dataset_name="Empty")

    assert report.total_samples == 0
    assert report.total_time_sec >= 0
    assert report.throughput_fps == 0.0
    assert report.latency_ms_per_batch == 0.0
    assert report.latency_ms_per_sample == 0.0


def test_benchmark_runner_cuda_memory():
    """Test benchmark runner handles CUDA memory when specified."""
    dataset = DummyDictDataset(size=5)
    dataloader = DataLoader(dataset, batch_size=5)
    
    # We test the logic even if CUDA is not available on the test machine
    runner = BenchmarkRunner(inference_fn=lambda x: None, device="cuda")
    report = runner.run(dataloader, dataset_name="CUDA_Test")
    
    if torch.cuda.is_available():
        assert report.gpu_memory_peak_mb is not None
        assert report.gpu_memory_peak_mb >= 0
    else:
        assert report.gpu_memory_peak_mb is None


def test_benchmark_runner_hai_swat_datasets():
    """Test benchmark runner handles explicitly required IEEE datasets correctly."""
    dataset = DummyDictDataset(size=5)
    dataloader = DataLoader(dataset, batch_size=5)
    runner = BenchmarkRunner(inference_fn=lambda x: None, device="cpu")
    
    report_hai = runner.run(dataloader, dataset_name="HAI")
    assert report_hai.dataset_name == "HAI"
    
    report_swat = runner.run(dataloader, dataset_name="SWaT")
    assert report_swat.dataset_name == "SWaT"
