import os
import tempfile
from unittest import mock

import numpy as np
import pytest
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.evaluation.visualization import (
    AblationVisualizer,
    BenchmarkVisualizer,
    ConfusionMatrixVisualizer,
    DataExporter,
    DistributionVisualizer,
    EmbeddingVisualizer,
    PRCurveVisualizer,
    ROCVisualizer,
    TrainingVisualizer,
    style_manager,
)


@pytest.fixture(autouse=True)
def clean_up_figures():
    """Ensure all figures are closed after each test."""
    yield
    plt.close("all")


@pytest.fixture
def temp_dir():
    with tempfile.TemporaryDirectory() as d:
        yield d


def test_roc_visualizer(temp_dir):
    y_true = [0, 0, 1, 1]
    y_score = [0.1, 0.4, 0.35, 0.8]
    
    vis = ROCVisualizer().style("default")
    vis.plot(y_true, y_score)
    
    filepath = os.path.join(temp_dir, "roc.pdf")
    vis.save(filepath, format="pdf", dpi=300)
    
    assert os.path.exists(filepath)
    vis.close()


def test_roc_visualizer_invalid_inputs():
    vis = ROCVisualizer()
    with pytest.raises(ValueError):
        vis.plot([], [])
        
    with pytest.raises(ValueError):
        vis.plot([0, 0], [0.1, 0.2])  # No positive class


def test_pr_curve_visualizer(temp_dir):
    y_true = [0, 1, 1, 0]
    y_score = [0.2, 0.8, 0.6, 0.4]
    
    vis = PRCurveVisualizer()
    vis.plot(y_true, y_score)
    
    filepath = os.path.join(temp_dir, "pr.png")
    vis.save(filepath, format="png", dpi=600)
    
    assert os.path.exists(filepath)
    vis.close()


def test_confusion_matrix_visualizer(temp_dir):
    y_true = [0, 1, 0, 1, 0, 1]
    y_pred = [0, 0, 0, 1, 1, 1]
    
    vis = ConfusionMatrixVisualizer()
    vis.plot(y_true, y_pred, normalize=True)
    
    filepath = os.path.join(temp_dir, "cm.svg")
    vis.save(filepath)
    
    assert os.path.exists(filepath)
    vis.close()


def test_embedding_visualizer_pca(temp_dir):
    embeddings = np.random.rand(20, 10)
    labels = np.random.randint(0, 2, 20)
    
    vis = EmbeddingVisualizer()
    vis.plot(embeddings, labels, method="pca")
    
    filepath = os.path.join(temp_dir, "embed.pdf")
    vis.save(filepath)
    assert os.path.exists(filepath)
    vis.close()


@mock.patch("src.evaluation.visualization.embeddings.TSNE")
def test_embedding_visualizer_tsne(mock_tsne, temp_dir):
    mock_tsne_instance = mock.Mock()
    mock_tsne_instance.fit_transform.return_value = np.random.rand(10, 2)
    mock_tsne.return_value = mock_tsne_instance
    
    embeddings = np.random.rand(10, 10)
    
    vis = EmbeddingVisualizer()
    vis.plot(embeddings, method="tsne")
    
    mock_tsne.assert_called_once()
    mock_tsne_instance.fit_transform.assert_called_once()


def test_training_visualizer(temp_dir):
    metrics = {
        "Train Loss": [0.9, 0.5, 0.2, 0.1],
        "Val Loss": [1.0, 0.6, 0.3, 0.2]
    }
    
    vis = TrainingVisualizer()
    vis.plot(metrics, log_scale=True)
    
    filepath = os.path.join(temp_dir, "training.eps")
    vis.save(filepath)
    assert os.path.exists(filepath)
    vis.close()


def test_distribution_visualizer(temp_dir):
    scores = np.random.randn(100)
    labels = (scores > 0).astype(int)
    
    vis = DistributionVisualizer()
    vis.plot(scores, labels, threshold=0.5)
    
    filepath = os.path.join(temp_dir, "dist.pdf")
    vis.save(filepath)
    assert os.path.exists(filepath)
    vis.close()


def test_ablation_visualizer(temp_dir):
    results = {
        "ConfigA": {"f1": 0.8, "auc": 0.9},
        "ConfigB": {"f1": 0.85, "auc": 0.92}
    }
    
    vis = AblationVisualizer()
    vis.plot(results, metric_keys=["f1", "auc"])
    
    filepath = os.path.join(temp_dir, "ablation.png")
    vis.save(filepath)
    assert os.path.exists(filepath)
    vis.close()


def test_benchmark_visualizer(temp_dir):
    results = {
        "Latency": 10.5,
        "Throughput": 95.0
    }
    
    vis = BenchmarkVisualizer()
    vis.plot(results)
    
    filepath = os.path.join(temp_dir, "bench.pdf")
    vis.save(filepath)
    assert os.path.exists(filepath)
    vis.close()


def test_data_exporter(temp_dir):
    data = [{"name": "test1", "val": 1}, {"name": "test2", "val": 2}]
    
    csv_path = os.path.join(temp_dir, "out.csv")
    json_path = os.path.join(temp_dir, "out.json")
    
    DataExporter.export_csv(csv_path, data)
    assert os.path.exists(csv_path)
    
    DataExporter.export_json(json_path, data)
    assert os.path.exists(json_path)


def test_style_manager_fallback():
    # Test that style manager doesn't crash when falling back
    with mock.patch("src.evaluation.visualization.styles.VisualizationStyle._is_scienceplots_available", return_value=False):
        style_manager.apply_style("ieee", figsize=(5, 5), dpi=100, font_family="Arial", font_size=12, line_width=2.0)
        assert plt.rcParams["figure.dpi"] == 100
        assert plt.rcParams["font.size"] == 12.0
