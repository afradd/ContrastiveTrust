import json
import math
import pytest
import numpy as np
from typing import Dict

from src.evaluation.metrics import EvaluationMetrics

@pytest.fixture
def evaluator():
    return EvaluationMetrics()

def test_numerical_correctness_binary_labels(evaluator):
    """Test every metric for numerical correctness using standard binary labels."""
    y_true = [0, 0, 0, 1, 1, 1]
    y_score = [0.1, 0.4, 0.35, 0.8, 0.45, 0.9]
    # threshold = 0.5 -> y_pred = [0, 0, 0, 1, 0, 1]
    
    metrics = evaluator.compute(y_true, y_score, threshold=0.5, inference_times=[0.1, 0.1, 0.2, 0.1, 0.1, 0.1])
    
    # Classification & CM
    assert metrics["tp"] == 2
    assert metrics["tn"] == 3
    assert metrics["fp"] == 0
    assert metrics["fn"] == 1
    
    # accuracy: (2+3)/6 = 5/6 = 0.8333...
    assert math.isclose(metrics["accuracy"], 5/6)
    # precision: TP / (TP+FP) = 2 / 2 = 1.0
    assert math.isclose(metrics["precision"], 1.0)
    # recall: TP / (TP+FN) = 2 / 3 = 0.6666...
    assert math.isclose(metrics["recall"], 2/3)
    # f1_score: 2*(1*2/3)/(1+2/3) = 0.8
    assert math.isclose(metrics["f1_score"], 0.8)
    
    # balanced_accuracy: (recall_pos + recall_neg)/2 = (2/3 + 3/3)/2 = 5/6
    assert math.isclose(metrics["balanced_accuracy"], 5/6)
    
    # Ranking metrics
    # roc_auc for these scores...
    # y_true = [0, 0, 0, 1, 1, 1]
    # y_score= [0.1, 0.4, 0.35, 0.8, 0.45, 0.9]
    # pairs: (0.1,0.8), (0.1,0.45), (0.1,0.9) - all 1s > 0s (3 pairs)
    # (0.4,0.8), (0.4,0.9), (0.4,0.45) - all 1s > 0s (3 pairs)
    # (0.35,0.8), (0.35,0.45), (0.35,0.9) - all 1s > 0s (3 pairs)
    # Wait, all anomalies have higher scores than all normals!
    # Let's check: max(normals) = 0.4. min(anomalies) = 0.45. Yes.
    # Therefore ROC-AUC should be 1.0.
    assert math.isclose(metrics["roc_auc"], 1.0)
    # Average precision should also be 1.0
    assert math.isclose(metrics["average_precision"], 1.0)
    
    # specificty: TN / (TN+FP) = 3 / 3 = 1.0
    assert math.isclose(metrics["specificity"], 1.0)
    # sensitivity: TP / (TP+FN) = 2 / 3
    assert math.isclose(metrics["sensitivity"], 2/3)
    # false_alarm_rate: FP / (FP+TN) = 0.0
    assert math.isclose(metrics["false_alarm_rate"], 0.0)
    # miss_rate: FN / (FN+TP) = 1/3
    assert math.isclose(metrics["miss_rate"], 1/3)
    # detection_rate: same as sensitivity = 2/3
    assert math.isclose(metrics["detection_rate"], 2/3)
    
    # Timing
    assert math.isclose(metrics["avg_inference_latency"], 7/60) # sum is 0.7, avg is 0.7/6 = 7/60
    assert math.isclose(metrics["throughput"], 60/7)

def test_all_normal_dataset(evaluator):
    """Test with all-normal dataset."""
    y_true = [0, 0, 0, 0]
    y_score = [0.1, 0.2, 0.1, 0.3]
    
    metrics = evaluator.compute(y_true, y_score, threshold=0.5)
    
    # ROC-AUC and PR-AUC should be NaN
    assert math.isnan(metrics["roc_auc"])
    assert math.isnan(metrics["average_precision"])
    assert math.isnan(metrics["pr_auc"])
    
    # accuracy should be 1.0
    assert metrics["accuracy"] == 1.0
    assert metrics["fp"] == 0
    assert metrics["tn"] == 4
    
def test_all_anomaly_dataset(evaluator):
    """Test with all-anomaly dataset."""
    y_true = [1, 1, 1, 1]
    y_score = [0.8, 0.9, 0.95, 0.7]
    
    metrics = evaluator.compute(y_true, y_score, threshold=0.5)
    
    assert math.isnan(metrics["roc_auc"])
    assert math.isnan(metrics["average_precision"])
    assert math.isnan(metrics["pr_auc"])
    
    assert metrics["accuracy"] == 1.0
    assert metrics["tp"] == 4

def test_empty_inputs(evaluator):
    with pytest.raises(ValueError, match="Input arrays must not be empty"):
        evaluator.compute([], [])

def test_invalid_shapes(evaluator):
    with pytest.raises(ValueError, match="Shape mismatch"):
        evaluator.compute([0, 1], [0.1, 0.2, 0.3])
        
    with pytest.raises(ValueError, match="Shape mismatch"):
        evaluator.compute([0, 1], [0.1, 0.2], y_pred=[1])
        
    with pytest.raises(ValueError, match="Inputs must be 1D arrays"):
        evaluator.compute(np.array([[0], [1]]), np.array([[0.1], [0.2]]))

def test_deterministic_output(evaluator):
    y_true = [0, 1, 0, 1]
    y_score = [0.1, 0.9, 0.3, 0.8]
    
    metrics1 = evaluator.compute(y_true, y_score, threshold=0.5)
    metrics2 = evaluator.compute(y_true, y_score, threshold=0.5)
    
    for k in metrics1:
        if math.isnan(metrics1[k]):
            assert math.isnan(metrics2[k])
        else:
            assert metrics1[k] == metrics2[k]

def test_serialization(evaluator):
    y_true = [0, 1, 0, 1]
    y_score = [0.1, 0.9, 0.3, 0.8]
    
    evaluator.compute(y_true, y_score, threshold=0.5)
    d = evaluator.to_dict()
    
    # To handle NaNs during serialization if any, though none here
    json_str = json.dumps(d)
    d2 = json.loads(json_str)
    
    assert d2["accuracy"] == 1.0
    assert d2["tp"] == 2

def test_compute_batch(evaluator):
    batches = [
        {"y_true": [0, 1], "y_score": [0.1, 0.9], "inference_times": [0.01, 0.02]},
        {"y_true": [0, 0], "y_score": [0.3, 0.2], "inference_times": [0.01, 0.01]}
    ]
    
    metrics = evaluator.compute_batch(batches)
    assert not math.isnan(metrics["roc_auc"])
    assert math.isclose(metrics["avg_inference_latency"], 0.0125)

def test_summary_and_to_dict(evaluator):
    assert evaluator.summary() == "No metrics computed yet."
    assert evaluator.to_dict() == {}
    
    evaluator.compute([0, 1], [0.1, 0.9], threshold=0.5)
    
    summary = evaluator.summary()
    assert "Evaluation Metrics Summary" in summary
    assert "accuracy" in summary
    
    d = evaluator.to_dict()
    assert d["accuracy"] == 1.0
