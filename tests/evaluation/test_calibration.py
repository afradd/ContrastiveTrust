"""Tests for the calibration module."""

import numpy as np
import pytest

from src.evaluation.calibration import ScoreCalibrator


def test_min_max_calibration():
    scores = np.array([0.0, 5.0, 10.0])
    calibrator = ScoreCalibrator(method="min-max")
    
    probs = calibrator.fit_transform(scores)
    
    assert calibrator.is_fitted
    assert np.allclose(probs, [0.0, 0.5, 1.0])
    
    # Test clipping
    out_of_bounds = np.array([-5.0, 15.0])
    clipped_probs = calibrator.transform(out_of_bounds)
    assert np.allclose(clipped_probs, [0.0, 1.0])


def test_min_max_zero_variance():
    scores = np.array([5.0, 5.0, 5.0])
    calibrator = ScoreCalibrator(method="min-max")
    probs = calibrator.fit_transform(scores)
    
    assert np.allclose(probs, [0.0, 0.0, 0.0])


def test_z_score_calibration():
    scores = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    calibrator = ScoreCalibrator(method="z-score")
    
    probs = calibrator.fit_transform(scores)
    
    assert calibrator.is_fitted
    assert probs.shape == (5,)
    assert 0.0 <= probs.min() <= probs.max() <= 1.0
    # Center should be close to 0.5 for a symmetric distribution
    assert np.isclose(probs[2], 0.5)


def test_logistic_calibration():
    scores = np.array([0.1, 0.2, 0.8, 0.9])
    labels = np.array([0, 0, 1, 1])
    
    calibrator = ScoreCalibrator(method="logistic")
    probs = calibrator.fit_transform(scores, labels=labels)
    
    assert calibrator.is_fitted
    assert probs.shape == (4,)
    assert 0.0 <= probs.min() <= probs.max() <= 1.0
    # High scores should have higher probabilities
    assert probs[0] < probs[3]


def test_logistic_requires_labels():
    scores = np.array([0.1, 0.2, 0.8, 0.9])
    calibrator = ScoreCalibrator(method="logistic")
    
    with pytest.raises(ValueError, match="Labels are required"):
        calibrator.fit(scores)


def test_temperature_scaling():
    scores = np.array([1.0, 2.0, 3.0])
    calibrator = ScoreCalibrator(method="temperature", temperature=2.0)
    
    probs = calibrator.fit_transform(scores)
    
    assert calibrator.is_fitted
    assert probs.shape == (3,)
    assert 0.0 <= probs.min() <= probs.max() <= 1.0


def test_invalid_method():
    with pytest.raises(ValueError, match="Unknown calibration method"):
        calibrator = ScoreCalibrator(method="invalid_method")
        calibrator.fit(np.array([1.0, 2.0]))


def test_nan_inf_handling():
    scores = np.array([1.0, np.nan, np.inf, -np.inf, 5.0])
    calibrator = ScoreCalibrator(method="min-max")
    
    probs = calibrator.fit_transform(scores)
    
    # NaN and Inf are replaced with 0.0
    # The array becomes [1.0, 0.0, 0.0, 0.0, 5.0]
    # Min is 0.0, Max is 5.0
    expected = [1.0/5.0, 0.0, 0.0, 0.0, 1.0]
    assert np.allclose(probs, expected)


def test_transform_before_fit():
    calibrator = ScoreCalibrator(method="min-max")
    with pytest.raises(RuntimeError, match="Calibrator must be fitted"):
        calibrator.transform(np.array([1.0]))
