"""Tests for the reporting module."""

import csv
import json
import os
from pathlib import Path

import pytest

from src.evaluation.reporting import ReportGenerator


@pytest.fixture
def sample_metrics():
    return {
        "f1_score": 0.95,
        "roc_auc": 0.98,
        "detection_rate": 0.92,
        "false_alarm_rate": 0.01,
        "tp": 920,
        "fp": 10
    }


@pytest.fixture
def sample_metadata():
    return {
        "dataset": "SWaT",
        "model": "ContrastiveTrust",
        "threshold_method": "otsu"
    }


def test_json_export(tmp_path, sample_metrics, sample_metadata):
    reporter = ReportGenerator(metrics=sample_metrics, metadata=sample_metadata)
    out_file = tmp_path / "report.json"
    
    reporter.to_json(str(out_file))
    
    assert out_file.exists()
    
    with open(out_file, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    assert "timestamp" in data
    assert data["metadata"]["dataset"] == "SWaT"
    assert data["metrics"]["f1_score"] == 0.95


def test_csv_export(tmp_path, sample_metrics, sample_metadata):
    reporter = ReportGenerator(metrics=sample_metrics, metadata=sample_metadata)
    out_file = tmp_path / "report.csv"
    
    # Export first time (creates header)
    reporter.to_csv(str(out_file))
    
    assert out_file.exists()
    
    with open(out_file, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        
    assert len(rows) == 1
    assert rows[0]["meta_dataset"] == "SWaT"
    assert float(rows[0]["f1_score"]) == 0.95

    # Export second time (appends to file)
    reporter.to_csv(str(out_file))
    
    with open(out_file, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        
    assert len(rows) == 2


def test_pdf_export(tmp_path, sample_metrics, sample_metadata):
    reporter = ReportGenerator(metrics=sample_metrics, metadata=sample_metadata)
    out_file = tmp_path / "report.pdf"
    
    reporter.to_pdf(str(out_file), title="Test Report")
    
    assert out_file.exists()
    assert out_file.stat().st_size > 0


def test_export_all(tmp_path, sample_metrics):
    reporter = ReportGenerator(metrics=sample_metrics)
    base_path = tmp_path / "full_report"
    
    reporter.export_all(str(base_path), title="Full Report")
    
    assert (tmp_path / "full_report.json").exists()
    assert (tmp_path / "full_report.csv").exists()
    assert (tmp_path / "full_report.pdf").exists()


def test_nan_inf_handling(tmp_path):
    metrics = {
        "valid": 1.0,
        "nan_val": float("nan"),
        "inf_val": float("inf")
    }
    
    reporter = ReportGenerator(metrics=metrics)
    out_file = tmp_path / "nan_report.json"
    
    reporter.to_json(str(out_file))
    
    with open(out_file, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    # JSON cannot natively encode NaN/Inf if strict, but Python's json does it by default.
    # We converted it to string in ReportGenerator to be safe and clean.
    assert data["metrics"]["valid"] == 1.0
    assert data["metrics"]["nan_val"] == "nan"
    assert data["metrics"]["inf_val"] == "inf"
