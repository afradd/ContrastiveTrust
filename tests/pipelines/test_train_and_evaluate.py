import tempfile
import pytest
import pandas as pd
import numpy as np
import sys
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.pipelines.train_and_evaluate import (
    load_data, build_model, train, build_detector, evaluate
)

@pytest.fixture
def mock_data_paths():
    """Generates synthetic SWaT and HAI data, saving to temporary CSVs."""
    np.random.seed(42)
    
    # Generate SWaT-like data
    swat_dates = pd.date_range("2019-12-01", periods=100, freq="s")
    swat_df = pd.DataFrame({
        "t_stamp": swat_dates,
        "FIT101": np.random.randn(100),
        "LIT101": np.random.randn(100),
        "PIT101": np.random.randn(100),
        "MV101": np.random.randint(0, 2, 100),
        "P1_STATE": np.random.randint(0, 2, 100),
    })

    # Generate HAI-like data
    hai_dates = pd.date_range("2019-12-01", periods=50, freq="s")
    hai_df = pd.DataFrame({
        "timestamp": hai_dates,
        "FIT101": np.random.randn(50),
        "LIT101": np.random.randn(50),
        "PIT101": np.random.randn(50),
        "MV101": np.random.randint(0, 2, 50),
        "label": np.random.randint(0, 2, 50)
    })

    with tempfile.TemporaryDirectory() as temp_dir:
        swat_path = Path(temp_dir) / "swat.csv"
        hai_path = Path(temp_dir) / "hai.csv"
        
        swat_df.to_csv(swat_path, index=False)
        hai_df.to_csv(hai_path, index=False)
        
        yield str(swat_path), str(hai_path)

def test_pipeline_end_to_end(mock_data_paths):
    swat_path, hai_path = mock_data_paths
    
    # 1. Load Data
    (train_loader, val_loader, normal_loader, 
     val_score_loader, swat_test_loader, hai_loader_eval) = load_data(
        swat_path=swat_path,
        hai_path=hai_path,
        swat_meta_path="",
        window_size=4,
        stride=2,
        batch_size=4
    )
    
    assert len(train_loader) > 0
    assert len(val_loader) > 0
    
    # 2. Build Model
    encoder, projection_head, loss_fn = build_model(device="cpu")
    
    # 3. Train
    with tempfile.TemporaryDirectory() as temp_dir:
        log_dir = Path(temp_dir)
        history = train(
            encoder=encoder,
            projection_head=projection_head,
            loss_fn=loss_fn,
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=1,
            log_dir=log_dir,
            device="cpu"
        )
        assert len(history) == 1
        
        # 4. Build Detector
        detector = build_detector(
            encoder=encoder,
            normal_loader=normal_loader,
            val_score_loader=val_score_loader,
            device="cpu"
        )
        assert detector.threshold_estimator.predict_threshold() > 0
        
        # 5. Evaluate (mock plt.savefig to prevent plotting errors in headless environment if any)
        with mock.patch("matplotlib.pyplot.savefig"):
            evaluate(
                detector=detector,
                hai_loader_eval=hai_loader_eval,
                swat_test_loader=swat_test_loader,
                history=history,
                out_dir=log_dir,
                device="cpu"
            )
        
        # Assert outputs were generated
        assert (log_dir / "tables" / "training_history.json").exists()
        assert (log_dir / "tables" / "hai_zero_shot_metrics.json").exists()
        assert (log_dir / "tables" / "metrics.csv").exists()
