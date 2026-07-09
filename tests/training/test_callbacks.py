import os
import tempfile
import pytest
import torch

from src.training.callbacks import Callback
from src.training.early_stopping import EarlyStopping
from src.training.checkpoint import ModelCheckpoint

class MockTrainer:
    def __init__(self):
        self.should_stop = False
        self.encoder = torch.nn.Linear(10, 2)
        self.projection_head = torch.nn.Linear(2, 2)
        self.optimizer = torch.optim.SGD(self.encoder.parameters(), lr=0.1)
        self.scheduler = None

def test_early_stopping():
    trainer = MockTrainer()
    es = EarlyStopping(monitor="val_loss", patience=2, mode="min")
    
    # Epoch 1
    es.on_epoch_end(trainer, 1, {"val_loss": 1.0})
    assert not trainer.should_stop
    assert es.best_score == 1.0
    
    # Epoch 2 (No improvement)
    es.on_epoch_end(trainer, 2, {"val_loss": 1.0})
    assert not trainer.should_stop
    assert es.wait == 1
    
    # Epoch 3 (No improvement -> Stop)
    es.on_epoch_end(trainer, 3, {"val_loss": 1.5})
    assert trainer.should_stop
    assert es.wait == 2
    assert es.stopped_epoch == 3

def test_early_stopping_max():
    trainer = MockTrainer()
    es = EarlyStopping(monitor="val_loss", patience=2, mode="max")
    
    es.on_epoch_end(trainer, 1, {"val_loss": 1.0})
    assert es.best_score == 1.0
    
    es.on_epoch_end(trainer, 2, {"val_loss": 1.5})
    assert es.best_score == 1.5
    assert es.wait == 0
    
    es.on_epoch_end(trainer, 3, {"val_loss": 1.0})
    assert not trainer.should_stop

def test_model_checkpoint():
    with tempfile.TemporaryDirectory() as tmpdir:
        filepath = os.path.join(tmpdir, "model.pt")
        trainer = MockTrainer()
        mc = ModelCheckpoint(filepath=filepath, monitor="val_loss", mode="min")
        
        # Epoch 1
        mc.on_epoch_end(trainer, 1, {"val_loss": 1.0})
        assert os.path.exists(filepath)
        assert mc.best_score == 1.0
        
        # Epoch 2 (Improvement)
        mc.on_epoch_end(trainer, 2, {"val_loss": 0.5})
        assert mc.best_score == 0.5
        
        # Check saved state
        state = torch.load(filepath, weights_only=False)
        assert state["epoch"] == 2
        assert "encoder_state_dict" in state

def test_model_checkpoint_max():
    with tempfile.TemporaryDirectory() as tmpdir:
        filepath = os.path.join(tmpdir, "model.pt")
        trainer = MockTrainer()
        mc = ModelCheckpoint(filepath=filepath, monitor="val_acc", mode="max")
        
        mc.on_epoch_end(trainer, 1, {"val_acc": 0.5})
        assert mc.best_score == 0.5
        
        mc.on_epoch_end(trainer, 2, {"val_acc": 0.8})
        assert mc.best_score == 0.8
        
        state = torch.load(filepath, weights_only=False)
        assert state["epoch"] == 2
