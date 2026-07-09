import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from src.training.trainer import Trainer
from src.models.encoder import DualStreamEncoder, EncoderConfig
from src.models.temporal_encoder import TemporalEncoderConfig
from src.models.physics_encoder import PhysicsEncoderConfig
from src.models.projection_head import ProjectionHead, ProjectionHeadConfig
from src.losses.contrastive_trust_loss import ContrastiveTrustLoss, ContrastiveTrustLossConfig

class DummyDataset(Dataset):
    def __len__(self):
        return 10
    def __getitem__(self, idx):
        return {
            "view1_window": torch.randn(100, 10),
            "view1_physics": torch.randn(18),
            "view2_window": torch.randn(100, 10),
            "view2_physics": torch.randn(18),
        }

@pytest.fixture
def dummy_components():
    encoder_config = EncoderConfig(
        temporal=TemporalEncoderConfig(input_channels=10),
        physics=PhysicsEncoderConfig(input_dim=18)
    )
    encoder = DualStreamEncoder(encoder_config)
    
    proj_config = ProjectionHeadConfig(input_dim=encoder_config.temporal.embedding_dim)
    projection_head = ProjectionHead(proj_config)
    
    loss_config = ContrastiveTrustLossConfig()
    loss_fn = ContrastiveTrustLoss(loss_config)
    
    optimizer = torch.optim.Adam(
        list(encoder.parameters()) + list(projection_head.parameters()), 
        lr=1e-3
    )
    return encoder, projection_head, loss_fn, optimizer

def test_trainer_init(dummy_components):
    encoder, projection_head, loss_fn, optimizer = dummy_components
    trainer = Trainer(
        encoder=encoder,
        projection_head=projection_head,
        loss_fn=loss_fn,
        optimizer=optimizer,
        device="cpu",
        mixed_precision=False
    )
    assert trainer.device.type == "cpu"
    assert trainer.amp.enabled is False

def test_trainer_train_epoch(dummy_components):
    encoder, projection_head, loss_fn, optimizer = dummy_components
    trainer = Trainer(
        encoder=encoder,
        projection_head=projection_head,
        loss_fn=loss_fn,
        optimizer=optimizer,
        device="cpu",
        mixed_precision=False,
        gradient_accumulation_steps=2
    )
    dataset = DummyDataset()
    dataloader = DataLoader(dataset, batch_size=2)
    metrics = trainer.train_epoch(dataloader)
    
    assert "train_loss" in metrics
    assert isinstance(metrics["train_loss"], float)

def test_trainer_validate_epoch(dummy_components):
    encoder, projection_head, loss_fn, optimizer = dummy_components
    trainer = Trainer(
        encoder=encoder,
        projection_head=projection_head,
        loss_fn=loss_fn,
        optimizer=optimizer,
        device="cpu",
        mixed_precision=False
    )
    dataset = DummyDataset()
    dataloader = DataLoader(dataset, batch_size=2)
    metrics = trainer.validate_epoch(dataloader)
    
    assert "val_loss" in metrics
    assert isinstance(metrics["val_loss"], float)

def test_trainer_fit(dummy_components):
    encoder, projection_head, loss_fn, optimizer = dummy_components
    trainer = Trainer(
        encoder=encoder,
        projection_head=projection_head,
        loss_fn=loss_fn,
        optimizer=optimizer,
        device="cpu",
        mixed_precision=False
    )
    dataset = DummyDataset()
    train_loader = DataLoader(dataset, batch_size=2)
    val_loader = DataLoader(dataset, batch_size=2)
    
    history = trainer.fit(train_loader, val_loader, epochs=2)
    
    assert len(history) == 2
    assert "epoch" in history[0]
    assert "train_loss" in history[0]
    assert "val_loss" in history[0]
