"""Main training entrypoint for ContrastiveTrust."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Dict

import torch
from torch.utils.data import DataLoader, Dataset

from src.losses.contrastive_trust_loss import (
    ContrastiveTrustLoss,
    ContrastiveTrustLossConfig,
)
from src.models.encoder import DualStreamEncoder, EncoderConfig
from src.models.physics_encoder import PhysicsEncoderConfig
from src.models.projection_head import ProjectionHead, ProjectionHeadConfig
from src.models.temporal_encoder import TemporalEncoderConfig
from src.training.callbacks import Callback
from src.training.checkpoint import ModelCheckpoint
from src.training.config import OptimizerConfig, SchedulerConfig
from src.training.early_stopping import EarlyStopping
from src.training.logger import MetricsLogger
from src.training.optimizer_factory import create_optimizer
from src.training.scheduler_factory import create_scheduler
from src.training.trainer import Trainer

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)


class DummyDataset(Dataset):
    """A dummy dataset generating random data for pipeline testing."""

    def __init__(self, num_samples: int = 100) -> None:
        self.num_samples = num_samples

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            "view1_window": torch.randn(100, 10),
            "view1_physics": torch.randn(18),
            "view2_window": torch.randn(100, 10),
            "view2_physics": torch.randn(18),
        }


def run_training_pipeline() -> None:
    """Instantiate components and run the training pipeline."""
    # 1. Configs
    encoder_config = EncoderConfig(
        temporal=TemporalEncoderConfig(input_channels=10),
        physics=PhysicsEncoderConfig(input_dim=18),
    )
    proj_config = ProjectionHeadConfig(input_dim=encoder_config.temporal.embedding_dim)
    loss_config = ContrastiveTrustLossConfig()

    # 2. Models
    encoder = DualStreamEncoder(encoder_config)
    projection_head = ProjectionHead(proj_config)
    loss_fn = ContrastiveTrustLoss(loss_config)

    # 3. Optimizer & Scheduler
    opt_config = OptimizerConfig(name="AdamW", lr=1e-3, weight_decay=1e-4)
    optimizer = create_optimizer(
        list(encoder.parameters()) + list(projection_head.parameters()), opt_config
    )

    sched_config = SchedulerConfig(name="CosineAnnealingLR", kwargs={"T_max": 10})
    scheduler = create_scheduler(optimizer, sched_config)

    # 4. Callbacks
    log_dir = Path("logs/training")
    callbacks: list[Callback] = [
        MetricsLogger(log_dir=log_dir),
        ModelCheckpoint(
            filepath=log_dir / "best_model.pt", monitor="val_loss", mode="min"
        ),
        EarlyStopping(monitor="val_loss", patience=3, mode="min"),
    ]

    # 5. Trainer
    trainer = Trainer(
        encoder=encoder,
        projection_head=projection_head,
        loss_fn=loss_fn,
        optimizer=optimizer,
        scheduler=scheduler,
        mixed_precision=torch.cuda.is_available(),
        gradient_accumulation_steps=2,
        callbacks=callbacks,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )

    # 6. Data Loaders
    train_dataset = DummyDataset(num_samples=64)
    val_dataset = DummyDataset(num_samples=16)

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

    # 7. Fit
    history = trainer.fit(train_loader, val_loader, epochs=5)
    logger.info("Training completed successfully. History: %s", history)


if __name__ == "__main__":
    run_training_pipeline()
