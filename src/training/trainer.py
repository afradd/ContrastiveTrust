"""Trainer for ContrastiveTrust.

This module provides the :class:`Trainer` which encapsulates the training
loop, validation loop, gradient accumulation, gradient clipping, and mixed
precision logic for the Dual-Stream Encoder.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Iterable, List, Optional

import torch
import torch.nn as nn
import torch.optim as optim

from src.losses.contrastive_trust_loss import ContrastiveTrustLoss
from src.models.encoder import DualStreamEncoder
from src.models.projection_head import ProjectionHead
from src.training.callbacks import Callback
from src.training.mixed_precision import MixedPrecisionManager

logger = logging.getLogger(__name__)


class Trainer:
    """Orchestrates the pre-training of the Dual-Stream Encoder.

    Parameters
    ----------
    encoder : DualStreamEncoder
        The dual-stream encoder to train.
    projection_head : ProjectionHead
        The projection head mapping to the contrastive latent space.
    loss_fn : ContrastiveTrustLoss
        The unified multi-objective loss function.
    optimizer : torch.optim.Optimizer
        The optimiser for updating model parameters.
    scheduler : torch.optim.lr_scheduler.LRScheduler, optional
        An optional learning rate scheduler.
    mixed_precision : bool, default=True
        Whether to enable Automatic Mixed Precision (AMP).
    gradient_accumulation_steps : int, default=1
        Number of steps to accumulate gradients before stepping.
    max_grad_norm : float, default=1.0
        Maximum norm for gradient clipping.  Set to 0.0 to disable.
    callbacks : list of Callback, optional
        A list of callbacks to invoke during the training lifecycle.
    device : str, default="cuda"
        Target device for training.
    """

    def __init__(
        self,
        encoder: DualStreamEncoder,
        projection_head: ProjectionHead,
        loss_fn: ContrastiveTrustLoss,
        optimizer: optim.Optimizer,
        scheduler: Optional[Any] = None,
        mixed_precision: bool = True,
        gradient_accumulation_steps: int = 1,
        max_grad_norm: float = 1.0,
        callbacks: Optional[List[Callback]] = None,
        device: str = "cuda",
    ) -> None:
        if gradient_accumulation_steps < 1:
            raise ValueError(
                f"gradient_accumulation_steps must be >= 1, "
                f"got {gradient_accumulation_steps}"
            )
        if max_grad_norm < 0.0:
            raise ValueError(
                f"max_grad_norm must be >= 0.0, got {max_grad_norm}"
            )

        self.device = torch.device(device)
        self.encoder = encoder.to(self.device)
        self.projection_head = projection_head.to(self.device)
        self.loss_fn = loss_fn.to(self.device)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.max_grad_norm = max_grad_norm

        self.amp = MixedPrecisionManager(
            enabled=mixed_precision, device=device
        )
        self.callbacks = callbacks or []
        self.should_stop = False

        logger.info(
            "Trainer initialised | device=%s | amp=%s | "
            "grad_acc_steps=%d | max_grad_norm=%.2f",
            self.device,
            mixed_precision,
            self.gradient_accumulation_steps,
            self.max_grad_norm,
        )

    def _get_trainable_parameters(self) -> List[nn.Parameter]:
        """Return all trainable parameters from encoder and projection head."""
        return [
            p for p in self.encoder.parameters() if p.requires_grad
        ] + [
            p for p in self.projection_head.parameters() if p.requires_grad
        ]

    def train_epoch(self, dataloader: Iterable[Dict[str, torch.Tensor]]) -> Dict[str, float]:
        """Run one epoch of training.

        Parameters
        ----------
        dataloader : iterable
            The training dataloader yielding dictionaries with keys:
            ``"view1_window"``, ``"view1_physics"``, ``"view2_window"``,
            ``"view2_physics"``.

        Returns
        -------
        dict[str, float]
            A dictionary containing the average training metrics
            (e.g., ``"train_loss"``).
        """
        self.encoder.train()
        self.projection_head.train()
        self.loss_fn.train()

        total_loss = 0.0
        total_physics_loss = 0.0
        num_batches = 0

        self.optimizer.zero_grad(set_to_none=True)

        for step, batch in enumerate(dataloader):
            for callback in self.callbacks:
                callback.on_batch_begin(self, step)

            v1_window = batch["view1_window"].to(self.device, non_blocking=True)
            v1_physics = batch["view1_physics"].to(self.device, non_blocking=True)
            v2_window = batch["view2_window"].to(self.device, non_blocking=True)
            v2_physics = batch["view2_physics"].to(self.device, non_blocking=True)

            with self.amp.autocast():
                enc1 = self.encoder(v1_window, v1_physics)
                enc2 = self.encoder(v2_window, v2_physics)

                proj1 = self.projection_head(enc1["embedding"])
                proj2 = self.projection_head(enc2["embedding"])

                # We use the fused embedding and physics embedding from view 1
                # to compute the physics consistency loss, as well as the two
                # projections for the contrastive loss.
                out = self.loss_fn(
                    projection_view_1=proj1,
                    projection_view_2=proj2,
                    temporal_embedding=enc1["temporal_embedding"],
                    physics_embedding=enc1["physics_embedding"],
                )
                loss = out["loss"] / self.gradient_accumulation_steps
                physics_loss_val = out["physics_loss"].item()

            self.amp.scale_and_backward(loss)

            if (step + 1) % self.gradient_accumulation_steps == 0:
                if self.max_grad_norm > 0.0:
                    self.amp.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self._get_trainable_parameters(),
                        self.max_grad_norm,
                    )
                self.amp.step_and_update(self.optimizer)
                self.optimizer.zero_grad(set_to_none=True)

            total_loss += loss.item() * self.gradient_accumulation_steps
            total_physics_loss += physics_loss_val
            num_batches += 1

            for callback in self.callbacks:
                callback.on_batch_end(self, step, {"loss": loss.item()})

        # Handle remaining gradients if len(dataloader) is not divisible by acc_steps
        if num_batches % self.gradient_accumulation_steps != 0:
            if self.max_grad_norm > 0.0:
                self.amp.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self._get_trainable_parameters(),
                    self.max_grad_norm,
                )
            self.amp.step_and_update(self.optimizer)
            self.optimizer.zero_grad(set_to_none=True)

        if self.scheduler is not None:
            self.scheduler.step()

        avg_loss = total_loss / max(1, num_batches)
        avg_physics_loss = total_physics_loss / max(1, num_batches)
        logger.debug("train_epoch complete | train_loss=%.4f | train_physics_loss=%.4f", avg_loss, avg_physics_loss)
        return {"train_loss": avg_loss, "train_physics_loss": avg_physics_loss}

    @torch.no_grad()
    def validate_epoch(self, dataloader: Iterable[Dict[str, torch.Tensor]]) -> Dict[str, float]:
        """Run one epoch of validation.

        Parameters
        ----------
        dataloader : iterable
            The validation dataloader.

        Returns
        -------
        dict[str, float]
            A dictionary containing the average validation metrics.
        """
        self.encoder.eval()
        self.projection_head.eval()
        self.loss_fn.eval()

        total_loss = 0.0
        total_physics_loss = 0.0
        num_batches = 0

        for batch in dataloader:
            v1_window = batch["view1_window"].to(self.device, non_blocking=True)
            v1_physics = batch["view1_physics"].to(self.device, non_blocking=True)
            v2_window = batch["view2_window"].to(self.device, non_blocking=True)
            v2_physics = batch["view2_physics"].to(self.device, non_blocking=True)

            with self.amp.autocast():
                enc1 = self.encoder(v1_window, v1_physics)
                enc2 = self.encoder(v2_window, v2_physics)

                proj1 = self.projection_head(enc1["embedding"])
                proj2 = self.projection_head(enc2["embedding"])

                out = self.loss_fn(
                    projection_view_1=proj1,
                    projection_view_2=proj2,
                    temporal_embedding=enc1["temporal_embedding"],
                    physics_embedding=enc1["physics_embedding"],
                )
                loss = out["loss"]
                physics_loss_val = out["physics_loss"].item()

            total_loss += loss.item()
            total_physics_loss += physics_loss_val
            num_batches += 1

        avg_loss = total_loss / max(1, num_batches)
        avg_physics_loss = total_physics_loss / max(1, num_batches)
        logger.debug("validate_epoch complete | val_loss=%.4f | val_physics_loss=%.4f", avg_loss, avg_physics_loss)
        return {"val_loss": avg_loss, "val_physics_loss": avg_physics_loss}

    def fit(
        self,
        train_loader: Iterable[Dict[str, torch.Tensor]],
        val_loader: Iterable[Dict[str, torch.Tensor]],
        epochs: int,
    ) -> List[Dict[str, Any]]:
        """Run the full training and validation loop.

        Parameters
        ----------
        train_loader : iterable
            The training dataloader.
        val_loader : iterable
            The validation dataloader.
        epochs : int
            The total number of epochs to train.

        Returns
        -------
        list of dict
            A history of metrics for each epoch.
        """
        history = []
        logger.info("Starting fit for %d epochs", epochs)

        for callback in self.callbacks:
            callback.on_train_begin(self)

        for epoch in range(epochs):
            if self.should_stop:
                break

            for callback in self.callbacks:
                callback.on_epoch_begin(self, epoch + 1)

            train_metrics = self.train_epoch(train_loader)
            val_metrics = self.validate_epoch(val_loader)

            epoch_results = {
                "epoch": epoch + 1,
                **train_metrics,
                **val_metrics,
            }
            history.append(epoch_results)

            logger.info(
                "Epoch %03d/%03d | Train Loss: %.4f | Val Loss: %.4f",
                epoch + 1,
                epochs,
                train_metrics["train_loss"],
                val_metrics["val_loss"],
            )

            for callback in self.callbacks:
                callback.on_epoch_end(self, epoch + 1, epoch_results)

        for callback in self.callbacks:
            callback.on_train_end(self)

        logger.info("Training complete.")
        return history
