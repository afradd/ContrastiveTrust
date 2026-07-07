"""Dual-stream encoder for ContrastiveTrust.

The :class:`DualStreamEncoder` orchestrates the three sub-modules

* :class:`~src.models.temporal_encoder.TemporalEncoder`
* :class:`~src.models.physics_encoder.PhysicsEncoder`
* :class:`~src.models.fusion.FeatureFusion`

into one reusable representation-learning backbone.  It accepts a raw
time-series window ``(B, T, S)`` together with precomputed physics
features ``(B, P)`` and produces a unified embedding ``(B, D)`` along
with the individual temporal and physics embeddings.

Configuration is centralised in the :class:`EncoderConfig` dataclass,
which composes sub-module configurations and validates consistency at
construction time.

Example
-------
>>> import torch
>>> from src.models.encoder import DualStreamEncoder, EncoderConfig
>>> from src.models.temporal_encoder import TemporalEncoderConfig
>>> from src.models.physics_encoder import PhysicsEncoderConfig
>>> from src.models.fusion import FusionConfig
>>> cfg = EncoderConfig(
...     temporal=TemporalEncoderConfig(input_channels=10),
...     physics=PhysicsEncoderConfig(input_dim=18),
...     fusion=FusionConfig(),
... )
>>> encoder = DualStreamEncoder(cfg)
>>> out = encoder(
...     window=torch.randn(4, 100, 10),
...     physics_features=torch.randn(4, 18),
... )
>>> out["embedding"].shape
torch.Size([4, 256])
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict

import torch
import torch.nn as nn

from src.models.fusion import FeatureFusion, FusionConfig
from src.models.physics_encoder import PhysicsEncoder, PhysicsEncoderConfig
from src.models.temporal_encoder import TemporalEncoder, TemporalEncoderConfig

logger = logging.getLogger(__name__)


# ======================================================================
# Configuration
# ======================================================================


@dataclass(frozen=True)
class EncoderConfig:
    """Unified configuration for :class:`DualStreamEncoder`.

    Composes the configurations for the temporal encoder, physics
    encoder, and feature fusion modules.  Consistency between embedding
    dimensions is validated at construction time.

    Parameters
    ----------
    temporal : TemporalEncoderConfig
        Hyper-parameters for the temporal (CNN) encoder.
    physics : PhysicsEncoderConfig
        Hyper-parameters for the physics (MLP) encoder.
    fusion : FusionConfig
        Hyper-parameters for the residual gated fusion module.

    Raises
    ------
    ValueError
        If the embedding dimensions of the three sub-modules are
        inconsistent.
    """

    temporal: TemporalEncoderConfig = field(
        default_factory=TemporalEncoderConfig
    )
    physics: PhysicsEncoderConfig = field(
        default_factory=PhysicsEncoderConfig
    )
    fusion: FusionConfig = field(default_factory=FusionConfig)

    def __post_init__(self) -> None:
        """Validate cross-module embedding-dimension consistency."""
        dims = {
            "temporal.embedding_dim": self.temporal.embedding_dim,
            "physics.embedding_dim": self.physics.embedding_dim,
            "fusion.embedding_dim": self.fusion.embedding_dim,
        }
        unique_dims = set(dims.values())
        if len(unique_dims) != 1:
            parts = ", ".join(f"{k}={v}" for k, v in dims.items())
            raise ValueError(
                f"Embedding dimensions must be consistent across all "
                f"sub-modules, got: {parts}"
            )


# ======================================================================
# Dual-Stream Encoder
# ======================================================================


class DualStreamEncoder(nn.Module):
    """Dual-stream encoder producing unified embeddings.

    Architecture::

        window (B, T, S)       physics_features (B, P)
              │                         │
              ▼                         ▼
        TemporalEncoder           PhysicsEncoder
              │                         │
        temporal_emb (B, D)     physics_emb (B, D)
              │                         │
              └────────┬────────────────┘
                       │
                 FeatureFusion
                       │
                embedding (B, D)

    The :meth:`forward` method returns a dictionary::

        {
            "embedding":          Tensor(B, D),
            "temporal_embedding": Tensor(B, D),
            "physics_embedding":  Tensor(B, D),
        }

    Parameters
    ----------
    config : EncoderConfig
        Full encoder configuration composing sub-module configs.

    Raises
    ------
    TypeError
        If *config* is not an :class:`EncoderConfig`.
    """

    def __init__(self, config: EncoderConfig) -> None:
        if not isinstance(config, EncoderConfig):
            raise TypeError(
                f"config must be an EncoderConfig, "
                f"got {type(config).__name__}"
            )
        super().__init__()

        self._config = config
        self._embedding_dim: int = config.temporal.embedding_dim

        # ---- sub-modules ------------------------------------------------
        self.temporal_encoder = TemporalEncoder(config.temporal)
        self.physics_encoder = PhysicsEncoder(config.physics)
        self.fusion = FeatureFusion(config.fusion)

        # ---- logging ----------------------------------------------------
        total, trainable = self._count_parameters()
        logger.info(
            "DualStreamEncoder initialised | "
            "embedding_dim=%d | "
            "temporal_params=%s | physics_params=%s | "
            "fusion_params=%s | total_params=%s | trainable_params=%s",
            self._embedding_dim,
            f"{self._submodule_params(self.temporal_encoder):,}",
            f"{self._submodule_params(self.physics_encoder):,}",
            f"{self._submodule_params(self.fusion):,}",
            f"{total:,}",
            f"{trainable:,}",
        )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def config(self) -> EncoderConfig:
        """Return the encoder configuration."""
        return self._config

    @property
    def embedding_dimension(self) -> int:
        """Return the unified embedding dimensionality."""
        return self._embedding_dim

    @property
    def device(self) -> torch.device:
        """Return the device of the encoder parameters."""
        return next(self.parameters()).device

    @property
    def dtype(self) -> torch.dtype:
        """Return the dtype of the encoder parameters."""
        return next(self.parameters()).dtype

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _submodule_params(module: nn.Module) -> int:
        """Return the total number of parameters in *module*.

        Parameters
        ----------
        module : nn.Module
            Any PyTorch module.

        Returns
        -------
        int
            Total parameter count.
        """
        return sum(p.numel() for p in module.parameters())

    def _count_parameters(self) -> tuple[int, int]:
        """Count total and trainable parameters.

        Returns
        -------
        tuple[int, int]
            ``(total_params, trainable_params)``
        """
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(
            p.numel() for p in self.parameters() if p.requires_grad
        )
        return total, trainable

    def _validate_inputs(
        self,
        window: torch.Tensor,
        physics_features: torch.Tensor,
    ) -> None:
        """Validate both input tensors before the forward pass.

        Parameters
        ----------
        window : torch.Tensor
            Raw time-series window of shape ``(B, T, S)``.
        physics_features : torch.Tensor
            Precomputed physics features of shape ``(B, P)``.

        Raises
        ------
        TypeError
            If either input is not a :class:`torch.Tensor`.
        ValueError
            If either input fails shape, dtype, or value checks, or if
            batch sizes are mismatched.
        """
        # ---- type checks ------------------------------------------------
        if not isinstance(window, torch.Tensor):
            raise TypeError(
                f"window must be a torch.Tensor, "
                f"got {type(window).__name__}"
            )
        if not isinstance(physics_features, torch.Tensor):
            raise TypeError(
                f"physics_features must be a torch.Tensor, "
                f"got {type(physics_features).__name__}"
            )

        # ---- dtype checks -----------------------------------------------
        if not window.is_floating_point():
            raise ValueError(
                f"window must have a floating-point dtype, "
                f"got {window.dtype}"
            )
        if not physics_features.is_floating_point():
            raise ValueError(
                f"physics_features must have a floating-point dtype, "
                f"got {physics_features.dtype}"
            )

        # ---- shape checks -----------------------------------------------
        if window.ndim != 3:
            raise ValueError(
                f"window must have exactly 3 dimensions (B, T, S), "
                f"got {window.ndim} dimensions with shape "
                f"{tuple(window.shape)}"
            )
        if physics_features.ndim != 2:
            raise ValueError(
                f"physics_features must have exactly 2 dimensions (B, P), "
                f"got {physics_features.ndim} dimensions with shape "
                f"{tuple(physics_features.shape)}"
            )

        # ---- batch-size consistency -------------------------------------
        if window.shape[0] != physics_features.shape[0]:
            raise ValueError(
                f"Batch size mismatch: window has batch size "
                f"{window.shape[0]} but physics_features has batch "
                f"size {physics_features.shape[0]}"
            )

        # ---- NaN / Inf checks -------------------------------------------
        if torch.isnan(window).any():
            raise ValueError("window contains NaN values")
        if torch.isinf(window).any():
            raise ValueError("window contains Inf values")
        if torch.isnan(physics_features).any():
            raise ValueError("physics_features contains NaN values")
        if torch.isinf(physics_features).any():
            raise ValueError("physics_features contains Inf values")

    # ------------------------------------------------------------------
    # Public methods
    # ------------------------------------------------------------------

    def forward(
        self,
        window: torch.Tensor,
        physics_features: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Run the full dual-stream encoding pipeline.

        Parameters
        ----------
        window : torch.Tensor
            Raw time-series window of shape ``(B, T, S)`` where *B* is
            the batch size, *T* is the window length, and *S* is the
            number of sensor channels.
        physics_features : torch.Tensor
            Precomputed physics features of shape ``(B, P)`` where *P*
            is the physics feature dimensionality.

        Returns
        -------
        dict[str, torch.Tensor]
            Dictionary with keys:

            - ``"embedding"`` — unified fused embedding ``(B, D)``
            - ``"temporal_embedding"`` — temporal encoder output
              ``(B, D)``
            - ``"physics_embedding"`` — physics encoder output
              ``(B, D)``

        Raises
        ------
        TypeError
            If either input is not a :class:`torch.Tensor`.
        ValueError
            If inputs fail validation.
        """
        self._validate_inputs(window, physics_features)
        logger.debug(
            "DualStreamEncoder forward | "
            "window shape=%s | physics shape=%s | dtype=%s",
            tuple(window.shape),
            tuple(physics_features.shape),
            window.dtype,
        )

        # ---- temporal stream --------------------------------------------
        temporal_embedding = self.temporal_encoder(window)  # (B, D)

        # ---- physics stream ---------------------------------------------
        physics_embedding = self.physics_encoder(
            physics_features
        )  # (B, D)

        # ---- fusion -----------------------------------------------------
        embedding = self.fusion(
            temporal_embedding, physics_embedding
        )  # (B, D)

        logger.debug(
            "DualStreamEncoder forward | "
            "temporal_emb shape=%s | physics_emb shape=%s | "
            "fused_emb shape=%s",
            tuple(temporal_embedding.shape),
            tuple(physics_embedding.shape),
            tuple(embedding.shape),
        )

        return {
            "embedding": embedding,
            "temporal_embedding": temporal_embedding,
            "physics_embedding": physics_embedding,
        }

    def encode(
        self,
        window: torch.Tensor,
        physics_features: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Encode inputs without computing gradients.

        Convenience wrapper around :meth:`forward` that disables
        gradient computation and sets the model to evaluation mode
        temporarily.  Useful during inference, evaluation, or
        visualisation.

        Parameters
        ----------
        window : torch.Tensor
            Raw time-series window of shape ``(B, T, S)``.
        physics_features : torch.Tensor
            Precomputed physics features of shape ``(B, P)``.

        Returns
        -------
        dict[str, torch.Tensor]
            Same dictionary format as :meth:`forward`.
        """
        was_training = self.training
        self.eval()
        try:
            with torch.no_grad():
                return self.forward(window, physics_features)
        finally:
            if was_training:
                self.train()

    def freeze_temporal(self) -> None:
        """Freeze all parameters in the temporal encoder.

        After calling this method the temporal encoder's parameters will
        not receive gradients during back-propagation.
        """
        for param in self.temporal_encoder.parameters():
            param.requires_grad_(False)
        logger.info(
            "DualStreamEncoder: temporal encoder frozen "
            "(%s parameters)",
            f"{self._submodule_params(self.temporal_encoder):,}",
        )

    def freeze_physics(self) -> None:
        """Freeze all parameters in the physics encoder.

        After calling this method the physics encoder's parameters will
        not receive gradients during back-propagation.
        """
        for param in self.physics_encoder.parameters():
            param.requires_grad_(False)
        logger.info(
            "DualStreamEncoder: physics encoder frozen "
            "(%s parameters)",
            f"{self._submodule_params(self.physics_encoder):,}",
        )

    def unfreeze_all(self) -> None:
        """Unfreeze all parameters in every sub-module.

        Restores gradient computation for all parameters in the
        temporal encoder, physics encoder, and fusion module.
        """
        for param in self.parameters():
            param.requires_grad_(True)
        total, trainable = self._count_parameters()
        logger.info(
            "DualStreamEncoder: all parameters unfrozen | "
            "total=%s | trainable=%s",
            f"{total:,}",
            f"{trainable:,}",
        )

    def parameter_count(self) -> Dict[str, int]:
        """Return parameter counts for the encoder and its sub-modules.

        Returns
        -------
        dict[str, int]
            Dictionary with keys ``"total"``, ``"trainable"``,
            ``"temporal"``, ``"physics"``, and ``"fusion"``.
        """
        total, trainable = self._count_parameters()
        return {
            "total": total,
            "trainable": trainable,
            "temporal": self._submodule_params(self.temporal_encoder),
            "physics": self._submodule_params(self.physics_encoder),
            "fusion": self._submodule_params(self.fusion),
        }
