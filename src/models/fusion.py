"""Residual gated fusion for temporal and physics embeddings.

The :class:`FeatureFusion` module combines temporal and physics encoder
embeddings ``(B, D)`` into a single unified representation ``(B, D)``
using a learned sigmoid gate with residual connection.

The gate learns *when* physics information is useful and *when*
temporal information should dominate, rather than treating both
modalities equally (as a naive concatenation or average would).

Architecture::

    Temporal (B, D)   Physics (B, D)
         │                 │
         └───────┬─────────┘
                 │ concat → (B, 2D)
         ┌───────┴───────┐
         │  Linear(2D→H) │
         │  LayerNorm(H)  │
         │  GELU          │
         │  Dropout        │
         │  Linear(H→D)   │
         │  Sigmoid        │
         └───────┬────────┘
                 │ gate g ∈ (0, 1)^D
                 │
         g * physics + (1 − g) * temporal     ← gated blend
                 │
         + temporal                            ← residual skip
                 │
         LayerNorm(D)
                 │
         L2 normalise
                 │
         Output (B, D)

Configuration is centralised in the :class:`FusionConfig` dataclass,
which validates all hyper-parameters at construction time.

Example
-------
>>> import torch
>>> from src.models.fusion import FeatureFusion, FusionConfig
>>> cfg = FusionConfig()
>>> fusion = FeatureFusion(cfg)
>>> t = torch.randn(4, 256)
>>> p = torch.randn(4, 256)
>>> z = fusion(t, p)
>>> z.shape
torch.Size([4, 256])
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


# ======================================================================
# Configuration
# ======================================================================


@dataclass(frozen=True)
class FusionConfig:
    """Hyper-parameters for :class:`FeatureFusion`.

    Parameters
    ----------
    embedding_dim : int
        Dimensionality of both input embeddings and the output
        embedding.
    hidden_dim : int
        Width of the hidden layer inside the gating network.
    dropout : float
        Dropout probability applied after the GELU activation in the
        gating network.
    bias : bool
        Whether the linear layers inside the gating network include a
        learnable bias.

    Raises
    ------
    ValueError
        If any hyper-parameter fails validation.
    """

    embedding_dim: int = 256
    hidden_dim: int = 512
    dropout: float = 0.2
    bias: bool = True

    def __post_init__(self) -> None:
        """Validate all configuration fields."""
        if self.embedding_dim < 1:
            raise ValueError(
                f"embedding_dim must be positive, got {self.embedding_dim}"
            )
        if self.hidden_dim < 1:
            raise ValueError(
                f"hidden_dim must be positive, got {self.hidden_dim}"
            )
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError(
                f"dropout must be in [0, 1), got {self.dropout}"
            )


# ======================================================================
# Feature Fusion
# ======================================================================


class FeatureFusion(nn.Module):
    """Residual gated fusion producing L2-normalised embeddings.

    The module concatenates temporal and physics embeddings, feeds the
    result through a two-layer gating network that outputs a
    per-dimension sigmoid gate, blends the two modalities via the gate,
    adds a residual skip from the temporal branch, and finally applies
    LayerNorm followed by L2 normalisation.

    Parameters
    ----------
    config : FusionConfig
        Full fusion configuration.

    Raises
    ------
    TypeError
        If *config* is not a :class:`FusionConfig`.
    """

    def __init__(self, config: FusionConfig) -> None:
        if not isinstance(config, FusionConfig):
            raise TypeError(
                f"config must be a FusionConfig, "
                f"got {type(config).__name__}"
            )
        super().__init__()

        # Store the dataclass in __dict__ directly so TorchScript does
        # not attempt to compile it.  Only JIT-visible primitives are
        # kept as proper attributes.
        self.__dict__["_config"] = config
        self._embedding_dim: int = config.embedding_dim

        # ---- gating network ---------------------------------------------
        self.gate_net = nn.Sequential(
            nn.Linear(
                config.embedding_dim * 2,
                config.hidden_dim,
                bias=config.bias,
            ),
            nn.LayerNorm(config.hidden_dim),
            nn.GELU(),
            nn.Dropout(p=config.dropout),
            nn.Linear(
                config.hidden_dim,
                config.embedding_dim,
                bias=config.bias,
            ),
            nn.Sigmoid(),
        )

        # ---- post-fusion layer norm -------------------------------------
        self.layer_norm = nn.LayerNorm(config.embedding_dim)

        # ---- logging ----------------------------------------------------
        total_params, trainable_params = self._count_parameters()
        logger.info(
            "FeatureFusion initialised | "
            "embedding_dim=%d | hidden_dim=%d | "
            "dropout=%.2f | bias=%s | "
            "total_params=%s | trainable_params=%s",
            config.embedding_dim,
            config.hidden_dim,
            config.dropout,
            config.bias,
            f"{total_params:,}",
            f"{trainable_params:,}",
        )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    @torch.jit.ignore
    def config(self) -> FusionConfig:
        """Return the fusion configuration."""
        return self.__dict__["_config"]

    @property
    @torch.jit.ignore
    def embedding_dim(self) -> int:
        """Return the embedding dimensionality."""
        return self._embedding_dim

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

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

    @torch.jit.ignore
    def _validate_inputs(
        self,
        temporal_embedding: torch.Tensor,
        physics_embedding: torch.Tensor,
    ) -> None:
        """Validate both input tensors before the forward pass.

        Parameters
        ----------
        temporal_embedding : torch.Tensor
            Expected shape ``(B, D)`` with floating-point dtype and no
            NaN or Inf values.
        physics_embedding : torch.Tensor
            Expected shape ``(B, D)`` with floating-point dtype and no
            NaN or Inf values.

        Raises
        ------
        TypeError
            If either input is not a :class:`torch.Tensor`.
        ValueError
            If either input fails shape, dtype, or value checks, or if
            the two inputs have mismatched batch sizes or embedding
            dimensions.
        """
        for name, tensor in [
            ("temporal_embedding", temporal_embedding),
            ("physics_embedding", physics_embedding),
        ]:
            if not isinstance(tensor, torch.Tensor):
                raise TypeError(
                    f"{name} must be a torch.Tensor, "
                    f"got {type(tensor).__name__}"
                )
            if not tensor.is_floating_point():
                raise ValueError(
                    f"{name} must have a floating-point dtype, "
                    f"got {tensor.dtype}"
                )
            if tensor.ndim != 2:
                raise ValueError(
                    f"{name} must have exactly 2 dimensions (B, D), "
                    f"got {tensor.ndim} dimensions with shape "
                    f"{tuple(tensor.shape)}"
                )
            if tensor.shape[1] != self._embedding_dim:
                raise ValueError(
                    f"{name} embedding dimension (dim 1) must be "
                    f"{self._embedding_dim}, got {tensor.shape[1]}"
                )
            if torch.isnan(tensor).any():
                raise ValueError(
                    f"{name} contains NaN values"
                )
            if torch.isinf(tensor).any():
                raise ValueError(
                    f"{name} contains Inf values"
                )

        # Cross-input consistency
        if temporal_embedding.shape[0] != physics_embedding.shape[0]:
            raise ValueError(
                f"Batch size mismatch: temporal_embedding has batch "
                f"size {temporal_embedding.shape[0]} but "
                f"physics_embedding has batch size "
                f"{physics_embedding.shape[0]}"
            )

    def _compute_gate(
        self,
        temporal_embedding: torch.Tensor,
        physics_embedding: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the per-dimension sigmoid gate from concatenated inputs.

        Parameters
        ----------
        temporal_embedding : torch.Tensor
            Shape ``(B, D)``.
        physics_embedding : torch.Tensor
            Shape ``(B, D)``.

        Returns
        -------
        torch.Tensor
            Gate vector of shape ``(B, D)`` with values in ``(0, 1)``.
        """
        concatenated = torch.cat(
            [temporal_embedding, physics_embedding], dim=-1
        )  # (B, 2D)
        gate: torch.Tensor = self.gate_net(concatenated)  # (B, D)
        return gate

    def _fuse(
        self,
        temporal_embedding: torch.Tensor,
        physics_embedding: torch.Tensor,
        gate: torch.Tensor,
    ) -> torch.Tensor:
        """Apply gated blending with residual addition and normalisation.

        Parameters
        ----------
        temporal_embedding : torch.Tensor
            Shape ``(B, D)``.
        physics_embedding : torch.Tensor
            Shape ``(B, D)``.
        gate : torch.Tensor
            Sigmoid gate of shape ``(B, D)``.

        Returns
        -------
        torch.Tensor
            L2-normalised fused embedding of shape ``(B, D)``.
        """
        # Gated blend
        blended = gate * physics_embedding + (1.0 - gate) * temporal_embedding

        # Residual addition from temporal branch
        residual = blended + temporal_embedding

        # Layer normalisation
        normed = self.layer_norm(residual)

        # L2 normalisation
        embedding: torch.Tensor = F.normalize(normed, p=2, dim=-1)
        return embedding

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        temporal_embedding: torch.Tensor,
        physics_embedding: torch.Tensor,
    ) -> torch.Tensor:
        """Fuse temporal and physics embeddings into one representation.

        Parameters
        ----------
        temporal_embedding : torch.Tensor
            Temporal encoder output of shape ``(B, D)`` where *D* is
            ``embedding_dim``.
        physics_embedding : torch.Tensor
            Physics encoder output of shape ``(B, D)`` where *D* is
            ``embedding_dim``.

        Returns
        -------
        torch.Tensor
            L2-normalised fused embedding of shape ``(B, D)``.

        Raises
        ------
        TypeError
            If either input is not a :class:`torch.Tensor`.
        ValueError
            If either input fails shape, dtype, or value validation.
        """
        self._validate_inputs(temporal_embedding, physics_embedding)
        logger.debug(
            "FeatureFusion forward | "
            "temporal shape=%s | physics shape=%s | dtype=%s",
            tuple(temporal_embedding.shape),
            tuple(physics_embedding.shape),
            temporal_embedding.dtype,
        )

        gate = self._compute_gate(temporal_embedding, physics_embedding)
        embedding = self._fuse(temporal_embedding, physics_embedding, gate)

        logger.debug(
            "FeatureFusion forward | output shape=%s",
            tuple(embedding.shape),
        )
        return embedding
