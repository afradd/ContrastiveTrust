"""CNN-based temporal encoder for multivariate ICS time-series windows.

The :class:`TemporalEncoder` maps a batch of multivariate sliding windows
``(B, T, S)`` to L2-normalised fixed-length embeddings ``(B, D)`` using a
stack of residual 1-D convolutional blocks followed by adaptive average
pooling and a two-layer projection head.

Configuration is centralised in the :class:`TemporalEncoderConfig`
dataclass, which validates all hyper-parameters at construction time.

Example
-------
>>> import torch
>>> from src.models.temporal_encoder import TemporalEncoder, TemporalEncoderConfig
>>> cfg = TemporalEncoderConfig(input_channels=10)
>>> encoder = TemporalEncoder(cfg)
>>> x = torch.randn(4, 100, 10)
>>> z = encoder(x)
>>> z.shape
torch.Size([4, 256])
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.blocks import ConvNormActivation, ResidualConvBlock

logger = logging.getLogger(__name__)


# ======================================================================
# Configuration
# ======================================================================


@dataclass(frozen=True)
class TemporalEncoderConfig:
    """Hyper-parameters for :class:`TemporalEncoder`.

    Parameters
    ----------
    input_channels : int
        Number of sensor channels (S dimension of the input tensor).
    embedding_dim : int
        Dimensionality of the output embedding vector.
    hidden_channels : tuple[int, ...]
        Number of output channels for each residual block.  The length
        of this tuple determines the number of residual stages.
    kernel_sizes : tuple[int, ...]
        Kernel size for each residual block.  Must have the same length
        as *hidden_channels*.
    dropout : float
        Dropout probability used in residual blocks and the projection
        head.
    bias : bool
        Whether convolutions include a learnable bias.

    Raises
    ------
    ValueError
        If any hyper-parameter fails validation.
    """

    input_channels: int = 1
    embedding_dim: int = 256
    hidden_channels: Tuple[int, ...] = (64, 128, 256)
    kernel_sizes: Tuple[int, ...] = (7, 5, 3)
    dropout: float = 0.2
    bias: bool = False

    def __post_init__(self) -> None:
        """Validate all configuration fields."""
        if self.input_channels < 1:
            raise ValueError(
                f"input_channels must be positive, got {self.input_channels}"
            )
        if self.embedding_dim < 1:
            raise ValueError(
                f"embedding_dim must be positive, got {self.embedding_dim}"
            )
        if len(self.hidden_channels) == 0:
            raise ValueError("hidden_channels must not be empty")
        for i, ch in enumerate(self.hidden_channels):
            if ch < 1:
                raise ValueError(
                    f"hidden_channels[{i}] must be positive, got {ch}"
                )
        if len(self.kernel_sizes) != len(self.hidden_channels):
            raise ValueError(
                f"kernel_sizes length ({len(self.kernel_sizes)}) must match "
                f"hidden_channels length ({len(self.hidden_channels)})"
            )
        for i, ks in enumerate(self.kernel_sizes):
            if ks < 1 or ks % 2 == 0:
                raise ValueError(
                    f"kernel_sizes[{i}] must be a positive odd integer, "
                    f"got {ks}"
                )
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError(
                f"dropout must be in [0, 1), got {self.dropout}"
            )


# ======================================================================
# Temporal Encoder
# ======================================================================


class TemporalEncoder(nn.Module):
    """CNN-based temporal encoder producing L2-normalised embeddings.

    Architecture::

        Input (B, T, S)
            │
            ▼ transpose → (B, S, T)
            │
        ┌───┴───┐
        │  Stem │  ConvNormActivation  S → hidden[0]
        └───┬───┘
            │
        ┌───┴───┐
        │ Res-1 │  ResidualConvBlock  hidden[0] → hidden[0]
        └───┬───┘
            │
        ┌───┴───┐
        │ Res-2 │  ResidualConvBlock  hidden[0] → hidden[1]
        └───┬───┘
            │
        ┌───┴───┐
        │ Res-3 │  ResidualConvBlock  hidden[1] → hidden[2]
        └───┬───┘
            │
            ▼ AdaptiveAvgPool1d(1) → Flatten
            │
        ┌───┴────────────┐
        │  Projection    │
        │  Linear → LN   │
        │  → GELU → Drop │
        │  → Linear      │
        └───┬────────────┘
            │
            ▼ L2 normalise
        Output (B, D)

    Parameters
    ----------
    config : TemporalEncoderConfig
        Full encoder configuration.

    Raises
    ------
    TypeError
        If *config* is not a :class:`TemporalEncoderConfig`.
    """

    def __init__(self, config: TemporalEncoderConfig) -> None:
        if not isinstance(config, TemporalEncoderConfig):
            raise TypeError(
                f"config must be a TemporalEncoderConfig, "
                f"got {type(config).__name__}"
            )
        super().__init__()

        self._config = config
        self._input_channels = config.input_channels

        # ---- stem -------------------------------------------------------
        first_hidden = config.hidden_channels[0]
        self.stem = ConvNormActivation(
            in_channels=config.input_channels,
            out_channels=first_hidden,
            kernel_size=config.kernel_sizes[0],
            bias=config.bias,
            activation=nn.GELU,
        )

        # ---- residual blocks --------------------------------------------
        blocks: list[nn.Module] = []
        in_ch = first_hidden
        for out_ch, ks in zip(
            config.hidden_channels, config.kernel_sizes
        ):
            blocks.append(
                ResidualConvBlock(
                    in_channels=in_ch,
                    out_channels=out_ch,
                    kernel_size=ks,
                    dropout=config.dropout,
                    bias=config.bias,
                )
            )
            in_ch = out_ch
        self.res_blocks = nn.Sequential(*blocks)

        # ---- pooling ----------------------------------------------------
        self.pool = nn.AdaptiveAvgPool1d(output_size=1)

        # ---- projection head -------------------------------------------
        final_channels = config.hidden_channels[-1]
        self.projection = nn.Sequential(
            nn.Linear(final_channels, final_channels),
            nn.LayerNorm(final_channels),
            nn.GELU(),
            nn.Dropout(p=config.dropout),
            nn.Linear(final_channels, config.embedding_dim),
        )

        # ---- logging ----------------------------------------------------
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(
            p.numel() for p in self.parameters() if p.requires_grad
        )
        logger.info(
            "TemporalEncoder initialised | "
            "input_channels=%d | embedding_dim=%d | "
            "hidden_channels=%s | kernel_sizes=%s | "
            "dropout=%.2f | total_params=%s | trainable_params=%s",
            config.input_channels,
            config.embedding_dim,
            config.hidden_channels,
            config.kernel_sizes,
            config.dropout,
            f"{total_params:,}",
            f"{trainable_params:,}",
        )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def config(self) -> TemporalEncoderConfig:
        """Return the encoder configuration."""
        return self._config

    @property
    def input_channels(self) -> int:
        """Return the expected number of sensor channels."""
        return self._input_channels

    @property
    def embedding_dim(self) -> int:
        """Return the embedding dimensionality."""
        return self._config.embedding_dim

    # ------------------------------------------------------------------
    # Input validation
    # ------------------------------------------------------------------

    def _validate_input(self, x: torch.Tensor) -> None:
        """Validate the input tensor before the forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Expected shape ``(B, T, S)`` with a floating-point dtype
            and no NaN or Inf values.

        Raises
        ------
        TypeError
            If *x* is not a :class:`torch.Tensor`.
        ValueError
            If *x* fails any shape, dtype, or value check.
        """
        if not isinstance(x, torch.Tensor):
            raise TypeError(
                f"Input must be a torch.Tensor, got {type(x).__name__}"
            )
        if not x.is_floating_point():
            raise ValueError(
                f"Input must have a floating-point dtype, got {x.dtype}"
            )
        if x.ndim != 3:
            raise ValueError(
                f"Input must have exactly 3 dimensions (B, T, S), "
                f"got {x.ndim} dimensions with shape {tuple(x.shape)}"
            )
        batch_size, window_len, sensor_count = x.shape
        if window_len < 1:
            raise ValueError(
                f"Window length (dim 1) must be positive, got {window_len}"
            )
        if sensor_count != self._input_channels:
            raise ValueError(
                f"Sensor count (dim 2) must be {self._input_channels}, "
                f"got {sensor_count}"
            )
        if torch.isnan(x).any():
            raise ValueError("Input tensor contains NaN values")
        if torch.isinf(x).any():
            raise ValueError("Input tensor contains Inf values")

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode a batch of multivariate time-series windows.

        Parameters
        ----------
        x : torch.Tensor
            Input of shape ``(B, T, S)`` where *B* is the batch size,
            *T* is the window length, and *S* is the number of sensors.

        Returns
        -------
        torch.Tensor
            L2-normalised embedding of shape ``(B, D)`` where *D* is
            the configured ``embedding_dim``.

        Raises
        ------
        TypeError
            If *x* is not a :class:`torch.Tensor`.
        ValueError
            If *x* fails shape, dtype, or value validation.
        """
        self._validate_input(x)
        logger.debug(
            "TemporalEncoder forward | input shape=%s | dtype=%s",
            tuple(x.shape),
            x.dtype,
        )

        # (B, T, S) → (B, S, T)  — channels-first for Conv1d
        h = x.transpose(1, 2)

        # Stem + residual blocks
        h = self.stem(h)
        h = self.res_blocks(h)

        # Pool → flatten
        h = self.pool(h)          # (B, C, 1)
        h = h.squeeze(-1)         # (B, C)

        # Projection head
        h = self.projection(h)    # (B, D)

        # L2 normalisation
        embedding = F.normalize(h, p=2, dim=-1)

        logger.debug(
            "TemporalEncoder forward | output shape=%s",
            tuple(embedding.shape),
        )
        return embedding
