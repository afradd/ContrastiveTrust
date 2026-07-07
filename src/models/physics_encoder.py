"""MLP-based physics encoder for engineered physics features.

The :class:`PhysicsEncoder` maps a batch of physics feature vectors
``(B, F)`` to L2-normalised fixed-length embeddings ``(B, D)`` using
a multi-layer perceptron with LayerNorm, GELU activation, and dropout.

The input feature dimensionality is determined at runtime via
``input_dim`` in :class:`PhysicsEncoderConfig`, making the encoder
fully dataset-agnostic.

Configuration is centralised in the :class:`PhysicsEncoderConfig`
dataclass, which validates all hyper-parameters at construction time.

Example
-------
>>> import torch
>>> from src.models.physics_encoder import PhysicsEncoder, PhysicsEncoderConfig
>>> cfg = PhysicsEncoderConfig(input_dim=18)
>>> encoder = PhysicsEncoder(cfg)
>>> x = torch.randn(4, 18)
>>> z = encoder(x)
>>> z.shape
torch.Size([4, 256])
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

# Mapping from activation name → nn.Module class for TorchScript compat.
_ACTIVATION_REGISTRY: Dict[str, type] = {
    "gelu": nn.GELU,
    "relu": nn.ReLU,
    "silu": nn.SiLU,
    "tanh": nn.Tanh,
}


# ======================================================================
# Configuration
# ======================================================================


@dataclass(frozen=True)
class PhysicsEncoderConfig:
    """Hyper-parameters for :class:`PhysicsEncoder`.

    Parameters
    ----------
    input_dim : int
        Dimensionality of the incoming physics feature vector.
        Determined at runtime by the physics feature extractor.
    hidden_dims : tuple[int, ...]
        Widths of the hidden layers preceding the final projection.
        The length of this tuple determines the number of hidden
        stages.
    embedding_dim : int
        Dimensionality of the output embedding vector.
    dropout : float
        Dropout probability applied after each activation.
    bias : bool
        Whether linear layers include a learnable bias.
    activation : str
        Name of the activation function.  Must be one of
        ``'gelu'``, ``'relu'``, ``'silu'``, ``'tanh'``.

    Raises
    ------
    ValueError
        If any hyper-parameter fails validation.
    """

    input_dim: int = 1
    hidden_dims: Tuple[int, ...] = (512, 256)
    embedding_dim: int = 256
    dropout: float = 0.2
    bias: bool = True
    activation: str = "gelu"

    def __post_init__(self) -> None:
        """Validate all configuration fields."""
        if self.input_dim < 1:
            raise ValueError(
                f"input_dim must be positive, got {self.input_dim}"
            )
        if self.embedding_dim < 1:
            raise ValueError(
                f"embedding_dim must be positive, got {self.embedding_dim}"
            )
        if len(self.hidden_dims) == 0:
            raise ValueError("hidden_dims must not be empty")
        for i, dim in enumerate(self.hidden_dims):
            if dim < 1:
                raise ValueError(
                    f"hidden_dims[{i}] must be positive, got {dim}"
                )
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError(
                f"dropout must be in [0, 1), got {self.dropout}"
            )
        activation_lower = self.activation.lower()
        if activation_lower not in _ACTIVATION_REGISTRY:
            supported = ", ".join(sorted(_ACTIVATION_REGISTRY.keys()))
            raise ValueError(
                f"activation must be one of [{supported}], "
                f"got '{self.activation}'"
            )


# ======================================================================
# Physics Encoder
# ======================================================================


class PhysicsEncoder(nn.Module):
    """MLP-based physics encoder producing L2-normalised embeddings.

    Architecture::

        Input (B, F)
            │
        ┌───┴───────────────┐
        │  Linear(F → 512)  │
        │  LayerNorm(512)   │
        │  GELU             │
        │  Dropout          │
        └───┬───────────────┘
            │
        ┌───┴───────────────┐
        │  Linear(512 → 256)│
        │  LayerNorm(256)   │
        │  GELU             │
        │  Dropout          │
        └───┬───────────────┘
            │
        ┌───┴───────────────┐
        │  Linear(256 → 256)│
        └───┬───────────────┘
            │
            ▼ L2 normalise
        Output (B, D)

    Parameters
    ----------
    config : PhysicsEncoderConfig
        Full encoder configuration.

    Raises
    ------
    TypeError
        If *config* is not a :class:`PhysicsEncoderConfig`.
    """

    def __init__(self, config: PhysicsEncoderConfig) -> None:
        if not isinstance(config, PhysicsEncoderConfig):
            raise TypeError(
                f"config must be a PhysicsEncoderConfig, "
                f"got {type(config).__name__}"
            )
        super().__init__()

        # Store the dataclass in __dict__ directly so TorchScript does
        # not attempt to compile it.  Only JIT-visible primitives are
        # kept as proper attributes.
        self.__dict__["_config"] = config
        self._input_dim: int = config.input_dim
        self._embedding_dim: int = config.embedding_dim

        activation_cls = _ACTIVATION_REGISTRY[config.activation.lower()]

        # ---- hidden layers ----------------------------------------------
        layers: list[nn.Module] = []
        in_features = config.input_dim
        for hidden_dim in config.hidden_dims:
            layers.append(nn.Linear(in_features, hidden_dim, bias=config.bias))
            layers.append(nn.LayerNorm(hidden_dim))
            layers.append(activation_cls())
            layers.append(nn.Dropout(p=config.dropout))
            in_features = hidden_dim
        self.hidden = nn.Sequential(*layers)

        # ---- projection head (no norm / activation) ---------------------
        self.projection = nn.Linear(
            in_features, config.embedding_dim, bias=config.bias
        )

        # ---- logging ----------------------------------------------------
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(
            p.numel() for p in self.parameters() if p.requires_grad
        )
        logger.info(
            "PhysicsEncoder initialised | "
            "input_dim=%d | embedding_dim=%d | "
            "hidden_dims=%s | dropout=%.2f | activation=%s | "
            "bias=%s | total_params=%s | trainable_params=%s",
            config.input_dim,
            config.embedding_dim,
            config.hidden_dims,
            config.dropout,
            config.activation,
            config.bias,
            f"{total_params:,}",
            f"{trainable_params:,}",
        )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------
    @property
    @torch.jit.ignore
    def config(self) -> PhysicsEncoderConfig:
        """Return the encoder configuration."""
        return self.__dict__["_config"]

    @property
    @torch.jit.ignore
    def input_dim(self) -> int:
        """Return the expected physics feature dimensionality."""
        return self._input_dim

    @property
    @torch.jit.ignore
    def embedding_dim(self) -> int:
        """Return the embedding dimensionality."""
        return self._embedding_dim

    # ------------------------------------------------------------------
    # Input validation
    # ------------------------------------------------------------------

    @torch.jit.ignore
    def _validate_input(self, x: torch.Tensor) -> None:
        """Validate the input tensor before the forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Expected shape ``(B, F)`` with a floating-point dtype
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
        if x.ndim != 2:
            raise ValueError(
                f"Input must have exactly 2 dimensions (B, F), "
                f"got {x.ndim} dimensions with shape {tuple(x.shape)}"
            )
        feature_dim = x.shape[1]
        if feature_dim != self._input_dim:
            raise ValueError(
                f"Feature dimension (dim 1) must be {self._input_dim}, "
                f"got {feature_dim}"
            )
        if torch.isnan(x).any():
            raise ValueError("Input tensor contains NaN values")
        if torch.isinf(x).any():
            raise ValueError("Input tensor contains Inf values")

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode a batch of physics feature vectors.

        Parameters
        ----------
        x : torch.Tensor
            Input of shape ``(B, F)`` where *B* is the batch size and
            *F* is the physics feature dimensionality.

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
            "PhysicsEncoder forward | input shape=%s | dtype=%s",
            tuple(x.shape),
            x.dtype,
        )

        # Hidden layers
        h = self.hidden(x)  # (B, hidden_dims[-1])

        # Projection
        h = self.projection(h)  # (B, embedding_dim)

        # L2 normalisation
        embedding: torch.Tensor = F.normalize(h, p=2, dim=-1)

        logger.debug(
            "PhysicsEncoder forward | output shape=%s",
            tuple(embedding.shape),
        )
        return embedding
