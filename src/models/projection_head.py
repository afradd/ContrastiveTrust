"""Projection head for self-supervised contrastive learning.

The :class:`ProjectionHead` maps encoder embeddings ``(B, D_in)`` into a
lower-dimensional latent space ``(B, D_out)`` where the contrastive loss
is applied.  During inference the projection head is **not** used; only
the encoder embeddings are retained.

Architecture::

    Input (256)
        │
    Linear(256 → 256)
        │
    LayerNorm(256)
        │
    GELU
        │
    Dropout
        │
    Linear(256 → 256)
        │
    LayerNorm(256)
        │
    GELU
        │
    Dropout
        │
    Linear(256 → 128)
        │
    L2 Normalise
        │
    Output (128)

Configuration is centralised in the :class:`ProjectionHeadConfig`
dataclass, which validates all hyper-parameters at construction time.

Example
-------
>>> import torch
>>> from src.models.projection_head import ProjectionHead, ProjectionHeadConfig
>>> cfg = ProjectionHeadConfig()
>>> head = ProjectionHead(cfg)
>>> z = head(torch.randn(4, 256))
>>> z.shape
torch.Size([4, 128])
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict

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
class ProjectionHeadConfig:
    """Hyper-parameters for :class:`ProjectionHead`.

    Parameters
    ----------
    input_dim : int
        Dimensionality of the incoming encoder embedding.
    hidden_dim : int
        Width of the hidden layers inside the projection MLP.
    output_dim : int
        Dimensionality of the projected embedding (contrastive
        latent space).
    dropout : float
        Dropout probability applied after each activation.
    bias : bool
        Whether the linear layers include a learnable bias.
    activation : str
        Name of the activation function.  Must be one of
        ``'gelu'``, ``'relu'``, ``'silu'``, ``'tanh'``.

    Raises
    ------
    ValueError
        If any hyper-parameter fails validation.
    """

    input_dim: int = 256
    hidden_dim: int = 256
    output_dim: int = 128
    dropout: float = 0.2
    bias: bool = True
    activation: str = "gelu"

    def __post_init__(self) -> None:
        """Validate all configuration fields."""
        if self.input_dim < 1:
            raise ValueError(
                f"input_dim must be positive, got {self.input_dim}"
            )
        if self.hidden_dim < 1:
            raise ValueError(
                f"hidden_dim must be positive, got {self.hidden_dim}"
            )
        if self.output_dim < 1:
            raise ValueError(
                f"output_dim must be positive, got {self.output_dim}"
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
# Projection Head
# ======================================================================


class ProjectionHead(nn.Module):
    """MLP projection head producing L2-normalised embeddings.

    The module transforms encoder embeddings into a contrastive latent
    space via a three-layer MLP (two hidden layers with LayerNorm,
    activation, and dropout, followed by a final linear projection).
    The output is L2-normalised.

    Parameters
    ----------
    config : ProjectionHeadConfig
        Full projection head configuration.

    Raises
    ------
    TypeError
        If *config* is not a :class:`ProjectionHeadConfig`.
    """

    def __init__(self, config: ProjectionHeadConfig) -> None:
        if not isinstance(config, ProjectionHeadConfig):
            raise TypeError(
                f"config must be a ProjectionHeadConfig, "
                f"got {type(config).__name__}"
            )
        super().__init__()

        # Store the dataclass in __dict__ directly so TorchScript does
        # not attempt to compile it.  Only JIT-visible primitives are
        # kept as proper attributes.
        self.__dict__["_config"] = config
        self._input_dim: int = config.input_dim
        self._output_dim: int = config.output_dim

        # Resolve activation class from registry.
        activation_cls = _ACTIVATION_REGISTRY[config.activation.lower()]

        # ---- projection MLP ---------------------------------------------
        self.projection = nn.Sequential(
            # Hidden layer 1
            nn.Linear(
                config.input_dim,
                config.hidden_dim,
                bias=config.bias,
            ),
            nn.LayerNorm(config.hidden_dim),
            activation_cls(),
            nn.Dropout(p=config.dropout),
            # Hidden layer 2
            nn.Linear(
                config.hidden_dim,
                config.hidden_dim,
                bias=config.bias,
            ),
            nn.LayerNorm(config.hidden_dim),
            activation_cls(),
            nn.Dropout(p=config.dropout),
            # Final projection
            nn.Linear(
                config.hidden_dim,
                config.output_dim,
                bias=config.bias,
            ),
        )

        # ---- logging ----------------------------------------------------
        total_params, trainable_params = self._count_parameters()
        logger.info(
            "ProjectionHead initialised | "
            "input_dim=%d | hidden_dim=%d | output_dim=%d | "
            "dropout=%.2f | bias=%s | activation=%s | "
            "total_params=%s | trainable_params=%s",
            config.input_dim,
            config.hidden_dim,
            config.output_dim,
            config.dropout,
            config.bias,
            config.activation,
            f"{total_params:,}",
            f"{trainable_params:,}",
        )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    @torch.jit.ignore
    def config(self) -> ProjectionHeadConfig:
        """Return the projection head configuration."""
        return self.__dict__["_config"]

    @property
    @torch.jit.ignore
    def input_dim(self) -> int:
        """Return the expected input embedding dimensionality."""
        return self._input_dim

    @property
    @torch.jit.ignore
    def output_dim(self) -> int:
        """Return the projected embedding dimensionality."""
        return self._output_dim

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
    def _validate_embedding(self, embedding: torch.Tensor) -> None:
        """Validate the input embedding tensor.

        Parameters
        ----------
        embedding : torch.Tensor
            Expected shape ``(B, D_in)`` with floating-point dtype and
            no NaN or Inf values.

        Raises
        ------
        TypeError
            If *embedding* is not a :class:`torch.Tensor`.
        ValueError
            If *embedding* fails shape, dtype, or value checks.
        """
        if not isinstance(embedding, torch.Tensor):
            raise TypeError(
                f"embedding must be a torch.Tensor, "
                f"got {type(embedding).__name__}"
            )
        if not embedding.is_floating_point():
            raise ValueError(
                f"embedding must have a floating-point dtype, "
                f"got {embedding.dtype}"
            )
        if embedding.ndim != 2:
            raise ValueError(
                f"embedding must have exactly 2 dimensions (B, D), "
                f"got {embedding.ndim} dimensions with shape "
                f"{tuple(embedding.shape)}"
            )
        if embedding.shape[1] != self._input_dim:
            raise ValueError(
                f"embedding dimension (dim 1) must be "
                f"{self._input_dim}, got {embedding.shape[1]}"
            )
        if torch.isnan(embedding).any():
            raise ValueError("embedding contains NaN values")
        if torch.isinf(embedding).any():
            raise ValueError("embedding contains Inf values")

    # ------------------------------------------------------------------
    # Public methods
    # ------------------------------------------------------------------

    def forward(self, embedding: torch.Tensor) -> torch.Tensor:
        """Project an encoder embedding into the contrastive latent space.

        Parameters
        ----------
        embedding : torch.Tensor
            Encoder embedding of shape ``(B, D_in)`` where *D_in* is
            ``input_dim``.

        Returns
        -------
        torch.Tensor
            L2-normalised projected embedding of shape
            ``(B, D_out)`` where *D_out* is ``output_dim``.

        Raises
        ------
        TypeError
            If *embedding* is not a :class:`torch.Tensor`.
        ValueError
            If *embedding* fails shape, dtype, or value validation.
        """
        self._validate_embedding(embedding)
        logger.debug(
            "ProjectionHead forward | "
            "input shape=%s | dtype=%s",
            tuple(embedding.shape),
            embedding.dtype,
        )

        projected: torch.Tensor = self.projection(embedding)
        projected = F.normalize(projected, p=2, dim=-1)

        logger.debug(
            "ProjectionHead forward | output shape=%s",
            tuple(projected.shape),
        )
        return projected

    def project(self, embedding: torch.Tensor) -> torch.Tensor:
        """Project without computing gradients (inference convenience).

        Sets the module to evaluation mode temporarily, runs
        :meth:`forward` under :func:`torch.no_grad`, and restores the
        original training state.

        Parameters
        ----------
        embedding : torch.Tensor
            Encoder embedding of shape ``(B, D_in)``.

        Returns
        -------
        torch.Tensor
            L2-normalised projected embedding of shape
            ``(B, D_out)``.
        """
        was_training = self.training
        self.eval()
        try:
            with torch.no_grad():
                return self.forward(embedding)
        finally:
            if was_training:
                self.train()

    def parameter_count(self) -> Dict[str, int]:
        """Return parameter counts for the projection head.

        Returns
        -------
        dict[str, int]
            Dictionary with keys ``"total"`` and ``"trainable"``.
        """
        total, trainable = self._count_parameters()
        return {
            "total": total,
            "trainable": trainable,
        }
