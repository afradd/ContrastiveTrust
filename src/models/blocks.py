"""Reusable neural network building blocks for temporal encoders.

This module provides composable 1-D convolutional blocks used throughout
the ContrastiveTrust encoder architectures:

* :class:`ConvNormActivation` — a fused Conv1D → Norm → Activation unit.
* :class:`ResidualConvBlock` — a two-layer residual block with optional
  channel projection, dropout, and configurable normalisation.

All blocks operate on **channels-first** tensors of shape ``(B, C, T)``.
"""

from __future__ import annotations

import logging
from typing import Optional, Type

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


# ======================================================================
# ConvNormActivation
# ======================================================================


class ConvNormActivation(nn.Module):
    """Fused Conv1D → BatchNorm1d → Activation block.

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    kernel_size : int
        Size of the convolving kernel.
    stride : int
        Stride of the convolution.
    padding : int or None
        Zero-padding added to both sides.  When *None* the padding is
        computed automatically as ``(kernel_size - 1) * dilation // 2``
        to preserve the temporal dimension (same padding).
    dilation : int
        Spacing between kernel elements.
    bias : bool
        If *True*, add a learnable bias to the convolution (typically
        *False* when followed by batch-norm).
    activation : type[nn.Module] or None
        Activation class to instantiate.  Pass *None* to omit the
        activation (e.g. before a residual addition).

    Raises
    ------
    ValueError
        If *in_channels* or *out_channels* is not positive, or if
        *kernel_size* is not a positive odd integer.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: Optional[int] = None,
        dilation: int = 1,
        bias: bool = False,
        activation: Optional[Type[nn.Module]] = nn.GELU,
    ) -> None:
        super().__init__()

        # ---- validation ------------------------------------------------
        if in_channels < 1:
            raise ValueError(
                f"in_channels must be positive, got {in_channels}"
            )
        if out_channels < 1:
            raise ValueError(
                f"out_channels must be positive, got {out_channels}"
            )
        if kernel_size < 1 or kernel_size % 2 == 0:
            raise ValueError(
                f"kernel_size must be a positive odd integer, got {kernel_size}"
            )

        if padding is None:
            padding = (kernel_size - 1) * dilation // 2

        # ---- layers -----------------------------------------------------
        layers: list[nn.Module] = [
            nn.Conv1d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                bias=bias,
            ),
            nn.BatchNorm1d(out_channels),
        ]
        if activation is not None:
            layers.append(activation())

        self.block = nn.Sequential(*layers)

    # -----------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply convolution → normalisation → activation.

        Parameters
        ----------
        x : torch.Tensor
            Input of shape ``(B, C_in, T)``.

        Returns
        -------
        torch.Tensor
            Output of shape ``(B, C_out, T')``.
        """
        return self.block(x)


# ======================================================================
# ResidualConvBlock
# ======================================================================


class ResidualConvBlock(nn.Module):
    """Two-layer residual convolutional block.

    Architecture::

        Input ─┬─────────────────────────────────┐
               │                                  │
               ▼                                  │ (projection if
        ConvNormActivation(in → out, act)         │  channels differ)
               │                                  │
               ▼                                  │
        ConvNormActivation(out → out, NO act)     │
               │                                  │
               ▼                                  ▼
              (+)────────────────────────────── shortcut
               │
               ▼
           Activation
               │
               ▼ (optional Dropout)
            Output

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    kernel_size : int
        Kernel size for both convolutions (must be positive odd).
    dropout : float
        Dropout probability applied after the residual addition
        and activation.  Set to ``0.0`` to disable.
    activation : type[nn.Module]
        Activation class applied after each convolution and after
        the residual merge.
    norm_layer : type[nn.Module]
        Not used for inner convolution norms (which are always
        ``BatchNorm1d``).  Reserved for future extensibility.
    bias : bool
        Whether to use bias in convolutions.

    Raises
    ------
    ValueError
        If *in_channels* or *out_channels* is non-positive, if
        *kernel_size* is not a positive odd integer, or if *dropout*
        is outside ``[0, 1)``.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        dropout: float = 0.0,
        activation: Type[nn.Module] = nn.GELU,
        norm_layer: Type[nn.Module] = nn.BatchNorm1d,
        bias: bool = False,
    ) -> None:
        super().__init__()

        # ---- validation ------------------------------------------------
        if in_channels < 1:
            raise ValueError(
                f"in_channels must be positive, got {in_channels}"
            )
        if out_channels < 1:
            raise ValueError(
                f"out_channels must be positive, got {out_channels}"
            )
        if kernel_size < 1 or kernel_size % 2 == 0:
            raise ValueError(
                f"kernel_size must be a positive odd integer, got {kernel_size}"
            )
        if not 0.0 <= dropout < 1.0:
            raise ValueError(
                f"dropout must be in [0, 1), got {dropout}"
            )

        # ---- main path --------------------------------------------------
        self.conv1 = ConvNormActivation(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            bias=bias,
            activation=activation,
        )
        self.conv2 = ConvNormActivation(
            out_channels,
            out_channels,
            kernel_size=kernel_size,
            bias=bias,
            activation=None,  # activation applied after residual add
        )

        # ---- shortcut / projection -------------------------------------
        if in_channels != out_channels:
            self.shortcut: nn.Module = nn.Sequential(
                nn.Conv1d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    bias=False,
                ),
                nn.BatchNorm1d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()

        # ---- post-merge -------------------------------------------------
        self.activation = activation()
        self.dropout = nn.Dropout(p=dropout) if dropout > 0.0 else nn.Identity()

    # -----------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply residual convolution block.

        Parameters
        ----------
        x : torch.Tensor
            Input of shape ``(B, C_in, T)``.

        Returns
        -------
        torch.Tensor
            Output of shape ``(B, C_out, T)``.
        """
        identity = self.shortcut(x)
        out = self.conv1(x)
        out = self.conv2(out)
        out = out + identity
        out = self.activation(out)
        out = self.dropout(out)
        return out
