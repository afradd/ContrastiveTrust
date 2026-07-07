"""Neural network models for ContrastiveTrust.

This package provides modular, reusable encoder architectures for
self-supervised contrastive learning on multivariate industrial
control-system time series.

Public API
----------
ConvNormActivation
    Fused Conv1D → BatchNorm1d → GELU building block.
ResidualConvBlock
    Two-layer residual convolutional block with optional projection.
TemporalEncoder
    CNN-based temporal encoder producing L2-normalised embeddings.
TemporalEncoderConfig
    Dataclass holding all hyper-parameters for :class:`TemporalEncoder`.
PhysicsEncoder
    MLP-based physics encoder producing L2-normalised embeddings.
PhysicsEncoderConfig
    Dataclass holding all hyper-parameters for :class:`PhysicsEncoder`.
FeatureFusion
    Residual gated fusion combining temporal and physics embeddings.
FusionConfig
    Dataclass holding all hyper-parameters for :class:`FeatureFusion`.
DualStreamEncoder
    Dual-stream encoder orchestrating temporal, physics, and fusion.
EncoderConfig
    Dataclass composing sub-module configs for :class:`DualStreamEncoder`.
ProjectionHead
    MLP projection head for contrastive latent space.
ProjectionHeadConfig
    Dataclass holding all hyper-parameters for :class:`ProjectionHead`.
"""

from src.models.blocks import ConvNormActivation, ResidualConvBlock
from src.models.encoder import DualStreamEncoder, EncoderConfig
from src.models.fusion import FeatureFusion, FusionConfig
from src.models.physics_encoder import PhysicsEncoder, PhysicsEncoderConfig
from src.models.projection_head import ProjectionHead, ProjectionHeadConfig
from src.models.temporal_encoder import TemporalEncoder, TemporalEncoderConfig

__all__: list[str] = [
    "ConvNormActivation",
    "DualStreamEncoder",
    "EncoderConfig",
    "FeatureFusion",
    "FusionConfig",
    "PhysicsEncoder",
    "PhysicsEncoderConfig",
    "ProjectionHead",
    "ProjectionHeadConfig",
    "ResidualConvBlock",
    "TemporalEncoder",
    "TemporalEncoderConfig",
]
