"""Data utilities for ContrastiveTrust.

This package exposes dataset loading, preprocessing, augmentation, and
contrastive view generation for multivariate industrial control-system
time series.

Public API
----------
Augmentation primitives
    :class:`GaussianNoise`, :class:`RandomScaling`, :class:`RandomJitter`,
    :class:`RandomTimeMask`, :class:`RandomChannelMask`,
    :class:`RandomTimeShift`, :class:`Identity`

Augmentation configuration dataclasses
    :class:`GaussianNoiseConfig`, :class:`RandomScalingConfig`,
    :class:`RandomJitterConfig`, :class:`RandomTimeMaskConfig`,
    :class:`RandomChannelMaskConfig`, :class:`RandomTimeShiftConfig`,
    :class:`IdentityConfig`

Composition
    :class:`Compose`, :class:`ComposeConfig`

Base class
    :class:`BaseAugmentation`

View generation
    :class:`ContrastiveViewGenerator`,
    :class:`ContrastiveViewGeneratorConfig`
"""

from src.data.augmentations import (
    BaseAugmentation,
    Compose,
    ComposeConfig,
    GaussianNoise,
    GaussianNoiseConfig,
    Identity,
    IdentityConfig,
    RandomChannelMask,
    RandomChannelMaskConfig,
    RandomJitter,
    RandomJitterConfig,
    RandomScaling,
    RandomScalingConfig,
    RandomTimeMask,
    RandomTimeMaskConfig,
    RandomTimeShift,
    RandomTimeShiftConfig,
)
from src.data.view_generator import (
    ContrastiveViewGenerator,
    ContrastiveViewGeneratorConfig,
)

__all__: list[str] = [
    # Base
    "BaseAugmentation",
    # Augmentations
    "Compose",
    "GaussianNoise",
    "Identity",
    "RandomChannelMask",
    "RandomJitter",
    "RandomScaling",
    "RandomTimeMask",
    "RandomTimeShift",
    # Configs
    "ComposeConfig",
    "GaussianNoiseConfig",
    "IdentityConfig",
    "RandomChannelMaskConfig",
    "RandomJitterConfig",
    "RandomScalingConfig",
    "RandomTimeMaskConfig",
    "RandomTimeShiftConfig",
    # View generator
    "ContrastiveViewGenerator",
    "ContrastiveViewGeneratorConfig",
]
