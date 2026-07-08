"""Loss functions for ContrastiveTrust.

This package provides contrastive and auxiliary loss functions used
during self-supervised pre-training of the dual-stream encoder.

Public API
----------
NTXentLoss
    NT-Xent (Normalized Temperature-scaled Cross-Entropy) loss for
    self-supervised contrastive learning (SimCLR formulation).
NTXentConfig
    Dataclass holding all hyper-parameters for :class:`NTXentLoss`.
PhysicsConsistencyLoss
    Physics-guided consistency regularisation loss.  Wraps a pluggable
    strategy (cosine, mse, huber, hybrid, or any custom strategy
    registered via :func:`register_strategy`).
PhysicsConsistencyConfig
    Dataclass holding all hyper-parameters for
    :class:`PhysicsConsistencyLoss`.
BaseConsistencyLoss
    Abstract base class for custom consistency strategies.
register_strategy
    Decorator that registers a new consistency strategy so it can be
    selected by name through :class:`PhysicsConsistencyConfig`.
ContrastiveTrustLoss
    Unified multi-objective criterion that orchestrates
    :class:`NTXentLoss` and :class:`PhysicsConsistencyLoss` into a single
    weighted training loss.
ContrastiveTrustLossConfig
    Dataclass holding all hyper-parameters for
    :class:`ContrastiveTrustLoss`.
"""

from src.losses.contrastive_trust_loss import (
    ContrastiveTrustLoss,
    ContrastiveTrustLossConfig,
)
from src.losses.nt_xent import NTXentConfig, NTXentLoss
from src.losses.physics_consistency import (
    BaseConsistencyLoss,
    PhysicsConsistencyConfig,
    PhysicsConsistencyLoss,
    register_strategy,
)

__all__: list[str] = [
    # NT-Xent contrastive loss
    "NTXentConfig",
    "NTXentLoss",
    # Physics consistency loss
    "BaseConsistencyLoss",
    "PhysicsConsistencyConfig",
    "PhysicsConsistencyLoss",
    "register_strategy",
    # Unified multi-objective loss
    "ContrastiveTrustLoss",
    "ContrastiveTrustLossConfig",
]
