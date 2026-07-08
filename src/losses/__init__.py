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
"""

from src.losses.nt_xent import NTXentConfig, NTXentLoss

__all__: list[str] = [
    "NTXentConfig",
    "NTXentLoss",
]
