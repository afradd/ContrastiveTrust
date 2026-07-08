"""Physics Consistency Loss for physics-guided contrastive regularisation.

This module implements the **Physics Consistency Loss** — the primary
physics-guided regularisation component of ContrastiveTrust.

Its purpose is to encourage the learned latent representations to remain
physically meaningful while the NT-Xent contrastive objective drives
discriminability.

Architecture (Strategy Pattern)
--------------------------------
::

    BaseConsistencyLoss          <- abstract base class
           |
    +------+---------------------------+
    |             |          |         |
    CosineConsistency  MSEConsistency  HuberConsistency  HybridConsistency
    Loss               Loss            Loss               Loss
    +------------------------------------------------------------+
                           |
                  PhysicsConsistencyLoss
              (factory / strategy wrapper)

Design Principles
-----------------
* **Open/Closed Principle** -- new strategies require only a new class and a
  single registry call; no existing code changes.
* **Strategy Pattern** -- the consistency metric is an injected collaborator,
  not a branching conditional.
* **Factory Pattern** -- :class:`PhysicsConsistencyLoss` selects and
  instantiates the appropriate strategy from configuration.
* **Dependency Injection** -- ``allow_custom_strategy`` enables callers to
  register domain-specific strategies without forking the codebase.

Usage
-----
>>> import torch
>>> import torch.nn.functional as F
>>> from src.losses.physics_consistency import PhysicsConsistencyLoss, PhysicsConsistencyConfig
>>> cfg = PhysicsConsistencyConfig(mode="cosine")
>>> loss_fn = PhysicsConsistencyLoss(cfg)
>>> B, D = 8, 256
>>> enc = F.normalize(torch.randn(B, D), p=2, dim=1)
>>> phy = F.normalize(torch.randn(B, D), p=2, dim=1)
>>> out = loss_fn(enc, phy)
>>> out["loss"].shape
torch.Size([])

References
----------
.. [1] He, K., et al. (2020). Momentum Contrast for Unsupervised Visual
       Representation Learning. CVPR.
.. [2] Gidaris, S., et al. (2018). Unsupervised Representation Learning
       by Predicting Image Rotations. ICLR.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Type

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


# ======================================================================
# Strategy Registry
# ======================================================================

# Maps strategy name -> BaseConsistencyLoss subclass.
# Future strategies register here; nothing else in the codebase changes.
_STRATEGY_REGISTRY: Dict[str, Type["BaseConsistencyLoss"]] = {}


def register_strategy(name: str):
    """Class decorator that registers a consistency strategy.

    Parameters
    ----------
    name : str
        Canonical lower-case key used in :class:`PhysicsConsistencyConfig`.

    Returns
    -------
    Callable
        The unmodified class (decorator does not alter the class).

    Examples
    --------
    >>> @register_strategy("my_custom")
    ... class MyCustomConsistencyLoss(BaseConsistencyLoss):
    ...     def compute(self, enc, phy): ...
    ...     @property
    ...     def metric_name(self): return "my_custom"
    """
    def _decorator(cls: Type["BaseConsistencyLoss"]):
        _STRATEGY_REGISTRY[name.lower()] = cls
        logger.debug("PhysicsConsistency | registered strategy '%s'", name)
        return cls
    return _decorator


# ======================================================================
# Configuration
# ======================================================================


@dataclass
class PhysicsConsistencyConfig:
    """Hyper-parameters for :class:`PhysicsConsistencyLoss`.

    Parameters
    ----------
    mode : str
        Active consistency metric.  One of ``"cosine"``, ``"mse"``,
        ``"huber"``, ``"hybrid"``, or any key registered via
        :func:`register_strategy`.  Default ``"cosine"``.
    cosine_weight : float
        Weight alpha applied to the cosine component in
        :class:`HybridConsistencyLoss`.  Must be in ``[0, 1]``.
        Default ``0.7``.
    mse_weight : float
        Weight beta applied to the MSE component in
        :class:`HybridConsistencyLoss`.  Must be in ``[0, 1]``.
        Default ``0.3``.
    reduction : str
        How to aggregate per-sample losses.  One of ``"mean"`` or
        ``"sum"``.  Default ``"mean"``.
    eps : float
        Small constant for numerical stability (e.g. cosine denominator).
        Must be strictly positive.  Default ``1e-8``.
    allow_custom_strategy : bool
        When ``True``, :meth:`PhysicsConsistencyLoss.set_metric` accepts
        strategy names beyond the four built-ins, provided they have been
        registered via :func:`register_strategy`.  Default ``True``.

    Raises
    ------
    ValueError
        If any parameter fails its constraint check.
    """

    mode: str = "cosine"
    cosine_weight: float = 0.7
    mse_weight: float = 0.3
    reduction: str = "mean"
    eps: float = 1e-8
    allow_custom_strategy: bool = True

    # Populated after __post_init__; excluded from constructor signature.
    _resolved_mode: str = field(default="", init=False, repr=False)

    def __post_init__(self) -> None:
        """Validate all configuration fields."""
        if not self.mode:
            raise ValueError("mode must be a non-empty string")
        if not (0.0 <= self.cosine_weight <= 1.0):
            raise ValueError(
                f"cosine_weight must be in [0, 1], got {self.cosine_weight}"
            )
        if not (0.0 <= self.mse_weight <= 1.0):
            raise ValueError(
                f"mse_weight must be in [0, 1], got {self.mse_weight}"
            )
        if self.reduction not in {"mean", "sum"}:
            raise ValueError(
                f"reduction must be 'mean' or 'sum', got '{self.reduction}'"
            )
        if self.eps <= 0.0:
            raise ValueError(
                f"eps must be strictly positive, got {self.eps}"
            )
        # Use object.__setattr__ because dataclass may be used in a frozen
        # context by subclasses; this is a computed, non-init field.
        object.__setattr__(self, "_resolved_mode", self.mode.lower())


# ======================================================================
# Abstract Base Class
# ======================================================================


class BaseConsistencyLoss(ABC):
    """Abstract base for all physics-consistency strategies.

    Each concrete strategy must implement :meth:`compute` and expose a
    :attr:`metric_name` property.  The base class provides shared
    reduction logic via :meth:`_reduce`.

    Parameters
    ----------
    config : PhysicsConsistencyConfig
        Shared configuration (reduction, eps, weights, ...).
    """

    def __init__(self, config: PhysicsConsistencyConfig) -> None:
        self._config = config
        self._reduction: str = config.reduction
        self._eps: float = config.eps

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    @abstractmethod
    def compute(
        self,
        encoder_embedding: torch.Tensor,
        physics_embedding: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the per-batch consistency loss scalar.

        Parameters
        ----------
        encoder_embedding : torch.Tensor
            L2-normalised encoder representation of shape ``(B, D)``.
        physics_embedding : torch.Tensor
            L2-normalised physics representation of shape ``(B, D)``.

        Returns
        -------
        torch.Tensor
            Scalar loss value (already reduced).
        """

    @property
    @abstractmethod
    def metric_name(self) -> str:
        """Short human-readable identifier for this strategy."""

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------

    def _reduce(self, per_sample: torch.Tensor) -> torch.Tensor:
        """Aggregate a per-sample 1-D tensor to a scalar.

        Parameters
        ----------
        per_sample : torch.Tensor
            1-D tensor of shape ``(B,)`` containing per-sample losses.

        Returns
        -------
        torch.Tensor
            Scalar reduced loss.
        """
        if self._reduction == "mean":
            return per_sample.mean()
        return per_sample.sum()


# ======================================================================
# Concrete Strategies
# ======================================================================


@register_strategy("cosine")
class CosineConsistencyLoss(BaseConsistencyLoss):
    """Cosine-distance consistency loss.

    Minimises the angular divergence between encoder and physics
    embeddings::

        L = mean(1 - cosine_similarity(enc, phy))

    Because both embeddings are already L2-normalised, cosine similarity
    reduces to the dot product ``enc . phy``, but an explicit
    ``F.normalize`` guard is applied for mixed-precision safety.

    Parameters
    ----------
    config : PhysicsConsistencyConfig
        Shared configuration (``reduction``, ``eps``).
    """

    def __init__(self, config: PhysicsConsistencyConfig) -> None:
        super().__init__(config)

    @property
    def metric_name(self) -> str:
        """Return ``'cosine'``."""
        return "cosine"

    def compute(
        self,
        encoder_embedding: torch.Tensor,
        physics_embedding: torch.Tensor,
    ) -> torch.Tensor:
        """Compute mean cosine-distance loss.

        Parameters
        ----------
        encoder_embedding : torch.Tensor
            L2-normalised tensor of shape ``(B, D)``.
        physics_embedding : torch.Tensor
            L2-normalised tensor of shape ``(B, D)``.

        Returns
        -------
        torch.Tensor
            Scalar in ``[0, 2]``; zero when embeddings are identical.
        """
        enc = F.normalize(encoder_embedding, p=2, dim=1, eps=self._eps)
        phy = F.normalize(physics_embedding, p=2, dim=1, eps=self._eps)
        # Cosine similarity per sample: (B,)
        cos_sim = (enc * phy).sum(dim=1)
        per_sample = 1.0 - cos_sim
        return self._reduce(per_sample)


@register_strategy("mse")
class MSEConsistencyLoss(BaseConsistencyLoss):
    """Mean-squared-error consistency loss.

    Measures Euclidean proximity between encoder and physics embeddings::

        L = mean( ||enc - phy||^2 )    (mean over embedding dim)

    Parameters
    ----------
    config : PhysicsConsistencyConfig
        Shared configuration (``reduction``).
    """

    def __init__(self, config: PhysicsConsistencyConfig) -> None:
        super().__init__(config)

    @property
    def metric_name(self) -> str:
        """Return ``'mse'``."""
        return "mse"

    def compute(
        self,
        encoder_embedding: torch.Tensor,
        physics_embedding: torch.Tensor,
    ) -> torch.Tensor:
        """Compute mean-squared-error loss.

        Parameters
        ----------
        encoder_embedding : torch.Tensor
            L2-normalised tensor of shape ``(B, D)``.
        physics_embedding : torch.Tensor
            L2-normalised tensor of shape ``(B, D)``.

        Returns
        -------
        torch.Tensor
            Non-negative scalar; zero when embeddings are identical.
        """
        diff = encoder_embedding - physics_embedding
        # Per-sample squared L2 norm averaged across the embedding dim (B,)
        per_sample = (diff * diff).mean(dim=1)
        return self._reduce(per_sample)


@register_strategy("huber")
class HuberConsistencyLoss(BaseConsistencyLoss):
    """Huber (smooth-L1) consistency loss.

    Combines the quadratic regime of MSE for small errors with the linear
    regime of MAE for large errors, providing robustness against outlier
    physics predictions::

        L = smooth_l1(enc, phy)   (element-wise, then reduced)

    Parameters
    ----------
    config : PhysicsConsistencyConfig
        Shared configuration (``reduction``).
    """

    def __init__(self, config: PhysicsConsistencyConfig) -> None:
        super().__init__(config)

    @property
    def metric_name(self) -> str:
        """Return ``'huber'``."""
        return "huber"

    def compute(
        self,
        encoder_embedding: torch.Tensor,
        physics_embedding: torch.Tensor,
    ) -> torch.Tensor:
        """Compute Huber (smooth L1) loss.

        Parameters
        ----------
        encoder_embedding : torch.Tensor
            L2-normalised tensor of shape ``(B, D)``.
        physics_embedding : torch.Tensor
            L2-normalised tensor of shape ``(B, D)``.

        Returns
        -------
        torch.Tensor
            Non-negative scalar.
        """
        # F.smooth_l1_loss with reduction="none" -> (B, D)
        element_wise = F.smooth_l1_loss(
            encoder_embedding,
            physics_embedding,
            reduction="none",
        )
        per_sample = element_wise.mean(dim=1)   # (B,)
        return self._reduce(per_sample)


@register_strategy("hybrid")
class HybridConsistencyLoss(BaseConsistencyLoss):
    """Weighted combination of cosine and MSE consistency losses.

    Allows simultaneous optimisation of angular alignment (cosine) and
    Euclidean proximity (MSE) with independently configurable weights::

        L = alpha * CosineConsistencyLoss + beta * MSEConsistencyLoss

    where ``alpha = cosine_weight`` and ``beta = mse_weight``.

    Parameters
    ----------
    config : PhysicsConsistencyConfig
        Shared configuration.  ``cosine_weight`` and ``mse_weight``
        control the blend.

    Notes
    -----
    The cosine and MSE sub-losses are computed independently to ensure
    each receives its own gradients.  There is no automatic weight
    normalisation; the user is responsible for choosing weight values
    appropriate for their loss scale.
    """

    def __init__(self, config: PhysicsConsistencyConfig) -> None:
        super().__init__(config)
        self._cosine_weight: float = config.cosine_weight
        self._mse_weight: float = config.mse_weight
        self._cosine = CosineConsistencyLoss(config)
        self._mse = MSEConsistencyLoss(config)

    @property
    def metric_name(self) -> str:
        """Return ``'hybrid'``."""
        return "hybrid"

    def compute(
        self,
        encoder_embedding: torch.Tensor,
        physics_embedding: torch.Tensor,
    ) -> torch.Tensor:
        """Compute alpha * CosineLoss + beta * MSELoss.

        Parameters
        ----------
        encoder_embedding : torch.Tensor
            L2-normalised tensor of shape ``(B, D)``.
        physics_embedding : torch.Tensor
            L2-normalised tensor of shape ``(B, D)``.

        Returns
        -------
        torch.Tensor
            Non-negative scalar.
        """
        cosine_val = self._cosine.compute(encoder_embedding, physics_embedding)
        mse_val = self._mse.compute(encoder_embedding, physics_embedding)
        return self._cosine_weight * cosine_val + self._mse_weight * mse_val


# ======================================================================
# PhysicsConsistencyLoss  (factory / strategy wrapper)
# ======================================================================


class PhysicsConsistencyLoss(nn.Module):
    """Physics consistency loss -- factory wrapper over pluggable strategies.

    Selects the requested consistency strategy through configuration,
    validates inputs at runtime, and returns a standardised output dict.

    The *Trainer* never requires modification when a new strategy is added;
    only :func:`register_strategy` and a new class are required.

    Parameters
    ----------
    config : PhysicsConsistencyConfig
        Loss configuration.  ``config.mode`` selects the strategy.

    Raises
    ------
    TypeError
        If *config* is not a :class:`PhysicsConsistencyConfig`.
    ValueError
        If ``config.mode`` refers to an unregistered strategy.

    Examples
    --------
    >>> import torch, torch.nn.functional as F
    >>> from src.losses.physics_consistency import (
    ...     PhysicsConsistencyLoss, PhysicsConsistencyConfig
    ... )
    >>> loss_fn = PhysicsConsistencyLoss(PhysicsConsistencyConfig(mode="cosine"))
    >>> enc = F.normalize(torch.randn(4, 256), p=2, dim=1)
    >>> phy = F.normalize(torch.randn(4, 256), p=2, dim=1)
    >>> out = loss_fn(enc, phy)
    >>> out["loss"].shape
    torch.Size([])
    >>> out["metric"]
    'cosine'
    """

    def __init__(self, config: PhysicsConsistencyConfig) -> None:
        if not isinstance(config, PhysicsConsistencyConfig):
            raise TypeError(
                f"config must be a PhysicsConsistencyConfig, "
                f"got {type(config).__name__}"
            )
        super().__init__()

        # Store config outside TorchScript-visible scope.
        self.__dict__["_config"] = config

        # Build and hold the active strategy.
        self._strategy: BaseConsistencyLoss = self._build_strategy(
            config._resolved_mode, config
        )

        logger.info(
            "PhysicsConsistencyLoss initialised | "
            "mode=%s | reduction=%s | eps=%.2e | "
            "cosine_weight=%.2f | mse_weight=%.2f",
            config.mode,
            config.reduction,
            config.eps,
            config.cosine_weight,
            config.mse_weight,
        )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    @torch.jit.ignore
    def config(self) -> PhysicsConsistencyConfig:
        """Return the active configuration."""
        return self.__dict__["_config"]

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @torch.jit.ignore
    def _build_strategy(
        self,
        mode: str,
        config: PhysicsConsistencyConfig,
    ) -> BaseConsistencyLoss:
        """Instantiate the strategy for *mode*.

        Parameters
        ----------
        mode : str
            Lower-case strategy key.
        config : PhysicsConsistencyConfig
            Configuration forwarded to the strategy constructor.

        Returns
        -------
        BaseConsistencyLoss
            Concrete strategy instance.

        Raises
        ------
        ValueError
            If *mode* is not found in :data:`_STRATEGY_REGISTRY`.
        """
        if mode not in _STRATEGY_REGISTRY:
            available = sorted(_STRATEGY_REGISTRY.keys())
            raise ValueError(
                f"Unknown consistency mode '{mode}'. "
                f"Available strategies: {available}. "
                f"Register custom strategies with @register_strategy."
            )
        strategy_cls = _STRATEGY_REGISTRY[mode]
        strategy = strategy_cls(config)
        logger.debug(
            "PhysicsConsistencyLoss | built strategy '%s' -> %s",
            mode,
            type(strategy).__name__,
        )
        return strategy

    @torch.jit.ignore
    def _validate_inputs(
        self,
        encoder_embedding: torch.Tensor,
        physics_embedding: torch.Tensor,
    ) -> None:
        """Validate encoder and physics embedding tensors.

        Parameters
        ----------
        encoder_embedding : torch.Tensor
            Encoder representation; expected shape ``(B, D)``.
        physics_embedding : torch.Tensor
            Physics representation; expected shape ``(B, D)``.

        Raises
        ------
        TypeError
            If either argument is not a :class:`torch.Tensor`.
        ValueError
            If shape, dtype, or value constraints are violated.
        """
        pairs = [
            ("encoder_embedding", encoder_embedding),
            ("physics_embedding", physics_embedding),
        ]
        for name, emb in pairs:
            if not isinstance(emb, torch.Tensor):
                raise TypeError(
                    f"{name} must be a torch.Tensor, "
                    f"got {type(emb).__name__}"
                )
            if not emb.is_floating_point():
                raise ValueError(
                    f"{name} must have a floating-point dtype, "
                    f"got {emb.dtype}"
                )
            if emb.ndim != 2:
                raise ValueError(
                    f"{name} must have exactly 2 dimensions (B, D), "
                    f"got {emb.ndim} dimensions with shape "
                    f"{tuple(emb.shape)}"
                )
            if torch.isnan(emb).any():
                raise ValueError(f"{name} contains NaN values")
            if torch.isinf(emb).any():
                raise ValueError(f"{name} contains Inf values")

        if encoder_embedding.shape[0] != physics_embedding.shape[0]:
            raise ValueError(
                f"encoder_embedding and physics_embedding must have the "
                f"same batch size, got "
                f"encoder_embedding.shape={tuple(encoder_embedding.shape)}, "
                f"physics_embedding.shape={tuple(physics_embedding.shape)}"
            )
        if encoder_embedding.shape[1] != physics_embedding.shape[1]:
            raise ValueError(
                f"encoder_embedding and physics_embedding must have the "
                f"same embedding dimension, got "
                f"encoder_embedding.shape={tuple(encoder_embedding.shape)}, "
                f"physics_embedding.shape={tuple(physics_embedding.shape)}"
            )

        logger.debug(
            "PhysicsConsistencyLoss | validated inputs: "
            "encoder_embedding=%s, physics_embedding=%s, dtype=%s",
            tuple(encoder_embedding.shape),
            tuple(physics_embedding.shape),
            encoder_embedding.dtype,
        )

    @torch.jit.ignore
    def _normalize_output(
        self,
        loss: torch.Tensor,
    ) -> Dict[str, object]:
        """Build the standardised output dictionary.

        Parameters
        ----------
        loss : torch.Tensor
            Scalar loss value produced by the active strategy.

        Returns
        -------
        dict
            ``{"loss": Tensor, "metric": str, "value": Tensor}``.
        """
        return {
            "loss": loss,
            "metric": self._strategy.metric_name,
            "value": loss.detach(),
        }

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @torch.jit.ignore
    def available_metrics(self) -> List[str]:
        """Return a sorted list of all registered strategy names.

        Returns
        -------
        list[str]
            Sorted list of registered consistency metric keys.

        Examples
        --------
        >>> from src.losses.physics_consistency import PhysicsConsistencyLoss, PhysicsConsistencyConfig
        >>> fn = PhysicsConsistencyLoss(PhysicsConsistencyConfig())
        >>> "cosine" in fn.available_metrics()
        True
        """
        return sorted(_STRATEGY_REGISTRY.keys())

    @torch.jit.ignore
    def current_metric(self) -> str:
        """Return the name of the currently active consistency metric.

        Returns
        -------
        str
            Strategy key (e.g. ``"cosine"``).

        Examples
        --------
        >>> from src.losses.physics_consistency import PhysicsConsistencyLoss, PhysicsConsistencyConfig
        >>> fn = PhysicsConsistencyLoss(PhysicsConsistencyConfig(mode="mse"))
        >>> fn.current_metric()
        'mse'
        """
        return self._strategy.metric_name

    @torch.jit.ignore
    def set_metric(self, mode: str) -> None:
        """Switch the active consistency strategy at runtime.

        Parameters
        ----------
        mode : str
            Key of the strategy to activate.  Must be registered in
            :data:`_STRATEGY_REGISTRY`.  If ``config.allow_custom_strategy``
            is ``False``, must be one of the four built-in modes.

        Raises
        ------
        ValueError
            If *mode* is not registered, or if custom strategies are
            disabled and *mode* is not a built-in.

        Examples
        --------
        >>> from src.losses.physics_consistency import PhysicsConsistencyLoss, PhysicsConsistencyConfig
        >>> fn = PhysicsConsistencyLoss(PhysicsConsistencyConfig())
        >>> fn.set_metric("mse")
        >>> fn.current_metric()
        'mse'
        """
        mode_lower = mode.lower()
        built_ins = {"cosine", "mse", "huber", "hybrid"}

        config: PhysicsConsistencyConfig = self.__dict__["_config"]
        if not config.allow_custom_strategy and mode_lower not in built_ins:
            raise ValueError(
                f"Custom strategy '{mode_lower}' is not permitted "
                f"(allow_custom_strategy=False). "
                f"Built-in modes: {sorted(built_ins)}"
            )

        self._strategy = self._build_strategy(mode_lower, config)
        logger.info(
            "PhysicsConsistencyLoss | strategy switched to '%s'",
            mode_lower,
        )

    def forward(
        self,
        encoder_embedding: torch.Tensor,
        physics_embedding: torch.Tensor,
    ) -> Dict[str, object]:
        """Compute the physics consistency loss.

        Parameters
        ----------
        encoder_embedding : torch.Tensor
            L2-normalised encoder representation of shape ``(B, D)``.
        physics_embedding : torch.Tensor
            L2-normalised physics representation of shape ``(B, D)``.

        Returns
        -------
        dict
            A standardised output dictionary:

            ``"loss"`` : torch.Tensor
                Scalar physics consistency loss (gradient-enabled).
            ``"metric"`` : str
                Name of the active consistency strategy.
            ``"value"`` : torch.Tensor
                Detached scalar for logging / monitoring.

        Raises
        ------
        TypeError
            If either input is not a :class:`torch.Tensor`.
        ValueError
            If input validation fails (dtype, shape, NaN, Inf, ...).

        Notes
        -----
        Both inputs must share the same batch size and embedding dimension.
        The ``value`` key contains a detached copy of ``loss`` intended
        for tensorboard / metric loggers that must not retain the graph.
        """
        self._validate_inputs(encoder_embedding, physics_embedding)

        batch_size = encoder_embedding.shape[0]
        logger.info(
            "PhysicsConsistencyLoss forward | "
            "batch_size=%d | metric=%s | "
            "encoder_embedding=%s | physics_embedding=%s",
            batch_size,
            self._strategy.metric_name,
            tuple(encoder_embedding.shape),
            tuple(physics_embedding.shape),
        )

        loss: torch.Tensor = self._strategy.compute(
            encoder_embedding, physics_embedding
        )

        logger.debug(
            "PhysicsConsistencyLoss forward | loss=%.6f",
            loss.item(),
        )

        return self._normalize_output(loss)

    @torch.jit.ignore
    def parameter_summary(self) -> Dict[str, object]:
        """Return a human-readable configuration summary.

        Returns
        -------
        dict
            Keys: ``"mode"``, ``"reduction"``, ``"eps"``,
            ``"cosine_weight"``, ``"mse_weight"``,
            ``"allow_custom_strategy"``, ``"num_parameters"``,
            ``"available_metrics"``.
        """
        config: PhysicsConsistencyConfig = self.__dict__["_config"]
        return {
            "mode": config.mode,
            "reduction": config.reduction,
            "eps": config.eps,
            "cosine_weight": config.cosine_weight,
            "mse_weight": config.mse_weight,
            "allow_custom_strategy": config.allow_custom_strategy,
            "num_parameters": sum(p.numel() for p in self.parameters()),
            "available_metrics": self.available_metrics(),
        }
