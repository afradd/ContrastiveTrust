"""Unified multi-objective training criterion for ContrastiveTrust.

This module implements :class:`ContrastiveTrustLoss` -- the top-level
optimisation objective that *orchestrates* the two lower-level losses of
the ContrastiveTrust pipeline:

* :class:`~src.losses.nt_xent.NTXentLoss`
  -- the self-supervised contrastive objective (SimCLR / NT-Xent) that
  drives discriminability of the projected views.
* :class:`~src.losses.physics_consistency.PhysicsConsistencyLoss`
  -- the physics-guided regulariser that keeps the learned latent space
  physically meaningful.

The unified loss is a (optionally normalised, optionally learnable)
weighted sum::

    L = w_c * L_contrastive + w_p * L_physics

Design principles
-----------------
* **Composition over reimplementation** -- the sub-losses are injected
  collaborators (dependency injection); their mathematics live in their
  own modules and are never duplicated here.
* **Open/Closed** -- objectives are keyed by name in an internal
  registry (:attr:`ContrastiveTrustLoss.objective_names`).  Adding a
  third objective requires only a new weight entry, a new config field,
  and one line in :meth:`forward`; the aggregation, normalisation,
  freezing, and serialisation logic are objective-agnostic.
* **Differentiable & mixed-precision safe** -- weights are cast to the
  incoming loss dtype/device before multiplication, so the criterion
  composes cleanly with ``torch.autocast`` and fp16/bf16 training.

Example
-------
>>> import torch
>>> import torch.nn.functional as F
>>> from src.losses.contrastive_trust_loss import (
...     ContrastiveTrustLoss, ContrastiveTrustLossConfig,
... )
>>> loss_fn = ContrastiveTrustLoss(ContrastiveTrustLossConfig())
>>> B = 8
>>> v1 = F.normalize(torch.randn(B, 128), p=2, dim=1)
>>> v2 = F.normalize(torch.randn(B, 128), p=2, dim=1)
>>> enc = F.normalize(torch.randn(B, 256), p=2, dim=1)
>>> phy = F.normalize(torch.randn(B, 256), p=2, dim=1)
>>> out = loss_fn(v1, v2, enc, phy)
>>> out["loss"].shape
torch.Size([])
>>> sorted(out["weights"])
['contrastive', 'physics']

References
----------
.. [1] Chen, T., et al. (2020). A Simple Framework for Contrastive
       Learning of Visual Representations. ICML.
.. [2] Kendall, A., Gal, Y., & Cipolla, R. (2018). Multi-Task Learning
       Using Uncertainty to Weigh Losses. CVPR.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, Mapping, Optional, Tuple

import torch
import torch.nn as nn

from src.losses.nt_xent import NTXentConfig, NTXentLoss
from src.losses.physics_consistency import (
    PhysicsConsistencyConfig,
    PhysicsConsistencyLoss,
)

logger = logging.getLogger(__name__)


# ======================================================================
# Configuration
# ======================================================================


@dataclass(frozen=True)
class ContrastiveTrustLossConfig:
    """Hyper-parameters for :class:`ContrastiveTrustLoss`.

    Parameters
    ----------
    contrastive_weight : float
        Initial weight ``w_c`` applied to the NT-Xent contrastive loss.
        Must be finite and non-negative.  Default ``1.0``.
    physics_weight : float
        Initial weight ``w_p`` applied to the physics-consistency loss.
        Must be finite and non-negative.  Default ``1.0``.
    normalize_weights : bool
        When ``True``, the effective weights are normalised to sum to one
        before aggregation (``w_i / Σ_j w_j``).  This decouples the
        relative objective balance from the overall loss scale / learning
        rate.  Default ``False``.
    learnable_weights : bool
        When ``True``, the objective weights are trainable
        :class:`torch.nn.Parameter` values (``requires_grad=True``) so the
        optimiser can adapt the multi-objective balance during training.
        When ``False`` they are still stored as parameters but with
        ``requires_grad=False`` (constant).  Default ``False``.
    log_individual_losses : bool
        When ``True``, each :meth:`ContrastiveTrustLoss.forward` call emits
        an ``INFO`` log line reporting the individual and weighted losses.
        Default ``True``.
    eps : float
        Small constant added to the weight sum during normalisation for
        numerical stability.  Must be strictly positive.  Default ``1e-8``.

    Raises
    ------
    ValueError
        If any hyper-parameter fails validation.
    """

    contrastive_weight: float = 1.0
    physics_weight: float = 1.0
    normalize_weights: bool = False
    learnable_weights: bool = False
    log_individual_losses: bool = True
    eps: float = 1e-8

    def __post_init__(self) -> None:
        """Validate all configuration fields."""
        for name, value in (
            ("contrastive_weight", self.contrastive_weight),
            ("physics_weight", self.physics_weight),
        ):
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                raise ValueError(
                    f"{name} must be a real number, got "
                    f"{type(value).__name__}"
                )
            if not _is_finite_number(float(value)):
                raise ValueError(
                    f"{name} must be finite, got {value}"
                )
            if float(value) < 0.0:
                raise ValueError(
                    f"{name} must be non-negative, got {value}"
                )
        if self.eps <= 0.0:
            raise ValueError(
                f"eps must be strictly positive, got {self.eps}"
            )


def _is_finite_number(value: float) -> bool:
    """Return ``True`` if *value* is neither NaN nor +/-Inf."""
    return value == value and value not in (float("inf"), float("-inf"))


# ======================================================================
# ContrastiveTrust Loss
# ======================================================================


class ContrastiveTrustLoss(nn.Module):
    """Unified multi-objective loss for ContrastiveTrust pre-training.

    Combines the contrastive (NT-Xent) and physics-consistency objectives
    into a single differentiable scalar via a weighted sum.  The component
    losses are *injected* collaborators and are never reimplemented here;
    this class is purely an orchestrator responsible for weighting,
    optional normalisation, optional learnable weights, aggregation,
    validation, and structured logging.

    Parameters
    ----------
    config : ContrastiveTrustLossConfig
        Loss configuration.
    contrastive_loss : torch.nn.Module, optional
        Contrastive objective.  Must be callable as
        ``module(projection_view_1, projection_view_2)`` and return a
        dict containing a scalar ``"loss"`` entry.  Defaults to a
        :class:`~src.losses.nt_xent.NTXentLoss` with default hyper-params.
    physics_loss : torch.nn.Module, optional
        Physics-consistency objective.  Must be callable as
        ``module(encoder_embedding, physics_embedding)`` and return a dict
        containing a scalar ``"loss"`` entry.  Defaults to a
        :class:`~src.losses.physics_consistency.PhysicsConsistencyLoss`
        with default hyper-params.

    Raises
    ------
    TypeError
        If *config* is not a :class:`ContrastiveTrustLossConfig`, or if a
        provided sub-loss is not a :class:`torch.nn.Module`.

    Notes
    -----
    The objective weights are held in an :class:`torch.nn.ParameterDict`
    keyed by objective name (``"contrastive"``, ``"physics"``).  This makes
    the aggregation logic objective-agnostic: a future objective is added
    by extending :attr:`objective_names`, registering its weight, and
    computing its loss inside :meth:`forward`.
    """

    def __init__(
        self,
        config: ContrastiveTrustLossConfig,
        contrastive_loss: Optional[nn.Module] = None,
        physics_loss: Optional[nn.Module] = None,
    ) -> None:
        if not isinstance(config, ContrastiveTrustLossConfig):
            raise TypeError(
                f"config must be a ContrastiveTrustLossConfig, "
                f"got {type(config).__name__}"
            )
        super().__init__()

        # Keep the config out of the TorchScript-visible attribute scope.
        self.__dict__["_config"] = config

        # ---- Sub-losses (dependency injection) ------------------------
        if contrastive_loss is None:
            contrastive_loss = NTXentLoss(NTXentConfig())
        if physics_loss is None:
            physics_loss = PhysicsConsistencyLoss(PhysicsConsistencyConfig())

        if not isinstance(contrastive_loss, nn.Module):
            raise TypeError(
                f"contrastive_loss must be a torch.nn.Module, "
                f"got {type(contrastive_loss).__name__}"
            )
        if not isinstance(physics_loss, nn.Module):
            raise TypeError(
                f"physics_loss must be a torch.nn.Module, "
                f"got {type(physics_loss).__name__}"
            )

        # Registered as submodules so .to(device) / state_dict propagate.
        self._contrastive_loss = contrastive_loss
        self._physics_loss = physics_loss

        # ---- Ordered objective registry -------------------------------
        # Extend this tuple (and forward()) to add further objectives.
        self._objective_names: Tuple[str, ...] = ("contrastive", "physics")

        # ---- Weights (always parameters; grad toggled by config) ------
        self._normalize_weights: bool = config.normalize_weights
        self._log_individual_losses: bool = config.log_individual_losses
        self._eps: float = config.eps

        self.weights = nn.ParameterDict(
            {
                "contrastive": nn.Parameter(
                    torch.tensor(
                        float(config.contrastive_weight),
                        dtype=torch.float32,
                    ),
                    requires_grad=config.learnable_weights,
                ),
                "physics": nn.Parameter(
                    torch.tensor(
                        float(config.physics_weight),
                        dtype=torch.float32,
                    ),
                    requires_grad=config.learnable_weights,
                ),
            }
        )

        logger.info(
            "ContrastiveTrustLoss initialised | "
            "contrastive_weight=%.4f | physics_weight=%.4f | "
            "normalize_weights=%s | learnable_weights=%s | "
            "objectives=%s",
            config.contrastive_weight,
            config.physics_weight,
            config.normalize_weights,
            config.learnable_weights,
            self._objective_names,
        )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    @torch.jit.ignore
    def config(self) -> ContrastiveTrustLossConfig:
        """Return the loss configuration."""
        return self.__dict__["_config"]

    @property
    @torch.jit.ignore
    def objective_names(self) -> Tuple[str, ...]:
        """Return the ordered tuple of objective names."""
        return self._objective_names

    @property
    @torch.jit.ignore
    def contrastive_loss(self) -> nn.Module:
        """Return the injected contrastive sub-loss module."""
        return self._contrastive_loss

    @property
    @torch.jit.ignore
    def physics_loss(self) -> nn.Module:
        """Return the injected physics-consistency sub-loss module."""
        return self._physics_loss

    # ------------------------------------------------------------------
    # Validation helpers
    # ------------------------------------------------------------------

    @torch.jit.ignore
    def _validate_pair(
        self,
        name_a: str,
        a: torch.Tensor,
        name_b: str,
        b: torch.Tensor,
    ) -> None:
        """Validate a pair of ``(B, D)`` embedding tensors.

        Parameters
        ----------
        name_a, name_b : str
            Human-readable argument names used in error messages.
        a, b : torch.Tensor
            Tensors that must both be 2-D, floating-point, finite, and
            share the same batch size and embedding dimension.

        Raises
        ------
        TypeError
            If either argument is not a :class:`torch.Tensor`.
        ValueError
            If any dtype, shape, or value constraint is violated.
        """
        for name, t in ((name_a, a), (name_b, b)):
            if not isinstance(t, torch.Tensor):
                raise TypeError(
                    f"{name} must be a torch.Tensor, "
                    f"got {type(t).__name__}"
                )
            if not t.is_floating_point():
                raise ValueError(
                    f"{name} must have a floating-point dtype, "
                    f"got {t.dtype}"
                )
            if t.ndim != 2:
                raise ValueError(
                    f"{name} must have exactly 2 dimensions (B, D), "
                    f"got {t.ndim} dimensions with shape {tuple(t.shape)}"
                )
            if torch.isnan(t).any():
                raise ValueError(f"{name} contains NaN values")
            if torch.isinf(t).any():
                raise ValueError(f"{name} contains Inf values")

        if a.shape[0] != b.shape[0]:
            raise ValueError(
                f"{name_a} and {name_b} must have the same batch size, "
                f"got {name_a}.shape={tuple(a.shape)}, "
                f"{name_b}.shape={tuple(b.shape)}"
            )
        if a.shape[1] != b.shape[1]:
            raise ValueError(
                f"{name_a} and {name_b} must have the same embedding "
                f"dimension, got {name_a}.shape={tuple(a.shape)}, "
                f"{name_b}.shape={tuple(b.shape)}"
            )

    @torch.jit.ignore
    def _validate_inputs(
        self,
        projection_view_1: torch.Tensor,
        projection_view_2: torch.Tensor,
        encoder_embedding: torch.Tensor,
        physics_embedding: torch.Tensor,
    ) -> None:
        """Validate all four public forward inputs.

        The two projection views must be mutually consistent (as required
        by NT-Xent) and the encoder / physics embeddings must be mutually
        consistent (as required by the physics loss).  Additionally, all
        four tensors must share the same batch size ``B`` since they
        originate from a single training batch.

        Raises
        ------
        TypeError
            If any argument is not a :class:`torch.Tensor`.
        ValueError
            If any dtype, shape, value, or cross-tensor constraint fails.
        """
        self._validate_pair(
            "projection_view_1", projection_view_1,
            "projection_view_2", projection_view_2,
        )
        self._validate_pair(
            "encoder_embedding", encoder_embedding,
            "physics_embedding", physics_embedding,
        )

        batch = projection_view_1.shape[0]
        if encoder_embedding.shape[0] != batch:
            raise ValueError(
                f"All inputs must share the same batch size; "
                f"projection views have batch {batch} but "
                f"encoder_embedding has batch {encoder_embedding.shape[0]}"
            )

        logger.debug(
            "ContrastiveTrustLoss | validated inputs | batch=%d | "
            "projection_dim=%d | embedding_dim=%d | dtype=%s",
            batch,
            projection_view_1.shape[1],
            encoder_embedding.shape[1],
            projection_view_1.dtype,
        )

    # ------------------------------------------------------------------
    # Weight management
    # ------------------------------------------------------------------

    @torch.jit.ignore
    def _resolve_effective_weights(self) -> Dict[str, torch.Tensor]:
        """Return the effective (possibly normalised) objective weights.

        Returns
        -------
        dict[str, torch.Tensor]
            Mapping from objective name to a scalar weight tensor.  When
            ``config.normalize_weights`` is ``True`` the weights are scaled
            to sum to one (``w_i / (Σ_j w_j + eps)``); otherwise the raw
            parameter values are returned.  The returned tensors remain
            attached to the autograd graph when the weights are learnable.
        """
        raw = {name: self.weights[name] for name in self._objective_names}
        if not self._normalize_weights:
            return raw

        stacked = torch.stack([raw[name] for name in self._objective_names])
        total = stacked.sum() + self._eps
        return {name: raw[name] / total for name in self._objective_names}

    @torch.jit.ignore
    def get_weights(self) -> Dict[str, float]:
        """Return the raw (pre-normalisation) objective weights.

        Returns
        -------
        dict[str, float]
            Mapping from objective name to its current raw weight value.

        Examples
        --------
        >>> fn = ContrastiveTrustLoss(ContrastiveTrustLossConfig())
        >>> fn.get_weights()["contrastive"]
        1.0
        """
        return {
            name: float(self.weights[name].detach().cpu())
            for name in self._objective_names
        }

    @torch.jit.ignore
    def get_effective_weights(self) -> Dict[str, float]:
        """Return the effective weights actually applied during aggregation.

        Returns
        -------
        dict[str, float]
            Mapping from objective name to its effective weight.  Identical
            to :meth:`get_weights` unless ``normalize_weights`` is enabled,
            in which case the values sum to one.
        """
        effective = self._resolve_effective_weights()
        return {
            name: float(effective[name].detach().cpu())
            for name in self._objective_names
        }

    @torch.jit.ignore
    def set_weights(self, weights: Mapping[str, float]) -> None:
        """Update one or more raw objective weights in place.

        Parameters
        ----------
        weights : Mapping[str, float]
            Mapping from objective name to a new non-negative, finite
            weight value.  Only the provided keys are updated; omitted
            objectives keep their current weight.

        Raises
        ------
        TypeError
            If *weights* is not a mapping.
        KeyError
            If a key is not a known objective name.
        ValueError
            If any weight is negative or non-finite.

        Examples
        --------
        >>> fn = ContrastiveTrustLoss(ContrastiveTrustLossConfig())
        >>> fn.set_weights({"contrastive": 2.0, "physics": 0.5})
        >>> fn.get_weights()["physics"]
        0.5
        """
        if not isinstance(weights, Mapping):
            raise TypeError(
                f"weights must be a mapping of objective->value, "
                f"got {type(weights).__name__}"
            )

        for name, value in weights.items():
            if name not in self._objective_names:
                raise KeyError(
                    f"Unknown objective '{name}'. "
                    f"Known objectives: {list(self._objective_names)}"
                )
            fvalue = float(value)
            if not _is_finite_number(fvalue):
                raise ValueError(
                    f"weight for '{name}' must be finite, got {value}"
                )
            if fvalue < 0.0:
                raise ValueError(
                    f"weight for '{name}' must be non-negative, got {value}"
                )

        with torch.no_grad():
            for name, value in weights.items():
                self.weights[name].fill_(float(value))

        logger.info(
            "ContrastiveTrustLoss | weights updated to %s",
            self.get_weights(),
        )

    @torch.jit.ignore
    def freeze_weights(self) -> None:
        """Disable gradients on the objective weights (make them constant).

        After this call the weights are excluded from optimisation but
        still participate in the forward computation and serialisation.
        """
        for name in self._objective_names:
            self.weights[name].requires_grad_(False)
        logger.info("ContrastiveTrustLoss | objective weights frozen")

    @torch.jit.ignore
    def unfreeze_weights(self) -> None:
        """Enable gradients on the objective weights (make them learnable)."""
        for name in self._objective_names:
            self.weights[name].requires_grad_(True)
        logger.info("ContrastiveTrustLoss | objective weights unfrozen")

    # ------------------------------------------------------------------
    # Aggregation
    # ------------------------------------------------------------------

    @torch.jit.ignore
    def compute_total_loss(
        self,
        individual_losses: Mapping[str, torch.Tensor],
    ) -> Dict[str, object]:
        """Aggregate individual objective losses into the weighted total.

        Parameters
        ----------
        individual_losses : Mapping[str, torch.Tensor]
            Mapping from objective name to its scalar loss tensor.  Must
            contain an entry for every name in :attr:`objective_names`.

        Returns
        -------
        dict
            A dictionary with keys:

            ``"loss"`` : torch.Tensor
                The weighted-sum scalar total (gradient-enabled).
            ``"weighted"`` : dict[str, torch.Tensor]
                Per-objective weighted loss contributions.
            ``"weights"`` : dict[str, float]
                Effective weights applied to each objective.

        Raises
        ------
        KeyError
            If any objective's loss is missing from *individual_losses*.

        Notes
        -----
        Each effective weight is cast to the dtype and device of its
        corresponding loss before multiplication, keeping the operation
        mixed-precision and multi-device safe.
        """
        missing = [
            name
            for name in self._objective_names
            if name not in individual_losses
        ]
        if missing:
            raise KeyError(
                f"individual_losses is missing objectives {missing}; "
                f"expected keys {list(self._objective_names)}"
            )

        effective = self._resolve_effective_weights()

        weighted: Dict[str, torch.Tensor] = {}
        total: Optional[torch.Tensor] = None
        for name in self._objective_names:
            loss = individual_losses[name]
            weight = effective[name].to(device=loss.device, dtype=loss.dtype)
            contribution = weight * loss
            weighted[name] = contribution
            total = contribution if total is None else total + contribution

        assert total is not None  # objective_names is never empty

        weights_out = {
            name: float(effective[name].detach().cpu())
            for name in self._objective_names
        }

        return {
            "loss": total,
            "weighted": weighted,
            "weights": weights_out,
        }

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        projection_view_1: torch.Tensor,
        projection_view_2: torch.Tensor,
        encoder_embedding: torch.Tensor,
        physics_embedding: torch.Tensor,
    ) -> Dict[str, object]:
        """Compute the unified multi-objective ContrastiveTrust loss.

        Parameters
        ----------
        projection_view_1 : torch.Tensor
            L2-normalised projected embeddings of the first augmented
            view, shape ``(B, P)``.
        projection_view_2 : torch.Tensor
            L2-normalised projected embeddings of the second augmented
            view, shape ``(B, P)``.
        encoder_embedding : torch.Tensor
            Fused encoder representation, shape ``(B, D)``.
        physics_embedding : torch.Tensor
            Physics-stream representation, shape ``(B, D)``.

        Returns
        -------
        dict
            A dictionary with the following keys:

            ``"loss"`` : torch.Tensor
                Scalar weighted total loss (gradient-enabled).
            ``"contrastive_loss"`` : torch.Tensor
                Raw (un-weighted) NT-Xent contrastive loss.
            ``"physics_loss"`` : torch.Tensor
                Raw (un-weighted) physics-consistency loss.
            ``"weighted_contrastive"`` : torch.Tensor
                Effective-weight-scaled contrastive contribution.
            ``"weighted_physics"`` : torch.Tensor
                Effective-weight-scaled physics contribution.
            ``"weights"`` : dict[str, float]
                Effective weights applied to each objective.

        Raises
        ------
        TypeError
            If any input is not a :class:`torch.Tensor`.
        ValueError
            If input validation fails (dtype, shape, NaN, Inf, batch
            mismatch, ...).
        """
        self._validate_inputs(
            projection_view_1,
            projection_view_2,
            encoder_embedding,
            physics_embedding,
        )

        # ---- 1. Delegate to the injected sub-losses -------------------
        contrastive_out = self._contrastive_loss(
            projection_view_1, projection_view_2
        )
        physics_out = self._physics_loss(
            encoder_embedding, physics_embedding
        )
        contrastive_loss = contrastive_out["loss"]
        physics_loss = physics_out["loss"]

        # ---- 2. Weighted aggregation ----------------------------------
        aggregated = self.compute_total_loss(
            {
                "contrastive": contrastive_loss,
                "physics": physics_loss,
            }
        )
        weighted = aggregated["weighted"]

        result: Dict[str, object] = {
            "loss": aggregated["loss"],
            "contrastive_loss": contrastive_loss,
            "physics_loss": physics_loss,
            "weighted_contrastive": weighted["contrastive"],
            "weighted_physics": weighted["physics"],
            "weights": aggregated["weights"],
        }

        if self._log_individual_losses:
            logger.info(
                "ContrastiveTrustLoss forward | total=%.6f | "
                "contrastive=%.6f (w=%.4f) | physics=%.6f (w=%.4f)",
                float(aggregated["loss"].detach()),
                float(contrastive_loss.detach()),
                aggregated["weights"]["contrastive"],
                float(physics_loss.detach()),
                aggregated["weights"]["physics"],
            )

        return result

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    @torch.jit.ignore
    def parameter_summary(self) -> Dict[str, object]:
        """Return a human-readable summary of the loss configuration.

        Returns
        -------
        dict
            Keys: ``"objectives"``, ``"weights"``, ``"effective_weights"``,
            ``"normalize_weights"``, ``"learnable_weights"``,
            ``"log_individual_losses"``, and ``"num_parameters"``.
        """
        config: ContrastiveTrustLossConfig = self.__dict__["_config"]
        return {
            "objectives": list(self._objective_names),
            "weights": self.get_weights(),
            "effective_weights": self.get_effective_weights(),
            "normalize_weights": config.normalize_weights,
            "learnable_weights": config.learnable_weights,
            "log_individual_losses": config.log_individual_losses,
            "num_parameters": sum(p.numel() for p in self.parameters()),
        }
