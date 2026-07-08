"""NT-Xent loss for self-supervised contrastive learning.

Implements the **Normalized Temperature-scaled Cross-Entropy** (NT-Xent)
objective introduced by Chen et al. (2020) in SimCLR.  The formulation
is adapted for multivariate industrial time-series produced by the
ContrastiveTrust dual-stream encoder.

Given two augmented views ``z_i`` and ``z_j`` of shape ``(B, D)``,
both **already L2-normalised** by the :class:`~src.models.ProjectionHead`,
the loss maximises agreement between positive pairs while pushing all
other ``2(B-1)`` in-batch samples apart.

Mathematical formulation
------------------------
For a positive pair ``(z_i[k], z_j[k])`` (indices ``k`` and ``k+B`` in
the concatenated representation)::

    ℓ(k) = -log  exp(sim(z_i[k], z_j[k]) / τ)
                 ──────────────────────────────────────────────
                  Σ_{m ≠ k} exp(sim(z_i[k], z_m) / τ)

The total loss is the mean over both views::

    L = (1 / 2B) Σ_k [ ℓ(k, i→j) + ℓ(k, j→i) ]

where ``τ`` is the temperature, and self-similarities on the diagonal
are masked out.

References
----------
.. [1] Chen, T., Kornblith, S., Norouzi, M., & Hinton, G. (2020).
       A Simple Framework for Contrastive Learning of Visual
       Representations. ICML. https://arxiv.org/abs/2002.05709

Example
-------
>>> import torch
>>> from src.losses.nt_xent import NTXentLoss, NTXentConfig
>>> cfg = NTXentConfig(temperature=0.07)
>>> loss_fn = NTXentLoss(cfg)
>>> B, D = 8, 128
>>> z_i = torch.nn.functional.normalize(torch.randn(B, D), p=2, dim=1)
>>> z_j = torch.nn.functional.normalize(torch.randn(B, D), p=2, dim=1)
>>> out = loss_fn(z_i, z_j)
>>> out["loss"].shape
torch.Size([])
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


# ======================================================================
# Configuration
# ======================================================================


@dataclass(frozen=True)
class NTXentConfig:
    """Hyper-parameters for :class:`NTXentLoss`.

    Parameters
    ----------
    temperature : float
        Softmax temperature ``τ``.  Smaller values sharpen the
        distribution; must be strictly positive.  Typical range:
        ``[0.05, 0.5]``.  Default ``0.07`` follows the original SimCLR
        paper for fine-grained representations.
    reduction : str
        How to reduce the per-sample losses.  One of ``"mean"`` or
        ``"sum"``.  Default ``"mean"``.
    eps : float
        Small constant for numerical stability in cosine-similarity
        computations.  Added to the denominator before normalisation.
        Default ``1e-8``.

    Raises
    ------
    ValueError
        If any hyper-parameter fails validation.
    """

    temperature: float = 0.07
    reduction: Literal["mean", "sum"] = "mean"
    eps: float = 1e-8

    def __post_init__(self) -> None:
        """Validate all configuration fields."""
        if self.temperature <= 0.0:
            raise ValueError(
                f"temperature must be strictly positive, "
                f"got {self.temperature}"
            )
        if self.reduction not in {"mean", "sum"}:
            raise ValueError(
                f"reduction must be 'mean' or 'sum', "
                f"got '{self.reduction}'"
            )
        if self.eps <= 0.0:
            raise ValueError(
                f"eps must be strictly positive, got {self.eps}"
            )


# ======================================================================
# NT-Xent Loss
# ======================================================================


class NTXentLoss(nn.Module):
    """NT-Xent (Normalized Temperature-scaled Cross-Entropy) loss.

    Computes the SimCLR contrastive objective over two augmented views
    of a batch.  Both views are expected to be **L2-normalised** tensors
    of shape ``(B, D)``; the :class:`~src.models.ProjectionHead` already
    applies this normalisation.

    The implementation is fully vectorised: no Python loops over samples.
    It runs on CPU and CUDA and supports mixed-precision training.

    Parameters
    ----------
    config : NTXentConfig
        Loss hyper-parameters.

    Raises
    ------
    TypeError
        If *config* is not a :class:`NTXentConfig`.

    Notes
    -----
    The concatenated representation ``z = cat([z_i, z_j])`` of shape
    ``(2B, D)`` is used to build a ``(2B, 2B)`` cosine-similarity
    matrix.  Self-similarities (diagonal) are masked to ``-inf`` before
    the softmax so they never contribute to the denominator.

    Positive-pair indices
        For row ``k`` in ``[0, B)``, the positive column is ``k + B``.
        For row ``k`` in ``[B, 2B)``, the positive column is ``k - B``.
    """

    def __init__(self, config: NTXentConfig) -> None:
        if not isinstance(config, NTXentConfig):
            raise TypeError(
                f"config must be an NTXentConfig, "
                f"got {type(config).__name__}"
            )
        super().__init__()

        # Store config in __dict__ to keep TorchScript from compiling it.
        self.__dict__["_config"] = config

        # Expose primitives as typed attributes for TorchScript.
        self._temperature: float = config.temperature
        self._reduction: str = config.reduction
        self._eps: float = config.eps

        logger.info(
            "NTXentLoss initialised | temperature=%.4f | "
            "reduction=%s | eps=%.2e",
            config.temperature,
            config.reduction,
            config.eps,
        )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    @torch.jit.ignore
    def config(self) -> NTXentConfig:
        """Return the loss configuration."""
        return self.__dict__["_config"]

    @property
    @torch.jit.ignore
    def temperature(self) -> float:
        """Return the softmax temperature."""
        return self._temperature

    # ------------------------------------------------------------------
    # Validation helpers
    # ------------------------------------------------------------------

    @torch.jit.ignore
    def _validate_inputs(
        self,
        z_i: torch.Tensor,
        z_j: torch.Tensor,
    ) -> None:
        """Validate projected embedding tensors.

        Parameters
        ----------
        z_i : torch.Tensor
            First-view embeddings of shape ``(B, D)``.
        z_j : torch.Tensor
            Second-view embeddings of shape ``(B, D)``.

        Raises
        ------
        TypeError
            If either input is not a :class:`torch.Tensor`.
        ValueError
            If any shape, dtype, or value constraint is violated.
        """
        for name, z in (("z_i", z_i), ("z_j", z_j)):
            if not isinstance(z, torch.Tensor):
                raise TypeError(
                    f"{name} must be a torch.Tensor, "
                    f"got {type(z).__name__}"
                )
            if not z.is_floating_point():
                raise ValueError(
                    f"{name} must have a floating-point dtype, "
                    f"got {z.dtype}"
                )
            if z.ndim != 2:
                raise ValueError(
                    f"{name} must have exactly 2 dimensions (B, D), "
                    f"got {z.ndim} dimensions with shape "
                    f"{tuple(z.shape)}"
                )
            if torch.isnan(z).any():
                raise ValueError(f"{name} contains NaN values")
            if torch.isinf(z).any():
                raise ValueError(f"{name} contains Inf values")

        if z_i.shape[0] != z_j.shape[0]:
            raise ValueError(
                f"z_i and z_j must have the same batch size, "
                f"got z_i.shape={tuple(z_i.shape)}, "
                f"z_j.shape={tuple(z_j.shape)}"
            )
        if z_i.shape[1] != z_j.shape[1]:
            raise ValueError(
                f"z_i and z_j must have the same embedding dimension, "
                f"got z_i.shape={tuple(z_i.shape)}, "
                f"z_j.shape={tuple(z_j.shape)}"
            )

    # ------------------------------------------------------------------
    # Public methods
    # ------------------------------------------------------------------

    def compute_similarity_matrix(
        self,
        z: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the pairwise cosine-similarity matrix.

        Parameters
        ----------
        z : torch.Tensor
            Concatenated embeddings of shape ``(2B, D)``, already
            L2-normalised.  If not normalised, the method applies
            F.normalize for numerical safety.

        Returns
        -------
        torch.Tensor
            Cosine-similarity matrix of shape ``(2B, 2B)``, with values
            in ``[-1, 1]``.

        Notes
        -----
        Because the projection head already L2-normalises embeddings,
        ``cosine_sim(u, v) = u · v`` and the matrix is simply
        ``z @ z.T``.  An explicit normalisation guard with ``eps``
        is applied to handle mixed-precision edge cases.
        """
        # Re-normalise defensively to handle fp16/bf16 drift.
        z_norm = F.normalize(z, p=2, dim=1, eps=self._eps)
        sim: torch.Tensor = torch.matmul(z_norm, z_norm.T)

        logger.debug(
            "NTXentLoss | similarity matrix shape=%s",
            tuple(sim.shape),
        )
        return sim

    def mask_self_similarity(
        self,
        sim: torch.Tensor,
        n: int,
    ) -> torch.Tensor:
        """Replace diagonal (self-similarity) entries with ``-inf``.

        Parameters
        ----------
        sim : torch.Tensor
            Similarity matrix of shape ``(N, N)``.
        n : int
            Side length of the square matrix.  Must equal
            ``sim.shape[0]`` and ``sim.shape[1]``.

        Returns
        -------
        torch.Tensor
            The input matrix with diagonal filled with ``-inf``, so
            self-similarities are excluded from the softmax denominator.
        """
        mask_val = torch.finfo(sim.dtype).min
        eye = torch.eye(n, dtype=torch.bool, device=sim.device)
        return sim.masked_fill(eye, mask_val)

    def create_labels(
        self,
        batch_size: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Create positive-pair ground-truth labels.

        The concatenated embedding ``z = cat([z_i, z_j])`` has shape
        ``(2B, D)``.  For the cross-entropy loss:

        * Rows ``0 … B-1``   (from ``z_i``) have their positive at
          columns ``B … 2B-1`` (the matching sample in ``z_j``).
        * Rows ``B … 2B-1``  (from ``z_j``) have their positive at
          columns ``0 … B-1`` (the matching sample in ``z_i``).

        This yields labels ``[B, B+1, …, 2B-1, 0, 1, …, B-1]``.

        Parameters
        ----------
        batch_size : int
            Number of samples per view, ``B``.
        device : torch.device
            Target device for the label tensor.

        Returns
        -------
        torch.Tensor
            Long tensor of shape ``(2B,)`` containing column indices of
            the positive pair for each row.
        """
        labels_i = torch.arange(
            batch_size, 2 * batch_size, dtype=torch.long, device=device
        )
        labels_j = torch.arange(
            0, batch_size, dtype=torch.long, device=device
        )
        return torch.cat([labels_i, labels_j], dim=0)

    def forward(
        self,
        z_i: torch.Tensor,
        z_j: torch.Tensor,
    ) -> Dict[str, object]:
        """Compute the NT-Xent contrastive loss.

        Parameters
        ----------
        z_i : torch.Tensor
            L2-normalised projected embeddings from the first augmented
            view.  Shape ``(B, D)``.
        z_j : torch.Tensor
            L2-normalised projected embeddings from the second augmented
            view.  Shape ``(B, D)``.

        Returns
        -------
        dict
            A dictionary with the following keys:

            ``"loss"`` : torch.Tensor
                Scalar NT-Xent loss (gradient-enabled).
            ``"logits"`` : torch.Tensor
                Scaled similarity logits of shape ``(2B, 2B)`` after
                masking self-similarities and dividing by temperature.
            ``"labels"`` : torch.Tensor
                Ground-truth positive-pair indices of shape ``(2B,)``.
            ``"temperature"`` : float
                The temperature used for this forward pass.

        Raises
        ------
        TypeError
            If either input is not a :class:`torch.Tensor`.
        ValueError
            If input validation fails (dtype, shape, NaN, Inf, etc.).

        Notes
        -----
        The loss is computed with
        :func:`torch.nn.functional.cross_entropy` over the masked and
        temperature-scaled similarity matrix, which is numerically stable
        via PyTorch's internal log-sum-exp trick.
        """
        self._validate_inputs(z_i, z_j)

        batch_size: int = z_i.shape[0]
        logger.info(
            "NTXentLoss forward | batch_size=%d | temperature=%.4f",
            batch_size,
            self._temperature,
        )

        # ---- 1. Concatenate views: (2B, D) ----------------------------
        z: torch.Tensor = torch.cat([z_i, z_j], dim=0)

        # ---- 2. Cosine-similarity matrix: (2B, 2B) --------------------
        sim = self.compute_similarity_matrix(z)

        # ---- 3. Mask self-similarities --------------------------------
        n: int = 2 * batch_size
        sim_masked = self.mask_self_similarity(sim, n)

        # ---- 4. Temperature scaling -----------------------------------
        logits: torch.Tensor = sim_masked / self._temperature

        # ---- 5. Positive-pair labels: (2B,) ---------------------------
        labels = self.create_labels(batch_size, device=z_i.device)

        # ---- 6. Cross-entropy loss (numerically stable via log-softmax)
        loss: torch.Tensor = F.cross_entropy(
            logits,
            labels,
            reduction=self._reduction,
        )

        logger.debug(
            "NTXentLoss forward | loss=%.6f",
            loss.item(),
        )

        return {
            "loss": loss,
            "logits": logits,
            "labels": labels,
            "temperature": self._temperature,
        }

    @torch.jit.ignore
    def parameter_summary(self) -> Dict[str, object]:
        """Return a human-readable summary of loss hyper-parameters.

        Returns
        -------
        dict
            Dictionary with keys ``"temperature"``, ``"reduction"``,
            ``"eps"``, and ``"num_parameters"`` (always 0 since
            :class:`NTXentLoss` has no learnable parameters).

        Examples
        --------
        >>> from src.losses.nt_xent import NTXentLoss, NTXentConfig
        >>> loss_fn = NTXentLoss(NTXentConfig())
        >>> summary = loss_fn.parameter_summary()
        >>> summary["temperature"]
        0.07
        """
        return {
            "temperature": self._temperature,
            "reduction": self._reduction,
            "eps": self._eps,
            "num_parameters": sum(p.numel() for p in self.parameters()),
        }
