"""Time-series augmentation primitives for self-supervised contrastive learning.

Each augmentation is implemented as a self-contained class with a dedicated
configuration dataclass.  All classes implement the same callable interface::

    augmented = augmentation(x)

where *x* is a ``torch.Tensor`` of shape ``(T, S)`` or ``(B, T, S)``.

Augmentations can be composed with :class:`Compose` (analogous to
``torchvision.transforms.Compose``) and each supports an application
probability *p* — when the random draw fails the original tensor is
returned unchanged.

Example
-------
>>> import torch
>>> from src.data.augmentations import Compose, GaussianNoise, RandomScaling
>>> aug = Compose([GaussianNoise(), RandomScaling()])
>>> x = torch.randn(60, 10)      # (T, S)
>>> x_aug = aug(x)
>>> x_aug.shape
torch.Size([60, 10])
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import torch

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_SUPPORTED_DTYPES: Tuple[torch.dtype, ...] = (
    torch.float16,
    torch.bfloat16,
    torch.float32,
    torch.float64,
)


def _validate_tensor(x: torch.Tensor, caller: str) -> None:
    """Validate that *x* is a well-formed time-series tensor.

    Parameters
    ----------
    x : torch.Tensor
        Tensor to validate.
    caller : str
        Name of the calling augmentation (used in error messages).

    Raises
    ------
    TypeError
        If *x* is not a :class:`torch.Tensor`.
    ValueError
        If *x* has wrong dtype, wrong number of dimensions, NaN, or Inf.
    """
    if not isinstance(x, torch.Tensor):
        raise TypeError(
            f"{caller}: expected torch.Tensor, got {type(x).__name__}"
        )
    if x.dtype not in _SUPPORTED_DTYPES:
        raise ValueError(
            f"{caller}: tensor must have a floating-point dtype "
            f"(float16, bfloat16, float32, float64), got {x.dtype}"
        )
    if x.ndim not in (2, 3):
        raise ValueError(
            f"{caller}: tensor must be 2-D (T, S) or 3-D (B, T, S), "
            f"got {x.ndim}-D tensor with shape {tuple(x.shape)}"
        )
    if torch.isnan(x).any():
        raise ValueError(f"{caller}: input tensor contains NaN values")
    if torch.isinf(x).any():
        raise ValueError(f"{caller}: input tensor contains Inf values")


def _make_generator(seed: Optional[int], device: torch.device) -> torch.Generator:
    """Create a :class:`torch.Generator` optionally seeded.

    Parameters
    ----------
    seed : int or None
        If *None*, the generator is not explicitly seeded (non-deterministic).
    device : torch.device
        Device on which the generator lives.

    Returns
    -------
    torch.Generator
    """
    gen = torch.Generator(device=device)
    if seed is not None:
        gen.manual_seed(seed)
    return gen


# ======================================================================
# Configuration dataclasses
# ======================================================================


@dataclass(frozen=True)
class GaussianNoiseConfig:
    """Hyper-parameters for :class:`GaussianNoise`.

    Parameters
    ----------
    std : float
        Standard deviation of the zero-mean Gaussian noise.
    p : float
        Probability of applying the augmentation (in [0, 1]).
    seed : int or None
        Fixed seed for reproducible noise; *None* for non-deterministic.
    """

    std: float = 0.05
    p: float = 1.0
    seed: Optional[int] = None

    def __post_init__(self) -> None:
        if self.std < 0.0:
            raise ValueError(f"std must be non-negative, got {self.std}")
        if not 0.0 <= self.p <= 1.0:
            raise ValueError(f"p must be in [0, 1], got {self.p}")


@dataclass(frozen=True)
class RandomScalingConfig:
    """Hyper-parameters for :class:`RandomScaling`.

    Parameters
    ----------
    scale_min : float
        Lower bound of the scaling factor range.
    scale_max : float
        Upper bound of the scaling factor range.
    p : float
        Probability of applying the augmentation.
    seed : int or None
        Fixed seed; *None* for non-deterministic.
    """

    scale_min: float = 0.8
    scale_max: float = 1.2
    p: float = 1.0
    seed: Optional[int] = None

    def __post_init__(self) -> None:
        if self.scale_min <= 0.0:
            raise ValueError(
                f"scale_min must be positive, got {self.scale_min}"
            )
        if self.scale_max < self.scale_min:
            raise ValueError(
                f"scale_max ({self.scale_max}) must be >= scale_min "
                f"({self.scale_min})"
            )
        if not 0.0 <= self.p <= 1.0:
            raise ValueError(f"p must be in [0, 1], got {self.p}")


@dataclass(frozen=True)
class RandomJitterConfig:
    """Hyper-parameters for :class:`RandomJitter`.

    Parameters
    ----------
    jitter_std : float
        Standard deviation of the per-step jitter perturbations.
    p : float
        Probability of applying the augmentation.
    seed : int or None
        Fixed seed; *None* for non-deterministic.
    """

    jitter_std: float = 0.01
    p: float = 1.0
    seed: Optional[int] = None

    def __post_init__(self) -> None:
        if self.jitter_std < 0.0:
            raise ValueError(
                f"jitter_std must be non-negative, got {self.jitter_std}"
            )
        if not 0.0 <= self.p <= 1.0:
            raise ValueError(f"p must be in [0, 1], got {self.p}")


@dataclass(frozen=True)
class RandomTimeMaskConfig:
    """Hyper-parameters for :class:`RandomTimeMask`.

    Parameters
    ----------
    mask_ratio : float
        Fraction of time steps to mask (in (0, 1]).
    fill_value : float
        Value used to fill masked positions.
    p : float
        Probability of applying the augmentation.
    seed : int or None
        Fixed seed; *None* for non-deterministic.
    """

    mask_ratio: float = 0.1
    fill_value: float = 0.0
    p: float = 1.0
    seed: Optional[int] = None

    def __post_init__(self) -> None:
        if not 0.0 < self.mask_ratio <= 1.0:
            raise ValueError(
                f"mask_ratio must be in (0, 1], got {self.mask_ratio}"
            )
        if not 0.0 <= self.p <= 1.0:
            raise ValueError(f"p must be in [0, 1], got {self.p}")


@dataclass(frozen=True)
class RandomChannelMaskConfig:
    """Hyper-parameters for :class:`RandomChannelMask`.

    Parameters
    ----------
    mask_ratio : float
        Fraction of sensor channels to zero out (in (0, 1]).
    fill_value : float
        Value used to fill masked channels.
    p : float
        Probability of applying the augmentation.
    seed : int or None
        Fixed seed; *None* for non-deterministic.
    """

    mask_ratio: float = 0.1
    fill_value: float = 0.0
    p: float = 1.0
    seed: Optional[int] = None

    def __post_init__(self) -> None:
        if not 0.0 < self.mask_ratio <= 1.0:
            raise ValueError(
                f"mask_ratio must be in (0, 1], got {self.mask_ratio}"
            )
        if not 0.0 <= self.p <= 1.0:
            raise ValueError(f"p must be in [0, 1], got {self.p}")


@dataclass(frozen=True)
class RandomTimeShiftConfig:
    """Hyper-parameters for :class:`RandomTimeShift`.

    Parameters
    ----------
    max_shift : int
        Maximum circular shift in time steps (absolute value).
    p : float
        Probability of applying the augmentation.
    seed : int or None
        Fixed seed; *None* for non-deterministic.
    """

    max_shift: int = 10
    p: float = 1.0
    seed: Optional[int] = None

    def __post_init__(self) -> None:
        if self.max_shift < 0:
            raise ValueError(
                f"max_shift must be non-negative, got {self.max_shift}"
            )
        if not 0.0 <= self.p <= 1.0:
            raise ValueError(f"p must be in [0, 1], got {self.p}")


@dataclass(frozen=True)
class IdentityConfig:
    """Hyper-parameters for :class:`Identity`.

    Parameters
    ----------
    p : float
        Probability of applying the (no-op) augmentation.  Typically 1.0.
    """

    p: float = 1.0

    def __post_init__(self) -> None:
        if not 0.0 <= self.p <= 1.0:
            raise ValueError(f"p must be in [0, 1], got {self.p}")


@dataclass
class ComposeConfig:
    """Hyper-parameters for :class:`Compose`.

    Parameters
    ----------
    p : float
        Probability of applying the entire composed pipeline.
    """

    p: float = 1.0

    def __post_init__(self) -> None:
        if not 0.0 <= self.p <= 1.0:
            raise ValueError(f"p must be in [0, 1], got {self.p}")


# ======================================================================
# Base class
# ======================================================================


class BaseAugmentation:
    """Abstract base for all time-series augmentations.

    Subclasses must override :meth:`_apply` and may override
    :meth:`_name` for logging purposes.

    Parameters
    ----------
    p : float
        Probability that the augmentation is applied.  When the random
        draw exceeds *p* the original tensor is returned unchanged.
    seed : int or None
        Optional seed for the internal :class:`torch.Generator`.
    """

    def __init__(self, p: float, seed: Optional[int]) -> None:
        self._p = p
        self._seed = seed
        logger.info(
            "%s initialised | p=%.2f | seed=%s",
            self._name,
            p,
            seed,
        )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def _name(self) -> str:
        """Return the class name for logging."""
        return type(self).__name__

    # ------------------------------------------------------------------
    # Internal interface
    # ------------------------------------------------------------------

    def _apply(self, x: torch.Tensor) -> torch.Tensor:  # pragma: no cover
        """Apply the augmentation unconditionally.

        Parameters
        ----------
        x : torch.Tensor
            Validated input tensor.

        Returns
        -------
        torch.Tensor
            Augmented tensor of the same shape and dtype as *x*.
        """
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Apply augmentation with probability *p*.

        Parameters
        ----------
        x : torch.Tensor
            Time-series window of shape ``(T, S)`` or ``(B, T, S)``
            with a floating-point dtype; must not contain NaN or Inf.

        Returns
        -------
        torch.Tensor
            Augmented (or unchanged) tensor of the same shape.

        Raises
        ------
        TypeError
            If *x* is not a :class:`torch.Tensor`.
        ValueError
            If *x* has wrong dtype, wrong dimensions, NaN, or Inf.
        """
        _validate_tensor(x, self._name)

        # Probability gate — use CPU generator to keep device-independence.
        if self._p < 1.0:
            gate_gen = _make_generator(self._seed, device=torch.device("cpu"))
            if torch.rand(1, generator=gate_gen).item() > self._p:
                logger.debug(
                    "%s skipped (probability gate) | shape=%s",
                    self._name,
                    tuple(x.shape),
                )
                return x

        logger.debug(
            "%s applying | shape=%s | dtype=%s",
            self._name,
            tuple(x.shape),
            x.dtype,
        )

        result = self._apply(x)

        logger.debug(
            "%s applied | output shape=%s",
            self._name,
            tuple(result.shape),
        )
        return result

    def __repr__(self) -> str:
        return f"{self._name}(p={self._p!r}, seed={self._seed!r})"


# ======================================================================
# Augmentation implementations
# ======================================================================


class GaussianNoise(BaseAugmentation):
    """Add zero-mean Gaussian noise to every element.

    Parameters
    ----------
    config : GaussianNoiseConfig
        Augmentation configuration.

    Notes
    -----
    The noise is drawn from ``N(0, std²)`` and has the same shape as *x*.

    Example
    -------
    >>> import torch
    >>> from src.data.augmentations import GaussianNoise, GaussianNoiseConfig
    >>> aug = GaussianNoise(GaussianNoiseConfig(std=0.1, seed=0))
    >>> x = torch.ones(5, 3)
    >>> y = aug(x)
    >>> y.shape
    torch.Size([5, 3])
    """

    def __init__(self, config: GaussianNoiseConfig = GaussianNoiseConfig()) -> None:
        if not isinstance(config, GaussianNoiseConfig):
            raise TypeError(
                f"config must be GaussianNoiseConfig, got {type(config).__name__}"
            )
        super().__init__(p=config.p, seed=config.seed)
        self._std = config.std
        self._config = config

    def _apply(self, x: torch.Tensor) -> torch.Tensor:
        gen = _make_generator(self._seed, device=x.device)
        noise = torch.zeros_like(x).normal_(mean=0.0, std=self._std, generator=gen)
        return x + noise

    def __repr__(self) -> str:
        return (
            f"GaussianNoise(std={self._std!r}, p={self._p!r}, "
            f"seed={self._seed!r})"
        )


class RandomScaling(BaseAugmentation):
    """Scale each sensor channel independently by a random factor.

    The scaling factor is drawn uniformly from
    ``[scale_min, scale_max]`` per channel.

    Parameters
    ----------
    config : RandomScalingConfig
        Augmentation configuration.

    Example
    -------
    >>> import torch
    >>> from src.data.augmentations import RandomScaling, RandomScalingConfig
    >>> aug = RandomScaling(RandomScalingConfig(scale_min=0.9, scale_max=1.1, seed=1))
    >>> x = torch.ones(10, 5)
    >>> y = aug(x)
    >>> y.shape
    torch.Size([10, 5])
    """

    def __init__(self, config: RandomScalingConfig = RandomScalingConfig()) -> None:
        if not isinstance(config, RandomScalingConfig):
            raise TypeError(
                f"config must be RandomScalingConfig, got {type(config).__name__}"
            )
        super().__init__(p=config.p, seed=config.seed)
        self._scale_min = config.scale_min
        self._scale_max = config.scale_max
        self._config = config

    def _apply(self, x: torch.Tensor) -> torch.Tensor:
        # x: (..., T, S) — last dim is S (channels)
        num_channels = x.shape[-1]
        gen = _make_generator(self._seed, device=x.device)
        scales = (
            torch.rand(num_channels, dtype=x.dtype, device=x.device, generator=gen)
            * (self._scale_max - self._scale_min)
            + self._scale_min
        )
        # Broadcast over batch and time dims: (..., 1, S) × (..., T, S)
        return x * scales

    def __repr__(self) -> str:
        return (
            f"RandomScaling(scale_min={self._scale_min!r}, "
            f"scale_max={self._scale_max!r}, p={self._p!r}, "
            f"seed={self._seed!r})"
        )


class RandomJitter(BaseAugmentation):
    """Apply small independent per-element random perturbations.

    Each element of the tensor is perturbed by a value drawn from
    ``N(0, jitter_std²)``.  Unlike :class:`GaussianNoise` (which is
    often used at a higher amplitude for data augmentation), this is
    intended as a fine-grained jitter that preserves local structure.

    Parameters
    ----------
    config : RandomJitterConfig
        Augmentation configuration.

    Example
    -------
    >>> import torch
    >>> from src.data.augmentations import RandomJitter, RandomJitterConfig
    >>> aug = RandomJitter(RandomJitterConfig(jitter_std=0.02, seed=2))
    >>> x = torch.zeros(8, 4)
    >>> y = aug(x)
    >>> y.shape
    torch.Size([8, 4])
    """

    def __init__(self, config: RandomJitterConfig = RandomJitterConfig()) -> None:
        if not isinstance(config, RandomJitterConfig):
            raise TypeError(
                f"config must be RandomJitterConfig, got {type(config).__name__}"
            )
        super().__init__(p=config.p, seed=config.seed)
        self._jitter_std = config.jitter_std
        self._config = config

    def _apply(self, x: torch.Tensor) -> torch.Tensor:
        gen = _make_generator(self._seed, device=x.device)
        jitter = torch.zeros_like(x).normal_(
            mean=0.0, std=self._jitter_std, generator=gen
        )
        return x + jitter

    def __repr__(self) -> str:
        return (
            f"RandomJitter(jitter_std={self._jitter_std!r}, "
            f"p={self._p!r}, seed={self._seed!r})"
        )


class RandomTimeMask(BaseAugmentation):
    """Mask a contiguous random temporal segment with a fixed fill value.

    A single contiguous block of ``ceil(mask_ratio * T)`` time steps is
    zeroed-out (or filled with *fill_value*).  The start position is
    chosen uniformly.

    Parameters
    ----------
    config : RandomTimeMaskConfig
        Augmentation configuration.

    Example
    -------
    >>> import torch
    >>> from src.data.augmentations import RandomTimeMask, RandomTimeMaskConfig
    >>> aug = RandomTimeMask(RandomTimeMaskConfig(mask_ratio=0.2, seed=3))
    >>> x = torch.ones(50, 6)
    >>> y = aug(x)
    >>> y.shape
    torch.Size([50, 6])
    """

    def __init__(self, config: RandomTimeMaskConfig = RandomTimeMaskConfig()) -> None:
        if not isinstance(config, RandomTimeMaskConfig):
            raise TypeError(
                f"config must be RandomTimeMaskConfig, got {type(config).__name__}"
            )
        super().__init__(p=config.p, seed=config.seed)
        self._mask_ratio = config.mask_ratio
        self._fill_value = config.fill_value
        self._config = config

    def _apply(self, x: torch.Tensor) -> torch.Tensor:
        # T is the second-to-last dimension for both (T, S) and (B, T, S)
        T = x.shape[-2]
        mask_len = max(1, math.ceil(self._mask_ratio * T))
        max_start = T - mask_len
        gen = _make_generator(self._seed, device=torch.device("cpu"))
        start = (
            torch.randint(0, max_start + 1, (1,), generator=gen).item()
            if max_start > 0
            else 0
        )
        end = int(start) + mask_len
        out = x.clone()
        # Apply mask along the time axis
        if x.ndim == 2:
            out[start:end, :] = self._fill_value
        else:  # (B, T, S)
            out[:, start:end, :] = self._fill_value
        return out

    def __repr__(self) -> str:
        return (
            f"RandomTimeMask(mask_ratio={self._mask_ratio!r}, "
            f"fill_value={self._fill_value!r}, p={self._p!r}, "
            f"seed={self._seed!r})"
        )


class RandomChannelMask(BaseAugmentation):
    """Randomly zero out complete sensor channels.

    ``ceil(mask_ratio * S)`` channels are selected uniformly at random
    and replaced with *fill_value* across all time steps.

    Parameters
    ----------
    config : RandomChannelMaskConfig
        Augmentation configuration.

    Example
    -------
    >>> import torch
    >>> from src.data.augmentations import RandomChannelMask, RandomChannelMaskConfig
    >>> aug = RandomChannelMask(RandomChannelMaskConfig(mask_ratio=0.3, seed=4))
    >>> x = torch.ones(30, 8)
    >>> y = aug(x)
    >>> y.shape
    torch.Size([30, 8])
    """

    def __init__(
        self, config: RandomChannelMaskConfig = RandomChannelMaskConfig()
    ) -> None:
        if not isinstance(config, RandomChannelMaskConfig):
            raise TypeError(
                f"config must be RandomChannelMaskConfig, "
                f"got {type(config).__name__}"
            )
        super().__init__(p=config.p, seed=config.seed)
        self._mask_ratio = config.mask_ratio
        self._fill_value = config.fill_value
        self._config = config

    def _apply(self, x: torch.Tensor) -> torch.Tensor:
        S = x.shape[-1]
        num_masked = max(1, math.ceil(self._mask_ratio * S))
        gen = _make_generator(self._seed, device=torch.device("cpu"))
        # Draw indices without replacement
        perm = torch.randperm(S, generator=gen)[:num_masked]
        out = x.clone()
        if x.ndim == 2:
            out[:, perm] = self._fill_value
        else:  # (B, T, S)
            out[:, :, perm] = self._fill_value
        return out

    def __repr__(self) -> str:
        return (
            f"RandomChannelMask(mask_ratio={self._mask_ratio!r}, "
            f"fill_value={self._fill_value!r}, p={self._p!r}, "
            f"seed={self._seed!r})"
        )


class RandomTimeShift(BaseAugmentation):
    """Apply a circular (roll) shift along the time axis.

    A shift amount is drawn uniformly from ``[-max_shift, max_shift]``
    and :func:`torch.roll` is used so that no information is lost.

    Parameters
    ----------
    config : RandomTimeShiftConfig
        Augmentation configuration.

    Example
    -------
    >>> import torch
    >>> from src.data.augmentations import RandomTimeShift, RandomTimeShiftConfig
    >>> aug = RandomTimeShift(RandomTimeShiftConfig(max_shift=5, seed=5))
    >>> x = torch.arange(20, dtype=torch.float32).reshape(10, 2)
    >>> y = aug(x)
    >>> y.shape
    torch.Size([10, 2])
    """

    def __init__(
        self, config: RandomTimeShiftConfig = RandomTimeShiftConfig()
    ) -> None:
        if not isinstance(config, RandomTimeShiftConfig):
            raise TypeError(
                f"config must be RandomTimeShiftConfig, "
                f"got {type(config).__name__}"
            )
        super().__init__(p=config.p, seed=config.seed)
        self._max_shift = config.max_shift
        self._config = config

    def _apply(self, x: torch.Tensor) -> torch.Tensor:
        if self._max_shift == 0:
            return x
        gen = _make_generator(self._seed, device=torch.device("cpu"))
        # Draw shift in [-max_shift, max_shift]
        shift = (
            torch.randint(0, 2 * self._max_shift + 1, (1,), generator=gen).item()
            - self._max_shift
        )
        # Time axis is dim -2 for both (T, S) and (B, T, S)
        return torch.roll(x, shifts=int(shift), dims=-2)

    def __repr__(self) -> str:
        return (
            f"RandomTimeShift(max_shift={self._max_shift!r}, "
            f"p={self._p!r}, seed={self._seed!r})"
        )


class Identity(BaseAugmentation):
    """Return the input unchanged.

    Useful as a baseline or for ablation studies where one augmentation
    branch is deactivated.

    Parameters
    ----------
    config : IdentityConfig
        Augmentation configuration.

    Example
    -------
    >>> import torch
    >>> from src.data.augmentations import Identity, IdentityConfig
    >>> aug = Identity(IdentityConfig())
    >>> x = torch.randn(10, 3)
    >>> assert aug(x) is x
    """

    def __init__(self, config: IdentityConfig = IdentityConfig()) -> None:
        if not isinstance(config, IdentityConfig):
            raise TypeError(
                f"config must be IdentityConfig, got {type(config).__name__}"
            )
        super().__init__(p=config.p, seed=None)
        self._config = config

    def _apply(self, x: torch.Tensor) -> torch.Tensor:
        return x

    def __repr__(self) -> str:
        return f"Identity(p={self._p!r})"


# ======================================================================
# Compose
# ======================================================================


class Compose:
    """Chain multiple augmentations sequentially.

    Analogous to ``torchvision.transforms.Compose``.  Each augmentation
    in the list is applied in order to the output of the previous one.

    Parameters
    ----------
    augmentations : list[BaseAugmentation]
        Ordered list of augmentations to apply.
    config : ComposeConfig
        Top-level configuration (outer probability gate).

    Example
    -------
    >>> import torch
    >>> from src.data.augmentations import (
    ...     Compose, GaussianNoise, RandomScaling, RandomTimeMask
    ... )
    >>> aug = Compose([GaussianNoise(), RandomScaling(), RandomTimeMask()])
    >>> x = torch.randn(60, 10)
    >>> y = aug(x)
    >>> y.shape
    torch.Size([60, 10])
    """

    def __init__(
        self,
        augmentations: List[BaseAugmentation],
        config: ComposeConfig = ComposeConfig(),
    ) -> None:
        if not isinstance(config, ComposeConfig):
            raise TypeError(
                f"config must be ComposeConfig, got {type(config).__name__}"
            )
        if not isinstance(augmentations, list):
            raise TypeError(
                f"augmentations must be a list, got {type(augmentations).__name__}"
            )
        for i, aug in enumerate(augmentations):
            if not isinstance(aug, BaseAugmentation):
                raise TypeError(
                    f"augmentations[{i}] must be a BaseAugmentation subclass, "
                    f"got {type(aug).__name__}"
                )
        self._augmentations = augmentations
        self._config = config
        self._p = config.p
        logger.info(
            "Compose initialised | steps=%d | p=%.2f | pipeline=%s",
            len(augmentations),
            config.p,
            [type(a).__name__ for a in augmentations],
        )

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    @property
    def augmentations(self) -> List[BaseAugmentation]:
        """Return the list of constituent augmentations."""
        return self._augmentations

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the augmentation pipeline.

        Parameters
        ----------
        x : torch.Tensor
            Time-series tensor of shape ``(T, S)`` or ``(B, T, S)``.

        Returns
        -------
        torch.Tensor
            Augmented tensor of the same shape.

        Raises
        ------
        TypeError
            If *x* is not a :class:`torch.Tensor`.
        ValueError
            If *x* fails validation.
        """
        _validate_tensor(x, "Compose")

        if self._p < 1.0:
            if torch.rand(1).item() > self._p:
                logger.debug(
                    "Compose skipped (probability gate) | shape=%s",
                    tuple(x.shape),
                )
                return x

        out = x
        for aug in self._augmentations:
            out = aug(out)
        return out

    def __len__(self) -> int:
        return len(self._augmentations)

    def __repr__(self) -> str:
        inner = ", ".join(repr(a) for a in self._augmentations)
        return f"Compose([{inner}], p={self._p!r})"
