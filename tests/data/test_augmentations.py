"""Tests for :mod:`src.data.augmentations`.

Comprehensive test suite covering:
- Every augmentation class (GaussianNoise, RandomScaling, RandomJitter,
  RandomTimeMask, RandomChannelMask, RandomTimeShift, Identity)
- Compose chaining
- Configuration validation
- Probability handling (p=0 never applies, p=1 always applies)
- Shape preservation for (T, S) and (B, T, S) inputs
- Float dtype support (float16, float32, float64)
- NaN / Inf rejection
- Non-tensor rejection
- Wrong dimension rejection
- Deterministic behaviour via seed
- CPU and CUDA (when available)
"""

from __future__ import annotations

import math
from dataclasses import FrozenInstanceError
from typing import Any, List, Tuple

import pytest
import torch

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
    _validate_tensor,
)


# ======================================================================
# Constants & helpers
# ======================================================================

_T = 60      # time steps
_S = 10      # sensor channels
_B = 4       # batch size

_DEVICES: List[torch.device] = [torch.device("cpu")]
if torch.cuda.is_available():
    _DEVICES.append(torch.device("cuda"))


def _rand(shape: Tuple[int, ...], device: torch.device = torch.device("cpu")) -> torch.Tensor:
    """Return a reproducible random float32 tensor."""
    torch.manual_seed(42)
    return torch.randn(*shape, device=device)


# ======================================================================
# Fixtures
# ======================================================================


@pytest.fixture()
def x_2d() -> torch.Tensor:
    """Single window (T, S)."""
    return _rand((_T, _S))


@pytest.fixture()
def x_3d() -> torch.Tensor:
    """Batched windows (B, T, S)."""
    return _rand((_B, _T, _S))


# ======================================================================
# _validate_tensor
# ======================================================================


class TestValidateTensor:
    """Unit tests for the internal validation helper."""

    def test_accepts_2d_float32(self, x_2d: torch.Tensor) -> None:
        _validate_tensor(x_2d, "test")  # must not raise

    def test_accepts_3d_float32(self, x_3d: torch.Tensor) -> None:
        _validate_tensor(x_3d, "test")

    def test_accepts_float16(self) -> None:
        _validate_tensor(torch.randn(5, 3).half(), "test")

    def test_accepts_float64(self) -> None:
        _validate_tensor(torch.randn(5, 3).double(), "test")

    def test_rejects_non_tensor(self) -> None:
        with pytest.raises(TypeError, match="torch.Tensor"):
            _validate_tensor([1.0, 2.0], "test")  # type: ignore[arg-type]

    def test_rejects_integer_dtype(self) -> None:
        with pytest.raises(ValueError, match="floating-point"):
            _validate_tensor(torch.zeros(5, 3, dtype=torch.int32), "test")

    def test_rejects_1d(self) -> None:
        with pytest.raises(ValueError, match="2-D.*3-D"):
            _validate_tensor(torch.randn(10), "test")

    def test_rejects_4d(self) -> None:
        with pytest.raises(ValueError, match="2-D.*3-D"):
            _validate_tensor(torch.randn(2, 3, 4, 5), "test")

    def test_rejects_nan(self) -> None:
        x = torch.randn(5, 3)
        x[0, 0] = float("nan")
        with pytest.raises(ValueError, match="NaN"):
            _validate_tensor(x, "test")

    def test_rejects_inf(self) -> None:
        x = torch.randn(5, 3)
        x[1, 2] = float("inf")
        with pytest.raises(ValueError, match="Inf"):
            _validate_tensor(x, "test")


# ======================================================================
# GaussianNoiseConfig
# ======================================================================


class TestGaussianNoiseConfig:
    def test_defaults(self) -> None:
        cfg = GaussianNoiseConfig()
        assert cfg.std == pytest.approx(0.05)
        assert cfg.p == pytest.approx(1.0)
        assert cfg.seed is None

    def test_custom(self) -> None:
        cfg = GaussianNoiseConfig(std=0.1, p=0.8, seed=7)
        assert cfg.std == pytest.approx(0.1)
        assert cfg.p == pytest.approx(0.8)
        assert cfg.seed == 7

    def test_negative_std_raises(self) -> None:
        with pytest.raises(ValueError, match="std"):
            GaussianNoiseConfig(std=-0.1)

    def test_invalid_p_raises(self) -> None:
        with pytest.raises(ValueError, match="p must be"):
            GaussianNoiseConfig(p=1.5)

    def test_frozen(self) -> None:
        cfg = GaussianNoiseConfig()
        with pytest.raises((FrozenInstanceError, AttributeError)):
            cfg.std = 99.0  # type: ignore[misc]


# ======================================================================
# GaussianNoise
# ======================================================================


class TestGaussianNoise:
    """Tests for :class:`GaussianNoise`."""

    @pytest.mark.parametrize("device", _DEVICES)
    @pytest.mark.parametrize("shape", [(_T, _S), (_B, _T, _S)])
    def test_shape_preserved(self, shape: Tuple[int, ...], device: torch.device) -> None:
        x = _rand(shape, device=device)
        aug = GaussianNoise(GaussianNoiseConfig(seed=0))
        assert aug(x).shape == x.shape

    def test_dtype_preserved_float32(self, x_2d: torch.Tensor) -> None:
        aug = GaussianNoise()
        assert aug(x_2d).dtype == torch.float32

    def test_dtype_preserved_float64(self) -> None:
        x = torch.randn(_T, _S).double()
        aug = GaussianNoise()
        assert aug(x).dtype == torch.float64

    def test_values_differ_from_input(self, x_2d: torch.Tensor) -> None:
        aug = GaussianNoise(GaussianNoiseConfig(std=1.0, seed=1))
        y = aug(x_2d)
        assert not torch.allclose(y, x_2d)

    def test_zero_std_is_identity(self, x_2d: torch.Tensor) -> None:
        aug = GaussianNoise(GaussianNoiseConfig(std=0.0, seed=0))
        assert torch.allclose(aug(x_2d), x_2d)

    def test_deterministic_with_seed(self, x_2d: torch.Tensor) -> None:
        aug1 = GaussianNoise(GaussianNoiseConfig(std=0.1, seed=42))
        aug2 = GaussianNoise(GaussianNoiseConfig(std=0.1, seed=42))
        assert torch.allclose(aug1(x_2d), aug2(x_2d))

    def test_p_zero_never_applies(self, x_2d: torch.Tensor) -> None:
        aug = GaussianNoise(GaussianNoiseConfig(std=10.0, p=0.0, seed=0))
        y = aug(x_2d)
        assert y is x_2d  # same object

    def test_invalid_config_type(self) -> None:
        with pytest.raises(TypeError, match="GaussianNoiseConfig"):
            GaussianNoise(config=42)  # type: ignore[arg-type]

    def test_rejects_nan(self) -> None:
        aug = GaussianNoise()
        x = torch.randn(_T, _S)
        x[0, 0] = float("nan")
        with pytest.raises(ValueError, match="NaN"):
            aug(x)

    def test_rejects_inf(self) -> None:
        aug = GaussianNoise()
        x = torch.randn(_T, _S)
        x[0, 0] = float("inf")
        with pytest.raises(ValueError, match="Inf"):
            aug(x)

    def test_rejects_non_tensor(self) -> None:
        aug = GaussianNoise()
        with pytest.raises(TypeError):
            aug([[1.0]])  # type: ignore[arg-type]

    def test_rejects_1d(self) -> None:
        aug = GaussianNoise()
        with pytest.raises(ValueError, match="2-D.*3-D"):
            aug(torch.randn(10))

    def test_repr_contains_class_name(self) -> None:
        aug = GaussianNoise(GaussianNoiseConfig(std=0.2, p=0.5, seed=1))
        assert "GaussianNoise" in repr(aug)


# ======================================================================
# RandomScalingConfig
# ======================================================================


class TestRandomScalingConfig:
    def test_defaults(self) -> None:
        cfg = RandomScalingConfig()
        assert cfg.scale_min == pytest.approx(0.8)
        assert cfg.scale_max == pytest.approx(1.2)
        assert cfg.p == pytest.approx(1.0)

    def test_invalid_scale_min(self) -> None:
        with pytest.raises(ValueError, match="scale_min"):
            RandomScalingConfig(scale_min=0.0)

    def test_scale_max_less_than_min(self) -> None:
        with pytest.raises(ValueError, match="scale_max"):
            RandomScalingConfig(scale_min=1.0, scale_max=0.5)


# ======================================================================
# RandomScaling
# ======================================================================


class TestRandomScaling:
    @pytest.mark.parametrize("device", _DEVICES)
    @pytest.mark.parametrize("shape", [(_T, _S), (_B, _T, _S)])
    def test_shape_preserved(self, shape: Tuple[int, ...], device: torch.device) -> None:
        x = _rand(shape, device=device)
        aug = RandomScaling(RandomScalingConfig(seed=0))
        assert aug(x).shape == x.shape

    def test_values_in_scale_range(self) -> None:
        """Each channel should be uniformly scaled."""
        # Use all-ones input so output == scale factor per channel
        x = torch.ones(_T, _S)
        aug = RandomScaling(RandomScalingConfig(scale_min=0.5, scale_max=2.0, seed=7))
        y = aug(x)
        # Each column is a constant (since x=1)
        col_values = y[0]  # (S,)
        assert (col_values >= 0.5).all()
        assert (col_values <= 2.0).all()

    def test_channels_scaled_independently(self) -> None:
        x = torch.ones(_T, _S)
        aug = RandomScaling(RandomScalingConfig(scale_min=0.8, scale_max=1.2, seed=10))
        y = aug(x)
        # Not all channels have identical scale
        col_values = y[0]
        assert not torch.allclose(col_values, col_values[0].expand_as(col_values))

    def test_deterministic_with_seed(self, x_2d: torch.Tensor) -> None:
        aug1 = RandomScaling(RandomScalingConfig(seed=5))
        aug2 = RandomScaling(RandomScalingConfig(seed=5))
        assert torch.allclose(aug1(x_2d), aug2(x_2d))

    def test_p_zero_never_applies(self, x_2d: torch.Tensor) -> None:
        aug = RandomScaling(RandomScalingConfig(p=0.0))
        assert aug(x_2d) is x_2d

    def test_batched_shape(self, x_3d: torch.Tensor) -> None:
        aug = RandomScaling(RandomScalingConfig(seed=0))
        assert aug(x_3d).shape == x_3d.shape


# ======================================================================
# RandomJitterConfig
# ======================================================================


class TestRandomJitterConfig:
    def test_defaults(self) -> None:
        cfg = RandomJitterConfig()
        assert cfg.jitter_std == pytest.approx(0.01)

    def test_negative_std_raises(self) -> None:
        with pytest.raises(ValueError, match="jitter_std"):
            RandomJitterConfig(jitter_std=-0.01)


# ======================================================================
# RandomJitter
# ======================================================================


class TestRandomJitter:
    @pytest.mark.parametrize("shape", [(_T, _S), (_B, _T, _S)])
    def test_shape_preserved(self, shape: Tuple[int, ...]) -> None:
        x = _rand(shape)
        aug = RandomJitter(RandomJitterConfig(seed=0))
        assert aug(x).shape == x.shape

    def test_values_differ_from_input(self, x_2d: torch.Tensor) -> None:
        aug = RandomJitter(RandomJitterConfig(jitter_std=1.0, seed=1))
        assert not torch.allclose(aug(x_2d), x_2d)

    def test_deterministic_with_seed(self, x_2d: torch.Tensor) -> None:
        aug1 = RandomJitter(RandomJitterConfig(seed=9))
        aug2 = RandomJitter(RandomJitterConfig(seed=9))
        assert torch.allclose(aug1(x_2d), aug2(x_2d))

    def test_p_zero_never_applies(self, x_2d: torch.Tensor) -> None:
        aug = RandomJitter(RandomJitterConfig(p=0.0))
        assert aug(x_2d) is x_2d


# ======================================================================
# RandomTimeMaskConfig
# ======================================================================


class TestRandomTimeMaskConfig:
    def test_defaults(self) -> None:
        cfg = RandomTimeMaskConfig()
        assert cfg.mask_ratio == pytest.approx(0.1)
        assert cfg.fill_value == pytest.approx(0.0)

    def test_zero_mask_ratio_raises(self) -> None:
        with pytest.raises(ValueError, match="mask_ratio"):
            RandomTimeMaskConfig(mask_ratio=0.0)

    def test_above_one_raises(self) -> None:
        with pytest.raises(ValueError, match="mask_ratio"):
            RandomTimeMaskConfig(mask_ratio=1.5)


# ======================================================================
# RandomTimeMask
# ======================================================================


class TestRandomTimeMask:
    @pytest.mark.parametrize("shape", [(_T, _S), (_B, _T, _S)])
    def test_shape_preserved(self, shape: Tuple[int, ...]) -> None:
        x = _rand(shape)
        aug = RandomTimeMask(RandomTimeMaskConfig(seed=0))
        assert aug(x).shape == x.shape

    def test_masked_region_is_fill_value(self) -> None:
        """At least some time steps should be zeroed."""
        x = torch.ones(_T, _S)
        aug = RandomTimeMask(RandomTimeMaskConfig(mask_ratio=0.2, fill_value=0.0, seed=3))
        y = aug(x)
        num_zeroed = (y == 0.0).all(dim=-1).sum().item()
        expected = math.ceil(0.2 * _T)
        assert num_zeroed == expected

    def test_custom_fill_value(self) -> None:
        x = torch.ones(_T, _S)
        aug = RandomTimeMask(RandomTimeMaskConfig(mask_ratio=0.1, fill_value=-1.0, seed=4))
        y = aug(x)
        assert (y == -1.0).any()

    def test_deterministic_with_seed(self, x_2d: torch.Tensor) -> None:
        aug1 = RandomTimeMask(RandomTimeMaskConfig(seed=11))
        aug2 = RandomTimeMask(RandomTimeMaskConfig(seed=11))
        assert torch.equal(aug1(x_2d), aug2(x_2d))

    def test_p_zero_never_applies(self, x_2d: torch.Tensor) -> None:
        aug = RandomTimeMask(RandomTimeMaskConfig(p=0.0))
        assert aug(x_2d) is x_2d

    def test_full_mask_ratio(self) -> None:
        """mask_ratio=1.0 masks all time steps."""
        x = torch.ones(_T, _S)
        aug = RandomTimeMask(RandomTimeMaskConfig(mask_ratio=1.0, seed=0))
        y = aug(x)
        assert torch.all(y == 0.0)

    def test_batched_masked_correctly(self, x_3d: torch.Tensor) -> None:
        aug = RandomTimeMask(RandomTimeMaskConfig(mask_ratio=0.1, seed=0))
        y = aug(x_3d)
        assert y.shape == x_3d.shape


# ======================================================================
# RandomChannelMaskConfig
# ======================================================================


class TestRandomChannelMaskConfig:
    def test_defaults(self) -> None:
        cfg = RandomChannelMaskConfig()
        assert cfg.mask_ratio == pytest.approx(0.1)

    def test_zero_raises(self) -> None:
        with pytest.raises(ValueError, match="mask_ratio"):
            RandomChannelMaskConfig(mask_ratio=0.0)


# ======================================================================
# RandomChannelMask
# ======================================================================


class TestRandomChannelMask:
    @pytest.mark.parametrize("shape", [(_T, _S), (_B, _T, _S)])
    def test_shape_preserved(self, shape: Tuple[int, ...]) -> None:
        x = _rand(shape)
        aug = RandomChannelMask(RandomChannelMaskConfig(seed=0))
        assert aug(x).shape == x.shape

    def test_masked_channels_are_fill_value(self) -> None:
        x = torch.ones(_T, _S)
        aug = RandomChannelMask(
            RandomChannelMaskConfig(mask_ratio=0.3, fill_value=0.0, seed=2)
        )
        y = aug(x)
        # At least one full column should be zero
        zeroed_cols = (y == 0.0).all(dim=0).sum().item()
        expected = math.ceil(0.3 * _S)
        assert zeroed_cols == expected

    def test_deterministic_with_seed(self, x_2d: torch.Tensor) -> None:
        aug1 = RandomChannelMask(RandomChannelMaskConfig(seed=13))
        aug2 = RandomChannelMask(RandomChannelMaskConfig(seed=13))
        assert torch.equal(aug1(x_2d), aug2(x_2d))

    def test_p_zero_never_applies(self, x_2d: torch.Tensor) -> None:
        aug = RandomChannelMask(RandomChannelMaskConfig(p=0.0))
        assert aug(x_2d) is x_2d

    def test_batched_shape(self, x_3d: torch.Tensor) -> None:
        aug = RandomChannelMask(RandomChannelMaskConfig(seed=0))
        assert aug(x_3d).shape == x_3d.shape


# ======================================================================
# RandomTimeShiftConfig
# ======================================================================


class TestRandomTimeShiftConfig:
    def test_defaults(self) -> None:
        cfg = RandomTimeShiftConfig()
        assert cfg.max_shift == 10

    def test_negative_shift_raises(self) -> None:
        with pytest.raises(ValueError, match="max_shift"):
            RandomTimeShiftConfig(max_shift=-1)


# ======================================================================
# RandomTimeShift
# ======================================================================


class TestRandomTimeShift:
    @pytest.mark.parametrize("shape", [(_T, _S), (_B, _T, _S)])
    def test_shape_preserved(self, shape: Tuple[int, ...]) -> None:
        x = _rand(shape)
        aug = RandomTimeShift(RandomTimeShiftConfig(seed=0))
        assert aug(x).shape == x.shape

    def test_zero_shift_is_identity(self, x_2d: torch.Tensor) -> None:
        aug = RandomTimeShift(RandomTimeShiftConfig(max_shift=0, seed=0))
        assert torch.equal(aug(x_2d), x_2d)

    def test_circular_shift_preserves_values(self, x_2d: torch.Tensor) -> None:
        """A circular shift should not change the set of values."""
        aug = RandomTimeShift(RandomTimeShiftConfig(max_shift=5, seed=3))
        y = aug(x_2d)
        assert torch.allclose(x_2d.sort(dim=0).values, y.sort(dim=0).values)

    def test_deterministic_with_seed(self, x_2d: torch.Tensor) -> None:
        aug1 = RandomTimeShift(RandomTimeShiftConfig(max_shift=10, seed=17))
        aug2 = RandomTimeShift(RandomTimeShiftConfig(max_shift=10, seed=17))
        assert torch.equal(aug1(x_2d), aug2(x_2d))

    def test_p_zero_never_applies(self, x_2d: torch.Tensor) -> None:
        aug = RandomTimeShift(RandomTimeShiftConfig(p=0.0))
        assert aug(x_2d) is x_2d

    def test_batched_shape(self, x_3d: torch.Tensor) -> None:
        aug = RandomTimeShift(RandomTimeShiftConfig(seed=0))
        assert aug(x_3d).shape == x_3d.shape


# ======================================================================
# IdentityConfig
# ======================================================================


class TestIdentityConfig:
    def test_defaults(self) -> None:
        cfg = IdentityConfig()
        assert cfg.p == pytest.approx(1.0)

    def test_invalid_p(self) -> None:
        with pytest.raises(ValueError):
            IdentityConfig(p=-0.1)


# ======================================================================
# Identity
# ======================================================================


class TestIdentity:
    @pytest.mark.parametrize("shape", [(_T, _S), (_B, _T, _S)])
    def test_returns_same_object(self, shape: Tuple[int, ...]) -> None:
        x = _rand(shape)
        aug = Identity()
        assert aug(x) is x

    def test_dtype_preserved(self, x_2d: torch.Tensor) -> None:
        aug = Identity()
        assert aug(x_2d).dtype == x_2d.dtype

    def test_invalid_config_type(self) -> None:
        with pytest.raises(TypeError, match="IdentityConfig"):
            Identity(config="bad")  # type: ignore[arg-type]

    def test_rejects_nan(self) -> None:
        aug = Identity()
        x = torch.randn(_T, _S)
        x[0, 0] = float("nan")
        with pytest.raises(ValueError, match="NaN"):
            aug(x)


# ======================================================================
# ComposeConfig
# ======================================================================


class TestComposeConfig:
    def test_defaults(self) -> None:
        cfg = ComposeConfig()
        assert cfg.p == pytest.approx(1.0)

    def test_invalid_p(self) -> None:
        with pytest.raises(ValueError):
            ComposeConfig(p=2.0)


# ======================================================================
# Compose
# ======================================================================


class TestCompose:
    def _make_compose(self) -> Compose:
        return Compose(
            [
                GaussianNoise(GaussianNoiseConfig(std=0.01, seed=0)),
                RandomScaling(RandomScalingConfig(seed=1)),
                RandomTimeMask(RandomTimeMaskConfig(seed=2)),
            ]
        )

    def test_shape_preserved_2d(self, x_2d: torch.Tensor) -> None:
        assert self._make_compose()(x_2d).shape == x_2d.shape

    def test_shape_preserved_3d(self, x_3d: torch.Tensor) -> None:
        assert self._make_compose()(x_3d).shape == x_3d.shape

    def test_empty_compose_is_identity(self, x_2d: torch.Tensor) -> None:
        aug = Compose([])
        assert aug(x_2d) is x_2d

    def test_single_augmentation(self, x_2d: torch.Tensor) -> None:
        aug = Compose([Identity()])
        assert aug(x_2d) is x_2d

    def test_len(self) -> None:
        compose = self._make_compose()
        assert len(compose) == 3

    def test_p_zero_skips_all(self, x_2d: torch.Tensor) -> None:
        aug = Compose(
            [GaussianNoise(GaussianNoiseConfig(std=10.0))],
            config=ComposeConfig(p=0.0),
        )
        assert aug(x_2d) is x_2d

    def test_invalid_augmentation_type_raises(self) -> None:
        with pytest.raises(TypeError, match="BaseAugmentation"):
            Compose(["not_an_aug"])  # type: ignore[list-item]

    def test_invalid_augmentations_not_list(self) -> None:
        with pytest.raises(TypeError, match="list"):
            Compose(GaussianNoise())  # type: ignore[arg-type]

    def test_invalid_config_type(self) -> None:
        with pytest.raises(TypeError, match="ComposeConfig"):
            Compose([], config=42)  # type: ignore[arg-type]

    def test_rejects_nan(self) -> None:
        aug = Compose([Identity()])
        x = torch.randn(_T, _S)
        x[0, 0] = float("nan")
        with pytest.raises(ValueError, match="NaN"):
            aug(x)

    def test_rejects_inf(self) -> None:
        aug = Compose([Identity()])
        x = torch.randn(_T, _S)
        x[0, 0] = float("inf")
        with pytest.raises(ValueError, match="Inf"):
            aug(x)

    def test_augmentation_property(self) -> None:
        compose = self._make_compose()
        assert len(compose.augmentations) == 3
        assert isinstance(compose.augmentations[0], GaussianNoise)

    def test_repr_contains_compose(self) -> None:
        compose = self._make_compose()
        assert "Compose" in repr(compose)

    @pytest.mark.parametrize("device", _DEVICES)
    def test_device_compatibility(self, device: torch.device) -> None:
        x = _rand((_T, _S), device=device)
        aug = Compose(
            [
                GaussianNoise(GaussianNoiseConfig(seed=0)),
                RandomScaling(RandomScalingConfig(seed=1)),
            ]
        )
        y = aug(x)
        assert y.device.type == device.type
        assert y.shape == x.shape


# ======================================================================
# Cross-augmentation: probability = 1 always augments
# ======================================================================


class TestProbabilityAlwaysApplies:
    """Verify that p=1.0 always applies the augmentation."""

    @pytest.mark.parametrize(
        "aug_cls, cfg",
        [
            (GaussianNoise, GaussianNoiseConfig(std=1.0, p=1.0, seed=0)),
            (RandomScaling, RandomScalingConfig(scale_min=2.0, scale_max=3.0, p=1.0, seed=0)),
            (RandomJitter, RandomJitterConfig(jitter_std=1.0, p=1.0, seed=0)),
            (RandomTimeMask, RandomTimeMaskConfig(mask_ratio=0.5, p=1.0, seed=0)),
            (RandomChannelMask, RandomChannelMaskConfig(mask_ratio=0.5, p=1.0, seed=0)),
            (RandomTimeShift, RandomTimeShiftConfig(max_shift=5, p=1.0, seed=0)),
        ],
    )
    def test_p1_changes_input(
        self, aug_cls: type, cfg: Any, x_2d: torch.Tensor
    ) -> None:
        aug = aug_cls(cfg)
        y = aug(x_2d)
        # Identity and zero-std noise are excluded from this test
        if not isinstance(aug, Identity):
            assert not torch.equal(y, x_2d) or isinstance(
                aug, (RandomTimeShift,)
            ), "Augmentation should have changed the tensor"


# ======================================================================
# Float dtype round-trip
# ======================================================================


class TestDtypeRoundTrip:
    """Augmentations must preserve the input dtype."""

    @pytest.mark.parametrize(
        "aug",
        [
            GaussianNoise(GaussianNoiseConfig(seed=0)),
            RandomScaling(RandomScalingConfig(seed=0)),
            RandomJitter(RandomJitterConfig(seed=0)),
            RandomTimeMask(RandomTimeMaskConfig(seed=0)),
            RandomChannelMask(RandomChannelMaskConfig(seed=0)),
            RandomTimeShift(RandomTimeShiftConfig(seed=0)),
            Identity(),
        ],
    )
    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    def test_dtype_preserved(self, aug: BaseAugmentation, dtype: torch.dtype) -> None:
        x = torch.randn(_T, _S, dtype=dtype)
        y = aug(x)
        assert y.dtype == dtype
