"""Tests for :mod:`src.data.view_generator`.

Covers:
- Default pipeline construction
- Custom pipeline injection
- Shape preservation for (T, S) and (B, T, S)
- Both views differ from each other and from the original
- View independence (the two pipelines are decorrelated)
- Deterministic behaviour via seed
- generate() alias
- Invalid config / transform types
- NaN / Inf rejection
- Non-tensor rejection
- Wrong dimensions rejected
- CPU and CUDA (when available)
- ContrastiveViewGeneratorConfig validation
- repr()
"""

from __future__ import annotations

from typing import List, Tuple

import pytest
import torch

from src.data.augmentations import (
    Compose,
    ComposeConfig,
    GaussianNoise,
    GaussianNoiseConfig,
    Identity,
    IdentityConfig,
    RandomScaling,
    RandomScalingConfig,
)
from src.data.view_generator import (
    ContrastiveViewGenerator,
    ContrastiveViewGeneratorConfig,
    _build_default_pipeline,
)


# ======================================================================
# Constants & helpers
# ======================================================================

_T = 60
_S = 10
_B = 4

_DEVICES: List[torch.device] = [torch.device("cpu")]
if torch.cuda.is_available():
    _DEVICES.append(torch.device("cuda"))


def _rand(shape: Tuple[int, ...], device: torch.device = torch.device("cpu")) -> torch.Tensor:
    torch.manual_seed(42)
    return torch.randn(*shape, device=device)


# ======================================================================
# Fixtures
# ======================================================================


@pytest.fixture()
def x_2d() -> torch.Tensor:
    return _rand((_T, _S))


@pytest.fixture()
def x_3d() -> torch.Tensor:
    return _rand((_B, _T, _S))


@pytest.fixture()
def generator() -> ContrastiveViewGenerator:
    return ContrastiveViewGenerator()


# ======================================================================
# _build_default_pipeline
# ======================================================================


class TestBuildDefaultPipeline:
    def test_returns_compose(self) -> None:
        pipeline = _build_default_pipeline(seed_offset=0)
        assert isinstance(pipeline, Compose)

    def test_pipeline_has_augmentations(self) -> None:
        pipeline = _build_default_pipeline(seed_offset=0)
        assert len(pipeline) > 0

    def test_different_offsets_differ(self, x_2d: torch.Tensor) -> None:
        p1 = _build_default_pipeline(seed_offset=0)
        p2 = _build_default_pipeline(seed_offset=100)
        # Apply both; at least one sensor value should differ
        y1 = p1(x_2d)
        y2 = p2(x_2d)
        assert not torch.equal(y1, y2)

    def test_preserves_shape(self, x_2d: torch.Tensor) -> None:
        pipeline = _build_default_pipeline(seed_offset=0)
        assert pipeline(x_2d).shape == x_2d.shape


# ======================================================================
# ContrastiveViewGeneratorConfig
# ======================================================================


class TestContrastiveViewGeneratorConfig:
    def test_defaults(self) -> None:
        cfg = ContrastiveViewGeneratorConfig()
        assert cfg.seed is None
        assert cfg.use_default_pipeline is True

    def test_custom_seed(self) -> None:
        cfg = ContrastiveViewGeneratorConfig(seed=42)
        assert cfg.seed == 42

    def test_invalid_seed_type(self) -> None:
        with pytest.raises(TypeError, match="seed"):
            ContrastiveViewGeneratorConfig(seed=3.14)  # type: ignore[arg-type]


# ======================================================================
# ContrastiveViewGenerator — construction
# ======================================================================


class TestContrastiveViewGeneratorConstruction:
    def test_default_construction(self) -> None:
        gen = ContrastiveViewGenerator()
        assert gen.config.use_default_pipeline is True

    def test_invalid_config_type(self) -> None:
        with pytest.raises(TypeError, match="ContrastiveViewGeneratorConfig"):
            ContrastiveViewGenerator(config=42)  # type: ignore[arg-type]

    def test_custom_transforms_injected(self) -> None:
        t1 = GaussianNoise(GaussianNoiseConfig(seed=0))
        t2 = RandomScaling(RandomScalingConfig(seed=1))
        gen = ContrastiveViewGenerator(
            config=ContrastiveViewGeneratorConfig(use_default_pipeline=False),
            transform_1=t1,
            transform_2=t2,
        )
        assert gen.transform_1 is t1
        assert gen.transform_2 is t2

    def test_custom_compose_transforms_injected(self) -> None:
        t1 = Compose([GaussianNoise(GaussianNoiseConfig(seed=0))], config=ComposeConfig())
        t2 = Compose([Identity(IdentityConfig())], config=ComposeConfig())
        gen = ContrastiveViewGenerator(
            config=ContrastiveViewGeneratorConfig(use_default_pipeline=False),
            transform_1=t1,
            transform_2=t2,
        )
        assert gen.transform_1 is t1
        assert gen.transform_2 is t2

    def test_missing_transform_1_raises(self) -> None:
        with pytest.raises(ValueError, match="transform_1"):
            ContrastiveViewGenerator(
                config=ContrastiveViewGeneratorConfig(use_default_pipeline=False),
                transform_1=None,
                transform_2=GaussianNoise(),
            )

    def test_missing_transform_2_raises(self) -> None:
        with pytest.raises(ValueError, match="transform_2"):
            ContrastiveViewGenerator(
                config=ContrastiveViewGeneratorConfig(use_default_pipeline=False),
                transform_1=GaussianNoise(),
                transform_2=None,
            )

    def test_invalid_transform_1_type(self) -> None:
        with pytest.raises(TypeError, match="transform_1"):
            ContrastiveViewGenerator(
                config=ContrastiveViewGeneratorConfig(use_default_pipeline=False),
                transform_1="bad",  # type: ignore[arg-type]
                transform_2=GaussianNoise(),
            )

    def test_invalid_transform_2_type(self) -> None:
        with pytest.raises(TypeError, match="transform_2"):
            ContrastiveViewGenerator(
                config=ContrastiveViewGeneratorConfig(use_default_pipeline=False),
                transform_1=GaussianNoise(),
                transform_2="bad",  # type: ignore[arg-type]
            )


# ======================================================================
# ContrastiveViewGenerator — shape & type correctness
# ======================================================================


class TestContrastiveViewGeneratorOutputShape:
    @pytest.mark.parametrize("device", _DEVICES)
    def test_2d_shape_preserved(self, device: torch.device) -> None:
        x = _rand((_T, _S), device=device)
        gen = ContrastiveViewGenerator()
        v1, v2 = gen(x)
        assert v1.shape == x.shape
        assert v2.shape == x.shape

    @pytest.mark.parametrize("device", _DEVICES)
    def test_3d_shape_preserved(self, device: torch.device) -> None:
        x = _rand((_B, _T, _S), device=device)
        gen = ContrastiveViewGenerator()
        v1, v2 = gen(x)
        assert v1.shape == x.shape
        assert v2.shape == x.shape

    def test_dtype_preserved_float32(self, x_2d: torch.Tensor) -> None:
        gen = ContrastiveViewGenerator()
        v1, v2 = gen(x_2d)
        assert v1.dtype == torch.float32
        assert v2.dtype == torch.float32

    def test_dtype_preserved_float64(self) -> None:
        x = torch.randn(_T, _S).double()
        gen = ContrastiveViewGenerator()
        v1, v2 = gen(x)
        assert v1.dtype == torch.float64
        assert v2.dtype == torch.float64

    def test_returns_tuple_of_two(self, x_2d: torch.Tensor) -> None:
        gen = ContrastiveViewGenerator()
        result = gen(x_2d)
        assert isinstance(result, tuple)
        assert len(result) == 2


# ======================================================================
# ContrastiveViewGenerator — view independence
# ======================================================================


class TestContrastiveViewGeneratorIndependence:
    def test_views_differ_from_each_other(self, x_2d: torch.Tensor) -> None:
        """The two augmented views should not be identical."""
        gen = ContrastiveViewGenerator()
        v1, v2 = gen(x_2d)
        assert not torch.equal(v1, v2)

    def test_views_differ_from_original(self, x_2d: torch.Tensor) -> None:
        """Views should generally differ from the original (with high-std noise)."""
        gen = ContrastiveViewGenerator(
            config=ContrastiveViewGeneratorConfig(use_default_pipeline=False),
            transform_1=GaussianNoise(GaussianNoiseConfig(std=1.0, p=1.0, seed=0)),
            transform_2=GaussianNoise(GaussianNoiseConfig(std=1.0, p=1.0, seed=100)),
        )
        v1, v2 = gen(x_2d)
        assert not torch.equal(v1, x_2d)
        assert not torch.equal(v2, x_2d)

    def test_identity_views_equal_input(self, x_2d: torch.Tensor) -> None:
        """Identity transforms should return the original tensor values."""
        gen = ContrastiveViewGenerator(
            config=ContrastiveViewGeneratorConfig(use_default_pipeline=False),
            transform_1=Identity(),
            transform_2=Identity(),
        )
        v1, v2 = gen(x_2d)
        assert torch.equal(v1, x_2d)
        assert torch.equal(v2, x_2d)


# ======================================================================
# ContrastiveViewGenerator — determinism
# ======================================================================


class TestContrastiveViewGeneratorDeterminism:
    def test_seeded_generator_is_deterministic(self, x_2d: torch.Tensor) -> None:
        gen1 = ContrastiveViewGenerator(
            config=ContrastiveViewGeneratorConfig(seed=42)
        )
        gen2 = ContrastiveViewGenerator(
            config=ContrastiveViewGeneratorConfig(seed=42)
        )
        v1a, v2a = gen1(x_2d)
        v1b, v2b = gen2(x_2d)
        assert torch.equal(v1a, v1b)
        assert torch.equal(v2a, v2b)

    def test_seeded_aug_pipelines_are_deterministic(self, x_2d: torch.Tensor) -> None:
        t1 = GaussianNoise(GaussianNoiseConfig(std=0.1, seed=7))
        t2 = GaussianNoise(GaussianNoiseConfig(std=0.1, seed=77))
        gen = ContrastiveViewGenerator(
            config=ContrastiveViewGeneratorConfig(use_default_pipeline=False),
            transform_1=t1,
            transform_2=t2,
        )
        v1a, v2a = gen(x_2d)
        v1b, v2b = gen(x_2d)
        assert torch.equal(v1a, v1b)
        assert torch.equal(v2a, v2b)


# ======================================================================
# ContrastiveViewGenerator — validation
# ======================================================================


class TestContrastiveViewGeneratorValidation:
    def test_rejects_non_tensor(self, generator: ContrastiveViewGenerator) -> None:
        with pytest.raises(TypeError, match="torch.Tensor"):
            generator([[1.0, 2.0]])  # type: ignore[arg-type]

    def test_rejects_integer_dtype(self, generator: ContrastiveViewGenerator) -> None:
        with pytest.raises(ValueError, match="floating-point"):
            generator(torch.zeros(_T, _S, dtype=torch.int32))

    def test_rejects_1d(self, generator: ContrastiveViewGenerator) -> None:
        with pytest.raises(ValueError, match="2-D.*3-D"):
            generator(torch.randn(_T))

    def test_rejects_4d(self, generator: ContrastiveViewGenerator) -> None:
        with pytest.raises(ValueError, match="2-D.*3-D"):
            generator(torch.randn(2, _B, _T, _S))

    def test_rejects_nan(self, generator: ContrastiveViewGenerator) -> None:
        x = torch.randn(_T, _S)
        x[0, 0] = float("nan")
        with pytest.raises(ValueError, match="NaN"):
            generator(x)

    def test_rejects_inf(self, generator: ContrastiveViewGenerator) -> None:
        x = torch.randn(_T, _S)
        x[0, 0] = float("inf")
        with pytest.raises(ValueError, match="Inf"):
            generator(x)


# ======================================================================
# ContrastiveViewGenerator — generate() alias
# ======================================================================


class TestContrastiveViewGeneratorGenerateAlias:
    def test_generate_returns_same_result_as_call(self, x_2d: torch.Tensor) -> None:
        gen = ContrastiveViewGenerator(
            config=ContrastiveViewGeneratorConfig(seed=0)
        )
        v1a, v2a = gen(x_2d)
        # Reset seed for second call
        gen2 = ContrastiveViewGenerator(
            config=ContrastiveViewGeneratorConfig(seed=0)
        )
        v1b, v2b = gen2.generate(x_2d)
        assert torch.equal(v1a, v1b)
        assert torch.equal(v2a, v2b)

    def test_generate_shape(self, x_2d: torch.Tensor) -> None:
        gen = ContrastiveViewGenerator()
        v1, v2 = gen.generate(x_2d)
        assert v1.shape == x_2d.shape
        assert v2.shape == x_2d.shape


# ======================================================================
# ContrastiveViewGenerator — properties and repr
# ======================================================================


class TestContrastiveViewGeneratorProperties:
    def test_config_property(self) -> None:
        cfg = ContrastiveViewGeneratorConfig(seed=7)
        gen = ContrastiveViewGenerator(config=cfg)
        assert gen.config is cfg

    def test_transform_1_property(self) -> None:
        t1 = Identity()
        gen = ContrastiveViewGenerator(
            config=ContrastiveViewGeneratorConfig(use_default_pipeline=False),
            transform_1=t1,
            transform_2=Identity(),
        )
        assert gen.transform_1 is t1

    def test_transform_2_property(self) -> None:
        t2 = Identity()
        gen = ContrastiveViewGenerator(
            config=ContrastiveViewGeneratorConfig(use_default_pipeline=False),
            transform_1=Identity(),
            transform_2=t2,
        )
        assert gen.transform_2 is t2

    def test_repr_contains_class_name(self) -> None:
        gen = ContrastiveViewGenerator()
        assert "ContrastiveViewGenerator" in repr(gen)


# ======================================================================
# Batch vs single input equivalence
# ======================================================================


class TestBatchSingleEquivalence:
    """Applying augmentations to a single window should give the same
    result as the corresponding slice of a batched application,
    provided the same seed is used."""

    def test_identity_2d_vs_3d_slice(self) -> None:
        gen = ContrastiveViewGenerator(
            config=ContrastiveViewGeneratorConfig(use_default_pipeline=False),
            transform_1=Identity(),
            transform_2=Identity(),
        )
        x_single = _rand((_T, _S))
        x_batch = x_single.unsqueeze(0).expand(_B, -1, -1).clone()

        v1s, v2s = gen(x_single)
        v1b, v2b = gen(x_batch)

        # The first batch item should equal the single output
        assert torch.equal(v1s, v1b[0])
        assert torch.equal(v2s, v2b[0])


# ======================================================================
# Public API surface (src.data)
# ======================================================================


class TestPublicAPIImports:
    """Ensure everything is importable from src.data."""

    def test_imports(self) -> None:
        from src.data import (  # noqa: F401
            BaseAugmentation,
            Compose,
            ComposeConfig,
            ContrastiveViewGenerator,
            ContrastiveViewGeneratorConfig,
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
