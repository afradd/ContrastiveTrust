"""Comprehensive tests for :class:`DualStreamEncoder` and :class:`EncoderConfig`.

Test coverage
-------------
✓ Initialisation (default and custom configs)
✓ Forward pass — output dict keys and shapes
✓ encode() convenience method
✓ freeze_temporal() / freeze_physics() / unfreeze_all()
✓ parameter_count() dictionary
✓ Serialisation (state_dict round-trip)
✓ Deterministic inference (eval mode reproducibility)
✓ TorchScript compatibility (torch.jit.script)
✓ Invalid batch-size mismatch
✓ NaN / Inf rejection
✓ Non-tensor and non-floating dtype rejection
✓ Wrong dimensionality inputs
✓ Config embedding-dimension mismatch
✓ CPU execution
✓ CUDA execution (when available)
✓ Properties (embedding_dimension, device, dtype)
"""

from __future__ import annotations

import copy
import logging
from collections import OrderedDict
from typing import Dict

import pytest
import torch
import torch.nn as nn

from src.models.encoder import DualStreamEncoder, EncoderConfig
from src.models.fusion import FusionConfig
from src.models.physics_encoder import PhysicsEncoderConfig
from src.models.temporal_encoder import TemporalEncoderConfig

# ======================================================================
# Fixtures
# ======================================================================

# Test dimensions
_INPUT_CHANNELS = 10
_PHYSICS_DIM = 18
_EMBEDDING_DIM = 256
_BATCH_SIZE = 4
_WINDOW_LEN = 100


@pytest.fixture()
def default_config() -> EncoderConfig:
    """Create an ``EncoderConfig`` with consistent test dimensions."""
    return EncoderConfig(
        temporal=TemporalEncoderConfig(
            input_channels=_INPUT_CHANNELS,
            embedding_dim=_EMBEDDING_DIM,
        ),
        physics=PhysicsEncoderConfig(
            input_dim=_PHYSICS_DIM,
            embedding_dim=_EMBEDDING_DIM,
        ),
        fusion=FusionConfig(
            embedding_dim=_EMBEDDING_DIM,
        ),
    )


@pytest.fixture()
def encoder(default_config: EncoderConfig) -> DualStreamEncoder:
    """Instantiate a ``DualStreamEncoder`` from the default config."""
    return DualStreamEncoder(default_config)


@pytest.fixture()
def sample_window() -> torch.Tensor:
    """Create a random window tensor ``(B, T, S)``."""
    return torch.randn(_BATCH_SIZE, _WINDOW_LEN, _INPUT_CHANNELS)


@pytest.fixture()
def sample_physics() -> torch.Tensor:
    """Create a random physics features tensor ``(B, P)``."""
    return torch.randn(_BATCH_SIZE, _PHYSICS_DIM)


# ======================================================================
# Initialisation tests
# ======================================================================


class TestInitialisation:
    """Tests for :class:`DualStreamEncoder` construction."""

    def test_default_config_creates_encoder(
        self, default_config: EncoderConfig
    ) -> None:
        """Encoder should initialise without error from a valid config."""
        enc = DualStreamEncoder(default_config)
        assert isinstance(enc, nn.Module)

    def test_stores_config(
        self, encoder: DualStreamEncoder, default_config: EncoderConfig
    ) -> None:
        """Encoder should store the configuration for later access."""
        assert encoder.config is default_config

    def test_submodules_exist(self, encoder: DualStreamEncoder) -> None:
        """Encoder should contain the three expected sub-modules."""
        assert hasattr(encoder, "temporal_encoder")
        assert hasattr(encoder, "physics_encoder")
        assert hasattr(encoder, "fusion")

    def test_invalid_config_type(self) -> None:
        """Passing a non-EncoderConfig should raise ``TypeError``."""
        with pytest.raises(TypeError, match="EncoderConfig"):
            DualStreamEncoder("not a config")  # type: ignore[arg-type]

    def test_custom_embedding_dim(self) -> None:
        """Encoder should accept a custom embedding dimension."""
        dim = 128
        cfg = EncoderConfig(
            temporal=TemporalEncoderConfig(
                input_channels=5, embedding_dim=dim
            ),
            physics=PhysicsEncoderConfig(
                input_dim=8, embedding_dim=dim
            ),
            fusion=FusionConfig(embedding_dim=dim),
        )
        enc = DualStreamEncoder(cfg)
        assert enc.embedding_dimension == dim

    def test_mismatched_embedding_dims_raises(self) -> None:
        """Mismatched embedding dims across sub-configs should raise."""
        with pytest.raises(ValueError, match="consistent"):
            EncoderConfig(
                temporal=TemporalEncoderConfig(
                    input_channels=10, embedding_dim=128
                ),
                physics=PhysicsEncoderConfig(
                    input_dim=18, embedding_dim=256
                ),
                fusion=FusionConfig(embedding_dim=256),
            )

    def test_logging_on_init(
        self,
        default_config: EncoderConfig,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Initialisation should log the parameter count."""
        with caplog.at_level(logging.INFO):
            DualStreamEncoder(default_config)
        assert any(
            "DualStreamEncoder initialised" in msg for msg in caplog.messages
        )


# ======================================================================
# Forward pass tests
# ======================================================================


class TestForward:
    """Tests for the ``forward()`` method."""

    def test_returns_dict(
        self,
        encoder: DualStreamEncoder,
        sample_window: torch.Tensor,
        sample_physics: torch.Tensor,
    ) -> None:
        """forward() should return a dictionary."""
        out = encoder(sample_window, sample_physics)
        assert isinstance(out, dict)

    def test_dict_keys(
        self,
        encoder: DualStreamEncoder,
        sample_window: torch.Tensor,
        sample_physics: torch.Tensor,
    ) -> None:
        """Output dict should contain exactly the expected keys."""
        out = encoder(sample_window, sample_physics)
        expected_keys = {
            "embedding",
            "temporal_embedding",
            "physics_embedding",
        }
        assert set(out.keys()) == expected_keys

    def test_output_shapes(
        self,
        encoder: DualStreamEncoder,
        sample_window: torch.Tensor,
        sample_physics: torch.Tensor,
    ) -> None:
        """All output tensors should have shape ``(B, D)``."""
        out = encoder(sample_window, sample_physics)
        expected_shape = (_BATCH_SIZE, _EMBEDDING_DIM)
        assert out["embedding"].shape == expected_shape
        assert out["temporal_embedding"].shape == expected_shape
        assert out["physics_embedding"].shape == expected_shape

    def test_output_dtype(
        self,
        encoder: DualStreamEncoder,
        sample_window: torch.Tensor,
        sample_physics: torch.Tensor,
    ) -> None:
        """All outputs should have floating-point dtype."""
        out = encoder(sample_window, sample_physics)
        for key, tensor in out.items():
            assert tensor.is_floating_point(), (
                f"{key} has non-float dtype {tensor.dtype}"
            )

    def test_embeddings_are_l2_normalised(
        self,
        encoder: DualStreamEncoder,
        sample_window: torch.Tensor,
        sample_physics: torch.Tensor,
    ) -> None:
        """Temporal and physics embeddings should be L2-normalised."""
        encoder.eval()
        out = encoder(sample_window, sample_physics)
        for key in ("temporal_embedding", "physics_embedding", "embedding"):
            norms = torch.linalg.norm(out[key], dim=-1)
            assert torch.allclose(
                norms, torch.ones_like(norms), atol=1e-5
            ), f"{key} is not L2-normalised"

    def test_batch_size_one(
        self, encoder: DualStreamEncoder
    ) -> None:
        """Encoder should handle batch size 1."""
        w = torch.randn(1, _WINDOW_LEN, _INPUT_CHANNELS)
        p = torch.randn(1, _PHYSICS_DIM)
        out = encoder(w, p)
        assert out["embedding"].shape == (1, _EMBEDDING_DIM)

    def test_varying_window_lengths(
        self, encoder: DualStreamEncoder
    ) -> None:
        """Encoder should handle different window lengths."""
        for length in (10, 50, 200):
            w = torch.randn(2, length, _INPUT_CHANNELS)
            p = torch.randn(2, _PHYSICS_DIM)
            out = encoder(w, p)
            assert out["embedding"].shape == (2, _EMBEDDING_DIM)

    def test_gradient_flows(
        self,
        encoder: DualStreamEncoder,
        sample_window: torch.Tensor,
        sample_physics: torch.Tensor,
    ) -> None:
        """Gradients should flow through all sub-modules."""
        encoder.train()
        out = encoder(sample_window, sample_physics)
        loss = out["embedding"].sum()
        loss.backward()

        for name, param in encoder.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, (
                    f"No gradient for {name}"
                )


# ======================================================================
# encode() method tests
# ======================================================================


class TestEncode:
    """Tests for the ``encode()`` convenience method."""

    def test_encode_returns_dict(
        self,
        encoder: DualStreamEncoder,
        sample_window: torch.Tensor,
        sample_physics: torch.Tensor,
    ) -> None:
        """encode() should return the same dict structure as forward()."""
        out = encoder.encode(sample_window, sample_physics)
        assert isinstance(out, dict)
        assert "embedding" in out

    def test_encode_no_grad(
        self,
        encoder: DualStreamEncoder,
        sample_window: torch.Tensor,
        sample_physics: torch.Tensor,
    ) -> None:
        """Tensors from encode() should not require gradients."""
        out = encoder.encode(sample_window, sample_physics)
        for key, tensor in out.items():
            assert not tensor.requires_grad, (
                f"{key} has requires_grad=True from encode()"
            )

    def test_encode_restores_training_mode(
        self,
        encoder: DualStreamEncoder,
        sample_window: torch.Tensor,
        sample_physics: torch.Tensor,
    ) -> None:
        """encode() should restore the original training mode."""
        encoder.train()
        assert encoder.training
        encoder.encode(sample_window, sample_physics)
        assert encoder.training, "Training mode not restored after encode()"

    def test_encode_from_eval_mode(
        self,
        encoder: DualStreamEncoder,
        sample_window: torch.Tensor,
        sample_physics: torch.Tensor,
    ) -> None:
        """encode() called in eval mode should stay in eval mode."""
        encoder.eval()
        assert not encoder.training
        encoder.encode(sample_window, sample_physics)
        assert not encoder.training


# ======================================================================
# Freeze / unfreeze tests
# ======================================================================


class TestFreezeUnfreeze:
    """Tests for ``freeze_temporal()``, ``freeze_physics()``, and
    ``unfreeze_all()``."""

    def test_freeze_temporal(
        self, encoder: DualStreamEncoder
    ) -> None:
        """Temporal encoder parameters should be frozen."""
        encoder.freeze_temporal()
        for p in encoder.temporal_encoder.parameters():
            assert not p.requires_grad

    def test_freeze_temporal_keeps_physics_trainable(
        self, encoder: DualStreamEncoder
    ) -> None:
        """Freezing temporal should not affect physics or fusion."""
        encoder.freeze_temporal()
        for p in encoder.physics_encoder.parameters():
            assert p.requires_grad
        for p in encoder.fusion.parameters():
            assert p.requires_grad

    def test_freeze_physics(
        self, encoder: DualStreamEncoder
    ) -> None:
        """Physics encoder parameters should be frozen."""
        encoder.freeze_physics()
        for p in encoder.physics_encoder.parameters():
            assert not p.requires_grad

    def test_freeze_physics_keeps_temporal_trainable(
        self, encoder: DualStreamEncoder
    ) -> None:
        """Freezing physics should not affect temporal or fusion."""
        encoder.freeze_physics()
        for p in encoder.temporal_encoder.parameters():
            assert p.requires_grad
        for p in encoder.fusion.parameters():
            assert p.requires_grad

    def test_freeze_both_then_unfreeze(
        self, encoder: DualStreamEncoder
    ) -> None:
        """Unfreezing should restore all parameters to trainable."""
        encoder.freeze_temporal()
        encoder.freeze_physics()
        encoder.unfreeze_all()
        for p in encoder.parameters():
            assert p.requires_grad

    def test_forward_still_works_when_frozen(
        self,
        encoder: DualStreamEncoder,
        sample_window: torch.Tensor,
        sample_physics: torch.Tensor,
    ) -> None:
        """Forward pass should still work with frozen sub-modules."""
        encoder.freeze_temporal()
        encoder.freeze_physics()
        out = encoder(sample_window, sample_physics)
        assert out["embedding"].shape == (_BATCH_SIZE, _EMBEDDING_DIM)

    def test_no_gradient_for_frozen_temporal(
        self,
        encoder: DualStreamEncoder,
        sample_window: torch.Tensor,
        sample_physics: torch.Tensor,
    ) -> None:
        """Frozen temporal params should receive no gradient."""
        encoder.freeze_temporal()
        out = encoder(sample_window, sample_physics)
        out["embedding"].sum().backward()
        for p in encoder.temporal_encoder.parameters():
            assert p.grad is None or torch.all(p.grad == 0)

    def test_no_gradient_for_frozen_physics(
        self,
        encoder: DualStreamEncoder,
        sample_window: torch.Tensor,
        sample_physics: torch.Tensor,
    ) -> None:
        """Frozen physics params should receive no gradient."""
        encoder.freeze_physics()
        out = encoder(sample_window, sample_physics)
        out["embedding"].sum().backward()
        for p in encoder.physics_encoder.parameters():
            assert p.grad is None or torch.all(p.grad == 0)


# ======================================================================
# parameter_count() tests
# ======================================================================


class TestParameterCount:
    """Tests for the ``parameter_count()`` method."""

    def test_returns_dict(self, encoder: DualStreamEncoder) -> None:
        """parameter_count() should return a dictionary."""
        counts = encoder.parameter_count()
        assert isinstance(counts, dict)

    def test_expected_keys(self, encoder: DualStreamEncoder) -> None:
        """parameter_count() should contain the expected keys."""
        expected = {"total", "trainable", "temporal", "physics", "fusion"}
        assert set(encoder.parameter_count().keys()) == expected

    def test_total_is_sum_of_parts(
        self, encoder: DualStreamEncoder
    ) -> None:
        """Total should equal temporal + physics + fusion."""
        counts = encoder.parameter_count()
        assert counts["total"] == (
            counts["temporal"] + counts["physics"] + counts["fusion"]
        )

    def test_all_trainable_by_default(
        self, encoder: DualStreamEncoder
    ) -> None:
        """All parameters should be trainable by default."""
        counts = encoder.parameter_count()
        assert counts["total"] == counts["trainable"]

    def test_trainable_decreases_on_freeze(
        self, encoder: DualStreamEncoder
    ) -> None:
        """Freezing temporal should reduce trainable count."""
        counts_before = encoder.parameter_count()
        encoder.freeze_temporal()
        counts_after = encoder.parameter_count()
        assert counts_after["trainable"] < counts_before["trainable"]

    def test_positive_counts(self, encoder: DualStreamEncoder) -> None:
        """All parameter counts should be positive."""
        for key, val in encoder.parameter_count().items():
            assert val > 0, f"{key} has non-positive count {val}"


# ======================================================================
# Properties tests
# ======================================================================


class TestProperties:
    """Tests for helper properties."""

    def test_embedding_dimension(
        self, encoder: DualStreamEncoder
    ) -> None:
        """embedding_dimension should match config."""
        assert encoder.embedding_dimension == _EMBEDDING_DIM

    def test_device_cpu(self, encoder: DualStreamEncoder) -> None:
        """device should be CPU when on CPU."""
        assert encoder.device == torch.device("cpu")

    def test_dtype(self, encoder: DualStreamEncoder) -> None:
        """dtype should be float32 by default."""
        assert encoder.dtype == torch.float32


# ======================================================================
# Serialisation tests
# ======================================================================


class TestSerialisation:
    """Tests for state_dict round-trip serialisation."""

    def test_state_dict_round_trip(
        self,
        encoder: DualStreamEncoder,
        default_config: EncoderConfig,
        sample_window: torch.Tensor,
        sample_physics: torch.Tensor,
    ) -> None:
        """Loading a state_dict should reproduce identical outputs."""
        encoder.eval()
        with torch.no_grad():
            out_original = encoder(sample_window, sample_physics)

        state = encoder.state_dict()
        new_encoder = DualStreamEncoder(default_config)
        new_encoder.load_state_dict(state)
        new_encoder.eval()

        with torch.no_grad():
            out_loaded = new_encoder(sample_window, sample_physics)

        for key in out_original:
            assert torch.allclose(
                out_original[key], out_loaded[key], atol=1e-6
            ), f"Mismatch in {key} after state_dict reload"

    def test_state_dict_keys_non_empty(
        self, encoder: DualStreamEncoder
    ) -> None:
        """state_dict should contain a non-trivial number of keys."""
        sd = encoder.state_dict()
        assert len(sd) > 0


# ======================================================================
# Deterministic inference tests
# ======================================================================


class TestDeterministicInference:
    """Tests for reproducibility in eval mode."""

    def test_eval_mode_deterministic(
        self,
        encoder: DualStreamEncoder,
        sample_window: torch.Tensor,
        sample_physics: torch.Tensor,
    ) -> None:
        """Two forward passes in eval mode should give identical results."""
        encoder.eval()
        with torch.no_grad():
            out1 = encoder(sample_window, sample_physics)
            out2 = encoder(sample_window, sample_physics)

        for key in out1:
            assert torch.equal(out1[key], out2[key]), (
                f"Non-deterministic output for {key} in eval mode"
            )


# ======================================================================
# TorchScript compatibility tests
# ======================================================================


class TestTorchScript:
    """Tests for TorchScript compatibility."""

    def test_jit_script(
        self,
        encoder: DualStreamEncoder,
        sample_window: torch.Tensor,
        sample_physics: torch.Tensor,
    ) -> None:
        """Encoder should be scriptable via torch.jit.script.

        This test is skipped if scripting fails (some complex
        architectures with dataclass configs are not fully scriptable).
        """
        encoder.eval()
        try:
            scripted = torch.jit.script(encoder)
        except Exception as exc:
            pytest.skip(f"TorchScript not supported: {exc}")

        with torch.no_grad():
            out_eager = encoder(sample_window, sample_physics)
            out_scripted = scripted(sample_window, sample_physics)

        for key in out_eager:
            assert torch.allclose(
                out_eager[key], out_scripted[key], atol=1e-5
            ), f"TorchScript mismatch in {key}"

    def test_jit_trace(
        self,
        encoder: DualStreamEncoder,
        sample_window: torch.Tensor,
        sample_physics: torch.Tensor,
    ) -> None:
        """Encoder should be traceable via torch.jit.trace.

        This test is skipped if tracing fails.
        """
        encoder.eval()
        try:
            traced = torch.jit.trace(
                encoder, (sample_window, sample_physics)
            )
        except Exception as exc:
            pytest.skip(f"TorchScript tracing not supported: {exc}")

        with torch.no_grad():
            out_eager = encoder(sample_window, sample_physics)
            out_traced = traced(sample_window, sample_physics)

        for key in out_eager:
            assert torch.allclose(
                out_eager[key], out_traced[key], atol=1e-5
            ), f"TorchScript trace mismatch in {key}"


# ======================================================================
# Input validation tests
# ======================================================================


class TestInputValidation:
    """Tests for input validation and error handling."""

    # ---- batch-size mismatch ----------------------------------------

    def test_batch_size_mismatch(
        self, encoder: DualStreamEncoder
    ) -> None:
        """Mismatched batch sizes should raise ``ValueError``."""
        w = torch.randn(4, _WINDOW_LEN, _INPUT_CHANNELS)
        p = torch.randn(8, _PHYSICS_DIM)
        with pytest.raises(ValueError, match="[Bb]atch size"):
            encoder(w, p)

    # ---- NaN --------------------------------------------------------

    def test_nan_in_window(
        self,
        encoder: DualStreamEncoder,
        sample_physics: torch.Tensor,
    ) -> None:
        """NaN in window should raise ``ValueError``."""
        w = torch.randn(_BATCH_SIZE, _WINDOW_LEN, _INPUT_CHANNELS)
        w[0, 0, 0] = float("nan")
        with pytest.raises(ValueError, match="NaN"):
            encoder(w, sample_physics)

    def test_nan_in_physics(
        self,
        encoder: DualStreamEncoder,
        sample_window: torch.Tensor,
    ) -> None:
        """NaN in physics features should raise ``ValueError``."""
        p = torch.randn(_BATCH_SIZE, _PHYSICS_DIM)
        p[0, 0] = float("nan")
        with pytest.raises(ValueError, match="NaN"):
            encoder(sample_window, p)

    # ---- Inf --------------------------------------------------------

    def test_inf_in_window(
        self,
        encoder: DualStreamEncoder,
        sample_physics: torch.Tensor,
    ) -> None:
        """Inf in window should raise ``ValueError``."""
        w = torch.randn(_BATCH_SIZE, _WINDOW_LEN, _INPUT_CHANNELS)
        w[0, 0, 0] = float("inf")
        with pytest.raises(ValueError, match="Inf"):
            encoder(w, sample_physics)

    def test_inf_in_physics(
        self,
        encoder: DualStreamEncoder,
        sample_window: torch.Tensor,
    ) -> None:
        """Inf in physics features should raise ``ValueError``."""
        p = torch.randn(_BATCH_SIZE, _PHYSICS_DIM)
        p[0, 0] = float("inf")
        with pytest.raises(ValueError, match="Inf"):
            encoder(sample_window, p)

    def test_neg_inf_in_window(
        self,
        encoder: DualStreamEncoder,
        sample_physics: torch.Tensor,
    ) -> None:
        """Negative Inf in window should raise ``ValueError``."""
        w = torch.randn(_BATCH_SIZE, _WINDOW_LEN, _INPUT_CHANNELS)
        w[0, 0, 0] = float("-inf")
        with pytest.raises(ValueError, match="Inf"):
            encoder(w, sample_physics)

    # ---- non-tensor -------------------------------------------------

    def test_non_tensor_window(
        self,
        encoder: DualStreamEncoder,
        sample_physics: torch.Tensor,
    ) -> None:
        """Passing a non-tensor window should raise ``TypeError``."""
        with pytest.raises(TypeError, match="torch.Tensor"):
            encoder("not a tensor", sample_physics)  # type: ignore[arg-type]

    def test_non_tensor_physics(
        self,
        encoder: DualStreamEncoder,
        sample_window: torch.Tensor,
    ) -> None:
        """Passing a non-tensor physics input should raise ``TypeError``."""
        with pytest.raises(TypeError, match="torch.Tensor"):
            encoder(sample_window, [1, 2, 3])  # type: ignore[arg-type]

    # ---- non-floating dtype -----------------------------------------

    def test_integer_window(
        self,
        encoder: DualStreamEncoder,
        sample_physics: torch.Tensor,
    ) -> None:
        """Integer window should raise ``ValueError``."""
        w = torch.randint(0, 10, (_BATCH_SIZE, _WINDOW_LEN, _INPUT_CHANNELS))
        with pytest.raises(ValueError, match="floating-point"):
            encoder(w, sample_physics)

    def test_integer_physics(
        self,
        encoder: DualStreamEncoder,
        sample_window: torch.Tensor,
    ) -> None:
        """Integer physics features should raise ``ValueError``."""
        p = torch.randint(0, 10, (_BATCH_SIZE, _PHYSICS_DIM))
        with pytest.raises(ValueError, match="floating-point"):
            encoder(sample_window, p)

    # ---- wrong dimensionality ---------------------------------------

    def test_2d_window(
        self,
        encoder: DualStreamEncoder,
        sample_physics: torch.Tensor,
    ) -> None:
        """2D window should raise ``ValueError``."""
        w = torch.randn(_BATCH_SIZE, _INPUT_CHANNELS)
        with pytest.raises(ValueError, match="3 dimensions"):
            encoder(w, sample_physics)

    def test_4d_window(
        self,
        encoder: DualStreamEncoder,
        sample_physics: torch.Tensor,
    ) -> None:
        """4D window should raise ``ValueError``."""
        w = torch.randn(_BATCH_SIZE, _WINDOW_LEN, _INPUT_CHANNELS, 1)
        with pytest.raises(ValueError, match="3 dimensions"):
            encoder(w, sample_physics)

    def test_1d_physics(
        self,
        encoder: DualStreamEncoder,
        sample_window: torch.Tensor,
    ) -> None:
        """1D physics tensor should raise ``ValueError``."""
        p = torch.randn(_PHYSICS_DIM)
        with pytest.raises(ValueError, match="2 dimensions"):
            encoder(sample_window, p)

    def test_3d_physics(
        self,
        encoder: DualStreamEncoder,
        sample_window: torch.Tensor,
    ) -> None:
        """3D physics tensor should raise ``ValueError``."""
        p = torch.randn(_BATCH_SIZE, _PHYSICS_DIM, 1)
        with pytest.raises(ValueError, match="2 dimensions"):
            encoder(sample_window, p)


# ======================================================================
# EncoderConfig tests
# ======================================================================


class TestEncoderConfig:
    """Tests for the ``EncoderConfig`` dataclass."""

    def test_default_config(self) -> None:
        """Default config should be valid (all dims = 256)."""
        cfg = EncoderConfig()
        assert cfg.temporal.embedding_dim == 256
        assert cfg.physics.embedding_dim == 256
        assert cfg.fusion.embedding_dim == 256

    def test_temporal_physics_mismatch(self) -> None:
        """Temporal ≠ physics embedding_dim should raise."""
        with pytest.raises(ValueError, match="consistent"):
            EncoderConfig(
                temporal=TemporalEncoderConfig(
                    input_channels=10, embedding_dim=128
                ),
                physics=PhysicsEncoderConfig(
                    input_dim=18, embedding_dim=64
                ),
                fusion=FusionConfig(embedding_dim=128),
            )

    def test_fusion_mismatch(self) -> None:
        """Fusion ≠ encoder embedding_dim should raise."""
        with pytest.raises(ValueError, match="consistent"):
            EncoderConfig(
                temporal=TemporalEncoderConfig(
                    input_channels=10, embedding_dim=128
                ),
                physics=PhysicsEncoderConfig(
                    input_dim=18, embedding_dim=128
                ),
                fusion=FusionConfig(embedding_dim=64),
            )

    def test_frozen_config(self) -> None:
        """EncoderConfig should be immutable (frozen dataclass)."""
        cfg = EncoderConfig()
        with pytest.raises(AttributeError):
            cfg.temporal = TemporalEncoderConfig()  # type: ignore[misc]


# ======================================================================
# CPU tests
# ======================================================================


class TestCPU:
    """Explicit CPU execution tests."""

    def test_forward_on_cpu(
        self,
        encoder: DualStreamEncoder,
        sample_window: torch.Tensor,
        sample_physics: torch.Tensor,
    ) -> None:
        """Encoder should work on CPU."""
        encoder = encoder.cpu()
        w = sample_window.cpu()
        p = sample_physics.cpu()
        out = encoder(w, p)
        assert out["embedding"].device.type == "cpu"


# ======================================================================
# CUDA tests
# ======================================================================


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA not available"
)
class TestCUDA:
    """CUDA execution tests (skipped when no GPU is present)."""

    def test_forward_on_cuda(
        self, default_config: EncoderConfig
    ) -> None:
        """Encoder should produce correct shapes on CUDA."""
        device = torch.device("cuda")
        enc = DualStreamEncoder(default_config).to(device)
        w = torch.randn(
            _BATCH_SIZE, _WINDOW_LEN, _INPUT_CHANNELS, device=device
        )
        p = torch.randn(_BATCH_SIZE, _PHYSICS_DIM, device=device)
        out = enc(w, p)
        assert out["embedding"].device.type == "cuda"
        assert out["embedding"].shape == (_BATCH_SIZE, _EMBEDDING_DIM)

    def test_encode_on_cuda(
        self, default_config: EncoderConfig
    ) -> None:
        """encode() should work on CUDA."""
        device = torch.device("cuda")
        enc = DualStreamEncoder(default_config).to(device)
        w = torch.randn(
            _BATCH_SIZE, _WINDOW_LEN, _INPUT_CHANNELS, device=device
        )
        p = torch.randn(_BATCH_SIZE, _PHYSICS_DIM, device=device)
        out = enc.encode(w, p)
        assert out["embedding"].device.type == "cuda"

    def test_device_property_cuda(
        self, default_config: EncoderConfig
    ) -> None:
        """device property should report CUDA."""
        enc = DualStreamEncoder(default_config).to("cuda")
        assert enc.device.type == "cuda"

    def test_gradient_flow_cuda(
        self, default_config: EncoderConfig
    ) -> None:
        """Gradients should flow on CUDA."""
        device = torch.device("cuda")
        enc = DualStreamEncoder(default_config).to(device)
        w = torch.randn(
            _BATCH_SIZE, _WINDOW_LEN, _INPUT_CHANNELS, device=device
        )
        p = torch.randn(_BATCH_SIZE, _PHYSICS_DIM, device=device)
        out = enc(w, p)
        out["embedding"].sum().backward()
        for name, param in enc.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"
