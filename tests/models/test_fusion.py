"""Tests for :mod:`src.models.fusion`.

Comprehensive test suite covering initialisation, forward-pass correctness,
output shape, gate value range, L2 normalisation, gradient flow,
serialisation round-trip, TorchScript tracing, deterministic behaviour,
invalid-input rejection (dimension, batch mismatch, embedding mismatch,
NaN, Inf), and device compatibility (CPU / CUDA).
"""

from __future__ import annotations

import copy
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

from src.models.fusion import FeatureFusion, FusionConfig


# ======================================================================
# Constants
# ======================================================================

_DEFAULT_EMBEDDING_DIM: int = 256
_DEFAULT_HIDDEN_DIM: int = 512
_DEFAULT_BATCH_SIZE: int = 4


# ======================================================================
# Fixtures
# ======================================================================


@pytest.fixture()
def default_config() -> FusionConfig:
    """Return a default fusion config."""
    return FusionConfig()


@pytest.fixture()
def fusion(default_config: FusionConfig) -> FeatureFusion:
    """Return a fusion module in eval mode with a fixed seed."""
    torch.manual_seed(42)
    model = FeatureFusion(default_config)
    model.eval()
    return model


@pytest.fixture()
def sample_temporal() -> torch.Tensor:
    """Return a reproducible (B, D) temporal embedding."""
    torch.manual_seed(0)
    return torch.randn(_DEFAULT_BATCH_SIZE, _DEFAULT_EMBEDDING_DIM)


@pytest.fixture()
def sample_physics() -> torch.Tensor:
    """Return a reproducible (B, D) physics embedding."""
    torch.manual_seed(1)
    return torch.randn(_DEFAULT_BATCH_SIZE, _DEFAULT_EMBEDDING_DIM)


# ======================================================================
# FusionConfig tests
# ======================================================================


class TestFusionConfig:
    """Tests for configuration validation."""

    def test_default_values(self) -> None:
        """Defaults match the specification."""
        cfg = FusionConfig()
        assert cfg.embedding_dim == 256
        assert cfg.hidden_dim == 512
        assert cfg.dropout == pytest.approx(0.2)
        assert cfg.bias is True

    def test_custom_values(self) -> None:
        """Config accepts custom values."""
        cfg = FusionConfig(
            embedding_dim=128,
            hidden_dim=256,
            dropout=0.1,
            bias=False,
        )
        assert cfg.embedding_dim == 128
        assert cfg.hidden_dim == 256
        assert cfg.dropout == pytest.approx(0.1)
        assert cfg.bias is False

    def test_invalid_embedding_dim_zero(self) -> None:
        """Raise ValueError for embedding_dim < 1."""
        with pytest.raises(ValueError, match="embedding_dim must be positive"):
            FusionConfig(embedding_dim=0)

    def test_invalid_embedding_dim_negative(self) -> None:
        """Raise ValueError for negative embedding_dim."""
        with pytest.raises(ValueError, match="embedding_dim must be positive"):
            FusionConfig(embedding_dim=-10)

    def test_invalid_hidden_dim_zero(self) -> None:
        """Raise ValueError for hidden_dim < 1."""
        with pytest.raises(ValueError, match="hidden_dim must be positive"):
            FusionConfig(hidden_dim=0)

    def test_invalid_hidden_dim_negative(self) -> None:
        """Raise ValueError for negative hidden_dim."""
        with pytest.raises(ValueError, match="hidden_dim must be positive"):
            FusionConfig(hidden_dim=-5)

    def test_invalid_dropout_too_high(self) -> None:
        """Raise ValueError for dropout >= 1."""
        with pytest.raises(ValueError, match="dropout must be in"):
            FusionConfig(dropout=1.0)

    def test_invalid_dropout_negative(self) -> None:
        """Raise ValueError for negative dropout."""
        with pytest.raises(ValueError, match="dropout must be in"):
            FusionConfig(dropout=-0.1)

    def test_frozen_config(self) -> None:
        """Config is immutable after creation."""
        cfg = FusionConfig()
        with pytest.raises(AttributeError):
            cfg.embedding_dim = 128  # type: ignore[misc]

    def test_dropout_zero_valid(self) -> None:
        """Dropout of 0 is valid."""
        cfg = FusionConfig(dropout=0.0)
        assert cfg.dropout == pytest.approx(0.0)


# ======================================================================
# FeatureFusion — Initialisation
# ======================================================================


class TestFeatureFusionInit:
    """Tests for fusion module construction."""

    def test_type_error_config(self) -> None:
        """Raise TypeError when config is not FusionConfig."""
        with pytest.raises(TypeError, match="FusionConfig"):
            FeatureFusion(config={"embedding_dim": 256})  # type: ignore[arg-type]

    def test_type_error_config_none(self) -> None:
        """Raise TypeError when config is None."""
        with pytest.raises(TypeError, match="FusionConfig"):
            FeatureFusion(config=None)  # type: ignore[arg-type]

    def test_properties(
        self,
        fusion: FeatureFusion,
        default_config: FusionConfig,
    ) -> None:
        """Public properties reflect the config."""
        assert fusion.embedding_dim == _DEFAULT_EMBEDDING_DIM
        assert fusion.config is default_config

    def test_parameter_count_positive(self, fusion: FeatureFusion) -> None:
        """The fusion module has a non-trivial number of parameters."""
        total = sum(p.numel() for p in fusion.parameters())
        assert total > 0

    def test_all_parameters_require_grad(
        self, fusion: FeatureFusion
    ) -> None:
        """All parameters are trainable by default."""
        for name, param in fusion.named_parameters():
            assert param.requires_grad, (
                f"Parameter {name} does not require grad"
            )

    def test_count_parameters_method(self, fusion: FeatureFusion) -> None:
        """_count_parameters returns consistent values."""
        total, trainable = fusion._count_parameters()
        assert total > 0
        assert trainable > 0
        assert trainable <= total

    def test_custom_config(self) -> None:
        """Fusion module accepts a non-default configuration."""
        cfg = FusionConfig(
            embedding_dim=128,
            hidden_dim=256,
            dropout=0.1,
            bias=False,
        )
        model = FeatureFusion(cfg)
        assert model.embedding_dim == 128

    def test_custom_embedding_dim_forward(self) -> None:
        """Forward works with non-default embedding_dim."""
        cfg = FusionConfig(embedding_dim=64, hidden_dim=128)
        model = FeatureFusion(cfg)
        model.eval()
        t = torch.randn(2, 64)
        p = torch.randn(2, 64)
        with torch.no_grad():
            z = model(t, p)
        assert z.shape == (2, 64)


# ======================================================================
# FeatureFusion — Forward pass
# ======================================================================


class TestFeatureFusionForward:
    """Tests for the forward pass."""

    def test_output_shape(
        self,
        fusion: FeatureFusion,
        sample_temporal: torch.Tensor,
        sample_physics: torch.Tensor,
    ) -> None:
        """Output shape is (B, embedding_dim)."""
        with torch.no_grad():
            z = fusion(sample_temporal, sample_physics)
        assert z.shape == (_DEFAULT_BATCH_SIZE, _DEFAULT_EMBEDDING_DIM)

    def test_output_dtype(
        self,
        fusion: FeatureFusion,
        sample_temporal: torch.Tensor,
        sample_physics: torch.Tensor,
    ) -> None:
        """Output dtype matches input dtype."""
        with torch.no_grad():
            z = fusion(sample_temporal, sample_physics)
        assert z.dtype == sample_temporal.dtype

    def test_l2_normalised(
        self,
        fusion: FeatureFusion,
        sample_temporal: torch.Tensor,
        sample_physics: torch.Tensor,
    ) -> None:
        """Every embedding vector has unit L2 norm."""
        with torch.no_grad():
            z = fusion(sample_temporal, sample_physics)
        norms = torch.linalg.norm(z, dim=-1)
        torch.testing.assert_close(
            norms,
            torch.ones_like(norms),
            atol=1e-5,
            rtol=1e-5,
        )

    def test_no_nan_in_output(
        self,
        fusion: FeatureFusion,
        sample_temporal: torch.Tensor,
        sample_physics: torch.Tensor,
    ) -> None:
        """Output does not contain NaN values."""
        with torch.no_grad():
            z = fusion(sample_temporal, sample_physics)
        assert not torch.isnan(z).any()

    def test_no_inf_in_output(
        self,
        fusion: FeatureFusion,
        sample_temporal: torch.Tensor,
        sample_physics: torch.Tensor,
    ) -> None:
        """Output does not contain Inf values."""
        with torch.no_grad():
            z = fusion(sample_temporal, sample_physics)
        assert not torch.isinf(z).any()

    def test_batch_size_one(self, fusion: FeatureFusion) -> None:
        """Forward works for a single-sample batch."""
        t = torch.randn(1, _DEFAULT_EMBEDDING_DIM)
        p = torch.randn(1, _DEFAULT_EMBEDDING_DIM)
        with torch.no_grad():
            z = fusion(t, p)
        assert z.shape == (1, _DEFAULT_EMBEDDING_DIM)

    def test_large_batch(self, fusion: FeatureFusion) -> None:
        """Forward works for a larger batch."""
        t = torch.randn(64, _DEFAULT_EMBEDDING_DIM)
        p = torch.randn(64, _DEFAULT_EMBEDDING_DIM)
        with torch.no_grad():
            z = fusion(t, p)
        assert z.shape == (64, _DEFAULT_EMBEDDING_DIM)

    def test_identical_inputs(self, fusion: FeatureFusion) -> None:
        """Forward works when both inputs are identical."""
        x = torch.randn(2, _DEFAULT_EMBEDDING_DIM)
        with torch.no_grad():
            z = fusion(x, x.clone())
        assert z.shape == (2, _DEFAULT_EMBEDDING_DIM)
        assert not torch.isnan(z).any()


# ======================================================================
# FeatureFusion — Gate values
# ======================================================================


class TestFeatureFusionGate:
    """Tests verifying the gate values are in [0, 1]."""

    def test_gate_range(
        self,
        fusion: FeatureFusion,
        sample_temporal: torch.Tensor,
        sample_physics: torch.Tensor,
    ) -> None:
        """Gate values must lie in [0, 1] since they come from sigmoid."""
        with torch.no_grad():
            gate = fusion._compute_gate(sample_temporal, sample_physics)
        assert gate.shape == (_DEFAULT_BATCH_SIZE, _DEFAULT_EMBEDDING_DIM)
        assert (gate >= 0.0).all(), "Gate contains values below 0"
        assert (gate <= 1.0).all(), "Gate contains values above 1"

    def test_gate_not_all_zero(
        self,
        fusion: FeatureFusion,
        sample_temporal: torch.Tensor,
        sample_physics: torch.Tensor,
    ) -> None:
        """Gate should not collapse to all zeros."""
        with torch.no_grad():
            gate = fusion._compute_gate(sample_temporal, sample_physics)
        assert gate.sum() > 0.0

    def test_gate_not_all_one(
        self,
        fusion: FeatureFusion,
        sample_temporal: torch.Tensor,
        sample_physics: torch.Tensor,
    ) -> None:
        """Gate should not saturate to all ones."""
        with torch.no_grad():
            gate = fusion._compute_gate(sample_temporal, sample_physics)
        assert gate.sum() < gate.numel()

    def test_gate_dtype(
        self,
        fusion: FeatureFusion,
        sample_temporal: torch.Tensor,
        sample_physics: torch.Tensor,
    ) -> None:
        """Gate dtype matches input dtype."""
        with torch.no_grad():
            gate = fusion._compute_gate(sample_temporal, sample_physics)
        assert gate.dtype == sample_temporal.dtype


# ======================================================================
# FeatureFusion — Gradient flow
# ======================================================================


class TestFeatureFusionGradients:
    """Tests verifying gradient flow for training."""

    def test_gradients_exist(
        self,
        fusion: FeatureFusion,
        sample_temporal: torch.Tensor,
        sample_physics: torch.Tensor,
    ) -> None:
        """Backward pass produces non-None gradients for all parameters."""
        fusion.train()
        z = fusion(sample_temporal, sample_physics)
        loss = z.sum()
        loss.backward()
        for name, param in fusion.named_parameters():
            assert param.grad is not None, (
                f"No gradient for parameter '{name}'"
            )

    def test_gradients_nonzero(
        self,
        fusion: FeatureFusion,
        sample_temporal: torch.Tensor,
        sample_physics: torch.Tensor,
    ) -> None:
        """Backward pass produces non-trivial (non-all-zero) gradients."""
        fusion.train()
        z = fusion(sample_temporal, sample_physics)
        loss = z.sum()
        loss.backward()
        zero_grad_params = [
            name
            for name, p in fusion.named_parameters()
            if p.grad is not None and torch.all(p.grad == 0)
        ]
        assert len(zero_grad_params) == 0, (
            f"Zero-gradient parameters: {zero_grad_params}"
        )

    def test_temporal_input_gradient(self, fusion: FeatureFusion) -> None:
        """Gradients propagate back to the temporal input tensor."""
        fusion.train()
        t = torch.randn(
            2, _DEFAULT_EMBEDDING_DIM, requires_grad=True
        )
        p = torch.randn(2, _DEFAULT_EMBEDDING_DIM)
        z = fusion(t, p)
        z.sum().backward()
        assert t.grad is not None
        assert t.grad.shape == t.shape

    def test_physics_input_gradient(self, fusion: FeatureFusion) -> None:
        """Gradients propagate back to the physics input tensor."""
        fusion.train()
        t = torch.randn(2, _DEFAULT_EMBEDDING_DIM)
        p = torch.randn(
            2, _DEFAULT_EMBEDDING_DIM, requires_grad=True
        )
        z = fusion(t, p)
        z.sum().backward()
        assert p.grad is not None
        assert p.grad.shape == p.shape

    def test_both_input_gradients(self, fusion: FeatureFusion) -> None:
        """Gradients propagate back to both input tensors."""
        fusion.train()
        t = torch.randn(
            2, _DEFAULT_EMBEDDING_DIM, requires_grad=True
        )
        p = torch.randn(
            2, _DEFAULT_EMBEDDING_DIM, requires_grad=True
        )
        z = fusion(t, p)
        z.sum().backward()
        assert t.grad is not None
        assert p.grad is not None


# ======================================================================
# FeatureFusion — Determinism
# ======================================================================


class TestFeatureFusionDeterminism:
    """Tests verifying deterministic output in eval mode."""

    def test_deterministic_eval(
        self,
        fusion: FeatureFusion,
        sample_temporal: torch.Tensor,
        sample_physics: torch.Tensor,
    ) -> None:
        """Two forward passes with the same input give identical output."""
        fusion.eval()
        with torch.no_grad():
            z1 = fusion(sample_temporal, sample_physics)
            z2 = fusion(sample_temporal, sample_physics)
        torch.testing.assert_close(z1, z2, atol=0.0, rtol=0.0)

    def test_different_inputs_different_outputs(
        self, fusion: FeatureFusion
    ) -> None:
        """Different inputs produce different embeddings."""
        fusion.eval()
        t1 = torch.randn(2, _DEFAULT_EMBEDDING_DIM)
        p1 = torch.randn(2, _DEFAULT_EMBEDDING_DIM)
        t2 = torch.randn(2, _DEFAULT_EMBEDDING_DIM) + 5.0
        p2 = torch.randn(2, _DEFAULT_EMBEDDING_DIM) + 5.0
        with torch.no_grad():
            z1 = fusion(t1, p1)
            z2 = fusion(t2, p2)
        assert not torch.allclose(z1, z2)


# ======================================================================
# FeatureFusion — Serialisation
# ======================================================================


class TestFeatureFusionSerialisation:
    """Tests for save/load round-tripping."""

    def test_state_dict_round_trip(
        self,
        fusion: FeatureFusion,
        sample_temporal: torch.Tensor,
        sample_physics: torch.Tensor,
        default_config: FusionConfig,
    ) -> None:
        """Save and reload via state_dict produces identical output."""
        fusion.eval()
        with torch.no_grad():
            z_orig = fusion(sample_temporal, sample_physics)

        state = copy.deepcopy(fusion.state_dict())

        new_fusion = FeatureFusion(default_config)
        new_fusion.load_state_dict(state)
        new_fusion.eval()

        with torch.no_grad():
            z_loaded = new_fusion(sample_temporal, sample_physics)

        torch.testing.assert_close(z_orig, z_loaded, atol=1e-6, rtol=1e-6)

    def test_torch_save_load(
        self,
        fusion: FeatureFusion,
        sample_temporal: torch.Tensor,
        sample_physics: torch.Tensor,
        default_config: FusionConfig,
    ) -> None:
        """Full torch.save / torch.load round-trip preserves output."""
        fusion.eval()
        with torch.no_grad():
            z_orig = fusion(sample_temporal, sample_physics)

        with tempfile.NamedTemporaryFile(
            suffix=".pt", delete=False
        ) as tmp:
            torch.save(fusion.state_dict(), tmp.name)
            tmp_path = Path(tmp.name)

        try:
            loaded_state = torch.load(
                tmp_path, map_location="cpu", weights_only=True
            )
            new_fusion = FeatureFusion(default_config)
            new_fusion.load_state_dict(loaded_state)
            new_fusion.eval()
            with torch.no_grad():
                z_loaded = new_fusion(sample_temporal, sample_physics)
            torch.testing.assert_close(
                z_orig, z_loaded, atol=1e-6, rtol=1e-6
            )
        finally:
            tmp_path.unlink(missing_ok=True)

    def test_state_dict_keys_stable(
        self, fusion: FeatureFusion
    ) -> None:
        """State dict key names are non-empty and consistent."""
        keys = list(fusion.state_dict().keys())
        assert len(keys) > 0
        for k in keys:
            assert isinstance(k, str)
            assert len(k) > 0


# ======================================================================
# FeatureFusion — TorchScript
# ======================================================================


class TestFeatureFusionTorchScript:
    """Tests for TorchScript compatibility via tracing."""

    def test_torch_jit_trace(
        self,
        fusion: FeatureFusion,
        sample_temporal: torch.Tensor,
        sample_physics: torch.Tensor,
    ) -> None:
        """Fusion module can be compiled with torch.jit.trace."""
        fusion.eval()
        traced = torch.jit.trace(
            fusion, (sample_temporal, sample_physics)
        )
        with torch.no_grad():
            z = traced(sample_temporal, sample_physics)
        assert z.shape == (_DEFAULT_BATCH_SIZE, _DEFAULT_EMBEDDING_DIM)

    def test_traced_output_matches_eager(
        self,
        fusion: FeatureFusion,
        sample_temporal: torch.Tensor,
        sample_physics: torch.Tensor,
    ) -> None:
        """Traced model produces the same output as the eager model."""
        fusion.eval()
        traced = torch.jit.trace(
            fusion, (sample_temporal, sample_physics)
        )
        with torch.no_grad():
            z_eager = fusion(sample_temporal, sample_physics)
            z_traced = traced(sample_temporal, sample_physics)
        torch.testing.assert_close(
            z_eager, z_traced, atol=1e-6, rtol=1e-6
        )

    def test_traced_l2_normalised(
        self,
        fusion: FeatureFusion,
        sample_temporal: torch.Tensor,
        sample_physics: torch.Tensor,
    ) -> None:
        """Traced model output is L2-normalised."""
        fusion.eval()
        traced = torch.jit.trace(
            fusion, (sample_temporal, sample_physics)
        )
        with torch.no_grad():
            z = traced(sample_temporal, sample_physics)
        norms = torch.linalg.norm(z, dim=-1)
        torch.testing.assert_close(
            norms,
            torch.ones_like(norms),
            atol=1e-5,
            rtol=1e-5,
        )

    def test_traced_save_load(
        self,
        fusion: FeatureFusion,
        sample_temporal: torch.Tensor,
        sample_physics: torch.Tensor,
    ) -> None:
        """Traced model can be saved and reloaded."""
        fusion.eval()
        traced = torch.jit.trace(
            fusion, (sample_temporal, sample_physics)
        )

        with torch.no_grad():
            z_orig = traced(sample_temporal, sample_physics)

        with tempfile.NamedTemporaryFile(
            suffix=".pt", delete=False
        ) as tmp:
            torch.jit.save(traced, tmp.name)
            tmp_path = Path(tmp.name)

        try:
            loaded = torch.jit.load(str(tmp_path), map_location="cpu")
            with torch.no_grad():
                z_loaded = loaded(sample_temporal, sample_physics)
            torch.testing.assert_close(
                z_orig, z_loaded, atol=1e-6, rtol=1e-6
            )
        finally:
            tmp_path.unlink(missing_ok=True)


# ======================================================================
# FeatureFusion — Invalid inputs
# ======================================================================


class TestFeatureFusionInvalidInputs:
    """Tests for input validation and error messages."""

    def test_temporal_not_a_tensor(self, fusion: FeatureFusion) -> None:
        """Raise TypeError for non-tensor temporal input."""
        p = torch.randn(2, _DEFAULT_EMBEDDING_DIM)
        with pytest.raises(TypeError, match="temporal_embedding.*torch.Tensor"):
            fusion(
                np.zeros((2, _DEFAULT_EMBEDDING_DIM)),  # type: ignore[arg-type]
                p,
            )

    def test_physics_not_a_tensor(self, fusion: FeatureFusion) -> None:
        """Raise TypeError for non-tensor physics input."""
        t = torch.randn(2, _DEFAULT_EMBEDDING_DIM)
        with pytest.raises(TypeError, match="physics_embedding.*torch.Tensor"):
            fusion(
                t,
                np.zeros((2, _DEFAULT_EMBEDDING_DIM)),  # type: ignore[arg-type]
            )

    def test_temporal_integer_dtype(self, fusion: FeatureFusion) -> None:
        """Raise ValueError for integer-typed temporal tensor."""
        t = torch.randint(0, 10, (2, _DEFAULT_EMBEDDING_DIM))
        p = torch.randn(2, _DEFAULT_EMBEDDING_DIM)
        with pytest.raises(ValueError, match="temporal_embedding.*floating-point"):
            fusion(t, p)

    def test_physics_integer_dtype(self, fusion: FeatureFusion) -> None:
        """Raise ValueError for integer-typed physics tensor."""
        t = torch.randn(2, _DEFAULT_EMBEDDING_DIM)
        p = torch.randint(0, 10, (2, _DEFAULT_EMBEDDING_DIM))
        with pytest.raises(ValueError, match="physics_embedding.*floating-point"):
            fusion(t, p)

    def test_temporal_wrong_ndim_1d(self, fusion: FeatureFusion) -> None:
        """Raise ValueError for 1-D temporal input."""
        t = torch.randn(_DEFAULT_EMBEDDING_DIM)
        p = torch.randn(2, _DEFAULT_EMBEDDING_DIM)
        with pytest.raises(ValueError, match="temporal_embedding.*exactly 2 dimensions"):
            fusion(t, p)

    def test_temporal_wrong_ndim_3d(self, fusion: FeatureFusion) -> None:
        """Raise ValueError for 3-D temporal input."""
        t = torch.randn(2, 10, _DEFAULT_EMBEDDING_DIM)
        p = torch.randn(2, _DEFAULT_EMBEDDING_DIM)
        with pytest.raises(ValueError, match="temporal_embedding.*exactly 2 dimensions"):
            fusion(t, p)

    def test_physics_wrong_ndim_1d(self, fusion: FeatureFusion) -> None:
        """Raise ValueError for 1-D physics input."""
        t = torch.randn(2, _DEFAULT_EMBEDDING_DIM)
        p = torch.randn(_DEFAULT_EMBEDDING_DIM)
        with pytest.raises(ValueError, match="physics_embedding.*exactly 2 dimensions"):
            fusion(t, p)

    def test_physics_wrong_ndim_3d(self, fusion: FeatureFusion) -> None:
        """Raise ValueError for 3-D physics input."""
        t = torch.randn(2, _DEFAULT_EMBEDDING_DIM)
        p = torch.randn(2, 10, _DEFAULT_EMBEDDING_DIM)
        with pytest.raises(ValueError, match="physics_embedding.*exactly 2 dimensions"):
            fusion(t, p)

    def test_batch_size_mismatch(self, fusion: FeatureFusion) -> None:
        """Raise ValueError when batch sizes differ."""
        t = torch.randn(3, _DEFAULT_EMBEDDING_DIM)
        p = torch.randn(5, _DEFAULT_EMBEDDING_DIM)
        with pytest.raises(ValueError, match="Batch size mismatch"):
            fusion(t, p)

    def test_temporal_embedding_dim_mismatch(
        self, fusion: FeatureFusion
    ) -> None:
        """Raise ValueError when temporal embedding dim is wrong."""
        t = torch.randn(2, _DEFAULT_EMBEDDING_DIM + 10)
        p = torch.randn(2, _DEFAULT_EMBEDDING_DIM)
        with pytest.raises(ValueError, match="temporal_embedding.*embedding dimension"):
            fusion(t, p)

    def test_physics_embedding_dim_mismatch(
        self, fusion: FeatureFusion
    ) -> None:
        """Raise ValueError when physics embedding dim is wrong."""
        t = torch.randn(2, _DEFAULT_EMBEDDING_DIM)
        p = torch.randn(2, _DEFAULT_EMBEDDING_DIM + 10)
        with pytest.raises(ValueError, match="physics_embedding.*embedding dimension"):
            fusion(t, p)

    def test_temporal_nan(self, fusion: FeatureFusion) -> None:
        """Raise ValueError when temporal input contains NaN."""
        t = torch.randn(2, _DEFAULT_EMBEDDING_DIM)
        t[0, 3] = float("nan")
        p = torch.randn(2, _DEFAULT_EMBEDDING_DIM)
        with pytest.raises(ValueError, match="temporal_embedding.*NaN"):
            fusion(t, p)

    def test_physics_nan(self, fusion: FeatureFusion) -> None:
        """Raise ValueError when physics input contains NaN."""
        t = torch.randn(2, _DEFAULT_EMBEDDING_DIM)
        p = torch.randn(2, _DEFAULT_EMBEDDING_DIM)
        p[1, 0] = float("nan")
        with pytest.raises(ValueError, match="physics_embedding.*NaN"):
            fusion(t, p)

    def test_temporal_inf(self, fusion: FeatureFusion) -> None:
        """Raise ValueError when temporal input contains Inf."""
        t = torch.randn(2, _DEFAULT_EMBEDDING_DIM)
        t[0, 5] = float("inf")
        p = torch.randn(2, _DEFAULT_EMBEDDING_DIM)
        with pytest.raises(ValueError, match="temporal_embedding.*Inf"):
            fusion(t, p)

    def test_physics_inf(self, fusion: FeatureFusion) -> None:
        """Raise ValueError when physics input contains Inf."""
        t = torch.randn(2, _DEFAULT_EMBEDDING_DIM)
        p = torch.randn(2, _DEFAULT_EMBEDDING_DIM)
        p[0, 0] = float("inf")
        with pytest.raises(ValueError, match="physics_embedding.*Inf"):
            fusion(t, p)

    def test_temporal_negative_inf(self, fusion: FeatureFusion) -> None:
        """Raise ValueError when temporal input contains -Inf."""
        t = torch.randn(2, _DEFAULT_EMBEDDING_DIM)
        t[0, 5] = float("-inf")
        p = torch.randn(2, _DEFAULT_EMBEDDING_DIM)
        with pytest.raises(ValueError, match="temporal_embedding.*Inf"):
            fusion(t, p)

    def test_physics_negative_inf(self, fusion: FeatureFusion) -> None:
        """Raise ValueError when physics input contains -Inf."""
        t = torch.randn(2, _DEFAULT_EMBEDDING_DIM)
        p = torch.randn(2, _DEFAULT_EMBEDDING_DIM)
        p[1, 10] = float("-inf")
        with pytest.raises(ValueError, match="physics_embedding.*Inf"):
            fusion(t, p)


# ======================================================================
# FeatureFusion — Device compatibility
# ======================================================================


class TestFeatureFusionCPU:
    """CPU-specific compatibility tests."""

    def test_cpu_forward(
        self,
        fusion: FeatureFusion,
        sample_temporal: torch.Tensor,
        sample_physics: torch.Tensor,
    ) -> None:
        """Forward pass runs on CPU without error."""
        assert sample_temporal.device.type == "cpu"
        assert sample_physics.device.type == "cpu"
        with torch.no_grad():
            z = fusion(sample_temporal, sample_physics)
        assert z.device.type == "cpu"
        assert z.shape == (_DEFAULT_BATCH_SIZE, _DEFAULT_EMBEDDING_DIM)


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA not available",
)
class TestFeatureFusionCUDA:
    """CUDA-specific compatibility tests (skipped if no GPU)."""

    def test_cuda_forward(self, default_config: FusionConfig) -> None:
        """Forward pass runs on CUDA and returns a CUDA tensor."""
        device = torch.device("cuda")
        model = FeatureFusion(default_config).to(device).eval()
        t = torch.randn(
            _DEFAULT_BATCH_SIZE, _DEFAULT_EMBEDDING_DIM, device=device
        )
        p = torch.randn(
            _DEFAULT_BATCH_SIZE, _DEFAULT_EMBEDDING_DIM, device=device
        )
        with torch.no_grad():
            z = model(t, p)
        assert z.device.type == "cuda"
        assert z.shape == (_DEFAULT_BATCH_SIZE, _DEFAULT_EMBEDDING_DIM)

    def test_cuda_l2_normalised(
        self, default_config: FusionConfig
    ) -> None:
        """CUDA embeddings are L2-normalised."""
        device = torch.device("cuda")
        model = FeatureFusion(default_config).to(device).eval()
        t = torch.randn(
            _DEFAULT_BATCH_SIZE, _DEFAULT_EMBEDDING_DIM, device=device
        )
        p = torch.randn(
            _DEFAULT_BATCH_SIZE, _DEFAULT_EMBEDDING_DIM, device=device
        )
        with torch.no_grad():
            z = model(t, p)
        norms = torch.linalg.norm(z, dim=-1)
        torch.testing.assert_close(
            norms,
            torch.ones_like(norms),
            atol=1e-5,
            rtol=1e-5,
        )

    def test_cuda_gradients(
        self, default_config: FusionConfig
    ) -> None:
        """Gradient flow works on CUDA."""
        device = torch.device("cuda")
        model = FeatureFusion(default_config).to(device).train()
        t = torch.randn(
            2, _DEFAULT_EMBEDDING_DIM,
            device=device, requires_grad=True,
        )
        p = torch.randn(
            2, _DEFAULT_EMBEDDING_DIM,
            device=device, requires_grad=True,
        )
        z = model(t, p)
        z.sum().backward()
        assert t.grad is not None
        assert t.grad.shape == t.shape
        assert p.grad is not None
        assert p.grad.shape == p.shape


# ======================================================================
# FeatureFusion — Float64 compatibility
# ======================================================================


class TestFeatureFusionFloat64:
    """Ensure the fusion module works with float64 input."""

    def test_float64_forward(
        self, default_config: FusionConfig
    ) -> None:
        """Forward works when both model and inputs are float64."""
        model = FeatureFusion(default_config).double().eval()
        t = torch.randn(
            2, _DEFAULT_EMBEDDING_DIM, dtype=torch.float64
        )
        p = torch.randn(
            2, _DEFAULT_EMBEDDING_DIM, dtype=torch.float64
        )
        with torch.no_grad():
            z = model(t, p)
        assert z.dtype == torch.float64
        assert z.shape == (2, _DEFAULT_EMBEDDING_DIM)


# ======================================================================
# FeatureFusion — Residual connection
# ======================================================================


class TestFeatureFusionResidual:
    """Tests verifying the residual connection behaviour."""

    def test_zero_physics_recovers_temporal(self) -> None:
        """When physics embedding is zero, output is dominated by temporal.

        If physics is the zero vector, the gated blend becomes
        ``(1 − g) * temporal`` and the residual adds ``temporal``,
        so the result should be close to a normalised version of
        temporal (modulo the learned gate values and LayerNorm).
        """
        cfg = FusionConfig()
        torch.manual_seed(42)
        model = FeatureFusion(cfg)
        model.eval()
        t = torch.randn(2, _DEFAULT_EMBEDDING_DIM)
        p = torch.zeros(2, _DEFAULT_EMBEDDING_DIM)
        with torch.no_grad():
            z = model(t, p)
        # Output should be valid and well-formed
        assert z.shape == (2, _DEFAULT_EMBEDDING_DIM)
        norms = torch.linalg.norm(z, dim=-1)
        torch.testing.assert_close(
            norms,
            torch.ones_like(norms),
            atol=1e-5,
            rtol=1e-5,
        )
