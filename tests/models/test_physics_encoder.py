"""Tests for :mod:`src.models.physics_encoder`.

Comprehensive test suite covering initialisation, forward-pass correctness,
output shape, L2 normalisation, gradient flow, serialisation round-trip,
TorchScript scripting, deterministic behaviour, invalid-input rejection
(dimension, feature count, NaN, Inf), and device compatibility.
"""

from __future__ import annotations

import copy
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

from src.models.physics_encoder import (
    PhysicsEncoder,
    PhysicsEncoderConfig,
    _ACTIVATION_REGISTRY,
)


# ======================================================================
# Fixtures
# ======================================================================

_DEFAULT_INPUT_DIM: int = 18
_DEFAULT_BATCH_SIZE: int = 4
_DEFAULT_EMBEDDING_DIM: int = 256


@pytest.fixture()
def default_config() -> PhysicsEncoderConfig:
    """Return a default encoder config for 18 physics features."""
    return PhysicsEncoderConfig(input_dim=_DEFAULT_INPUT_DIM)


@pytest.fixture()
def encoder(default_config: PhysicsEncoderConfig) -> PhysicsEncoder:
    """Return an encoder in eval mode with a fixed seed."""
    torch.manual_seed(42)
    model = PhysicsEncoder(default_config)
    model.eval()
    return model


@pytest.fixture()
def sample_input() -> torch.Tensor:
    """Return a reproducible (B, F) input tensor."""
    torch.manual_seed(0)
    return torch.randn(_DEFAULT_BATCH_SIZE, _DEFAULT_INPUT_DIM)


# ======================================================================
# PhysicsEncoderConfig tests
# ======================================================================


class TestPhysicsEncoderConfig:
    """Tests for configuration validation."""

    def test_default_values(self) -> None:
        """Defaults match the specification."""
        cfg = PhysicsEncoderConfig(input_dim=18)
        assert cfg.input_dim == 18
        assert cfg.hidden_dims == (512, 256)
        assert cfg.embedding_dim == 256
        assert cfg.dropout == pytest.approx(0.2)
        assert cfg.bias is True
        assert cfg.activation == "gelu"

    def test_invalid_input_dim(self) -> None:
        """Raise ValueError for input_dim < 1."""
        with pytest.raises(ValueError, match="input_dim must be positive"):
            PhysicsEncoderConfig(input_dim=0)

    def test_negative_input_dim(self) -> None:
        """Raise ValueError for negative input_dim."""
        with pytest.raises(ValueError, match="input_dim must be positive"):
            PhysicsEncoderConfig(input_dim=-5)

    def test_invalid_embedding_dim(self) -> None:
        """Raise ValueError for embedding_dim < 1."""
        with pytest.raises(ValueError, match="embedding_dim must be positive"):
            PhysicsEncoderConfig(input_dim=10, embedding_dim=0)

    def test_empty_hidden_dims(self) -> None:
        """Raise ValueError for empty hidden_dims."""
        with pytest.raises(ValueError, match="hidden_dims must not be empty"):
            PhysicsEncoderConfig(input_dim=10, hidden_dims=())

    def test_negative_hidden_dim(self) -> None:
        """Raise ValueError for negative hidden dimension."""
        with pytest.raises(ValueError, match=r"hidden_dims\[0\] must be positive"):
            PhysicsEncoderConfig(input_dim=10, hidden_dims=(-1, 256))

    def test_zero_hidden_dim(self) -> None:
        """Raise ValueError for zero hidden dimension."""
        with pytest.raises(ValueError, match=r"hidden_dims\[1\] must be positive"):
            PhysicsEncoderConfig(input_dim=10, hidden_dims=(512, 0))

    def test_invalid_dropout_too_high(self) -> None:
        """Raise ValueError for dropout >= 1."""
        with pytest.raises(ValueError, match="dropout must be in"):
            PhysicsEncoderConfig(input_dim=10, dropout=1.0)

    def test_invalid_dropout_negative(self) -> None:
        """Raise ValueError for negative dropout."""
        with pytest.raises(ValueError, match="dropout must be in"):
            PhysicsEncoderConfig(input_dim=10, dropout=-0.1)

    def test_invalid_activation(self) -> None:
        """Raise ValueError for unsupported activation name."""
        with pytest.raises(ValueError, match="activation must be one of"):
            PhysicsEncoderConfig(input_dim=10, activation="leaky_relu")

    def test_frozen_config(self) -> None:
        """Config is immutable after creation."""
        cfg = PhysicsEncoderConfig(input_dim=10)
        with pytest.raises(AttributeError):
            cfg.input_dim = 99  # type: ignore[misc]

    def test_custom_values(self) -> None:
        """Config accepts custom values."""
        cfg = PhysicsEncoderConfig(
            input_dim=32,
            hidden_dims=(128, 64),
            embedding_dim=128,
            dropout=0.1,
            bias=False,
            activation="relu",
        )
        assert cfg.input_dim == 32
        assert cfg.hidden_dims == (128, 64)
        assert cfg.embedding_dim == 128
        assert cfg.dropout == pytest.approx(0.1)
        assert cfg.bias is False
        assert cfg.activation == "relu"


# ======================================================================
# PhysicsEncoder — Initialisation
# ======================================================================


class TestPhysicsEncoderInit:
    """Tests for encoder construction."""

    def test_type_error_config(self) -> None:
        """Raise TypeError when config is not PhysicsEncoderConfig."""
        with pytest.raises(TypeError, match="PhysicsEncoderConfig"):
            PhysicsEncoder(config={"input_dim": 10})  # type: ignore[arg-type]

    def test_properties(
        self,
        encoder: PhysicsEncoder,
        default_config: PhysicsEncoderConfig,
    ) -> None:
        """Public properties reflect the config."""
        assert encoder.input_dim == _DEFAULT_INPUT_DIM
        assert encoder.embedding_dim == _DEFAULT_EMBEDDING_DIM
        assert encoder.config is default_config

    def test_parameter_count_positive(self, encoder: PhysicsEncoder) -> None:
        """The encoder has a non-trivial number of parameters."""
        total = sum(p.numel() for p in encoder.parameters())
        assert total > 0

    def test_all_parameters_require_grad(
        self, encoder: PhysicsEncoder
    ) -> None:
        """All parameters are trainable by default."""
        for name, param in encoder.named_parameters():
            assert param.requires_grad, f"Parameter {name} does not require grad"

    def test_custom_config(self) -> None:
        """Encoder accepts a non-default configuration."""
        cfg = PhysicsEncoderConfig(
            input_dim=32,
            hidden_dims=(128, 64),
            embedding_dim=128,
            dropout=0.1,
            bias=False,
            activation="relu",
        )
        model = PhysicsEncoder(cfg)
        assert model.embedding_dim == 128
        assert model.input_dim == 32

    def test_single_hidden_layer(self) -> None:
        """Encoder works with a single hidden layer."""
        cfg = PhysicsEncoderConfig(
            input_dim=10,
            hidden_dims=(64,),
            embedding_dim=32,
        )
        model = PhysicsEncoder(cfg)
        x = torch.randn(2, 10)
        model.eval()
        with torch.no_grad():
            z = model(x)
        assert z.shape == (2, 32)

    def test_three_hidden_layers(self) -> None:
        """Encoder works with three hidden layers."""
        cfg = PhysicsEncoderConfig(
            input_dim=10,
            hidden_dims=(128, 64, 32),
            embedding_dim=16,
        )
        model = PhysicsEncoder(cfg)
        x = torch.randn(2, 10)
        model.eval()
        with torch.no_grad():
            z = model(x)
        assert z.shape == (2, 16)

    @pytest.mark.parametrize("activation", list(_ACTIVATION_REGISTRY.keys()))
    def test_all_activations(self, activation: str) -> None:
        """Encoder initialises with each supported activation."""
        cfg = PhysicsEncoderConfig(
            input_dim=10,
            activation=activation,
        )
        model = PhysicsEncoder(cfg)
        x = torch.randn(2, 10)
        model.eval()
        with torch.no_grad():
            z = model(x)
        assert z.shape == (2, _DEFAULT_EMBEDDING_DIM)


# ======================================================================
# PhysicsEncoder — Forward pass
# ======================================================================


class TestPhysicsEncoderForward:
    """Tests for the forward pass."""

    def test_output_shape(
        self,
        encoder: PhysicsEncoder,
        sample_input: torch.Tensor,
    ) -> None:
        """Output shape is (B, embedding_dim)."""
        with torch.no_grad():
            z = encoder(sample_input)
        assert z.shape == (_DEFAULT_BATCH_SIZE, _DEFAULT_EMBEDDING_DIM)

    def test_output_dtype(
        self,
        encoder: PhysicsEncoder,
        sample_input: torch.Tensor,
    ) -> None:
        """Output dtype matches input dtype."""
        with torch.no_grad():
            z = encoder(sample_input)
        assert z.dtype == sample_input.dtype

    def test_l2_normalised(
        self,
        encoder: PhysicsEncoder,
        sample_input: torch.Tensor,
    ) -> None:
        """Every embedding vector has unit L2 norm."""
        with torch.no_grad():
            z = encoder(sample_input)
        norms = torch.linalg.norm(z, dim=-1)
        torch.testing.assert_close(
            norms,
            torch.ones_like(norms),
            atol=1e-5,
            rtol=1e-5,
        )

    def test_no_nan_in_output(
        self,
        encoder: PhysicsEncoder,
        sample_input: torch.Tensor,
    ) -> None:
        """Output does not contain NaN values."""
        with torch.no_grad():
            z = encoder(sample_input)
        assert not torch.isnan(z).any()

    def test_no_inf_in_output(
        self,
        encoder: PhysicsEncoder,
        sample_input: torch.Tensor,
    ) -> None:
        """Output does not contain Inf values."""
        with torch.no_grad():
            z = encoder(sample_input)
        assert not torch.isinf(z).any()

    def test_batch_size_one(self, encoder: PhysicsEncoder) -> None:
        """Forward works for a single-sample batch."""
        x = torch.randn(1, _DEFAULT_INPUT_DIM)
        with torch.no_grad():
            z = encoder(x)
        assert z.shape == (1, _DEFAULT_EMBEDDING_DIM)

    def test_large_batch(self, encoder: PhysicsEncoder) -> None:
        """Forward works for a larger batch."""
        x = torch.randn(64, _DEFAULT_INPUT_DIM)
        with torch.no_grad():
            z = encoder(x)
        assert z.shape == (64, _DEFAULT_EMBEDDING_DIM)

    @pytest.mark.parametrize("input_dim", [1, 5, 18, 50, 128])
    def test_variable_input_dims(self, input_dim: int) -> None:
        """Encoder supports arbitrary input feature dimensions."""
        cfg = PhysicsEncoderConfig(input_dim=input_dim)
        model = PhysicsEncoder(cfg)
        model.eval()
        x = torch.randn(2, input_dim)
        with torch.no_grad():
            z = model(x)
        assert z.shape == (2, _DEFAULT_EMBEDDING_DIM)
        norms = torch.linalg.norm(z, dim=-1)
        torch.testing.assert_close(
            norms,
            torch.ones_like(norms),
            atol=1e-5,
            rtol=1e-5,
        )


# ======================================================================
# PhysicsEncoder — Gradient flow
# ======================================================================


class TestPhysicsEncoderGradients:
    """Tests verifying gradient flow for training."""

    def test_gradients_exist(
        self,
        encoder: PhysicsEncoder,
        sample_input: torch.Tensor,
    ) -> None:
        """Backward pass produces non-None gradients for all parameters."""
        encoder.train()
        z = encoder(sample_input)
        loss = z.sum()
        loss.backward()
        for name, param in encoder.named_parameters():
            assert param.grad is not None, (
                f"No gradient for parameter '{name}'"
            )

    def test_gradients_nonzero(
        self,
        encoder: PhysicsEncoder,
        sample_input: torch.Tensor,
    ) -> None:
        """Backward pass produces non-trivial (non-all-zero) gradients."""
        encoder.train()
        z = encoder(sample_input)
        loss = z.sum()
        loss.backward()
        zero_grad_params = [
            name
            for name, p in encoder.named_parameters()
            if p.grad is not None and torch.all(p.grad == 0)
        ]
        assert len(zero_grad_params) == 0, (
            f"Zero-gradient parameters: {zero_grad_params}"
        )

    def test_input_gradient(self, encoder: PhysicsEncoder) -> None:
        """Gradients propagate back to the input tensor."""
        encoder.train()
        x = torch.randn(
            2, _DEFAULT_INPUT_DIM,
            requires_grad=True,
        )
        z = encoder(x)
        z.sum().backward()
        assert x.grad is not None
        assert x.grad.shape == x.shape


# ======================================================================
# PhysicsEncoder — Determinism
# ======================================================================


class TestPhysicsEncoderDeterminism:
    """Tests verifying deterministic output in eval mode."""

    def test_deterministic_eval(
        self,
        encoder: PhysicsEncoder,
        sample_input: torch.Tensor,
    ) -> None:
        """Two forward passes with the same input give identical output."""
        encoder.eval()
        with torch.no_grad():
            z1 = encoder(sample_input)
            z2 = encoder(sample_input)
        torch.testing.assert_close(z1, z2, atol=0.0, rtol=0.0)

    def test_different_inputs_different_outputs(
        self, encoder: PhysicsEncoder
    ) -> None:
        """Different inputs produce different embeddings."""
        encoder.eval()
        x1 = torch.randn(2, _DEFAULT_INPUT_DIM)
        x2 = torch.randn(2, _DEFAULT_INPUT_DIM) + 5.0
        with torch.no_grad():
            z1 = encoder(x1)
            z2 = encoder(x2)
        assert not torch.allclose(z1, z2)


# ======================================================================
# PhysicsEncoder — Serialisation
# ======================================================================


class TestPhysicsEncoderSerialisation:
    """Tests for save/load round-tripping."""

    def test_state_dict_round_trip(
        self,
        encoder: PhysicsEncoder,
        sample_input: torch.Tensor,
        default_config: PhysicsEncoderConfig,
    ) -> None:
        """Save and reload via state_dict produces identical output."""
        encoder.eval()
        with torch.no_grad():
            z_orig = encoder(sample_input)

        state = copy.deepcopy(encoder.state_dict())

        # Build a new model and load the state
        new_encoder = PhysicsEncoder(default_config)
        new_encoder.load_state_dict(state)
        new_encoder.eval()

        with torch.no_grad():
            z_loaded = new_encoder(sample_input)

        torch.testing.assert_close(z_orig, z_loaded, atol=1e-6, rtol=1e-6)

    def test_torch_save_load(
        self,
        encoder: PhysicsEncoder,
        sample_input: torch.Tensor,
        default_config: PhysicsEncoderConfig,
    ) -> None:
        """Full torch.save / torch.load round-trip preserves output."""
        encoder.eval()
        with torch.no_grad():
            z_orig = encoder(sample_input)

        with tempfile.NamedTemporaryFile(
            suffix=".pt", delete=False
        ) as tmp:
            torch.save(encoder.state_dict(), tmp.name)
            tmp_path = Path(tmp.name)

        try:
            loaded_state = torch.load(
                tmp_path, map_location="cpu", weights_only=True
            )
            new_encoder = PhysicsEncoder(default_config)
            new_encoder.load_state_dict(loaded_state)
            new_encoder.eval()
            with torch.no_grad():
                z_loaded = new_encoder(sample_input)
            torch.testing.assert_close(z_orig, z_loaded, atol=1e-6, rtol=1e-6)
        finally:
            tmp_path.unlink(missing_ok=True)

    def test_state_dict_keys_stable(
        self, encoder: PhysicsEncoder
    ) -> None:
        """State dict key names are non-empty and consistent."""
        keys = list(encoder.state_dict().keys())
        assert len(keys) > 0
        for k in keys:
            assert isinstance(k, str)
            assert len(k) > 0


# ======================================================================
# PhysicsEncoder — TorchScript
# ======================================================================


class TestPhysicsEncoderTorchScript:
    """Tests for TorchScript compatibility via tracing."""

    def test_torch_jit_trace(
        self,
        encoder: PhysicsEncoder,
        sample_input: torch.Tensor,
    ) -> None:
        """Encoder can be compiled with torch.jit.trace."""
        encoder.eval()
        traced = torch.jit.trace(encoder, sample_input)
        with torch.no_grad():
            z = traced(sample_input)
        assert z.shape == (_DEFAULT_BATCH_SIZE, _DEFAULT_EMBEDDING_DIM)

    def test_traced_output_matches_eager(
        self,
        encoder: PhysicsEncoder,
        sample_input: torch.Tensor,
    ) -> None:
        """Traced model produces the same output as the eager model."""
        encoder.eval()
        traced = torch.jit.trace(encoder, sample_input)
        with torch.no_grad():
            z_eager = encoder(sample_input)
            z_traced = traced(sample_input)
        torch.testing.assert_close(z_eager, z_traced, atol=1e-6, rtol=1e-6)

    def test_traced_l2_normalised(
        self,
        encoder: PhysicsEncoder,
        sample_input: torch.Tensor,
    ) -> None:
        """Traced model output is L2-normalised."""
        encoder.eval()
        traced = torch.jit.trace(encoder, sample_input)
        with torch.no_grad():
            z = traced(sample_input)
        norms = torch.linalg.norm(z, dim=-1)
        torch.testing.assert_close(
            norms,
            torch.ones_like(norms),
            atol=1e-5,
            rtol=1e-5,
        )

    def test_traced_save_load(
        self,
        encoder: PhysicsEncoder,
        sample_input: torch.Tensor,
    ) -> None:
        """Traced model can be saved and reloaded."""
        encoder.eval()
        traced = torch.jit.trace(encoder, sample_input)

        with torch.no_grad():
            z_orig = traced(sample_input)

        with tempfile.NamedTemporaryFile(
            suffix=".pt", delete=False
        ) as tmp:
            torch.jit.save(traced, tmp.name)
            tmp_path = Path(tmp.name)

        try:
            loaded = torch.jit.load(str(tmp_path), map_location="cpu")
            with torch.no_grad():
                z_loaded = loaded(sample_input)
            torch.testing.assert_close(z_orig, z_loaded, atol=1e-6, rtol=1e-6)
        finally:
            tmp_path.unlink(missing_ok=True)


# ======================================================================
# PhysicsEncoder — Invalid inputs
# ======================================================================


class TestPhysicsEncoderInvalidInputs:
    """Tests for input validation and error messages."""

    def test_not_a_tensor(self, encoder: PhysicsEncoder) -> None:
        """Raise TypeError for non-tensor input."""
        with pytest.raises(TypeError, match="torch.Tensor"):
            encoder(np.zeros((2, _DEFAULT_INPUT_DIM)))  # type: ignore[arg-type]

    def test_integer_dtype(self, encoder: PhysicsEncoder) -> None:
        """Raise ValueError for integer-typed tensor."""
        x = torch.randint(0, 10, (2, _DEFAULT_INPUT_DIM))
        with pytest.raises(ValueError, match="floating-point dtype"):
            encoder(x)

    def test_wrong_ndim_1d(self, encoder: PhysicsEncoder) -> None:
        """Raise ValueError for 1-D input."""
        x = torch.randn(_DEFAULT_INPUT_DIM)
        with pytest.raises(ValueError, match="exactly 2 dimensions"):
            encoder(x)

    def test_wrong_ndim_3d(self, encoder: PhysicsEncoder) -> None:
        """Raise ValueError for 3-D input."""
        x = torch.randn(2, 10, _DEFAULT_INPUT_DIM)
        with pytest.raises(ValueError, match="exactly 2 dimensions"):
            encoder(x)

    def test_wrong_ndim_4d(self, encoder: PhysicsEncoder) -> None:
        """Raise ValueError for 4-D input."""
        x = torch.randn(2, 5, _DEFAULT_INPUT_DIM, 1)
        with pytest.raises(ValueError, match="exactly 2 dimensions"):
            encoder(x)

    def test_wrong_feature_dim(self, encoder: PhysicsEncoder) -> None:
        """Raise ValueError when feature dimension mismatches config."""
        wrong_dim = _DEFAULT_INPUT_DIM + 3
        x = torch.randn(2, wrong_dim)
        with pytest.raises(ValueError, match="Feature dimension"):
            encoder(x)

    def test_nan_input(self, encoder: PhysicsEncoder) -> None:
        """Raise ValueError when input contains NaN."""
        x = torch.randn(2, _DEFAULT_INPUT_DIM)
        x[0, 3] = float("nan")
        with pytest.raises(ValueError, match="NaN"):
            encoder(x)

    def test_inf_input(self, encoder: PhysicsEncoder) -> None:
        """Raise ValueError when input contains Inf."""
        x = torch.randn(2, _DEFAULT_INPUT_DIM)
        x[1, 0] = float("inf")
        with pytest.raises(ValueError, match="Inf"):
            encoder(x)

    def test_negative_inf_input(self, encoder: PhysicsEncoder) -> None:
        """Raise ValueError when input contains negative Inf."""
        x = torch.randn(2, _DEFAULT_INPUT_DIM)
        x[0, 5] = float("-inf")
        with pytest.raises(ValueError, match="Inf"):
            encoder(x)


# ======================================================================
# PhysicsEncoder — Device compatibility
# ======================================================================


class TestPhysicsEncoderCPU:
    """CPU-specific compatibility tests."""

    def test_cpu_forward(
        self,
        encoder: PhysicsEncoder,
        sample_input: torch.Tensor,
    ) -> None:
        """Forward pass runs on CPU without error."""
        assert sample_input.device.type == "cpu"
        with torch.no_grad():
            z = encoder(sample_input)
        assert z.device.type == "cpu"
        assert z.shape == (_DEFAULT_BATCH_SIZE, _DEFAULT_EMBEDDING_DIM)


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA not available",
)
class TestPhysicsEncoderCUDA:
    """CUDA-specific compatibility tests (skipped if no GPU)."""

    def test_cuda_forward(self, default_config: PhysicsEncoderConfig) -> None:
        """Forward pass runs on CUDA and returns a CUDA tensor."""
        device = torch.device("cuda")
        model = PhysicsEncoder(default_config).to(device).eval()
        x = torch.randn(
            _DEFAULT_BATCH_SIZE,
            _DEFAULT_INPUT_DIM,
            device=device,
        )
        with torch.no_grad():
            z = model(x)
        assert z.device.type == "cuda"
        assert z.shape == (_DEFAULT_BATCH_SIZE, _DEFAULT_EMBEDDING_DIM)

    def test_cuda_l2_normalised(
        self, default_config: PhysicsEncoderConfig
    ) -> None:
        """CUDA embeddings are L2-normalised."""
        device = torch.device("cuda")
        model = PhysicsEncoder(default_config).to(device).eval()
        x = torch.randn(
            _DEFAULT_BATCH_SIZE,
            _DEFAULT_INPUT_DIM,
            device=device,
        )
        with torch.no_grad():
            z = model(x)
        norms = torch.linalg.norm(z, dim=-1)
        torch.testing.assert_close(
            norms,
            torch.ones_like(norms),
            atol=1e-5,
            rtol=1e-5,
        )

    def test_cuda_gradients(
        self, default_config: PhysicsEncoderConfig
    ) -> None:
        """Gradient flow works on CUDA."""
        device = torch.device("cuda")
        model = PhysicsEncoder(default_config).to(device).train()
        x = torch.randn(
            2,
            _DEFAULT_INPUT_DIM,
            device=device,
            requires_grad=True,
        )
        z = model(x)
        z.sum().backward()
        assert x.grad is not None
        assert x.grad.shape == x.shape


# ======================================================================
# PhysicsEncoder — Float64 compatibility
# ======================================================================


class TestPhysicsEncoderFloat64:
    """Ensure the encoder works with float64 input."""

    def test_float64_forward(
        self, default_config: PhysicsEncoderConfig
    ) -> None:
        """Forward works when both model and input are float64."""
        model = PhysicsEncoder(default_config).double().eval()
        x = torch.randn(
            2, _DEFAULT_INPUT_DIM, dtype=torch.float64
        )
        with torch.no_grad():
            z = model(x)
        assert z.dtype == torch.float64
        assert z.shape == (2, _DEFAULT_EMBEDDING_DIM)


# ======================================================================
# PhysicsEncoder — Single feature
# ======================================================================


class TestPhysicsEncoderSingleFeature:
    """Edge case: single physics feature."""

    def test_single_feature(self) -> None:
        """Encoder works with input_dim=1."""
        cfg = PhysicsEncoderConfig(input_dim=1)
        model = PhysicsEncoder(cfg).eval()
        x = torch.randn(2, 1)
        with torch.no_grad():
            z = model(x)
        assert z.shape == (2, _DEFAULT_EMBEDDING_DIM)
        norms = torch.linalg.norm(z, dim=-1)
        torch.testing.assert_close(
            norms,
            torch.ones_like(norms),
            atol=1e-5,
            rtol=1e-5,
        )
