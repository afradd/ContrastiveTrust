"""Tests for :mod:`src.models.temporal_encoder` and :mod:`src.models.blocks`.

Comprehensive test suite covering initialisation, forward-pass correctness,
output shape, L2 normalisation, gradient flow, serialisation round-trip,
deterministic behaviour, variable window lengths, invalid-input rejection
(dimension, sensor count, NaN, Inf), and device compatibility.
"""

from __future__ import annotations

import copy
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn

from src.models.blocks import ConvNormActivation, ResidualConvBlock
from src.models.temporal_encoder import TemporalEncoder, TemporalEncoderConfig


# ======================================================================
# Fixtures
# ======================================================================

_DEFAULT_INPUT_CHANNELS: int = 10
_DEFAULT_WINDOW_LEN: int = 100
_DEFAULT_BATCH_SIZE: int = 4
_DEFAULT_EMBEDDING_DIM: int = 256


@pytest.fixture()
def default_config() -> TemporalEncoderConfig:
    """Return a default encoder config for 10 sensor channels."""
    return TemporalEncoderConfig(input_channels=_DEFAULT_INPUT_CHANNELS)


@pytest.fixture()
def encoder(default_config: TemporalEncoderConfig) -> TemporalEncoder:
    """Return an encoder in eval mode with a fixed seed."""
    torch.manual_seed(42)
    model = TemporalEncoder(default_config)
    model.eval()
    return model


@pytest.fixture()
def sample_input() -> torch.Tensor:
    """Return a reproducible (B, T, S) input tensor."""
    torch.manual_seed(0)
    return torch.randn(
        _DEFAULT_BATCH_SIZE,
        _DEFAULT_WINDOW_LEN,
        _DEFAULT_INPUT_CHANNELS,
    )


# ======================================================================
# ConvNormActivation tests
# ======================================================================


class TestConvNormActivation:
    """Tests for the :class:`ConvNormActivation` building block."""

    def test_output_shape(self) -> None:
        """Output preserves temporal dimension with default same-padding."""
        block = ConvNormActivation(8, 16, kernel_size=5)
        x = torch.randn(2, 8, 50)
        y = block(x)
        assert y.shape == (2, 16, 50)

    def test_no_activation(self) -> None:
        """Block runs without error when activation is None."""
        block = ConvNormActivation(4, 8, kernel_size=3, activation=None)
        x = torch.randn(1, 4, 20)
        y = block(x)
        assert y.shape == (1, 8, 20)

    def test_invalid_in_channels(self) -> None:
        """Raise ValueError for non-positive in_channels."""
        with pytest.raises(ValueError, match="in_channels must be positive"):
            ConvNormActivation(0, 8)

    def test_invalid_out_channels(self) -> None:
        """Raise ValueError for non-positive out_channels."""
        with pytest.raises(ValueError, match="out_channels must be positive"):
            ConvNormActivation(8, 0)

    def test_invalid_kernel_size_even(self) -> None:
        """Raise ValueError for an even kernel size."""
        with pytest.raises(ValueError, match="kernel_size must be a positive odd"):
            ConvNormActivation(8, 8, kernel_size=4)

    def test_invalid_kernel_size_zero(self) -> None:
        """Raise ValueError for zero kernel size."""
        with pytest.raises(ValueError, match="kernel_size must be a positive odd"):
            ConvNormActivation(8, 8, kernel_size=0)

    def test_custom_dilation(self) -> None:
        """Block works with dilation > 1 and auto-padding."""
        block = ConvNormActivation(4, 8, kernel_size=3, dilation=2)
        x = torch.randn(1, 4, 30)
        y = block(x)
        assert y.shape == (1, 8, 30)

    def test_with_bias(self) -> None:
        """Block works when bias is enabled."""
        block = ConvNormActivation(4, 8, kernel_size=3, bias=True)
        x = torch.randn(1, 4, 20)
        y = block(x)
        assert y.shape == (1, 8, 20)


# ======================================================================
# ResidualConvBlock tests
# ======================================================================


class TestResidualConvBlock:
    """Tests for the :class:`ResidualConvBlock`."""

    def test_same_channels(self) -> None:
        """Identity shortcut when in == out channels."""
        block = ResidualConvBlock(16, 16, kernel_size=3)
        x = torch.randn(2, 16, 40)
        y = block(x)
        assert y.shape == x.shape

    def test_channel_projection(self) -> None:
        """1×1 shortcut projection when channels differ."""
        block = ResidualConvBlock(8, 32, kernel_size=5)
        x = torch.randn(2, 8, 40)
        y = block(x)
        assert y.shape == (2, 32, 40)

    def test_with_dropout(self) -> None:
        """Block runs with nonzero dropout."""
        block = ResidualConvBlock(16, 16, kernel_size=3, dropout=0.5)
        block.train()
        x = torch.randn(2, 16, 40)
        y = block(x)
        assert y.shape == x.shape

    def test_invalid_dropout(self) -> None:
        """Raise ValueError for dropout >= 1."""
        with pytest.raises(ValueError, match="dropout must be in"):
            ResidualConvBlock(8, 8, dropout=1.0)

    def test_invalid_negative_dropout(self) -> None:
        """Raise ValueError for negative dropout."""
        with pytest.raises(ValueError, match="dropout must be in"):
            ResidualConvBlock(8, 8, dropout=-0.1)

    def test_gradient_flow(self) -> None:
        """Gradients flow through both the residual and shortcut paths."""
        block = ResidualConvBlock(8, 16, kernel_size=3)
        x = torch.randn(1, 8, 30, requires_grad=True)
        y = block(x)
        loss = y.sum()
        loss.backward()
        assert x.grad is not None
        assert torch.all(x.grad != 0)


# ======================================================================
# TemporalEncoderConfig tests
# ======================================================================


class TestTemporalEncoderConfig:
    """Tests for configuration validation."""

    def test_default_values(self) -> None:
        """Defaults match the specification."""
        cfg = TemporalEncoderConfig(input_channels=5)
        assert cfg.input_channels == 5
        assert cfg.embedding_dim == 256
        assert cfg.hidden_channels == (64, 128, 256)
        assert cfg.kernel_sizes == (7, 5, 3)
        assert cfg.dropout == pytest.approx(0.2)
        assert cfg.bias is False

    def test_invalid_input_channels(self) -> None:
        """Raise ValueError for input_channels < 1."""
        with pytest.raises(ValueError, match="input_channels must be positive"):
            TemporalEncoderConfig(input_channels=0)

    def test_invalid_embedding_dim(self) -> None:
        """Raise ValueError for embedding_dim < 1."""
        with pytest.raises(ValueError, match="embedding_dim must be positive"):
            TemporalEncoderConfig(input_channels=5, embedding_dim=0)

    def test_empty_hidden_channels(self) -> None:
        """Raise ValueError for empty hidden_channels."""
        with pytest.raises(ValueError, match="hidden_channels must not be empty"):
            TemporalEncoderConfig(input_channels=5, hidden_channels=())

    def test_mismatched_kernel_sizes(self) -> None:
        """Raise ValueError when kernel_sizes length != hidden_channels."""
        with pytest.raises(ValueError, match="kernel_sizes length"):
            TemporalEncoderConfig(
                input_channels=5,
                hidden_channels=(32, 64),
                kernel_sizes=(3,),
            )

    def test_even_kernel_size(self) -> None:
        """Raise ValueError for even kernel size."""
        with pytest.raises(ValueError, match="positive odd integer"):
            TemporalEncoderConfig(
                input_channels=5,
                hidden_channels=(32,),
                kernel_sizes=(4,),
            )

    def test_invalid_dropout(self) -> None:
        """Raise ValueError for dropout >= 1."""
        with pytest.raises(ValueError, match="dropout must be in"):
            TemporalEncoderConfig(input_channels=5, dropout=1.0)

    def test_negative_hidden_channels(self) -> None:
        """Raise ValueError for negative channel count."""
        with pytest.raises(ValueError, match="hidden_channels\\[0\\] must be positive"):
            TemporalEncoderConfig(
                input_channels=5,
                hidden_channels=(-1,),
                kernel_sizes=(3,),
            )

    def test_frozen_config(self) -> None:
        """Config is immutable after creation."""
        cfg = TemporalEncoderConfig(input_channels=5)
        with pytest.raises(AttributeError):
            cfg.input_channels = 99  # type: ignore[misc]


# ======================================================================
# TemporalEncoder — Initialisation
# ======================================================================


class TestTemporalEncoderInit:
    """Tests for encoder construction."""

    def test_type_error_config(self) -> None:
        """Raise TypeError when config is not TemporalEncoderConfig."""
        with pytest.raises(TypeError, match="TemporalEncoderConfig"):
            TemporalEncoder(config={"input_channels": 5})  # type: ignore[arg-type]

    def test_properties(
        self,
        encoder: TemporalEncoder,
        default_config: TemporalEncoderConfig,
    ) -> None:
        """Public properties reflect the config."""
        assert encoder.input_channels == _DEFAULT_INPUT_CHANNELS
        assert encoder.embedding_dim == _DEFAULT_EMBEDDING_DIM
        assert encoder.config is default_config

    def test_parameter_count_positive(self, encoder: TemporalEncoder) -> None:
        """The encoder has a non-trivial number of parameters."""
        total = sum(p.numel() for p in encoder.parameters())
        assert total > 0

    def test_all_parameters_require_grad(
        self, encoder: TemporalEncoder
    ) -> None:
        """All parameters are trainable by default."""
        for name, param in encoder.named_parameters():
            assert param.requires_grad, f"Parameter {name} does not require grad"

    def test_custom_config(self) -> None:
        """Encoder accepts a non-default configuration."""
        cfg = TemporalEncoderConfig(
            input_channels=20,
            embedding_dim=128,
            hidden_channels=(32, 64),
            kernel_sizes=(5, 3),
            dropout=0.1,
            bias=True,
        )
        model = TemporalEncoder(cfg)
        assert model.embedding_dim == 128
        assert model.input_channels == 20


# ======================================================================
# TemporalEncoder — Forward pass
# ======================================================================


class TestTemporalEncoderForward:
    """Tests for the forward pass."""

    def test_output_shape(
        self,
        encoder: TemporalEncoder,
        sample_input: torch.Tensor,
    ) -> None:
        """Output shape is (B, embedding_dim)."""
        with torch.no_grad():
            z = encoder(sample_input)
        assert z.shape == (_DEFAULT_BATCH_SIZE, _DEFAULT_EMBEDDING_DIM)

    def test_output_dtype(
        self,
        encoder: TemporalEncoder,
        sample_input: torch.Tensor,
    ) -> None:
        """Output dtype matches input dtype."""
        with torch.no_grad():
            z = encoder(sample_input)
        assert z.dtype == sample_input.dtype

    def test_l2_normalised(
        self,
        encoder: TemporalEncoder,
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
        encoder: TemporalEncoder,
        sample_input: torch.Tensor,
    ) -> None:
        """Output does not contain NaN values."""
        with torch.no_grad():
            z = encoder(sample_input)
        assert not torch.isnan(z).any()

    def test_no_inf_in_output(
        self,
        encoder: TemporalEncoder,
        sample_input: torch.Tensor,
    ) -> None:
        """Output does not contain Inf values."""
        with torch.no_grad():
            z = encoder(sample_input)
        assert not torch.isinf(z).any()

    def test_batch_size_one(self, encoder: TemporalEncoder) -> None:
        """Forward works for a single-sample batch."""
        x = torch.randn(1, _DEFAULT_WINDOW_LEN, _DEFAULT_INPUT_CHANNELS)
        with torch.no_grad():
            z = encoder(x)
        assert z.shape == (1, _DEFAULT_EMBEDDING_DIM)

    def test_large_batch(self, encoder: TemporalEncoder) -> None:
        """Forward works for a larger batch."""
        x = torch.randn(32, _DEFAULT_WINDOW_LEN, _DEFAULT_INPUT_CHANNELS)
        with torch.no_grad():
            z = encoder(x)
        assert z.shape == (32, _DEFAULT_EMBEDDING_DIM)


# ======================================================================
# TemporalEncoder — Gradient flow
# ======================================================================


class TestTemporalEncoderGradients:
    """Tests verifying gradient flow for training."""

    def test_gradients_exist(
        self,
        encoder: TemporalEncoder,
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
        encoder: TemporalEncoder,
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

    def test_input_gradient(self, encoder: TemporalEncoder) -> None:
        """Gradients propagate back to the input tensor."""
        encoder.train()
        x = torch.randn(
            2, _DEFAULT_WINDOW_LEN, _DEFAULT_INPUT_CHANNELS,
            requires_grad=True,
        )
        z = encoder(x)
        z.sum().backward()
        assert x.grad is not None
        assert x.grad.shape == x.shape


# ======================================================================
# TemporalEncoder — Determinism
# ======================================================================


class TestTemporalEncoderDeterminism:
    """Tests verifying deterministic output in eval mode."""

    def test_deterministic_eval(
        self,
        encoder: TemporalEncoder,
        sample_input: torch.Tensor,
    ) -> None:
        """Two forward passes with the same input give identical output."""
        encoder.eval()
        with torch.no_grad():
            z1 = encoder(sample_input)
            z2 = encoder(sample_input)
        torch.testing.assert_close(z1, z2, atol=0.0, rtol=0.0)

    def test_different_inputs_different_outputs(
        self, encoder: TemporalEncoder
    ) -> None:
        """Different inputs produce different embeddings."""
        encoder.eval()
        x1 = torch.randn(2, _DEFAULT_WINDOW_LEN, _DEFAULT_INPUT_CHANNELS)
        x2 = torch.randn(2, _DEFAULT_WINDOW_LEN, _DEFAULT_INPUT_CHANNELS) + 5.0
        with torch.no_grad():
            z1 = encoder(x1)
            z2 = encoder(x2)
        assert not torch.allclose(z1, z2)


# ======================================================================
# TemporalEncoder — Variable window lengths
# ======================================================================


class TestTemporalEncoderVariableWindows:
    """Tests demonstrating support for arbitrary window lengths."""

    @pytest.mark.parametrize("window_len", [10, 25, 50, 100, 200, 500])
    def test_variable_window_length(
        self,
        encoder: TemporalEncoder,
        window_len: int,
    ) -> None:
        """Encoder produces correct shape for varied window lengths."""
        x = torch.randn(2, window_len, _DEFAULT_INPUT_CHANNELS)
        with torch.no_grad():
            z = encoder(x)
        assert z.shape == (2, _DEFAULT_EMBEDDING_DIM)

    @pytest.mark.parametrize("window_len", [10, 25, 50, 100, 200, 500])
    def test_variable_window_normalised(
        self,
        encoder: TemporalEncoder,
        window_len: int,
    ) -> None:
        """Embeddings are L2-normalised regardless of window length."""
        x = torch.randn(2, window_len, _DEFAULT_INPUT_CHANNELS)
        with torch.no_grad():
            z = encoder(x)
        norms = torch.linalg.norm(z, dim=-1)
        torch.testing.assert_close(
            norms,
            torch.ones_like(norms),
            atol=1e-5,
            rtol=1e-5,
        )

    def test_window_length_one(self, encoder: TemporalEncoder) -> None:
        """Forward works for the minimal window length of 1."""
        x = torch.randn(1, 1, _DEFAULT_INPUT_CHANNELS)
        with torch.no_grad():
            z = encoder(x)
        assert z.shape == (1, _DEFAULT_EMBEDDING_DIM)


# ======================================================================
# TemporalEncoder — Serialisation
# ======================================================================


class TestTemporalEncoderSerialisation:
    """Tests for save/load round-tripping."""

    def test_state_dict_round_trip(
        self,
        encoder: TemporalEncoder,
        sample_input: torch.Tensor,
        default_config: TemporalEncoderConfig,
    ) -> None:
        """Save and reload via state_dict produces identical output."""
        encoder.eval()
        with torch.no_grad():
            z_orig = encoder(sample_input)

        state = copy.deepcopy(encoder.state_dict())

        # Build a new model and load the state
        new_encoder = TemporalEncoder(default_config)
        new_encoder.load_state_dict(state)
        new_encoder.eval()

        with torch.no_grad():
            z_loaded = new_encoder(sample_input)

        torch.testing.assert_close(z_orig, z_loaded, atol=1e-6, rtol=1e-6)

    def test_torch_save_load(
        self,
        encoder: TemporalEncoder,
        sample_input: torch.Tensor,
        default_config: TemporalEncoderConfig,
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
            new_encoder = TemporalEncoder(default_config)
            new_encoder.load_state_dict(loaded_state)
            new_encoder.eval()
            with torch.no_grad():
                z_loaded = new_encoder(sample_input)
            torch.testing.assert_close(z_orig, z_loaded, atol=1e-6, rtol=1e-6)
        finally:
            tmp_path.unlink(missing_ok=True)

    def test_state_dict_keys_stable(
        self, encoder: TemporalEncoder
    ) -> None:
        """State dict key names are non-empty and consistent."""
        keys = list(encoder.state_dict().keys())
        assert len(keys) > 0
        for k in keys:
            assert isinstance(k, str)
            assert len(k) > 0


# ======================================================================
# TemporalEncoder — Invalid inputs
# ======================================================================


class TestTemporalEncoderInvalidInputs:
    """Tests for input validation and error messages."""

    def test_not_a_tensor(self, encoder: TemporalEncoder) -> None:
        """Raise TypeError for non-tensor input."""
        with pytest.raises(TypeError, match="torch.Tensor"):
            encoder(np.zeros((2, 50, _DEFAULT_INPUT_CHANNELS)))  # type: ignore[arg-type]

    def test_integer_dtype(self, encoder: TemporalEncoder) -> None:
        """Raise ValueError for integer-typed tensor."""
        x = torch.randint(0, 10, (2, 50, _DEFAULT_INPUT_CHANNELS))
        with pytest.raises(ValueError, match="floating-point dtype"):
            encoder(x)

    def test_wrong_ndim_2d(self, encoder: TemporalEncoder) -> None:
        """Raise ValueError for 2-D input."""
        x = torch.randn(50, _DEFAULT_INPUT_CHANNELS)
        with pytest.raises(ValueError, match="exactly 3 dimensions"):
            encoder(x)

    def test_wrong_ndim_4d(self, encoder: TemporalEncoder) -> None:
        """Raise ValueError for 4-D input."""
        x = torch.randn(2, 50, _DEFAULT_INPUT_CHANNELS, 1)
        with pytest.raises(ValueError, match="exactly 3 dimensions"):
            encoder(x)

    def test_wrong_sensor_count(self, encoder: TemporalEncoder) -> None:
        """Raise ValueError when sensor dimension mismatches config."""
        wrong_sensors = _DEFAULT_INPUT_CHANNELS + 3
        x = torch.randn(2, 50, wrong_sensors)
        with pytest.raises(ValueError, match="Sensor count"):
            encoder(x)

    def test_nan_input(self, encoder: TemporalEncoder) -> None:
        """Raise ValueError when input contains NaN."""
        x = torch.randn(2, 50, _DEFAULT_INPUT_CHANNELS)
        x[0, 10, 3] = float("nan")
        with pytest.raises(ValueError, match="NaN"):
            encoder(x)

    def test_inf_input(self, encoder: TemporalEncoder) -> None:
        """Raise ValueError when input contains Inf."""
        x = torch.randn(2, 50, _DEFAULT_INPUT_CHANNELS)
        x[1, 0, 0] = float("inf")
        with pytest.raises(ValueError, match="Inf"):
            encoder(x)

    def test_negative_inf_input(self, encoder: TemporalEncoder) -> None:
        """Raise ValueError when input contains negative Inf."""
        x = torch.randn(2, 50, _DEFAULT_INPUT_CHANNELS)
        x[0, 5, 2] = float("-inf")
        with pytest.raises(ValueError, match="Inf"):
            encoder(x)

    def test_wrong_ndim_1d(self, encoder: TemporalEncoder) -> None:
        """Raise ValueError for 1-D input."""
        x = torch.randn(50)
        with pytest.raises(ValueError, match="exactly 3 dimensions"):
            encoder(x)

    def test_wrong_ndim_5d(self, encoder: TemporalEncoder) -> None:
        """Raise ValueError for 5-D input."""
        x = torch.randn(2, 1, 50, _DEFAULT_INPUT_CHANNELS, 1)
        with pytest.raises(ValueError, match="exactly 3 dimensions"):
            encoder(x)


# ======================================================================
# TemporalEncoder — Device compatibility
# ======================================================================


class TestTemporalEncoderCPU:
    """CPU-specific compatibility tests."""

    def test_cpu_forward(
        self,
        encoder: TemporalEncoder,
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
class TestTemporalEncoderCUDA:
    """CUDA-specific compatibility tests (skipped if no GPU)."""

    def test_cuda_forward(self, default_config: TemporalEncoderConfig) -> None:
        """Forward pass runs on CUDA and returns a CUDA tensor."""
        device = torch.device("cuda")
        model = TemporalEncoder(default_config).to(device).eval()
        x = torch.randn(
            _DEFAULT_BATCH_SIZE,
            _DEFAULT_WINDOW_LEN,
            _DEFAULT_INPUT_CHANNELS,
            device=device,
        )
        with torch.no_grad():
            z = model(x)
        assert z.device.type == "cuda"
        assert z.shape == (_DEFAULT_BATCH_SIZE, _DEFAULT_EMBEDDING_DIM)

    def test_cuda_l2_normalised(
        self, default_config: TemporalEncoderConfig
    ) -> None:
        """CUDA embeddings are L2-normalised."""
        device = torch.device("cuda")
        model = TemporalEncoder(default_config).to(device).eval()
        x = torch.randn(
            _DEFAULT_BATCH_SIZE,
            _DEFAULT_WINDOW_LEN,
            _DEFAULT_INPUT_CHANNELS,
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
        self, default_config: TemporalEncoderConfig
    ) -> None:
        """Gradient flow works on CUDA."""
        device = torch.device("cuda")
        model = TemporalEncoder(default_config).to(device).train()
        x = torch.randn(
            2,
            _DEFAULT_WINDOW_LEN,
            _DEFAULT_INPUT_CHANNELS,
            device=device,
            requires_grad=True,
        )
        z = model(x)
        z.sum().backward()
        assert x.grad is not None
        assert x.grad.shape == x.shape


# ======================================================================
# TemporalEncoder — Float16 compatibility
# ======================================================================


class TestTemporalEncoderFloat64:
    """Ensure the encoder works with float64 input."""

    def test_float64_forward(
        self, default_config: TemporalEncoderConfig
    ) -> None:
        """Forward works when both model and input are float64."""
        model = TemporalEncoder(default_config).double().eval()
        x = torch.randn(
            2, _DEFAULT_WINDOW_LEN, _DEFAULT_INPUT_CHANNELS, dtype=torch.float64
        )
        with torch.no_grad():
            z = model(x)
        assert z.dtype == torch.float64
        assert z.shape == (2, _DEFAULT_EMBEDDING_DIM)


# ======================================================================
# TemporalEncoder — Single sensor channel
# ======================================================================


class TestTemporalEncoderSingleChannel:
    """Edge case: single sensor column."""

    def test_single_channel(self) -> None:
        """Encoder works with input_channels=1."""
        cfg = TemporalEncoderConfig(input_channels=1)
        model = TemporalEncoder(cfg).eval()
        x = torch.randn(2, 50, 1)
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
