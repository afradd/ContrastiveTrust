"""Tests for :mod:`src.models.projection_head`.

Comprehensive test suite covering initialisation, forward-pass correctness,
output shape, L2 normalisation, gradient flow, deterministic behaviour,
serialisation round-trip, TorchScript tracing, ``project()`` convenience
method, ``parameter_count()``, invalid-input rejection (type, dtype,
dimension, embedding size, NaN, Inf), and device compatibility
(CPU / CUDA).
"""

from __future__ import annotations

import copy
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

from src.models.projection_head import ProjectionHead, ProjectionHeadConfig


# ======================================================================
# Constants
# ======================================================================

_DEFAULT_INPUT_DIM: int = 256
_DEFAULT_HIDDEN_DIM: int = 256
_DEFAULT_OUTPUT_DIM: int = 128
_DEFAULT_BATCH_SIZE: int = 4


# ======================================================================
# Fixtures
# ======================================================================


@pytest.fixture()
def default_config() -> ProjectionHeadConfig:
    """Return a default projection head config."""
    return ProjectionHeadConfig()


@pytest.fixture()
def head(default_config: ProjectionHeadConfig) -> ProjectionHead:
    """Return a projection head in eval mode with a fixed seed."""
    torch.manual_seed(42)
    model = ProjectionHead(default_config)
    model.eval()
    return model


@pytest.fixture()
def sample_embedding() -> torch.Tensor:
    """Return a reproducible (B, D_in) embedding."""
    torch.manual_seed(0)
    return torch.randn(_DEFAULT_BATCH_SIZE, _DEFAULT_INPUT_DIM)


# ======================================================================
# ProjectionHeadConfig tests
# ======================================================================


class TestProjectionHeadConfig:
    """Tests for configuration validation."""

    def test_default_values(self) -> None:
        """Defaults match the specification."""
        cfg = ProjectionHeadConfig()
        assert cfg.input_dim == 256
        assert cfg.hidden_dim == 256
        assert cfg.output_dim == 128
        assert cfg.dropout == pytest.approx(0.2)
        assert cfg.bias is True
        assert cfg.activation == "gelu"

    def test_custom_values(self) -> None:
        """Config accepts custom values."""
        cfg = ProjectionHeadConfig(
            input_dim=512,
            hidden_dim=384,
            output_dim=64,
            dropout=0.1,
            bias=False,
            activation="relu",
        )
        assert cfg.input_dim == 512
        assert cfg.hidden_dim == 384
        assert cfg.output_dim == 64
        assert cfg.dropout == pytest.approx(0.1)
        assert cfg.bias is False
        assert cfg.activation == "relu"

    def test_frozen(self) -> None:
        """Config is immutable (frozen dataclass)."""
        cfg = ProjectionHeadConfig()
        with pytest.raises(AttributeError):
            cfg.input_dim = 512  # type: ignore[misc]

    def test_invalid_input_dim_zero(self) -> None:
        """Raise ValueError for input_dim < 1."""
        with pytest.raises(ValueError, match="input_dim must be positive"):
            ProjectionHeadConfig(input_dim=0)

    def test_invalid_input_dim_negative(self) -> None:
        """Raise ValueError for negative input_dim."""
        with pytest.raises(ValueError, match="input_dim must be positive"):
            ProjectionHeadConfig(input_dim=-10)

    def test_invalid_hidden_dim_zero(self) -> None:
        """Raise ValueError for hidden_dim < 1."""
        with pytest.raises(ValueError, match="hidden_dim must be positive"):
            ProjectionHeadConfig(hidden_dim=0)

    def test_invalid_hidden_dim_negative(self) -> None:
        """Raise ValueError for negative hidden_dim."""
        with pytest.raises(ValueError, match="hidden_dim must be positive"):
            ProjectionHeadConfig(hidden_dim=-5)

    def test_invalid_output_dim_zero(self) -> None:
        """Raise ValueError for output_dim < 1."""
        with pytest.raises(ValueError, match="output_dim must be positive"):
            ProjectionHeadConfig(output_dim=0)

    def test_invalid_output_dim_negative(self) -> None:
        """Raise ValueError for negative output_dim."""
        with pytest.raises(ValueError, match="output_dim must be positive"):
            ProjectionHeadConfig(output_dim=-1)

    def test_invalid_dropout_too_high(self) -> None:
        """Raise ValueError for dropout >= 1."""
        with pytest.raises(ValueError, match="dropout must be in"):
            ProjectionHeadConfig(dropout=1.0)

    def test_invalid_dropout_negative(self) -> None:
        """Raise ValueError for negative dropout."""
        with pytest.raises(ValueError, match="dropout must be in"):
            ProjectionHeadConfig(dropout=-0.1)

    def test_invalid_activation(self) -> None:
        """Raise ValueError for unsupported activation."""
        with pytest.raises(ValueError, match="activation must be one of"):
            ProjectionHeadConfig(activation="swish")

    def test_zero_dropout_valid(self) -> None:
        """Zero dropout is valid."""
        cfg = ProjectionHeadConfig(dropout=0.0)
        assert cfg.dropout == 0.0

    def test_all_activations_valid(self) -> None:
        """All supported activations are accepted."""
        for act in ("gelu", "relu", "silu", "tanh"):
            cfg = ProjectionHeadConfig(activation=act)
            assert cfg.activation == act


# ======================================================================
# ProjectionHead — Initialisation
# ======================================================================


class TestProjectionHeadInit:
    """Tests for module initialisation."""

    def test_creates_module(self, default_config: ProjectionHeadConfig) -> None:
        """ProjectionHead is a valid nn.Module."""
        head = ProjectionHead(default_config)
        assert isinstance(head, torch.nn.Module)

    def test_config_property(
        self, head: ProjectionHead, default_config: ProjectionHeadConfig
    ) -> None:
        """Config property returns the original config."""
        assert head.config is default_config

    def test_input_dim_property(self, head: ProjectionHead) -> None:
        """input_dim property returns the configured value."""
        assert head.input_dim == _DEFAULT_INPUT_DIM

    def test_output_dim_property(self, head: ProjectionHead) -> None:
        """output_dim property returns the configured value."""
        assert head.output_dim == _DEFAULT_OUTPUT_DIM

    def test_invalid_config_type(self) -> None:
        """Raise TypeError for non-config argument."""
        with pytest.raises(TypeError, match="config must be a ProjectionHeadConfig"):
            ProjectionHead({"input_dim": 256})  # type: ignore[arg-type]

    def test_invalid_config_none(self) -> None:
        """Raise TypeError for None config."""
        with pytest.raises(TypeError, match="config must be a ProjectionHeadConfig"):
            ProjectionHead(None)  # type: ignore[arg-type]

    def test_custom_config(self) -> None:
        """ProjectionHead accepts custom config."""
        cfg = ProjectionHeadConfig(
            input_dim=512,
            hidden_dim=384,
            output_dim=64,
            dropout=0.1,
            bias=False,
            activation="relu",
        )
        head = ProjectionHead(cfg)
        assert head.input_dim == 512
        assert head.output_dim == 64

    def test_has_projection_sequential(self, head: ProjectionHead) -> None:
        """Module contains a projection Sequential."""
        assert hasattr(head, "projection")
        assert isinstance(head.projection, torch.nn.Sequential)


# ======================================================================
# ProjectionHead — Forward pass
# ======================================================================


class TestProjectionHeadForward:
    """Tests for the forward method."""

    def test_output_shape(
        self,
        head: ProjectionHead,
        sample_embedding: torch.Tensor,
    ) -> None:
        """Output has shape (B, output_dim)."""
        with torch.no_grad():
            z = head(sample_embedding)
        assert z.shape == (_DEFAULT_BATCH_SIZE, _DEFAULT_OUTPUT_DIM)

    def test_output_dtype_float32(
        self,
        head: ProjectionHead,
        sample_embedding: torch.Tensor,
    ) -> None:
        """Output dtype matches input dtype (float32)."""
        with torch.no_grad():
            z = head(sample_embedding)
        assert z.dtype == torch.float32

    def test_output_l2_normalised(
        self,
        head: ProjectionHead,
        sample_embedding: torch.Tensor,
    ) -> None:
        """Output embeddings are L2-normalised (unit norm)."""
        with torch.no_grad():
            z = head(sample_embedding)
        norms = torch.linalg.norm(z, dim=-1)
        torch.testing.assert_close(
            norms,
            torch.ones_like(norms),
            atol=1e-5,
            rtol=1e-5,
        )

    def test_output_no_nan(
        self,
        head: ProjectionHead,
        sample_embedding: torch.Tensor,
    ) -> None:
        """Output contains no NaN values."""
        with torch.no_grad():
            z = head(sample_embedding)
        assert not torch.isnan(z).any()

    def test_output_no_inf(
        self,
        head: ProjectionHead,
        sample_embedding: torch.Tensor,
    ) -> None:
        """Output contains no Inf values."""
        with torch.no_grad():
            z = head(sample_embedding)
        assert not torch.isinf(z).any()

    def test_batch_size_one(self, head: ProjectionHead) -> None:
        """Forward works with batch size 1."""
        x = torch.randn(1, _DEFAULT_INPUT_DIM)
        with torch.no_grad():
            z = head(x)
        assert z.shape == (1, _DEFAULT_OUTPUT_DIM)

    def test_large_batch(self, head: ProjectionHead) -> None:
        """Forward works with a large batch."""
        x = torch.randn(64, _DEFAULT_INPUT_DIM)
        with torch.no_grad():
            z = head(x)
        assert z.shape == (64, _DEFAULT_OUTPUT_DIM)

    def test_custom_dimensions(self) -> None:
        """Forward works with non-default dimensions."""
        cfg = ProjectionHeadConfig(
            input_dim=512,
            hidden_dim=384,
            output_dim=64,
        )
        head = ProjectionHead(cfg)
        head.eval()
        x = torch.randn(2, 512)
        with torch.no_grad():
            z = head(x)
        assert z.shape == (2, 64)


# ======================================================================
# ProjectionHead — Deterministic inference
# ======================================================================


class TestProjectionHeadDeterministic:
    """Tests for deterministic output in eval mode."""

    def test_same_input_same_output(
        self,
        head: ProjectionHead,
        sample_embedding: torch.Tensor,
    ) -> None:
        """Two forward passes with the same input give identical output."""
        head.eval()
        with torch.no_grad():
            z1 = head(sample_embedding)
            z2 = head(sample_embedding)
        torch.testing.assert_close(z1, z2, atol=0.0, rtol=0.0)

    def test_different_inputs_different_outputs(
        self, head: ProjectionHead
    ) -> None:
        """Different inputs produce different embeddings."""
        head.eval()
        x1 = torch.randn(2, _DEFAULT_INPUT_DIM)
        x2 = torch.randn(2, _DEFAULT_INPUT_DIM) + 5.0
        with torch.no_grad():
            z1 = head(x1)
            z2 = head(x2)
        assert not torch.allclose(z1, z2)


# ======================================================================
# ProjectionHead — project() method
# ======================================================================


class TestProjectionHeadProject:
    """Tests for the project() convenience method."""

    def test_project_output_shape(
        self,
        head: ProjectionHead,
        sample_embedding: torch.Tensor,
    ) -> None:
        """project() returns the correct shape."""
        z = head.project(sample_embedding)
        assert z.shape == (_DEFAULT_BATCH_SIZE, _DEFAULT_OUTPUT_DIM)

    def test_project_l2_normalised(
        self,
        head: ProjectionHead,
        sample_embedding: torch.Tensor,
    ) -> None:
        """project() output is L2-normalised."""
        z = head.project(sample_embedding)
        norms = torch.linalg.norm(z, dim=-1)
        torch.testing.assert_close(
            norms,
            torch.ones_like(norms),
            atol=1e-5,
            rtol=1e-5,
        )

    def test_project_matches_forward_eval(
        self,
        head: ProjectionHead,
        sample_embedding: torch.Tensor,
    ) -> None:
        """project() gives same output as forward() in eval mode."""
        head.eval()
        with torch.no_grad():
            z_forward = head(sample_embedding)
        z_project = head.project(sample_embedding)
        torch.testing.assert_close(z_forward, z_project, atol=0.0, rtol=0.0)

    def test_project_no_grad(
        self,
        head: ProjectionHead,
        sample_embedding: torch.Tensor,
    ) -> None:
        """project() does not accumulate gradients."""
        embedding = sample_embedding.clone().requires_grad_(True)
        z = head.project(embedding)
        assert not z.requires_grad

    def test_project_restores_training_mode(
        self, default_config: ProjectionHeadConfig
    ) -> None:
        """project() restores original training mode after call."""
        head = ProjectionHead(default_config)
        head.train()
        assert head.training is True
        x = torch.randn(2, _DEFAULT_INPUT_DIM)
        head.project(x)
        assert head.training is True


# ======================================================================
# ProjectionHead — parameter_count()
# ======================================================================


class TestProjectionHeadParameterCount:
    """Tests for the parameter_count() method."""

    def test_returns_dict(self, head: ProjectionHead) -> None:
        """parameter_count() returns a dict."""
        counts = head.parameter_count()
        assert isinstance(counts, dict)

    def test_keys(self, head: ProjectionHead) -> None:
        """parameter_count() has expected keys."""
        counts = head.parameter_count()
        assert "total" in counts
        assert "trainable" in counts

    def test_positive_counts(self, head: ProjectionHead) -> None:
        """All counts are positive."""
        counts = head.parameter_count()
        assert counts["total"] > 0
        assert counts["trainable"] > 0

    def test_total_equals_trainable(self, head: ProjectionHead) -> None:
        """All parameters are trainable by default."""
        counts = head.parameter_count()
        assert counts["total"] == counts["trainable"]

    def test_matches_manual_count(self, head: ProjectionHead) -> None:
        """parameter_count() matches manual enumeration."""
        counts = head.parameter_count()
        manual_total = sum(p.numel() for p in head.parameters())
        assert counts["total"] == manual_total

    def test_frozen_changes_trainable(
        self, default_config: ProjectionHeadConfig
    ) -> None:
        """Freezing parameters changes trainable count."""
        head = ProjectionHead(default_config)
        for param in head.parameters():
            param.requires_grad_(False)
        counts = head.parameter_count()
        assert counts["trainable"] == 0
        assert counts["total"] > 0


# ======================================================================
# ProjectionHead — Gradient propagation
# ======================================================================


class TestProjectionHeadGradient:
    """Tests for gradient flow through the projection head."""

    def test_gradients_flow(
        self, default_config: ProjectionHeadConfig
    ) -> None:
        """Gradients flow from output back to input."""
        head = ProjectionHead(default_config)
        head.train()
        x = torch.randn(
            2, _DEFAULT_INPUT_DIM, requires_grad=True
        )
        z = head(x)
        z.sum().backward()
        assert x.grad is not None
        assert x.grad.shape == x.shape

    def test_model_params_receive_grads(
        self, default_config: ProjectionHeadConfig
    ) -> None:
        """All model parameters receive gradients."""
        head = ProjectionHead(default_config)
        head.train()
        x = torch.randn(2, _DEFAULT_INPUT_DIM)
        z = head(x)
        z.sum().backward()
        for name, param in head.named_parameters():
            assert param.grad is not None, (
                f"Parameter '{name}' has no gradient"
            )
            assert param.grad.shape == param.shape

    def test_grad_shapes_match_params(
        self, default_config: ProjectionHeadConfig
    ) -> None:
        """Gradient shapes match parameter shapes."""
        head = ProjectionHead(default_config)
        head.train()
        x = torch.randn(2, _DEFAULT_INPUT_DIM)
        z = head(x)
        z.sum().backward()
        for name, param in head.named_parameters():
            assert param.grad.shape == param.shape, (
                f"Gradient shape mismatch for '{name}'"
            )


# ======================================================================
# ProjectionHead — Serialisation
# ======================================================================


class TestProjectionHeadSerialisation:
    """Tests for save/load round-tripping."""

    def test_state_dict_round_trip(
        self,
        head: ProjectionHead,
        sample_embedding: torch.Tensor,
        default_config: ProjectionHeadConfig,
    ) -> None:
        """Save and reload via state_dict produces identical output."""
        head.eval()
        with torch.no_grad():
            z_orig = head(sample_embedding)

        state = copy.deepcopy(head.state_dict())

        new_head = ProjectionHead(default_config)
        new_head.load_state_dict(state)
        new_head.eval()

        with torch.no_grad():
            z_loaded = new_head(sample_embedding)

        torch.testing.assert_close(z_orig, z_loaded, atol=1e-6, rtol=1e-6)

    def test_torch_save_load(
        self,
        head: ProjectionHead,
        sample_embedding: torch.Tensor,
        default_config: ProjectionHeadConfig,
    ) -> None:
        """Full torch.save / torch.load round-trip preserves output."""
        head.eval()
        with torch.no_grad():
            z_orig = head(sample_embedding)

        with tempfile.NamedTemporaryFile(
            suffix=".pt", delete=False
        ) as tmp:
            torch.save(head.state_dict(), tmp.name)
            tmp_path = Path(tmp.name)

        try:
            loaded_state = torch.load(
                tmp_path, map_location="cpu", weights_only=True
            )
            new_head = ProjectionHead(default_config)
            new_head.load_state_dict(loaded_state)
            new_head.eval()
            with torch.no_grad():
                z_loaded = new_head(sample_embedding)
            torch.testing.assert_close(
                z_orig, z_loaded, atol=1e-6, rtol=1e-6
            )
        finally:
            tmp_path.unlink(missing_ok=True)

    def test_state_dict_keys_stable(
        self, head: ProjectionHead
    ) -> None:
        """State dict key names are non-empty and consistent."""
        keys = list(head.state_dict().keys())
        assert len(keys) > 0
        for k in keys:
            assert isinstance(k, str)
            assert len(k) > 0


# ======================================================================
# ProjectionHead — TorchScript
# ======================================================================


class TestProjectionHeadTorchScript:
    """Tests for TorchScript compatibility via tracing."""

    def test_torch_jit_trace(
        self,
        head: ProjectionHead,
        sample_embedding: torch.Tensor,
    ) -> None:
        """ProjectionHead can be compiled with torch.jit.trace."""
        head.eval()
        traced = torch.jit.trace(head, (sample_embedding,))
        with torch.no_grad():
            z = traced(sample_embedding)
        assert z.shape == (_DEFAULT_BATCH_SIZE, _DEFAULT_OUTPUT_DIM)

    def test_traced_output_matches_eager(
        self,
        head: ProjectionHead,
        sample_embedding: torch.Tensor,
    ) -> None:
        """Traced model produces the same output as the eager model."""
        head.eval()
        traced = torch.jit.trace(head, (sample_embedding,))
        with torch.no_grad():
            z_eager = head(sample_embedding)
            z_traced = traced(sample_embedding)
        torch.testing.assert_close(
            z_eager, z_traced, atol=1e-6, rtol=1e-6
        )

    def test_traced_l2_normalised(
        self,
        head: ProjectionHead,
        sample_embedding: torch.Tensor,
    ) -> None:
        """Traced model output is L2-normalised."""
        head.eval()
        traced = torch.jit.trace(head, (sample_embedding,))
        with torch.no_grad():
            z = traced(sample_embedding)
        norms = torch.linalg.norm(z, dim=-1)
        torch.testing.assert_close(
            norms,
            torch.ones_like(norms),
            atol=1e-5,
            rtol=1e-5,
        )

    def test_traced_save_load(
        self,
        head: ProjectionHead,
        sample_embedding: torch.Tensor,
    ) -> None:
        """Traced model can be saved and reloaded."""
        head.eval()
        traced = torch.jit.trace(head, (sample_embedding,))

        with torch.no_grad():
            z_orig = traced(sample_embedding)

        with tempfile.NamedTemporaryFile(
            suffix=".pt", delete=False
        ) as tmp:
            torch.jit.save(traced, tmp.name)
            tmp_path = Path(tmp.name)

        try:
            loaded = torch.jit.load(str(tmp_path), map_location="cpu")
            with torch.no_grad():
                z_loaded = loaded(sample_embedding)
            torch.testing.assert_close(
                z_orig, z_loaded, atol=1e-6, rtol=1e-6
            )
        finally:
            tmp_path.unlink(missing_ok=True)


# ======================================================================
# ProjectionHead — Invalid inputs
# ======================================================================


class TestProjectionHeadInvalidInputs:
    """Tests for input validation and error messages."""

    def test_not_a_tensor(self, head: ProjectionHead) -> None:
        """Raise TypeError for non-tensor input."""
        with pytest.raises(TypeError, match="embedding.*torch.Tensor"):
            head(
                np.zeros((2, _DEFAULT_INPUT_DIM)),  # type: ignore[arg-type]
            )

    def test_none_input(self, head: ProjectionHead) -> None:
        """Raise TypeError for None input."""
        with pytest.raises(TypeError, match="embedding.*torch.Tensor"):
            head(None)  # type: ignore[arg-type]

    def test_list_input(self, head: ProjectionHead) -> None:
        """Raise TypeError for list input."""
        with pytest.raises(TypeError, match="embedding.*torch.Tensor"):
            head([[1.0, 2.0]])  # type: ignore[arg-type]

    def test_integer_dtype(self, head: ProjectionHead) -> None:
        """Raise ValueError for integer-typed tensor."""
        x = torch.randint(0, 10, (2, _DEFAULT_INPUT_DIM))
        with pytest.raises(ValueError, match="embedding.*floating-point"):
            head(x)

    def test_wrong_ndim_1d(self, head: ProjectionHead) -> None:
        """Raise ValueError for 1-D input."""
        x = torch.randn(_DEFAULT_INPUT_DIM)
        with pytest.raises(ValueError, match="embedding.*exactly 2 dimensions"):
            head(x)

    def test_wrong_ndim_3d(self, head: ProjectionHead) -> None:
        """Raise ValueError for 3-D input."""
        x = torch.randn(2, 10, _DEFAULT_INPUT_DIM)
        with pytest.raises(ValueError, match="embedding.*exactly 2 dimensions"):
            head(x)

    def test_wrong_ndim_4d(self, head: ProjectionHead) -> None:
        """Raise ValueError for 4-D input."""
        x = torch.randn(2, 3, 10, _DEFAULT_INPUT_DIM)
        with pytest.raises(ValueError, match="embedding.*exactly 2 dimensions"):
            head(x)

    def test_wrong_embedding_dim(self, head: ProjectionHead) -> None:
        """Raise ValueError when embedding dim doesn't match config."""
        x = torch.randn(2, _DEFAULT_INPUT_DIM + 10)
        with pytest.raises(ValueError, match="embedding dimension.*must be"):
            head(x)

    def test_wrong_embedding_dim_smaller(self, head: ProjectionHead) -> None:
        """Raise ValueError when embedding dim is smaller than expected."""
        x = torch.randn(2, _DEFAULT_INPUT_DIM - 50)
        with pytest.raises(ValueError, match="embedding dimension.*must be"):
            head(x)

    def test_nan_input(self, head: ProjectionHead) -> None:
        """Raise ValueError when input contains NaN."""
        x = torch.randn(2, _DEFAULT_INPUT_DIM)
        x[0, 3] = float("nan")
        with pytest.raises(ValueError, match="embedding.*NaN"):
            head(x)

    def test_inf_input(self, head: ProjectionHead) -> None:
        """Raise ValueError when input contains Inf."""
        x = torch.randn(2, _DEFAULT_INPUT_DIM)
        x[0, 5] = float("inf")
        with pytest.raises(ValueError, match="embedding.*Inf"):
            head(x)

    def test_negative_inf_input(self, head: ProjectionHead) -> None:
        """Raise ValueError when input contains -Inf."""
        x = torch.randn(2, _DEFAULT_INPUT_DIM)
        x[1, 0] = float("-inf")
        with pytest.raises(ValueError, match="embedding.*Inf"):
            head(x)


# ======================================================================
# ProjectionHead — CPU
# ======================================================================


class TestProjectionHeadCPU:
    """CPU-specific compatibility tests."""

    def test_cpu_forward(
        self,
        head: ProjectionHead,
        sample_embedding: torch.Tensor,
    ) -> None:
        """Forward pass runs on CPU without error."""
        assert sample_embedding.device.type == "cpu"
        with torch.no_grad():
            z = head(sample_embedding)
        assert z.device.type == "cpu"
        assert z.shape == (_DEFAULT_BATCH_SIZE, _DEFAULT_OUTPUT_DIM)

    def test_cpu_l2_normalised(
        self,
        head: ProjectionHead,
        sample_embedding: torch.Tensor,
    ) -> None:
        """CPU embeddings are L2-normalised."""
        with torch.no_grad():
            z = head(sample_embedding)
        norms = torch.linalg.norm(z, dim=-1)
        torch.testing.assert_close(
            norms,
            torch.ones_like(norms),
            atol=1e-5,
            rtol=1e-5,
        )


# ======================================================================
# ProjectionHead — CUDA
# ======================================================================


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA not available",
)
class TestProjectionHeadCUDA:
    """CUDA-specific compatibility tests (skipped if no GPU)."""

    def test_cuda_forward(
        self, default_config: ProjectionHeadConfig
    ) -> None:
        """Forward pass runs on CUDA and returns a CUDA tensor."""
        device = torch.device("cuda")
        model = ProjectionHead(default_config).to(device).eval()
        x = torch.randn(
            _DEFAULT_BATCH_SIZE, _DEFAULT_INPUT_DIM, device=device
        )
        with torch.no_grad():
            z = model(x)
        assert z.device.type == "cuda"
        assert z.shape == (_DEFAULT_BATCH_SIZE, _DEFAULT_OUTPUT_DIM)

    def test_cuda_l2_normalised(
        self, default_config: ProjectionHeadConfig
    ) -> None:
        """CUDA embeddings are L2-normalised."""
        device = torch.device("cuda")
        model = ProjectionHead(default_config).to(device).eval()
        x = torch.randn(
            _DEFAULT_BATCH_SIZE, _DEFAULT_INPUT_DIM, device=device
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
        self, default_config: ProjectionHeadConfig
    ) -> None:
        """Gradient flow works on CUDA."""
        device = torch.device("cuda")
        model = ProjectionHead(default_config).to(device).train()
        x = torch.randn(
            2, _DEFAULT_INPUT_DIM,
            device=device, requires_grad=True,
        )
        z = model(x)
        z.sum().backward()
        assert x.grad is not None
        assert x.grad.shape == x.shape


# ======================================================================
# ProjectionHead — Float64 compatibility
# ======================================================================


class TestProjectionHeadFloat64:
    """Ensure the projection head works with float64 input."""

    def test_float64_forward(
        self, default_config: ProjectionHeadConfig
    ) -> None:
        """Forward works when both model and inputs are float64."""
        model = ProjectionHead(default_config).double().eval()
        x = torch.randn(
            2, _DEFAULT_INPUT_DIM, dtype=torch.float64
        )
        with torch.no_grad():
            z = model(x)
        assert z.dtype == torch.float64
        assert z.shape == (2, _DEFAULT_OUTPUT_DIM)

    def test_float64_l2_normalised(
        self, default_config: ProjectionHeadConfig
    ) -> None:
        """Float64 embeddings are L2-normalised."""
        model = ProjectionHead(default_config).double().eval()
        x = torch.randn(
            2, _DEFAULT_INPUT_DIM, dtype=torch.float64
        )
        with torch.no_grad():
            z = model(x)
        norms = torch.linalg.norm(z, dim=-1)
        torch.testing.assert_close(
            norms,
            torch.ones_like(norms),
            atol=1e-7,
            rtol=1e-7,
        )
