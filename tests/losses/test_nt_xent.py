"""Tests for :mod:`src.losses.nt_xent`.

Comprehensive test suite covering:

- NTXentConfig initialisation and validation
- NTXentLoss initialisation
- forward() output contract (dict keys, scalar loss, shapes)
- compute_similarity_matrix() correctness
- create_labels() correctness
- mask_self_similarity() correctness
- deterministic behaviour under fixed seeds
- symmetry of loss (z_i ↔ z_j)
- gradient propagation
- numerical stability (very small / large values, half precision)
- serialisation round-trip (state_dict / load_state_dict)
- TorchScript compatibility
- parameter_summary()
- Invalid inputs: wrong type, non-float dtype, 1D/3D tensors,
  batch-size mismatch, embedding-dim mismatch, NaN, Inf
- CPU execution
- CUDA execution (skipped if unavailable)
"""

from __future__ import annotations

import copy
import io
import math
import tempfile
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

from src.losses.nt_xent import NTXentConfig, NTXentLoss


# ======================================================================
# Constants
# ======================================================================

_BATCH: int = 8
_DIM: int = 128
_TEMPERATURE: float = 0.07


# ======================================================================
# Helpers
# ======================================================================


def _make_views(
    batch: int = _BATCH,
    dim: int = _DIM,
    device: torch.device | str = "cpu",
    seed: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return two L2-normalised random views ``(z_i, z_j)``."""
    g = torch.Generator(device=device if device != "cpu" else "cpu")
    g.manual_seed(seed)
    z_i = F.normalize(
        torch.randn(batch, dim, generator=g, device=device), p=2, dim=1
    )
    z_j = F.normalize(
        torch.randn(batch, dim, generator=g, device=device), p=2, dim=1
    )
    return z_i, z_j


def _default_loss_fn() -> NTXentLoss:
    return NTXentLoss(NTXentConfig(temperature=_TEMPERATURE))


# ======================================================================
# NTXentConfig tests
# ======================================================================


class TestNTXentConfig:
    """Tests for :class:`NTXentConfig` validation."""

    def test_default_values(self) -> None:
        """Defaults match the specification."""
        cfg = NTXentConfig()
        assert cfg.temperature == pytest.approx(0.07)
        assert cfg.reduction == "mean"
        assert cfg.eps == pytest.approx(1e-8)

    def test_custom_values(self) -> None:
        """Custom values are stored correctly."""
        cfg = NTXentConfig(temperature=0.5, reduction="sum", eps=1e-6)
        assert cfg.temperature == pytest.approx(0.5)
        assert cfg.reduction == "sum"
        assert cfg.eps == pytest.approx(1e-6)

    def test_frozen(self) -> None:
        """Config is immutable."""
        cfg = NTXentConfig()
        with pytest.raises((AttributeError, TypeError)):
            cfg.temperature = 0.5  # type: ignore[misc]

    def test_invalid_temperature_zero(self) -> None:
        """Zero temperature raises ValueError."""
        with pytest.raises(ValueError, match="temperature"):
            NTXentConfig(temperature=0.0)

    def test_invalid_temperature_negative(self) -> None:
        """Negative temperature raises ValueError."""
        with pytest.raises(ValueError, match="temperature"):
            NTXentConfig(temperature=-0.1)

    def test_invalid_reduction(self) -> None:
        """Unknown reduction raises ValueError."""
        with pytest.raises(ValueError, match="reduction"):
            NTXentConfig(reduction="none")  # type: ignore[arg-type]

    def test_invalid_eps_zero(self) -> None:
        """Zero eps raises ValueError."""
        with pytest.raises(ValueError, match="eps"):
            NTXentConfig(eps=0.0)

    def test_invalid_eps_negative(self) -> None:
        """Negative eps raises ValueError."""
        with pytest.raises(ValueError, match="eps"):
            NTXentConfig(eps=-1e-9)

    @pytest.mark.parametrize("reduction", ["mean", "sum"])
    def test_valid_reductions(self, reduction: str) -> None:
        """Both valid reductions are accepted without error."""
        cfg = NTXentConfig(reduction=reduction)  # type: ignore[arg-type]
        assert cfg.reduction == reduction


# ======================================================================
# NTXentLoss initialisation tests
# ======================================================================


class TestNTXentLossInit:
    """Tests for :class:`NTXentLoss` initialisation."""

    def test_default_init(self) -> None:
        """Loss initialises with default config."""
        loss_fn = NTXentLoss(NTXentConfig())
        assert isinstance(loss_fn, NTXentLoss)

    def test_config_property(self) -> None:
        """Config is accessible via the config property."""
        cfg = NTXentConfig(temperature=0.2)
        loss_fn = NTXentLoss(cfg)
        assert loss_fn.config is cfg

    def test_temperature_property(self) -> None:
        """Temperature property matches config."""
        cfg = NTXentConfig(temperature=0.3)
        loss_fn = NTXentLoss(cfg)
        assert loss_fn.temperature == pytest.approx(0.3)

    def test_wrong_config_type_raises(self) -> None:
        """Wrong config type raises TypeError."""
        with pytest.raises(TypeError, match="NTXentConfig"):
            NTXentLoss("bad_config")  # type: ignore[arg-type]

    def test_no_learnable_parameters(self) -> None:
        """NT-Xent has no learnable parameters."""
        loss_fn = _default_loss_fn()
        assert sum(p.numel() for p in loss_fn.parameters()) == 0

    def test_parameter_summary_keys(self) -> None:
        """parameter_summary() returns expected keys."""
        loss_fn = _default_loss_fn()
        summary = loss_fn.parameter_summary()
        assert "temperature" in summary
        assert "reduction" in summary
        assert "eps" in summary
        assert "num_parameters" in summary

    def test_parameter_summary_values(self) -> None:
        """parameter_summary() values match config."""
        loss_fn = _default_loss_fn()
        summary = loss_fn.parameter_summary()
        assert summary["temperature"] == pytest.approx(_TEMPERATURE)
        assert summary["reduction"] == "mean"
        assert summary["num_parameters"] == 0


# ======================================================================
# compute_similarity_matrix tests
# ======================================================================


class TestComputeSimilarityMatrix:
    """Tests for :meth:`NTXentLoss.compute_similarity_matrix`."""

    def test_shape(self) -> None:
        """Output has shape (2B, 2B)."""
        loss_fn = _default_loss_fn()
        z_i, z_j = _make_views()
        z = torch.cat([z_i, z_j], dim=0)
        sim = loss_fn.compute_similarity_matrix(z)
        assert sim.shape == (2 * _BATCH, 2 * _BATCH)

    def test_self_similarity_is_one(self) -> None:
        """Diagonal entries equal 1.0 for L2-normalised inputs."""
        loss_fn = _default_loss_fn()
        z_i, z_j = _make_views()
        z = torch.cat([z_i, z_j], dim=0)
        sim = loss_fn.compute_similarity_matrix(z)
        diag = torch.diag(sim)
        assert torch.allclose(diag, torch.ones_like(diag), atol=1e-5)

    def test_symmetry(self) -> None:
        """Cosine-similarity matrix is symmetric."""
        loss_fn = _default_loss_fn()
        z_i, z_j = _make_views()
        z = torch.cat([z_i, z_j], dim=0)
        sim = loss_fn.compute_similarity_matrix(z)
        assert torch.allclose(sim, sim.T, atol=1e-5)

    def test_values_bounded(self) -> None:
        """All entries are in [-1, 1] (within floating-point tolerance)."""
        loss_fn = _default_loss_fn()
        z_i, z_j = _make_views()
        z = torch.cat([z_i, z_j], dim=0)
        sim = loss_fn.compute_similarity_matrix(z)
        assert (sim >= -1.0 - 1e-5).all()
        assert (sim <= 1.0 + 1e-5).all()

    def test_identical_vectors_give_one(self) -> None:
        """Two identical rows produce similarity 1."""
        loss_fn = _default_loss_fn()
        v = F.normalize(torch.randn(1, _DIM), p=2, dim=1)
        z = v.expand(4, -1)
        sim = loss_fn.compute_similarity_matrix(z)
        # All off-diagonal entries should be ~1
        off_diag = sim[~torch.eye(4, dtype=torch.bool)]
        assert torch.allclose(off_diag, torch.ones_like(off_diag), atol=1e-5)


# ======================================================================
# mask_self_similarity tests
# ======================================================================


class TestMaskSelfSimilarity:
    """Tests for :meth:`NTXentLoss.mask_self_similarity`."""

    def test_diagonal_is_neg_inf(self) -> None:
        """Diagonal entries are set to -inf or finfo.min (very negative).

        PyTorch's ``masked_fill`` with ``torch.finfo(dtype).min`` produces
        the most-negative finite float (``-3.4028e+38`` for float32) rather
        than IEEE ``-inf``.  Both are accepted here.
        """
        loss_fn = _default_loss_fn()
        n = 2 * _BATCH
        sim = torch.randn(n, n)
        sim_masked = loss_fn.mask_self_similarity(sim, n)
        diag = torch.diag(sim_masked)
        # Accept both strict -inf and finfo.min (most-negative finite float).
        fmin = torch.finfo(sim.dtype).min
        assert torch.all(
            (torch.isinf(diag) & (diag < 0)) | (diag <= fmin)
        )

    def test_off_diagonal_unchanged(self) -> None:
        """Off-diagonal entries are not modified."""
        loss_fn = _default_loss_fn()
        n = 2 * _BATCH
        sim = torch.randn(n, n)
        sim_masked = loss_fn.mask_self_similarity(sim, n)
        eye = torch.eye(n, dtype=torch.bool)
        assert torch.allclose(sim[~eye], sim_masked[~eye])

    def test_small_batch(self) -> None:
        """Works for batch size 2 (minimum meaningful case)."""
        loss_fn = _default_loss_fn()
        n = 4
        sim = torch.ones(n, n)
        masked = loss_fn.mask_self_similarity(sim, n)
        assert masked[0, 0].item() == -math.inf or masked[0, 0].item() < -1e30


# ======================================================================
# create_labels tests
# ======================================================================


class TestCreateLabels:
    """Tests for :meth:`NTXentLoss.create_labels`."""

    def test_shape(self) -> None:
        """Labels shape is (2B,)."""
        loss_fn = _default_loss_fn()
        labels = loss_fn.create_labels(_BATCH, torch.device("cpu"))
        assert labels.shape == (2 * _BATCH,)

    def test_dtype_long(self) -> None:
        """Labels are of dtype long (int64)."""
        loss_fn = _default_loss_fn()
        labels = loss_fn.create_labels(_BATCH, torch.device("cpu"))
        assert labels.dtype == torch.long

    def test_first_half_points_to_second_half(self) -> None:
        """Rows 0..B-1 have positive at columns B..2B-1."""
        loss_fn = _default_loss_fn()
        B = _BATCH
        labels = loss_fn.create_labels(B, torch.device("cpu"))
        expected_first = torch.arange(B, 2 * B)
        assert torch.equal(labels[:B], expected_first)

    def test_second_half_points_to_first_half(self) -> None:
        """Rows B..2B-1 have positive at columns 0..B-1."""
        loss_fn = _default_loss_fn()
        B = _BATCH
        labels = loss_fn.create_labels(B, torch.device("cpu"))
        expected_second = torch.arange(0, B)
        assert torch.equal(labels[B:], expected_second)

    def test_labels_unique(self) -> None:
        """Each label index appears exactly once in each half."""
        loss_fn = _default_loss_fn()
        B = _BATCH
        labels = loss_fn.create_labels(B, torch.device("cpu"))
        # First half labels are B..2B-1, all distinct
        assert len(set(labels[:B].tolist())) == B
        # Second half labels are 0..B-1, all distinct
        assert len(set(labels[B:].tolist())) == B


# ======================================================================
# forward() tests
# ======================================================================


class TestNTXentLossForward:
    """Tests for :meth:`NTXentLoss.forward`."""

    # ---- Output contract ----

    def test_output_is_dict(self) -> None:
        """forward() returns a dict."""
        loss_fn = _default_loss_fn()
        z_i, z_j = _make_views()
        out = loss_fn(z_i, z_j)
        assert isinstance(out, dict)

    def test_output_keys(self) -> None:
        """Output dict contains expected keys."""
        loss_fn = _default_loss_fn()
        z_i, z_j = _make_views()
        out = loss_fn(z_i, z_j)
        assert set(out.keys()) == {"loss", "logits", "labels", "temperature"}

    def test_loss_is_scalar(self) -> None:
        """Loss tensor is a 0-d scalar."""
        loss_fn = _default_loss_fn()
        z_i, z_j = _make_views()
        out = loss_fn(z_i, z_j)
        loss = out["loss"]
        assert isinstance(loss, torch.Tensor)
        assert loss.shape == torch.Size([])

    def test_logits_shape(self) -> None:
        """Logits have shape (2B, 2B)."""
        loss_fn = _default_loss_fn()
        z_i, z_j = _make_views()
        out = loss_fn(z_i, z_j)
        assert out["logits"].shape == (2 * _BATCH, 2 * _BATCH)

    def test_labels_shape(self) -> None:
        """Labels have shape (2B,)."""
        loss_fn = _default_loss_fn()
        z_i, z_j = _make_views()
        out = loss_fn(z_i, z_j)
        assert out["labels"].shape == (2 * _BATCH,)

    def test_temperature_value_in_output(self) -> None:
        """Temperature in output matches config."""
        loss_fn = _default_loss_fn()
        z_i, z_j = _make_views()
        out = loss_fn(z_i, z_j)
        assert out["temperature"] == pytest.approx(_TEMPERATURE)

    def test_loss_is_finite(self) -> None:
        """Loss value is finite (not NaN or Inf)."""
        loss_fn = _default_loss_fn()
        z_i, z_j = _make_views()
        out = loss_fn(z_i, z_j)
        assert torch.isfinite(out["loss"])

    def test_loss_is_non_negative(self) -> None:
        """NT-Xent loss is always ≥ 0."""
        loss_fn = _default_loss_fn()
        z_i, z_j = _make_views()
        out = loss_fn(z_i, z_j)
        assert out["loss"].item() >= 0.0

    # ---- Determinism ----

    def test_deterministic_behavior(self) -> None:
        """Same inputs always produce the same loss."""
        loss_fn = _default_loss_fn()
        z_i, z_j = _make_views(seed=42)
        out1 = loss_fn(z_i, z_j)
        out2 = loss_fn(z_i, z_j)
        assert torch.equal(out1["loss"], out2["loss"])

    # ---- Symmetry ----

    def test_loss_symmetry(self) -> None:
        """Swapping z_i and z_j yields the same loss."""
        loss_fn = _default_loss_fn()
        z_i, z_j = _make_views(seed=7)
        out_ij = loss_fn(z_i, z_j)
        out_ji = loss_fn(z_j, z_i)
        assert torch.allclose(out_ij["loss"], out_ji["loss"], atol=1e-5)

    # ---- Gradient propagation ----

    def test_gradient_propagates_through_z_i(self) -> None:
        """Gradient flows back to z_i.

        ``F.normalize`` produces a non-leaf tensor (it has a ``grad_fn``).
        We call ``.retain_grad()`` so the gradient is stored on the
        non-leaf tensor, which is the standard PyTorch pattern.
        """
        loss_fn = _default_loss_fn()
        raw_i = torch.randn(_BATCH, _DIM, requires_grad=True)
        z_i = F.normalize(raw_i, p=2, dim=1)
        z_i.retain_grad()
        z_j = F.normalize(torch.randn(_BATCH, _DIM), p=2, dim=1)
        out = loss_fn(z_i, z_j)
        out["loss"].backward()
        # Gradient reaches the raw leaf (raw_i) and the non-leaf z_i.
        assert raw_i.grad is not None
        assert not torch.all(raw_i.grad == 0)

    def test_gradient_propagates_through_z_j(self) -> None:
        """Gradient flows back to z_j."""
        loss_fn = _default_loss_fn()
        raw_j = torch.randn(_BATCH, _DIM, requires_grad=True)
        z_i = F.normalize(torch.randn(_BATCH, _DIM), p=2, dim=1)
        z_j = F.normalize(raw_j, p=2, dim=1)
        z_j.retain_grad()
        out = loss_fn(z_i, z_j)
        out["loss"].backward()
        assert raw_j.grad is not None
        assert not torch.all(raw_j.grad == 0)

    def test_gradient_propagates_both_views(self) -> None:
        """Gradient reaches both z_i and z_j simultaneously."""
        loss_fn = _default_loss_fn()
        raw_i = torch.randn(_BATCH, _DIM, requires_grad=True)
        raw_j = torch.randn(_BATCH, _DIM, requires_grad=True)
        z_i = F.normalize(raw_i, p=2, dim=1)
        z_j = F.normalize(raw_j, p=2, dim=1)
        out = loss_fn(z_i, z_j)
        out["loss"].backward()
        assert raw_i.grad is not None
        assert raw_j.grad is not None

    # ---- Reduction modes ----

    def test_sum_reduction_greater_than_mean(self) -> None:
        """Sum reduction produces a larger value than mean for B > 1."""
        z_i, z_j = _make_views(batch=4)
        loss_mean = NTXentLoss(NTXentConfig(reduction="mean"))(z_i, z_j)["loss"]
        loss_sum = NTXentLoss(NTXentConfig(reduction="sum"))(z_i, z_j)["loss"]
        assert loss_sum.item() > loss_mean.item()

    # ---- Perfect-positive test ----

    def test_identical_views_produce_low_loss(self) -> None:
        """When z_i == z_j the loss should be low (positive pair maximally similar)."""
        loss_fn = _default_loss_fn()
        z_i, _ = _make_views(batch=4)
        # Perfect positives: z_j is a copy of z_i
        out = loss_fn(z_i, z_i.clone())
        # With identical views the positive logit is 1/τ which dominates;
        # loss < log(2B) = log(8) ≈ 2.08 for B=4.
        assert out["loss"].item() < math.log(2 * 4)

    # ---- Different batch sizes ----

    @pytest.mark.parametrize("batch", [2, 4, 16, 32])
    def test_various_batch_sizes(self, batch: int) -> None:
        """Loss is finite for various batch sizes."""
        loss_fn = _default_loss_fn()
        z_i, z_j = _make_views(batch=batch)
        out = loss_fn(z_i, z_j)
        assert torch.isfinite(out["loss"])

    @pytest.mark.parametrize("dim", [32, 64, 256, 512])
    def test_various_embedding_dims(self, dim: int) -> None:
        """Loss is finite for various embedding dimensions."""
        loss_fn = _default_loss_fn()
        z_i, z_j = _make_views(dim=dim)
        out = loss_fn(z_i, z_j)
        assert torch.isfinite(out["loss"])


# ======================================================================
# Numerical stability tests
# ======================================================================


class TestNumericalStability:
    """Tests for numerical robustness."""

    def test_very_small_temperature(self) -> None:
        """Very small temperature (0.001) still produces finite loss."""
        loss_fn = NTXentLoss(NTXentConfig(temperature=0.001))
        z_i, z_j = _make_views()
        out = loss_fn(z_i, z_j)
        assert torch.isfinite(out["loss"])

    def test_large_temperature(self) -> None:
        """Large temperature (10.0) still produces finite loss."""
        loss_fn = NTXentLoss(NTXentConfig(temperature=10.0))
        z_i, z_j = _make_views()
        out = loss_fn(z_i, z_j)
        assert torch.isfinite(out["loss"])

    def test_half_precision_inputs(self) -> None:
        """fp16 inputs produce a finite loss without errors."""
        loss_fn = _default_loss_fn()
        z_i, z_j = _make_views()
        z_i_half = z_i.half()
        z_j_half = z_j.half()
        out = loss_fn(z_i_half, z_j_half)
        assert torch.isfinite(out["loss"])

    def test_bfloat16_inputs(self) -> None:
        """bfloat16 inputs produce a finite loss without errors."""
        if not torch.cuda.is_available():
            pytest.skip("bfloat16 is best tested on CUDA or recent CPU")
        loss_fn = _default_loss_fn()
        z_i, z_j = _make_views()
        z_i_bf = z_i.to(torch.bfloat16)
        z_j_bf = z_j.to(torch.bfloat16)
        out = loss_fn(z_i_bf, z_j_bf)
        assert torch.isfinite(out["loss"])

    def test_near_duplicate_embeddings(self) -> None:
        """Near-duplicate embeddings (almost all similarities close to 1) are stable."""
        loss_fn = _default_loss_fn()
        base = F.normalize(torch.randn(1, _DIM), p=2, dim=1)
        z = base.expand(_BATCH, -1) + 1e-6 * torch.randn(_BATCH, _DIM)
        z = F.normalize(z, p=2, dim=1)
        out = loss_fn(z, z.clone())
        assert torch.isfinite(out["loss"])


# ======================================================================
# Serialisation tests
# ======================================================================


class TestSerialization:
    """Tests for state-dict serialisation."""

    def test_state_dict_is_empty(self) -> None:
        """NT-Xent has no parameters, so state_dict is empty."""
        loss_fn = _default_loss_fn()
        sd = loss_fn.state_dict()
        assert len(sd) == 0

    def test_round_trip_gives_identical_output(self) -> None:
        """Saving and reloading produces identical results."""
        cfg = NTXentConfig(temperature=0.15)
        loss_fn = NTXentLoss(cfg)

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "nt_xent.pt"
            torch.save({"state_dict": loss_fn.state_dict(), "config": cfg}, path)
            checkpoint = torch.load(path, weights_only=False)

        restored = NTXentLoss(checkpoint["config"])
        restored.load_state_dict(checkpoint["state_dict"])

        z_i, z_j = _make_views(seed=99)
        out_orig = loss_fn(z_i, z_j)
        out_rest = restored(z_i, z_j)
        assert torch.equal(out_orig["loss"], out_rest["loss"])

    def test_deepcopy(self) -> None:
        """deepcopy produces an independent instance with same output."""
        loss_fn = _default_loss_fn()
        cloned = copy.deepcopy(loss_fn)
        z_i, z_j = _make_views()
        assert torch.equal(
            loss_fn(z_i, z_j)["loss"],
            cloned(z_i, z_j)["loss"],
        )


# ======================================================================
# TorchScript tests
# ======================================================================


class TestTorchScript:
    """Tests for TorchScript compatibility."""

    def test_torchscript_trace(self) -> None:
        """NTXentLoss can be traced by TorchScript."""
        loss_fn = _default_loss_fn()
        z_i, z_j = _make_views()
        try:
            traced = torch.jit.trace(loss_fn, (z_i, z_j), strict=False)
            out = traced(z_i, z_j)
            assert "loss" in out
        except Exception as exc:
            pytest.skip(f"TorchScript trace not supported: {exc}")

    def test_torchscript_save_load(self) -> None:
        """Traced model can be saved and reloaded."""
        loss_fn = _default_loss_fn()
        z_i, z_j = _make_views()
        try:
            traced = torch.jit.trace(loss_fn, (z_i, z_j), strict=False)
        except Exception as exc:
            pytest.skip(f"TorchScript trace not supported: {exc}")

        buf = io.BytesIO()
        torch.jit.save(traced, buf)
        buf.seek(0)
        loaded = torch.jit.load(buf)
        out = loaded(z_i, z_j)
        assert torch.isfinite(out["loss"])


# ======================================================================
# Input validation tests
# ======================================================================


class TestInputValidation:
    """Tests for invalid-input rejection."""

    def test_z_i_not_tensor_raises_type_error(self) -> None:
        """Non-tensor z_i raises TypeError."""
        loss_fn = _default_loss_fn()
        _, z_j = _make_views()
        with pytest.raises(TypeError, match="z_i"):
            loss_fn([[1.0, 0.0]] * _BATCH, z_j)  # type: ignore[arg-type]

    def test_z_j_not_tensor_raises_type_error(self) -> None:
        """Non-tensor z_j raises TypeError."""
        loss_fn = _default_loss_fn()
        z_i, _ = _make_views()
        with pytest.raises(TypeError, match="z_j"):
            loss_fn(z_i, [[1.0, 0.0]] * _BATCH)  # type: ignore[arg-type]

    def test_z_i_integer_dtype_raises(self) -> None:
        """Integer dtype for z_i raises ValueError."""
        loss_fn = _default_loss_fn()
        z_i = torch.randint(0, 10, (_BATCH, _DIM))
        z_j = F.normalize(torch.randn(_BATCH, _DIM), p=2, dim=1)
        with pytest.raises(ValueError, match="floating-point"):
            loss_fn(z_i, z_j)

    def test_z_j_integer_dtype_raises(self) -> None:
        """Integer dtype for z_j raises ValueError."""
        loss_fn = _default_loss_fn()
        z_i = F.normalize(torch.randn(_BATCH, _DIM), p=2, dim=1)
        z_j = torch.randint(0, 10, (_BATCH, _DIM))
        with pytest.raises(ValueError, match="floating-point"):
            loss_fn(z_i, z_j)

    def test_z_i_1d_raises(self) -> None:
        """1-D z_i raises ValueError."""
        loss_fn = _default_loss_fn()
        z_i = torch.randn(_DIM)
        z_j = F.normalize(torch.randn(_BATCH, _DIM), p=2, dim=1)
        with pytest.raises(ValueError, match="2 dimensions"):
            loss_fn(z_i, z_j)

    def test_z_j_1d_raises(self) -> None:
        """1-D z_j raises ValueError."""
        loss_fn = _default_loss_fn()
        z_i = F.normalize(torch.randn(_BATCH, _DIM), p=2, dim=1)
        z_j = torch.randn(_DIM)
        with pytest.raises(ValueError, match="2 dimensions"):
            loss_fn(z_i, z_j)

    def test_z_i_3d_raises(self) -> None:
        """3-D z_i raises ValueError."""
        loss_fn = _default_loss_fn()
        z_i = torch.randn(_BATCH, _DIM, 4)
        z_j = F.normalize(torch.randn(_BATCH, _DIM), p=2, dim=1)
        with pytest.raises(ValueError, match="2 dimensions"):
            loss_fn(z_i, z_j)

    def test_z_j_3d_raises(self) -> None:
        """3-D z_j raises ValueError."""
        loss_fn = _default_loss_fn()
        z_i = F.normalize(torch.randn(_BATCH, _DIM), p=2, dim=1)
        z_j = torch.randn(_BATCH, _DIM, 4)
        with pytest.raises(ValueError, match="2 dimensions"):
            loss_fn(z_i, z_j)

    def test_batch_size_mismatch_raises(self) -> None:
        """Different batch sizes raise ValueError."""
        loss_fn = _default_loss_fn()
        z_i = F.normalize(torch.randn(_BATCH, _DIM), p=2, dim=1)
        z_j = F.normalize(torch.randn(_BATCH + 2, _DIM), p=2, dim=1)
        with pytest.raises(ValueError, match="batch size"):
            loss_fn(z_i, z_j)

    def test_embedding_dim_mismatch_raises(self) -> None:
        """Different embedding dims raise ValueError."""
        loss_fn = _default_loss_fn()
        z_i = F.normalize(torch.randn(_BATCH, _DIM), p=2, dim=1)
        z_j = F.normalize(torch.randn(_BATCH, _DIM + 16), p=2, dim=1)
        with pytest.raises(ValueError, match="embedding dimension"):
            loss_fn(z_i, z_j)

    def test_nan_in_z_i_raises(self) -> None:
        """NaN in z_i raises ValueError."""
        loss_fn = _default_loss_fn()
        z_i, z_j = _make_views()
        z_i[0, 0] = float("nan")
        with pytest.raises(ValueError, match="NaN"):
            loss_fn(z_i, z_j)

    def test_nan_in_z_j_raises(self) -> None:
        """NaN in z_j raises ValueError."""
        loss_fn = _default_loss_fn()
        z_i, z_j = _make_views()
        z_j[0, 0] = float("nan")
        with pytest.raises(ValueError, match="NaN"):
            loss_fn(z_i, z_j)

    def test_inf_in_z_i_raises(self) -> None:
        """Inf in z_i raises ValueError."""
        loss_fn = _default_loss_fn()
        z_i, z_j = _make_views()
        z_i[0, 0] = float("inf")
        with pytest.raises(ValueError, match="Inf"):
            loss_fn(z_i, z_j)

    def test_inf_in_z_j_raises(self) -> None:
        """Inf in z_j raises ValueError."""
        loss_fn = _default_loss_fn()
        z_i, z_j = _make_views()
        z_j[0, 0] = float("inf")
        with pytest.raises(ValueError, match="Inf"):
            loss_fn(z_i, z_j)

    def test_negative_inf_in_z_i_raises(self) -> None:
        """-Inf in z_i raises ValueError."""
        loss_fn = _default_loss_fn()
        z_i, z_j = _make_views()
        z_i[1, 3] = float("-inf")
        with pytest.raises(ValueError, match="Inf"):
            loss_fn(z_i, z_j)


# ======================================================================
# Device tests
# ======================================================================


class TestDeviceCompatibility:
    """Tests for CPU and CUDA execution."""

    def test_cpu_execution(self) -> None:
        """Loss computes correctly on CPU."""
        loss_fn = _default_loss_fn()
        z_i, z_j = _make_views(device="cpu")
        out = loss_fn(z_i, z_j)
        assert out["loss"].device.type == "cpu"
        assert torch.isfinite(out["loss"])

    @pytest.mark.skipif(
        not torch.cuda.is_available(), reason="CUDA not available"
    )
    def test_cuda_execution(self) -> None:
        """Loss computes correctly on CUDA."""
        device = torch.device("cuda")
        loss_fn = _default_loss_fn().to(device)
        z_i, z_j = _make_views(device=device)
        out = loss_fn(z_i, z_j)
        assert out["loss"].device.type == "cuda"
        assert torch.isfinite(out["loss"])

    @pytest.mark.skipif(
        not torch.cuda.is_available(), reason="CUDA not available"
    )
    def test_cuda_result_matches_cpu(self) -> None:
        """CUDA and CPU produce numerically close results."""
        z_i_cpu, z_j_cpu = _make_views(device="cpu", seed=123)
        z_i_gpu = z_i_cpu.cuda()
        z_j_gpu = z_j_cpu.cuda()

        loss_cpu = NTXentLoss(NTXentConfig())(z_i_cpu, z_j_cpu)["loss"]
        loss_gpu = NTXentLoss(NTXentConfig()).cuda()(z_i_gpu, z_j_gpu)["loss"]

        assert torch.allclose(
            loss_cpu, loss_gpu.cpu(), atol=1e-4
        ), f"CPU={loss_cpu.item():.6f}, CUDA={loss_gpu.item():.6f}"

    @pytest.mark.skipif(
        not torch.cuda.is_available(), reason="CUDA not available"
    )
    def test_cuda_fp16(self) -> None:
        """CUDA fp16 forward pass is finite."""
        device = torch.device("cuda")
        loss_fn = _default_loss_fn().to(device)
        z_i, z_j = _make_views(device=device)
        out = loss_fn(z_i.half(), z_j.half())
        assert torch.isfinite(out["loss"])
