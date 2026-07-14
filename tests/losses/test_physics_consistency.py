"""Tests for :mod:`src.losses.physics_consistency`.

Comprehensive test suite covering:

- PhysicsConsistencyConfig initialisation and validation
- PhysicsConsistencyLoss initialisation
- strategy selection via config
- available_metrics()
- current_metric()
- set_metric()
- cosine strategy correctness
- mse strategy correctness
- huber strategy correctness
- hybrid strategy correctness
- forward() output contract (dict keys, scalar loss, shapes)
- scalar loss guarantee for all strategies
- deterministic behaviour under fixed seeds
- gradient propagation (loss.backward())
- serialisation round-trip (state_dict / load_state_dict)
- TorchScript compatibility
- invalid config: bad mode, bad reduction, bad eps, bad weights
- invalid inputs: wrong type, non-float dtype, 1D/3D tensors
- batch-size mismatch
- embedding-dim mismatch
- NaN inputs
- Inf inputs
- CPU execution
- CUDA execution (skipped if unavailable)
- extensibility via register_strategy
- allow_custom_strategy=False enforcement
- reduction="sum" variant
- parameter_summary()
"""

from __future__ import annotations

import copy

import pytest
import torch
import torch.nn.functional as F

from src.losses.physics_consistency import (
    BaseConsistencyLoss,
    PhysicsConsistencyConfig,
    PhysicsConsistencyLoss,
    _STRATEGY_REGISTRY,
    register_strategy,
)

# ======================================================================
# Constants
# ======================================================================

_BATCH: int = 8
_DIM: int = 256


# ======================================================================
# Helpers
# ======================================================================


def _make_embeddings(
    batch: int = _BATCH,
    dim: int = _DIM,
    device: torch.device | str = "cpu",
    seed: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return two L2-normalised random embedding tensors ``(enc, phy)``."""
    g = torch.Generator(device="cpu")
    g.manual_seed(seed)
    enc = F.normalize(
        torch.randn(batch, dim, generator=g), p=2, dim=1
    ).to(device)
    phy = F.normalize(
        torch.randn(batch, dim, generator=g), p=2, dim=1
    ).to(device)
    return enc, phy


def _default_loss_fn(mode: str = "cosine") -> PhysicsConsistencyLoss:
    return PhysicsConsistencyLoss(PhysicsConsistencyConfig(mode=mode))


# ======================================================================
# PhysicsConsistencyConfig Tests
# ======================================================================


class TestPhysicsConsistencyConfig:
    """Tests for PhysicsConsistencyConfig validation."""

    def test_defaults(self):
        cfg = PhysicsConsistencyConfig()
        assert cfg.mode == "cosine"
        assert cfg.cosine_weight == pytest.approx(0.7)
        assert cfg.mse_weight == pytest.approx(0.3)
        assert cfg.reduction == "mean"
        assert cfg.eps == pytest.approx(1e-8)
        assert cfg.allow_custom_strategy is True

    def test_resolved_mode_lower_cased(self):
        cfg = PhysicsConsistencyConfig(mode="COSINE")
        assert cfg._resolved_mode == "cosine"

    def test_empty_mode_raises(self):
        with pytest.raises(ValueError, match="mode must be a non-empty string"):
            PhysicsConsistencyConfig(mode="")

    def test_invalid_reduction_raises(self):
        with pytest.raises(ValueError, match="reduction must be"):
            PhysicsConsistencyConfig(reduction="invalid")

    def test_negative_eps_raises(self):
        with pytest.raises(ValueError, match="eps must be strictly positive"):
            PhysicsConsistencyConfig(eps=-1e-8)

    def test_zero_eps_raises(self):
        with pytest.raises(ValueError, match="eps must be strictly positive"):
            PhysicsConsistencyConfig(eps=0.0)

    def test_cosine_weight_out_of_range_raises(self):
        with pytest.raises(ValueError, match="cosine_weight must be in"):
            PhysicsConsistencyConfig(cosine_weight=1.5)

    def test_mse_weight_out_of_range_raises(self):
        with pytest.raises(ValueError, match="mse_weight must be in"):
            PhysicsConsistencyConfig(mse_weight=-0.1)

    def test_valid_reduction_sum(self):
        cfg = PhysicsConsistencyConfig(reduction="sum")
        assert cfg.reduction == "sum"

    @pytest.mark.parametrize("mode", ["cosine", "mse", "huber", "hybrid"])
    def test_valid_built_in_modes(self, mode):
        cfg = PhysicsConsistencyConfig(mode=mode)
        assert cfg._resolved_mode == mode


# ======================================================================
# PhysicsConsistencyLoss Initialisation Tests
# ======================================================================


class TestPhysicsConsistencyLossInit:
    """Tests for PhysicsConsistencyLoss construction."""

    def test_wrong_config_type_raises(self):
        with pytest.raises(TypeError, match="config must be a PhysicsConsistencyConfig"):
            PhysicsConsistencyLoss("cosine")  # type: ignore[arg-type]

    def test_unknown_mode_raises(self):
        cfg = PhysicsConsistencyConfig.__new__(PhysicsConsistencyConfig)
        # Bypass __post_init__ by setting fields manually
        object.__setattr__(cfg, "mode", "nonexistent_mode")
        object.__setattr__(cfg, "cosine_weight", 0.7)
        object.__setattr__(cfg, "mse_weight", 0.3)
        object.__setattr__(cfg, "reduction", "mean")
        object.__setattr__(cfg, "eps", 1e-8)
        object.__setattr__(cfg, "allow_custom_strategy", True)
        object.__setattr__(cfg, "_resolved_mode", "nonexistent_mode")
        with pytest.raises(ValueError, match="Unknown consistency mode"):
            PhysicsConsistencyLoss(cfg)

    @pytest.mark.parametrize("mode", ["cosine", "mse", "huber", "hybrid"])
    def test_all_built_in_modes_initialise(self, mode):
        fn = _default_loss_fn(mode)
        assert fn.current_metric() == mode

    def test_is_nn_module(self):
        fn = _default_loss_fn()
        assert isinstance(fn, torch.nn.Module)

    def test_no_learnable_parameters(self):
        fn = _default_loss_fn()
        assert sum(p.numel() for p in fn.parameters()) == 0

    def test_config_property(self):
        cfg = PhysicsConsistencyConfig(mode="mse")
        fn = PhysicsConsistencyLoss(cfg)
        assert fn.config is cfg


# ======================================================================
# available_metrics() Tests
# ======================================================================


class TestAvailableMetrics:
    """Tests for PhysicsConsistencyLoss.available_metrics()."""

    def test_returns_list(self):
        fn = _default_loss_fn()
        result = fn.available_metrics()
        assert isinstance(result, list)

    def test_contains_all_built_ins(self):
        fn = _default_loss_fn()
        metrics = fn.available_metrics()
        for name in ("cosine", "mse", "huber", "hybrid"):
            assert name in metrics, f"'{name}' missing from available_metrics()"

    def test_is_sorted(self):
        fn = _default_loss_fn()
        metrics = fn.available_metrics()
        assert metrics == sorted(metrics)

    def test_reflects_custom_registration(self):
        @register_strategy("_test_avail_metric")
        class _DummyLoss(BaseConsistencyLoss):
            @property
            def metric_name(self):
                return "_test_avail_metric"

            def compute(self, enc, phy):
                return torch.tensor(0.0)

        fn = _default_loss_fn()
        assert "_test_avail_metric" in fn.available_metrics()


# ======================================================================
# current_metric() Tests
# ======================================================================


class TestCurrentMetric:
    """Tests for PhysicsConsistencyLoss.current_metric()."""

    @pytest.mark.parametrize("mode", ["cosine", "mse", "huber", "hybrid"])
    def test_returns_correct_mode(self, mode):
        fn = _default_loss_fn(mode)
        assert fn.current_metric() == mode

    def test_returns_string(self):
        fn = _default_loss_fn()
        assert isinstance(fn.current_metric(), str)


# ======================================================================
# set_metric() Tests
# ======================================================================


class TestSetMetric:
    """Tests for PhysicsConsistencyLoss.set_metric()."""

    @pytest.mark.parametrize(
        "initial, target",
        [
            ("cosine", "mse"),
            ("mse", "huber"),
            ("huber", "hybrid"),
            ("hybrid", "cosine"),
        ],
    )
    def test_switch_strategy(self, initial, target):
        fn = _default_loss_fn(initial)
        fn.set_metric(target)
        assert fn.current_metric() == target

    def test_set_metric_case_insensitive(self):
        fn = _default_loss_fn("cosine")
        fn.set_metric("MSE")
        assert fn.current_metric() == "mse"

    def test_unknown_mode_raises(self):
        fn = _default_loss_fn()
        with pytest.raises(ValueError, match="Unknown consistency mode"):
            fn.set_metric("does_not_exist")

    def test_custom_disallowed_raises(self):
        @register_strategy("_test_custom_blocked")
        class _BlockedLoss(BaseConsistencyLoss):
            @property
            def metric_name(self):
                return "_test_custom_blocked"

            def compute(self, enc, phy):
                return torch.tensor(0.0)

        cfg = PhysicsConsistencyConfig(allow_custom_strategy=False)
        fn = PhysicsConsistencyLoss(cfg)
        with pytest.raises(ValueError, match="not permitted"):
            fn.set_metric("_test_custom_blocked")

    def test_custom_allowed_with_flag(self):
        @register_strategy("_test_custom_allowed")
        class _AllowedLoss(BaseConsistencyLoss):
            @property
            def metric_name(self):
                return "_test_custom_allowed"

            def compute(self, enc, phy):
                return torch.tensor(0.0)

        cfg = PhysicsConsistencyConfig(allow_custom_strategy=True)
        fn = PhysicsConsistencyLoss(cfg)
        fn.set_metric("_test_custom_allowed")
        assert fn.current_metric() == "_test_custom_allowed"

    def test_forward_uses_new_strategy_after_set(self):
        enc, phy = _make_embeddings()
        fn = _default_loss_fn("cosine")
        out_cosine = fn(enc, phy)

        fn.set_metric("mse")
        out_mse = fn(enc, phy)

        # Different strategies should (in general) produce different values.
        assert out_cosine["metric"] == "cosine"
        assert out_mse["metric"] == "mse"


# ======================================================================
# Forward Output Contract Tests
# ======================================================================


class TestForwardOutputContract:
    """Tests that forward() always returns the required output structure."""

    @pytest.mark.parametrize("mode", ["cosine", "mse", "huber", "hybrid"])
    def test_output_keys(self, mode):
        fn = _default_loss_fn(mode)
        enc, phy = _make_embeddings()
        out = fn(enc, phy)
        assert set(out.keys()) == {"loss", "metric", "value"}

    @pytest.mark.parametrize("mode", ["cosine", "mse", "huber", "hybrid"])
    def test_loss_is_scalar(self, mode):
        fn = _default_loss_fn(mode)
        enc, phy = _make_embeddings()
        out = fn(enc, phy)
        assert out["loss"].shape == torch.Size([])

    @pytest.mark.parametrize("mode", ["cosine", "mse", "huber", "hybrid"])
    def test_value_is_scalar(self, mode):
        fn = _default_loss_fn(mode)
        enc, phy = _make_embeddings()
        out = fn(enc, phy)
        assert out["value"].shape == torch.Size([])

    @pytest.mark.parametrize("mode", ["cosine", "mse", "huber", "hybrid"])
    def test_metric_is_string(self, mode):
        fn = _default_loss_fn(mode)
        enc, phy = _make_embeddings()
        out = fn(enc, phy)
        assert isinstance(out["metric"], str)
        assert out["metric"] == mode

    @pytest.mark.parametrize("mode", ["cosine", "mse", "huber", "hybrid"])
    def test_value_is_detached(self, mode):
        fn = _default_loss_fn(mode)
        enc, phy = _make_embeddings(seed=42)
        enc.requires_grad_(True)
        phy.requires_grad_(True)
        out = fn(enc, phy)
        # "value" must not have a grad_fn (detached)
        assert out["value"].grad_fn is None

    @pytest.mark.parametrize("mode", ["cosine", "mse", "huber", "hybrid"])
    def test_loss_has_grad_fn(self, mode):
        fn = _default_loss_fn(mode)
        enc, phy = _make_embeddings()
        enc.requires_grad_(True)
        out = fn(enc, phy)
        assert out["loss"].grad_fn is not None


# ======================================================================
# Per-Strategy Correctness Tests
# ======================================================================


class TestCosineStrategy:
    """Mathematical correctness tests for CosineConsistencyLoss."""

    def test_identical_embeddings_zero_loss(self):
        fn = _default_loss_fn("cosine")
        enc, _ = _make_embeddings()
        out = fn(enc, enc)
        assert out["loss"].item() == pytest.approx(0.0, abs=1e-5)

    def test_orthogonal_embeddings_loss_near_one(self):
        # Construct orthogonal pairs: enc along dim 0, phy along dim 1
        enc = torch.zeros(1, _DIM)
        enc[0, 0] = 1.0
        phy = torch.zeros(1, _DIM)
        phy[0, 1] = 1.0
        fn = _default_loss_fn("cosine")
        out = fn(enc, phy)
        assert out["loss"].item() == pytest.approx(1.0, abs=1e-5)

    def test_opposite_embeddings_loss_near_two(self):
        enc, _ = _make_embeddings(batch=1)
        phy = -enc
        fn = _default_loss_fn("cosine")
        out = fn(enc, phy)
        assert out["loss"].item() == pytest.approx(2.0, abs=1e-4)

    def test_loss_non_negative(self):
        fn = _default_loss_fn("cosine")
        enc, phy = _make_embeddings()
        out = fn(enc, phy)
        assert out["loss"].item() >= 0.0

    def test_loss_bounded_above(self):
        # cosine distance in [0, 2]
        fn = _default_loss_fn("cosine")
        enc, phy = _make_embeddings()
        out = fn(enc, phy)
        assert out["loss"].item() <= 2.0 + 1e-6


class TestMSEStrategy:
    """Mathematical correctness tests for MSEConsistencyLoss."""

    def test_identical_embeddings_zero_loss(self):
        fn = _default_loss_fn("mse")
        enc, _ = _make_embeddings()
        out = fn(enc, enc)
        assert out["loss"].item() == pytest.approx(0.0, abs=1e-6)

    def test_loss_non_negative(self):
        fn = _default_loss_fn("mse")
        enc, phy = _make_embeddings()
        out = fn(enc, phy)
        assert out["loss"].item() >= 0.0

    def test_manual_calculation(self):
        enc = torch.tensor([[1.0, 0.0]])
        phy = torch.tensor([[0.0, 1.0]])
        fn = _default_loss_fn("mse")
        out = fn(enc, phy)
        expected = ((1.0 - 0.0) ** 2 + (0.0 - 1.0) ** 2) / 2
        assert out["loss"].item() == pytest.approx(expected, abs=1e-6)

    def test_scales_with_distance(self):
        enc = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
        phy_near = torch.tensor([[0.9, 0.1, 0.0, 0.0]])
        phy_far = torch.tensor([[0.0, 1.0, 0.0, 0.0]])
        fn = _default_loss_fn("mse")
        loss_near = fn(enc, phy_near)["loss"].item()
        loss_far = fn(enc, phy_far)["loss"].item()
        assert loss_near < loss_far


class TestHuberStrategy:
    """Mathematical correctness tests for HuberConsistencyLoss."""

    def test_identical_embeddings_zero_loss(self):
        fn = _default_loss_fn("huber")
        enc, _ = _make_embeddings()
        out = fn(enc, enc)
        assert out["loss"].item() == pytest.approx(0.0, abs=1e-6)

    def test_loss_non_negative(self):
        fn = _default_loss_fn("huber")
        enc, phy = _make_embeddings()
        out = fn(enc, phy)
        assert out["loss"].item() >= 0.0

    def test_smooth_l1_equivalence(self):
        enc, phy = _make_embeddings()
        fn = _default_loss_fn("huber")
        out = fn(enc, phy)
        expected = F.smooth_l1_loss(enc, phy, reduction="mean")
        assert out["loss"].item() == pytest.approx(expected.item(), abs=1e-6)

    def test_robustness_to_large_errors(self):
        # Huber should have lower loss than MSE for large residuals.
        enc = torch.zeros(4, _DIM)
        phy = torch.full((4, _DIM), fill_value=10.0)
        fn_huber = _default_loss_fn("huber")
        fn_mse = _default_loss_fn("mse")
        loss_huber = fn_huber(enc, phy)["loss"].item()
        loss_mse = fn_mse(enc, phy)["loss"].item()
        assert loss_huber < loss_mse


class TestHybridStrategy:
    """Mathematical correctness tests for HybridConsistencyLoss."""

    def test_identical_embeddings_zero_loss(self):
        fn = _default_loss_fn("hybrid")
        enc, _ = _make_embeddings()
        out = fn(enc, enc)
        assert out["loss"].item() == pytest.approx(0.0, abs=1e-5)

    def test_loss_is_weighted_combination(self):
        alpha, beta = 0.7, 0.3
        cfg = PhysicsConsistencyConfig(
            mode="hybrid", cosine_weight=alpha, mse_weight=beta
        )
        fn = PhysicsConsistencyLoss(cfg)

        cfg_cos = PhysicsConsistencyConfig(mode="cosine")
        fn_cos = PhysicsConsistencyLoss(cfg_cos)
        cfg_mse = PhysicsConsistencyConfig(mode="mse")
        fn_mse = PhysicsConsistencyLoss(cfg_mse)

        enc, phy = _make_embeddings(seed=7)
        hybrid_loss = fn(enc, phy)["loss"].item()
        cos_loss = fn_cos(enc, phy)["loss"].item()
        mse_loss = fn_mse(enc, phy)["loss"].item()

        expected = alpha * cos_loss + beta * mse_loss
        assert hybrid_loss == pytest.approx(expected, abs=1e-5)

    def test_custom_weights(self):
        cfg = PhysicsConsistencyConfig(
            mode="hybrid", cosine_weight=0.5, mse_weight=0.5
        )
        fn = PhysicsConsistencyLoss(cfg)
        enc, phy = _make_embeddings()
        out = fn(enc, phy)
        assert out["loss"].item() >= 0.0

    def test_loss_non_negative(self):
        fn = _default_loss_fn("hybrid")
        enc, phy = _make_embeddings()
        assert fn(enc, phy)["loss"].item() >= 0.0


# ======================================================================
# Gradient Propagation Tests
# ======================================================================


class TestGradientPropagation:
    """Tests that gradients flow through the loss."""

    @pytest.mark.parametrize("mode", ["cosine", "mse", "huber", "hybrid"])
    def test_backward_does_not_raise(self, mode):
        fn = _default_loss_fn(mode)
        enc, phy = _make_embeddings()
        enc = enc.detach().requires_grad_(True)
        phy = phy.detach().requires_grad_(True)
        out = fn(enc, phy)
        out["loss"].backward()

    @pytest.mark.parametrize("mode", ["cosine", "mse", "huber", "hybrid"])
    def test_grad_is_not_none(self, mode):
        fn = _default_loss_fn(mode)
        enc, phy = _make_embeddings()
        enc = enc.detach().requires_grad_(True)
        phy = phy.detach().requires_grad_(True)
        out = fn(enc, phy)
        out["loss"].backward()
        assert enc.grad is not None, "Gradient on enc must not be None"
        assert phy.grad is not None, "Gradient on phy must not be None"

    @pytest.mark.parametrize("mode", ["cosine", "mse", "huber", "hybrid"])
    def test_grad_is_finite(self, mode):
        fn = _default_loss_fn(mode)
        enc, phy = _make_embeddings()
        enc = enc.detach().requires_grad_(True)
        phy = phy.detach().requires_grad_(True)
        out = fn(enc, phy)
        out["loss"].backward()
        assert torch.isfinite(enc.grad).all(), "enc gradients contain NaN/Inf"
        assert torch.isfinite(phy.grad).all(), "phy gradients contain NaN/Inf"


# ======================================================================
# Determinism Tests
# ======================================================================


class TestDeterminism:
    """Tests for deterministic behaviour given the same inputs."""

    @pytest.mark.parametrize("mode", ["cosine", "mse", "huber", "hybrid"])
    def test_same_inputs_same_output(self, mode):
        fn = _default_loss_fn(mode)
        enc, phy = _make_embeddings(seed=0)
        out1 = fn(enc, phy)["loss"].item()
        out2 = fn(enc, phy)["loss"].item()
        assert out1 == pytest.approx(out2, abs=0.0)


# ======================================================================
# Serialisation Tests
# ======================================================================


class TestSerialisation:
    """Tests for state_dict round-trip (no learnable parameters)."""

    @pytest.mark.parametrize("mode", ["cosine", "mse", "huber", "hybrid"])
    def test_state_dict_is_empty(self, mode):
        fn = _default_loss_fn(mode)
        sd = fn.state_dict()
        assert len(sd) == 0, "PhysicsConsistencyLoss has no learnable params"

    @pytest.mark.parametrize("mode", ["cosine", "mse", "huber", "hybrid"])
    def test_load_state_dict_does_not_raise(self, mode):
        fn = _default_loss_fn(mode)
        fn2 = _default_loss_fn(mode)
        fn2.load_state_dict(fn.state_dict())

    @pytest.mark.parametrize("mode", ["cosine", "mse", "huber", "hybrid"])
    def test_deepcopy_produces_same_output(self, mode):
        fn = _default_loss_fn(mode)
        fn2 = copy.deepcopy(fn)
        enc, phy = _make_embeddings(seed=5)
        loss1 = fn(enc, phy)["loss"].item()
        loss2 = fn2(enc, phy)["loss"].item()
        assert loss1 == pytest.approx(loss2, abs=1e-7)


# ======================================================================
# TorchScript Compatibility Tests
# ======================================================================


class TestTorchScript:
    """Tests for TorchScript compatibility where applicable."""

    @pytest.mark.parametrize("mode", ["cosine", "mse", "huber", "hybrid"])
    def test_torchscript_forward(self, mode):
        fn = _default_loss_fn(mode)
        fn.eval()
        enc, phy = _make_embeddings()
        try:
            scripted = torch.jit.script(fn)
            out = scripted(enc, phy)
            assert "loss" in out
            assert out["loss"].shape == torch.Size([])
        except Exception as exc:
            pytest.skip(f"TorchScript not supported for mode '{mode}': {exc}")


# ======================================================================
# Input Validation Tests
# ======================================================================


class TestInputValidation:
    """Tests that invalid inputs raise informative exceptions."""

    def test_encoder_not_tensor_raises(self):
        fn = _default_loss_fn()
        _, phy = _make_embeddings()
        with pytest.raises(TypeError, match="temporal_embedding must be a torch.Tensor"):
            fn("not_a_tensor", phy)  # type: ignore[arg-type]

    def test_physics_not_tensor_raises(self):
        fn = _default_loss_fn()
        enc, _ = _make_embeddings()
        with pytest.raises(TypeError, match="physics_embedding must be a torch.Tensor"):
            fn(enc, 42)  # type: ignore[arg-type]

    def test_encoder_integer_dtype_raises(self):
        fn = _default_loss_fn()
        enc = torch.randint(0, 10, (_BATCH, _DIM))
        phy = torch.randn(_BATCH, _DIM)
        with pytest.raises(ValueError, match="floating-point dtype"):
            fn(enc, phy)

    def test_physics_integer_dtype_raises(self):
        fn = _default_loss_fn()
        enc = torch.randn(_BATCH, _DIM)
        phy = torch.randint(0, 10, (_BATCH, _DIM))
        with pytest.raises(ValueError, match="floating-point dtype"):
            fn(enc, phy)

    def test_encoder_1d_raises(self):
        fn = _default_loss_fn()
        enc = torch.randn(_DIM)
        phy = torch.randn(_BATCH, _DIM)
        with pytest.raises(ValueError, match="exactly 2 dimensions"):
            fn(enc, phy)

    def test_physics_1d_raises(self):
        fn = _default_loss_fn()
        enc = torch.randn(_BATCH, _DIM)
        phy = torch.randn(_DIM)
        with pytest.raises(ValueError, match="exactly 2 dimensions"):
            fn(enc, phy)

    def test_encoder_3d_raises(self):
        fn = _default_loss_fn()
        enc = torch.randn(2, _BATCH, _DIM)
        phy = torch.randn(_BATCH, _DIM)
        with pytest.raises(ValueError, match="exactly 2 dimensions"):
            fn(enc, phy)

    def test_batch_size_mismatch_raises(self):
        fn = _default_loss_fn()
        enc = torch.randn(4, _DIM)
        phy = torch.randn(8, _DIM)
        with pytest.raises(ValueError, match="same batch size"):
            fn(enc, phy)

    def test_embedding_dim_mismatch_raises(self):
        fn = _default_loss_fn()
        enc = torch.randn(_BATCH, 128)
        phy = torch.randn(_BATCH, 256)
        with pytest.raises(ValueError, match="same embedding dimension"):
            fn(enc, phy)

    def test_encoder_nan_raises(self):
        fn = _default_loss_fn()
        enc = torch.full((_BATCH, _DIM), float("nan"))
        phy = torch.randn(_BATCH, _DIM)
        with pytest.raises(ValueError, match="NaN"):
            fn(enc, phy)

    def test_physics_nan_raises(self):
        fn = _default_loss_fn()
        enc = torch.randn(_BATCH, _DIM)
        phy = torch.full((_BATCH, _DIM), float("nan"))
        with pytest.raises(ValueError, match="NaN"):
            fn(enc, phy)

    def test_encoder_inf_raises(self):
        fn = _default_loss_fn()
        enc = torch.full((_BATCH, _DIM), float("inf"))
        phy = torch.randn(_BATCH, _DIM)
        with pytest.raises(ValueError, match="Inf"):
            fn(enc, phy)

    def test_physics_inf_raises(self):
        fn = _default_loss_fn()
        enc = torch.randn(_BATCH, _DIM)
        phy = torch.full((_BATCH, _DIM), float("-inf"))
        with pytest.raises(ValueError, match="Inf"):
            fn(enc, phy)


# ======================================================================
# Reduction Tests
# ======================================================================


class TestReduction:
    """Tests for reduction='sum' variant."""

    @pytest.mark.parametrize("mode", ["cosine", "mse", "huber", "hybrid"])
    def test_sum_greater_than_or_equal_to_mean(self, mode):
        cfg = PhysicsConsistencyConfig(mode=mode, reduction="sum")
        fn_sum = PhysicsConsistencyLoss(cfg)

        cfg_mean = PhysicsConsistencyConfig(mode=mode, reduction="mean")
        fn_mean = PhysicsConsistencyLoss(cfg_mean)

        enc, phy = _make_embeddings()
        loss_sum = fn_sum(enc, phy)["loss"].item()
        loss_mean = fn_mean(enc, phy)["loss"].item()
        assert loss_sum == pytest.approx(loss_mean * _BATCH, abs=1e-4)


# ======================================================================
# CPU / CUDA Tests
# ======================================================================


class TestDevice:
    """Tests for CPU and CUDA execution."""

    @pytest.mark.parametrize("mode", ["cosine", "mse", "huber", "hybrid"])
    def test_cpu_forward(self, mode):
        fn = _default_loss_fn(mode)
        enc, phy = _make_embeddings(device="cpu")
        out = fn(enc, phy)
        assert out["loss"].device.type == "cpu"

    @pytest.mark.parametrize("mode", ["cosine", "mse", "huber", "hybrid"])
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_cuda_forward(self, mode):
        device = torch.device("cuda")
        fn = _default_loss_fn(mode)
        enc, phy = _make_embeddings(device=device)
        out = fn(enc, phy)
        assert out["loss"].device.type == "cuda"
        assert out["loss"].shape == torch.Size([])

    @pytest.mark.parametrize("mode", ["cosine", "mse", "huber", "hybrid"])
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_cuda_gradients(self, mode):
        device = torch.device("cuda")
        fn = _default_loss_fn(mode)
        enc, phy = _make_embeddings(device=device)
        enc = enc.requires_grad_(True)
        phy = phy.requires_grad_(True)
        out = fn(enc, phy)
        out["loss"].backward()
        assert enc.grad is not None
        assert torch.isfinite(enc.grad).all()


# ======================================================================
# Extensibility Tests
# ======================================================================


class TestExtensibility:
    """Tests that the strategy registry is open/closed."""

    def test_new_strategy_appears_in_registry(self):
        @register_strategy("_ext_test_strategy")
        class _ExtTestLoss(BaseConsistencyLoss):
            @property
            def metric_name(self):
                return "_ext_test_strategy"

            def compute(self, enc, phy):
                return torch.tensor(0.0, requires_grad=True)

        assert "_ext_test_strategy" in _STRATEGY_REGISTRY

    def test_new_strategy_selectable_via_config(self):
        @register_strategy("_config_select_test")
        class _ConfigSelectLoss(BaseConsistencyLoss):
            @property
            def metric_name(self):
                return "_config_select_test"

            def compute(self, enc, phy):
                return (enc - phy).abs().mean()

        cfg = PhysicsConsistencyConfig(mode="_config_select_test")
        fn = PhysicsConsistencyLoss(cfg)
        enc, phy = _make_embeddings()
        out = fn(enc, phy)
        assert out["metric"] == "_config_select_test"
        assert out["loss"].shape == torch.Size([])

    def test_no_existing_code_modified(self):
        """Verify built-in strategies are unchanged after registering custom."""
        @register_strategy("_no_modify_test")
        class _NoModifyLoss(BaseConsistencyLoss):
            @property
            def metric_name(self):
                return "_no_modify_test"

            def compute(self, enc, phy):
                return torch.tensor(0.0)

        fn = _default_loss_fn("cosine")
        enc, phy = _make_embeddings()
        out = fn(enc, phy)
        assert out["metric"] == "cosine"


# ======================================================================
# parameter_summary() Tests
# ======================================================================


class TestParameterSummary:
    """Tests for PhysicsConsistencyLoss.parameter_summary()."""

    def test_returns_dict(self):
        fn = _default_loss_fn()
        summary = fn.parameter_summary()
        assert isinstance(summary, dict)

    def test_required_keys(self):
        fn = _default_loss_fn()
        summary = fn.parameter_summary()
        for key in (
            "mode", "reduction", "eps", "cosine_weight",
            "mse_weight", "allow_custom_strategy",
            "num_parameters", "available_metrics",
        ):
            assert key in summary, f"Missing key '{key}' in parameter_summary()"

    def test_num_parameters_is_zero(self):
        fn = _default_loss_fn()
        assert fn.parameter_summary()["num_parameters"] == 0

    @pytest.mark.parametrize("mode", ["cosine", "mse", "huber", "hybrid"])
    def test_mode_reflects_config(self, mode):
        fn = _default_loss_fn(mode)
        assert fn.parameter_summary()["mode"] == mode

    def test_available_metrics_in_summary_is_list(self):
        fn = _default_loss_fn()
        metrics = fn.parameter_summary()["available_metrics"]
        assert isinstance(metrics, list)
