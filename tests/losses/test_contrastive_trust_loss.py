"""Tests for :mod:`src.losses.contrastive_trust_loss`.

Comprehensive test suite covering:

- ContrastiveTrustLossConfig initialisation and validation
- ContrastiveTrustLoss initialisation and dependency injection
- forward() output contract (dict keys, scalar loss, shapes)
- correct weighted aggregation (total == Σ weighted contributions)
- weight normalisation (effective weights sum to one)
- learnable weights (trainable parameters receive gradients)
- freeze_weights / unfreeze_weights
- get_weights / set_weights / get_effective_weights
- compute_total_loss aggregation + missing-objective handling
- serialisation round-trip (state_dict / load_state_dict / deepcopy)
- gradient propagation to the encoder / projection inputs
- deterministic behaviour under fixed seeds
- mixed-precision compatibility (autocast + fp16 inputs)
- CPU execution
- CUDA execution (skipped if unavailable)
- invalid inputs: wrong type, non-float dtype, 1D/3D tensors,
  batch-size mismatch, embedding-dim mismatch
- NaN / Inf input handling
- parameter_summary()
- TorchScript best-effort compatibility
"""

from __future__ import annotations

import copy
import io
import tempfile
from pathlib import Path

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.losses.contrastive_trust_loss import (
    ContrastiveTrustLoss,
    ContrastiveTrustLossConfig,
)
from src.losses.nt_xent import NTXentConfig, NTXentLoss
from src.losses.physics_consistency import (
    PhysicsConsistencyConfig,
    PhysicsConsistencyLoss,
)

# ======================================================================
# Constants
# ======================================================================

_BATCH: int = 8
_PROJ_DIM: int = 128
_EMB_DIM: int = 256


# ======================================================================
# Helpers
# ======================================================================


def _make_inputs(
    batch: int = _BATCH,
    proj_dim: int = _PROJ_DIM,
    emb_dim: int = _EMB_DIM,
    device: torch.device | str = "cpu",
    seed: int = 0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return ``(view_1, view_2, encoder_embedding, physics_embedding)``.

    Projection views are L2-normalised (as produced by the projection
    head); encoder / physics embeddings are L2-normalised as expected by
    the physics-consistency loss.
    """
    g = torch.Generator(device="cpu")
    g.manual_seed(seed)
    v1 = F.normalize(torch.randn(batch, proj_dim, generator=g), p=2, dim=1)
    v2 = F.normalize(torch.randn(batch, proj_dim, generator=g), p=2, dim=1)
    enc = F.normalize(torch.randn(batch, emb_dim, generator=g), p=2, dim=1)
    phy = F.normalize(torch.randn(batch, emb_dim, generator=g), p=2, dim=1)
    return (
        v1.to(device),
        v2.to(device),
        enc.to(device),
        phy.to(device),
    )


def _default_loss_fn(**cfg_kwargs) -> ContrastiveTrustLoss:
    return ContrastiveTrustLoss(ContrastiveTrustLossConfig(**cfg_kwargs))


# ======================================================================
# ContrastiveTrustLossConfig tests
# ======================================================================


class TestContrastiveTrustLossConfig:
    """Tests for ContrastiveTrustLossConfig validation."""

    def test_defaults(self) -> None:
        cfg = ContrastiveTrustLossConfig()
        assert cfg.contrastive_weight == pytest.approx(1.0)
        assert cfg.physics_weight == pytest.approx(1.0)
        assert cfg.normalize_weights is False
        assert cfg.learnable_weights is False
        assert cfg.log_individual_losses is True
        assert cfg.eps == pytest.approx(1e-8)

    def test_custom_values(self) -> None:
        cfg = ContrastiveTrustLossConfig(
            contrastive_weight=2.0,
            physics_weight=0.5,
            normalize_weights=True,
            learnable_weights=True,
            log_individual_losses=False,
        )
        assert cfg.contrastive_weight == pytest.approx(2.0)
        assert cfg.physics_weight == pytest.approx(0.5)
        assert cfg.normalize_weights is True
        assert cfg.learnable_weights is True
        assert cfg.log_individual_losses is False

    def test_frozen(self) -> None:
        cfg = ContrastiveTrustLossConfig()
        with pytest.raises((AttributeError, TypeError)):
            cfg.contrastive_weight = 3.0  # type: ignore[misc]

    def test_negative_contrastive_weight_raises(self) -> None:
        with pytest.raises(ValueError, match="contrastive_weight must be non-negative"):
            ContrastiveTrustLossConfig(contrastive_weight=-1.0)

    def test_negative_physics_weight_raises(self) -> None:
        with pytest.raises(ValueError, match="physics_weight must be non-negative"):
            ContrastiveTrustLossConfig(physics_weight=-0.5)

    def test_nan_weight_raises(self) -> None:
        with pytest.raises(ValueError, match="must be finite"):
            ContrastiveTrustLossConfig(contrastive_weight=float("nan"))

    def test_inf_weight_raises(self) -> None:
        with pytest.raises(ValueError, match="must be finite"):
            ContrastiveTrustLossConfig(physics_weight=float("inf"))

    def test_zero_eps_raises(self) -> None:
        with pytest.raises(ValueError, match="eps must be strictly positive"):
            ContrastiveTrustLossConfig(eps=0.0)

    def test_negative_eps_raises(self) -> None:
        with pytest.raises(ValueError, match="eps must be strictly positive"):
            ContrastiveTrustLossConfig(eps=-1e-9)

    def test_zero_weights_allowed(self) -> None:
        cfg = ContrastiveTrustLossConfig(
            contrastive_weight=0.0, physics_weight=0.0
        )
        assert cfg.contrastive_weight == pytest.approx(0.0)


# ======================================================================
# Initialisation tests
# ======================================================================


class TestInitialisation:
    """Tests for ContrastiveTrustLoss construction."""

    def test_default_init(self) -> None:
        fn = _default_loss_fn()
        assert isinstance(fn, ContrastiveTrustLoss)
        assert isinstance(fn, nn.Module)

    def test_wrong_config_type_raises(self) -> None:
        with pytest.raises(TypeError, match="ContrastiveTrustLossConfig"):
            ContrastiveTrustLoss("bad_config")  # type: ignore[arg-type]

    def test_default_sublosses_created(self) -> None:
        fn = _default_loss_fn()
        assert isinstance(fn.contrastive_loss, NTXentLoss)
        assert isinstance(fn.physics_loss, PhysicsConsistencyLoss)

    def test_dependency_injection(self) -> None:
        nt = NTXentLoss(NTXentConfig(temperature=0.2))
        phy = PhysicsConsistencyLoss(PhysicsConsistencyConfig(mode="mse"))
        fn = ContrastiveTrustLoss(
            ContrastiveTrustLossConfig(),
            contrastive_loss=nt,
            physics_loss=phy,
        )
        assert fn.contrastive_loss is nt
        assert fn.physics_loss is phy

    def test_injected_contrastive_wrong_type_raises(self) -> None:
        with pytest.raises(TypeError, match="contrastive_loss must be a torch.nn.Module"):
            ContrastiveTrustLoss(
                ContrastiveTrustLossConfig(),
                contrastive_loss="not_a_module",  # type: ignore[arg-type]
            )

    def test_injected_physics_wrong_type_raises(self) -> None:
        with pytest.raises(TypeError, match="physics_loss must be a torch.nn.Module"):
            ContrastiveTrustLoss(
                ContrastiveTrustLossConfig(),
                physics_loss=123,  # type: ignore[arg-type]
            )

    def test_config_property(self) -> None:
        cfg = ContrastiveTrustLossConfig(contrastive_weight=3.0)
        fn = ContrastiveTrustLoss(cfg)
        assert fn.config is cfg

    def test_objective_names(self) -> None:
        fn = _default_loss_fn()
        assert fn.objective_names == ("contrastive", "physics")

    def test_weights_are_parameters(self) -> None:
        fn = _default_loss_fn()
        assert isinstance(fn.weights["contrastive"], nn.Parameter)
        assert isinstance(fn.weights["physics"], nn.Parameter)

    def test_non_learnable_weights_have_no_grad(self) -> None:
        fn = _default_loss_fn(learnable_weights=False)
        assert fn.weights["contrastive"].requires_grad is False
        assert fn.weights["physics"].requires_grad is False

    def test_learnable_weights_have_grad(self) -> None:
        fn = _default_loss_fn(learnable_weights=True)
        assert fn.weights["contrastive"].requires_grad is True
        assert fn.weights["physics"].requires_grad is True


# ======================================================================
# Forward output contract tests
# ======================================================================


class TestForwardContract:
    """Tests that forward() always returns the required structure."""

    def test_output_is_dict(self) -> None:
        fn = _default_loss_fn()
        out = fn(*_make_inputs())
        assert isinstance(out, dict)

    def test_output_keys(self) -> None:
        fn = _default_loss_fn()
        out = fn(*_make_inputs())
        expected = {
            "loss",
            "contrastive_loss",
            "physics_loss",
            "weighted_contrastive",
            "weighted_physics",
            "weights",
        }
        assert expected.issubset(set(out.keys()))

    def test_loss_is_scalar(self) -> None:
        fn = _default_loss_fn()
        out = fn(*_make_inputs())
        assert isinstance(out["loss"], torch.Tensor)
        assert out["loss"].shape == torch.Size([])

    def test_component_losses_are_scalar(self) -> None:
        fn = _default_loss_fn()
        out = fn(*_make_inputs())
        for key in ("contrastive_loss", "physics_loss",
                    "weighted_contrastive", "weighted_physics"):
            assert out[key].shape == torch.Size([]), key

    def test_weights_is_dict_of_floats(self) -> None:
        fn = _default_loss_fn()
        out = fn(*_make_inputs())
        assert isinstance(out["weights"], dict)
        assert set(out["weights"]) == {"contrastive", "physics"}
        for value in out["weights"].values():
            assert isinstance(value, float)

    def test_loss_is_finite(self) -> None:
        fn = _default_loss_fn()
        out = fn(*_make_inputs())
        assert torch.isfinite(out["loss"])

    def test_loss_has_grad_fn(self) -> None:
        fn = _default_loss_fn()
        v1, v2, enc, phy = _make_inputs()
        enc = enc.requires_grad_(True)
        out = fn(v1, v2, enc, phy)
        assert out["loss"].grad_fn is not None


# ======================================================================
# Aggregation correctness tests
# ======================================================================


class TestAggregation:
    """Tests for correct weighted aggregation of the objectives."""

    def test_total_equals_sum_of_weighted(self) -> None:
        fn = _default_loss_fn(
            contrastive_weight=2.0, physics_weight=3.0
        )
        out = fn(*_make_inputs())
        expected = out["weighted_contrastive"] + out["weighted_physics"]
        assert out["loss"].item() == pytest.approx(expected.item(), abs=1e-6)

    def test_weighted_equals_weight_times_raw(self) -> None:
        wc, wp = 2.0, 0.5
        fn = _default_loss_fn(contrastive_weight=wc, physics_weight=wp)
        out = fn(*_make_inputs())
        assert out["weighted_contrastive"].item() == pytest.approx(
            wc * out["contrastive_loss"].item(), abs=1e-6
        )
        assert out["weighted_physics"].item() == pytest.approx(
            wp * out["physics_loss"].item(), abs=1e-6
        )

    def test_raw_losses_match_sublosses(self) -> None:
        fn = _default_loss_fn()
        v1, v2, enc, phy = _make_inputs(seed=11)
        out = fn(v1, v2, enc, phy)

        expected_c = fn.contrastive_loss(v1, v2)["loss"].item()
        expected_p = fn.physics_loss(enc, phy)["loss"].item()
        assert out["contrastive_loss"].item() == pytest.approx(expected_c, abs=1e-6)
        assert out["physics_loss"].item() == pytest.approx(expected_p, abs=1e-6)

    def test_zero_physics_weight_isolates_contrastive(self) -> None:
        fn = _default_loss_fn(contrastive_weight=1.0, physics_weight=0.0)
        out = fn(*_make_inputs())
        assert out["loss"].item() == pytest.approx(
            out["contrastive_loss"].item(), abs=1e-6
        )

    def test_compute_total_loss_directly(self) -> None:
        fn = _default_loss_fn(contrastive_weight=1.0, physics_weight=2.0)
        losses = {
            "contrastive": torch.tensor(3.0),
            "physics": torch.tensor(4.0),
        }
        agg = fn.compute_total_loss(losses)
        assert agg["loss"].item() == pytest.approx(1.0 * 3.0 + 2.0 * 4.0)

    def test_compute_total_loss_missing_objective_raises(self) -> None:
        fn = _default_loss_fn()
        with pytest.raises(KeyError, match="missing objectives"):
            fn.compute_total_loss({"contrastive": torch.tensor(1.0)})


# ======================================================================
# Weight normalisation tests
# ======================================================================


class TestWeightNormalisation:
    """Tests for normalize_weights behaviour."""

    def test_effective_weights_sum_to_one(self) -> None:
        fn = _default_loss_fn(
            contrastive_weight=3.0, physics_weight=1.0,
            normalize_weights=True,
        )
        eff = fn.get_effective_weights()
        assert sum(eff.values()) == pytest.approx(1.0, abs=1e-6)

    def test_effective_weights_ratio_preserved(self) -> None:
        fn = _default_loss_fn(
            contrastive_weight=3.0, physics_weight=1.0,
            normalize_weights=True,
        )
        eff = fn.get_effective_weights()
        assert eff["contrastive"] == pytest.approx(0.75, abs=1e-6)
        assert eff["physics"] == pytest.approx(0.25, abs=1e-6)

    def test_normalised_weights_used_in_forward(self) -> None:
        fn = _default_loss_fn(
            contrastive_weight=3.0, physics_weight=1.0,
            normalize_weights=True,
        )
        out = fn(*_make_inputs())
        assert out["weighted_contrastive"].item() == pytest.approx(
            0.75 * out["contrastive_loss"].item(), abs=1e-6
        )
        assert out["weights"]["contrastive"] == pytest.approx(0.75, abs=1e-6)

    def test_raw_weights_used_when_not_normalised(self) -> None:
        fn = _default_loss_fn(
            contrastive_weight=3.0, physics_weight=1.0,
            normalize_weights=False,
        )
        eff = fn.get_effective_weights()
        assert eff["contrastive"] == pytest.approx(3.0)
        assert eff["physics"] == pytest.approx(1.0)

    def test_get_weights_returns_raw_even_when_normalised(self) -> None:
        fn = _default_loss_fn(
            contrastive_weight=3.0, physics_weight=1.0,
            normalize_weights=True,
        )
        raw = fn.get_weights()
        assert raw["contrastive"] == pytest.approx(3.0)
        assert raw["physics"] == pytest.approx(1.0)


# ======================================================================
# Weight management tests
# ======================================================================


class TestWeightManagement:
    """Tests for get_weights / set_weights / freeze / unfreeze."""

    def test_get_weights(self) -> None:
        fn = _default_loss_fn(contrastive_weight=1.5, physics_weight=2.5)
        w = fn.get_weights()
        assert w["contrastive"] == pytest.approx(1.5)
        assert w["physics"] == pytest.approx(2.5)

    def test_set_weights_updates_values(self) -> None:
        fn = _default_loss_fn()
        fn.set_weights({"contrastive": 4.0, "physics": 0.25})
        w = fn.get_weights()
        assert w["contrastive"] == pytest.approx(4.0)
        assert w["physics"] == pytest.approx(0.25)

    def test_set_weights_partial(self) -> None:
        fn = _default_loss_fn(contrastive_weight=1.0, physics_weight=1.0)
        fn.set_weights({"physics": 5.0})
        w = fn.get_weights()
        assert w["contrastive"] == pytest.approx(1.0)
        assert w["physics"] == pytest.approx(5.0)

    def test_set_weights_affects_forward(self) -> None:
        fn = _default_loss_fn()
        inputs = _make_inputs()
        fn.set_weights({"contrastive": 10.0, "physics": 0.0})
        out = fn(*inputs)
        assert out["loss"].item() == pytest.approx(
            10.0 * out["contrastive_loss"].item(), abs=1e-5
        )

    def test_set_weights_unknown_objective_raises(self) -> None:
        fn = _default_loss_fn()
        with pytest.raises(KeyError, match="Unknown objective"):
            fn.set_weights({"unknown": 1.0})

    def test_set_weights_negative_raises(self) -> None:
        fn = _default_loss_fn()
        with pytest.raises(ValueError, match="non-negative"):
            fn.set_weights({"contrastive": -1.0})

    def test_set_weights_nan_raises(self) -> None:
        fn = _default_loss_fn()
        with pytest.raises(ValueError, match="finite"):
            fn.set_weights({"physics": float("nan")})

    def test_set_weights_not_mapping_raises(self) -> None:
        fn = _default_loss_fn()
        with pytest.raises(TypeError, match="mapping"):
            fn.set_weights([1.0, 2.0])  # type: ignore[arg-type]

    def test_freeze_weights(self) -> None:
        fn = _default_loss_fn(learnable_weights=True)
        fn.freeze_weights()
        assert fn.weights["contrastive"].requires_grad is False
        assert fn.weights["physics"].requires_grad is False

    def test_unfreeze_weights(self) -> None:
        fn = _default_loss_fn(learnable_weights=False)
        fn.unfreeze_weights()
        assert fn.weights["contrastive"].requires_grad is True
        assert fn.weights["physics"].requires_grad is True

    def test_freeze_then_unfreeze_round_trip(self) -> None:
        fn = _default_loss_fn(learnable_weights=True)
        fn.freeze_weights()
        fn.unfreeze_weights()
        assert all(
            fn.weights[name].requires_grad for name in fn.objective_names
        )


# ======================================================================
# Learnable weights tests
# ======================================================================


class TestLearnableWeights:
    """Tests for learnable weight training dynamics."""

    def test_weights_appear_in_parameters(self) -> None:
        fn = _default_loss_fn(learnable_weights=True)
        num = sum(p.numel() for p in fn.parameters() if p.requires_grad)
        assert num == 2

    def test_weights_receive_gradients(self) -> None:
        fn = _default_loss_fn(learnable_weights=True)
        out = fn(*_make_inputs())
        out["loss"].backward()
        assert fn.weights["contrastive"].grad is not None
        assert fn.weights["physics"].grad is not None

    def test_non_learnable_weights_no_gradient(self) -> None:
        fn = _default_loss_fn(learnable_weights=False)
        v1, v2, enc, phy = _make_inputs()
        # A graph must exist (via an input) for backward to run at all.
        enc = enc.detach().requires_grad_(True)
        out = fn(v1, v2, enc, phy)
        out["loss"].backward()
        assert enc.grad is not None
        assert fn.weights["contrastive"].grad is None
        assert fn.weights["physics"].grad is None

    def test_optimizer_step_changes_learnable_weights(self) -> None:
        fn = _default_loss_fn(learnable_weights=True)
        opt = torch.optim.SGD(fn.parameters(), lr=0.1)
        before = fn.get_weights()["contrastive"]
        out = fn(*_make_inputs())
        opt.zero_grad()
        out["loss"].backward()
        opt.step()
        after = fn.get_weights()["contrastive"]
        assert after != pytest.approx(before)

    def test_gradient_flows_through_normalised_learnable_weights(self) -> None:
        fn = _default_loss_fn(
            learnable_weights=True, normalize_weights=True
        )
        out = fn(*_make_inputs())
        out["loss"].backward()
        assert torch.isfinite(fn.weights["contrastive"].grad).all()


# ======================================================================
# Gradient propagation tests
# ======================================================================


class TestGradientPropagation:
    """Tests that gradients reach the input embeddings."""

    def test_gradients_reach_projection_views(self) -> None:
        fn = _default_loss_fn()
        v1, v2, enc, phy = _make_inputs()
        v1 = v1.detach().requires_grad_(True)
        v2 = v2.detach().requires_grad_(True)
        out = fn(v1, v2, enc, phy)
        out["loss"].backward()
        assert v1.grad is not None
        assert v2.grad is not None
        assert torch.isfinite(v1.grad).all()

    def test_gradients_reach_encoder_and_physics(self) -> None:
        fn = _default_loss_fn()
        v1, v2, enc, phy = _make_inputs()
        enc = enc.detach().requires_grad_(True)
        phy = phy.detach().requires_grad_(True)
        out = fn(v1, v2, enc, phy)
        out["loss"].backward()
        assert enc.grad is not None
        assert phy.grad is not None
        assert torch.isfinite(enc.grad).all()
        assert torch.isfinite(phy.grad).all()

    def test_backward_does_not_raise(self) -> None:
        fn = _default_loss_fn(learnable_weights=True)
        v1, v2, enc, phy = _make_inputs()
        v1 = v1.detach().requires_grad_(True)
        enc = enc.detach().requires_grad_(True)
        fn(v1, v2, enc, phy)["loss"].backward()


# ======================================================================
# Determinism tests
# ======================================================================


class TestDeterminism:
    """Tests for deterministic behaviour given identical inputs."""

    def test_same_inputs_same_output(self) -> None:
        fn = _default_loss_fn()
        inputs = _make_inputs(seed=3)
        out1 = fn(*inputs)["loss"].item()
        out2 = fn(*inputs)["loss"].item()
        assert out1 == pytest.approx(out2, abs=0.0)

    def test_independent_instances_agree(self) -> None:
        fn_a = _default_loss_fn(contrastive_weight=2.0, physics_weight=0.5)
        fn_b = _default_loss_fn(contrastive_weight=2.0, physics_weight=0.5)
        inputs = _make_inputs(seed=17)
        assert fn_a(*inputs)["loss"].item() == pytest.approx(
            fn_b(*inputs)["loss"].item(), abs=1e-6
        )


# ======================================================================
# Serialisation tests
# ======================================================================


class TestSerialisation:
    """Tests for state_dict round-trip and deepcopy."""

    def test_state_dict_contains_weights(self) -> None:
        fn = _default_loss_fn()
        sd = fn.state_dict()
        assert "weights.contrastive" in sd
        assert "weights.physics" in sd

    def test_load_state_dict_round_trip(self) -> None:
        fn = _default_loss_fn(contrastive_weight=2.0, physics_weight=3.0)
        fn.set_weights({"contrastive": 7.0, "physics": 0.1})

        fn2 = _default_loss_fn()
        fn2.load_state_dict(fn.state_dict())
        assert fn2.get_weights()["contrastive"] == pytest.approx(7.0)
        assert fn2.get_weights()["physics"] == pytest.approx(0.1)

    def test_save_load_gives_identical_output(self) -> None:
        cfg = ContrastiveTrustLossConfig(
            contrastive_weight=1.5, physics_weight=2.0
        )
        fn = ContrastiveTrustLoss(cfg)

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "ct_loss.pt"
            torch.save(
                {"state_dict": fn.state_dict(), "config": cfg}, path
            )
            checkpoint = torch.load(path, weights_only=False)

        restored = ContrastiveTrustLoss(checkpoint["config"])
        restored.load_state_dict(checkpoint["state_dict"])

        inputs = _make_inputs(seed=88)
        assert fn(*inputs)["loss"].item() == pytest.approx(
            restored(*inputs)["loss"].item(), abs=1e-6
        )

    def test_deepcopy(self) -> None:
        fn = _default_loss_fn(contrastive_weight=2.0)
        cloned = copy.deepcopy(fn)
        inputs = _make_inputs(seed=5)
        assert fn(*inputs)["loss"].item() == pytest.approx(
            cloned(*inputs)["loss"].item(), abs=1e-6
        )


# ======================================================================
# Mixed-precision tests
# ======================================================================


class TestMixedPrecision:
    """Tests for mixed-precision compatibility."""

    def test_autocast_cpu_bfloat16(self) -> None:
        fn = _default_loss_fn()
        inputs = _make_inputs()
        try:
            with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
                out = fn(*inputs)
        except (RuntimeError, ValueError) as exc:  # pragma: no cover
            pytest.skip(f"CPU autocast unsupported: {exc}")
        assert torch.isfinite(out["loss"])

    def test_fp16_inputs(self) -> None:
        fn = _default_loss_fn()
        v1, v2, enc, phy = _make_inputs()
        try:
            out = fn(v1.half(), v2.half(), enc.half(), phy.half())
        except RuntimeError as exc:  # pragma: no cover
            pytest.skip(f"fp16 op unsupported on this platform: {exc}")
        assert torch.isfinite(out["loss"])

    def test_weight_cast_preserves_finiteness(self) -> None:
        # Weights are fp32; losses may be fp16.  Aggregation must stay finite.
        fn = _default_loss_fn(contrastive_weight=2.0, physics_weight=3.0)
        losses = {
            "contrastive": torch.tensor(1.0, dtype=torch.float16),
            "physics": torch.tensor(2.0, dtype=torch.float16),
        }
        agg = fn.compute_total_loss(losses)
        assert torch.isfinite(agg["loss"])
        assert agg["loss"].item() == pytest.approx(2.0 * 1.0 + 3.0 * 2.0, abs=1e-2)


# ======================================================================
# Input validation tests
# ======================================================================


class TestInputValidation:
    """Tests for invalid-input rejection."""

    def test_view1_not_tensor_raises(self) -> None:
        fn = _default_loss_fn()
        _, v2, enc, phy = _make_inputs()
        with pytest.raises(TypeError, match="projection_view_1"):
            fn("bad", v2, enc, phy)  # type: ignore[arg-type]

    def test_encoder_not_tensor_raises(self) -> None:
        fn = _default_loss_fn()
        v1, v2, _, phy = _make_inputs()
        with pytest.raises(TypeError, match="encoder_embedding"):
            fn(v1, v2, 42, phy)  # type: ignore[arg-type]

    def test_integer_dtype_raises(self) -> None:
        fn = _default_loss_fn()
        v1, v2, _, phy = _make_inputs()
        enc = torch.randint(0, 5, (_BATCH, _EMB_DIM))
        with pytest.raises(ValueError, match="floating-point"):
            fn(v1, v2, enc, phy)

    def test_view_1d_raises(self) -> None:
        fn = _default_loss_fn()
        _, v2, enc, phy = _make_inputs()
        v1 = torch.randn(_PROJ_DIM)
        with pytest.raises(ValueError, match="2 dimensions"):
            fn(v1, v2, enc, phy)

    def test_encoder_3d_raises(self) -> None:
        fn = _default_loss_fn()
        v1, v2, _, phy = _make_inputs()
        enc = torch.randn(2, _BATCH, _EMB_DIM)
        with pytest.raises(ValueError, match="2 dimensions"):
            fn(v1, v2, enc, phy)

    def test_view_batch_mismatch_raises(self) -> None:
        fn = _default_loss_fn()
        v1 = F.normalize(torch.randn(4, _PROJ_DIM), p=2, dim=1)
        v2 = F.normalize(torch.randn(8, _PROJ_DIM), p=2, dim=1)
        enc = F.normalize(torch.randn(4, _EMB_DIM), p=2, dim=1)
        phy = F.normalize(torch.randn(4, _EMB_DIM), p=2, dim=1)
        with pytest.raises(ValueError, match="same batch size"):
            fn(v1, v2, enc, phy)

    def test_view_dim_mismatch_raises(self) -> None:
        fn = _default_loss_fn()
        v1 = F.normalize(torch.randn(_BATCH, 64), p=2, dim=1)
        v2 = F.normalize(torch.randn(_BATCH, 128), p=2, dim=1)
        enc = F.normalize(torch.randn(_BATCH, _EMB_DIM), p=2, dim=1)
        phy = F.normalize(torch.randn(_BATCH, _EMB_DIM), p=2, dim=1)
        with pytest.raises(ValueError, match="same embedding"):
            fn(v1, v2, enc, phy)

    def test_encoder_physics_dim_mismatch_raises(self) -> None:
        fn = _default_loss_fn()
        v1, v2, _, _ = _make_inputs()
        enc = F.normalize(torch.randn(_BATCH, 128), p=2, dim=1)
        phy = F.normalize(torch.randn(_BATCH, 256), p=2, dim=1)
        with pytest.raises(ValueError, match="same embedding"):
            fn(v1, v2, enc, phy)

    def test_cross_pair_batch_mismatch_raises(self) -> None:
        fn = _default_loss_fn()
        v1 = F.normalize(torch.randn(_BATCH, _PROJ_DIM), p=2, dim=1)
        v2 = F.normalize(torch.randn(_BATCH, _PROJ_DIM), p=2, dim=1)
        enc = F.normalize(torch.randn(_BATCH + 1, _EMB_DIM), p=2, dim=1)
        phy = F.normalize(torch.randn(_BATCH + 1, _EMB_DIM), p=2, dim=1)
        with pytest.raises(ValueError, match="same batch size"):
            fn(v1, v2, enc, phy)


# ======================================================================
# NaN / Inf handling tests
# ======================================================================


class TestNaNInfHandling:
    """Tests that NaN / Inf inputs are rejected before computation."""

    def test_nan_in_view_raises(self) -> None:
        fn = _default_loss_fn()
        v1, v2, enc, phy = _make_inputs()
        v1[0, 0] = float("nan")
        with pytest.raises(ValueError, match="NaN"):
            fn(v1, v2, enc, phy)

    def test_nan_in_encoder_raises(self) -> None:
        fn = _default_loss_fn()
        v1, v2, enc, phy = _make_inputs()
        enc[1, 2] = float("nan")
        with pytest.raises(ValueError, match="NaN"):
            fn(v1, v2, enc, phy)

    def test_inf_in_view_raises(self) -> None:
        fn = _default_loss_fn()
        v1, v2, enc, phy = _make_inputs()
        v2[0, 0] = float("inf")
        with pytest.raises(ValueError, match="Inf"):
            fn(v1, v2, enc, phy)

    def test_negative_inf_in_physics_raises(self) -> None:
        fn = _default_loss_fn()
        v1, v2, enc, phy = _make_inputs()
        phy[3, 1] = float("-inf")
        with pytest.raises(ValueError, match="Inf"):
            fn(v1, v2, enc, phy)


# ======================================================================
# parameter_summary tests
# ======================================================================


class TestParameterSummary:
    """Tests for parameter_summary()."""

    def test_returns_dict(self) -> None:
        fn = _default_loss_fn()
        assert isinstance(fn.parameter_summary(), dict)

    def test_required_keys(self) -> None:
        fn = _default_loss_fn()
        summary = fn.parameter_summary()
        for key in (
            "objectives", "weights", "effective_weights",
            "normalize_weights", "learnable_weights",
            "log_individual_losses", "num_parameters",
        ):
            assert key in summary, f"Missing key '{key}'"

    def test_num_parameters_counts_weights(self) -> None:
        fn = _default_loss_fn()
        assert fn.parameter_summary()["num_parameters"] == 2

    def test_objectives_reflect_registry(self) -> None:
        fn = _default_loss_fn()
        assert fn.parameter_summary()["objectives"] == ["contrastive", "physics"]


# ======================================================================
# Device tests
# ======================================================================


class TestDevice:
    """Tests for CPU and CUDA execution."""

    def test_cpu_forward(self) -> None:
        fn = _default_loss_fn()
        out = fn(*_make_inputs(device="cpu"))
        assert out["loss"].device.type == "cpu"
        assert torch.isfinite(out["loss"])

    @pytest.mark.skipif(
        not torch.cuda.is_available(), reason="CUDA not available"
    )
    def test_cuda_forward(self) -> None:
        device = torch.device("cuda")
        fn = _default_loss_fn().to(device)
        out = fn(*_make_inputs(device=device))
        assert out["loss"].device.type == "cuda"
        assert torch.isfinite(out["loss"])

    @pytest.mark.skipif(
        not torch.cuda.is_available(), reason="CUDA not available"
    )
    def test_cuda_gradients(self) -> None:
        device = torch.device("cuda")
        fn = _default_loss_fn(learnable_weights=True).to(device)
        v1, v2, enc, phy = _make_inputs(device=device)
        enc = enc.requires_grad_(True)
        out = fn(v1, v2, enc, phy)
        out["loss"].backward()
        assert enc.grad is not None
        assert torch.isfinite(enc.grad).all()
        assert fn.weights["contrastive"].grad is not None

    @pytest.mark.skipif(
        not torch.cuda.is_available(), reason="CUDA not available"
    )
    def test_cuda_matches_cpu(self) -> None:
        inputs_cpu = _make_inputs(device="cpu", seed=321)
        inputs_gpu = tuple(t.cuda() for t in inputs_cpu)

        loss_cpu = _default_loss_fn()(*inputs_cpu)["loss"]
        loss_gpu = _default_loss_fn().cuda()(*inputs_gpu)["loss"]
        assert torch.allclose(loss_cpu, loss_gpu.cpu(), atol=1e-4)


# ======================================================================
# TorchScript tests
# ======================================================================


class TestTorchScript:
    """Best-effort TorchScript compatibility tests."""

    def test_torchscript_trace(self) -> None:
        fn = _default_loss_fn()
        fn.eval()
        inputs = _make_inputs()
        try:
            traced = torch.jit.trace(fn, inputs, strict=False)
            out = traced(*inputs)
        except Exception as exc:
            pytest.skip(f"TorchScript trace unsupported: {exc}")
        assert "loss" in out

    def test_torchscript_save_load(self) -> None:
        fn = _default_loss_fn()
        fn.eval()
        inputs = _make_inputs()
        try:
            traced = torch.jit.trace(fn, inputs, strict=False)
        except Exception as exc:
            pytest.skip(f"TorchScript trace unsupported: {exc}")
        buf = io.BytesIO()
        torch.jit.save(traced, buf)
        buf.seek(0)
        loaded = torch.jit.load(buf)
        out = loaded(*inputs)
        assert torch.isfinite(out["loss"])
