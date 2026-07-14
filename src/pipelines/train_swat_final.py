"""Final multi-source SWaT training pipeline — Config B.

This script performs the final, non-exploratory training run using
the finalized Config B hyperparameters. It is designed to run ONCE,
correctly, and stop after reporting the loss curve for human review.

Splits (by day, zero overlap)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
- **TRAIN** : Feb 19 full day  (pure normal)
- **VAL**   : Feb 20 full day  (pure normal)
- **TEST A**: Dec 2019 full file (reserved — attack labels from
  ``SWaTAttackLabeler``, zero overlap with train/val)
- **TEST B**: Mar 11 full file (reserved — coarse block label,
  zero overlap with train/val)

Config B (finalized)
~~~~~~~~~~~~~~~~~~~~
stride=20, contrastive_weight=1.0, physics_weight=3.0, dropout=0.2,
weight_decay=1e-4, max epochs=50, early stopping patience=8.
"""

from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from preprocessing.cleaner import DataCleaner
from preprocessing.normalizer import FeatureNormalizer
from preprocessing.windowing import SlidingWindowGenerator
from src.data.swat_multi_loader import SWaTMultiLoader
from src.data.view_generator import ContrastiveViewGenerator, ContrastiveViewGeneratorConfig
from src.models.encoder import DualStreamEncoder, EncoderConfig
from src.models.fusion import FusionConfig
from src.models.physics_encoder import PhysicsEncoderConfig
from src.models.temporal_encoder import TemporalEncoderConfig
from src.models.projection_head import ProjectionHead, ProjectionHeadConfig
from src.losses.contrastive_trust_loss import ContrastiveTrustLoss, ContrastiveTrustLossConfig
from src.training.config import OptimizerConfig, SchedulerConfig
from src.training.optimizer_factory import create_optimizer
from src.training.scheduler_factory import create_scheduler
from src.training.trainer import Trainer
from src.training.callbacks import Callback
from src.training.checkpoint import ModelCheckpoint
from src.training.early_stopping import EarlyStopping
from src.training.logger import MetricsLogger
from src.features.channel_alignment import build_typed_frame, NUM_CHANNELS, PHYSICS_DIM

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────
# Config B — finalized, do not change
# ──────────────────────────────────────────────────────────────────────
WINDOW_SIZE = 60
STRIDE = 20
BATCH_SIZE = 32
CONTRASTIVE_WEIGHT = 1.0
PHYSICS_WEIGHT = 3.0
DROPOUT = 0.2
WEIGHT_DECAY = 1e-4
MAX_EPOCHS = 50
PATIENCE = 8
SEED = 42


# ──────────────────────────────────────────────────────────────────────
# Datasets
# ──────────────────────────────────────────────────────────────────────

def physics_vector(window: np.ndarray) -> np.ndarray:
    """Extract a simple physics feature vector from a single window."""
    mean = window.mean(axis=0)
    std = window.std(axis=0)
    roc = (
        np.abs(np.diff(window, axis=0)).mean(axis=0)
        if window.shape[0] > 1
        else np.zeros(window.shape[1])
    )
    return np.concatenate([mean, std, roc]).astype(np.float32)


class ContrastivePretrainDataset(Dataset):
    """Wraps sliding windows for contrastive pre-training."""

    def __init__(
        self, windows: np.ndarray, physics: np.ndarray, seed: int = 0
    ) -> None:
        self.windows = torch.from_numpy(windows)
        self.physics = torch.from_numpy(physics)
        self.view_gen = ContrastiveViewGenerator(
            ContrastiveViewGeneratorConfig(seed=seed)
        )

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        w = self.windows[idx]
        v1, v2 = self.view_gen.generate(w)
        p = self.physics[idx]
        return {
            "view1_window": v1.float(),
            "view1_physics": p,
            "view2_window": v2.float(),
            "view2_physics": p,
        }


# ──────────────────────────────────────────────────────────────────────
# Pipeline
# ──────────────────────────────────────────────────────────────────────

def _split_by_day(
    pooled: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split pooled DataFrame into train, val, test_a, test_b by source_day.

    Returns
    -------
    train_df, val_df, test_a_df, test_b_df
    """
    train_df = pooled[pooled["source_day"] == "feb19"].copy()
    val_df = pooled[pooled["source_day"] == "feb20"].copy()
    test_a_df = pooled[pooled["source_day"] == "dec2019"].copy()
    test_b_df = pooled[pooled["source_day"] == "mar11"].copy()
    return train_df, val_df, test_a_df, test_b_df


def _verify_zero_overlap(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_a_df: pd.DataFrame,
    test_b_df: pd.DataFrame,
) -> None:
    """Print row counts and date ranges, and assert zero overlap."""
    splits = {
        "TRAIN (Feb19)": train_df,
        "VAL   (Feb20)": val_df,
        "TEST_A (Dec2019)": test_a_df,
        "TEST_B (Mar11)": test_b_df,
    }

    logger.info("=" * 72)
    logger.info("SPLIT VERIFICATION — zero-overlap check")
    logger.info("=" * 72)

    for name, df in splits.items():
        ts = pd.to_datetime(df["t_stamp"], errors="coerce")
        ts_min = ts.min()
        ts_max = ts.max()
        source_days = sorted(df["source_day"].unique())
        label_dist = df["label"].value_counts().to_dict()
        logger.info(
            "%-20s | rows=%6d | range=[%s → %s] | "
            "source_days=%s | label_dist=%s",
            name,
            len(df),
            ts_min,
            ts_max,
            source_days,
            label_dist,
        )

    # Assert no source_day appears in more than one split
    train_days = set(train_df["source_day"].unique())
    val_days = set(val_df["source_day"].unique())
    testa_days = set(test_a_df["source_day"].unique())
    testb_days = set(test_b_df["source_day"].unique())

    all_pairs = [
        ("train", train_days, "val", val_days),
        ("train", train_days, "test_a", testa_days),
        ("train", train_days, "test_b", testb_days),
        ("val", val_days, "test_a", testa_days),
        ("val", val_days, "test_b", testb_days),
        ("test_a", testa_days, "test_b", testb_days),
    ]
    for name_a, days_a, name_b, days_b in all_pairs:
        overlap = days_a & days_b
        if overlap:
            raise RuntimeError(
                f"DATA LEAKAGE: {name_a} and {name_b} share "
                f"source_day(s): {sorted(overlap)}"
            )

    logger.info("✓ Zero source_day overlap confirmed across all splits.")
    logger.info("=" * 72)


def _clean_split(
    df: pd.DataFrame, split_name: str
) -> pd.DataFrame:
    """Clean a single split with the DataCleaner."""
    logger.info("Cleaning %s (%d rows)...", split_name, len(df))

    # Preserve source_day and label during cleaning — add them to the
    # preserved set by passing label_column explicitly; source_day is not
    # a standard candidate so we drop it before cleaning and re-attach.
    source_day_values = df["source_day"].to_numpy()
    label_values = df["label"].to_numpy()
    working = df.drop(columns=["source_day", "label"])

    cleaner = DataCleaner(
        timestamp_column="t_stamp",
        missing_value_strategy="forward_fill",
    )
    cleaned, meta = cleaner.clean(working)

    # Re-attach preserved metadata columns.
    cleaned["label"] = label_values[: len(cleaned)]
    cleaned["source_day"] = source_day_values[: len(cleaned)]

    logger.info(
        "%s cleaned: %s → %s rows | Bad Input coerced: %s | "
        "missing remaining: %s",
        split_name,
        meta["input_shape"][0],
        meta["output_shape"][0],
        meta.get("operations", [{}])[0].get("details", {}).get(
            "total_coerced", 0
        )
        if meta.get("operations")
        else 0,
        meta["missing_values_remaining"],
    )
    return cleaned


def _prepare_windows(
    df: pd.DataFrame,
    feature_columns: list[str],
    normalizer: FeatureNormalizer,
    split_name: str,
    fit: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """Project → normalize → window → physics vectors."""
    typed = build_typed_frame(df, feature_columns, keep=["t_stamp", "label"])

    if fit:
        norm = normalizer.fit_transform(
            typed.drop(columns=["label"], errors="ignore")
        )
    else:
        norm = normalizer.transform(
            typed.drop(columns=["label"], errors="ignore")
        )

    # Re-attach label for windowing label extraction later if needed.
    if "label" in typed.columns:
        norm["label"] = typed["label"].to_numpy()
    if "t_stamp" not in norm.columns and "t_stamp" in typed.columns:
        norm["t_stamp"] = typed["t_stamp"].to_numpy()

    win_gen = SlidingWindowGenerator(
        window_size=WINDOW_SIZE,
        stride=STRIDE,
        timestamp_column="t_stamp",
    )
    batch = win_gen.generate(norm)
    windows = batch.windows.astype(np.float32)
    physics = np.stack([physics_vector(w) for w in windows])

    logger.info(
        "%s: %d windows of shape (%d, %d), physics (%d,)",
        split_name,
        windows.shape[0],
        windows.shape[1],
        windows.shape[2],
        physics.shape[1],
    )
    return windows, physics


def _get_feature_columns(df: pd.DataFrame) -> list[str]:
    """Return feature columns (all except metadata/label/timestamp)."""
    exclude = {"t_stamp", "label", "source_day"}
    return [c for c in df.columns if c not in exclude]


def main() -> None:
    """Run the final Config B training pipeline."""
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    data_dir = Path("data/raw/SWaT")
    log_dir = Path("logs/final_configB")
    out_dir = Path("outputs/final_configB")
    log_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Device: %s", device)

    # ── 1. Load all sources ──────────────────────────────────────────
    t0 = time.time()
    loader = SWaTMultiLoader(data_dir)
    multi_data = loader.load()
    pooled = multi_data.dataframe
    logger.info("Loaded %d pooled rows in %.1fs", len(pooled), time.time() - t0)

    # ── 2. Split by day ──────────────────────────────────────────────
    train_df, val_df, test_a_df, test_b_df = _split_by_day(pooled)

    # ── 3. Verify zero overlap ───────────────────────────────────────
    _verify_zero_overlap(train_df, val_df, test_a_df, test_b_df)

    # ── 4. Clean each split independently ────────────────────────────
    #    (each goes through Bad Input → NaN, Active/Inactive → 1/0,
    #     numeric coercion, forward-fill imputation)
    train_clean = _clean_split(train_df, "TRAIN")
    val_clean = _clean_split(val_df, "VAL")
    # Test sets cleaned but NOT used for training or normalization fitting.
    test_a_clean = _clean_split(test_a_df, "TEST_A")
    test_b_clean = _clean_split(test_b_df, "TEST_B")

    # ── 5. Feature columns ───────────────────────────────────────────
    feature_cols = _get_feature_columns(train_clean)
    logger.info("Feature columns (%d): %s", len(feature_cols), feature_cols[:10])

    # ── 6. Normalize (fit on train only) and window ──────────────────
    normalizer = FeatureNormalizer(timestamp_column="t_stamp")

    train_win, train_phy = _prepare_windows(
        train_clean, feature_cols, normalizer, "TRAIN", fit=True
    )
    val_win, val_phy = _prepare_windows(
        val_clean, feature_cols, normalizer, "VAL", fit=False
    )

    # Sanity: no NaN/inf in windowed data
    for name, arr in [
        ("train_win", train_win), ("train_phy", train_phy),
        ("val_win", val_win), ("val_phy", val_phy),
    ]:
        assert np.isfinite(arr).all(), f"NaN/inf detected in {name}!"
    logger.info("✓ No NaN/inf in train/val windowed data.")

    # ── 7. Build DataLoaders ─────────────────────────────────────────
    train_ds = ContrastivePretrainDataset(train_win, train_phy, seed=SEED)
    val_ds = ContrastivePretrainDataset(val_win, val_phy, seed=SEED + 1)
    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False,
    )

    logger.info(
        "DataLoaders: train=%d batches, val=%d batches",
        len(train_loader),
        len(val_loader),
    )

    # ── 8. Build model (Config B) ────────────────────────────────────
    encoder_config = EncoderConfig(
        temporal=TemporalEncoderConfig(
            input_channels=NUM_CHANNELS, dropout=DROPOUT,
        ),
        physics=PhysicsEncoderConfig(
            input_dim=PHYSICS_DIM, dropout=DROPOUT,
        ),
    )
    proj_config = ProjectionHeadConfig(
        input_dim=encoder_config.temporal.embedding_dim, dropout=DROPOUT,
    )
    loss_config = ContrastiveTrustLossConfig()

    encoder = DualStreamEncoder(encoder_config).to(device)
    projection_head = ProjectionHead(proj_config).to(device)
    loss_fn = ContrastiveTrustLoss(loss_config).to(device)
    loss_fn.set_weights({
        "contrastive": CONTRASTIVE_WEIGHT,
        "physics": PHYSICS_WEIGHT,
    })

    param_count = encoder.parameter_count()
    logger.info(
        "Model built | params=%s | dropout=%.2f | "
        "contrastive_weight=%.1f | physics_weight=%.1f",
        param_count, DROPOUT, CONTRASTIVE_WEIGHT, PHYSICS_WEIGHT,
    )

    # ── 9. Train ─────────────────────────────────────────────────────
    logger.info(
        "Training Config B: max_epochs=%d, patience=%d, "
        "weight_decay=%.0e, stride=%d",
        MAX_EPOCHS, PATIENCE, WEIGHT_DECAY, STRIDE,
    )

    opt_config = OptimizerConfig(
        name="AdamW", lr=1e-3, weight_decay=WEIGHT_DECAY,
    )
    optimizer = create_optimizer(
        list(encoder.parameters()) + list(projection_head.parameters()),
        opt_config,
    )
    sched_config = SchedulerConfig(
        name="CosineAnnealingLR", kwargs={"T_max": MAX_EPOCHS},
    )
    scheduler = create_scheduler(optimizer, sched_config)

    callbacks: list[Callback] = [
        MetricsLogger(log_dir=log_dir),
        ModelCheckpoint(
            filepath=log_dir / "best_model.pt",
            monitor="val_loss",
            mode="min",
        ),
        EarlyStopping(
            monitor="val_loss", patience=PATIENCE, mode="min",
        ),
    ]

    trainer = Trainer(
        encoder=encoder,
        projection_head=projection_head,
        loss_fn=loss_fn,
        optimizer=optimizer,
        scheduler=scheduler,
        mixed_precision=(device == "cuda"),
        gradient_accumulation_steps=1,
        callbacks=callbacks,
        device=device,
    )

    history: List[Dict[str, Any]] = trainer.fit(
        train_loader, val_loader, epochs=MAX_EPOCHS,
    )

    # ── 10. Verify loss curve ────────────────────────────────────────
    logger.info("=" * 72)
    logger.info("TRAINING COMPLETE — Loss Curve Report")
    logger.info("=" * 72)

    has_nan_inf = False
    for h in history:
        epoch = h["epoch"]
        tl = h["train_loss"]
        vl = h["val_loss"]
        if not (np.isfinite(tl) and np.isfinite(vl)):
            has_nan_inf = True
            logger.error(
                "⚠ Epoch %03d: NaN/inf detected! train_loss=%.6f val_loss=%.6f",
                epoch, tl, vl,
            )
        else:
            logger.info(
                "Epoch %03d: train_loss=%.6f  val_loss=%.6f",
                epoch, tl, vl,
            )

    if has_nan_inf:
        logger.error("❌ TRAINING DIVERGED — NaN/inf values detected in loss.")
    else:
        logger.info("✓ No NaN/inf in any epoch.")

    # ── 11. Save outputs ─────────────────────────────────────────────
    tables_dir = out_dir / "tables"
    figures_dir = out_dir / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    # Save history JSON
    with open(tables_dir / "training_history.json", "w") as f:
        json.dump(history, f, indent=2)
    logger.info("Saved training history to %s", tables_dir / "training_history.json")

    # Save config summary
    config_summary = {
        "window_size": WINDOW_SIZE,
        "stride": STRIDE,
        "batch_size": BATCH_SIZE,
        "contrastive_weight": CONTRASTIVE_WEIGHT,
        "physics_weight": PHYSICS_WEIGHT,
        "dropout": DROPOUT,
        "weight_decay": WEIGHT_DECAY,
        "max_epochs": MAX_EPOCHS,
        "patience": PATIENCE,
        "seed": SEED,
        "device": device,
        "param_count": param_count,
        "actual_epochs": len(history),
        "best_val_loss": min(h["val_loss"] for h in history),
        "final_train_loss": history[-1]["train_loss"],
        "final_val_loss": history[-1]["val_loss"],
        "splits": {
            "train": {
                "source": "feb19",
                "rows": len(train_clean),
                "windows": len(train_win),
            },
            "val": {
                "source": "feb20",
                "rows": len(val_clean),
                "windows": len(val_win),
            },
            "test_a": {
                "source": "dec2019",
                "rows": len(test_a_clean),
                "status": "reserved (not used in training)",
            },
            "test_b": {
                "source": "mar11",
                "rows": len(test_b_clean),
                "status": "reserved (not used in training)",
            },
        },
    }
    with open(tables_dir / "config_summary.json", "w") as f:
        json.dump(config_summary, f, indent=2)
    logger.info("Saved config summary to %s", tables_dir / "config_summary.json")

    # ── 12. Plot loss curve ──────────────────────────────────────────
    epochs_list = [h["epoch"] for h in history]
    train_losses = [h["train_loss"] for h in history]
    val_losses = [h["val_loss"] for h in history]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs_list, train_losses, "o-", label="Train loss", linewidth=2)
    ax.plot(epochs_list, val_losses, "s-", label="Val loss", linewidth=2)
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("ContrastiveTrust Loss", fontsize=12)
    ax.set_title(
        f"Config B Final Run — "
        f"cw={CONTRASTIVE_WEIGHT}, pw={PHYSICS_WEIGHT}, "
        f"wd={WEIGHT_DECAY:.0e}, do={DROPOUT}",
        fontsize=11,
    )
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(figures_dir / "loss_curve.png", dpi=200)
    fig.savefig(figures_dir / "loss_curve.pdf")
    plt.close(fig)
    logger.info("Saved loss curve to %s", figures_dir / "loss_curve.png")

    # Also save a zoomed version excluding epoch 1 if > 3 epochs
    if len(history) > 3:
        fig2, ax2 = plt.subplots(figsize=(8, 5))
        ax2.plot(
            epochs_list[1:], train_losses[1:], "o-",
            label="Train loss", linewidth=2,
        )
        ax2.plot(
            epochs_list[1:], val_losses[1:], "s-",
            label="Val loss", linewidth=2,
        )
        ax2.set_xlabel("Epoch", fontsize=12)
        ax2.set_ylabel("ContrastiveTrust Loss", fontsize=12)
        ax2.set_title("Config B — Zoomed (epoch 2+)", fontsize=11)
        ax2.legend(fontsize=11)
        ax2.grid(True, alpha=0.3)
        fig2.tight_layout()
        fig2.savefig(figures_dir / "loss_curve_zoomed.png", dpi=200)
        fig2.savefig(figures_dir / "loss_curve_zoomed.pdf")
        plt.close(fig2)
        logger.info(
            "Saved zoomed loss curve to %s",
            figures_dir / "loss_curve_zoomed.png",
        )

    # ── 13. Final summary ────────────────────────────────────────────
    logger.info("=" * 72)
    logger.info("PIPELINE STOPPED — awaiting human review of loss curve")
    logger.info(
        "Best val loss: %.6f at epoch %d",
        config_summary["best_val_loss"],
        min(history, key=lambda h: h["val_loss"])["epoch"],
    )
    logger.info(
        "Final (epoch %d): train_loss=%.6f, val_loss=%.6f",
        history[-1]["epoch"],
        history[-1]["train_loss"],
        history[-1]["val_loss"],
    )
    logger.info(
        "Model checkpoint: %s",
        log_dir / "best_model.pt",
    )
    logger.info(
        "Loss curve: %s",
        figures_dir / "loss_curve.png",
    )
    logger.info("=" * 72)


if __name__ == "__main__":
    main()
