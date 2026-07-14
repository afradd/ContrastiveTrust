"""Real, data-driven integration pipeline for ContrastiveTrust.

Loads real SWaT (normal) data, trains the DualStreamEncoder with the
real ContrastiveTrustLoss via the real Trainer, builds a zero-shot detector
from the trained encoder, and evaluates it on real, labeled HAI test data.
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import time
from pathlib import Path
from typing import Dict, Any, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_curve, precision_recall_curve, confusion_matrix
)
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from preprocessing.cleaner import DataCleaner
from preprocessing.normalizer import FeatureNormalizer
from preprocessing.windowing import SlidingWindowGenerator
from src.data.swat_loader import SWaTLoader
from src.data.swat_attack_labeler import SWaTAttackLabeler
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
from src.evaluation.embedding_bank import EmbeddingBank
from src.evaluation.anomaly_scorer import AnomalyScorer
from src.evaluation.threshold import ThresholdEstimator
from src.evaluation.zero_shot_detector import ZeroShotDetector
from src.features.channel_alignment import build_typed_frame, NUM_CHANNELS, PHYSICS_DIM

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)



def physics_vector(window: np.ndarray) -> np.ndarray:
    mean = window.mean(axis=0)
    std = window.std(axis=0)
    roc = np.abs(np.diff(window, axis=0)).mean(axis=0) if window.shape[0] > 1 else np.zeros(window.shape[1])
    return np.concatenate([mean, std, roc]).astype(np.float32)

class ContrastivePretrainDataset(Dataset):
    def __init__(self, windows: np.ndarray, physics: np.ndarray, seed: int = 0):
        self.windows = torch.from_numpy(windows)
        self.physics = torch.from_numpy(physics)
        self.view_gen = ContrastiveViewGenerator(ContrastiveViewGeneratorConfig(seed=seed))

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        w = self.windows[idx]
        v1, v2 = self.view_gen.generate(w)
        p = self.physics[idx]
        return {
            "view1_window": v1.float(),
            "view1_physics": p,
            "view2_window": v2.float(),
            "view2_physics": p,
        }

class ScoringDataset(Dataset):
    def __init__(self, windows: np.ndarray, physics: np.ndarray, labels: np.ndarray = None):
        self.windows = torch.from_numpy(windows).float()
        self.physics = torch.from_numpy(physics).float()
        self.labels = torch.from_numpy(labels).float() if labels is not None else None

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        if self.labels is not None:
            return self.windows[idx], self.physics[idx], self.labels[idx]
        return self.windows[idx], self.physics[idx]


# ---------------------------------------------------------------------------
# Pipeline steps
# ---------------------------------------------------------------------------
def load_data(
    swat_path: str,
    hai_path: str,
    swat_meta_path: str,
    window_size: int,
    stride: int,
    batch_size: int
) -> Tuple[DataLoader, DataLoader, DataLoader, DataLoader, DataLoader, DataLoader]:
    """Loads and preprocesses real data, returning DataLoaders for train, val, test, and evaluation."""
    logger.info("Loading SWaT and HAI data...")
    t0 = time.time()
    swat_df = pd.read_pickle(swat_path) if swat_path.endswith('.pkl') else pd.read_csv(swat_path)
    hai_df = pd.read_pickle(hai_path) if hai_path.endswith('.pkl') else pd.read_csv(hai_path)
    logger.info(f"SWaT: {swat_df.shape}, HAI: {hai_df.shape} ({time.time()-t0:.1f}s)")
    
    # Feature cols
    if "t_stamp" not in swat_df.columns and "timestamp" in swat_df.columns:
        swat_df = swat_df.rename(columns={"timestamp": "t_stamp"})

    # Extract swat features. Usually we use an excel loader, but here we can just use columns.
    swat_feature_cols = [
        c for c in swat_df.columns if c not in ("t_stamp",) and not c.endswith(".Alarm")
        and not c.startswith("P") or c.startswith(("P1_STATE",))
    ]
    if swat_meta_path:
        loader_tmp = SWaTLoader(file_path=swat_meta_path)
        swat_sensor_cols = loader_tmp._categorize_sensor_columns(swat_df.columns)
        swat_actuator_cols = loader_tmp._categorize_actuator_columns(swat_df.columns)
        swat_feature_cols = swat_sensor_cols + swat_actuator_cols

    hai_feature_cols = [c for c in hai_df.columns if c not in ("timestamp", "label")]

    # Label SWaT dataset
    logger.info("Labeling SWaT data...")
    swat_df = SWaTAttackLabeler(timestamp_column="t_stamp").label(swat_df)

    logger.info("Cleaning data...")
    swat_clean, swat_meta = DataCleaner(timestamp_column="t_stamp").clean(swat_df)
    hai_clean, hai_meta = DataCleaner(timestamp_column="timestamp", label_column="label").clean(hai_df)

    logger.info("Grouping columns by physics type...")
    swat_typed = build_typed_frame(swat_clean, swat_feature_cols, keep=["t_stamp", "label"])
    hai_typed = build_typed_frame(hai_clean, hai_feature_cols, keep=["timestamp", "label"])

    logger.info("Chronological split (SWaT: 70/15/15 train/val/test-normal)...")
    n = len(swat_typed)
    n_train, n_val = int(n * 0.70), int(n * 0.15)
    swat_train = swat_typed.iloc[:n_train].reset_index(drop=True)
    swat_val = swat_typed.iloc[n_train:n_train + n_val].reset_index(drop=True)
    swat_test = swat_typed.iloc[n_train + n_val:].reset_index(drop=True)

    logger.info("Normalizing...")
    normalizer = FeatureNormalizer(timestamp_column="t_stamp")
    swat_train_norm = normalizer.fit_transform(swat_train.drop(columns=["label"], errors="ignore"))
    if "label" in swat_train.columns:
        swat_train_norm["label"] = swat_train["label"].to_numpy()
    if "t_stamp" not in swat_train_norm.columns and "t_stamp" in swat_train.columns:
        swat_train_norm["t_stamp"] = swat_train["t_stamp"].to_numpy()

    swat_val_norm = normalizer.transform(swat_val.drop(columns=["label"], errors="ignore"))
    if "label" in swat_val.columns:
        swat_val_norm["label"] = swat_val["label"].to_numpy()
    if "t_stamp" not in swat_val_norm.columns and "t_stamp" in swat_val.columns:
        swat_val_norm["t_stamp"] = swat_val["t_stamp"].to_numpy()

    swat_test_norm = normalizer.transform(swat_test.drop(columns=["label"], errors="ignore"))
    if "label" in swat_test.columns:
        swat_test_norm["label"] = swat_test["label"].to_numpy()
    if "t_stamp" not in swat_test_norm.columns and "t_stamp" in swat_test.columns:
        swat_test_norm["t_stamp"] = swat_test["t_stamp"].to_numpy()

    hai_normalizer_input = hai_typed.rename(columns={"timestamp": "t_stamp"})
    hai_norm = normalizer.transform(hai_normalizer_input.drop(columns=["label"], errors="ignore"))
    if "label" in hai_typed.columns:
        hai_norm["label"] = hai_typed["label"].to_numpy()
    if "t_stamp" in hai_norm.columns:
        hai_norm = hai_norm.rename(columns={"t_stamp": "timestamp"})
    if "timestamp" not in hai_norm.columns and "timestamp" in hai_typed.columns:
        hai_norm["timestamp"] = hai_typed["timestamp"].to_numpy()

    logger.info("Extracting sliding windows...")
    win_gen_swat = SlidingWindowGenerator(window_size=window_size, stride=stride, timestamp_column="t_stamp")
    
    # SWAT TRAIN
    swat_train_win = win_gen_swat.generate(swat_train_norm).windows.astype(np.float32)
    
    # SWAT VAL
    swat_val_win = win_gen_swat.generate(swat_val_norm).windows.astype(np.float32)
    
    # SWAT TEST (with labels)
    has_labels_swat = "label" in swat_test_norm.columns
    win_gen_swat_test = SlidingWindowGenerator(
        window_size=window_size, stride=stride, timestamp_column="t_stamp",
        label_column="label" if has_labels_swat else None,
        return_labels=has_labels_swat, label_method="max" if has_labels_swat else None,
    )
    swat_test_batch = win_gen_swat_test.generate(swat_test_norm)
    swat_test_win = swat_test_batch.windows.astype(np.float32)
    swat_test_labels = swat_test_batch.labels.astype(np.int64) if has_labels_swat else None

    has_labels = "label" in hai_norm.columns
    win_gen_hai = SlidingWindowGenerator(
        window_size=window_size, stride=stride, timestamp_column="timestamp",
        label_column="label" if has_labels else None, 
        return_labels=has_labels, label_method="max" if has_labels else None,
    )
    hai_batch = win_gen_hai.generate(hai_norm)
    hai_win = hai_batch.windows.astype(np.float32)
    hai_labels = hai_batch.labels.astype(np.int64) if has_labels else np.zeros(len(hai_win), dtype=np.int64)

    logger.info("Extracting physics vectors...")
    swat_train_phy = np.stack([physics_vector(w) for w in swat_train_win])
    swat_val_phy = np.stack([physics_vector(w) for w in swat_val_win])
    swat_test_phy = np.stack([physics_vector(w) for w in swat_test_win])
    hai_phy = np.stack([physics_vector(w) for w in hai_win])

    # Datasets and Loaders
    train_ds = ContrastivePretrainDataset(swat_train_win, swat_train_phy, seed=0)
    val_ds = ContrastivePretrainDataset(swat_val_win, swat_val_phy, seed=1)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    normal_ds = ScoringDataset(swat_train_win, swat_train_phy)
    normal_loader = DataLoader(normal_ds, batch_size=batch_size, shuffle=False)
    
    val_score_ds = ScoringDataset(swat_val_win, swat_val_phy)
    val_score_loader = DataLoader(val_score_ds, batch_size=batch_size, shuffle=False)

    swat_test_ds = ScoringDataset(swat_test_win, swat_test_phy, swat_test_labels)
    swat_test_loader = DataLoader(swat_test_ds, batch_size=batch_size, shuffle=False)

    hai_ds = ScoringDataset(hai_win, hai_phy, hai_labels)
    hai_loader_eval = DataLoader(hai_ds, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, normal_loader, val_score_loader, swat_test_loader, hai_loader_eval

def build_model(device: str, contrastive_weight: float = 1.0, physics_weight: float = 1.0,
                dropout: float = 0.2, small_model: bool = False) -> Tuple[DualStreamEncoder, ProjectionHead, ContrastiveTrustLoss]:
    if small_model:
        encoder_config = EncoderConfig(
            temporal=TemporalEncoderConfig(
                input_channels=NUM_CHANNELS, embedding_dim=128,
                hidden_channels=(32, 64, 128), kernel_sizes=(7, 5, 3), dropout=dropout,
            ),
            physics=PhysicsEncoderConfig(
                input_dim=PHYSICS_DIM, embedding_dim=128,
                hidden_dims=(256, 128), dropout=dropout,
            ),
            fusion=FusionConfig(embedding_dim=128),
        )
        proj_config = ProjectionHeadConfig(
            input_dim=128, hidden_dim=128, output_dim=64, dropout=dropout,
        )
    else:
        encoder_config = EncoderConfig(
            temporal=TemporalEncoderConfig(input_channels=NUM_CHANNELS, dropout=dropout),
            physics=PhysicsEncoderConfig(input_dim=PHYSICS_DIM, dropout=dropout),
        )
        proj_config = ProjectionHeadConfig(
            input_dim=encoder_config.temporal.embedding_dim, dropout=dropout,
        )
    loss_config = ContrastiveTrustLossConfig()

    encoder = DualStreamEncoder(encoder_config).to(device)
    projection_head = ProjectionHead(proj_config).to(device)
    loss_fn = ContrastiveTrustLoss(loss_config).to(device)
    loss_fn.set_weights({"contrastive": contrastive_weight, "physics": physics_weight})
    logger.info("Model built | small_model=%s | dropout=%.2f | params=%s",
                small_model, dropout, encoder.parameter_count())
    return encoder, projection_head, loss_fn

def train(
    encoder: DualStreamEncoder,
    projection_head: ProjectionHead,
    loss_fn: ContrastiveTrustLoss,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int,
    log_dir: Path,
    device: str,
    weight_decay: float = 1e-4,
    patience: int = 8,
) -> list[Dict[str, float]]:
    logger.info(f"Training for {epochs} epochs | weight_decay={weight_decay} | patience={patience}")
    opt_config = OptimizerConfig(name="AdamW", lr=1e-3, weight_decay=weight_decay)
    optimizer = create_optimizer(list(encoder.parameters()) + list(projection_head.parameters()), opt_config)
    sched_config = SchedulerConfig(name="CosineAnnealingLR", kwargs={"T_max": epochs})
    scheduler = create_scheduler(optimizer, sched_config)

    callbacks: list[Callback] = [
        MetricsLogger(log_dir=log_dir),
        ModelCheckpoint(filepath=log_dir / "best_model.pt", monitor="val_loss", mode="min"),
        EarlyStopping(monitor="val_loss", patience=patience, mode="min"),
    ]

    trainer = Trainer(
        encoder=encoder, projection_head=projection_head, loss_fn=loss_fn,
        optimizer=optimizer, scheduler=scheduler, mixed_precision=(device == "cuda"),
        gradient_accumulation_steps=1, callbacks=callbacks, device=device,
    )

    history = trainer.fit(train_loader, val_loader, epochs=epochs)
    return history

def build_detector(
    encoder: DualStreamEncoder,
    normal_loader: DataLoader,
    val_score_loader: DataLoader,
    device: str
) -> ZeroShotDetector:
    logger.info("Building and fitting ZeroShotDetector...")
    encoder.eval()
    embedding_dim = encoder.config.temporal.embedding_dim
    bank = EmbeddingBank(embedding_dim=embedding_dim, device=device, normalize=True)
    scorer = AnomalyScorer(bank=bank, metric="cosine", strategy="robust_z", k=5)
    threshold_estimator = ThresholdEstimator(strategy="percentile", percentile=95.0)

    detector = ZeroShotDetector(encoder=encoder, scorer=scorer, threshold_estimator=threshold_estimator, device=device)
    detector.fit(normal_loader, val_score_loader)
    
    logger.info(f"Embedding bank size: {len(bank)}")
    logger.info(f"Fitted threshold: {threshold_estimator.predict_threshold():.4f}")
    return detector

def evaluate(
    detector: ZeroShotDetector,
    hai_loader_eval: DataLoader,
    swat_test_loader: DataLoader,
    history: list[Dict[str, float]],
    out_dir: Path,
    device: str
):
    # --- HAI Evaluation ---
    logger.info("Evaluating on HAI test set...")
    hai_metrics = detector.evaluate(hai_loader_eval)
    
    all_scores, all_labels = [], []
    for window, physics, labels in hai_loader_eval:
        scores = detector.score(window, physics)
        all_scores.append(scores.cpu().view(-1).numpy())
        all_labels.append(labels.cpu().view(-1).numpy())
    hai_scores = np.concatenate(all_scores)
    hai_labels_arr = np.concatenate(all_labels).astype(int)
    hai_threshold = detector.threshold_estimator.predict_threshold()
    hai_preds = (hai_scores > hai_threshold).astype(int)
    
    hai_full_metrics = {
        "accuracy": float(accuracy_score(hai_labels_arr, hai_preds)),
        "precision": float(precision_score(hai_labels_arr, hai_preds, zero_division=0)),
        "recall": float(recall_score(hai_labels_arr, hai_preds, zero_division=0)),
        "f1": float(f1_score(hai_labels_arr, hai_preds, zero_division=0)),
        "roc_auc": float(hai_metrics["auroc"]),
        "pr_auc": float(hai_metrics["auprc"]),
        "threshold": float(hai_threshold),
        "n_windows": int(len(hai_labels_arr)),
        "n_attack_windows": int(hai_labels_arr.sum()),
    }
    
    logger.info(f"HAI zero-shot metrics: {json.dumps(hai_full_metrics, indent=2)}")

    # --- SWaT Evaluation ---
    logger.info("Evaluating on SWaT test set...")
    swat_metrics = detector.evaluate(swat_test_loader)
    
    all_swat_scores, all_swat_labels = [], []
    for batch in swat_test_loader:
        if len(batch) == 3:
            window, physics, labels = batch
        else:
            window, physics = batch
            labels = torch.zeros(window.shape[0])
            
        scores = detector.score(window, physics)
        all_swat_scores.append(scores.cpu().view(-1).numpy())
        all_swat_labels.append(labels.cpu().view(-1).numpy())
        
    swat_scores = np.concatenate(all_swat_scores)
    swat_labels_arr = np.concatenate(all_swat_labels).astype(int)
    swat_preds = (swat_scores > hai_threshold).astype(int)
    
    swat_full_metrics = {
        "accuracy": float(accuracy_score(swat_labels_arr, swat_preds)),
        "precision": float(precision_score(swat_labels_arr, swat_preds, zero_division=0)),
        "recall": float(recall_score(swat_labels_arr, swat_preds, zero_division=0)),
        "f1": float(f1_score(swat_labels_arr, swat_preds, zero_division=0)),
        "roc_auc": float(swat_metrics["auroc"]),
        "pr_auc": float(swat_metrics["auprc"]),
        "threshold": float(hai_threshold),
        "n_windows": int(len(swat_labels_arr)),
        "n_attack_windows": int(swat_labels_arr.sum()),
    }
    
    logger.info(f"SWaT in-domain metrics: {json.dumps(swat_full_metrics, indent=2)}")
    
    # Compute Delta-F1
    delta_f1 = hai_full_metrics["f1"] - swat_full_metrics["f1"]
    logger.info(f"Delta-F1 Transfer Gap (HAI - SWaT): {delta_f1:.4f}")

    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "tables").mkdir(parents=True, exist_ok=True)
        (out_dir / "figures").mkdir(parents=True, exist_ok=True)
        
        with open(out_dir / "tables" / "training_history.json", "w") as f:
            json.dump(history, f, indent=2)
        with open(out_dir / "tables" / "hai_zero_shot_metrics.json", "w") as f:
            json.dump(hai_full_metrics, f, indent=2)
        with open(out_dir / "tables" / "swat_indomain_metrics.json", "w") as f:
            json.dump(swat_full_metrics, f, indent=2)
        
        metrics_df = pd.DataFrame([
            {"dataset": "HAI", **hai_full_metrics},
            {"dataset": "SWaT", **swat_full_metrics},
            {"dataset": "Delta", "f1": delta_f1}
        ])
        metrics_df.to_csv(out_dir / "tables" / "metrics.csv", index=False)

        # Plotting
        logger.info("Generating plots...")
        FIG = out_dir / "figures"
        
        # Training curve
        epochs_ = [h["epoch"] for h in history]
        train_losses = [h["train_loss"] for h in history]
        val_losses = [h["val_loss"] for h in history]
        plt.figure(figsize=(6, 4))
        plt.plot(epochs_, train_losses, marker="o", label="Train loss (SWaT)")
        plt.plot(epochs_, val_losses, marker="o", label="Val loss (SWaT)")
        plt.xlabel("Epoch"); plt.ylabel("ContrastiveTrust loss"); plt.title("Training Curve")
        plt.legend(); plt.tight_layout()
        plt.savefig(FIG / "training_loss_curve.pdf"); plt.savefig(FIG / "training_loss_curve.png", dpi=150); plt.close()

        # Score distribution
        plt.figure(figsize=(7, 4.5))
        plt.hist(swat_scores[swat_labels_arr == 0], bins=30, alpha=0.5, label=f"SWaT normal (n={(swat_labels_arr==0).sum()})", density=True)
        if (swat_labels_arr == 1).sum() > 0:
            plt.hist(swat_scores[swat_labels_arr == 1], bins=30, alpha=0.5, label=f"SWaT attack (n={(swat_labels_arr==1).sum()})", density=True)
        plt.hist(hai_scores[hai_labels_arr == 0], bins=30, alpha=0.5, label=f"HAI normal (n={(hai_labels_arr==0).sum()})", density=True)
        if (hai_labels_arr == 1).sum() > 0:
            plt.hist(hai_scores[hai_labels_arr == 1], bins=30, alpha=0.5, label=f"HAI attack (n={(hai_labels_arr==1).sum()})", density=True)
        plt.axvline(hai_threshold, color="k", linestyle="--", label=f"Threshold ({hai_threshold:.2f})")
        plt.xlabel("Anomaly score"); plt.ylabel("Density"); plt.title("Anomaly Score Distribution")
        plt.legend(fontsize=8); plt.tight_layout()
        plt.savefig(FIG / "score_distribution.pdf"); plt.savefig(FIG / "score_distribution.png", dpi=150); plt.close()

        # ROC Curve
        plt.figure(figsize=(5, 5))
        if len(np.unique(hai_labels_arr)) > 1:
            fpr, tpr, _ = roc_curve(hai_labels_arr, hai_scores)
            plt.plot(fpr, tpr, label=f"HAI ROC-AUC = {hai_full_metrics['roc_auc']:.3f}")
        if len(np.unique(swat_labels_arr)) > 1:
            fpr, tpr, _ = roc_curve(swat_labels_arr, swat_scores)
            plt.plot(fpr, tpr, label=f"SWaT ROC-AUC = {swat_full_metrics['roc_auc']:.3f}")
            plt.plot([0, 1], [0, 1], "k--", alpha=0.3)
            plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
            plt.title("Zero-Shot ROC Curve (SWaT to HAI)"); plt.legend(); plt.tight_layout()
            plt.savefig(FIG / "roc_curve.pdf"); plt.savefig(FIG / "roc_curve.png", dpi=150); plt.close()

        # PR Curve
        plt.figure(figsize=(5, 5))
        if len(np.unique(hai_labels_arr)) > 1:
            prec, rec, _ = precision_recall_curve(hai_labels_arr, hai_scores)
            plt.plot(rec, prec, label=f"HAI PR-AUC = {hai_full_metrics['pr_auc']:.3f}")
            plt.axhline(hai_labels_arr.mean(), color="C0", linestyle="--", alpha=0.3)
        if len(np.unique(swat_labels_arr)) > 1:
            prec, rec, _ = precision_recall_curve(swat_labels_arr, swat_scores)
            plt.plot(rec, prec, label=f"SWaT PR-AUC = {swat_full_metrics['pr_auc']:.3f}")
            plt.axhline(swat_labels_arr.mean(), color="C1", linestyle="--", alpha=0.3)
            plt.xlabel("Recall"); plt.ylabel("Precision")
            plt.title("Zero-Shot PR Curve (SWaT to HAI)"); plt.legend(); plt.tight_layout()
            plt.savefig(FIG / "pr_curve.pdf"); plt.savefig(FIG / "pr_curve.png", dpi=150); plt.close()

        # Confusion Matrix
        cm = confusion_matrix(hai_labels_arr, hai_preds)
        plt.figure(figsize=(4.5, 4))
        plt.imshow(cm, cmap="Blues")
        for (i, j), v in np.ndenumerate(cm):
            plt.text(j, i, str(v), ha="center", va="center", color="white" if v > cm.max() / 2 else "black")
        plt.xticks([0, 1], ["Normal", "Attack"]); plt.yticks([0, 1], ["Normal", "Attack"])
        plt.xlabel("Predicted"); plt.ylabel("True"); plt.title("Zero-Shot Confusion Matrix (HAI)")
        plt.colorbar(fraction=0.046); plt.tight_layout()
        plt.savefig(FIG / "confusion_matrix.pdf"); plt.savefig(FIG / "confusion_matrix.png", dpi=150); plt.close()

def main():
    parser = argparse.ArgumentParser(description="ContrastiveTrust Training and Evaluation Pipeline")
    parser.add_argument("--swat-path", type=str, required=True, help="Path to SWaT dataset (pickle or csv)")
    parser.add_argument("--hai-path", type=str, required=True, help="Path to HAI dataset (pickle or csv)")
    parser.add_argument("--swat-meta-path", type=str, default="", help="Path to SWaT original excel for metadata")
    parser.add_argument("--window-size", type=int, default=60, help="Window size for SlidingWindowGenerator")
    parser.add_argument("--stride", type=int, default=20, help="Stride for SlidingWindowGenerator")
    parser.add_argument("--epochs", type=int, default=15, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--contrastive-weight", type=float, default=1.0)
    parser.add_argument("--physics-weight", type=float, default=1.0)
    parser.add_argument("--dropout", type=float, default=0.2, help="Dropout probability for all model components")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Weight decay for AdamW optimizer")
    parser.add_argument("--patience", type=int, default=8, help="Early stopping patience")
    parser.add_argument("--small-model", action="store_true", help="Use reduced-capacity model (halved hidden dims)")
    parser.add_argument("--out-dir", type=str, default="outputs", help="Output directory for plots and metrics")
    parser.add_argument("--log-dir", type=str, default="logs/training_real", help="Output directory for logs and models")
    parser.add_argument("--device", type=str, default="cpu", help="Device to use for training")
    
    args = parser.parse_args()
    
    out_dir = Path(args.out_dir)
    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load Data
    (train_loader, val_loader, normal_loader, 
     val_score_loader, swat_test_loader, hai_loader_eval) = load_data(
        swat_path=args.swat_path,
        hai_path=args.hai_path,
        swat_meta_path=args.swat_meta_path,
        window_size=args.window_size,
        stride=args.stride,
        batch_size=args.batch_size
    )

    # 2. Build Model
    encoder, projection_head, loss_fn = build_model(
        device=args.device, contrastive_weight=args.contrastive_weight,
        physics_weight=args.physics_weight, dropout=args.dropout,
        small_model=args.small_model,
    )

    # 3. Train
    history = train(
        encoder=encoder,
        projection_head=projection_head,
        loss_fn=loss_fn,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=args.epochs,
        log_dir=log_dir,
        device=args.device,
        weight_decay=args.weight_decay,
        patience=args.patience,
    )

    # 4. Build Detector
    detector = build_detector(
        encoder=encoder,
        normal_loader=normal_loader,
        val_score_loader=val_score_loader,
        device=args.device
    )

    # 5. Evaluate
    evaluate(
        detector=detector,
        hai_loader_eval=hai_loader_eval,
        swat_test_loader=swat_test_loader,
        history=history,
        out_dir=out_dir,
        device=args.device
    )
    
if __name__ == "__main__":
    main()
