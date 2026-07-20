"""Final multi-set evaluation script for ContrastiveTrust.

This script orchestrates the final zero-shot and held-out evaluations
across three datasets (HAI zero-shot, SWaT Dec2019, SWaT Mar11) using
the finalized Config B best_model checkpoint.

Generates requested figures and tables.
"""

import json
import logging
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
from sklearn.metrics import (
    accuracy_score, average_precision_score, confusion_matrix,
    f1_score, precision_score, recall_score, roc_auc_score,
    roc_curve, precision_recall_curve, ConfusionMatrixDisplay
)
import torch
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from preprocessing.cleaner import DataCleaner
from preprocessing.normalizer import FeatureNormalizer
from preprocessing.windowing import SlidingWindowGenerator
from src.data.swat_multi_loader import SWaTMultiLoader
from src.models.encoder import DualStreamEncoder, EncoderConfig
from src.models.physics_encoder import PhysicsEncoderConfig
from src.models.temporal_encoder import TemporalEncoderConfig
from src.evaluation.anomaly_scorer import AnomalyScorer, RawDistanceStrategy
from src.evaluation.distance_metrics import DistanceMetricFactory
from src.evaluation.embedding_bank import EmbeddingBank, EmbeddingBankConfig
from src.evaluation.threshold import ThresholdEstimator, MedianMADThreshold
from src.evaluation.zero_shot_detector import ZeroShotDetector
from src.features.channel_alignment import build_typed_frame, NUM_CHANNELS, PHYSICS_DIM

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# Config B Finalized parameters
WINDOW_SIZE = 60
STRIDE = 20
BATCH_SIZE = 64  # Increased for faster inference
DROPOUT = 0.2
SEED = 42

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

class InferenceDataset(Dataset):
    """Wraps sliding windows for inference."""
    def __init__(self, windows: np.ndarray, physics: np.ndarray, labels: np.ndarray = None) -> None:
        self.windows = torch.from_numpy(windows)
        self.physics = torch.from_numpy(physics)
        self.labels = torch.from_numpy(labels) if labels is not None else None

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, idx: int) -> tuple:
        if self.labels is not None:
            return self.windows[idx].float(), self.physics[idx], self.labels[idx].float()
        return self.windows[idx].float(), self.physics[idx]

def _prepare_windows(
    df: pd.DataFrame,
    feature_columns: list[str],
    normalizer: FeatureNormalizer,
    split_name: str,
    fit_normalizer: bool = False,
    is_train: bool = False
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    
    typed = build_typed_frame(df, feature_columns, keep=["t_stamp", "label"])
    
    if fit_normalizer:
        norm = normalizer.fit_transform(typed.drop(columns=["label"], errors="ignore"))
    else:
        norm = normalizer.transform(typed.drop(columns=["label"], errors="ignore"))
        
    if "label" in typed.columns:
        norm["label"] = typed["label"].to_numpy()
    if "t_stamp" not in norm.columns and "t_stamp" in typed.columns:
        norm["t_stamp"] = typed["t_stamp"].to_numpy()
        
    # Use smaller stride for test sets to get denser predictions, but 
    # original stride for training to match bank size
    stride_to_use = STRIDE if is_train else WINDOW_SIZE // 2
    
    win_gen = SlidingWindowGenerator(
        window_size=WINDOW_SIZE,
        stride=stride_to_use,
        timestamp_column="t_stamp",
        return_labels=True,
        label_method="max"
    )
    batch = win_gen.generate(norm)
    
    windows = batch.windows.astype(np.float32)
    physics = np.stack([physics_vector(w) for w in windows])
    
    # Use built-in label aggregation if labels exist
    if batch.labels is not None:
        labels = batch.labels.astype(int)
    else:
        labels = np.zeros(len(windows), dtype=int)
        
    logger.info(
        "%s: %d windows, %d anomalies",
        split_name, len(windows), int(labels.sum())
    )
    
    return windows, physics, labels

def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    base_dir = Path(__file__).resolve().parents[2]
    data_dir = base_dir / "data" / "raw"
    log_dir = base_dir / "logs" / "final_configB"
    paper_dir = base_dir / "paper"
    
    tables_dir = paper_dir / "tables"
    figures_dir = paper_dir / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    
    # ── 1. Load SWaT Data ──────────────────────────────────────────────
    logger.info("Loading SWaT data...")
    swat_loader = SWaTMultiLoader(data_dir / "SWaT")
    pooled = swat_loader.load().dataframe
    
    train_df = pooled[pooled["source_day"] == "feb19"].copy()
    val_df = pooled[pooled["source_day"] == "feb20"].copy()
    test_a_df = pooled[pooled["source_day"] == "dec2019"].copy()
    test_b_df = pooled[pooled["source_day"] == "mar11"].copy()
    
    def _clean(df, name):
        source_day = df["source_day"].to_numpy()
        label = df["label"].to_numpy()
        working = df.drop(columns=["source_day", "label"])
        cleaner = DataCleaner(timestamp_column="t_stamp", missing_value_strategy="forward_fill")
        cleaned, _ = cleaner.clean(working)
        cleaned["label"] = label[:len(cleaned)]
        cleaned["source_day"] = source_day[:len(cleaned)]
        return cleaned

    train_clean = _clean(train_df, "TRAIN")
    val_clean = _clean(val_df, "VAL")
    test_a_clean = _clean(test_a_df, "TEST_A")
    test_b_clean = _clean(test_b_df, "TEST_B")
    
    feature_cols_swat = [c for c in train_clean.columns if c not in {"t_stamp", "label", "source_day"}]
    normalizer = FeatureNormalizer(timestamp_column="t_stamp")
    
    train_win, train_phy, train_lbl = _prepare_windows(train_clean, feature_cols_swat, normalizer, "TRAIN", fit_normalizer=True, is_train=True)
    val_win, val_phy, val_lbl = _prepare_windows(val_clean, feature_cols_swat, normalizer, "VAL", fit_normalizer=False, is_train=True)
    test_a_win, test_a_phy, test_a_lbl = _prepare_windows(test_a_clean, feature_cols_swat, normalizer, "TEST_A", fit_normalizer=False)
    test_b_win, test_b_phy, test_b_lbl = _prepare_windows(test_b_clean, feature_cols_swat, normalizer, "TEST_B", fit_normalizer=False)

    # ── 2. Load HAI Data ───────────────────────────────────────────────
    logger.info("Loading HAI data...")
    hai_df = pd.read_csv(data_dir / "HAI" / "hai_test1.csv")
    hai_lbl_df = pd.read_csv(data_dir / "HAI" / "hai_test1_label.csv")
    
    if "label" in hai_lbl_df.columns:
        hai_df["label"] = hai_lbl_df["label"]
    else:
        hai_df["label"] = hai_lbl_df.iloc[:, 0]
        
    hai_df["t_stamp"] = pd.to_datetime(hai_df["timestamp"], errors="coerce")
    hai_df = hai_df.drop(columns=["timestamp"])
    
    feature_cols_hai = [c for c in hai_df.columns if c not in {"t_stamp", "label"}]
    # We use the SAME normalizer fitted on SWaT train for zero-shot
    hai_win, hai_phy, hai_lbl = _prepare_windows(hai_df, feature_cols_hai, normalizer, "HAI_TEST", fit_normalizer=False)

    # ── 3. Dataset Statistics ─────────────────────────────────────────
    stats = []
    
    def _add_stat(name, split, win, lbl, num_sensors):
        stats.append({
            "Dataset": name,
            "Split": split,
            "Total Windows": len(win),
            "Normal Windows": int((lbl == 0).sum()),
            "Attack Windows": int((lbl == 1).sum()),
            "Sensors": num_sensors
        })
        
    _add_stat("SWaT Feb19", "Train", train_win, train_lbl, len(feature_cols_swat))
    _add_stat("SWaT Feb20", "Validation", val_win, val_lbl, len(feature_cols_swat))
    _add_stat("SWaT Dec2019", "Test A", test_a_win, test_a_lbl, len(feature_cols_swat))
    _add_stat("SWaT Mar11", "Test B", test_b_win, test_b_lbl, len(feature_cols_swat))
    _add_stat("HAI Test 1", "Zero-shot Test", hai_win, hai_lbl, len(feature_cols_hai))
    
    stats_df = pd.DataFrame(stats)
    stats_df.to_csv(tables_dir / "dataset_statistics.csv", index=False)
    logger.info("Saved dataset_statistics.csv")

    # ── 4. Load Model ──────────────────────────────────────────────────
    encoder_config = EncoderConfig(
        temporal=TemporalEncoderConfig(input_channels=NUM_CHANNELS, dropout=DROPOUT),
        physics=PhysicsEncoderConfig(input_dim=PHYSICS_DIM, dropout=DROPOUT)
    )
    encoder = DualStreamEncoder(encoder_config).to(device)
    
    ckpt_path = log_dir / "best_model.pt"
    logger.info(f"Loading checkpoint from {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location=device)
    encoder.load_state_dict(checkpoint["encoder_state_dict"])
    encoder.eval()

    # ── 5. Setup ZeroShotDetector ──────────────────────────────────────
    bank_config = EmbeddingBankConfig(embedding_dim=256, device=device)
    bank = EmbeddingBank(config=bank_config)
    
    metric = DistanceMetricFactory.create("cosine")
    strategy = RawDistanceStrategy()
    scorer = AnomalyScorer(bank=bank, metric=metric, k=1, strategy=strategy)
    
    threshold_est = ThresholdEstimator(strategy=MedianMADThreshold(k=3.0))
    
    detector = ZeroShotDetector(encoder=encoder, scorer=scorer, threshold_estimator=threshold_est, device=device)

    # Create loaders
    train_ds = InferenceDataset(train_win, train_phy)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=False)
    
    val_ds = InferenceDataset(val_win, val_phy)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
    
    # Fit detector (populates bank with train, estimates threshold with val)
    logger.info("Fitting ZeroShotDetector...")
    detector.fit(normal_loader=train_loader, val_loader=val_loader)
    
    threshold = detector.threshold_estimator.predict_threshold()
    logger.info(f"Estimated threshold: {threshold:.4f}")

    # ── 6. Evaluate and get scores ─────────────────────────────────────
    def _get_scores(windows, physics, labels):
        ds = InferenceDataset(windows, physics, labels)
        loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False)
        all_scores = []
        all_labels = []
        with torch.no_grad():
            for w, p, l in loader:
                scores = detector.score(w, p)
                all_scores.append(scores.cpu().numpy())
                all_labels.append(l.numpy())
        return np.concatenate(all_scores), np.concatenate(all_labels)

    logger.info("Scoring Test A (Dec2019)...")
    scores_a, labels_a = _get_scores(test_a_win, test_a_phy, test_a_lbl)
    
    logger.info("Scoring Test B (Mar11)...")
    scores_b, labels_b = _get_scores(test_b_win, test_b_phy, test_b_lbl)
    
    logger.info("Scoring HAI...")
    scores_hai, labels_hai = _get_scores(hai_win, hai_phy, hai_lbl)
    
    # For plotting (SWaT-normal, SWaT-attack)
    # Combine Test A and Test B
    scores_swat = np.concatenate([scores_a, scores_b])
    labels_swat = np.concatenate([labels_a, labels_b])

    # ── 7. Calculate Metrics ───────────────────────────────────────────
    def _compute_metrics(scores, labels, thresh):
        preds = (scores > thresh).astype(int)
        
        # Avoid undefined metric warnings if no positive labels
        if labels.sum() == 0:
            return {
                "Accuracy": accuracy_score(labels, preds),
                "Precision": 0.0,
                "Recall": 0.0,
                "F1": 0.0,
                "ROC-AUC": 0.0,
                "PR-AUC": 0.0
            }
            
        return {
            "Accuracy": accuracy_score(labels, preds),
            "Precision": precision_score(labels, preds, zero_division=0),
            "Recall": recall_score(labels, preds, zero_division=0),
            "F1": f1_score(labels, preds, zero_division=0),
            "ROC-AUC": roc_auc_score(labels, scores),
            "PR-AUC": average_precision_score(labels, scores)
        }

    metrics_a = _compute_metrics(scores_a, labels_a, threshold)
    metrics_b = _compute_metrics(scores_b, labels_b, threshold)
    metrics_hai = _compute_metrics(scores_hai, labels_hai, threshold)
    metrics_swat_pooled = _compute_metrics(scores_swat, labels_swat, threshold)
    
    metrics_df = pd.DataFrame([
        {"Dataset": "TEST_A (Dec2019)", **metrics_a},
        {"Dataset": "TEST_B (Mar11)", **metrics_b},
        {"Dataset": "HAI (Zero-shot)", **metrics_hai}
    ])
    
    # Calculate Transfer Gap (Best-case SWaT - HAI)
    best_swat_f1 = max(metrics_a["F1"], metrics_b["F1"])
    transfer_gap = best_swat_f1 - metrics_hai["F1"]
    metrics_df["Delta-F1 Transfer Gap"] = transfer_gap
    
    metrics_df.to_csv(tables_dir / "metrics.csv", index=False)
    logger.info("Saved metrics.csv")
    
    # Print metrics for verification
    for _, row in metrics_df.iterrows():
        logger.info("--- %s ---", row["Dataset"])
        for k, v in row.items():
            if k != "Dataset":
                logger.info("  %s: %.4f", k, v)

    # ── 8. Generate Figures ────────────────────────────────────────────
    # a. Score Distribution
    plt.figure(figsize=(10, 6))
    plt.hist(scores_swat[labels_swat == 0], bins=50, alpha=0.5, label='SWaT Normal', density=True, color='blue')
    plt.hist(scores_swat[labels_swat == 1], bins=50, alpha=0.5, label='SWaT Attack', density=True, color='red')
    plt.hist(scores_hai[labels_hai == 0], bins=50, alpha=0.5, label='HAI Normal', density=True, color='green')
    plt.hist(scores_hai[labels_hai == 1], bins=50, alpha=0.5, label='HAI Attack', density=True, color='orange')
    plt.axvline(threshold, color='black', linestyle='dashed', linewidth=2, label=f'Threshold ({threshold:.2f})')
    plt.title('Anomaly Score Distribution (SWaT vs HAI)')
    plt.xlabel('Anomaly Score')
    plt.ylabel('Density')
    plt.legend()
    plt.tight_layout()
    plt.savefig(figures_dir / "score_distribution.png", dpi=200)
    plt.savefig(figures_dir / "score_distribution.pdf")
    plt.close()

    # b. ROC Curve
    plt.figure(figsize=(8, 8))
    if labels_a.sum() > 0:
        fpr_a, tpr_a, _ = roc_curve(labels_a, scores_a)
        plt.plot(fpr_a, tpr_a, label=f'Dec2019 (AUC = {metrics_a["ROC-AUC"]:.3f})', color='blue')
    if labels_b.sum() > 0:
        fpr_b, tpr_b, _ = roc_curve(labels_b, scores_b)
        plt.plot(fpr_b, tpr_b, label=f'Mar11 (AUC = {metrics_b["ROC-AUC"]:.3f})', color='orange')
    if labels_hai.sum() > 0:
        fpr_h, tpr_h, _ = roc_curve(labels_hai, scores_hai)
        plt.plot(fpr_h, tpr_h, label=f'HAI (AUC = {metrics_hai["ROC-AUC"]:.3f})', color='green')
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve Comparison')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(figures_dir / "roc_curve.png", dpi=200)
    plt.savefig(figures_dir / "roc_curve.pdf")
    plt.close()

    # c. PR Curve
    plt.figure(figsize=(8, 8))
    if labels_a.sum() > 0:
        p_a, r_a, _ = precision_recall_curve(labels_a, scores_a)
        plt.plot(r_a, p_a, label=f'Dec2019 (PR-AUC = {metrics_a["PR-AUC"]:.3f})', color='blue')
    if labels_b.sum() > 0:
        p_b, r_b, _ = precision_recall_curve(labels_b, scores_b)
        plt.plot(r_b, p_b, label=f'Mar11 (PR-AUC = {metrics_b["PR-AUC"]:.3f})', color='orange')
    if labels_hai.sum() > 0:
        p_h, r_h, _ = precision_recall_curve(labels_hai, scores_hai)
        plt.plot(r_h, p_h, label=f'HAI (PR-AUC = {metrics_hai["PR-AUC"]:.3f})', color='green')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve Comparison')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(figures_dir / "pr_curve.png", dpi=200)
    plt.savefig(figures_dir / "pr_curve.pdf")
    plt.close()

    # d. Confusion Matrix
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    cm_swat = confusion_matrix(labels_swat, (scores_swat > threshold).astype(int))
    ConfusionMatrixDisplay(cm_swat, display_labels=['Normal', 'Attack']).plot(ax=axes[0], cmap='Blues', colorbar=False)
    axes[0].set_title('SWaT (Dec2019 + Mar11)')
    
    cm_hai = confusion_matrix(labels_hai, (scores_hai > threshold).astype(int))
    ConfusionMatrixDisplay(cm_hai, display_labels=['Normal', 'Attack']).plot(ax=axes[1], cmap='Greens', colorbar=False)
    axes[1].set_title('HAI (Zero-shot)')
    
    plt.tight_layout()
    plt.savefig(figures_dir / "confusion_matrix.png", dpi=200)
    plt.savefig(figures_dir / "confusion_matrix.pdf")
    plt.close()

    # e. t-SNE Embeddings
    logger.info("Computing t-SNE...")
    
    # Extract embeddings directly using the encoder
    def _extract_embeddings(windows, physics):
        if len(windows) == 0:
            return np.zeros((0, 256), dtype=np.float32)
        ds = InferenceDataset(windows, physics)
        loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False)
        embs = []
        with torch.no_grad():
            for w, p in loader:
                out = encoder(w.to(device), p.to(device))
                embs.append(out["embedding"].cpu().numpy())
        return np.concatenate(embs)
    
    # Randomly subsample for t-SNE to keep computation reasonable
    def _subsample(arrs, n=1000):
        if len(arrs[0]) <= n: return arrs
        idx = np.random.choice(len(arrs[0]), n, replace=False)
        return [a[idx] for a in arrs]
    
    # 1. Feb19 train-domain normal
    emb_train = _extract_embeddings(*_subsample([train_win, train_phy], 1500))
    # 2. HAI normal
    mask_hn = labels_hai == 0
    emb_hai_n = _extract_embeddings(*_subsample([hai_win[mask_hn], hai_phy[mask_hn]], 1000))
    # 3. HAI attack
    mask_ha = labels_hai == 1
    emb_hai_a = _extract_embeddings(*_subsample([hai_win[mask_ha], hai_phy[mask_ha]], 500))
    # 4. SWaT attack (Dec2019 + Mar11)
    mask_sa = labels_swat == 1
    emb_swat_a = _extract_embeddings(*_subsample([np.concatenate([test_a_win, test_b_win])[mask_sa], 
                                                  np.concatenate([test_a_phy, test_b_phy])[mask_sa]], 1000))
                                                  
    all_embs = np.concatenate([emb_train, emb_hai_n, emb_hai_a, emb_swat_a])
    group_labels = (
        ['SWaT Train (Normal)'] * len(emb_train) +
        ['HAI (Normal)'] * len(emb_hai_n) +
        ['HAI (Attack)'] * len(emb_hai_a) +
        ['SWaT Test (Attack)'] * len(emb_swat_a)
    )
    
    tsne = TSNE(n_components=2, random_state=SEED, init='pca', learning_rate='auto')
    embs_2d = tsne.fit_transform(all_embs)
    
    plt.figure(figsize=(10, 8))
    colors = {'SWaT Train (Normal)': 'blue', 'HAI (Normal)': 'green', 
              'HAI (Attack)': 'orange', 'SWaT Test (Attack)': 'red'}
    markers = {'SWaT Train (Normal)': 'o', 'HAI (Normal)': 'o', 
               'HAI (Attack)': 'x', 'SWaT Test (Attack)': 'x'}
               
    for label in set(group_labels):
        mask = np.array(group_labels) == label
        plt.scatter(
            embs_2d[mask, 0], embs_2d[mask, 1], 
            label=label, alpha=0.6, 
            c=colors[label], marker=markers[label], s=30
        )
        
    plt.title('t-SNE Embeddings (Cross-Domain Analysis)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(figures_dir / "embedding_tsne.png", dpi=200)
    plt.savefig(figures_dir / "embedding_tsne.pdf")
    plt.close()
    
    logger.info("Figures generated successfully.")
    
if __name__ == "__main__":
    main()
