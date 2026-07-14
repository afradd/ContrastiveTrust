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

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

WINDOW_SIZE = 60
STRIDE = 20
BATCH_SIZE = 64
DROPOUT = 0.2
SEED = 42

def physics_vector(window: np.ndarray) -> np.ndarray:
    mean = window.mean(axis=0)
    std = window.std(axis=0)
    roc = (
        np.abs(np.diff(window, axis=0)).mean(axis=0)
        if window.shape[0] > 1
        else np.zeros(window.shape[1])
    )
    return np.concatenate([mean, std, roc]).astype(np.float32)

class InferenceDataset(Dataset):
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
    
    if batch.labels is not None:
        labels = batch.labels.astype(int)
    else:
        labels = np.zeros(len(windows), dtype=int)
    return windows, physics, labels

def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    base_dir = Path(__file__).resolve().parents[2]
    data_dir = base_dir / "data" / "raw"
    log_dir = base_dir / "logs" / "final_configB"
    out_dir = base_dir / "outputs" / "final_configB"
    paper_dir = base_dir / "paper"
    
    figures_dir = paper_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    
    # ── 1. Load Training History for Fig 1 ──────────────────────────────────
    history_file = out_dir / "tables" / "training_history.json"
    if history_file.exists():
        with open(history_file, 'r') as f:
            history = json.load(f)
        epochs = [h["epoch"] for h in history]
        train_loss = [h["train_loss"] for h in history]
        val_loss = [h["val_loss"] for h in history]
        
        plt.figure(figsize=(6, 4))
        plt.plot(epochs, train_loss, label='SWaT-N1 (Train)', marker='o')
        plt.plot(epochs, val_loss, label='SWaT-N2 (Val)', marker='s')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(figures_dir / "fig1_training_loss.png", dpi=300, bbox_inches="tight")
        plt.savefig(figures_dir / "fig1_training_loss.pdf", bbox_inches="tight")
        plt.close()

    # ── 2. Load Data for inference ──────────────────────────────────────────
    logger.info("Loading SWaT data...")
    swat_loader = SWaTMultiLoader(data_dir / "SWaT")
    pooled = swat_loader.load().dataframe
    
    train_df = pooled[pooled["source_day"] == "feb19"].copy()
    val_df = pooled[pooled["source_day"] == "feb20"].copy()
    test_a_df = pooled[pooled["source_day"] == "dec2019"].copy()
    test_b_df = pooled[pooled["source_day"] == "mar11"].copy()
    
    def _clean(df):
        label = df["label"].to_numpy()
        working = df.drop(columns=["source_day", "label"])
        cleaner = DataCleaner(timestamp_column="t_stamp", missing_value_strategy="forward_fill")
        cleaned, _ = cleaner.clean(working)
        cleaned["label"] = label[:len(cleaned)]
        return cleaned

    train_clean = _clean(train_df)
    val_clean = _clean(val_df)
    test_a_clean = _clean(test_a_df)
    test_b_clean = _clean(test_b_df)
    
    feature_cols = [c for c in train_clean.columns if c not in {"t_stamp", "label"}]
    normalizer = FeatureNormalizer(timestamp_column="t_stamp")
    
    train_win, train_phy, train_lbl = _prepare_windows(train_clean, feature_cols, normalizer, "TRAIN", fit_normalizer=True, is_train=True)
    val_win, val_phy, val_lbl = _prepare_windows(val_clean, feature_cols, normalizer, "VAL", fit_normalizer=False, is_train=True)
    test_a_win, test_a_phy, test_a_lbl = _prepare_windows(test_a_clean, feature_cols, normalizer, "SWaT-M1", fit_normalizer=False)
    test_b_win, test_b_phy, test_b_lbl = _prepare_windows(test_b_clean, feature_cols, normalizer, "SWaT-M2", fit_normalizer=False)

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
    hai_win, hai_phy, hai_lbl = _prepare_windows(hai_df, feature_cols_hai, normalizer, "HAI_TEST", fit_normalizer=False)

    # ── 3. Load Model ──────────────────────────────────────────────────
    encoder_config = EncoderConfig(
        temporal=TemporalEncoderConfig(input_channels=NUM_CHANNELS, dropout=DROPOUT),
        physics=PhysicsEncoderConfig(input_dim=PHYSICS_DIM, dropout=DROPOUT)
    )
    encoder = DualStreamEncoder(encoder_config).to(device)
    
    ckpt_path = log_dir / "best_model.pt"
    checkpoint = torch.load(ckpt_path, map_location=device)
    encoder.load_state_dict(checkpoint["encoder_state_dict"])
    encoder.eval()

    bank_config = EmbeddingBankConfig(embedding_dim=256, device=device)
    bank = EmbeddingBank(config=bank_config)
    
    metric = DistanceMetricFactory.create("cosine")
    strategy = RawDistanceStrategy()
    scorer = AnomalyScorer(bank=bank, metric=metric, k=1, strategy=strategy)
    
    threshold_est = ThresholdEstimator(strategy=MedianMADThreshold(k=3.0))
    
    detector = ZeroShotDetector(encoder=encoder, scorer=scorer, threshold_estimator=threshold_est, device=device)

    train_ds = InferenceDataset(train_win, train_phy)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=False)
    
    val_ds = InferenceDataset(val_win, val_phy)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
    
    detector.fit(normal_loader=train_loader, val_loader=val_loader)
    threshold = detector.threshold_estimator.predict_threshold()

    def _get_scores(windows, physics, labels):
        ds = InferenceDataset(windows, physics, labels)
        loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False)
        all_scores = []
        with torch.no_grad():
            for w, p, _ in loader:
                scores = detector.score(w, p)
                all_scores.append(scores.cpu().numpy())
        return np.concatenate(all_scores)

    scores_swat = _get_scores(test_a_win, test_a_phy, test_a_lbl)
    scores_m2 = _get_scores(test_b_win, test_b_phy, test_b_lbl)
    scores_hai = _get_scores(hai_win, hai_phy, hai_lbl)

    # ── Fig 2: Score Distribution ──────────────────────────────────────
    plt.figure(figsize=(7, 5))
    plt.hist(scores_swat[test_a_lbl == 0], bins=50, alpha=0.5, label='SWaT-M1 Normal', density=True)
    plt.hist(scores_swat[test_a_lbl == 1], bins=50, alpha=0.5, label='SWaT-M1 Attack', density=True)
    plt.hist(scores_m2[test_b_lbl == 0], bins=50, alpha=0.5, label='SWaT-M2 Normal', density=True)
    plt.hist(scores_m2[test_b_lbl == 1], bins=50, alpha=0.5, label='SWaT-M2 Attack', density=True)
    plt.hist(scores_hai[hai_lbl == 0], bins=50, alpha=0.5, label='HAI-Test1 Normal', density=True)
    plt.hist(scores_hai[hai_lbl == 1], bins=50, alpha=0.5, label='HAI-Test1 Attack', density=True)
    plt.axvline(threshold, color='black', linestyle='dashed', linewidth=2)
    plt.xlabel('Anomaly Score')
    plt.ylabel('Density')
    plt.legend()
    plt.tight_layout()
    plt.savefig(figures_dir / "fig2_score_distribution.png", dpi=300, bbox_inches="tight")
    plt.savefig(figures_dir / "fig2_score_distribution.pdf", bbox_inches="tight")
    plt.close()

    # ── Fig 3: ROC Curve ─────────────────────────────────────────────
    plt.figure(figsize=(6, 5))
    fpr_swat, tpr_swat, _ = roc_curve(test_a_lbl, scores_swat)
    fpr_m2, tpr_m2, _ = roc_curve(test_b_lbl, scores_m2)
    fpr_hai, tpr_hai, _ = roc_curve(hai_lbl, scores_hai)
    plt.plot(fpr_swat, tpr_swat, label='SWaT-M1')
    plt.plot(fpr_m2, tpr_m2, label='SWaT-M2')
    plt.plot(fpr_hai, tpr_hai, label='HAI-Test1')
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(figures_dir / "fig3_roc_curve.png", dpi=300, bbox_inches="tight")
    plt.savefig(figures_dir / "fig3_roc_curve.pdf", bbox_inches="tight")
    plt.close()

    # ── Fig 4: PR Curve ──────────────────────────────────────────────
    plt.figure(figsize=(6, 5))
    p_swat, r_swat, _ = precision_recall_curve(test_a_lbl, scores_swat)
    p_m2, r_m2, _ = precision_recall_curve(test_b_lbl, scores_m2)
    p_hai, r_hai, _ = precision_recall_curve(hai_lbl, scores_hai)
    plt.plot(r_swat, p_swat, label='SWaT-M1')
    plt.plot(r_m2, p_m2, label='SWaT-M2')
    plt.plot(r_hai, p_hai, label='HAI-Test1')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(figures_dir / "fig4_pr_curve.png", dpi=300, bbox_inches="tight")
    plt.savefig(figures_dir / "fig4_pr_curve.pdf", bbox_inches="tight")
    plt.close()

    # ── Fig 5: Confusion Matrices ──────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    cm_swat = confusion_matrix(test_a_lbl, (scores_swat > threshold).astype(int))
    ConfusionMatrixDisplay(cm_swat, display_labels=['Normal', 'Attack']).plot(ax=axes[0], cmap='Blues', colorbar=False)
    axes[0].set_title('SWaT-M1')
    
    cm_m2 = confusion_matrix(test_b_lbl, (scores_m2 > threshold).astype(int))
    ConfusionMatrixDisplay(cm_m2, display_labels=['Normal', 'Attack']).plot(ax=axes[1], cmap='Purples', colorbar=False)
    axes[1].set_title('SWaT-M2')
    
    cm_hai = confusion_matrix(hai_lbl, (scores_hai > threshold).astype(int))
    ConfusionMatrixDisplay(cm_hai, display_labels=['Normal', 'Attack']).plot(ax=axes[2], cmap='Greens', colorbar=False)
    axes[2].set_title('HAI-Test1')
    
    plt.tight_layout()
    plt.savefig(figures_dir / "fig5_confusion_matrix.png", dpi=300, bbox_inches="tight")
    plt.savefig(figures_dir / "fig5_confusion_matrix.pdf", bbox_inches="tight")
    plt.close()

    # ── Fig 6: t-SNE Embeddings ──────────────────────────────────────
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
    
    def _subsample(arrs, n=1000):
        if len(arrs[0]) <= n: return arrs
        idx = np.random.choice(len(arrs[0]), n, replace=False)
        return [a[idx] for a in arrs]
    
    emb_train = _extract_embeddings(*_subsample([train_win, train_phy], 1500))
    mask_hn = hai_lbl == 0
    emb_hai_n = _extract_embeddings(*_subsample([hai_win[mask_hn], hai_phy[mask_hn]], 1000))
    mask_ha = hai_lbl == 1
    emb_hai_a = _extract_embeddings(*_subsample([hai_win[mask_ha], hai_phy[mask_ha]], 500))
    mask_sa = test_a_lbl == 1
    emb_swat_a = _extract_embeddings(*_subsample([test_a_win[mask_sa], test_a_phy[mask_sa]], 1000))
                                                  
    all_embs = np.concatenate([emb_train, emb_swat_a, emb_hai_n, emb_hai_a])
    group_labels = (
        ['SWaT Normal'] * len(emb_train) +
        ['SWaT Attack'] * len(emb_swat_a) +
        ['HAI Normal'] * len(emb_hai_n) +
        ['HAI Attack'] * len(emb_hai_a)
    )
    
    tsne = TSNE(n_components=2, random_state=SEED, init='pca', learning_rate='auto')
    embs_2d = tsne.fit_transform(all_embs)
    
    plt.figure(figsize=(7, 5))
    colors = {'SWaT Normal': 'blue', 'SWaT Attack': 'red', 
              'HAI Normal': 'green', 'HAI Attack': 'orange'}
    markers = {'SWaT Normal': 'o', 'SWaT Attack': 'x', 
               'HAI Normal': 'o', 'HAI Attack': 'x'}
               
    for label in ['SWaT Normal', 'SWaT Attack', 'HAI Normal', 'HAI Attack']:
        mask = np.array(group_labels) == label
        if mask.any():
            plt.scatter(
                embs_2d[mask, 0], embs_2d[mask, 1], 
                label=label, alpha=0.6, 
                c=colors[label], marker=markers[label], s=30
            )
        
    plt.legend()
    plt.tight_layout()
    plt.savefig(figures_dir / "fig6_embedding_tsne.png", dpi=300, bbox_inches="tight")
    plt.savefig(figures_dir / "fig6_embedding_tsne.pdf", bbox_inches="tight")
    plt.close()

if __name__ == "__main__":
    main()
