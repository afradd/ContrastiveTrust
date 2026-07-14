"""Overfitting diagnosis sweep for ContrastiveTrust.

Runs 4 configurations for 12 epochs each and reports train/val gap:
  A: stride=5 baseline (existing behavior)
  B: stride=20 baseline (reduced window redundancy)
  C: stride=20 + regularization (dropout=0.3, weight_decay=1e-2)
  D: stride=20 + regularization + small model (halved hidden dims)
"""
import subprocess
import json
import sys
from pathlib import Path

CONFIGS = [
    {
        "name": "A_stride5_baseline",
        "stride": 5,
        "dropout": 0.2,
        "weight_decay": 1e-4,
        "small_model": False,
        "label": "stride=5, default model",
    },
    {
        "name": "B_stride20_baseline",
        "stride": 20,
        "dropout": 0.2,
        "weight_decay": 1e-4,
        "small_model": False,
        "label": "stride=20, default model",
    },
    {
        "name": "C_stride20_regularized",
        "stride": 20,
        "dropout": 0.3,
        "weight_decay": 1e-2,
        "small_model": False,
        "label": "stride=20, drop=0.3, wd=1e-2",
    },
    {
        "name": "D_stride20_small",
        "stride": 20,
        "dropout": 0.3,
        "weight_decay": 1e-2,
        "small_model": True,
        "label": "stride=20, drop=0.3, wd=1e-2, small",
    },
]

EPOCHS = 12


def main():
    results = []

    for cfg in CONFIGS:
        name = cfg["name"]
        print(f"\n{'=' * 70}")
        print(f"  Config: {cfg['label']}")
        print(f"{'=' * 70}")

        out_dir = f"outputs/overfit_sweep/{name}"
        log_dir = f"logs/overfit_sweep/{name}"

        cmd = [
            sys.executable,
            "src/pipelines/train_and_evaluate.py",
            "--swat-path", "data/raw/SWaT/SWaT_Dec2019.pkl",
            "--hai-path", "data/raw/HAI/hai_test1_combined.csv",
            "--swat-meta-path", "data/raw/SWaT/SWaT_Dec2019.xlsx",
            "--epochs", str(EPOCHS),
            "--stride", str(cfg["stride"]),
            "--contrastive-weight", "1.0",
            "--physics-weight", "1.0",
            "--dropout", str(cfg["dropout"]),
            "--weight-decay", str(cfg["weight_decay"]),
            "--patience", "999",  # disable early stopping for fair comparison
            "--out-dir", out_dir,
            "--log-dir", log_dir,
        ]
        if cfg["small_model"]:
            cmd.append("--small-model")

        print(f"  Command: {' '.join(cmd)}\n")
        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError:
            print(f"  *** FAILED: {name} ***")
            results.append({"config": cfg["label"], "status": "FAILED"})
            continue

        # Parse training history
        history_path = Path(out_dir) / "tables" / "training_history.json"
        if not history_path.exists():
            print(f"  No history file: {history_path}")
            results.append({"config": cfg["label"], "status": "NO HISTORY"})
            continue

        with open(history_path) as f:
            history = json.load(f)

        final = history[-1]
        train_loss = final["train_loss"]
        val_loss = final["val_loss"]
        gap = val_loss - train_loss

        # Determine val-loss trend from last 4 epochs
        val_losses = [h["val_loss"] for h in history]
        if len(val_losses) >= 4:
            recent = val_losses[-4:]
            if recent[-1] > recent[0] * 1.02:
                trend = "Diverging"
            elif abs(recent[-1] - recent[0]) < 0.02 * recent[0]:
                trend = "Plateau"
            else:
                trend = "Decreasing"
        else:
            trend = "Too short"

        # Track first-epoch val loss for initial gap comparison
        first_val = history[0]["val_loss"]
        first_train = history[0]["train_loss"]

        # Unweighted physics loss at final epoch
        raw_phys = final.get("val_physics_loss", float("nan"))

        results.append({
            "config": cfg["label"],
            "train_loss": train_loss,
            "val_loss": val_loss,
            "gap": gap,
            "first_train": first_train,
            "first_val": first_val,
            "trend": trend,
            "epochs_run": len(history),
            "raw_physics": raw_phys,
        })

    # ---- Print comparison table ----
    print(f"\n\n{'=' * 110}")
    print("  OVERFITTING DIAGNOSIS SWEEP — 12-EPOCH COMPARISON")
    print(f"{'=' * 110}")
    header = (
        f"{'Config':<40} | {'Train':>8} | {'Val':>8} | "
        f"{'Gap':>8} | {'1st Val':>8} | {'Phys':>8} | "
        f"{'Trend':<12} | {'Ep':>3}"
    )
    print(header)
    print("-" * len(header))
    for r in results:
        if "train_loss" in r:
            print(
                f"{r['config']:<40} | {r['train_loss']:>8.4f} | "
                f"{r['val_loss']:>8.4f} | {r['gap']:>8.4f} | "
                f"{r['first_val']:>8.4f} | {r['raw_physics']:>8.4f} | "
                f"{r['trend']:<12} | {r['epochs_run']:>3}"
            )
        else:
            status = r.get("status", "UNKNOWN")
            print(
                f"{r['config']:<40} | {'N/A':>8} | {'N/A':>8} | "
                f"{'N/A':>8} | {'N/A':>8} | {'N/A':>8} | "
                f"{status:<12} | {'N/A':>3}"
            )

    # Also save as JSON for easy post-processing
    out_path = Path("outputs/overfit_sweep/sweep_results.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
