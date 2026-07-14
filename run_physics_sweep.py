import subprocess
import json
import sys
from pathlib import Path

def main():
    weights = [1.0, 3.0, 5.0, 8.0]
    epochs = 12
    contrastive_weight = 1.0

    results = []

    for w in weights:
        print(f"--- Running with physics_weight={w} ---")
        out_dir = f"outputs/sweep_{w}"
        log_dir = f"logs/sweep_{w}"

        cmd = [
            sys.executable,
            "src/pipelines/train_and_evaluate.py",
            "--swat-path", "data/raw/SWaT/SWaT_Dec2019.pkl",
            "--hai-path", "data/raw/HAI/hai_test1_combined.csv",
            "--swat-meta-path", "data/raw/SWaT/SWaT_Dec2019.xlsx",
            "--epochs", str(epochs),
            "--contrastive-weight", str(contrastive_weight),
            "--physics-weight", str(w),
            "--out-dir", out_dir,
            "--log-dir", log_dir,
        ]

        print(f"Executing: {' '.join(cmd)}")
        # Run it
        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            print(f"Run failed for physics_weight={w}")
            continue

        # Parse history
        history_path = Path(out_dir) / "tables" / "training_history.json"
        if not history_path.exists():
            print(f"History file not found: {history_path}")
            continue
        
        with open(history_path, "r") as f:
            history = json.load(f)
        
        final_epoch = history[-1]
        final_train_loss = final_epoch.get("train_loss")
        final_val_loss = final_epoch.get("val_loss")
        final_train_physics_loss = final_epoch.get("train_physics_loss")
        final_val_physics_loss = final_epoch.get("val_physics_loss")

        # Check plateau/diverge
        val_losses = [h["val_loss"] for h in history]
        if len(val_losses) > 3:
            recent_losses = val_losses[-4:]
            if recent_losses[-1] > recent_losses[0]:
                status = "Diverging/Increasing"
            elif max(recent_losses) - min(recent_losses) < 0.05 * recent_losses[0]:
                status = "Plateaued"
            else:
                status = "Decreasing"
        else:
            status = "Too short"

        res = {
            "physics_weight": w,
            "final_train_loss": final_train_loss,
            "final_val_loss": final_val_loss,
            "status": status,
            "final_val_physics_loss": final_val_physics_loss
        }
        results.append(res)
    
    print("\n\n--- SWEEP RESULTS ---")
    print(f"{'Physics W':<12} | {'Train Loss':<12} | {'Val Loss':<12} | {'Val Physics':<12} | {'Status':<20}")
    print("-" * 75)
    for r in results:
        pw = r["physics_weight"]
        tl = r["final_train_loss"]
        vl = r["final_val_loss"]
        vp = r["final_val_physics_loss"]
        st = r["status"]
        print(f"{pw:<12.1f} | {tl:<12.4f} | {vl:<12.4f} | {vp:<12.4f} | {st:<20}")

if __name__ == '__main__':
    main()
