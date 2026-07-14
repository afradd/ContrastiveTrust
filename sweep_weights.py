import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.pipelines.train_and_evaluate import load_data, build_model, train
import json
import numpy as np

def run_sweep():
    weights = [1.0, 3.0, 5.0, 8.0]
    results = []

    # 1. Load Data
    print("Loading data...")
    (train_loader, val_loader, normal_loader, 
     val_score_loader, swat_test_loader, hai_loader_eval) = load_data(
        swat_path="data/raw/SWaT/SWaT_Dec2019.pkl",
        hai_path="data/raw/HAI/hai_test1_combined.csv",
        swat_meta_path="data/raw/SWaT/SWaT_Dec2019.xlsx",
        window_size=60,
        stride=20, # Use default 20 stride for faster sweep
        batch_size=32
    )

    for p_w in weights:
        print(f"\n--- Running Sweep for physics_weight={p_w} ---")
        log_dir = Path(f"logs/sweep_pw_{p_w}")
        log_dir.mkdir(parents=True, exist_ok=True)

        encoder, projection_head, loss_fn = build_model(device="cpu")
        loss_fn.set_weights({"contrastive": 1.0, "physics": p_w})
        
        history = train(
            encoder=encoder,
            projection_head=projection_head,
            loss_fn=loss_fn,
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=12,
            log_dir=log_dir,
            device="cpu"
        )
        
        final_train_loss = history[-1]["train_loss"]
        final_val_loss = history[-1]["val_loss"]
        
        val_losses = [h["val_loss"] for h in history]
        if min(val_losses) == val_losses[-1]:
            status = "Improving"
        elif val_losses[-1] > min(val_losses) * 1.05:
            status = "Diverging"
        else:
            status = "Plateauing"
            
        # Parse log for raw unweighted physics loss
        import re
        log_file = log_dir / "metrics.jsonl" # actually metrics logger just logs to metrics.jsonl in dict format
        
        # Wait, the physics loss logged in metrics.jsonl is the combined total. We need the raw one. 
        # But wait, trainer doesn't log unweighted physics loss per epoch to history!
        # It logs it to the main logging console. We need to parse the python logging output for that epoch.
        # It's easier to just read metrics.jsonl for train/val loss, and parse the console output.
        pass

if __name__ == "__main__":
    run_sweep()
