import subprocess
import json
import re
import numpy as np

def parse_last_epoch_physics_loss(log_file_content):
    p = re.compile(r'total=([\d.]+).*contrastive=([\d.]+).*physics=([\d.]+)')
    lines = log_file_content.split('\n')
    
    # We want the physics losses for the last epoch.
    # An epoch in the logs starts with: `Trainer | epoch 12/12`
    # Let's just find the last "epoch X/12"
    
    last_epoch_idx = 0
    for i, line in enumerate(lines):
        if "Trainer | epoch 12/12" in line:
            last_epoch_idx = i
            
    epoch_lines = lines[last_epoch_idx:]
    
    ph = []
    for line in epoch_lines:
        m = p.search(line)
        if m:
            ph.append(float(m.group(3)))
            
    if ph:
        return np.mean(ph)
    return 0.0

def main():
    weights = [1.0, 3.0, 5.0, 8.0]
    
    results = []
    
    for w in weights:
        print(f"Running sweep for physics_weight={w}")
        cmd = [
            "python", "src/pipelines/train_and_evaluate.py",
            "--swat-path", "data/raw/SWaT/SWaT_Dec2019.pkl",
            "--hai-path", "data/raw/HAI/hai_test1_combined.csv",
            "--swat-meta-path", "data/raw/SWaT/SWaT_Dec2019.xlsx",
            "--stride", "5",
            "--epochs", "12",
            "--contrastive-weight", "1.0",
            "--physics-weight", str(w),
            "--out-dir", f"outputs_sweep_{w}",
            "--log-dir", f"logs_sweep_{w}"
        ]
        
        process = subprocess.run(cmd, capture_output=True, text=True)
        
        # Check history
        try:
            with open(f"outputs_sweep_{w}/tables/training_history.json") as f:
                history = json.load(f)
                
            final_train = history[-1]["train_loss"]
            final_val = history[-1]["val_loss"]
            
            val_losses = [h["val_loss"] for h in history]
            if val_losses[-1] <= min(val_losses):
                status = "Improving"
            elif val_losses[-1] > min(val_losses) * 1.05:
                status = "Diverging"
            else:
                status = "Plateauing"
        except Exception as e:
            print(f"Failed to read history for {w}: {e}")
            final_train = -1
            final_val = -1
            status = "Error"
            
        raw_physics = parse_last_epoch_physics_loss(process.stderr)
        
        results.append({
            "weight": w,
            "train_loss": final_train,
            "val_loss": final_val,
            "status": status,
            "raw_physics": raw_physics
        })
        
    print("\n--- SWEEP RESULTS ---")
    for r in results:
        print(r)

if __name__ == "__main__":
    main()
