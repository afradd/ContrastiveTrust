import pandas as pd
from datetime import time

class SWaTAttackLabeler:
    """
    Labels the SWaT Dec 2019 dataset for physical attacks.
    
    Attack 1 (Exfiltrate Historian Data) is labeled Normal (0) deliberately 
    because it is a network/historian-layer event with no physical actuation 
    signature, so a physics-based detector cannot and should not be expected 
    to catch it — this is a stated methodological choice, not a missed detection.
    """
    def __init__(self, timestamp_column: str = "t_stamp"):
        self.timestamp_column = timestamp_column
        
    def label(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["label"] = 0
        
        if self.timestamp_column not in df.columns:
            return df
            
        ts = pd.to_datetime(df[self.timestamp_column])
        time_series = ts.dt.time
        
        attack2_windows = [
            ("12:30:00", "12:33:00"),
            ("12:43:00", "12:46:00"),
            ("12:56:00", "12:59:00"),
            ("13:09:00", "13:12:00"),
            ("13:22:00", "13:25:00")
        ]
        
        for start_str, end_str in attack2_windows:
            start_time = pd.to_datetime(start_str).time()
            end_time = pd.to_datetime(end_str).time()
            mask = (time_series >= start_time) & (time_series <= end_time)
            df.loc[mask, "label"] = 1
            
        return df
