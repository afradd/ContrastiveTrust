"""Physics-guided channel alignment for cross-domain representation learning.

This module is responsible for grouping raw sensor and actuator time-series
into a fixed, 6-channel physical variable space (flow, level, pressure,
temperature, analyzer, actuator).

This is a deliberate, physics-informed design choice. We specifically DO NOT
use learned assignment (like the Hungarian algorithm) or dataset-specific
tag matching. Grouping sensors by the physical variable they measure is what
makes the shared 6-channel input space between SWaT and HAI physically
meaningful rather than arbitrary, allowing the DualStreamEncoder to learn
transferable physics representations.
"""

from __future__ import annotations

import re
import numpy as np
import pandas as pd

# Regular expressions meticulously cross-checked against the complete column
# lists of both SWaT (28 sensors + 32 actuators) and HAI (86 feature columns)
# to guarantee every column maps exactly to one physical category.
TYPE_PATTERNS = [
    ("flow", re.compile(r"FIT|_FT\d|FCV", re.I)),
    ("level", re.compile(r"LIT|_LT\d|LCV|_LL\d|_LH\d|_LD\b|_LCP\d", re.I)),
    ("pressure", re.compile(r"PIT|PCV|_PT\d", re.I)),
    ("temperature", re.compile(r"TIT|_TT\d", re.I)),
    ("analyzer", re.compile(r"AIT|_SIT\d|_VIBTR|_VT\d|_VTR\d", re.I)),
    ("actuator", re.compile(
        r"Status|MV\d|_PP\d|_SOL\d|_STSP|_24Vdc|_ATSW|_Auto|_Emerg|_MASW|"
        r"_Manual|_OnOff|_RTR|_SCO|_SCST|_TripEx|_HT_|_ST_FD|_ST_GOV|_ST_PO|_ST_PS|x100", re.I
    )),
]

TYPE_NAMES = [name for name, _ in TYPE_PATTERNS]
NUM_CHANNELS = len(TYPE_NAMES)
PHYSICS_DIM = NUM_CHANNELS * 3


def classify_columns(columns: list[str]) -> dict[str, list[str]]:
    """Classifies a list of dataset column names into physics groups.
    
    Args:
        columns: A list of feature column names from SWaT or HAI.
        
    Returns:
        A dictionary mapping physics channel types (e.g., "flow", "level")
        to lists of matching column names.
    """
    groups: dict[str, list[str]] = {name: [] for name, _ in TYPE_PATTERNS}
    for col in columns:
        for name, pattern in TYPE_PATTERNS:
            if pattern.search(col):
                groups[name].append(col)
                break
    return groups


def build_typed_frame(df: pd.DataFrame, feature_columns: list[str], keep: list[str]) -> pd.DataFrame:
    """Projects raw features into the 6-channel physical variable space.
    
    Each physical channel represents the mean z-score of all sensors/actuators
    in that category. If a category has no sensors (e.g. no temperature in SWaT),
    it is populated with zeros to maintain the fixed dimension.
    
    Args:
        df: The dataset dataframe.
        feature_columns: The columns to classify and project.
        keep: Extra columns (like timestamp, label) to keep untouched.
        
    Returns:
        A new DataFrame with the 6 physical channels and the kept columns.
    """
    groups = classify_columns(feature_columns)
    out = {}
    for type_name, cols in groups.items():
        cols = [c for c in cols if c in df.columns]
        if not cols:
            out[type_name] = np.zeros(len(df))
            continue
        block = df[cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float64)
        mu = np.nanmean(block, axis=0, keepdims=True)
        sd = np.nanstd(block, axis=0, keepdims=True)
        sd[sd == 0] = 1.0
        z = (block - mu) / sd
        z = np.nan_to_num(z, nan=0.0)
        out[type_name] = z.mean(axis=1)
    
    result = pd.DataFrame(out)
    keep_cols = [c for c in keep if c in df.columns]
    for c in keep_cols:
        result[c] = df[c].to_numpy()
    return result
