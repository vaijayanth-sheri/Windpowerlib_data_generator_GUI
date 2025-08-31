# lineage.py
from __future__ import annotations

import pandas as pd


def attach_source_tag(df: pd.DataFrame, source_name: str) -> pd.DataFrame:
    """
    Add (or overwrite) a 'source' column that users can inspect/export.
    For MultiIndex weather DataFrames we attach a parallel Series (same index),
    returned separately for convenience in the UI when exporting.
    """
    tag = pd.Series(source_name, index=df.index, name="source")
    return tag


def merge_sources(primary: pd.DataFrame, secondary: pd.DataFrame, prefer: str = "primary") -> pd.DataFrame:
    """
    Simple row-wise merge: if a row is fully NA in primary, take secondary.
    Assumes both are normalized to the same schema and index.
    """
    out = primary.copy()
    mask = primary.isna().all(axis=1)
    out[mask] = secondary.reindex(primary.index)[mask]
    return out
