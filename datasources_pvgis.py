# datasources_pvgis.py
from __future__ import annotations
from typing import Dict, Tuple
import re
import pandas as pd
from pvlib import iotools

def _find_col(df: pd.DataFrame, patterns: list[str]) -> str | None:
    """Return the first column name matching any regex in `patterns` (case-insensitive)."""
    cols = {c.lower(): c for c in df.columns}
    for pat in patterns:
        cre = re.compile(pat, re.IGNORECASE)
        for lo, orig in cols.items():
            if cre.fullmatch(lo) or cre.search(lo):
                return orig
    return None

def _clean_and_map_columns(data: pd.DataFrame) -> pd.DataFrame:
    """
    Map pvlib's PVGIS TMY columns to wind-friendly names:
      - WS10m  [m/s]  -> 'WS10m'
      - T2m    [°C]   -> 'T2m_K' (K)
      - SP     [Pa]   -> 'SP_Pa' (Pa) if present
    """
    if data is None or data.empty:
        return pd.DataFrame(index=pd.DatetimeIndex([], tz="UTC"))

    df = data.copy()
    # pvlib should already give a datetime-like index; make it tz-aware UTC
    idx = pd.to_datetime(df.index, utc=True, errors="coerce")
    df.index = idx
    df = df[~df.index.isna()].sort_index()

    out = pd.DataFrame(index=df.index)

    # wind @10 m
    c_ws = _find_col(df, [r"^ws10m$", r"\bwind_speed\b", r"wind.*10m", r"^ws$"])
    if c_ws is not None:
        out["WS10m"] = pd.to_numeric(df[c_ws], errors="coerce")

    # temperature -> Kelvin
    c_t = _find_col(df, [r"t2m", r"temp.*2m", r"dry.*bulb", r"temperature"])
    if c_t is not None:
        out["T2m_K"] = pd.to_numeric(df[c_t], errors="coerce") + 273.15

    # pressure (Pa)
    c_p = _find_col(df, [r"^sp$", r"station.*press", r"pressure"])
    if c_p is not None:
        # PVGIS SP is Pa; if in hPa/kPa, values would be too small — no automatic scaling here.
        out["SP_Pa"] = pd.to_numeric(df[c_p], errors="coerce")

    # drop rows with all NaN
    out = out.dropna(how="all")

    # enforce TMY length if too long (should be exactly 8760)
    if len(out) > 8760:
        out = out.iloc[:8760]

    return out

def fetch_pvgis_tmy(lat: float, lon: float, *,
                    usehorizon: bool = True,
                    startyear: int | None = None,
                    endyear: int | None = None) -> Tuple[pd.DataFrame, Dict]:
    """
    Fetch PVGIS TMY via pvlib and return (df, meta).
    df columns: 'WS10m' [m/s], 'T2m_K' [K], optional 'SP_Pa' [Pa].
    """
    data, meta = iotools.get_pvgis_tmy(
        latitude=float(lat),
        longitude=float(lon),
        map_variables=True,   # standardized names when possible
        usehorizon=bool(usehorizon),
        startyear=startyear,
        endyear=endyear,
    )
    df = _clean_and_map_columns(data)
    return df, {"source": "PVGIS TMY via pvlib", "meta": meta, "rows": int(len(df))}
