# normalize.py
from __future__ import annotations
from typing import Dict, Optional

import pandas as pd
import numpy as np


def assemble_weather_df(
    idx: pd.DatetimeIndex,
    wind10: Optional[pd.Series] = None,
    wind50: Optional[pd.Series] = None,
    wind100: Optional[pd.Series] = None,
    temp2: Optional[pd.Series] = None,
    press: Optional[pd.Series] = None,
    z0_series: Optional[pd.Series] = None,
) -> pd.DataFrame:
    """
    Assemble MultiIndex weather DataFrame expected by windpowerlib ModelChain.
    Columns: ('wind_speed', 10/50/100), ('temperature', 2/10), ('pressure', 0), ('roughness_length', 0)
    Units: m/s, K, Pa, m
    """
    cols = []
    data = []

    def add(col_name, height, series):
        if series is None:
            return
        s = series.reindex(idx).astype(float)
        cols.append((col_name, int(height)))
        data.append(s)

    add("wind_speed", 10, wind10)
    add("wind_speed", 50, wind50)
    add("wind_speed", 100, wind100)
    add("temperature", 2, temp2)
    # By default we will duplicate T2M to 10 m later if needed.
    add("pressure", 0, press)
    add("roughness_length", 0, z0_series)

    if not cols:
        raise ValueError("No weather variables provided.")
    df = pd.concat(data, axis=1)
    df.columns = pd.MultiIndex.from_tuples(cols, names=["variable_name", "height"])
    return df


def ensure_two_temps_if_needed(df: pd.DataFrame, temp_default_K: float = 288.15) -> pd.DataFrame:
    """
    windpowerlib's default temperature_model is 'linear_gradient'. If only temperature at 2 m is present,
    duplicate it to 10 m to avoid errors (conservative assumption: no gradient in surface layer).
    """
    has_t2 = ("temperature", 2) in df.columns
    has_t10 = ("temperature", 10) in df.columns
    if has_t2 and not has_t10:
        t2 = df[("temperature", 2)]
        df[("temperature", 10)] = t2
    elif not has_t2 and has_t10:
        t10 = df[("temperature", 10)]
        df[("temperature", 2)] = t10
    elif (not has_t2) and (not has_t10):
        # set default temperature if none provided
        df[("temperature", 2)] = temp_default_K
        df[("temperature", 10)] = temp_default_K
    return df


def to_timezone_and_hourly(df: pd.DataFrame, tz: str = "UTC") -> pd.DataFrame:
    """
    Ensure UTC timezone and hourly frequency with asfreq (no interpolation).
    """
    idx = pd.to_datetime(df.index, utc=True)
    df = df.copy()
    df.index = idx.tz_convert(tz)
    # Align to exact hourly stamps
    df = df.sort_index()
    # unique hourly index
    hourly_index = pd.date_range(df.index.min().ceil("H"), df.index.max().floor("H"), freq="H", tz=tz)
    df = df.reindex(hourly_index)
    return df


# ---- From user mapping ----
def _convert_series_units(s: pd.Series, unit: str) -> pd.Series:
    unit = unit.strip().lower()
    if unit in ["ms", "m/s", "mps"]:
        return s.astype(float)
    if unit in ["km/h", "kmh"]:
        return s.astype(float) / 3.6
    if unit in ["c", "°c", "degc"]:
        return s.astype(float) + 273.15
    if unit in ["k", "kelvin"]:
        return s.astype(float)
    if unit in ["pa"]:
        return s.astype(float)
    if unit in ["hpa", "mbar", "mb"]:
        return s.astype(float) * 100.0
    if unit in ["kpa"]:
        return s.astype(float) * 1000.0
    if unit in ["bar"]:
        return s.astype(float) * 1e5
    if unit in ["m"]:
        return s.astype(float)
    # default: no change
    return s.astype(float)


def _add_from_mapping(df_out: pd.DataFrame, idx: pd.DatetimeIndex, src: pd.DataFrame, mapping: Dict, units: Dict):
    for var_name, submap in mapping.items():
        for h_str, col in submap.items():
            h = int(h_str)
            if col not in src.columns:
                continue
            s = src[col].reindex(idx)
            if col in units:
                s = _convert_series_units(s, units[col])
            if var_name == "wind_speed":
                df_out[("wind_speed", h)] = s
            elif var_name == "temperature":
                df_out[("temperature", h)] = s
            elif var_name == "pressure":
                df_out[("pressure", 0)] = s
            elif var_name == "roughness_length":
                df_out[("roughness_length", 0)] = s


class assemble_weather_df_from_user:
    pass


def from_user_mapping(src: pd.DataFrame, mapping: Dict, units: Dict) -> pd.DataFrame:
    idx = pd.to_datetime(src.index, utc=True)
    out = pd.DataFrame(index=idx)
    _add_from_mapping(out, idx, src, mapping, units)
    # Ensure MultiIndex columns
    out.columns = pd.MultiIndex.from_tuples(out.columns, names=["variable_name", "height"])
    return out


# Expose names for app import
assemble_weather_df.from_user_mapping = staticmethod(from_user_mapping)
