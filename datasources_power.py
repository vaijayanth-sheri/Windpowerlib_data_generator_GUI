# datasources_power.py
from __future__ import annotations
import requests
import pandas as pd

# NASA POWER Hourly API docs:
# https://power.larc.nasa.gov/docs/services/api/temporal/hourly/
# Parameter names reference:
# WS10M, WS50M [m/s], WD10M, WD50M [deg], T2M [°C], PS [kPa], Z0M [m]
# We'll request JSON, UTC; convert T2M to K, PS to Pa.

def _build_url(lat, lon, start_dt, end_dt, community="ag"):
    base = "https://power.larc.nasa.gov/api/temporal/hourly/point"
    params = {
        "latitude": float(lat),
        "longitude": float(lon),
        "start": start_dt.strftime("%Y%m%d"),
        "end": end_dt.strftime("%Y%m%d"),
        "parameters": ",".join(["WS10M", "WS50M", "T2M", "PS", "Z0M"]),
        "community": community,  # 'ag' or 'sb'
        "format": "JSON",
        "time-standard": "UTC",
    }
    return base, params

def fetch_power_hourly(lat, lon, start_dt, end_dt, community="re") -> pd.DataFrame:
    base, params = _build_url(lat, lon, start_dt, end_dt, community)
    r = requests.get(base, params=params, timeout=60)
    r.raise_for_status()
    js = r.json()
    hr = js["properties"]["parameter"]
    # POWER hourly timestamps are local standard time by default in some tools; here we requested UTC
    # Build index from provided "time" keys if present; fallback to regular range
    # Many POWER responses provide values keyed by YYYYMMDDHH; collect aligned series
    # Assemble DataFrame from available parameters:
    series = {}
    for key in ["WS10M", "WS50M", "T2M", "PS", "Z0M"]:
        if key in hr:
            # Keys are timestamps like '2024010100', values numeric
            s = pd.Series(hr[key])
            s.index = pd.to_datetime(s.index, format="%Y%m%d%H", utc=True)
            series[key] = s.sort_index()

    # Combine
    df = pd.DataFrame(series).sort_index()
    if "T2M" in df:
        df["T2M_K"] = df["T2M"] + 273.15
    if "PS" in df:
        df["PS_PA"] = df["PS"] * 1000.0  # kPa -> Pa

    return df
